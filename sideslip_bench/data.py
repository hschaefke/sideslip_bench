"""ReV-StED data: download, rebuild the frozen split, verify it, and load recordings as tables.

    import sideslip_bench as ssb
    recs = ssb.load("train")          # {"ReVSTED_recording_1.h5": DataFrame, ...}

The first call downloads the seven recording CSVs from Zenodo (record 15270060, ~0.6 GB), applies
the unit conversions, splits time-continuously 70/10/20 (recordings 1..7 concatenated, then cut;
each recording portion is its own HDF5 file, so no window spans a join) and checks the result
against ``results/split_manifest.json``. Later calls only read the HDF5 files.
"""
import hashlib
import json
from datetime import datetime
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import requests
from tqdm import tqdm

from .inputs import INPUT_SETS, TARGET

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "revsted"
MANIFEST = ROOT / "results" / "split_manifest.json"
ZENODO_RECORD_ID = "15270060"
SPLIT_RATIOS = {"train": 0.70, "valid": 0.10, "test": 0.20}
SPLITS = tuple(SPLIT_RATIOS)
NUM_RECORDINGS = 7
RECORDING_STEM = "Recording_{i}_ReVSTED"
#: Columns ``load`` returns by default: the nine CAN channels an estimator may use (input set
#: u=1), the Correvit target and the ADMA reference (never inputs), and the time stamp.
LOAD_COLUMNS = [*INPUT_SETS[1], TARGET, "sideslip_angle_ADMA_deg", "time"]

# Column rename mapping
RENAME_MAP = {
    "LatAcc_obd": "acc_y_CAN_ms2",                               # Lateral acceleration onboard
    "SW_pos_obd": "delta_sw_deg",                                 # Steering wheel position onboard
    "Yawrate_obd": "yaw_rate_CAN_degs",                          # Yaw rate onboard
    "VelFL_obd": "wheel_vel_fl_CAN_kmh",                         # Velocity front left tyre onboard
    "VelFR_obd": "wheel_vel_fr_CAN_kmh",                         # Velocity front right tyre onboard
    "VelRL_obd": "wheel_vel_rl_CAN_kmh",                         # Velocity rear left tyre onboard
    "VelRR_obd": "wheel_vel_rr_CAN_kmh",                         # Velocity rear right tyre onboard
    "INS_slip_angle_COG": "sideslip_angle_ADMA_deg",             # Vehicle side slip angle at COG (INS)
    "INS_velCOG_hor_y": "vel_y_ADMA_ms",                        # INS velocity in y-direction at COG
    "INS_velCOG_hor_x": "vel_x_ADMA_ms",                        # INS velocity in y-direction at COG
    "Correvit_Vel_COG_x": "vel_x_COR_ms",                       # Correvit velocity transformed to COG
    "Correvit_Vel_COG_y": "vel_y_COR_ms",                       # Correvit velocity transformed to COG
    "Correvit_vehiclesideslip_angle_COG": "sideslip_angle_COR_deg",  # Correvit sideslip angle at COG
    "INS_time_sec": "time",
}
REQUIRED_COLS = [
    "acc_y_CAN_ms2",
    "wheel_vel_fl_CAN_kmh",
    "wheel_vel_fr_CAN_kmh",
    "wheel_vel_rl_CAN_kmh",
    "wheel_vel_rr_CAN_kmh",
    "delta_sw_deg",
    "yaw_rate_CAN_degs",
    "sideslip_angle_ADMA_deg",
    "sideslip_angle_COR_deg",
]


def apply_unit_conversions(df: pd.DataFrame) -> pd.DataFrame:
    """Apply unit conversions."""

    # Negate lateral acceleration to match ISO 8855 convention (positive = left).
    # Revsted uses SAE convention (positive = right), CarMaker uses ISO 8855.
    df["acc_y_CAN_ms2"] = -df["acc_y_CAN_ms2"].astype("float64")

    # km/h to m/s
    df["wheel_vel_fl_CAN_ms"] = df["wheel_vel_fl_CAN_kmh"].astype("float64") * (1000.0 / 3600.0)
    df["wheel_vel_fr_CAN_ms"] = df["wheel_vel_fr_CAN_kmh"].astype("float64") * (1000.0 / 3600.0)
    df["wheel_vel_rl_CAN_ms"] = df["wheel_vel_rl_CAN_kmh"].astype("float64") * (1000.0 / 3600.0)
    df["wheel_vel_rr_CAN_ms"] = df["wheel_vel_rr_CAN_kmh"].astype("float64") * (1000.0 / 3600.0)

    # Axis-wise mean wheel velocities (naming: pos1 = front/rear, pos2 = left/right, x = averaged over that axis)
    df["wheel_vel_xl_CAN_ms"] = (df["wheel_vel_fl_CAN_ms"] + df["wheel_vel_rl_CAN_ms"]) / 2.0  # left mean (front + rear left)
    df["wheel_vel_xr_CAN_ms"] = (df["wheel_vel_fr_CAN_ms"] + df["wheel_vel_rr_CAN_ms"]) / 2.0  # right mean (front + rear right)
    df["wheel_vel_fx_CAN_ms"] = (df["wheel_vel_fl_CAN_ms"] + df["wheel_vel_fr_CAN_ms"]) / 2.0  # front mean (front left + right)
    df["wheel_vel_rx_CAN_ms"] = (df["wheel_vel_rl_CAN_ms"] + df["wheel_vel_rr_CAN_ms"]) / 2.0  # rear mean (rear left + right)
    df["wheel_vel_x_CAN_ms"] = (df["wheel_vel_fl_CAN_ms"] + df["wheel_vel_fr_CAN_ms"] + df["wheel_vel_rl_CAN_ms"] + df["wheel_vel_rr_CAN_ms"]) / 4.0  # grand mean over all four corners

    # degrees to radians
    df["delta_sw_rad"] = np.deg2rad(df["delta_sw_deg"].astype("float64"))
    df["yaw_rate_CAN_rads"] = np.deg2rad(df["yaw_rate_CAN_degs"].astype("float64"))
    df["sideslip_angle_ADMA_rad"] = np.deg2rad(df["sideslip_angle_ADMA_deg"].astype("float64"))
    df["sideslip_angle_COR_rad"] = np.deg2rad(df["sideslip_angle_COR_deg"].astype("float64"))

    return df


# ----------------------------
# Zenodo download
# ----------------------------
def fetch_zenodo_record(record_id: str) -> dict:
    url = f"https://zenodo.org/api/records/{record_id}"
    r = requests.get(url, timeout=60)
    r.raise_for_status()
    return r.json()


def download_if_missing(url: str, out_path: Path) -> None:
    if out_path.exists() and out_path.stat().st_size > 0:
        return

    out_path.parent.mkdir(parents=True, exist_ok=True)
    part = out_path.with_suffix(out_path.suffix + ".part")  # an interrupted download never looks complete
    with requests.get(url, stream=True, timeout=120) as r:
        r.raise_for_status()
        total = int(r.headers.get("Content-Length", 0))
        with part.open("wb") as f, tqdm(
            total=total, unit="B", unit_scale=True, desc=out_path.name, leave=False
        ) as pbar:
            for chunk in r.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    f.write(chunk)
                    pbar.update(len(chunk))
    part.rename(out_path)


def list_recording_files(record_json: dict) -> list[str]:
    keys = []
    for f in record_json.get("files", []):
        key = f.get("key") or f.get("filename")
        if key and key.lower().endswith(".csv") and "recording_" in key.lower():
            keys.append(key)
    return sorted(keys)


def download(raw_dir: Path) -> None:
    """Download the Recording_*.csv files of the Zenodo record into ``raw_dir`` (skips complete files)."""
    record = fetch_zenodo_record(ZENODO_RECORD_ID)
    recording_files = list_recording_files(record)
    if not recording_files:
        raise RuntimeError("No Recording_*.csv files found in the Zenodo record metadata.")
    for fname in tqdm(recording_files, desc="Downloading CSVs"):
        url = f"https://zenodo.org/api/records/{ZENODO_RECORD_ID}/files/{fname}/content"
        download_if_missing(url, Path(raw_dir) / fname)


# ----------------------------
# Split + HDF5 export
# ----------------------------
def to_h5_1d(df: pd.DataFrame, out_h5: Path) -> None:
    out_h5.parent.mkdir(parents=True, exist_ok=True)

    with h5py.File(out_h5, "w") as h5:
        for col in df.columns:
            series = df[col]
            if pd.api.types.is_numeric_dtype(series):
                arr = series.to_numpy(dtype=np.float64, copy=False)
                h5.create_dataset(col, data=arr, shape=(arr.shape[0],), dtype="float64")
            else:
                arr = series.astype(str).to_numpy()
                dt = h5py.string_dtype(encoding="utf-8")
                h5.create_dataset(col, data=arr, shape=(arr.shape[0],), dtype=dt)


def build(root: str | Path = DATA_DIR) -> dict[str, list[tuple[str, int]]]:
    """Rebuild ``root/{train,valid,test}/*.h5`` from ``root/raw_csv``; returns ``{split: [(file, rows)]}``."""
    root = Path(root)
    raw_dir = root / "raw_csv"
    for split_name in SPLIT_RATIOS:
        (root / split_name).mkdir(parents=True, exist_ok=True)

    # Load all 7 recordings in order 1→7, tracking each recording's row span in the
    # combined DataFrame so we can later assign rows to splits per-recording.
    frames = []
    recording_spans: list[tuple[int, int, int]] = []  # (recording_index, global_start, global_end)
    cursor = 0
    for i in tqdm(range(1, NUM_RECORDINGS + 1), desc="Loading CSVs"):
        csv_path = raw_dir / (RECORDING_STEM.format(i=i) + ".csv")
        if not csv_path.exists():
            raise FileNotFoundError(f"Expected file not found: {csv_path}")
        df = pd.read_csv(csv_path)
        df = df.rename(columns=RENAME_MAP)
        recording_spans.append((i, cursor, cursor + len(df)))
        cursor += len(df)
        frames.append(df)

    combined = pd.concat(frames, ignore_index=True)

    # Basic validation: ensure required columns exist after renaming
    missing = [c for c in REQUIRED_COLS if c not in combined.columns]
    if missing:
        raise RuntimeError(f"Missing required columns after rename: {missing}")

    combined = apply_unit_conversions(combined)

    # Continuous time-axis split boundaries
    n = len(combined)
    n_train = int(n * SPLIT_RATIOS["train"])
    n_valid = int(n * SPLIT_RATIOS["valid"])
    split_ranges = {
        "train": (0, n_train),
        "valid": (n_train, n_train + n_valid),
        "test":  (n_train + n_valid, n),
    }

    # For each recording, find which portion(s) fall into each split and save them
    # as individual H5 files.  A recording that straddles a split boundary produces
    # one file per split it touches (e.g. train/ReVSTED_recording_5.h5 and
    # valid/ReVSTED_recording_5.h5).
    saved_files: dict[str, list[tuple[str, int]]] = {s: [] for s in split_ranges}
    for rec_idx, rec_start, rec_end in tqdm(recording_spans, desc="Saving H5"):
        for split_name, (split_start, split_end) in split_ranges.items():
            overlap_start = max(rec_start, split_start)
            overlap_end = min(rec_end, split_end)
            if overlap_start >= overlap_end:
                continue
            chunk = combined.iloc[overlap_start:overlap_end].reset_index(drop=True)
            out_h5 = root / split_name / f"ReVSTED_recording_{rec_idx}.h5"
            to_h5_1d(chunk, out_h5)
            saved_files[split_name].append((str(out_h5), len(chunk)))
    return saved_files


# ----------------------------
# Split manifest (why per-column statistics and not a content hash: scripts/make_split_manifest.py)
# ----------------------------
#: Relative tolerance when comparing per-column statistics. Measured last-bit spread
#: between laptop and cluster is ~1e-13 relative; the smallest real change we ever made to
#: a channel (the 2026-07-25 geometry correction) was ~1e-2.
RTOL = 1e-9


def sha256(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while block := f.read(chunk):
            h.update(block)
    return h.hexdigest()


def column_stats(path: Path) -> dict[str, list]:
    """Per column ``[n_nan, sum, min, max]`` over the finite values.

    Read column by column so a 270 k x 78 recording never lands in memory as a whole.
    """
    out: dict[str, list] = {}
    with h5py.File(path, "r") as f:
        for name in sorted(f.keys()):
            ds = f[name]
            if not isinstance(ds, h5py.Dataset):
                continue
            a = np.asarray(ds[:], dtype=np.float64)
            fin = np.isfinite(a)
            sub = a[fin]
            out[name] = [int(a.size - fin.sum()),
                         float(sub.sum()) if sub.size else 0.0,
                         float(sub.min()) if sub.size else 0.0,
                         float(sub.max()) if sub.size else 0.0]
    return out


def describe(path: Path, root: Path) -> dict:
    with h5py.File(path, "r") as f:
        cols = sorted(f.keys())
        first = f[cols[0]]
        n = int(first.shape[0]) if isinstance(first, h5py.Dataset) else -1
    return {
        "file": str(path.relative_to(root)),
        "n": n,
        "stats": column_stats(path),   # [n_nan, sum, min, max] -- the portable fingerprint
        "sha256": sha256(path),        # file bytes; machine-local, see make_split_manifest.py
        "columns": cols,
    }


def manifest(root: str | Path = DATA_DIR) -> dict:
    """Manifest of the split under ``root``: per file sample count, column statistics, hash, columns."""
    root = Path(root)
    entries: dict[str, list[dict]] = {}
    for split in SPLITS:
        d = root / split
        if not d.is_dir():
            continue
        entries[split] = [describe(p, root) for p in sorted(d.glob("*.h5"))]
    totals = {s: sum(e["n"] for e in v) for s, v in entries.items()}
    return {
        "dataset": "ReV-StED",
        "layout": "time-continuous 70/10/20, author-confirmed (email 2026-01-31): "
                  "recordings 1..7 concatenated, then cut; recording 5 straddles "
                  "train/valid and 6 straddles valid/test, so no window spans a join",
        "created": datetime.now().isoformat(timespec="seconds"),
        "splits": entries,
        "samples": totals,
    }


def check(root: str | Path = DATA_DIR, frozen: str | Path = MANIFEST) -> tuple[list[str], list[str], int]:
    """Compare the split under ``root`` with the frozen manifest; returns ``(drift, benign, n_samples)``.

    ``drift`` lists real differences (files, columns, counts, column statistics beyond ``RTOL``);
    ``benign`` lists files whose bytes differ but whose content matches."""
    current = manifest(root)
    old = json.loads(Path(frozen).read_text())
    drift, benign = [], []
    for split in SPLITS:
        was = {e["file"]: e for e in old.get("splits", {}).get(split, [])}
        now = {e["file"]: e for e in current["splits"].get(split, [])}
        for name in sorted(set(was) | set(now)):
            if name not in now:
                drift.append(f"{split}/{name}: GONE")
                continue
            if name not in was:
                drift.append(f"{split}/{name}: NEW")
                continue
            added = sorted(set(now[name]["columns"]) - set(was[name]["columns"]))
            removed = sorted(set(was[name]["columns"]) - set(now[name]["columns"]))
            if added or removed:
                drift.append(f"{split}/{name}: columns +{added} -{removed}")
                continue
            if was[name]["n"] != now[name]["n"]:
                drift.append(f"{split}/{name}: n {was[name]['n']} -> {now[name]['n']}")
                continue
            if "stats" not in was[name]:
                drift.append(f"{split}/{name}: manifest predates per-column stats — regenerate it")
                continue
            changed = [
                col for col, ref in was[name]["stats"].items()
                # atol carries the case of a statistic that is legitimately 0 (a constant
                # channel), where a pure relative tolerance would demand exact equality.
                if not np.allclose(ref, now[name]["stats"][col], rtol=RTOL, atol=1e-12, equal_nan=True)
            ]
            if changed:
                drift.append(f"{split}/{name}: {len(changed)} column(s) changed beyond "
                             f"rtol={RTOL:g}: {changed[:5]}")
            elif was[name]["sha256"] != now[name]["sha256"]:
                benign.append(f"{split}/{name}")
    return drift, benign, sum(current["samples"].values())


# ----------------------------
# One call: download + rebuild + verify, then load
# ----------------------------
def prepare(root: str | Path = DATA_DIR, splits=SPLITS, force: bool = False) -> Path:
    """Make sure ``root/<split>/*.h5`` exist for ``splits``; if not (or ``force``), download, rebuild and verify.

    Raises ``RuntimeError`` if the rebuilt split differs from ``results/split_manifest.json``."""
    root = Path(root)
    if not force and all(any((root / s).glob("*.h5")) for s in splits):
        return root
    download(root / "raw_csv")
    build(root)
    drift, _, n = check(root)
    if drift:
        raise RuntimeError(f"rebuilt split under {root} differs from {MANIFEST.name}:\n  " + "\n  ".join(drift))
    print(f"[data] {n:,} samples written to {root} and verified against {MANIFEST.name}")
    return root


def load(split: str, columns: list[str] | None = LOAD_COLUMNS, root: str | Path = DATA_DIR) -> dict[str, pd.DataFrame]:
    """``{recording file name: DataFrame}`` of one split at 50 Hz; downloads and builds the data on first use.

    ``columns=None`` returns all 78 channels. Estimators may use the CAN channels of
    ``INPUT_SETS[1]`` only; ``sideslip_angle_COR_deg`` is the target, ``sideslip_angle_ADMA_deg``
    the second reference."""
    if split not in SPLITS:
        raise ValueError(f"unknown split {split!r}, choose from {SPLITS}")
    root = prepare(root, (split,))
    out = {}
    for f in sorted((root / split).glob("*.h5")):
        with h5py.File(f, "r") as h:
            cols = sorted(h.keys()) if columns is None else columns
            out[f.name] = pd.DataFrame({c: np.asarray(h[c]) for c in cols})
    return out
