"""One-call evaluation of a sideslip angle estimator against the bundled reference methods.

    import sideslip_bench as ssb
    rep = ssb.evaluate(my_estimator, name="my_ukf")   # reports/my_ukf/{metrics.json, row.md, row.tex, report.pdf, report.png}

``method`` is one of
  * a callable ``f(rec: pandas.DataFrame) -> array``, called once per test recording with the
    allowed CAN channels of input set ``inputs`` at 50 Hz; it returns beta in degrees for every
    sample, computed causally from the first sample;
  * a dict ``{recording file name: array}``;
  * a directory of per-recording prediction files (the ``scripts/evaluate.py --preds`` contract).
Warm-up skip, speed gate and pooling are those of ``metrics.score``.
"""
import json
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

from .data import DATA_DIR, prepare
from .inputs import GATE_KMH, INPUT_SETS
from .metrics import error_stats, errors, read_preds, score

ROOT = Path(__file__).resolve().parents[1]
BASELINE_DIR = ROOT / "results" / "baseline_preds"
SPEED_BINS = (0, 5, 10, 20, 30, 50, np.inf)  # km/h; 0-5 is gated out of the paper metrics
PROTOCOL = "revsted-ssa-1"  # bump when warm-up, gate, split or metric definitions change


def run_estimator(fn, split_dir: str | Path, inputs: int = 1) -> dict[str, np.ndarray]:
    """Call ``fn`` once per recording of ``split_dir`` with only the channels of ``INPUT_SETS[inputs]``."""
    preds = {}
    for f in sorted(Path(split_dir).glob("*.h5")):
        with h5py.File(f, "r") as h:
            rec = pd.DataFrame({c: np.asarray(h[c]) for c in INPUT_SETS[inputs]})
        p = np.asarray(fn(rec), dtype=np.float64).reshape(-1)
        if p.size != len(rec):
            raise ValueError(f"{f.name}: estimator returned {p.size} values for {len(rec)} samples")
        if not np.isfinite(p).all():
            raise ValueError(f"{f.name}: estimator returned {np.count_nonzero(~np.isfinite(p))} non-finite values")
        preds[f.name] = p
    return preds


def speed_bins(per_rec: dict[str, tuple[np.ndarray, np.ndarray]]) -> list[dict]:
    """Error statistics per speed bin of ``SPEED_BINS``, pooled over the recordings."""
    err, v = (np.concatenate(x) for x in zip(*per_rec.values()))
    out = []
    for lo, hi in zip(SPEED_BINS[:-1], SPEED_BINS[1:]):
        m = (v >= lo) & (v < hi)
        out.append({"lo_kmh": lo, "hi_kmh": None if np.isinf(hi) else hi, "share": float(m.mean()),
                    **(error_stats(err[m]) if m.any() else {"n": 0})})
    return out


def load_baselines(split_dir: str | Path, root: str | Path = BASELINE_DIR) -> dict:
    """``{name: {"label", "family", "seed", "errors": per-recording errors}}`` of the bundled methods (one shipped model each)."""
    root = Path(root)
    manifest = json.loads((root / "manifest.json").read_text())["methods"]
    return {name: {"label": d["label"], "family": d["family"], "seed": d["seed"],
                   "errors": errors(read_preds(root / name), split_dir)}
            for name, d in manifest.items()}


def _row(name: str, m: dict) -> tuple[str, str]:
    vals = [m[k] for k in ("RMSE_g5", "MAE_g5", "P95_g5", "MaxE_g5")]
    md = f"| {name} | " + " | ".join(f"{x:.3f}" for x in vals[:3]) + f" | {vals[3]:.2f} |"
    tex = f"{name} & " + " & ".join(f"{x:.3f}" for x in vals[:3]) + f" & {vals[3]:.2f} \\\\"
    return md, tex


def evaluate(method, name: str, *, data: str | Path = DATA_DIR, inputs: int = 1,
             baselines: str | Path | None = BASELINE_DIR, out_dir: str | Path = "reports",
             report: bool = True) -> dict:
    """Score ``method`` on the test split, compare it with the bundled baselines and write the report.

    Returns ``{"metrics", "speed_bins", "rank", "baselines", "paths"}``; ``rank`` is the place of
    ``method`` by gated RMSE among the shipped models of the comparison methods (1 = best). ``baselines=None`` skips the
    comparison, ``report=False`` the figure (for search loops). The data are downloaded and
    built on first use (``sideslip_bench.data.prepare``)."""
    test_dir = prepare(data, ("test",)) / "test"  # downloads and builds the data on first use
    if callable(method):
        preds = run_estimator(method, test_dir, inputs)
    elif isinstance(method, dict):
        preds = method
    else:
        preds = read_preds(method)
    per_rec = errors(preds, test_dir)
    res = {"name": name, "protocol": PROTOCOL, "inputs": inputs, "gate_kmh": GATE_KMH,
           "metrics": score(preds, test_dir), "speed_bins": speed_bins(per_rec)}

    base = load_baselines(test_dir, baselines) if baselines is not None else {}
    res["baselines"] = {b: {"label": d["label"], "family": d["family"], "seed": d["seed"], **score_from_errors(d["errors"])}
                        for b, d in base.items()}
    others = sorted(x["RMSE_g5"] for x in res["baselines"].values())
    res["rank"] = 1 + int(np.searchsorted(others, res["metrics"]["RMSE_g5"])) if others else None

    out = Path(out_dir) / name
    out.mkdir(parents=True, exist_ok=True)
    md, tex = _row(name, res["metrics"])
    res["paths"] = {"metrics": str(out / "metrics.json"), "row_md": str(out / "row.md"), "row_tex": str(out / "row.tex")}
    (out / "row.md").write_text("| method | RMSE | MAE | P95 | MaxE |\n|---|---|---|---|---|\n" + md + "\n")
    (out / "row.tex").write_text(tex + "\n")
    if report:
        from .report import draw_report  # matplotlib only when a figure is wanted
        res["paths"].update(draw_report(name, per_rec, base, res, out, test_dir))
    (out / "metrics.json").write_text(json.dumps(res, indent=1) + "\n")
    return res


def score_from_errors(per_rec: dict[str, tuple[np.ndarray, np.ndarray]]) -> dict:
    """Gated statistics (``*_g5`` keys) from ``errors`` output, as ``metrics.score`` computes them."""
    err, v = (np.concatenate(x) for x in zip(*per_rec.values()))
    return {k + f"_g{GATE_KMH:g}": val for k, val in error_stats(err[v >= GATE_KMH]).items()}
