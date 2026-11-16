"""Paper metrics: RMSE, MAE, P95 and MaxE of the beta error in degrees.

Protocol: every test recording is predicted as one sequence from its first sample; the
first ``SKIP_N`` = 20 samples of each recording are dropped (warm-up of stateful nets);
the errors of all recordings are pooled. Reported twice:
  * gated (paper): only samples with v_x >= 5 km/h on ``speedo_obd``, keys ``*_g5``;
  * pooled: all samples, plain keys (comparable to the ungated literature rows).
"""
from pathlib import Path

import h5py
import numpy as np

from .inputs import GATE_KMH, SKIP_N, SPEED_CHANNEL, TARGET


def error_stats(err: np.ndarray) -> dict:
    """RMSE / MAE / P95 / MaxE of a signed error vector (deg); P95 uses linear interpolation."""
    err = np.asarray(err, dtype=np.float64)
    ae = np.abs(err)
    return {"RMSE": float(np.sqrt(np.mean(err ** 2))), "MAE": float(ae.mean()),
            "P95": float(np.quantile(ae, 0.95)), "MaxE": float(ae.max()), "n": int(err.size)}


def errors(preds: dict[str, np.ndarray], split_dir: str | Path,
           skip_n: int = SKIP_N) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Per recording ``{file name: (signed error in deg, speed in km/h)}`` after the warm-up skip.

    Every recording of the split must be present and have the full recording length."""
    files = sorted(Path(split_dir).glob("*.h5"))
    if sorted(preds) != [f.name for f in files]:
        raise ValueError(f"predictions for {sorted(preds)}, recordings are {[f.name for f in files]}")
    out = {}
    for f in files:
        with h5py.File(f, "r") as h:
            # float32 like the training pipeline; error formed in float32, statistics in float64
            t = np.asarray(h[TARGET], dtype=np.float32)
            v = np.asarray(h[SPEED_CHANNEL], dtype=np.float32)
        p = np.asarray(preds[f.name], dtype=np.float32).reshape(-1)
        if p.shape != t.shape:
            raise ValueError(f"{f.name}: {p.size} predictions for {t.size} samples")
        out[f.name] = ((p - t)[skip_n:], v[skip_n:])
    return out


def score(preds: dict[str, np.ndarray], split_dir: str | Path, skip_n: int = SKIP_N,
          gate_kmh: float = GATE_KMH) -> dict:
    """Score per-recording predictions ``{file name: beta in deg}`` against ``split_dir/*.h5``."""
    err, v = (np.concatenate(x) for x in zip(*errors(preds, split_dir, skip_n).values()))
    gated = error_stats(err[v >= gate_kmh])
    tag = f"_g{gate_kmh:g}"
    return {**error_stats(err), **{k + tag: val for k, val in gated.items()}, "skip_n": skip_n}


def read_preds(pred_dir: str | Path) -> dict[str, np.ndarray]:
    """Read ``pred_dir/<recording>.h5`` files that each hold exactly one 1-D dataset."""
    out = {}
    for f in sorted(Path(pred_dir).glob("*.h5")):
        with h5py.File(f, "r") as h:
            keys = list(h.keys())
            if len(keys) != 1:
                raise ValueError(f"{f}: expected one dataset, found {keys}")
            out[f.name] = np.asarray(h[keys[0]])
    return out


def write_preds(preds: dict[str, np.ndarray], pred_dir: str | Path) -> None:
    pred_dir = Path(pred_dir)
    pred_dir.mkdir(parents=True, exist_ok=True)
    for name, p in preds.items():
        with h5py.File(pred_dir / name, "w") as h:
            h.create_dataset("beta_deg", data=np.asarray(p, dtype=np.float32))
