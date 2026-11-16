"""Shipped networks: the bundled weights reproduce the bundled test predictions, the built-in
mirror symmetry is exact (both need the [train] extra), and the bundled predictions of the
mirror-symmetric networks reproduce their recorded scores (needs the test split).

    uv run --extra train python tests/test_baselines.py
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import sideslip_bench as ssb  # noqa: E402
from sideslip_bench.evaluation import BASELINE_DIR, run_estimator  # noqa: E402
from sideslip_bench.inputs import INPUT_SETS  # noqa: E402
from sideslip_bench.metrics import read_preds, score  # noqa: E402

# deg; the predictions were made on an L40S GPU, this runs on the CPU (measured per sample <= 1.5e-4,
# one outlier seed 1.1e-3; RMSE_g5 <= 1.5e-6 over all 50 seeds)
TOL, TOL_RMSE = 5e-3, 1e-5


def _torch():
    try:
        import torch
        return torch
    except ImportError:
        print("skip: torch not installed (uv sync --extra train)")
        return None


def test_networks():
    if _torch() is None:
        return
    manifest = json.loads((BASELINE_DIR / "manifest.json").read_text())["methods"]
    test_dir = ssb.data.prepare(splits=("test",)) / "test"
    for name in ssb.baselines.NETWORKS:
        preds = run_estimator(ssb.load_baseline(name), test_dir)
        ref = read_preds(BASELINE_DIR / name)
        d = max(float(np.abs(preds[r] - ref[r]).max()) for r in ref)
        d_rmse = score(preds, test_dir)["RMSE_g5"] - manifest[name]["test"]["RMSE_g5"]
        assert d < TOL and abs(d_rmse) < TOL_RMSE, f"{name}: max |diff| {d:.1e}, RMSE_g5 {d_rmse:+.1e} deg"
        print(f"{name:4s} max |diff| {d:.1e} deg")
    try:
        ssb.load_baseline("mlp")
        raise AssertionError("unknown method not caught")
    except ValueError:
        pass


def test_sym_wrap():
    """beta(mirrored manoeuvre) = -beta exactly, for every architecture."""
    torch = _torch()
    if torch is None:
        return
    from sideslip_bench.learner import build_model
    from sideslip_bench.models import SignAntisymmetricWrapper
    u = INPUT_SETS[1]
    x = torch.randn(2, 50, len(u))
    xm = x.clone()
    for c in ("acc_y_CAN_ms2", "yaw_rate_CAN_rads", "delta_sw_rad"):
        xm[:, :, u.index(c)] *= -1
    for a, b in (("wheel_vel_fl_CAN_ms", "wheel_vel_fr_CAN_ms"), ("wheel_vel_rl_CAN_ms", "wheel_vel_rr_CAN_ms")):
        xm[:, :, [u.index(a), u.index(b)]] = x[:, :, [u.index(b), u.index(a)]]
    for name in ssb.baselines.NETWORKS:
        cfg = json.loads((ssb.baselines.ROOT / "configs" / f"{name}.json").read_text())
        m = SignAntisymmetricWrapper(build_model(cfg, len(u), 1), u).eval()
        with torch.no_grad():
            assert torch.allclose(m(xm), -m(x), atol=1e-6), name


def test_symmetric_preds():
    """Shipped seed = lowest validation RMSE_g5, and the bundled predictions give the recorded test score."""
    sym = ssb.baselines.ROOT / "results" / "symmetric_preds"
    man = json.loads((sym / "manifest.json").read_text())["methods"]
    rows = json.loads((ssb.baselines.ROOT / "results" / "reference_rows_symmetric.json").read_text())["rows"]
    test_dir = ssb.data.prepare(splits=("test",)) / "test"
    for name in ssb.baselines.NETWORKS:
        seeds = rows[name]["seeds"]
        best = min(seeds, key=lambda s: seeds[s]["valid"]["RMSE_g5"])
        assert man[name]["seed"] == rows[name]["shipped_seed"] == int(best[1:]), name
        rmse = score(read_preds(sym / name), test_dir)["RMSE_g5"]
        assert rmse == man[name]["test"]["RMSE_g5"] == seeds[best]["test"]["RMSE_g5"], (name, rmse)


if __name__ == "__main__":
    test_networks()
    test_sym_wrap()
    test_symmetric_preds()
    print("ok")
