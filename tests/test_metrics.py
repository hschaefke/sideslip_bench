"""Self-check of the metric protocol on synthetic recordings (warm-up skip, speed gate, pooling).

    uv run python tests/test_metrics.py
"""
import sys
import tempfile
from pathlib import Path

import h5py
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sideslip_bench.metrics import read_preds, score, write_preds  # noqa: E402


def test_score():
    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        # rec A: 30 samples, error 9 in the 20 warm-up samples (must be ignored), then 1.0 at 10 km/h
        # rec B: 25 samples, error 9 in warm-up, then 2.0 at 3 km/h (excluded by the 5 km/h gate)
        for name, n, err, v in [("A.h5", 30, 1.0, 10.0), ("B.h5", 25, 2.0, 3.0)]:
            with h5py.File(d / name, "w") as h:
                h["sideslip_angle_COR_deg"] = np.zeros(n)
                h["speedo_obd"] = np.full(n, v)
        preds = {"A.h5": np.r_[np.full(20, 9.0), np.full(10, 1.0)],
                 "B.h5": np.r_[np.full(20, 9.0), np.full(5, -2.0)]}
        m = score(preds, d)
        assert m["n"] == 15 and m["n_g5"] == 10
        assert np.isclose(m["MAE"], (10 * 1 + 5 * 2) / 15) and np.isclose(m["MaxE"], 2.0)
        assert np.isclose(m["RMSE"], np.sqrt((10 * 1 + 5 * 4) / 15))
        assert m["MAE_g5"] == m["RMSE_g5"] == m["MaxE_g5"] == m["P95_g5"] == 1.0
        write_preds(preds, d / "preds")
        assert score(read_preds(d / "preds"), d) == m


if __name__ == "__main__":
    test_score()
    print("ok")
