"""Observers and their PSO tuning: the cost is the benchmark metric, the swarm finds a known optimum and
survives diverging particles, and the shipped observers reproduce the bundled predictions.

    uv run python tests/test_observers.py
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import sideslip_bench as ssb  # noqa: E402
from sideslip_bench import observers, pso  # noqa: E402
from sideslip_bench.evaluation import BASELINE_DIR, run_estimator  # noqa: E402
from sideslip_bench.metrics import read_preds, score  # noqa: E402


def test_pso_toy():
    """Shifted sphere in 4-D with a NaN region: the optimum is found and NaN never becomes the best."""
    lb, ub, opt = np.full(4, -5.0), np.full(4, 5.0), np.array([1.0, -2.0, 0.5, 3.0])

    def f(X):
        c = ((X - opt) ** 2).sum(1)
        return np.where(X[:, 0] < -2, np.nan, c)  # a "diverged" region

    x, c, hist = pso.pso(lambda X: np.where(np.isfinite(f(X)), f(X), np.inf), lb, ub, 40, 60, 0)
    assert np.allclose(x, opt, atol=1e-3) and c < 1e-5 and np.all(np.diff(hist) <= 0), (x, c)
    again = pso.pso(lambda X: np.where(np.isfinite(f(X)), f(X), np.inf), lb, ub, 40, 60, 0)
    assert np.array_equal(again[0], x)  # a seed fixes the run


def test_cost_is_benchmark_metric():
    """The batched PSO cost equals metrics.score's RMSE_g5 of observers.estimate on the training split."""
    train = pso.split_arrays("train")
    for name in observers.OBSERVERS:
        p = observers.params(name)
        c = pso.cost(name, np.array([[p[k] for k in observers.PARAMS[name]]]), train)[0]
        ref = score(run_estimator(lambda r: observers.estimate(name, r, p), ssb.data.DATA_DIR / "train", inputs=9),
                    ssb.data.DATA_DIR / "train")["RMSE_g5"]
        assert abs(c - ref) < 1e-8, (name, c, ref)


def test_shipped_observers():
    manifest = json.loads((BASELINE_DIR / "manifest.json").read_text())["methods"]
    test_dir = ssb.data.prepare(splits=("test",)) / "test"
    for name in observers.OBSERVERS:
        preds = run_estimator(ssb.load_baseline(name), test_dir)
        ref = read_preds(BASELINE_DIR / name)
        d = max(float(np.abs(preds[r].astype(np.float32) - ref[r]).max()) for r in ref)
        assert d == 0.0, f"{name}: max |diff| {d:.2e} deg"
        assert abs(score(preds, test_dir)["RMSE_g5"] - manifest[name]["test"]["RMSE_g5"]) < 1e-9


if __name__ == "__main__":
    test_pso_toy()
    test_cost_is_benchmark_metric()
    test_shipped_observers()
    print("ok")
