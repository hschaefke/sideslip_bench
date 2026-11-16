"""Self-check of the one-call evaluation on synthetic recordings (input contract, baselines, rank, report, window).

    uv run python tests/test_evaluate.py
"""
import json
import sys
import tempfile
from pathlib import Path

import h5py
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sideslip_bench.evaluation import evaluate  # noqa: E402
from sideslip_bench.inputs import INPUT_SETS  # noqa: E402
from sideslip_bench.metrics import score, write_preds  # noqa: E402


def make_split(d: Path) -> dict[str, int]:
    """Two recordings with target 0 and speeds spanning the gate; all CAN channels present."""
    lens = {"A.h5": 6000, "B.h5": 2000}  # A is long enough for the report's 60 s window at v_x >= 10 km/h
    rng = np.random.default_rng(0)
    for name, n in lens.items():
        with h5py.File(d / "test" / name, "w") as h:
            h["sideslip_angle_COR_deg"] = np.zeros(n)
            h["speedo_obd"] = np.linspace(0, 60, n)
            for c in set(INPUT_SETS[1]) - {"speedo_obd"}:
                h[c] = rng.normal(size=n)
    return lens


def test_evaluate():
    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        (d / "test").mkdir()
        lens = make_split(d)
        # bundled baselines: a network with error 0.5 and a KF with error 1.0, one shipped model each
        man = {"methods": {}}
        for name, fam, e in (("net", "network", 0.5), ("kf", "observer", 1.0)):
            write_preds({r: np.full(n, e) for r, n in lens.items()}, d / "base" / name)
            man["methods"][name] = {"label": name, "family": fam, "seed": 0}
        (d / "base" / "manifest.json").write_text(json.dumps(man))

        seen = []

        def est(rec):  # constant error 0.2 deg, records which channels it was given
            seen.append(sorted(rec.columns))
            return np.full(len(rec), 0.2)

        res = evaluate(est, "est", data=d, inputs=7, baselines=d / "base", out_dir=d / "rep")
        assert seen == [sorted(INPUT_SETS[7])] * 2, seen
        assert res["metrics"] == score({r: np.full(n, 0.2) for r, n in lens.items()}, d / "test")
        assert np.isclose(res["metrics"]["MAE_g5"], 0.2)
        assert sum(b["n"] for b in res["speed_bins"]) == res["metrics"]["n"]
        assert res["rank"] == 1 and res["baselines"]["kf"]["RMSE_g5"] == 1.0
        for k in ("metrics", "row_md", "row_tex", "report_pdf", "report_png"):
            assert Path(res["paths"][k]).stat().st_size > 0, k
        # a dict and a prediction folder give the same metrics as the callable
        write_preds({r: np.full(n, 0.2) for r, n in lens.items()}, d / "mine")
        assert evaluate(d / "mine", "dir", data=d, baselines=None, out_dir=d / "rep", report=False)["metrics"] == res["metrics"]
        # contract violations
        for bad, msg in ((lambda rec: np.zeros(len(rec) - 1), "values for"),
                         (lambda rec: np.full(len(rec), np.nan), "non-finite")):
            try:
                evaluate(bad, "bad", data=d, baselines=None, out_dir=d / "rep", report=False)
                raise AssertionError("contract violation not caught")
            except ValueError as err:
                assert msg in str(err), err


def test_report_window():
    """Report panel (ii): the 60 s window at v_x >= 10 km/h with the most corners, the earliest on ties."""
    from sideslip_bench.report import FS, pick_window
    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        n = int(180 * FS)
        ay, vx = np.zeros(n), np.full(n, 30.0)
        for t0 in (70, 80, 90, 150):  # 2 s corners at 3 m/s^2: the first three fit into one 60 s window
            ay[int(t0 * FS):int((t0 + 2) * FS)] = 3.0
        vx[:int(20 * FS)] = 5.0      # too slow for the first 20 s
        for name, v in (("A.h5", vx), ("B.h5", np.full(n, 8.0))):  # B has the same corners but is too slow
            with h5py.File(d / name, "w") as h:
                h["acc_y_CAN_ms2"], h["speedo_obd"] = ay, v
        assert pick_window(d) == ("A.h5", 1620, 3), pick_window(d)  # earliest 1 s step that holds 70-92 s
        (d / "A.h5").unlink()
        assert pick_window(d) is None


if __name__ == "__main__":
    test_evaluate()
    test_report_window()
    print("ok")
