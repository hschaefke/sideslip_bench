"""Self-check of the data API on synthetic CSVs (split boundaries, manifest check, ssb.load), and on
the real split if it has been built.

    uv run python tests/test_data.py
"""
import json
import sys
import tempfile
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sideslip_bench import data  # noqa: E402


def make_csvs(raw: Path, lens: list[int]) -> None:
    """Seven recordings with every raw column the preprocessing renames or the loader returns."""
    rng = np.random.default_rng(0)
    raw.mkdir(parents=True)
    for i, n in enumerate(lens, 1):
        df = pd.DataFrame({c: rng.normal(size=n) for c in [*data.RENAME_MAP, "speedo_obd", "brake_pressure_obd"]})
        df.to_csv(raw / f"Recording_{i}_ReVSTED.csv", index=False)


def test_synthetic():
    with tempfile.TemporaryDirectory() as d:
        root = Path(d)
        lens = [300, 200, 250, 250, 100, 200, 200]  # 1500 rows -> 1050 / 150 / 300
        make_csvs(root / "raw_csv", lens)
        files = data.build(root)
        assert {s: sum(n for _, n in f) for s, f in files.items()} == {"train": 1050, "valid": 150, "test": 300}
        # rec 5 (rows 1000..1100) straddles the cut at 1050, rec 6 (1100..1300) the cut at 1200
        assert [Path(f).name for f, _ in files["valid"]] == ["ReVSTED_recording_5.h5", "ReVSTED_recording_6.h5"]
        frozen = root / "manifest.json"
        frozen.write_text(json.dumps(data.manifest(root)))
        assert data.check(root, frozen)[0] == []
        # load: present data is not rebuilt, default columns, all speeds of a split, in file order
        recs = data.load("test", root=root)
        assert list(recs) == ["ReVSTED_recording_6.h5", "ReVSTED_recording_7.h5"]
        assert list(recs["ReVSTED_recording_7.h5"].columns) == data.LOAD_COLUMNS
        assert sum(len(r) for r in recs.values()) == 300
        assert np.allclose(recs["ReVSTED_recording_7.h5"]["acc_y_CAN_ms2"],  # SAE -> ISO sign flip
                           -pd.read_csv(root / "raw_csv/Recording_7_ReVSTED.csv")["LatAcc_obd"])
        assert len(data.load("valid", columns=None, root=root)["ReVSTED_recording_5.h5"].columns) > len(data.LOAD_COLUMNS)
        # a changed value is drift
        with h5py.File(root / "test/ReVSTED_recording_7.h5", "r+") as h:
            h["speedo_obd"][0] += 1.0
        drift = data.check(root, frozen)[0]
        assert len(drift) == 1 and "speedo_obd" in drift[0], drift
        try:
            data.load("nope", root=root)
            raise AssertionError("unknown split not caught")
        except ValueError:
            pass


def test_real_split():
    """The built split matches the frozen manifest (skipped when data/revsted has not been built)."""
    if not any((data.DATA_DIR / "test").glob("*.h5")):
        print("skip test_real_split: no data/revsted (run ssb.load once)")
        return
    drift, _, n = data.check()
    assert drift == [] and n == 938_324, drift
    assert sum(len(r) for r in data.load("test").values()) == 187_666


if __name__ == "__main__":
    test_synthetic()
    test_real_split()
    print("ok")
