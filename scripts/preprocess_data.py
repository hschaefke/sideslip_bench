"""Download ReV-StED, rebuild the 70/10/20 split and verify it (CLI for ``sideslip_bench.data``).

    uv run python scripts/preprocess_data.py      # data/revsted/{raw_csv,train,valid,test}

Rebuilds even if the split exists (``ssb.load`` builds only when it is missing); stops with an
error if the result differs from results/split_manifest.json.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(next(p for p in Path(__file__).resolve().parents if (p / "pyproject.toml").exists())))

from sideslip_bench.data import DATA_DIR, prepare  # noqa: E402


if __name__ == "__main__":
    prepare(DATA_DIR, force=True)
