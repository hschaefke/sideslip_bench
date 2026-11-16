# -*- coding: utf-8 -*-
"""Freeze the ReV-StED split: one committed JSON that pins files, hashes and sample counts.

WHY THIS EXISTS. The split is the single most load-bearing convention of the benchmark:
time-continuous, author-confirmed, and cut so that no window spans a join. Without a
manifest it lives only as a directory layout that anyone could change without noticing.

The manifest records, per file: split, sample count, two hashes, and the dataset columns.
That makes three different questions answerable after the fact:

  * did the data change since the numbers were produced? (hash)
  * does a reproduction attempt use the same split? (file list + counts)
  * did someone write model outputs back into the measurements? (column list)

WHY NOT A CONTENT HASH (measured 2026-07-28). The obvious design -- hash the column data --
does not survive contact with two machines, for two separate reasons:

  1. ``sha256`` over the FILE bytes is machine-local anyway: deleting a dataset does not
     shrink an HDF5 file, it frees space internally, and the layout differs between builds.
     On two machines with identical data, all nine file hashes
     differed while every column and sample count matched.
  2. A hash over the DATA fails too. 15 of the 78 channels differ between laptop and
     cluster by **1e-14 to 1e-21** -- exactly the *computed* ones (deg<->rad, the POI
     transform, velocity magnitudes), each preprocessed independently from the CSV on a
     different CPU. Numerically identical, bitwise not. Rounding before hashing only moves
     the problem: with ~73 M values, a value sitting on a rounding boundary is likelier
     than not.

So the manifest pins **per-column statistics** (n_nan, sum, min, max) and ``--check``
compares them with a relative tolerance of 1e-9. That is loose enough to ignore
last-bit noise and far tighter than any real change: the 2026-07-25 geometry correction
moved β by ~1e-2. The file ``sha256`` is kept as a "was this file touched at all" signal,
and a difference in it alone is reported as benign.

    uv run python scripts/make_split_manifest.py            # write results/split_manifest.json (maintainer)
    uv run python scripts/make_split_manifest.py --check     # verify, exit 1 on drift

It lands in ``results/`` rather than beside the data, because ``data/`` is gitignored
wholesale and a manifest nobody can commit pins nothing.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = next(p for p in Path(__file__).resolve().parents if (p / "pyproject.toml").exists())
sys.path.insert(0, str(ROOT))

from sideslip_bench.data import DATA_DIR, MANIFEST, check, manifest  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--check", action="store_true",
                    help="compare against the committed manifest instead of writing it")
    args = ap.parse_args()

    if not args.check:
        current = manifest(DATA_DIR)
        MANIFEST.write_text(json.dumps(current, indent=2) + "\n")
        print(f"[manifest] wrote {MANIFEST.relative_to(ROOT)}")
        for s, n in current["samples"].items():
            print(f"  {s:6s} {len(current['splits'][s])} file(s), {n:,} samples")
        return 0

    if not MANIFEST.exists():
        print(f"[manifest] MISSING: {MANIFEST.relative_to(ROOT)} -- run without --check first")
        return 1
    drift, benign, n = check(DATA_DIR)
    if benign:
        print(f"[manifest] {len(benign)} file(s) differ in FILE bytes but not in content — "
              f"expected after an HDF5 delete/repack or across machines; not drift.")
    if drift:
        print("[manifest] DRIFT against the frozen split:")
        for d in drift:
            print(f"  {d}")
        return 1
    print(f"[manifest] OK -- {n:,} samples match the frozen split")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
