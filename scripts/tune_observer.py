"""Tune an observer's parameters by PSO on the training split, one run per seed (``sideslip_bench.pso``).

    uv run python scripts/tune_observer.py kf --seeds 0 1 2     # -> results/observer_tuning/kf_s<seed>.json

Prints the validation RMSE of every seed and checks that the runs agree to within ``--tol`` deg,
i.e. that the swarm budget is large enough for a reproducible optimum. scripts/bundle_baselines.py
then ships the seed with the lowest validation RMSE. Existing records are never overwritten.
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = next(p for p in Path(__file__).resolve().parents if (p / "pyproject.toml").exists())
sys.path.insert(0, str(ROOT))

from sideslip_bench import pso  # noqa: E402
from sideslip_bench.data import DATA_DIR, prepare  # noqa: E402
from sideslip_bench.observers import OBSERVERS  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("observer", choices=OBSERVERS)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--particles", type=int, default=pso.N_PARTICLES)
    ap.add_argument("--iters", type=int, default=pso.N_ITERS)
    ap.add_argument("--restarts", type=int, default=pso.N_RESTARTS)
    ap.add_argument("--out", type=Path, default=ROOT / "results" / "observer_tuning")
    ap.add_argument("--tol", type=float, default=0.005, help="max spread of the validation RMSE_g5 over the seeds")
    args = ap.parse_args()

    prepare(DATA_DIR)
    paths = {s: args.out / f"{args.observer}_s{s}.json" for s in args.seeds}
    for p in paths.values():
        if p.exists():
            raise SystemExit(f"{p} exists; records are never overwritten, pass --out")
    args.out.mkdir(parents=True, exist_ok=True)
    valid = []
    for s, path in paths.items():
        rec = pso.tune(args.observer, s, args.particles, args.iters, args.restarts)
        path.write_text(json.dumps(rec, indent=1) + "\n")
        valid.append(rec["valid"]["RMSE_g5"])
        print(f"[{args.observer}] seed {s}: train {rec['train_RMSE_g5']:.4f}  valid {rec['valid']['RMSE_g5']:.4f}  "
              f"test {rec['test']['RMSE_g5']:.4f} RMSE_g5  ({rec['seconds']:.0f} s"
              f"{', at bound: ' + ', '.join(rec['bound_hits']) if rec['bound_hits'] else ''})", flush=True)
    spread = max(valid) - min(valid)
    print(f"[{args.observer}] validation spread over seeds {spread:.4f} deg: "
          f"{'reproducible' if spread <= args.tol else f'NOT reproducible (> {args.tol}), raise --restarts/--particles/--iters'}")


if __name__ == "__main__":
    main()
