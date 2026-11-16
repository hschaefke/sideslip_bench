"""Score beta predictions on the ReV-StED test split with the paper metrics.

    # predictions of any estimator: one <recording>.h5 per test recording, one 1-D dataset (deg)
    uv run python scripts/evaluate.py --preds runs/rnn/s0/preds/test
    # a model trained by scripts/train.py: rebuild it from config.json + model.pt and predict
    uv run python scripts/evaluate.py --run runs/rnn/s0
    # a shipped comparison method (tdnn tcn rnn gru lstm: bundled weights; kf ekf: tuned observers)
    uv run python scripts/evaluate.py --baseline kf --save-preds runs/kf/preds/test
    # with --name: also compare with the bundled baselines and write reports/<name>/ (report.pdf/png, metrics.json, row.md/tex)
    uv run python scripts/evaluate.py --preds my_estimator/test --name my_ukf
"""
import argparse
import json
import os
import sys
from pathlib import Path

ROOT = next(p for p in Path(__file__).resolve().parents if (p / "pyproject.toml").exists())
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))

from sideslip_bench.baselines import load_baseline  # noqa: E402
from sideslip_bench.data import prepare  # noqa: E402
from sideslip_bench.evaluation import evaluate, run_estimator  # noqa: E402
from sideslip_bench.metrics import read_preds, score, write_preds  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--preds", type=Path, help="directory of per-recording prediction files")
    src.add_argument("--run", type=Path, help="run directory written by scripts/train.py")
    src.add_argument("--baseline", help="bundled comparison method (see sideslip_bench.baselines.METHODS)")
    ap.add_argument("--save-preds", type=Path, help="also write the predictions to this directory")
    ap.add_argument("--data", type=Path, default=Path("data/revsted"), help="dataset root")
    ap.add_argument("--name", help="write the standard report to reports/<name>/")
    args = ap.parse_args()

    test_dir = prepare(args.data, ("test",)) / "test"
    if args.preds:
        preds = read_preds(args.preds)
    elif args.baseline:
        preds = run_estimator(load_baseline(args.baseline), test_dir)
    else:
        import torch
        from sideslip_bench.learner import build_learner, predict
        cfg = json.loads((args.run / "config.json").read_text())
        lrn = build_learner(cfg, args.data)
        lrn.model.load_state_dict(torch.load(args.run / "model.pt", map_location="cpu"))
        preds = predict(lrn, sorted(test_dir.glob("*.h5")))
    if args.save_preds:
        if args.save_preds.exists():
            raise SystemExit(f"{args.save_preds} exists; predictions are never overwritten")
        write_preds(preds, args.save_preds)
    if args.name:
        res = evaluate(preds, args.name, data=args.data)
        print(json.dumps({k: res[k] for k in ("metrics", "rank", "paths")}, indent=2))
    else:
        print(json.dumps(score(preds, test_dir), indent=2))


if __name__ == "__main__":
    main()
    # tsfast's background loader threads block the interpreter at exit; everything is written.
    sys.stdout.flush()
    os._exit(0)
