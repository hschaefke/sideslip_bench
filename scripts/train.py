"""Train one benchmark row with N seeds and score it with the paper metrics.

For every seed: seed torch/numpy/random, build the learner from the config, train with a
flat-then-cosine learning-rate schedule (tsfast ``fit_flat_cos``), predict every test
recording as one sequence, score it (``sideslip_bench.metrics``), and save

    runs/<name>/s<seed>/config.json     the config used
    runs/<name>/s<seed>/model.pt        state_dict (incl. the input scaler statistics)
    runs/<name>/s<seed>/preds/test/     beta predictions, one h5 per test recording
    runs/<name>/s<seed>/metrics.json
    runs/<name>/summary.json            median and quartiles over the seeds (the paper's aggregation)

    uv run python scripts/train.py configs/gru.json               # seeds 0-9, as in Tab. 1
    # the same champion with built-in mirror symmetry
    uv run python scripts/train.py configs/gru.json --set sym_wrap=1 --name gru_sym
"""
import argparse
import json
import os
import platform
import sys
import time
from pathlib import Path

ROOT = next(p for p in Path(__file__).resolve().parents if (p / "pyproject.toml").exists())
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from sideslip_bench.learner import build_learner, predict, seed_all  # noqa: E402
from sideslip_bench.metrics import score, write_preds  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("config", type=Path)
    ap.add_argument("--seeds", type=int, nargs="+", default=list(range(10)))
    ap.add_argument("--data", type=Path, default=Path("data/revsted"), help="dataset root")
    ap.add_argument("--out", type=Path, default=Path("runs"))
    ap.add_argument("--name", help="run name (default: config file stem)")
    ap.add_argument("--set", nargs="+", default=[], metavar="KEY=VALUE",
                    help="override config entries; VALUE is parsed as JSON")
    args = ap.parse_args()

    cfg = json.loads(args.config.read_text())
    for kv in args.set:
        key, value = kv.split("=", 1)
        if key not in cfg:
            raise SystemExit(f"unknown config key {key!r}")
        cfg[key] = json.loads(value)
    out = args.out / (args.name or args.config.stem)
    test_files = sorted((args.data / "test").glob("*.h5"))
    for seed in args.seeds:
        if (out / f"s{seed}").exists():
            raise SystemExit(f"{out / f's{seed}'} exists; results are never overwritten, pass --name")
    runs = []
    for seed in args.seeds:
        run_dir = out / f"s{seed}"
        seed_all(seed)
        t0 = time.time()
        lrn = build_learner(cfg, args.data)
        lrn.fit_flat_cos(n_epoch=cfg["n_epoch"], lr=cfg["lr"])
        train_s = time.time() - t0
        preds = predict(lrn, test_files)
        m = score(preds, args.data / "test")
        n_params = sum(p.numel() for p in lrn.model.parameters())

        run_dir.mkdir(parents=True)
        (run_dir / "config.json").write_text(json.dumps(cfg, indent=2) + "\n")
        torch.save(lrn.model.state_dict(), run_dir / "model.pt")
        write_preds(preds, run_dir / "preds" / "test")
        env = {"device": str(lrn.device), "torch": torch.__version__, "host": platform.node(),
               "cuda": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}
        rec = {"seed": seed, "metrics": m, "n_params": n_params, "train_seconds": train_s, "env": env}
        (run_dir / "metrics.json").write_text(json.dumps(rec, indent=2) + "\n")
        runs.append(rec)
        print(f"[{out.name}] seed {seed}: RMSE_g5 {m['RMSE_g5']:.4f}  MAE_g5 {m['MAE_g5']:.4f}  "
              f"P95_g5 {m['P95_g5']:.3f}  MaxE_g5 {m['MaxE_g5']:.3f}  | pooled RMSE {m['RMSE']:.4f}  "
              f"({train_s:.0f} s, {n_params} params)", flush=True)

    keys = [k for k, v in runs[0]["metrics"].items() if isinstance(v, float)]
    summary = {k: {q: float(np.percentile(v, p)) for q, p in (("median", 50), ("q25", 25), ("q75", 75))}
               for k in keys for v in [[r["metrics"][k] for r in runs]]}
    (out / "summary.json").write_text(json.dumps({"seeds": args.seeds, "metrics": summary}, indent=2) + "\n")
    print(f"[{out.name}] {len(runs)} seeds: " + "  ".join(
        f"{k} {summary[k]['median']:.4f} [{summary[k]['q25']:.4f}, {summary[k]['q75']:.4f}]" for k in ("RMSE_g5", "MAE_g5", "P95_g5", "MaxE_g5")))


if __name__ == "__main__":
    main()
    # tsfast's background loader threads block the interpreter at exit; everything is written.
    sys.stdout.flush()
    os._exit(0)
