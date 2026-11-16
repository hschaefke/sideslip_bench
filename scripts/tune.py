"""Lean hyperparameter search for one network: Ray Tune random search + ASHA (needs the [tune] extra).

    uv sync --extra tune
    uv run python scripts/tune.py gru                      # 100 trials -> runs/tune_gru/best_config.json
    uv run python scripts/tune.py gru --gpus-per-trial 0.5 # two trials per GPU

Protocol of the paper: only the learning rate (log-uniform 3e-4 to 3e-3), the width
{16, 32, 64, 128} and the depth (layers 1-3; TCN 4-7 blocks and kernel 3 or 5; TDNN 1-3 head
layers and a lag window of 5, 10, 20 or 50 samples) are searched; every other setting is the one
in ``configs/<net>.json``. Each trial trains up to 100 epochs and reports the validation RMSE
(``fun_rmse``, 400-sample windows) after every epoch; ASHA stops weak trials at 20, 40 and 80
epochs (grace period 20, reduction factor 2). The champion is the trial with the lowest validation
RMSE after its last epoch; retrain it with ``scripts/train.py``. The test split is never touched.
"""
import argparse
import json
import os
import sys
from pathlib import Path

ROOT = next(p for p in Path(__file__).resolve().parents if (p / "pyproject.toml").exists())
sys.path.insert(0, str(ROOT))

from ray import tune  # noqa: E402
from ray.tune.schedulers import ASHAScheduler  # noqa: E402
from tsfast.tune import report_metrics  # noqa: E402

from sideslip_bench.data import DATA_DIR, prepare  # noqa: E402
from sideslip_bench.learner import build_learner  # noqa: E402

DEPTH = {**{n: {"num_layers": tune.choice([1, 2, 3])} for n in ("rnn", "gru", "lstm")},
         "tdnn": {"hidden_layer": tune.choice([1, 2, 3]), "delay": tune.choice([5, 10, 20, 50])},
         # receptive field 1 + 2 (k - 1) (2^blocks - 1): 61..509 samples (k = 3), 121..1017 (k = 5)
         "tcn": {"num_layers": tune.choice([4, 5, 6, 7]), "tcn_kernel_size": tune.choice([3, 5])}}


def trainable(cfg: dict) -> None:
    lrn = build_learner(cfg, cfg.pop("data_dir"))
    with lrn.no_bar(), report_metrics(lrn, checkpoint_every=None):
        lrn.fit_flat_cos(n_epoch=cfg["n_epoch"], lr=cfg["lr"])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("net", choices=sorted(DEPTH))
    ap.add_argument("--num-samples", type=int, default=100)
    ap.add_argument("--epochs", type=int, default=None, help="override n_epoch (ASHA max_t), e.g. for a smoke test")
    ap.add_argument("--cpus-per-trial", type=float, default=1)
    ap.add_argument("--gpus-per-trial", type=float, default=0)
    ap.add_argument("--out", type=Path, default=None, help="default runs/tune_<net>")
    args = ap.parse_args()

    out = (args.out or ROOT / "runs" / f"tune_{args.net}").resolve()
    if out.exists():
        raise SystemExit(f"{out} exists; results are never overwritten, pass --out")
    base = json.loads((ROOT / "configs" / f"{args.net}.json").read_text())
    base.pop("source")
    if args.epochs:
        base["n_epoch"] = args.epochs
    space = {**base, "lr": tune.loguniform(3e-4, 3e-3), "hidden_size": tune.choice([16, 32, 64, 128]),
             **DEPTH[args.net], "data_dir": str(prepare(DATA_DIR))}  # absolute: trials run in their own directory
    scheduler = ASHAScheduler(time_attr="training_iteration", metric="fun_rmse", mode="min",
                              max_t=base["n_epoch"], grace_period=min(20, base["n_epoch"]), reduction_factor=2)
    tuner = tune.Tuner(
        tune.with_resources(trainable, {"cpu": args.cpus_per_trial, "gpu": args.gpus_per_trial}),
        param_space=space,
        tune_config=tune.TuneConfig(scheduler=scheduler, num_samples=args.num_samples),
        run_config=tune.RunConfig(name="trials", storage_path=str(out)))
    grid = tuner.fit()
    grid.get_dataframe().to_csv(out / "trials.csv", index=False)
    best = grid.get_best_result(metric="fun_rmse", mode="min", scope="last")
    cfg = {k: v for k, v in best.config.items() if k != "data_dir"}
    cfg["source"] = {"search": f"scripts/tune.py, {args.num_samples} trials", "valid_fun_rmse": best.metrics["fun_rmse"]}
    (out / "best_config.json").write_text(json.dumps(cfg, indent=2) + "\n")
    print(f"[tune:{args.net}] best valid fun_rmse {best.metrics['fun_rmse']:.4f}: "
          + ", ".join(f"{k}={cfg[k]}" for k in ("lr", "hidden_size", *DEPTH[args.net])) + f" -> {out / 'best_config.json'}")


if __name__ == "__main__":
    main()
    sys.stdout.flush()
    os._exit(0)  # tsfast and Ray keep threads alive that block a normal exit; everything is written
