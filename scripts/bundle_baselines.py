"""Ship one model per comparison method and record all seeds' scores (maintainer script).

    uv run python scripts/bundle_baselines.py --nets <dir>

For every method and every training seed, the validation and test splits are scored with the
benchmark metrics. The shipped model is the seed with the lowest RMSE_g5 on the VALIDATION split
(the test split never enters the choice). Written:

    results/baseline_preds/<name>/<recording>.h5   test predictions of the shipped model
    results/baseline_preds/manifest.json           label, family, shipped seed and its scores
    results/reference_rows.json                    valid + test scores of every seed, median and IQR (the paper's Tab. 1)
    weights/<net>.pt                               weights of the shipped network
    configs/observers/<name>.json                  parameters of the shipped observer

``--nets`` holds, per network and seed, ``<net>_lean_asha_seeded_lean_s<seed>/`` with the research
repo's prediction sidecars ``valid/`` and ``test/`` and its weights ``best_model.pt``. The observers
come from ``scripts/tune_observer.py`` (``results/observer_tuning/<name>_s<seed>.json``); their
predictions are recomputed here. Existing outputs are never overwritten: move them away first.
"""
import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np

ROOT = next(p for p in Path(__file__).resolve().parents if (p / "pyproject.toml").exists())
sys.path.insert(0, str(ROOT))

from sideslip_bench import observers  # noqa: E402
from sideslip_bench.data import DATA_DIR, prepare  # noqa: E402
from sideslip_bench.evaluation import run_estimator  # noqa: E402
from sideslip_bench.metrics import read_preds, score, write_preds  # noqa: E402

NETS = {"tdnn": "TDNN", "tcn": "TCN", "rnn": "RNN", "gru": "GRU", "lstm": "LSTM"}
OBS = {"kf": "linear KF", "ekf": "EKF"}
NET_SEEDS, OBS_SEEDS = range(10), range(3)
KEYS = ("RMSE_g5", "MAE_g5", "P95_g5", "MaxE_g5", "RMSE", "MAE", "P95", "MaxE")


def stats(runs: dict) -> dict:
    """Median and quartiles over the seeds of every test metric."""
    return {k: {q: float(np.percentile([r["test"][k] for r in runs.values()], p))
                for q, p in (("median", 50), ("q25", 25), ("q75", 75))} for k in KEYS}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--nets", type=Path, required=True, help="directory of <net>_lean_asha_seeded_lean_s<seed>/ folders")
    ap.add_argument("--tuning", type=Path, default=ROOT / "results" / "observer_tuning")
    ap.add_argument("--symmetric", action="store_true",
                    help="bundle the mirror-symmetric networks from <net>_lean_asha_seeded_lean_symwrap_s<seed>/ instead")
    args = ap.parse_args()
    if args.symmetric:
        return symmetric(args.nets)

    out_preds, out_w = ROOT / "results" / "baseline_preds", ROOT / "weights"
    out_rows, out_obs = ROOT / "results" / "reference_rows.json", ROOT / "configs" / "observers"
    for p in (out_preds, out_w, out_rows, *(out_obs / f"{n}.json" for n in OBS)):
        if p.exists():
            sys.exit(f"{p.relative_to(ROOT)} exists; move it away first, nothing written")
    split = {s: prepare(DATA_DIR) / s for s in ("valid", "test")}

    rows, manifest, ship = {}, {}, {}
    for name, label in NETS.items():
        runs = {}
        for s in NET_SEEDS:
            d = args.nets / f"{name}_lean_asha_seeded_lean_s{s}"
            preds = {sp: read_preds(d / sp) for sp in split}
            runs[f"s{s}"] = {sp: {k: score(preds[sp], split[sp])[k] for k in KEYS} for sp in split}
            runs[f"s{s}"]["_preds"], runs[f"s{s}"]["_weights"] = preds["test"], d / "best_model.pt"
        cfg = json.loads((ROOT / "configs" / f"{name}.json").read_text())
        rows[name] = {"label": label, "family": "network", "n_params": cfg["source"]["n_params"], "runs": runs}
    for name, label in OBS.items():
        runs = {}
        for s in OBS_SEEDS:
            rec = json.loads((args.tuning / f"{name}_s{s}.json").read_text())
            p = {**rec["fixed"], **rec["tuned"]}
            preds = {sp: run_estimator(lambda r: observers.estimate(name, r, p), split[sp], inputs=9) for sp in split}
            runs[f"s{s}"] = {sp: {k: score(preds[sp], split[sp])[k] for k in KEYS} for sp in split}
            assert abs(runs[f"s{s}"]["valid"]["RMSE_g5"] - rec["valid"]["RMSE_g5"]) < 1e-9, (name, s)
            runs[f"s{s}"]["_preds"], runs[f"s{s}"]["_params"] = preds["test"], rec
        rows[name] = {"label": label, "family": "observer", "runs": runs}

    for name, row in rows.items():
        v = {s: r["valid"]["RMSE_g5"] for s, r in row["runs"].items()}
        if not all(np.isfinite(list(v.values()))):
            sys.exit(f"{name}: non-finite validation score {v}, nothing written")
        best = min(v, key=v.get)  # lowest validation RMSE_g5; ties go to the lower seed
        ship[name] = row["runs"][best]
        manifest[name] = {"label": row["label"], "family": row["family"], "seed": int(best[1:]),
                          "valid_RMSE_g5": v[best], "test": ship[name]["test"]}
        row["shipped_seed"] = int(best[1:])
        print(f"{name:5s} shipped s{best[1:]}: valid {v[best]:.4f}  test {ship[name]['test']['RMSE_g5']:.4f} | "
              f"median over {len(v)} seeds test {np.median([r['test']['RMSE_g5'] for r in row['runs'].values()]):.4f}")

    for name, r in ship.items():
        write_preds(r["_preds"], out_preds / name)
    (out_preds / "manifest.json").write_text(json.dumps(
        {"what": "Test predictions (beta in deg) of the shipped model of every comparison method: the training "
                 "seed with the lowest RMSE_g5 on the validation split. Written by scripts/bundle_baselines.py.",
         "methods": manifest}, indent=1) + "\n")
    out_w.mkdir()
    for name in NETS:
        shutil.copyfile(ship[name]["_weights"], out_w / f"{name}.pt")
    for name in OBS:
        rec = ship[name]["_params"]
        out_obs.joinpath(f"{name}.json").write_text(json.dumps(
            {"observer": name, "label": OBS[name], "seed": rec["seed"], "fixed": rec["fixed"], "tuned": rec["tuned"],
             "bound_hits": rec["bound_hits"], "train_RMSE_g5": rec["train_RMSE_g5"], "pso": rec["pso"],
             "source": f"results/observer_tuning/{name}_s{rec['seed']}.json (scripts/tune_observer.py)"}, indent=2) + "\n")
    out_rows.write_text(json.dumps(
        {"what": "Scores of every training seed (validation and test split; *_g5 = v_x >= 5 km/h, the benchmark "
                 "metrics; plain keys = all speeds; deg). The paper's table reports the median and quartiles over "
                 "the seeds; the shipped model is the seed with the lowest validation RMSE_g5.",
         "rows": {n: {**{k: v for k, v in r.items() if k != "runs"}, "test_over_seeds": stats(r["runs"]),
                      "seeds": {s: {sp: x[sp] for sp in split} for s, x in r["runs"].items()}}
                  for n, r in rows.items()}}, indent=1) + "\n")
    print("wrote", out_preds.relative_to(ROOT), out_w.relative_to(ROOT), out_rows.relative_to(ROOT), out_obs.relative_to(ROOT))


def symmetric(nets: Path) -> None:
    """The networks with built-in mirror symmetry (shipped config + ``"sym_wrap": 1``), the paper's
    "+ symmetry" columns: test predictions of the seed with the lowest validation RMSE_g5 and every
    seed's scores, no weights. Written: ``results/symmetric_preds/`` and ``results/reference_rows_symmetric.json``."""
    out_preds, out_rows = ROOT / "results" / "symmetric_preds", ROOT / "results" / "reference_rows_symmetric.json"
    for p in (out_preds, out_rows):
        if p.exists():
            sys.exit(f"{p.relative_to(ROOT)} exists; move it away first, nothing written")
    split = {s: prepare(DATA_DIR) / s for s in ("valid", "test")}
    rows, manifest = {}, {}
    for name, label in NETS.items():
        runs, test_preds = {}, {}
        for s in NET_SEEDS:
            d = nets / f"{name}_lean_asha_seeded_lean_symwrap_s{s}"
            preds = {sp: read_preds(d / sp) for sp in split}
            runs[f"s{s}"] = {sp: {k: score(preds[sp], split[sp])[k] for k in KEYS} for sp in split}
            test_preds[f"s{s}"] = preds["test"]
        v = {s: r["valid"]["RMSE_g5"] for s, r in runs.items()}
        best = min(v, key=v.get)  # lowest validation RMSE_g5; ties go to the lower seed
        write_preds(test_preds[best], out_preds / name)
        manifest[name] = {"label": f"{label} + symmetry", "family": "network", "seed": int(best[1:]),
                          "valid_RMSE_g5": v[best], "test": runs[best]["test"]}
        n_params = json.loads((ROOT / "configs" / f"{name}.json").read_text())["source"]["n_params"]
        rows[name] = {"label": f"{label} + symmetry", "family": "network", "n_params": n_params,
                      "shipped_seed": int(best[1:]), "test_over_seeds": stats(runs), "seeds": runs}
        print(f"{name:5s} + symmetry shipped {best}: valid {v[best]:.4f}  test {runs[best]['test']['RMSE_g5']:.4f} | "
              f"median over {len(v)} seeds test {np.median([r['test']['RMSE_g5'] for r in runs.values()]):.4f}")
    (out_preds / "manifest.json").write_text(json.dumps(
        {"what": "Test predictions (beta in deg) of the networks with built-in mirror symmetry (config + "
                 "\"sym_wrap\": 1): the training seed with the lowest RMSE_g5 on the validation split. No weights "
                 "are shipped. Written by scripts/bundle_baselines.py --symmetric.",
         "methods": manifest}, indent=1) + "\n")
    out_rows.write_text(json.dumps(
        {"what": "Scores of every training seed of the networks with built-in mirror symmetry (validation and "
                 "test split; *_g5 = v_x >= 5 km/h; plain keys = all speeds; deg), the paper's \"+ symmetry\" "
                 "columns: median over the seeds.",
         "rows": rows}, indent=1) + "\n")
    print("wrote", out_preds.relative_to(ROOT), out_rows.relative_to(ROOT))


if __name__ == "__main__":
    main()
