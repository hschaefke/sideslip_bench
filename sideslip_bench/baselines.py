"""The paper's comparison methods as ready-to-run estimators, one shipped model each.

    import sideslip_bench as ssb
    gru = ssb.load_baseline("gru")          # bundled weights, needs the [train] extra (torch)
    kf = ssb.load_baseline("kf")            # tuned observer, numba only
    beta = gru(ssb.load("test")["ReVSTED_recording_7.h5"])   # beta in deg for every sample
    ssb.evaluate(kf, name="kf")            # reproduces the bundled KF row

The shipped model of a method is its training seed with the lowest gated RMSE on the validation
split: networks ``configs/<net>.json`` + ``weights/<net>.pt`` (``state_dict`` incl. the input
scaler statistics), observers ``configs/observers/<name>.json``. All seeds' scores are in
``results/reference_rows.json``.
"""
import json
from functools import partial
from pathlib import Path

import numpy as np

from . import observers
from .inputs import INPUT_SETS

ROOT = Path(__file__).resolve().parents[1]
NETWORKS = ("tdnn", "tcn", "rnn", "gru", "lstm")
METHODS = NETWORKS + observers.OBSERVERS


def load_baseline(name: str):
    """Estimator ``f(rec: DataFrame) -> beta in deg`` of comparison method ``name``.

    ``rec`` needs the CAN channels of input set u=1 (what ``ssb.evaluate`` passes by default)."""
    if name in observers.OBSERVERS:
        return partial(observers.estimate, name, p=observers.params(name))
    if name not in NETWORKS:
        raise ValueError(f"unknown method {name!r}, choose from {METHODS}")
    try:
        import torch  # networks only: the core install has no torch
    except ImportError as err:
        raise ImportError("the networks need the [train] extra: uv sync --extra train") from err

    from .learner import load_model
    cfg = json.loads((ROOT / "configs" / f"{name}.json").read_text())
    model, u_names = load_model(cfg, ROOT / "weights" / f"{name}.pt"), INPUT_SETS[cfg["u"]]

    @torch.no_grad()
    def net(rec) -> np.ndarray:  # the whole recording as one sequence, on CPU
        x = torch.as_tensor(np.stack([np.asarray(rec[c], dtype=np.float32) for c in u_names], -1))[None]
        return model(x)[0, :, 0].numpy()

    return net
