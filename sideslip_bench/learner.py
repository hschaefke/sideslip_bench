"""Windowed dataloader + model dispatch for the paper protocol, on top of tsfast.

``build_learner(cfg)`` reproduces the research code's learner construction for the five paper
architectures step by step and in the same order (data loaders, model), so that a given seed
draws the same random numbers. Training is fixed by the protocol: MAE loss, Adam without weight
decay, gradient clipping, flat-then-cosine learning rate (``Learner.fit_flat_cos``).
"""
from functools import partial
from pathlib import Path

import numpy as np
import torch
from torch import nn
from tsfast.models import ScaledModel, SimpleRNN, StandardScaler  # package-level import: tsfast 0.4 moved the submodules
from tsfast.training import Learner, RNNLearner
from tsfast.training.losses import fun_rmse
from tsfast.tsdata import create_dls, get_io_size, get_signal_names

from .inputs import INPUT_SETS, TARGET
from .models import TCN, TDNN, SignAntisymmetricWrapper

# AdamW at weight decay 0 is Adam; tsfast's default weight decay would otherwise apply.
OPTIMIZER = partial(torch.optim.AdamW, weight_decay=0.0)


def seed_all(seed: int) -> None:
    import random
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def build_model(cfg: dict, n_in: int, n_out: int) -> nn.Module:
    """The bare network of a config, without input scaling (``build_learner`` wraps it)."""
    model_type = cfg["model"]
    if model_type in ("rnn", "gru", "lstm"):  # as tsfast's RNNLearner builds it
        return SimpleRNN(n_in, n_out, cfg["num_layers"], cfg["hidden_size"], rnn_type=model_type)
    if model_type == "tdnn":
        return TDNN(n_in, n_out, cfg["delay"], cfg["hidden_layer"], cfg["hidden_size"])
    if model_type == "tcn":
        return TCN(n_in, n_out, cfg["num_layers"], cfg["hidden_size"],
                   kernel_size=cfg["tcn_kernel_size"], convs_per_block=cfg["tcn_convs_per_block"])
    raise ValueError(f"unknown model {model_type!r}")


def load_model(cfg: dict, weights: str | Path) -> nn.Module:
    """Network with input scaling from a ``state_dict`` file (the scaler statistics are stored in it), on CPU, eval mode."""
    u_names = INPUT_SETS[cfg["u"]]
    n_in = len(u_names)
    model = ScaledModel(build_model(cfg, n_in, 1), StandardScaler(torch.zeros(n_in), torch.ones(n_in)))
    if cfg.get("sym_wrap"):
        model = SignAntisymmetricWrapper(model, u_names)
    model.load_state_dict(torch.load(weights, map_location="cpu"))
    return model.eval()


def build_learner(cfg: dict, data_dir: str | Path = "data/revsted") -> Learner:
    """tsfast Learner for one config (see ``configs/*.json`` for the keys)."""
    u_names = INPUT_SETS[cfg["u"]]
    # Training windows of win_sz samples drawn with stride stp_sz from every train
    # recording, n_batches_train random batches per epoch; validation windows do not overlap.
    dls = create_dls(u=u_names, y=[TARGET], bs=cfg["batch_size"], win_sz=cfg["win_sz"],
                     stp_sz=cfg["stp_sz"], dataset=Path(data_dir), n_batches_train=cfg["n_batches_train"])
    dls.one_batch()  # materialise once, as the research runs did (this draws from the RNG)
    u_check, y_names = get_signal_names(dls.train)
    assert list(u_check) == u_names and list(y_names) == [TARGET]

    common = dict(loss_func=nn.L1Loss(), opt_func=OPTIMIZER, metrics=[fun_rmse], n_skip=cfg["n_skip"])
    model_type = cfg["model"]
    if model_type in ("rnn", "gru", "lstm"):
        lrn = RNNLearner(dls, rnn_type=model_type, num_layers=cfg["num_layers"],
                         hidden_size=cfg["hidden_size"], input_norm=StandardScaler, output_norm=None, **common)
    else:
        model = build_model(cfg, *get_io_size(dls))
        lrn = Learner(ScaledModel.from_dls(model, dls, StandardScaler, None), dls, **common)
    if cfg.get("sym_wrap"):  # built-in mirror symmetry, outside the scaler (it acts on raw channels)
        lrn.model = SignAntisymmetricWrapper(lrn.model, u_names)

    lrn.grad_clip = float(cfg["grad_clip"])

    # n_skip is a training-loss knob; validation is always scored with valid_skip.
    orig_validate = lrn.validate

    def validate(dl=None, chunk_sz=None):
        saved, lrn.n_skip = lrn.n_skip, cfg["valid_skip"]
        try:
            return orig_validate(dl=dl, chunk_sz=chunk_sz)
        finally:
            lrn.n_skip = saved

    lrn.validate = validate
    return lrn


@torch.no_grad()
def predict(lrn: Learner, files: list[Path]) -> dict[str, np.ndarray]:
    """beta (deg) for each whole recording, run as one sequence from its first sample."""
    u_names, y_names = get_signal_names(lrn.dls.train)
    out = {}
    # cuDNN's fused RNN kernel rejects sequences this long; the native kernel is used instead.
    with torch.backends.cudnn.flags(enabled=False):
        for f in files:
            full = create_dls(u=u_names, y=y_names, dataset={"train": [f], "valid": [f], "test": [f]},
                              win_sz=400, bs=1, n_batches_train=1, n_batches_valid=1)
            preds, _ = lrn.get_preds(dl=full.test)
            out[Path(f).name] = preds[0].float().reshape(-1).cpu().numpy()
    return out
