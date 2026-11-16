"""Particle swarm tuning of the observers' parameters on the training split.

    uv run python scripts/tune_observer.py kf --seeds 0 1 2

Protocol, the same split roles as for the networks: the parameters are fitted on the training
split, the tuning seed is chosen on the validation split, the test split is only scored.

  * Cost: the benchmark metric on the training split, i.e. the RMSE of beta in deg over all
    training recordings pooled, each run as one sequence from its first sample, without the first
    ``SKIP_N`` samples and only where v_x >= ``GATE_KMH``.
  * Search space: log10 of every parameter in the box ``observers.PARAMS`` (all are positive and
    span up to four decades, so a linear search would hardly visit the low ones).
  * Global-best PSO, inertia ``W``, cognitive and social weights ``C1 = C2``, positions clipped to
    the box, a non-finite cost (a diverged filter) counts as the worst possible cost.
  * ``N_RESTARTS`` independent swarms per seed, the one with the lowest training cost is kept (a
    decision on the training split only). One swarm finds the deepest of the EKF's local optima in
    only 15-30 % of the runs (measured); with thirty a miss is below 1 %.
  * The cost of all particles of one iteration is computed in parallel (numba ``prange``), in IEEE
    arithmetic (no fastmath), so a seed gives the same result on every machine.
"""
import time
from pathlib import Path

import numpy as np
from numba import njit, prange

from . import observers as obs
from .data import DATA_DIR, load
from .evaluation import run_estimator
from .inputs import GATE_KMH, SKIP_N, TARGET
from .metrics import score

W, C1, C2 = 0.7, 1.4, 1.4
N_PARTICLES, N_ITERS, N_RESTARTS = 100, 50, 30


def split_arrays(split: str, root=DATA_DIR) -> dict:
    """Filter inputs of all recordings of ``split`` concatenated, with recording starts and the cost mask."""
    recs = load(split, columns=[*obs.CHANNELS, TARGET], root=root)
    parts = [obs.inputs(r, obs.FIXED) for r in recs.values()]
    vx, delta, yaw, ay = (np.concatenate(x) for x in zip(*parts))
    lens = [len(r) for r in recs.values()]
    first = np.concatenate([np.arange(n) >= SKIP_N for n in lens])
    return {"vx": vx, "delta": delta, "yaw": yaw, "ay": ay,
            "ref": np.concatenate([r[TARGET].to_numpy(np.float64) for r in recs.values()]),
            "mask": first & (vx / obs.KMH_TO_MS >= GATE_KMH), "starts": np.cumsum([0, *lens])}


@njit(parallel=True, fastmath=False, cache=True)
def _cost_batch(which, P, vx, delta, yaw, ay, ref, mask, starts, dt, m, Iz, lf, lr, vx_min):
    """Gated RMSE (deg) per row of ``P`` (parameters in natural units, order of ``observers.PARAMS``)."""
    out = np.empty(P.shape[0])
    for p in prange(P.shape[0]):
        sse = 0.0
        n = 0
        for i in range(starts.size - 1):
            a, b = starts[i], starts[i + 1]
            v = vx[a:b]
            if which == 0:
                beta = obs._linkf_beta_core(v, delta[a:b], yaw[a:b], ay[a:b], dt, m, Iz, lf, lr, P[p, 3], P[p, 4],
                                            P[p, 0], P[p, 1], P[p, 2], vx_min, 0.0, 10.0) * obs._DEG
            else:
                vy = obs._ekf_vy_core_scalar(v, delta[a:b], yaw[a:b], ay[a:b], dt, m, Iz, lf, lr, P[p, 6], P[p, 7],
                                             P[p, 0], P[p, 1], P[p, 2], P[p, 3], P[p, 4], P[p, 5], vx_min, 0.0, 10.0)
                beta = np.arctan2(vy, np.maximum(v, obs.VX_FLOOR)) * obs._DEG
            for k in range(b - a):
                if mask[a + k]:
                    g = min(max((v[k] - obs.VX_FLOOR) / (obs.VX_FULL - obs.VX_FLOOR), 0.0), 1.0)
                    e = beta[k] * g - ref[a + k]
                    sse += e * e
                    n += 1
        out[p] = np.sqrt(sse / n)
    return out


def cost(name: str, P: np.ndarray, arr: dict) -> np.ndarray:
    """Gated train RMSE per parameter row ``P`` (natural units); non-finite costs become +inf."""
    f = obs.FIXED
    c = _cost_batch(obs.OBSERVERS.index(name), np.atleast_2d(np.asarray(P, dtype=np.float64)), arr["vx"], arr["delta"],
                    arr["yaw"], arr["ay"], arr["ref"], arr["mask"], arr["starts"], f["dt"], f["m"], f["Iz"], f["lf"],
                    f["lr"], f["vx_min"])
    return np.where(np.isfinite(c), c, np.inf)


def pso(f, lb: np.ndarray, ub: np.ndarray, n_particles: int, n_iters: int, seed):
    """Minimise ``f(X) -> costs`` over the box [lb, ub]; returns (best x, best cost, best cost per iteration)."""
    rng = np.random.default_rng(seed)
    x = rng.uniform(lb, ub, (n_particles, lb.size))
    v = np.zeros_like(x)
    pbest, pcost, hist = x.copy(), np.full(n_particles, np.inf), []
    for _ in range(n_iters):
        c = f(x)
        better = c < pcost  # strict: a non-finite cost (inf) never becomes a personal best
        pbest[better], pcost[better] = x[better], c[better]
        if not np.isfinite(pcost).any():
            raise RuntimeError("every particle diverged; check the search box")
        g = pbest[np.argmin(pcost)]
        hist.append(float(pcost.min()))
        v = W * v + C1 * rng.random(x.shape) * (pbest - x) + C2 * rng.random(x.shape) * (g - x)
        x = np.clip(x + v, lb, ub)
    return g, float(pcost.min()), hist


def tune(name: str, seed: int, n_particles: int = N_PARTICLES, n_iters: int = N_ITERS,
         n_restarts: int = N_RESTARTS, root=DATA_DIR) -> dict:
    """Tune observer ``name`` with seed ``seed``; returns the record (parameters, train cost, valid and test scores).

    The restarts draw from ``SeedSequence(seed).spawn(n_restarts)``, so a seed fixes all of them."""
    root, names = Path(root), list(obs.PARAMS[name])
    lb, ub = (np.log10([obs.PARAMS[name][k][j] for k in names]) for j in (0, 1))
    train = split_arrays("train", root)
    t0 = time.time()
    runs = [pso(lambda X: cost(name, 10.0 ** X, train), lb, ub, n_particles, n_iters, ss)
            for ss in np.random.SeedSequence(seed).spawn(n_restarts)]
    best = int(np.argmin([c for _, c, _ in runs]))
    x, c, hist = runs[best]
    seconds = time.time() - t0
    tuned = dict(zip(names, (10.0 ** x).tolist()))
    p = {**obs.FIXED, **tuned}
    scores = {s: score(run_estimator(lambda r: obs.estimate(name, r, p), root / s, inputs=9), root / s)
              for s in ("valid", "test")}
    pos = (x - lb) / (ub - lb)  # position in the log box, 0..1
    return {"observer": name, "seed": seed, "tuned": tuned, "fixed": obs.FIXED,
            "bound_hits": [k for k, q in zip(names, pos) if q < 0.01 or q > 0.99],
            "train_RMSE_g5": c, "valid": scores["valid"], "test": scores["test"],
            "restarts_train_RMSE_g5": [c for _, c, _ in runs], "restart_kept": best,
            "pso": {"particles": n_particles, "iterations": n_iters, "restarts": n_restarts, "w": W, "c1": C1, "c2": C2,
                    "space": "log10", "box": obs.PARAMS[name], "skip_n": SKIP_N, "gate_kmh": GATE_KMH},
            "history": hist, "seconds": seconds}
