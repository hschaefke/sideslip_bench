"""Standard report figure of ``evaluation.evaluate``: error per speed range, beta in a window, error over time.

(i) Per speed range, a bar for the MAE, a tick for the P95 and a dot for the maximum error, the
estimator under test next to the median of the shipped networks (whisker = their range over the
architectures) and the MAE of the two observers (open markers). (ii) Beta in the 60 s test window
at v_x >= 10 km/h with the most corners, chosen from the inputs only (the same window for every
estimator), against the reference, the networks' range and the observers, with the speed below.
(i) and (ii) are the paper's Fig. 2. (iii) The 30 s rolling RMS error over the test timeline
against the networks' range and the observers, with the speed below. Plain matplotlib, so the
release does not depend on a LaTeX installation.
"""
import warnings
from pathlib import Path

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.lines as mlines  # noqa: E402
import matplotlib.patches as mpatches  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from .evaluation import SPEED_BINS, speed_bins  # noqa: E402
from .inputs import GATE_KMH, SKIP_N, SPEED_CHANNEL, TARGET  # noqa: E402

FS = 50.0             # sample rate in Hz
WIN_S = 30.0          # rolling RMS window
DEC = 25              # timeline decimation for plotting (50 Hz -> 2 Hz)
GAP_MIN = 0.6         # gap between recordings on the timeline, min
C_MUT, C_BASE, C_OBS, SHADE = "#e37222", "#bdbdbd", "#3d3d3d", "#efefef"
OBS_STYLE = {"kf": ("s", "--"), "ekf": ("D", ":")}  # marker, line style per observer
WIN_B_S, V_MIN_KMH, AY_THR, CORNER_S = 60.0, 10.0, 2.0, 1.0  # window (ii): length, min speed, corner rule


def corners(ay: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Start and end indices of the corners: |a_y| (1 s moving average) above 2 m/s^2 for at least 1 s."""
    a = np.convolve(ay, np.ones(int(FS)) / FS, "same")
    d = np.diff(np.r_[0, (np.abs(a) > AY_THR).astype(int), 0])
    st, en = np.flatnonzero(d == 1), np.flatnonzero(d == -1)
    keep = en - st >= CORNER_S * FS
    return st[keep], en[keep]


def pick_window(split_dir: str | Path) -> tuple[str, int, int] | None:
    """``(recording, first sample, corners)`` of the 60 s window at v_x >= 10 km/h with the most corners
    that start and end inside it (the earliest on ties), from the inputs only. None if no window qualifies."""
    w, cand = int(WIN_B_S * FS), []
    for f in sorted(Path(split_dir).glob("*.h5")):
        with h5py.File(f, "r") as h:
            ay = np.asarray(h["acc_y_CAN_ms2"], dtype=np.float64)
            vx = np.asarray(h[SPEED_CHANNEL], dtype=np.float64)
        st, en = corners(ay)
        for s0 in range(SKIP_N, vx.size - w, int(FS)):  # 1 s steps
            if vx[s0:s0 + w].min() >= V_MIN_KMH:
                cand.append((-int(np.sum((st >= s0) & (en <= s0 + w))), f.name, s0))
    if not cand:
        return None
    n, rec, s0 = min(cand)
    return rec, s0, -n


def rolling_rms(e: np.ndarray, m: np.ndarray, w: int = int(WIN_S * FS)) -> np.ndarray:
    """Centred rolling RMS over the samples where ``m`` is true; NaN where less than half the window is."""
    c = np.concatenate([[0.0], np.cumsum(np.where(m, e.astype(np.float64) ** 2, 0.0))])
    k = np.concatenate([[0.0], np.cumsum(m.astype(np.float64))])
    i = np.arange(e.size)
    lo, hi = np.clip(i - w // 2, 0, e.size), np.clip(i + w // 2, 0, e.size)
    s, n = c[hi] - c[lo], k[hi] - k[lo]
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(n >= w / 2, np.sqrt(s / n), np.nan)


def _bin_table(runs: list[dict], key: str) -> np.ndarray:
    """``runs x bins`` array of one statistic per speed bin."""
    return np.array([[b.get(key, np.nan) for b in speed_bins(r)] for r in runs])


def draw_report(name: str, per_rec: dict, base: dict, res: dict, out: Path, split_dir: str | Path) -> dict:
    with warnings.catch_warnings():  # all-NaN rolling windows at standstill are expected
        warnings.simplefilter("ignore", RuntimeWarning)
        return _draw(name, per_rec, base, res, out, split_dir)


def _window(ax, axv, per_rec: dict, nets: list, obs: dict, split_dir: str | Path) -> None:
    """Panel (ii): beta in the window of ``pick_window``; predictions = error + reference."""
    win = pick_window(split_dir)
    if win is None:
        ax.text(0.5, 0.5, f"no {WIN_B_S:.0f} s window at v_x ≥ {V_MIN_KMH:g} km/h in the test split",
                transform=ax.transAxes, ha="center", va="center", fontsize=8)
        ax.set_title("(ii) Sideslip angle", fontsize=9, loc="left")
        return
    rec, s0, n = win
    with h5py.File(Path(split_dir) / rec, "r") as h:
        ref = np.asarray(h[TARGET], dtype=np.float32)[SKIP_N:]
    sl = slice(s0 - SKIP_N, s0 - SKIP_N + int(WIN_B_S * FS) + 1)  # error arrays start after the warm-up skip
    r, t = ref[sl], (np.arange(sl.start, sl.stop) + SKIP_N) / FS
    if nets:
        b = np.stack([x[rec][0][sl] + r for x in nets])
        ax.fill_between(t, b.min(0), b.max(0), color=C_BASE, lw=0, zorder=1)
    for o, d in obs.items():
        ax.plot(t, d["errors"][rec][0][sl] + r, color=C_OBS, lw=0.6, ls=OBS_STYLE[o][1], zorder=2)
    ax.plot(t, r, color="k", lw=0.7, zorder=3)
    e, v = per_rec[rec]
    ax.plot(t, e[sl] + r, color=C_MUT, lw=0.8, zorder=4)
    gmax = {k: np.abs(x[0][x[1] >= GATE_KMH]).max(initial=0.0) for k, x in per_rec.items()}
    m = v >= GATE_KMH
    if max(gmax, key=gmax.get) == rec and m.any():  # max error of the method under test inside the window
        i = np.flatnonzero(m)[np.argmax(np.abs(e[m]))]
        if sl.start <= i < sl.stop:
            ax.plot((i + SKIP_N) / FS, 1.0, marker="v", color=C_MUT, ms=5, transform=ax.get_xaxis_transform(),
                    clip_on=False, zorder=5)
    axv.plot(t, v[sl], color="0.4", lw=0.6)
    axv.set_ylim(0, None)
    axv.set_xlim(t[0], t[-1])
    ax.set_ylabel("beta in deg", fontsize=8)
    axv.set_ylabel("v_x in\nkm/h", fontsize=8)
    axv.set_xlabel(f"time in {rec.removesuffix('.h5').replace('ReVSTED_recording_', 'test recording ')} in s", fontsize=8)
    ax.tick_params(labelbottom=False, labelsize=7)
    axv.tick_params(labelsize=7)
    for a in (ax, axv):
        a.grid(True, axis="y", lw=0.4, alpha=0.6)
    ax.set_title(f"(ii) Sideslip angle, {WIN_B_S:.0f} s window at v_x ≥ {V_MIN_KMH:g} km/h with {n} corners",
                 fontsize=9, loc="left")


def _draw(name: str, per_rec: dict, base: dict, res: dict, out: Path, split_dir: str | Path) -> dict:
    nets = [d["errors"] for d in base.values() if d["family"] == "network"]
    obs = {b: d for b, d in base.items() if d["family"] == "observer" and b in OBS_STYLE}
    labels = [f"{lo:g}–{hi:g}" if np.isfinite(hi) else f"≥{lo:g}" for lo, hi in zip(SPEED_BINS[:-1], SPEED_BINS[1:])]
    share = [b["share"] for b in res["speed_bins"]]
    xb = np.arange(len(labels))

    fig = plt.figure(figsize=(10.0, 6.6))
    top = fig.add_gridspec(2, 2, height_ratios=[1, 0.36], width_ratios=[1, 1.35], hspace=0.12, wspace=0.22,
                           left=0.06, right=0.99, bottom=0.53, top=0.865)
    bot = fig.add_gridspec(2, 1, height_ratios=[1, 0.36], hspace=0.12, left=0.075, right=0.99, bottom=0.075, top=0.405)

    # (i) per speed range
    ax = fig.add_subplot(top[:, 0])
    w = 0.36
    ax.axvspan(-0.5, 0.5, color=SHADE, lw=0, zorder=0)
    mut = {k: [b.get(k, np.nan) for b in res["speed_bins"]] for k in ("MAE", "P95", "MaxE")}
    ax.bar(xb + w / 2, mut["MAE"], w, color=C_MUT, edgecolor="k", lw=0.3, zorder=2)
    ax.scatter(xb + w / 2, mut["P95"], marker="_", s=60, color="k", lw=1.2, zorder=4)
    ax.scatter(xb + w / 2, mut["MaxE"], marker="o", s=14, color=C_MUT, edgecolor="k", lw=0.4, zorder=4)
    if nets:
        mae = _bin_table(nets, "MAE")
        med = np.nanmedian(mae, 0)
        ax.bar(xb - w / 2, med, w, color=C_BASE, edgecolor="k", lw=0.3, zorder=2)
        ax.errorbar(xb - w / 2, med, yerr=[med - np.nanmin(mae, 0), np.nanmax(mae, 0) - med], fmt="none",
                    ecolor="k", elinewidth=0.6, capsize=1.5, zorder=3)
        ax.scatter(xb - w / 2, np.nanmedian(_bin_table(nets, "P95"), 0), marker="_", s=60, color="k", lw=1.2, zorder=4)
        ax.scatter(xb - w / 2, np.nanmedian(_bin_table(nets, "MaxE"), 0), marker="o", s=14, color=C_BASE,
                   edgecolor="k", lw=0.4, zorder=4)
    for b, d in obs.items():
        ax.plot(xb - w / 2, _bin_table([d["errors"]], "MAE")[0], ls="none", marker=OBS_STYLE[b][0], ms=4, mfc="white",
                mec=C_OBS, mew=0.9, zorder=5)
    ax.set_yscale("log")
    ax.set_xticks(xb, [f"{lab}\n{s:.0%}" for lab, s in zip(labels, share)], fontsize=7)
    ax.set_xlim(-0.5, len(labels) - 0.5)
    ax.set_xlabel("v_x in km/h (share of test samples)", fontsize=8)
    ax.set_ylabel("error in deg", fontsize=8)
    ax.set_title("(i) Per speed range: bar MAE, tick P95, dot max", fontsize=9, loc="left")
    ax.tick_params(labelsize=7)
    ax.grid(True, axis="y", lw=0.4, alpha=0.6)

    # (ii) beta in the window with the most corners
    axw = fig.add_subplot(top[0, 1])
    _window(axw, fig.add_subplot(top[1, 1], sharex=axw), per_rec, nets, obs, split_dir)

    # (iii) over time: recordings end to end
    axt = fig.add_subplot(bot[0])
    axv = fig.add_subplot(bot[1], sharex=axt)
    t0, bounds, tmax = 0.0, [], None
    emax = max(np.abs(e[v >= GATE_KMH]).max(initial=0.0) for e, v in per_rec.values())
    for rec, (e, v) in per_rec.items():
        m = v >= GATE_KMH
        t = t0 + np.arange(e.size) / FS / 60.0
        sl = slice(None, None, DEC)
        if nets:
            rb = np.stack([rolling_rms(r[rec][0], m) for r in nets])
            axt.fill_between(t[sl], np.nanmin(rb, 0)[sl], np.nanmax(rb, 0)[sl], color=C_BASE, alpha=0.8, lw=0, zorder=1)
        for b, d in obs.items():
            axt.plot(t[sl], rolling_rms(d["errors"][rec][0], m)[sl], color=C_OBS, lw=0.6, ls=OBS_STYLE[b][1], zorder=2)
        axt.plot(t[sl], rolling_rms(e, m)[sl], color=C_MUT, lw=0.8, zorder=3)
        for a in (axt, axv):
            a.fill_between(t[sl], 0, 1, where=~m[sl], transform=a.get_xaxis_transform(), color=SHADE, lw=0, zorder=0)
        axv.plot(t[sl], v[sl], color="0.4", lw=0.6)
        if m.any() and np.abs(e[m]).max() == emax:
            tmax = t[np.flatnonzero(m)[np.argmax(np.abs(e[m]))]]
        bounds.append((rec, t0, t[-1]))
        t0 = t[-1] + GAP_MIN
    for rec, a, _ in bounds:
        axt.text(a + 0.5, 0.96, rec.removesuffix(".h5").replace("ReVSTED_recording_", "rec. "),
                 transform=axt.get_xaxis_transform(), ha="left", va="top", fontsize=7, color="0.4")
    for _, a, _ in bounds[1:]:
        for x in (axt, axv):
            x.axvline(a - GAP_MIN / 2, color="k", lw=0.5)
    if tmax is not None:
        axt.plot(tmax, 1.0, marker="v", color=C_MUT, ms=5, transform=axt.get_xaxis_transform(), clip_on=False, zorder=5)
    axt.set_yscale("log")
    axt.set_ylabel("RMS error\nin deg", fontsize=8)
    axt.tick_params(labelbottom=False, labelsize=7)
    axt.set_title(f"(iii) Over time ({WIN_S:.0f} s rolling RMS, ▼ = max error)", fontsize=9, loc="left")
    axv.axhline(GATE_KMH, color="0.4", lw=0.5, ls=":")
    axv.set_ylim(0, None)
    axv.set_ylabel("v_x in\nkm/h", fontsize=8)
    axv.set_xlabel("test timeline in min", fontsize=8)
    axv.set_xlim(0, bounds[-1][2])
    axv.tick_params(labelsize=7)
    for a in (axt, axv):
        a.grid(True, axis="y", lw=0.4, alpha=0.6)

    m = res["metrics"]
    n_ref = len(res["baselines"])
    rank = f", rank {res['rank']} of {n_ref + 1} by RMSE" if res["rank"] else ""
    fig.suptitle(f"{name}: RMSE {m['RMSE_g5']:.3f}, MAE {m['MAE_g5']:.3f}, P95 {m['P95_g5']:.3f}, "
                 f"max {m['MaxE_g5']:.2f} deg for v_x ≥ {GATE_KMH:g} km/h{rank}", fontsize=10, x=0.06, ha="left", y=0.995)
    h = [mpatches.Patch(color=C_MUT, label=f"{name} (under test)"),
         mlines.Line2D([], [], color="k", lw=0.8, label="reference")]
    if nets:
        h.append(mpatches.Patch(color=C_BASE, label=f"shipped networks ({len(nets)}, median and range)"))
    for b, d in obs.items():
        h.append(mlines.Line2D([], [], color=C_OBS, marker=OBS_STYLE[b][0], mfc="white", ms=4, lw=0.8,
                               ls=OBS_STYLE[b][1], label=d["label"]))
    h.append(mpatches.Patch(color=SHADE, label=f"gated out (v_x < {GATE_KMH:g} km/h)"))
    fig.legend(handles=h, loc="upper left", bbox_to_anchor=(0.055, 0.962), ncol=len(h), fontsize=7.5, frameon=False)

    paths = {"report_pdf": str(out / "report.pdf"), "report_png": str(out / "report.png")}
    fig.savefig(paths["report_pdf"])
    fig.savefig(paths["report_png"], dpi=200)
    plt.close(fig)
    return paths
