"""Bird's-eye animation of a sideslip angle estimator on a ReV-StED test recording (the README GIF).

    uv run --extra anim python scripts/animate.py      # docs/birdseye.gif: GRU + symmetry, Fig. 2 window, 32-40 s
    uv run --extra anim python scripts/animate.py --preds my_estimator/test --label "my UKF" --out runs/my_ukf.gif
    uv run python scripts/animate.py --no-map --start 38.26 --out runs/frame.png   # one still, no map

The camera is locked to the car's centre of gravity (CoG), nose to the right; a faded
OpenStreetMap and a world-fixed 2 m dot grid move underneath. Two velocity vectors start at the
CoG, the reference beta (Correvit, black) and the estimator's (orange), each with a wedge against
the body axis. The scene draws the angles x3; the text line and the 10 s strip below show the true
values. The front tyres turn by the true road-wheel angle. A north-up inset shows the track
of the window. --window, --start and --duration select the frames: 50 Hz data, 4 samples per frame
at 12.5 fps, i.e. real time. --preds takes any prediction folder in the bundle format (one
<recording>.h5 per test recording, one 1-D dataset: beta in deg for every sample).

Conventions, asserted on load by ``check_conventions`` on the whole recording:
  * ins_lat_abs / ins_long_abs are WGS84 degrees; ins_yaw is in degrees, counter-clockwise from
    North, so the heading counted from East is psi = ins_yaw + 90 deg;
  * beta is ISO, positive left: the INS point moves along psi + beta minus a lever-arm term, because
    it sits about 0.9 m behind the CoG (INS_TO_COG_M); the camera sits on the CoG;
  * road-wheel angle = (delta_sw_rad - steer_offset) / steer_ratio (``Vehicle.road_wheel``).

Map: OpenStreetMap standard tiles at zoom 19, fetched by contextily (``uv sync --extra anim``;
without it the map is left out). As the OSM tile usage policy asks, the requests name this
application in their User-Agent and the tiles are cached in --tile-cache. Every frame with the
map credits "© OpenStreetMap contributors".
"""
import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np

ROOT = next(p for p in Path(__file__).resolve().parents if (p / "pyproject.toml").exists())
sys.path.insert(0, str(ROOT))

from sideslip_bench.data import DATA_DIR, prepare  # noqa: E402
from sideslip_bench.inputs import TARGET  # noqa: E402
from sideslip_bench.metrics import read_preds  # noqa: E402
from sideslip_bench.observers import FIXED  # noqa: E402
from sideslip_bench.report import C_MUT, FS, WIN_B_S, pick_window  # noqa: E402  (selects the Agg backend)

import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.animation import PillowWriter  # noqa: E402
from matplotlib.colors import to_rgba  # noqa: E402
from matplotlib.patches import Arc, Circle, FancyArrowPatch, FancyBboxPatch, Polygon, Wedge  # noqa: E402
from matplotlib.ticker import MaxNLocator  # noqa: E402
from matplotlib.transforms import Affine2D, blended_transform_factory  # noqa: E402

GAIN = 3.0                             # the scene draws beta x3; text and strip show the true values
FPS, DPI_GIF, DPI_PNG = 12.5, 100, 200
FIG_W, FIG_H, STRIP_H = 9.6, 6.4, 2.0  # in; the scene fills the figure above the strip
VIEW_X = (-3.0, 8.7)                   # scene x range in the car frame in m (x forward, y left, origin CoG)
VEC_LEN, GRID_M = 4.7, 2.0             # velocity-vector length, world-fixed dot grid, in m
STRIP_S = 10.0                         # history shown in the strip, s
INS_TO_COG_M = 0.92                    # ReV-StED: the INS point sits this far behind the CoG (fit of check_conventions)
MAP_ZOOM, MAP_MARGIN_M = 19, 30.0
USER_AGENT = "sideslip_bench-animate"  # OSM rejects anonymous clients; never a browser string
R_EARTH = 6_378_137.0

C_EST, C_REF, C_PAPER = C_MUT, "#000000", np.array([0.965, 0.965, 0.955])
C_BODY, C_BODY_IN, C_EDGE = "#e6e8eb", "#fcfcfb", "#59616a"
C_GLASS, C_HEAD, C_TAIL, C_TYRE = "#2b333b", "#f3e3a0", "#d64541", "#25292e"
C_TEXT, C_MUTED = "#22272c", "#6b7280"


# ------------------------------------------------------------------------------------------------
# Vehicle
# ------------------------------------------------------------------------------------------------
@dataclass(frozen=True)
class Vehicle:
    """Top-view geometry (m) and steering of a car; the defaults are roughly a VW Golf 7."""

    length: float = 4.26          # spec
    width: float = 1.79           # spec, without mirrors
    wheelbase: float = 2.63       # spec
    track_front: float = 1.55     # spec
    track_rear: float = 1.52      # spec
    lf: float = 1.16              # CoG -> front axle: assumed
    front_overhang: float = 0.86  # assumed: 53 % of length - wheelbase
    steer_ratio: float = 15.0     # assumed
    steer_offset: float = 0.0     # steering-wheel angle when driving straight, rad
    tyre_diameter: float = 0.63   # 205/55 R16; drawing only
    tyre_width: float = 0.205

    def road_wheel(self, delta_sw):
        """Road-wheel angle in rad from the steering-wheel angle in rad."""
        return (np.asarray(delta_sw) - self.steer_offset) / self.steer_ratio


# ReV-StED's car. Specs: length, width, wheelbase, tracks, tyres (165/65 R15 front, 185/60 R15 rear;
# the width drawn is their mean). lf, the identified steering ratio and the steering-wheel zero
# (estimated on the training split) are the observers' values. Assumed: the front overhang.
SMART_FORTWO_453 = Vehicle(length=2.695, width=1.663, wheelbase=1.873, track_front=1.469, track_rear=1.430,
                           lf=FIXED["lf"], front_overhang=0.436, steer_ratio=FIXED["r_s"],
                           steer_offset=FIXED["delta_sw_offset_rad"], tyre_diameter=0.60, tyre_width=0.175)
VEHICLES = {"smart": SMART_FORTWO_453, "generic": Vehicle()}


def _se(v, n):  # superellipse power that keeps the sign
    return np.sign(v) * np.abs(v) ** (2.0 / n)


def capsule(x_f, x_r, hw, d_f, d_r, n_f=3.0, n_r=4.0, taper=0.0, k=36) -> np.ndarray:
    """Top-view outline: straight flanks closed by superellipse caps (nose at x_f, tail at x_r)."""
    t = np.linspace(np.pi / 2, -np.pi / 2, k)
    nose = np.column_stack([x_f - d_f + d_f * _se(np.cos(t), n_f), hw * _se(np.sin(t), n_f)])
    t = t[::-1]
    tail = np.column_stack([x_r + d_r - d_r * _se(np.cos(t), n_r), hw * _se(np.sin(t), n_r)])
    p = np.vstack([nose, tail])
    p[:, 1] *= 1.0 - taper * (p[:, 0] - x_r) / (x_f - x_r)   # body a touch narrower at the nose
    return p


def band(pts, depth, ctr) -> np.ndarray:
    """A band of ``depth`` hugging the polyline ``pts`` on its inner side (``ctr`` = body centre)."""
    tng = np.gradient(pts, axis=0)
    nrm = np.column_stack([tng[:, 1], -tng[:, 0]])
    nrm /= np.linalg.norm(nrm, axis=1, keepdims=True)
    nrm *= np.sign(np.sum(nrm * (pts - ctr), axis=1))[:, None]   # outward
    return np.vstack([pts, (pts - depth * nrm)[::-1]])


def build_car(ax, v: Vehicle) -> list:
    """Draw the car in its body frame (x forward, y left, origin CoG); returns [(tyre, x, y, front)]."""
    hw, L, lr = v.width / 2.0, v.length, v.wheelbase - v.lf
    xf = v.lf + v.front_overhang
    xr = xf - L
    ro = -lr - xr                                         # rear overhang
    z = 5.0
    body = capsule(xf, xr, hw, 0.28 * v.width, 0.16 * v.width, taper=0.015)

    for k, g in enumerate((0.16, 0.10, 0.05)):            # soft drop shadow, light from the upper left
        c = body.mean(axis=0)
        ax.add_patch(Polygon(c + (body - c) * (1 + g * 0.35) + [0.07, -0.12], closed=True,
                             fc=(0, 0, 0, 0.06 + 0.02 * k), ec="none", zorder=z - 0.5))
    # two-tone body: slightly darker flank, light core = volume without gradients
    ax.add_patch(Polygon(body, closed=True, fc=C_BODY, ec=C_EDGE, lw=1.1, joinstyle="round", zorder=z))
    core = capsule(xf - 0.05 * L, xr + 0.04 * L, 0.84 * hw, 0.22 * v.width, 0.12 * v.width, taper=0.03)
    ax.add_patch(Polygon(core, closed=True, fc=C_BODY_IN, ec="none", zorder=z + 0.05))

    # greenhouse: one dark glass canopy with the roof floating in it
    cf = v.lf - 0.10 * v.wheelbase                       # windshield base
    cr = xr + max(0.20, 0.45 * ro)                        # rear-glass end
    gh = hw - 0.09 * v.width
    ws, rw = 0.27 * v.wheelbase, 0.16 * v.wheelbase
    glass = capsule(cf, cr, gh, 0.30 * gh, 0.12 * gh, n_f=2.4, n_r=3.0)
    roof = capsule(cf - ws, cr + rw, gh - 0.10, 0.10 * gh, 0.06 * gh, n_f=2.6, n_r=3.0)
    ax.add_patch(Polygon(glass, closed=True, fc=C_GLASS, ec=C_EDGE, lw=0.8, joinstyle="round", zorder=z + 0.2))
    ax.add_patch(Polygon(roof, closed=True, fc=C_BODY_IN, ec=C_EDGE, lw=0.6, joinstyle="round", zorder=z + 0.3))
    ax.add_patch(Polygon([[cf - 0.05, 0.10 * gh], [cf - 0.02, 0.62 * gh], [cf - ws + 0.08, 0.40 * gh],
                          [cf - ws + 0.08, 0.02 * gh]], closed=True, fc=(1, 1, 1, 0.22), ec="none", zorder=z + 0.25))

    # lamps: thin strips hugging the nose (warm white) and the tail (red); mirrors at the windshield base
    edge = capsule(xf - 0.025, xr + 0.025, hw - 0.025, 0.28 * v.width, 0.16 * v.width, taper=0.015)
    nose, tail = edge[edge[:, 0] > 0.5 * (xf + v.lf)], edge[edge[:, 0] < 0.5 * (xr - lr)]
    mid = np.array([0.5 * (xf + xr), 0.0])
    for s in (1, -1):
        h = nose[(s * nose[:, 1] > 0.42 * hw) & (s * nose[:, 1] < 0.86 * hw)]
        t = tail[(s * tail[:, 1] > 0.36 * hw) & (s * tail[:, 1] < 0.90 * hw)]
        ax.add_patch(Polygon(band(h, 0.07, mid), closed=True, fc=C_HEAD, ec=C_EDGE, lw=0.6, zorder=z + 0.4))
        ax.add_patch(Polygon(band(t, 0.06, mid), closed=True, fc=C_TAIL, ec=C_EDGE, lw=0.6, zorder=z + 0.4))
        ax.add_patch(FancyBboxPatch((cf - 0.22, s * (hw - 0.04) - (0.15 if s < 0 else 0)), 0.17, 0.15,
                                    boxstyle="round,pad=0,rounding_size=0.05", fc=C_BODY_IN, ec=C_EDGE,
                                    lw=0.8, zorder=z - 0.1))

    # tyres whole, on top of the body with a light rim, at the true track: the steer angle reads at its size
    tyres, tl, tw = [], v.tyre_diameter, v.tyre_width
    for x, tr, front in ((v.lf, v.track_front, True), (-lr, v.track_rear, False)):
        for s in (1, -1):
            p = FancyBboxPatch((x - tl / 2, s * tr / 2 - tw / 2), tl, tw, boxstyle="round,pad=0,rounding_size=0.07",
                               fc=C_TYRE, ec=C_BODY_IN, lw=1.0, alpha=0.9, zorder=z + 0.35)
            ax.add_patch(p)
            tyres.append((p, x, s * tr / 2, front))
    return tyres


# ------------------------------------------------------------------------------------------------
# Data
# ------------------------------------------------------------------------------------------------
CHANNELS = ("ins_lat_abs", "ins_long_abs", "ins_yaw", "vel_x_ADMA_ms", "vel_y_ADMA_ms", "yaw_rate_CAN_rads",
            "delta_sw_rad", TARGET)


def deg(b: float) -> str:
    """'+4.6°', '−2.2°', ' 0.0°' (a figure space keeps the column steady; no '−0.0')."""
    t = f"{b:+.1f}"
    return ("\u2007" + t[1:] if float(t) == 0 else t.replace("-", "\u2212")) + "°"


def wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


def project(lat, lon, lat_c, lon_c):
    """WGS84 degrees -> local metres (x East, y North) around (lat_c, lon_c)."""
    # ponytail: equirectangular, exact to well below a pixel over the test site; a real projection if ever km-scale
    return (R_EARTH * np.deg2rad(lon - lon_c) * np.cos(np.deg2rad(lat_c)), R_EARTH * np.deg2rad(lat - lat_c))


def check_conventions(x, y, psi, v, r, beta, k: int = 5) -> tuple[float, float, float, float]:
    """Fit course - psi = a*beta + c*r/v + e over the samples above 5 m/s and assert the conventions.

    The course is that of the INS point over +-k samples. Heading (psi = ins_yaw + 90 deg): e ~ 0 and a
    small residual. If beta varies enough (std >= 1 deg): a ~ +1 (ISO beta, positive left) and c ~ -0.9 m
    (the Correvit beta holds at the CoG, the INS point sits behind it). Returns (a, c, e in deg, rms in deg).
    """
    m = v[k:-k] >= 5.0
    if m.sum() < FS:
        raise ValueError("less than 1 s above 5 m/s: cannot check the conventions")
    d = wrap(np.arctan2(y[2 * k:] - y[:-2 * k], x[2 * k:] - x[:-2 * k]) - psi[k:-k])[m]
    A = np.column_stack([np.deg2rad(beta), r / np.maximum(v, 1.0), np.ones_like(v)])[k:-k][m]
    (a, c, e), *_ = np.linalg.lstsq(A, d, rcond=None)
    rms, e = np.rad2deg(np.sqrt(np.mean((d - A @ (a, c, e)) ** 2))), np.rad2deg(e)
    assert abs(e) < 1.0 and rms < 0.5, \
        f"heading convention wrong (psi = ins_yaw + 90 deg?): offset {e:+.2f} deg, rms {rms:.2f} deg"
    if np.std(beta[k:-k][m]) >= 1.0:
        assert 0.6 < a < 1.4, f"beta does not match the track (ISO, positive left?): course - psi = {a:+.2f} beta"
        assert -1.3 < c < -0.5, f"INS point {-c:+.2f} m behind the CoG, expected about {INS_TO_COG_M} m"
    return a, c, e, rms


def load(path: Path, est: np.ndarray, i0: int, n: int) -> dict:
    """Samples [i0, i0 + n) of one recording and its predictions ``est``; positions in local metres around the
    window, moved from the INS point to the CoG. Checks the conventions on the whole recording first."""
    with h5py.File(path, "r") as h:
        g = {k: np.asarray(h[k], dtype=np.float64) for k in CHANNELS}
    if est.shape != g[TARGET].shape:
        raise ValueError(f"{path.name}: {est.size} predictions for {g[TARGET].size} samples")
    sl = slice(i0, i0 + n)
    lat, lon = g["ins_lat_abs"], g["ins_long_abs"]
    centre = float(lat[sl].mean()), float(lon[sl].mean())
    x, y = project(lat, lon, *centre)
    psi = np.unwrap(np.deg2rad(g["ins_yaw"] + 90.0))
    v = np.hypot(g["vel_x_ADMA_ms"], g["vel_y_ADMA_ms"])
    a, c, e, rms = check_conventions(x, y, psi, v, g["yaw_rate_CAN_rads"], g[TARGET])
    print(f"conventions of {path.name}: course - psi = {a:+.2f} beta {c:+.2f} m r/v {e:+.2f} deg (rms {rms:.2f} deg)")
    x, y = x + INS_TO_COG_M * np.cos(psi), y + INS_TO_COG_M * np.sin(psi)
    return {"x": x[sl], "y": y[sl], "psi": psi[sl], "v": v[sl], "steer": g["delta_sw_rad"][sl], "ref": g[TARGET][sl],
            "est": est.astype(np.float64)[sl], "lat": lat[sl], "lon": lon[sl], "centre": centre}


# ------------------------------------------------------------------------------------------------
# Map
# ------------------------------------------------------------------------------------------------
def _filt(a, f, k):
    from numpy.lib.stride_tricks import sliding_window_view
    return f(sliding_window_view(np.pad(a, k // 2, mode="edge"), (k, k)), axis=(-2, -1))


def scrub_strokes(rgb, k=7):
    """Remove thin, dark, grey strokes (area labels, rail lines) from the tile mosaic.

    At this zoom one tile pixel spans ~15 screen pixels, so the area label of the test site turns into
    giant blurred letters under the car. A grey closing finds the strokes and fills them with the closed
    colour; saturated lines (the green site fence) and light ones (roads, parking bays) survive.
    """
    lum = rgb.mean(axis=2)
    stroke = (_filt(_filt(lum, np.max, k), np.min, k) - lum > 0.05) & (np.ptp(rgb, axis=2) < 0.25)
    stroke = _filt(stroke, np.max, 3)                                  # catch anti-aliased rims
    closed = np.stack([_filt(_filt(rgb[..., c], np.max, k), np.min, k) for c in range(3)], axis=-1)
    return np.where(stroke[..., None], closed, rgb)


def basemap(D: dict, cache: Path, user_agent: str = USER_AGENT):
    """OSM tile mosaic around the window: (rgb in [0, 1], extent in local m), or None without contextily."""
    try:
        import contextily as cx
    except ImportError:
        print("contextily is not installed (uv sync --extra anim): drawing without the map")
        return None
    cx.set_cache_dir(str(cache))
    lat_c, lon_c = D["centre"]
    dlat = MAP_MARGIN_M / 111_320.0
    dlon = dlat / np.cos(np.deg2rad(lat_c))
    try:
        img, ext = cx.bounds2img(D["lon"].min() - dlon, D["lat"].min() - dlat, D["lon"].max() + dlon,
                                 D["lat"].max() + dlat, zoom=MAP_ZOOM, ll=True,
                                 source=cx.providers.OpenStreetMap.Mapnik, headers={"user-agent": user_agent})
    except Exception as exc:
        raise SystemExit(f"could not load the map tiles ({exc}); --no-map draws without them") from exc
    lon_lr = np.rad2deg(np.array(ext[:2]) / R_EARTH)                    # Web Mercator -> WGS84
    lat_bt = np.rad2deg(2 * np.arctan(np.exp(np.array(ext[2:]) / R_EARTH)) - np.pi / 2)
    X, _ = project(np.full(2, lat_c), lon_lr, lat_c, lon_c)
    _, Y = project(lat_bt, np.full(2, lon_c), lat_c, lon_c)
    return scrub_strokes(img[..., :3] / 255.0), (X[0], X[1], Y[0], Y[1])


def fade(rgb, keep, sat):
    """Desaturate to ``sat`` and keep ``keep`` of the colour; the rest is paper."""
    rgb = sat * rgb + (1 - sat) * rgb.mean(axis=2, keepdims=True)
    return C_PAPER * (1 - keep) + rgb * keep


# ------------------------------------------------------------------------------------------------
# Scene
# ------------------------------------------------------------------------------------------------
class Scene:
    """One frame: the car-locked scene on top, the beta strip below; ``update(i)`` shows sample i of ``D``.

    ``tiles``: the output of ``basemap()`` or None for a plain background; ``ylim``: the strip's beta range."""

    def __init__(self, D: dict, veh: Vehicle, label: str, ylim: tuple[float, float], tiles=None):
        self.D, self.veh, self.label, self.ylim = D, veh, label, ylim
        self.fig = fig = plt.figure(figsize=(FIG_W, FIG_H), facecolor=C_PAPER)
        W, H = FIG_W, FIG_H - STRIP_H                                    # scene size, in
        fx = lambda x: x / FIG_W  # noqa: E731  scene inches -> figure fraction
        fy = lambda y: (STRIP_H + y) / FIG_H  # noqa: E731
        self.ax = ax = fig.add_axes((fx(0), fy(0), W / FIG_W, H / FIG_H))
        hy = (VIEW_X[1] - VIEW_X[0]) * H / W / 2
        self.view = (*VIEW_X, -hy, hy)
        ax.set_xlim(VIEW_X)
        ax.set_ylim(-hy, hy)
        ax.set_aspect("equal")
        ax.set_autoscale_on(False)
        ax.axis("off")

        self.map = None
        if tiles is not None:
            raw, self.map_ext = tiles
            self.map_rgb = fade(raw, 0.45, 0.35)
            self.map = ax.imshow(self.map_rgb[:2, :2], extent=(0, 1, 0, 1), origin="upper",
                                 interpolation="bilinear", zorder=0)
            fig.text(fx(W - 0.11), fy(0.09), "© OpenStreetMap contributors", fontsize=8.5, color=C_MUTED,
                     ha="right", va="bottom", zorder=10,
                     bbox=dict(boxstyle="round,pad=0.25,rounding_size=0.3", fc=(1, 1, 1, 0.75), ec="none"))
        fig.text(fx(W - 0.11), fy(0.42), f"angles drawn ×{GAIN:g}", fontsize=10.5, color=C_TEXT,
                 ha="right", va="bottom", zorder=10)

        gx = np.arange(np.floor(D["x"].min() / GRID_M) - 4, np.ceil(D["x"].max() / GRID_M) + 5) * GRID_M
        gy = np.arange(np.floor(D["y"].min() / GRID_M) - 4, np.ceil(D["y"].max() / GRID_M) + 5) * GRID_M
        self.grid_w = np.array(np.meshgrid(gx, gy)).reshape(2, -1).T
        self.grid = ax.scatter([], [], s=5, c=[(0.2, 0.22, 0.25, 0.30)], lw=0, zorder=1.8)

        # body axis (beta = 0) and the two wedges, below the car so that they emerge at the nose
        ax.plot([0, VEC_LEN * 1.08], [0, 0], color=C_MUTED, lw=0.9, ls=(0, (4, 3)), zorder=3)
        rw = 0.80 * VEC_LEN
        self.w_ref = Wedge((0, 0), rw, 0, 0, fc=(0, 0, 0, 0.12), ec="none", zorder=3.1)
        self.w_est = Wedge((0, 0), rw, 0, 0, fc=to_rgba(C_EST, 0.36), ec="none", zorder=3.2)
        self.a_ref = Arc((0, 0), 2 * rw, 2 * rw, theta1=0, theta2=0, color=C_REF, lw=1.2, zorder=3.3)
        for p in (self.w_ref, self.w_est, self.a_ref):
            ax.add_patch(p)

        self.tyres = build_car(ax, veh)

        # vectors on top: thick orange estimate, thin black reference inside it; CoG symbol
        self.v_est = FancyArrowPatch((0, 0), (VEC_LEN, 0), arrowstyle="-|>", mutation_scale=24, lw=3.8,
                                     color=C_EST, shrinkA=0, shrinkB=0, joinstyle="miter", zorder=8)
        self.v_ref = FancyArrowPatch((0, 0), (VEC_LEN, 0), arrowstyle="-|>", mutation_scale=14, lw=1.8,
                                     color=C_REF, shrinkA=0, shrinkB=0, joinstyle="miter", zorder=8.1)
        ax.add_patch(self.v_est)
        ax.add_patch(self.v_ref)
        ax.add_patch(Circle((0, 0), 0.12, fc="white", ec=C_TEXT, lw=1.0, zorder=8.2))
        for a0 in (0, 180):
            ax.add_patch(Wedge((0, 0), 0.12, a0, a0 + 90, fc=C_TEXT, ec="none", zorder=8.3))

        # one text line, colour-coded = legend; the white box fits the estimator's label
        kw = dict(fontsize=13, va="center", ha="left", transform=fig.transFigure, zorder=10)
        self.t_v = fig.text(fx(0.23), fy(H - 0.405), "", color=C_TEXT, **kw)
        self.t_ref = fig.text(fx(1.21), fy(H - 0.405), "", color=C_REF, **kw)
        self.t_est = fig.text(fx(2.92), fy(H - 0.405), f"β {label}  {deg(-8.8)}", color=C_EST, fontweight="bold", **kw)
        w_est = self.t_est.get_window_extent(fig.canvas.get_renderer()).width / fig.dpi
        fig.patches.append(FancyBboxPatch((fx(0.11), fy(H - 0.61)), (2.81 + w_est + 0.51) / FIG_W, 0.41 / FIG_H,
                                          transform=fig.transFigure, boxstyle="round,pad=0,rounding_size=0.012",
                                          fc=(1, 1, 1, 0.80), ec="none", zorder=9))

        # track inset, north up, top right
        x, y, mg = D["x"], D["y"], 8.0
        lim = (x.min() - mg, x.max() + mg, y.min() - mg, y.max() + mg)
        iw = min(1.8, 1.6 * (lim[1] - lim[0]) / (lim[3] - lim[2]))     # at most 1.8 in wide, 1.6 in tall
        ih, pad = iw * (lim[3] - lim[2]) / (lim[1] - lim[0]), 0.108
        x0, y0 = W - 0.216 - iw, H - 0.2025 - ih
        fig.patches.append(FancyBboxPatch((fx(x0 - pad), fy(y0 - pad)), (iw + 2 * pad) / FIG_W, (ih + 2 * pad) / FIG_H,
                                          transform=fig.transFigure, boxstyle="round,pad=0,rounding_size=0.015",
                                          fc=(1, 1, 1, 0.82), ec="#d6d9dd", lw=0.8, zorder=9))
        ia = fig.add_axes((fx(x0), fy(y0), iw / FIG_W, ih / FIG_H))
        ia.set_zorder(10)
        ia.patch.set_alpha(0)
        ia.axis("off")
        if tiles is not None:
            ia.imshow(fade(raw, 0.75, 0.6), extent=self.map_ext, origin="upper", interpolation="antialiased")
        ia.set_xlim(lim[:2])
        ia.set_ylim(lim[2:])
        ia.set_aspect("equal")
        ia.plot(x, y, color="#5f666d", lw=1.2, solid_capstyle="round")
        self.dot, = ia.plot([], [], "o", ms=6.5, mfc=C_TEXT, mec="white", mew=1.4, zorder=3)

        # strip: the last STRIP_S seconds, the live values at the line ends
        self.sx = sx = fig.add_axes((0.78 / FIG_W, 0.42 / FIG_H, 8.0 / FIG_W, 1.38 / FIG_H))
        sx.set_facecolor("none")
        for s in ("top", "right"):
            sx.spines[s].set_visible(False)
        for s in ("left", "bottom"):
            sx.spines[s].set_color("#a3a8ad")
            sx.spines[s].set_linewidth(0.8)
        sx.tick_params(colors=C_MUTED, labelsize=11, length=3, width=0.8)
        sx.axhline(0, color="#cfd3d7", lw=0.8, zorder=0)
        sx.set_xlim(-STRIP_S, 0)
        sx.set_xticks([-STRIP_S, -STRIP_S / 2, 0], [f"−{STRIP_S:g} s", f"−{STRIP_S / 2:g} s", "now"])
        sx.set_ylim(ylim)
        sx.yaxis.set_major_locator(MaxNLocator(5, steps=[1, 2, 5, 10]))
        sx.set_ylabel("β in deg", color=C_TEXT, fontsize=11, labelpad=3)
        style = {"est": dict(color=C_EST, lw=2.6), "ref": dict(color=C_REF, lw=1.3)}   # thin black on top
        self.lines = {k: sx.plot([], [], solid_capstyle="round", **st)[0] for k, st in style.items()}
        self.dots = {k: sx.plot([], [], "o", ms=5.5, color=st["color"], mec="white", mew=0.8, clip_on=False,
                                zorder=6)[0] for k, st in style.items()}
        tr = blended_transform_factory(sx.transAxes, sx.transData)
        self.vals = {k: sx.text(1.02, 0, "", transform=tr, color=style[k]["color"], fontsize=12, va="center",
                                ha="left", fontweight="bold" if k == "est" else "normal") for k in ("ref", "est")}

    def update(self, i: int) -> None:
        D, V = self.D, self.view
        x, y, psi = D["x"][i], D["y"][i], D["psi"][i]
        world = Affine2D().translate(-x, -y).rotate(-psi) + self.ax.transData   # world m -> car frame -> screen
        if self.map is not None:  # crop the mosaic around the car, then rotate it with the world
            x0, x1, y0, y1 = self.map_ext
            Hh, Ww = self.map_rgb.shape[:2]
            R = 0.75 * np.hypot(V[1] - V[0], V[3] - V[2])
            c0 = int(np.clip((x - R - x0) / (x1 - x0) * Ww, 0, Ww - 2))
            c1 = int(np.clip(np.ceil((x + R - x0) / (x1 - x0) * Ww), c0 + 2, Ww))
            r0 = int(np.clip((y1 - (y + R)) / (y1 - y0) * Hh, 0, Hh - 2))
            r1 = int(np.clip(np.ceil((y1 - (y - R)) / (y1 - y0) * Hh), r0 + 2, Hh))
            self.map.set_data(self.map_rgb[r0:r1, c0:c1])
            self.map.set_extent((x0 + c0 / Ww * (x1 - x0), x0 + c1 / Ww * (x1 - x0),
                                 y1 - r1 / Hh * (y1 - y0), y1 - r0 / Hh * (y1 - y0)))
            self.map.set_transform(world)
        c, s = np.cos(psi), np.sin(psi)
        g = (self.grid_w - (x, y)) @ np.array([[c, -s], [s, c]])               # world -> car frame
        self.grid.set_offsets(g[(g[:, 0] > V[0] - 1) & (g[:, 0] < V[1] + 1) & (np.abs(g[:, 1]) < V[3] + 1)])

        delta = float(self.veh.road_wheel(D["steer"][i]))                     # true road-wheel angle, gain 1
        for p, xt, yt, front in self.tyres:
            if front:
                p.set_transform(Affine2D().rotate_around(xt, yt, delta) + self.ax.transData)

        br, be = D["ref"][i], D["est"][i]
        for w, a in ((self.w_ref, GAIN * br), (self.w_est, GAIN * be)):
            w.set_theta1(min(0, a))
            w.set_theta2(max(0, a))
        self.a_ref.theta1, self.a_ref.theta2 = min(0, GAIN * br), max(0, GAIN * br)
        for arrow, a in ((self.v_ref, np.deg2rad(GAIN * br)), (self.v_est, np.deg2rad(GAIN * be))):
            arrow.set_positions((0, 0), (VEC_LEN * np.cos(a), VEC_LEN * np.sin(a)))
        self.t_v.set_text(f"{D['v'][i] * 3.6:.0f} km/h")
        self.t_ref.set_text(f"β reference  {deg(br)}")
        self.t_est.set_text(f"β {self.label}  {deg(be)}")
        self.dot.set_data([x], [y])

        j = max(0, i - int(STRIP_S * FS))
        t = (np.arange(j, i + 1) - i) / FS
        for k in ("ref", "est"):
            self.lines[k].set_data(t, D[k][j:i + 1])
            self.dots[k].set_data([0], [D[k][i]])
        gap = 0.13 * (self.ylim[1] - self.ylim[0])                            # labels pushed apart when they touch
        if abs(br - be) < gap:
            m, sgn = 0.5 * (br + be), (1 if br >= be else -1)
            br, be = m + sgn * gap / 2, m - sgn * gap / 2
        for k, yv in (("ref", br), ("est", be)):
            self.vals[k].set_y(yv)
            self.vals[k].set_text(deg(D[k][i]))


# ------------------------------------------------------------------------------------------------
# CLI
# ------------------------------------------------------------------------------------------------
def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--preds", type=Path, default=ROOT / "results/symmetric_preds/gru",
                    help="prediction folder in the bundle format (default: results/symmetric_preds/gru)")
    ap.add_argument("--label", default="GRU + symmetry", help="name of the estimator in the frame")
    ap.add_argument("--recording", help="test recording, e.g. ReVSTED_recording_7.h5 (default: that of the "
                                        "paper's Fig. 2 window, sideslip_bench.report.pick_window)")
    ap.add_argument("--window", type=float, help="start of the 60 s window shown in the inset, in s from the "
                                                 "recording start (default: the Fig. 2 window; 0 with --recording)")
    ap.add_argument("--start", type=float, default=32.0, help="first frame, in s after the window start (32)")
    ap.add_argument("--duration", type=float, default=8.0, help="length in s (8)")
    ap.add_argument("--vehicle", choices=VEHICLES, default="smart",
                    help="car drawn: the Smart Fortwo 453 of ReV-StED, or generic (about a VW Golf)")
    ap.add_argument("--no-map", action="store_true", help="plain background, no OpenStreetMap")
    ap.add_argument("--tile-cache", type=Path, default=ROOT / "data/osm_tiles", help="map tile cache (data/osm_tiles)")
    ap.add_argument("--user-agent", default=USER_AGENT, help=f"User-Agent of the tile requests ({USER_AGENT})")
    ap.add_argument("--data", type=Path, default=DATA_DIR, help="dataset root (data/revsted)")
    ap.add_argument("--out", type=Path, default=ROOT / "docs/birdseye.gif",
                    help=".gif: the animation (default: docs/birdseye.gif); .png: a still of the first frame")
    a = ap.parse_args(argv)

    test_dir = prepare(a.data, ("test",)) / "test"
    rec, i0 = a.recording, round((a.window or 0.0) * FS)
    if rec is None:
        win = pick_window(test_dir)
        if win is None:
            raise SystemExit("no window qualifies in the test split: pass --recording")
        rec, s0 = win[:2]
        i0 = s0 if a.window is None else i0
    preds = read_preds(a.preds)
    if rec not in preds:
        raise SystemExit(f"{a.preds} holds no {rec}")
    D = load(test_dir / rec, preds[rec], i0, round(max(WIN_B_S, a.start + a.duration) * FS))
    frames = range(round(a.start * FS), round((a.start + a.duration) * FS), round(FS / FPS))
    if a.start < 0 or not frames or frames.stop > D["x"].size:
        raise SystemExit(f"--start/--duration select no frames inside {rec}")
    j = max(0, frames[0] - int(STRIP_S * FS))
    b = np.r_[D["ref"][j:frames[-1] + 1], D["est"][j:frames[-1] + 1]]
    pad = 0.11 * max(np.ptp(b), 4.0)

    m = None if a.no_map else basemap(D, a.tile_cache, a.user_agent)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    with plt.rc_context({"font.sans-serif": ["Helvetica Neue", "Arial", *plt.rcParams["font.sans-serif"]]}):
        sc = Scene(D, VEHICLES[a.vehicle], a.label, (b.min() - pad, b.max() + pad), m)
        if a.out.suffix == ".png":
            sc.update(frames[0])
            sc.fig.savefig(a.out, dpi=DPI_PNG)
            print(f"wrote {a.out}: {rec}, {(i0 + frames[0]) / FS:.2f} s")
            return
        w = PillowWriter(fps=FPS)
        with w.saving(sc.fig, a.out, DPI_GIF):
            for i in frames:
                sc.update(i)
                w.grab_frame()
    print(f"wrote {a.out}: {rec} {(i0 + frames[0]) / FS:.1f}-{(i0 + frames.stop) / FS:.1f} s, {len(frames)} frames "
          f"at {FPS:g} fps, {FIG_W * DPI_GIF:.0f}x{FIG_H * DPI_GIF:.0f} px, {a.out.stat().st_size / 1e6:.2f} MB")


if __name__ == "__main__":
    main()
