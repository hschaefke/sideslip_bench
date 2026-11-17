"""Self-check of scripts/animate.py on synthetic samples (the drawn steering, one frame without the map,
the convention check), and on the real test split if it has been built (the conventions of the
recording, one --no-map still). Needs neither contextily nor internet.

    uv run python tests/test_animate.py
"""
import importlib.util
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("animate", ROOT / "scripts" / "animate.py")
animate = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(animate)


def test_scene():
    """Front tyres drawn at the true road-wheel angle (gain 1), rear tyres straight; a frame renders."""
    n = 200
    t = np.arange(n) / animate.FS
    D = {"x": 5.0 * t, "y": np.sin(t), "psi": 0.2 * np.sin(t), "v": np.full(n, 8.0), "steer": 2.0 * np.sin(t),
         "ref": 3.0 * np.sin(t), "est": 2.8 * np.sin(t)}
    veh = animate.SMART_FORTWO_453
    sc = animate.Scene(D, veh, "test", (-4.0, 4.0))
    for i in (0, 80, 199):
        sc.update(i)
        want = float(veh.road_wheel(D["steer"][i]))
        for p, _, _, front in sc.tyres:
            M = (p.get_transform() - sc.ax.transData).get_matrix()
            assert abs(np.arctan2(M[1, 0], M[0, 0]) - (want if front else 0.0)) < 1e-9, (i, front, want)
    assert np.isclose(want, (D["steer"][199] - 0.1911) / 20.297)
    with tempfile.TemporaryDirectory() as d:
        sc.fig.savefig(Path(d) / "frame.png", dpi=40)
        assert (Path(d) / "frame.png").stat().st_size > 0


def test_conventions_check():
    """A synthetic drive (INS point 0.92 m behind the CoG) passes; a 90 deg heading offset or a flipped beta fails."""
    t = np.arange(3000) / animate.FS
    r, beta, v = 0.3 * np.sin(0.2 * t), 2.0 * np.sin(0.5 * t), np.full(t.size, 10.0)
    psi = np.cumsum(r) / animate.FS
    course = psi + np.deg2rad(beta)  # of the CoG
    integral = lambda u: (np.cumsum(u) - u / 2) / animate.FS  # noqa: E731  trapezoidal, no half-sample lag
    x = integral(v * np.cos(course)) - 0.92 * np.cos(psi)
    y = integral(v * np.sin(course)) - 0.92 * np.sin(psi)
    a, c, e, _ = animate.check_conventions(x, y, psi, v, r, beta)
    assert abs(a - 1) < 0.05 and abs(c + 0.92) < 0.05 and abs(e) < 0.1, (a, c, e)
    for p, b in ((psi + np.pi / 2, beta), (psi, -beta)):
        try:
            animate.check_conventions(x, y, p, v, r, b)
        except AssertionError:
            continue
        raise AssertionError("wrong convention not caught")


def test_real_window():
    """Conventions hold on the real recording (asserted in animate.load); one --no-map still of the default window."""
    if not any((animate.DATA_DIR / "test").glob("*.h5")):
        print("skip test_real_window: no data/revsted (run ssb.load once)")
        return
    with tempfile.TemporaryDirectory() as d:
        png = Path(d) / "frame.png"
        animate.main(["--no-map", "--out", str(png)])
        assert png.stat().st_size > 0


if __name__ == "__main__":
    test_scene()
    test_conventions_check()
    test_real_window()
    print("ok")
