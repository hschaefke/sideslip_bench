"""Model-based comparison observers of the paper: linear KF and EKF on the single-track model.

    import sideslip_bench as ssb
    beta = ssb.observers.estimate("kf", rec)          # rec: DataFrame with the u=9 CAN channels
    ssb.evaluate(ssb.load_baseline("ekf"), name="ekf")

Both take {v_x, steering-wheel angle, yaw rate, a_y} (input set u=9), correct on a_y and the yaw
rate, and return beta in degrees for every sample, causally from the first one. Their parameters
(``configs/observers/<name>.json``) are tuned on the training split by ``sideslip_bench.pso``.
Before the steering ratio is applied, the steering-wheel zero offset of ReV-StED (0.1911 rad, i.e.
10.9 deg straight ahead, estimated on the training split) is subtracted.

  * ``kf``  linear single-track KF, state [beta, r], process noise on the steering input
            (Leanza, Reina, Blanco-Claraco 2021, Sensors 21(16):5409).
  * ``ekf`` single-track EKF, state [v_y, r, b_ay, b_r] with IMU bias random walks, linear tyres.

The kernels are the research code's, compiled without fastmath (IEEE arithmetic) so that every
machine computes the same numbers.
"""
import json
from pathlib import Path

import numpy as np
from numba import njit

ROOT = Path(__file__).resolve().parents[1]
PARAM_DIR = ROOT / "configs" / "observers"
OBSERVERS = ("kf", "ekf")
CHANNELS = ("speedo_obd", "delta_sw_rad", "yaw_rate_CAN_rads", "acc_y_CAN_ms2")  # input set u=9
KMH_TO_MS = 1.0 / 3.6
VX_FLOOR, VX_FULL = 0.5, 2.0  # m/s: beta is ramped from 0 at VX_FLOOR to fully on at VX_FULL
_DEG = 180.0 / np.pi
#: Fixed values: Smart Fortwo geometry and inertia, identified steering ratio, model floor, and
#: the steering-wheel zero offset; the filters start from beta = 0 (v_y = 0), r = yaw[0], P0 = 10 I.
FIXED = {"m": 900.0, "Iz": 700.0, "lf": 1.0488, "lr": 0.8242, "r_s": 20.297, "vx_min": 1.0, "dt": 0.02,
         "delta_sw_offset_rad": 0.1911}
#: Tuned parameters (noise standard deviations, cornering stiffnesses in N/rad) and their search box.
PARAMS = {"kf": {"q_steer": (1e-4, 5.0), "r_ay": (0.1, 10.0), "r_r": (1e-3, 1.0),
                 "Cf": (1e4, 2.5e5), "Cr": (1e4, 2.5e5)},
          "ekf": {"q_vy": (1e-3, 10.0), "q_r": (1e-3, 10.0), "q_bay": (1e-4, 1.0), "q_br": (1e-5, 0.1),
                  "r_ay": (0.1, 10.0), "r_r": (1e-3, 1.0), "Cf": (1e4, 2.5e5), "Cr": (1e4, 2.5e5)}}


@njit(cache=True, fastmath=False)
def _linkf_beta_core(vel_x, steer, yaw, acc_y, dt, m, Iz, lf, lr, Cf, Cr,
                     q_steer, r_ay, r_r, vx_min, b0, P0):
    """Run Leanza's linear single-track KF -> beta[rad] per sample.

    State [beta, r]; predict with delta_{k-1}, correct on z=[a_y, yaw] with delta_k feed-through
    (mirrors BetaStanfordLinear.m exactly). Returns the filtered beta in radians."""
    N = vel_x.shape[0]
    beta_hat = np.zeros(N, dtype=np.float64)

    x0 = float(b0)        # beta
    x1 = float(yaw[0])    # r (init from measured yaw, like ekf_st)

    # 2x2 covariance, flat scalars (numba-friendly).
    P00 = float(P0); P01 = 0.0
    P10 = 0.0;       P11 = float(P0)

    q2 = q_steer * q_steer
    Ray = r_ay * r_ay
    Rr = r_r * r_r

    beta_hat[0] = x0
    for k in range(1, N):
        v = vel_x[k]
        if v < 0.0:
            v = -v
        if v < vx_min:
            v = vx_min
        d_prev = steer[k - 1]   # input used in the prediction
        d = steer[k]            # input feed-through in the measurement

        # --- continuous-time A/B for state [beta, r] (linear single-track) ---
        a11 = -(Cf + Cr) / (m * v)
        a12 = -(1.0 + (Cf * lf - Cr * lr) / (m * v * v))
        a21 = -(Cf * lf - Cr * lr) / Iz
        a22 = -(Cf * lf * lf + Cr * lr * lr) / (Iz * v)
        b1 = Cf / (m * v)
        b2 = Cf * lf / Iz

        # --- Euler-discretized Ad, Bd ---
        Ad00 = 1.0 + dt * a11; Ad01 = dt * a12
        Ad10 = dt * a21;       Ad11 = 1.0 + dt * a22
        Bd0 = dt * b1;         Bd1 = dt * b2

        # --- predict state ---
        x0p = Ad00 * x0 + Ad01 * x1 + Bd0 * d_prev
        x1p = Ad10 * x0 + Ad11 * x1 + Bd1 * d_prev

        # --- predict covariance: Pp = Ad P Ad' + Qd,  Qd = Bd q2 Bd' ---
        AP00 = Ad00 * P00 + Ad01 * P10
        AP01 = Ad00 * P01 + Ad01 * P11
        AP10 = Ad10 * P00 + Ad11 * P10
        AP11 = Ad10 * P01 + Ad11 * P11
        Pp00 = AP00 * Ad00 + AP01 * Ad01 + Bd0 * Bd0 * q2
        Pp01 = AP00 * Ad10 + AP01 * Ad11 + Bd0 * Bd1 * q2
        Pp10 = AP10 * Ad00 + AP11 * Ad01 + Bd1 * Bd0 * q2
        Pp11 = AP10 * Ad10 + AP11 * Ad11 + Bd1 * Bd1 * q2

        # --- measurement model: ay = c00*beta + c01*r + dd0*delta ; yaw_meas = r ---
        c00 = -(Cf + Cr) / m
        c01 = -(Cf * lf - Cr * lr) / (m * v)
        dd0 = Cf / m
        z0_pred = c00 * x0p + c01 * x1p + dd0 * d
        z1_pred = x1p

        y0 = acc_y[k] - z0_pred
        y1 = yaw[k] - z1_pred

        # --- innovation cov S = H Pp H' + R, H = [[c00, c01],[0, 1]] ---
        HP00 = c00 * Pp00 + c01 * Pp10
        HP01 = c00 * Pp01 + c01 * Pp11
        S00 = HP00 * c00 + HP01 * c01 + Ray
        S01 = HP01
        S10 = Pp10 * c00 + Pp11 * c01
        S11 = Pp11 + Rr

        det_s = S00 * S11 - S01 * S10
        if abs(det_s) < 1e-12:
            x0, x1 = x0p, x1p
            P00, P01, P10, P11 = Pp00, Pp01, Pp10, Pp11
            beta_hat[k] = x0
            continue

        # --- Kalman gain K = Pp H' S^-1, Pp H' columns: ---
        PHt00 = Pp00 * c00 + Pp01 * c01; PHt01 = Pp01
        PHt10 = Pp10 * c00 + Pp11 * c01; PHt11 = Pp11
        inv00 = S11 / det_s; inv01 = -S01 / det_s
        inv10 = -S10 / det_s; inv11 = S00 / det_s
        K00 = PHt00 * inv00 + PHt01 * inv10
        K01 = PHt00 * inv01 + PHt01 * inv11
        K10 = PHt10 * inv00 + PHt11 * inv10
        K11 = PHt10 * inv01 + PHt11 * inv11

        # --- correct state ---
        x0 = x0p + K00 * y0 + K01 * y1
        x1 = x1p + K10 * y0 + K11 * y1

        # --- correct covariance: P = (I - K H) Pp ---
        KH00 = K00 * c00 + K01 * 0.0; KH01 = K00 * c01 + K01 * 1.0
        KH10 = K10 * c00 + K11 * 0.0; KH11 = K10 * c01 + K11 * 1.0
        m00 = 1.0 - KH00; m01 = -KH01
        m10 = -KH10;      m11 = 1.0 - KH11
        P00 = m00 * Pp00 + m01 * Pp10
        P01 = m00 * Pp01 + m01 * Pp11
        P10 = m10 * Pp00 + m11 * Pp10
        P11 = m10 * Pp01 + m11 * Pp11

        beta_hat[k] = x0

    return beta_hat


@njit(cache=True, fastmath=False)
def _ekf_vy_core_scalar(vel_x, steer, yaw, acc_y, dt, m, Iz, lf, lr, Cf, Cr,
                        q_vy, q_r, q_bay, q_br, r_ay, r_r, vx_min, v0, P0):
    N = vel_x.shape[0]
    vy_hat = np.zeros(N, dtype=np.float64)

    x0 = float(v0)          # v_y
    x1 = float(yaw[0])      # r
    x2 = 0.0                # b_ay
    x3 = 0.0                # b_r

    # P = P0 * I4, symmetric -> 10 unique entries
    p00 = P0; p01 = 0.0; p02 = 0.0; p03 = 0.0
    p11 = P0; p12 = 0.0; p13 = 0.0
    p22 = P0; p23 = 0.0
    p33 = P0

    qvy2 = q_vy * q_vy; qr2 = q_r * q_r; qbay2 = q_bay * q_bay; qbr2 = q_br * q_br
    ray2 = r_ay * r_ay; rr2 = r_r * r_r

    vy_hat[0] = x0
    for k in range(1, N):
        v = vel_x[k]
        if v < 0.0:
            v = -v
        if v < vx_min:
            v = vx_min
        d = steer[k]

        a11 = -(Cf + Cr) / (m * v)
        a12 = -v - (Cf * lf - Cr * lr) / (m * v)
        a21 = -(Cf * lf - Cr * lr) / (Iz * v)
        a22 = -(Cf * lf * lf + Cr * lr * lr) / (Iz * v)
        b1 = Cf / m
        b2 = Cf * lf / Iz

        # --- predict state ---
        f00 = 1.0 + dt * a11; f01 = dt * a12
        f10 = dt * a21;       f11 = 1.0 + dt * a22
        x0p = f00 * x0 + f01 * x1 + dt * b1 * d
        x1p = f10 * x0 + f11 * x1 + dt * b2 * d
        x2p = x2
        x3p = x3

        # --- predict covariance: top-left 2x2 = A Pa A^T, top-right = A Pb, bottom = Pc ---
        # A Pa  (Pa = [[p00,p01],[p01,p11]])
        m00 = f00 * p00 + f01 * p01; m01 = f00 * p01 + f01 * p11
        m10 = f10 * p00 + f11 * p01; m11 = f10 * p01 + f11 * p11
        # (A Pa) A^T
        pp00 = m00 * f00 + m01 * f01 + qvy2
        pp01 = m00 * f10 + m01 * f11
        pp11 = m10 * f10 + m11 * f11 + qr2
        # A Pb  (Pb = [[p02,p03],[p12,p13]])
        pp02 = f00 * p02 + f01 * p12
        pp03 = f00 * p03 + f01 * p13
        pp12 = f10 * p02 + f11 * p12
        pp13 = f10 * p03 + f11 * p13
        # Pc + bias process noise
        pp22 = p22 + qbay2
        pp23 = p23
        pp33 = p33 + qbr2

        # --- measurement: H = [[h0,h1,1,0],[0,1,0,1]] ---
        h0 = a11; h1 = a12 + v
        z0_pred = h0 * x0p + h1 * x1p + b1 * d + x2p
        z1_pred = x1p + x3p

        # PHt (4x2), using symmetric pp**
        ph00 = pp00 * h0 + pp01 * h1 + pp02; ph01 = pp01 + pp03
        ph10 = pp01 * h0 + pp11 * h1 + pp12; ph11 = pp11 + pp13
        ph20 = pp02 * h0 + pp12 * h1 + pp22; ph21 = pp12 + pp23
        ph30 = pp03 * h0 + pp13 * h1 + pp23; ph31 = pp13 + pp33

        # S = H PHt + R
        s00 = h0 * ph00 + h1 * ph10 + ph20 + ray2
        s01 = h0 * ph01 + h1 * ph11 + ph21
        s10 = ph10 + ph30
        s11 = ph11 + ph31 + rr2

        det_s = s00 * s11 - s01 * s10
        if abs(det_s) < 1e-12:
            x0, x1, x2, x3 = x0p, x1p, x2p, x3p
            p00 = pp00; p01 = pp01; p02 = pp02; p03 = pp03
            p11 = pp11; p12 = pp12; p13 = pp13
            p22 = pp22; p23 = pp23; p33 = pp33
            vy_hat[k] = x0
            continue

        i00 = s11 / det_s; i01 = -s01 / det_s
        i10 = -s10 / det_s; i11 = s00 / det_s

        # K = PHt Sinv (4x2)
        k00 = ph00 * i00 + ph01 * i10; k01 = ph00 * i01 + ph01 * i11
        k10 = ph10 * i00 + ph11 * i10; k11 = ph10 * i01 + ph11 * i11
        k20 = ph20 * i00 + ph21 * i10; k21 = ph20 * i01 + ph21 * i11
        k30 = ph30 * i00 + ph31 * i10; k31 = ph30 * i01 + ph31 * i11

        y0 = acc_y[k] - z0_pred
        y1 = yaw[k] - z1_pred
        x0 = x0p + k00 * y0 + k01 * y1
        x1 = x1p + k10 * y0 + k11 * y1
        x2 = x2p + k20 * y0 + k21 * y1
        x3 = x3p + k30 * y0 + k31 * y1

        # P = (I - K H) P_pred ; (K H)[i,:] = [k_i0*h0, k_i0*h1+k_i1, k_i0, k_i1]
        # ImKH row i = [d_i0 - k_i0*h0, d_i1 - (k_i0*h1+k_i1), d_i2 - k_i0, d_i3 - k_i1]
        a00 = 1.0 - k00 * h0; a01 = -(k00 * h1 + k01); a02 = -k00; a03 = -k01
        a10 = -k10 * h0;      a11r = 1.0 - (k10 * h1 + k11); a12r = -k10; a13 = -k11
        a20 = -k20 * h0;      a21r = -(k20 * h1 + k21);      a22r = 1.0 - k20; a23 = -k21
        a30 = -k30 * h0;      a31r = -(k30 * h1 + k31);      a32 = -k30; a33 = 1.0 - k31
        # P_new[i,j] = ImKH row i . column j of symmetric P_pred; keep upper triangle
        p00 = a00 * pp00 + a01 * pp01 + a02 * pp02 + a03 * pp03
        p01 = a00 * pp01 + a01 * pp11 + a02 * pp12 + a03 * pp13
        p02 = a00 * pp02 + a01 * pp12 + a02 * pp22 + a03 * pp23
        p03 = a00 * pp03 + a01 * pp13 + a02 * pp23 + a03 * pp33
        p11 = a10 * pp01 + a11r * pp11 + a12r * pp12 + a13 * pp13
        p12 = a10 * pp02 + a11r * pp12 + a12r * pp22 + a13 * pp23
        p13 = a10 * pp03 + a11r * pp13 + a12r * pp23 + a13 * pp33
        p22 = a20 * pp02 + a21r * pp12 + a22r * pp22 + a23 * pp23
        p23 = a20 * pp03 + a21r * pp13 + a22r * pp23 + a23 * pp33
        p33 = a30 * pp03 + a31r * pp13 + a32 * pp23 + a33 * pp33

        vy_hat[k] = x0

    return vy_hat


def params(name: str) -> dict:
    """Fixed and tuned parameters of the shipped observer ``name`` (one flat dict)."""
    if name not in OBSERVERS:
        raise ValueError(f"unknown observer {name!r}, choose from {OBSERVERS}")
    cfg = json.loads((PARAM_DIR / f"{name}.json").read_text())
    return {**cfg["fixed"], **cfg["tuned"]}


def inputs(rec, p: dict) -> tuple[np.ndarray, ...]:
    """(v_x in m/s, road-wheel angle in rad, yaw rate, a_y) as float64 arrays."""
    vx = np.asarray(rec["speedo_obd"], dtype=np.float64) * KMH_TO_MS
    delta = (np.asarray(rec["delta_sw_rad"], dtype=np.float64) - p["delta_sw_offset_rad"]) / p["r_s"]
    return vx, delta, np.asarray(rec["yaw_rate_CAN_rads"], dtype=np.float64), np.asarray(rec["acc_y_CAN_ms2"], dtype=np.float64)


def estimate(name: str, rec, p: dict | None = None) -> np.ndarray:
    """beta in deg for every sample of ``rec`` (DataFrame or dict with ``CHANNELS``); ``p`` overrides ``params(name)``."""
    p = params(name) if p is None else p
    vx, delta, yaw, ay = inputs(rec, p)
    car = (p["dt"], p["m"], p["Iz"], p["lf"], p["lr"])
    if name == "kf":
        beta = _linkf_beta_core(vx, delta, yaw, ay, *car, p["Cf"], p["Cr"], p["q_steer"], p["r_ay"], p["r_r"],
                                p["vx_min"], 0.0, 10.0) * _DEG
    elif name == "ekf":
        vy = _ekf_vy_core_scalar(vx, delta, yaw, ay, *car, p["Cf"], p["Cr"], p["q_vy"], p["q_r"], p["q_bay"],
                                 p["q_br"], p["r_ay"], p["r_r"], p["vx_min"], 0.0, 10.0)
        beta = np.arctan2(vy, np.maximum(vx, VX_FLOOR)) * _DEG
    else:
        raise ValueError(f"unknown observer {name!r}, choose from {OBSERVERS}")
    return beta * np.clip((vx - VX_FLOOR) / (VX_FULL - VX_FLOOR), 0.0, 1.0)
