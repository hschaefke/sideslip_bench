"""Input sets, target and evaluation constants of the paper protocol.

Signal names are the HDF5 channel names written by ``scripts/preprocess_data.py``.
ReV-StED's CAN bus carries no longitudinal acceleration, no axle steering angle and
no drive/brake torque. Unit quirk: ``speedo_obd`` is in km/h (only the wheel speeds
are converted to m/s during preprocessing).
"""

# The six input sets of the paper (the numbering follows the research code base, so
# u=4..6 and u=8 are deliberately absent: they are not part of the paper).
INPUT_SETS = {
    # full CAN, 9 signals (Kalyanasundaram et al. 2025, Eq. 16) -- benchmark default
    1: ["speedo_obd", "delta_sw_rad", "yaw_rate_CAN_rads", "acc_y_CAN_ms2", "brake_pressure_obd",
        "wheel_vel_fl_CAN_ms", "wheel_vel_fr_CAN_ms", "wheel_vel_rl_CAN_ms", "wheel_vel_rr_CAN_ms"],
    # u=1 without brake pressure
    2: ["speedo_obd", "delta_sw_rad", "yaw_rate_CAN_rads", "acc_y_CAN_ms2",
        "wheel_vel_fl_CAN_ms", "wheel_vel_fr_CAN_ms", "wheel_vel_rl_CAN_ms", "wheel_vel_rr_CAN_ms"],
    # u=2 without the speedometer (the four wheel speeds remain)
    3: ["delta_sw_rad", "yaw_rate_CAN_rads", "acc_y_CAN_ms2",
        "wheel_vel_fl_CAN_ms", "wheel_vel_fr_CAN_ms", "wheel_vel_rl_CAN_ms", "wheel_vel_rr_CAN_ms"],
    # classical single-track set {v_x, delta, yaw rate, a_y}
    9: ["speedo_obd", "delta_sw_rad", "yaw_rate_CAN_rads", "acc_y_CAN_ms2"],
    # minimal set {v_x, yaw rate, a_y}
    7: ["speedo_obd", "yaw_rate_CAN_rads", "acc_y_CAN_ms2"],
    # no-velocity floor {delta, yaw rate, a_y}
    10: ["delta_sw_rad", "yaw_rate_CAN_rads", "acc_y_CAN_ms2"],
}

# Target: Correvit sideslip angle at the CoG, in degrees (predicted directly).
TARGET = "sideslip_angle_COR_deg"

# Evaluation: each test recording runs as one sequence, the first SKIP_N samples per
# recording are dropped (warm-up of stateful nets), and the paper metrics are gated
# to v_x >= GATE_KMH on the speedometer channel.
SPEED_CHANNEL = "speedo_obd"
SKIP_N = 20
GATE_KMH = 5.0
