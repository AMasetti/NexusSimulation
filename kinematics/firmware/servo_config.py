"""
Servo configuration and calibration for the SpotMicro 3D-printed robot.

── Hardware ──────────────────────────────────────────────────────────
  MCU:    ESP32
  Driver: PCA9685 16-channel I2C PWM board  (address 0x40)
  Servos: MG995  (50 Hz, 500–2500 µs, ±90° mechanical range)

── Calibration workflow ──────────────────────────────────────────────
  1. Run calibrate_servos.py to find each servo's centre/min/max pulse.
  2. Fill in center_us, min_us, max_us, and direction for every entry.
  3. scale_us_per_rad translates kinematics angles (rad) → pulse offset.
     Default 637 µs/rad corresponds to a ±90° (±π/2 rad) servo range
     over a ±1000 µs window from centre.

── Joint ordering ────────────────────────────────────────────────────
  Matches the SpotMicroKinematics output:
    indices 0–2   front_left  (shoulder, hip, knee)
    indices 3–5   front_right (shoulder, hip, knee)
    indices 6–8   rear_left   (shoulder, hip, knee)
    indices 9–11  rear_right  (shoulder, hip, knee)
  Channels 12–13 are reserved for auxiliary servos (e.g. head pan/tilt).
"""

import math

# ── MG995 hardware defaults ───────────────────────────────────────────────────
_MG995_MIN_US    = 500
_MG995_MAX_US    = 2500
_MG995_CENTER_US = 1500
# µs per radian: maps ±π/2 rad → ±1000 µs  →  2000/π ≈ 637 µs/rad
_MG995_SCALE     = 2000.0 / math.pi


def _servo(channel, direction=1, center_us=_MG995_CENTER_US,
           scale_us_per_rad=_MG995_SCALE,
           min_us=_MG995_MIN_US, max_us=_MG995_MAX_US):
    """Convenience constructor for a single servo config dict."""
    return {
        "channel":         channel,
        "direction":       direction,      # +1 or -1 (depends on mounting)
        "center_us":       center_us,      # pulse at joint angle = 0 rad
        "scale_us_per_rad": scale_us_per_rad,
        "min_us":          min_us,         # hardware safety clamp
        "max_us":          max_us,
    }


# ── Servo map — edit channel and direction after physical installation ─────────
# Index in this list = joint index in the kinematics output array.
SERVO_MAP = [
    # ── Front Left ────────────────────────────────────────────────────
    _servo(channel=0,  direction=+1),   # 0  FL shoulder
    _servo(channel=1,  direction=+1),   # 1  FL hip
    _servo(channel=2,  direction=-1),   # 2  FL knee
    # ── Front Right ───────────────────────────────────────────────────
    _servo(channel=3,  direction=-1),   # 3  FR shoulder  (mirrored)
    _servo(channel=4,  direction=-1),   # 4  FR hip
    _servo(channel=5,  direction=+1),   # 5  FR knee
    # ── Rear Left ─────────────────────────────────────────────────────
    _servo(channel=6,  direction=+1),   # 6  RL shoulder
    _servo(channel=7,  direction=+1),   # 7  RL hip
    _servo(channel=8,  direction=-1),   # 8  RL knee
    # ── Rear Right ────────────────────────────────────────────────────
    _servo(channel=9,  direction=-1),   # 9  RR shoulder  (mirrored)
    _servo(channel=10, direction=-1),   # 10 RR hip
    _servo(channel=11, direction=+1),   # 11 RR knee
    # ── Auxiliary (not driven by kinematics) ──────────────────────────
    _servo(channel=12, direction=+1),   # 12 aux / head pan
    _servo(channel=13, direction=+1),   # 13 aux / head tilt
]

# ── Angle → pulse conversion ──────────────────────────────────────────────────

def angle_to_pulse_us(angle_rad: float, cfg: dict) -> float:
    """
    Convert a joint angle (rad) to a PWM pulse width (µs).

    pulse = center_us + direction * angle_rad * scale_us_per_rad
    Result is clamped to [min_us, max_us] to protect the servo.
    """
    pulse = (
        cfg["center_us"]
        + cfg["direction"] * angle_rad * cfg["scale_us_per_rad"]
    )
    return max(cfg["min_us"], min(cfg["max_us"], pulse))


def pulse_us_to_angle(pulse_us: float, cfg: dict) -> float:
    """Inverse of angle_to_pulse_us — used during calibration."""
    return (pulse_us - cfg["center_us"]) / (cfg["direction"] * cfg["scale_us_per_rad"])


# ── Joint limits from kinematics (for reference / validation) ─────────────────
JOINT_LIMITS_RAD = {
    "shoulder": (-0.548,  0.548),
    "hip":      (-2.666,  1.548),
    "knee":     (-2.590,  0.100),
}

# Joint type per index (0=shoulder, 1=hip, 2=knee per leg, repeating)
JOINT_TYPES = ["shoulder", "hip", "knee"] * 4
