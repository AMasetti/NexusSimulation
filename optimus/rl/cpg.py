"""
Central Pattern Generator (CPG) for Optimus biped.

Implements Matsuoka-style coupled oscillators producing a stable walking gait.
Each leg has phase-coupled hip/knee/ankle oscillators. Arms swing counter-phase.

Coordinate frame (from MuJoCo model):
  Forward = +Y,  Up = +Z,  Right = -X (robot's right)

Joint axes (world frame, upright pose):
  Hip-L/R   : Y axis (lateral lean / swing)
  Knee-L/R  : X axis (fore-aft flex)
  Ankle-L/R : Y axis (lateral)

Leg geometry (link lengths in metres):
  thigh = 0.079, shin = 0.108, foot = 0.026

Usage:
    cpg = CPG()
    while True:
        targets = cpg.step(dt)   # dict: joint_name -> target angle (rad)
        imu_pitch = get_imu()
        targets = cpg.stabilise(targets, imu_pitch)
"""

import numpy as np

# ── Gait parameters (tunable) ────────────────────────────────────────────────

FREQ        = 0.8       # Hz — step frequency

# Joint roles (verified from MuJoCo kinematics):
#   Hip   (Z axis) → swings foot in X (lateral)
#   Knee  (Z axis) → swings foot in Y (forward/back) — primary locomotion joint
#   Ankle (Z axis) → rotates foot in place

KNEE_AMP    = 0.25      # rad — knee swing (drives forward motion)
HIP_AMP     = 0.0       # rad — hip lateral: zeroed, causes drift; let RL handle balance
ANKLE_AMP   = 0.06      # rad — small ankle oscillation for natural gait
ARM_AMP     = 0.10      # rad — shoulder counter-swing

# Phase offsets
KNEE_PHASE  = 0.0
ANKLE_PHASE = -np.pi/4

LEFT_PHASE  = 0.0
RIGHT_PHASE = np.pi

PITCH_GAIN  = 0.8       # IMU pitch → knee correction
YAW_GAIN    = 0.5       # heading error → differential knee to steer straight


class CPG:
    """
    Coupled oscillator CPG. dt-independent via explicit phase integration.
    All outputs are joint angle targets in radians.
    """

    def __init__(self, freq=FREQ):
        self.freq    = freq
        self.phase_L = LEFT_PHASE
        self.phase_R = RIGHT_PHASE
        self._omega  = 2 * np.pi * freq

    # ── Public ───────────────────────────────────────────────────────────────

    def step(self, dt: float) -> dict:
        """Advance oscillator by dt seconds, return joint angle targets."""
        self.phase_L = (self.phase_L + self._omega * dt) % (2 * np.pi)
        self.phase_R = (self.phase_R + self._omega * dt) % (2 * np.pi)
        return self._angles()

    def stabilise(self, targets: dict, pitch: float, yaw: float = 0.0) -> dict:
        """
        Reactive corrections using IMU.
        pitch > 0 = leaning forward → pull knees back slightly.
        yaw   > 0 = drifting right  → speed up left leg to steer back.
        """
        pitch_corr = np.clip(-PITCH_GAIN * pitch, -0.15, 0.15)
        yaw_corr   = np.clip(-YAW_GAIN   * yaw,   -0.10, 0.10)

        for k in ("Servo-Knee-L-Top", "Servo-Knee-L-Bottom"):
            targets[k] = np.clip(targets[k] + pitch_corr + yaw_corr, -1.3, 1.3)
        for k in ("Servo-Knee-R-Top", "Servo-Knee-R-Bottom"):
            targets[k] = np.clip(targets[k] + pitch_corr - yaw_corr, -1.3, 1.3)
        return targets

    def set_freq(self, freq: float):
        self.freq   = max(0.1, freq)
        self._omega = 2 * np.pi * self.freq

    # ── Internal ─────────────────────────────────────────────────────────────

    def _leg(self, phase: float) -> tuple:
        knee  = KNEE_AMP  * np.sin(phase + KNEE_PHASE)
        ankle = ANKLE_AMP * np.sin(phase + ANKLE_PHASE)
        return knee, ankle

    def _angles(self) -> dict:
        lk, la = self._leg(self.phase_L)
        rk, ra = self._leg(self.phase_R)

        # Arms counter-swing with opposite leg (natural human gait)
        arm_L = ARM_AMP * np.sin(self.phase_R)
        arm_R = ARM_AMP * np.sin(self.phase_L)

        return {
            # Legs — hip stays zero (no lateral sway, avoids drift)
            "Servo-Hip-L":         0.0,
            "Servo-Knee-L-Top":    lk,
            "Servo-Knee-L-Bottom": lk,
            "Servo-Ankle-L":       la,
            "Servo-Hip-R":         0.0,
            "Servo-Knee-R-Top":    rk,
            "Servo-Knee-R-Bottom": rk,
            "Servo-Ankle-R":       ra,
            "Servo-Hip-Body-Rotation": 0.0,
            # Arms — only front/back swing, forearms locked at zero
            "Servo-Showlder-L-Front-Back":     arm_L,
            "Servo-Showlder-R-Front-Back":     arm_R,
            "Servo-Showlder-L-Inward-Outward": 0.0,
            "Servo-Showlder-R-Inward-Outward": 0.0,
            "Servo-Forearm-L": 0.0,
            "Servo-Forearm-R": 0.0,
        }
