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
#   Hip   (Z axis) → swings foot in X (lateral). Use for lateral balance, small amp.
#   Knee  (Z axis) → swings foot in Y (forward/back). This drives walking.
#   Ankle (Z axis) → rotates foot, no CoM displacement. Use for push-off timing.

KNEE_AMP    = 0.25      # rad — knee swing amplitude (drives forward motion)
HIP_AMP     = 0.06      # rad — hip lateral sway (small, for balance)
ANKLE_AMP   = 0.08      # rad — ankle push-off at end of stance
ARM_AMP     = 0.12      # rad — arm counter-swing

# Phase offsets
KNEE_PHASE  = 0.0       # knee is the primary driver — no offset
ANKLE_PHASE = -np.pi/4  # ankle pushes off slightly after knee peak
HIP_PHASE   = np.pi/2   # hip sway peaks at mid-swing (balance)

LEFT_PHASE  = 0.0
RIGHT_PHASE = np.pi     # legs anti-phase

PITCH_GAIN  = 1.0       # IMU pitch → knee correction (forward lean → extend knee)


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

    def stabilise(self, targets: dict, pitch: float) -> dict:
        """
        Reactive pitch correction using IMU.
        pitch > 0 = leaning forward → reduce knee extension to slow down.
        """
        correction = np.clip(-PITCH_GAIN * pitch, -0.15, 0.15)
        targets["Servo-Knee-L-Top"]    = np.clip(targets["Servo-Knee-L-Top"]    + correction, -1.3, 1.3)
        targets["Servo-Knee-L-Bottom"] = np.clip(targets["Servo-Knee-L-Bottom"] + correction, -1.3, 1.3)
        targets["Servo-Knee-R-Top"]    = np.clip(targets["Servo-Knee-R-Top"]    + correction, -1.3, 1.3)
        targets["Servo-Knee-R-Bottom"] = np.clip(targets["Servo-Knee-R-Bottom"] + correction, -1.3, 1.3)
        return targets

    def set_freq(self, freq: float):
        self.freq   = max(0.1, freq)
        self._omega = 2 * np.pi * self.freq

    # ── Internal ─────────────────────────────────────────────────────────────

    def _leg(self, phase: float) -> tuple:
        """
        Returns (hip, knee, ankle) target angles for a leg at given phase.

        Knee: drives forward swing (Y axis) — primary locomotion joint.
              Sinusoidal: positive = foot swings forward, negative = pushes back.
        Hip:  small lateral sway for balance (X axis).
        Ankle: push-off timing at end of stance.
        """
        knee  = KNEE_AMP  * np.sin(phase + KNEE_PHASE)
        hip   = HIP_AMP   * np.sin(phase + HIP_PHASE)
        ankle = ANKLE_AMP * np.sin(phase + ANKLE_PHASE)
        return hip, knee, ankle

    def _angles(self) -> dict:
        lh, lk, la = self._leg(self.phase_L)
        rh, rk, ra = self._leg(self.phase_R)

        # Arms counter-swing with opposite knee (natural gait)
        arm_L = ARM_AMP * np.sin(self.phase_R)
        arm_R = ARM_AMP * np.sin(self.phase_L)

        return {
            "Servo-Hip-L":         lh,
            "Servo-Knee-L-Top":    lk,
            "Servo-Knee-L-Bottom": lk,
            "Servo-Ankle-L":       la,
            "Servo-Hip-R":         rh,
            "Servo-Knee-R-Top":    rk,
            "Servo-Knee-R-Bottom": rk,
            "Servo-Ankle-R":       ra,
            "Servo-Hip-Body-Rotation": 0.0,
            "Servo-Showlder-L-Front-Back":     arm_L,
            "Servo-Showlder-R-Front-Back":     arm_R,
            "Servo-Showlder-L-Inward-Outward": 0.0,
            "Servo-Showlder-R-Inward-Outward": 0.0,
            "Servo-Forearm-L": 0.0,
            "Servo-Forearm-R": 0.0,
        }
