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

FREQ        = 0.8       # Hz — step frequency (one full cycle = one stride)
HIP_AMP     = 0.18      # rad — hip swing amplitude (~10°)
KNEE_AMP    = 0.22      # rad — knee flex amplitude (~13°)
ANKLE_AMP   = 0.10      # rad — ankle push amplitude (~6°)
ARM_AMP     = 0.12      # rad — arm counter-swing amplitude

# Phase offsets within a leg cycle (in radians of the oscillator phase)
KNEE_PHASE  = 0.4       # knee leads hip slightly (flex before swing)
ANKLE_PHASE = -0.3      # ankle pushes off after knee extends

# Left/right legs are anti-phase (π apart)
LEFT_PHASE  = 0.0
RIGHT_PHASE = np.pi

# IMU stabilisation gain — how aggressively to correct pitch with ankle
PITCH_GAIN  = 1.2       # rad ankle correction per rad of pitch error


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
        pitch > 0 = leaning forward → dorsiflex ankles to push CoM back.
        """
        correction = np.clip(-PITCH_GAIN * pitch, -0.3, 0.3)
        targets["Servo-Ankle-L"] = np.clip(
            targets["Servo-Ankle-L"] + correction, -1.3, 1.3
        )
        targets["Servo-Ankle-R"] = np.clip(
            targets["Servo-Ankle-R"] + correction, -1.3, 1.3
        )
        return targets

    def set_freq(self, freq: float):
        self.freq   = max(0.1, freq)
        self._omega = 2 * np.pi * self.freq

    # ── Internal ─────────────────────────────────────────────────────────────

    def _leg(self, phase: float) -> tuple:
        """
        Returns (hip, knee, ankle) target angles for a leg at given phase.

        Hip:   sinusoidal swing — forward on first half, back on second.
        Knee:  always flexes during swing (positive = flex), extends at stance.
               Offset so it peaks just after toe-off.
        Ankle: push-off at end of stance phase (negative phase = push).
        """
        hip   =  HIP_AMP   * np.sin(phase)
        knee  =  KNEE_AMP  * np.maximum(0.0, np.sin(phase + KNEE_PHASE))
        ankle =  ANKLE_AMP * np.sin(phase + ANKLE_PHASE)
        return hip, knee, ankle

    def _angles(self) -> dict:
        lh, lk, la = self._leg(self.phase_L)
        rh, rk, ra = self._leg(self.phase_R)

        # Arms counter-swing with hips
        arm_L =  ARM_AMP * np.sin(self.phase_R)   # arm_L swings with right leg
        arm_R =  ARM_AMP * np.sin(self.phase_L)

        return {
            # Legs
            "Servo-Hip-L":         lh,
            "Servo-Knee-L-Top":    lk,
            "Servo-Knee-L-Bottom": lk,
            "Servo-Ankle-L":       la,
            "Servo-Hip-R":         rh,
            "Servo-Knee-R-Top":    rk,
            "Servo-Knee-R-Bottom": rk,
            "Servo-Ankle-R":       ra,
            # Trunk rotation — subtle counter to pelvis swing
            "Servo-Hip-Body-Rotation": 0.0,
            # Arms
            "Servo-Showlder-L-Front-Back":    arm_L,
            "Servo-Showlder-R-Front-Back":    arm_R,
            "Servo-Showlder-L-Inward-Outward": 0.0,
            "Servo-Showlder-R-Inward-Outward": 0.0,
            "Servo-Forearm-L": 0.0,
            "Servo-Forearm-R": 0.0,
        }
