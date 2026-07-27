"""
Central Pattern Generator for Optimus — hardware-faithful version.

Hardware constraints extracted from firmware:
  - PCA9685 servo PWM at 50 Hz  → control loop cadence
  - MPU6050 complementary filter at 100 Hz, α=0.98
  - MG995 legs: +90° flex / -45° extension (hardware-measured asymmetric range)
  - Futaba S3003 arms: ±120° shoulder_fb, ±120° shoulder_lat, ±90° forearm
  - Geometry: L1=L2=90 mm thigh/shin, L3=30 mm foot

Gait design — rectified swing with IMU phase reset:
  CPG oscillator drives all joints. IMU roll modulates phase speed so weight
  transfer is confirmed before push-off, not assumed from a fixed timer.

  phase_dot = omega_base * (1 + K_RESET * (roll_actual - roll_expected))
    roll_expected = -ROLL_TARGET * sin(wt)   (negative = stance-L, positive = stance-R)

  ROLL_TARGET auto-calibrates during warmup from observed roll amplitude.

  Joint roles:
    Knee swing/stance: rectified sinusoid — swing leg lifts, stance leg extends (push-off)
    Hip L/R: in stabilise() only — IMU roll drives lateral weight transfer reactively
    Ankle: in stabilise() only — IMU roll tilts foot to recover lateral lean
    Shoulder: ±ARM_AMP counter-swing with opposite leg
    Yaw: hip body rotation damps yaw rate

  RL residual (±MAX_RESIDUAL) corrects everything on top of this.
"""

import numpy as np

# ── Gait parameters ──────────────────────────────────────────────────────────

PERIOD_S      = 1.2
KNEE_AMP      = np.radians(40.0)   # swing amplitude — ~75mm foot clearance
STANCE_ANGLE  = np.radians(15.0)   # stance crouch → extension generates push-off
ARM_AMP       = np.radians(25.0)   # shoulder counter-swing ±25°

# Phase reset: how strongly IMU roll modulates oscillator speed.
# Small value → robust to noise; large → fast adaptation.
# ponytail: tune K_RESET on hardware if gait drifts phase
K_RESET       = 0.4

# ROLL_TARGET: expected roll amplitude when weight is fully on one leg.
# Auto-calibrated during warmup; this is the fallback if calibration fails.
ROLL_TARGET_DEFAULT = np.radians(4.0)

# IMU stabiliser gains
PITCH_GAIN    = 0.4
ANKLE_GAIN    = 0.6    # roll → ankle tilt (foot roll joint, axis Y)
HIP_GAIN      = 0.8    # roll → hip lateral shift (active with ground contact)
YAW_RATE_GAIN = 0.5
PITCH_THRESH  = np.radians(0.6)
ROLL_THRESH   = np.radians(0.6)

# Joint limits from config.h
LIM_HIP_PITCH  = np.radians(45.0)
LIM_HIP_ROLL   = np.radians(45.0)
LIM_KNEE_FLEX  = np.radians(90.0)
LIM_KNEE_EXT   = np.radians(45.0)
LIM_KNEE       = np.radians(70.0)
LIM_ANKLE      = np.radians(90.0)
LIM_SHOULDER   = np.radians(120.0)
LIM_FOREARM    = np.radians(90.0)


class CPG:
    """
    CPG with IMU-modulated phase and auto-calibrated roll target.

    step(dt, roll) — advance oscillator, modulate phase by IMU roll.
    stabilise(targets, pitch, roll, yaw_rate) — reactive IMU corrections.
    calibrate(roll_samples) — set ROLL_TARGET from observed walking roll.
    """

    def __init__(self, period_s: float = PERIOD_S):
        self._omega      = 2 * np.pi / period_s
        self._t          = 0.0
        self._roll_target = ROLL_TARGET_DEFAULT
        self._calibrated  = False

    # ── Public API ───────────────────────────────────────────────────────────

    def step(self, dt: float, roll: float = 0.0) -> dict:
        """
        Advance oscillator by dt seconds, modulated by IMU roll.
        roll: current IMU roll in radians (positive = leaning right).
        """
        wt = self._omega * self._t

        # Expected roll at this phase: negative during stance-L, positive during stance-R.
        roll_expected = -self._roll_target * np.sin(wt)

        # Phase modulation: slow down if weight transfer hasn't happened yet, speed up if it has.
        # Clipped to [0.3, 1.7]× omega so noise can't stall or flip the oscillator.
        phase_error  = float(roll - roll_expected)
        omega_mod    = self._omega * float(np.clip(1.0 + K_RESET * phase_error, 0.3, 1.7))

        self._t += dt * (omega_mod / self._omega)
        return self._angles(self._t)

    def stabilise(self, targets: dict, pitch: float, roll: float,
                  yaw_rate: float = 0.0) -> dict:
        """
        Reactive IMU corrections on top of CPG joint targets.

        pitch → knee bias (forward/back lean recovery)
        roll  → ankle tilt + hip lateral shift (weight transfer assist)
        yaw   → hip body rotation damping
        """
        # Pitch → bias both knees to recover forward/back lean
        if abs(pitch) > PITCH_THRESH:
            p = float(np.clip(-PITCH_GAIN * pitch, -np.radians(8), np.radians(8)))
            for k in ("Servo-Knee-L-Top", "Servo-Knee-L-Bottom",
                      "Servo-Knee-R-Top", "Servo-Knee-R-Bottom"):
                targets[k] = float(np.clip(targets[k] + p, -LIM_KNEE, LIM_KNEE))

        # Roll → ankle tilt (restores foot contact on leaning side)
        # Roll → hip lateral push (shifts CoM toward stance leg, requires ground contact)
        # Both always written so they return to 0 below threshold.
        if abs(roll) > ROLL_THRESH:
            ankle = float(np.clip(-ANKLE_GAIN * roll, -np.radians(30), np.radians(30)))
            # Hip correction: positive roll (lean right) → push both hips right to recover.
            # Same sign both hips = feet move together in -X → CoM shifts +X.
            hip_corr = float(np.clip(HIP_GAIN * roll, -np.radians(20), np.radians(20)))
        else:
            ankle    = 0.0
            hip_corr = 0.0

        targets["Servo-Ankle-L"] = ankle
        targets["Servo-Ankle-R"] = ankle
        targets["Servo-Hip-L"]   = float(np.clip(targets.get("Servo-Hip-L", 0.0) + hip_corr,
                                                  -LIM_HIP_ROLL, LIM_HIP_ROLL))
        targets["Servo-Hip-R"]   = float(np.clip(targets.get("Servo-Hip-R", 0.0) + hip_corr,
                                                  -LIM_HIP_ROLL, LIM_HIP_ROLL))

        # Yaw rate → hip body rotation damping
        targets["Servo-Hip-Body-Rotation"] = float(
            np.clip(-YAW_RATE_GAIN * yaw_rate, -np.radians(20), np.radians(20))
        )
        return targets

    def calibrate(self, roll_samples: list) -> float:
        """
        Set ROLL_TARGET from observed roll amplitude during warmup walking.
        Uses 80th percentile of abs(roll) — robust to noise spikes.
        Only calibrates once; subsequent calls are no-ops.
        """
        if self._calibrated or len(roll_samples) < 10:
            return self._roll_target
        amp = float(np.percentile(np.abs(roll_samples), 80))
        # Clamp to sensible range — too small = phase reset does nothing,
        # too large = oscillator over-reacts to normal sway.
        self._roll_target = float(np.clip(amp, np.radians(1.5), np.radians(12.0)))
        self._calibrated  = True
        return self._roll_target

    def reset(self, t: float = 0.0):
        self._t = t

    def set_period(self, period_s: float):
        self._omega = 2 * np.pi / max(0.2, period_s)

    # ── Internal ─────────────────────────────────────────────────────────────

    def _angles(self, t: float) -> dict:
        wt = self._omega * t

        s            = float(np.sin(wt))
        swing_frac_L = max(0.0, -s)
        swing_frac_R = max(0.0,  s)
        stance_frac_L = 1.0 - swing_frac_L
        stance_frac_R = 1.0 - swing_frac_R

        # Stance leg extends (push-off) during mid-stance; swing leg flexes (clearance).
        stance_ext_L = STANCE_ANGLE * max(0.0,  s)
        stance_ext_R = STANCE_ANGLE * max(0.0, -s)

        knee_L_top = float(np.clip(-KNEE_AMP * swing_frac_L + stance_ext_L * stance_frac_L,
                                   -LIM_KNEE_EXT, LIM_KNEE_FLEX))
        knee_L_bot = float(np.clip(+KNEE_AMP * swing_frac_L - stance_ext_L * stance_frac_L,
                                   -LIM_KNEE_FLEX, LIM_KNEE_EXT))
        knee_R_top = float(np.clip(-KNEE_AMP * swing_frac_R + stance_ext_R * stance_frac_R,
                                   -LIM_KNEE_EXT, LIM_KNEE_FLEX))
        knee_R_bot = float(np.clip(+KNEE_AMP * swing_frac_R - stance_ext_R * stance_frac_R,
                                   -LIM_KNEE_FLEX, LIM_KNEE_EXT))

        # Arms counter-swing opposite leg — symmetric ±ARM_AMP
        arm_L  = float(np.clip( ARM_AMP * (swing_frac_R - swing_frac_L), -LIM_SHOULDER, LIM_SHOULDER))
        arm_R  = float(np.clip( ARM_AMP * (swing_frac_L - swing_frac_R), -LIM_SHOULDER, LIM_SHOULDER))
        fore_L = float(np.clip(0.4 * ARM_AMP * np.sin(wt + np.pi), -LIM_FOREARM, LIM_FOREARM))
        fore_R = float(np.clip(0.4 * ARM_AMP * np.sin(wt),          -LIM_FOREARM, LIM_FOREARM))

        # Hips: base=0, driven entirely by stabilise() IMU roll correction.
        return {
            "Servo-Hip-L":                     0.0,
            "Servo-Knee-L-Top":                knee_L_top,
            "Servo-Knee-L-Bottom":             knee_L_bot,
            "Servo-Ankle-L":                   0.0,
            "Servo-Hip-R":                     0.0,
            "Servo-Knee-R-Top":                knee_R_top,
            "Servo-Knee-R-Bottom":             knee_R_bot,
            "Servo-Ankle-R":                   0.0,
            "Servo-Hip-Body-Rotation":         0.0,
            "Servo-Showlder-L-Front-Back":     arm_L,
            "Servo-Showlder-R-Front-Back":     arm_R,
            "Servo-Showlder-L-Inward-Outward": 0.0,
            "Servo-Showlder-R-Inward-Outward": 0.0,
            "Servo-Forearm-L":                 fore_L,
            "Servo-Forearm-R":                 fore_R,
        }
