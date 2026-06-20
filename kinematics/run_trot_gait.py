"""
Trot gait with cycloid swing trajectories — kinematic simulation.

Gait pattern (trot):
  FL + RR  swing together  (phase = 0.0)
  FR + RL  swing together  (phase = 0.5)

Each foot traces a true cycloid arc during the swing phase:
  - slow lift-off  →  fast travel  →  slow touch-down
  - natural velocity profile matching real quadruped gaits

Body remains at constant height — pure kinematic replay (mj_forward, no physics).

Run:
    mjpython run_trot_gait.py
"""

import sys
import os
import time
import math

import numpy as np
import mujoco
from mujoco import viewer

# ── Path setup ────────────────────────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MUJUCO_DIR = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, MUJUCO_DIR)
sys.path.insert(0, SCRIPT_DIR)

from spotmicro_loader import load_spotmicro_xml
from spotmicro_kinematics import SpotMicroKinematics, LEG_NAMES

# ── Gait parameters  (edit these) ────────────────────────────────────────────
BODY_HEIGHT  = 0.200    # m       standing height
STEP_HEIGHT  = 0.035    # m       max foot lift during swing
STEP_LENGTH  = 0.040    # m       fore-aft extent of each swing arc
DUTY_CYCLE   = 0.50     # [0-1]   fraction of cycle the foot spends in stance
GAIT_PERIOD  = 1.6      # s       duration of one full trot cycle
DT           = 0.02     # s       simulation / render step  (50 Hz)

# Trot diagonal pairs: FL+RR in phase, FR+RL offset by half a cycle
PHASE_OFFSETS = {
    "front_left":  0.0,
    "front_right": 0.5,
    "rear_left":   0.5,
    "rear_right":  0.0,
}

# ── Cycloid trajectory ────────────────────────────────────────────────────────

def _cycloid_swing(swing_t: float):
    """
    Compute foot offset (dx_forward, dz_up) for one swing phase instant.

    swing_t : normalised swing progress  0.0 (lift-off) → 1.0 (touch-down)

    Parametric cycloid  φ = swing_t * 2π :
        x(φ) = R·(φ − sin φ)          — horizontal travel
        z(φ) = R·(1 − cos φ)          — vertical lift

    Horizontal component is centred so the foot starts STEP_LENGTH/2 behind
    its home position and lands STEP_LENGTH/2 ahead, making the motion
    symmetric around the stance contact point.
    """
    phi = swing_t * 2.0 * math.pi

    # Forward displacement: centred cycloid, spans full STEP_LENGTH
    dx = (STEP_LENGTH / (2.0 * math.pi)) * (phi - math.sin(phi)) - STEP_LENGTH / 2.0

    # Vertical displacement: cycloid lift, peaks at STEP_HEIGHT when φ = π
    dz = (STEP_HEIGHT / 2.0) * (1.0 - math.cos(phi))

    return dx, dz


def foot_target(home: np.ndarray, phase: float) -> np.ndarray:
    """
    World-frame toe target for one leg at the given cycle phase.

    home  : (3,) nominal ground contact position of this leg
    phase : normalised cycle phase [0, 1)
    """
    phase = phase % 1.0

    if phase < DUTY_CYCLE:
        # ── Stance ─────────────────────────────────────────────────
        # Foot planted at home, z = 0 (ground contact)
        return np.array([home[0], home[1], 0.0])
    else:
        # ── Swing  ─────────────────────────────────────────────────
        swing_t = (phase - DUTY_CYCLE) / (1.0 - DUTY_CYCLE)
        dx, dz  = _cycloid_swing(swing_t)
        return np.array([home[0] + dx, home[1], dz])


# ── Robot setup ───────────────────────────────────────────────────────────────
kin        = SpotMicroKinematics()
BODY_POS   = np.array([0.0, 0.0, BODY_HEIGHT])
BODY_EULER = np.zeros(3)

# Nominal stance foot positions — computed once, held as gait home positions
home_feet = kin.compute_stance_feet(BODY_POS, BODY_EULER)   # (4, 3)

# ── MuJoCo model ──────────────────────────────────────────────────────────────
xml_str = load_spotmicro_xml()
model   = mujoco.MjModel.from_xml_string(xml_str)
data    = mujoco.MjData(model)

# Initialise to standing pose
init_angles, _ = kin.inverse_kinematics(BODY_POS, BODY_EULER, home_feet)
data.qpos[:]   = kin.build_qpos(BODY_POS, BODY_EULER, init_angles)
data.qvel[:]   = 0.0
mujoco.mj_forward(model, data)

# ── Info ──────────────────────────────────────────────────────────────────────
print("═" * 56)
print("  SpotMicro — Trot gait (cycloid swing, kinematic)")
print("═" * 56)
print(f"  Body height  : {BODY_HEIGHT*100:.0f} cm")
print(f"  Step height  : {STEP_HEIGHT*100:.1f} cm")
print(f"  Step length  : {STEP_LENGTH*100:.1f} cm")
print(f"  Gait period  : {GAIT_PERIOD:.2f} s")
print(f"  Duty cycle   : {DUTY_CYCLE*100:.0f}%  stance / {(1-DUTY_CYCLE)*100:.0f}%  swing")
print("─" * 56)
print("  FL + RR  swing phase  →  0.0")
print("  FR + RL  swing phase  →  0.5")
print("─" * 56)
print("  Press 2 in viewer for mesh visuals.")
print("  Close viewer window to stop.")
print("═" * 56)

# ── Gait loop ─────────────────────────────────────────────────────────────────
with viewer.launch_passive(model, data) as v:
    v.opt.geomgroup[1] = 0   # hide collision boxes, show mesh visuals

    sim_time = 0.0

    while v.is_running():
        # -- Compute target toe positions for all 4 legs --
        targets = np.zeros((4, 3))
        for i, leg in enumerate(LEG_NAMES):
            phase       = (sim_time / GAIT_PERIOD + PHASE_OFFSETS[leg]) % 1.0
            targets[i]  = foot_target(home_feet[i], phase)

        # -- Solve IK --
        angles, reachable = kin.inverse_kinematics(BODY_POS, BODY_EULER, targets)
        # (joint-limit clamping is silently accepted in kinematic mode)

        # -- Apply to simulation --
        data.qpos[:] = kin.build_qpos(BODY_POS, BODY_EULER, angles)
        data.qvel[:] = 0.0
        mujoco.mj_forward(model, data)
        v.sync()

        sim_time += DT
        time.sleep(DT)
