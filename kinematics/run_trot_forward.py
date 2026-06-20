"""
Trot gait — kinematic simulation with forward locomotion.

Diagonal trot (FL+RR swing together, FR+RL swing together).
Each foot follows a pure cycloid arc during swing.  The body
advances at a constant forward velocity — no physics, no PD
oscillation.  Uses mj_forward (geometry only) for perfectly
smooth replay.

Run:
    mjpython run_trot_forward.py
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
from spotmicro_kinematics import (
    SpotMicroKinematics, LEG_NAMES,
    SHOULDER_ORIGINS, HIP_OFFSET_Y,
)

# ══════════════════════════════════════════════════════════════════════════════
# Parameters  — edit here
# ══════════════════════════════════════════════════════════════════════════════

BODY_HEIGHT  = 0.200   # m    standing body height
STEP_HEIGHT  = 0.045   # m    max foot lift during swing
STEP_FORWARD = 0.080   # m    how far ahead of hip to plant the foot
DUTY_CYCLE   = 0.60    # []   fraction of cycle in stance
GAIT_PERIOD  = 1.8     # s    one full trot cycle
N_CYCLES     = 5       # []   number of cycles to simulate

DT = 0.02              # s    render step (50 Hz)

# Forward body speed derived from step length and cycle timing so the
# body advances exactly one step-forward per half-cycle.
# Each diagonal pair swings once per full cycle → net advance = STEP_FORWARD/cycle
BODY_VEL = STEP_FORWARD / GAIT_PERIOD   # m/s

# ── Trot phase offsets ────────────────────────────────────────────────────────
PHASE_OFFSETS = {
    "front_left":  0.0,
    "front_right": 0.5,
    "rear_left":   0.5,
    "rear_right":  0.0,
}


# ══════════════════════════════════════════════════════════════════════════════
# Cycloid swing trajectory
# ══════════════════════════════════════════════════════════════════════════════

def cycloid_swing(p_lift: np.ndarray, p_land: np.ndarray,
                  swing_t: float) -> np.ndarray:
    """
    Smooth arc from lift-off to landing using a cycloid velocity profile.
    swing_t : 0.0 (lift-off) → 1.0 (touch-down)
    """
    phi    = swing_t * 2.0 * math.pi
    x_norm = (phi - math.sin(phi)) / (2.0 * math.pi)
    dz     = (STEP_HEIGHT / 2.0) * (1.0 - math.cos(phi))
    pos    = (1.0 - x_norm) * p_lift + x_norm * p_land
    pos    = pos.copy()
    pos[2] = pos[2] + dz
    return pos


# ══════════════════════════════════════════════════════════════════════════════
# Setup
# ══════════════════════════════════════════════════════════════════════════════

kin = SpotMicroKinematics()

body_pos   = np.array([0.0, 0.0, BODY_HEIGHT])
body_euler = np.zeros(3)

xml_str = load_spotmicro_xml()
model   = mujoco.MjModel.from_xml_string(xml_str)
data    = mujoco.MjData(model)

# Compute initial stance feet and set starting qpos
home_feet   = kin.compute_stance_feet(body_pos, body_euler)
init_angles, _ = kin.inverse_kinematics(body_pos, body_euler, home_feet)
data.qpos[:]   = kin.build_qpos(body_pos, body_euler, init_angles)
data.qvel[:]   = 0.0
mujoco.mj_forward(model, data)

# ── Gait state ────────────────────────────────────────────────────────────────

world_contact = home_feet.copy()   # planted foot positions (world frame)
lift_pos      = home_feet.copy()   # foot position at swing start
land_target   = home_feet.copy()   # planned touchdown position

in_swing = np.array([
    (PHASE_OFFSETS[leg] % 1.0) >= DUTY_CYCLE
    for leg in LEG_NAMES
])

total_time = N_CYCLES * GAIT_PERIOD
sim_time   = 0.0

# ══════════════════════════════════════════════════════════════════════════════
# Info
# ══════════════════════════════════════════════════════════════════════════════

print("═" * 56)
print("  SpotMicro — Trot  (kinematic + forward locomotion)")
print("═" * 56)
print(f"  Body height   : {BODY_HEIGHT*100:.0f} cm")
print(f"  Step height   : {STEP_HEIGHT*100:.1f} cm")
print(f"  Step forward  : {STEP_FORWARD*100:.1f} cm")
print(f"  Body velocity : {BODY_VEL*100:.1f} cm/s")
print(f"  Duty cycle    : {DUTY_CYCLE*100:.0f} % stance")
print(f"  Gait period   : {GAIT_PERIOD:.2f} s")
print(f"  Cycles        : {N_CYCLES}  ({total_time:.1f} s total)")
print("─" * 56)
print("  FL + RR swing first,  FR + RL swing second")
print("  Press 2 for mesh visuals.  Close to stop.")
print("═" * 56)

# ══════════════════════════════════════════════════════════════════════════════
# Main loop
# ══════════════════════════════════════════════════════════════════════════════

def hip_world(leg_idx: int) -> np.ndarray:
    leg      = LEG_NAMES[leg_idx]
    hip_body = SHOULDER_ORIGINS[leg] + np.array([0.0, HIP_OFFSET_Y[leg], 0.0])
    T        = kin._build_body_transform(body_pos, body_euler)
    return (T @ np.append(hip_body, 1.0))[:3]


with viewer.launch_passive(model, data) as v:
    v.opt.geomgroup[1] = 0   # hide collision boxes

    while v.is_running() and sim_time < total_time:

        foot_targets = np.zeros((4, 3))

        for i, leg in enumerate(LEG_NAMES):
            phase     = (sim_time / GAIT_PERIOD + PHASE_OFFSETS[leg]) % 1.0
            now_swing = phase >= DUTY_CYCLE

            # ── Stance → swing transition ─────────────────────────────
            if now_swing and not in_swing[i]:
                lift_pos[i] = world_contact[i].copy()
                hw = hip_world(i)
                land_target[i] = np.array([hw[0] + STEP_FORWARD,
                                           hw[1],
                                           0.0])

            # ── Swing → stance transition ─────────────────────────────
            if not now_swing and in_swing[i]:
                world_contact[i] = land_target[i].copy()

            in_swing[i] = now_swing

            # ── Foot target ───────────────────────────────────────────
            if not now_swing:
                foot_targets[i] = world_contact[i]
            else:
                swing_t = (phase - DUTY_CYCLE) / (1.0 - DUTY_CYCLE)
                foot_targets[i] = cycloid_swing(lift_pos[i], land_target[i], swing_t)

        # ── IK → set qpos directly (no physics) ──────────────────────
        angles, _ = kin.inverse_kinematics(body_pos, body_euler, foot_targets)
        data.qpos[:] = kin.build_qpos(body_pos, body_euler, angles)
        data.qvel[:] = 0.0
        mujoco.mj_forward(model, data)
        v.sync()

        # ── Advance body forward ──────────────────────────────────────
        body_pos[0] += BODY_VEL * DT
        sim_time    += DT
        time.sleep(DT)

    print(f"\nCompleted {N_CYCLES} cycles. "
          f"Body travelled {body_pos[0]:.3f} m forward.")
    print("Holding final pose — close viewer to exit.")

    # ── Hold final pose ───────────────────────────────────────────────
    final_feet = kin.compute_stance_feet(body_pos, body_euler)
    final_angles, _ = kin.inverse_kinematics(body_pos, body_euler, final_feet)
    while v.is_running():
        data.qpos[:] = kin.build_qpos(body_pos, body_euler, final_angles)
        data.qvel[:] = 0.0
        mujoco.mj_forward(model, data)
        v.sync()
        time.sleep(DT)
