"""
Full DOF kinematic sweep for SpotMicro.

Runs 9 movements back-to-back, each resetting to the neutral pose first,
then playing forward → target → back to neutral before moving to the next.
The full cycle loops indefinitely until the viewer is closed.

Run with:
    mjpython run_kinematics.py
"""

import sys
import os
import time

import numpy as np
import mujoco
from mujoco import viewer

# ── Path setup ────────────────────────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MUJUCO_DIR = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, MUJUCO_DIR)

from spotmicro_loader import load_spotmicro_xml
from spotmicro_kinematics import SpotMicroKinematics

# ── Shared trajectory settings ────────────────────────────────────────────────
N_STEPS = 80     # frames per movement
DT      = 0.04   # seconds per frame  →  ~3.2 s per movement, ~58 s full cycle

NEUTRAL_POS   = np.array([0.0, 0.0, 0.200])
NEUTRAL_EULER = np.zeros(3)

# ── Movement definitions ──────────────────────────────────────────────────────
# Each entry: (label, start_pos, start_euler, target_pos, target_euler)
# All movements start from and return to NEUTRAL so resets are clean.
MOVEMENTS = [
    (
        "1 · Squat / Rise  [Z height — hip + knee pitch]",
        NEUTRAL_POS, NEUTRAL_EULER,
        np.array([0.0,   0.0,  0.120]),
        np.zeros(3),
    ),
    (
        "2 · Forward Shift  [X translation — hip pitch front vs rear]",
        NEUTRAL_POS, NEUTRAL_EULER,
        np.array([0.06,  0.0,  0.200]),
        np.zeros(3),
    ),
    (
        "3 · Lateral Shift  [Y translation — shoulder abduction L vs R]",
        NEUTRAL_POS, NEUTRAL_EULER,
        np.array([0.0,   0.05, 0.200]),
        np.zeros(3),
    ),
    (
        "4 · Roll  [shoulder abduction max range]",
        NEUTRAL_POS, NEUTRAL_EULER,
        NEUTRAL_POS.copy(),
        np.array([0.4,   0.0,  0.0]),
    ),
    (
        "5 · Pitch  [hip pitch front vs rear asymmetry]",
        NEUTRAL_POS, NEUTRAL_EULER,
        NEUTRAL_POS.copy(),
        np.array([0.0,   0.3,  0.0]),
    ),
    (
        "6 · Yaw  [all shoulders + hip rotation around vertical]",
        NEUTRAL_POS, NEUTRAL_EULER,
        NEUTRAL_POS.copy(),
        np.array([0.0,   0.0,  0.5]),
    ),
    (
        "7 · Roll + Squat  [shoulder abduction + hip/knee range]",
        NEUTRAL_POS, NEUTRAL_EULER,
        np.array([0.0,   0.0,  0.150]),
        np.array([0.35,  0.0,  0.0]),
    ),
    (
        "8 · Pitch + Forward Lean  [front leg compression + rear extension]",
        NEUTRAL_POS, NEUTRAL_EULER,
        np.array([0.05,  0.0,  0.190]),
        np.array([0.0,   0.25, 0.0]),
    ),
    (
        "9 · Full 6-DOF  [all joints simultaneously]",
        NEUTRAL_POS, NEUTRAL_EULER,
        np.array([0.04,  0.03, 0.160]),
        np.array([0.25,  0.15, 0.3]),
    ),
]

# ── Pre-compute qpos trajectories ─────────────────────────────────────────────
def build_qpos_traj(kin, start_pos, start_euler, end_pos, end_euler, n_steps):
    joint_traj, _, reachable = kin.interpolate_poses(
        start_pos, start_euler, end_pos, end_euler,
        n_steps=n_steps, keep_feet_fixed=True,
    )
    bad = int((~reachable.all(axis=1)).sum())
    if bad:
        print(f"  [warn] {bad}/{n_steps} frames clamped to joint limits")

    q0 = kin._euler_to_quat_wxyz(start_euler)
    q1 = kin._euler_to_quat_wxyz(end_euler)
    qpos = []
    for i in range(n_steps):
        t = i / max(n_steps - 1, 1)
        bp = (1.0 - t) * start_pos + t * end_pos
        be = kin._quat_wxyz_to_euler(kin._slerp(q0, q1, t))
        qpos.append(kin.build_qpos(bp, be, joint_traj[i]))
    return np.array(qpos)


print("Building trajectories...")
kin = SpotMicroKinematics()

neutral_qpos = kin.build_qpos(
    NEUTRAL_POS, NEUTRAL_EULER,
    kin.inverse_kinematics(NEUTRAL_POS, NEUTRAL_EULER,
                           kin.compute_stance_feet(NEUTRAL_POS, NEUTRAL_EULER))[0],
)

trajectories = []
for label, sp, se, tp, te in MOVEMENTS:
    print(f"  {label}")
    traj = build_qpos_traj(kin, sp, se, tp, te, N_STEPS)
    trajectories.append((label, traj))

total_s = len(MOVEMENTS) * N_STEPS * DT * 2   # ×2 for forward+return
print(f"\nReady — {len(MOVEMENTS)} movements, ~{total_s:.0f} s per full cycle")
print("Tip: press 2 in the viewer to show mesh visuals.")

# ── Load MuJoCo model ─────────────────────────────────────────────────────────
xml_str = load_spotmicro_xml()
model   = mujoco.MjModel.from_xml_string(xml_str)
data    = mujoco.MjData(model)


def play_traj(traj, v):
    """Play one trajectory forward then in reverse. Returns False if viewer closed."""
    for qpos_t in traj:
        if not v.is_running():
            return False
        data.qpos[:] = qpos_t
        data.qvel[:] = 0.0
        mujoco.mj_forward(model, data)
        v.sync()
        time.sleep(DT)
    for qpos_t in reversed(traj):
        if not v.is_running():
            return False
        data.qpos[:] = qpos_t
        data.qvel[:] = 0.0
        mujoco.mj_forward(model, data)
        v.sync()
        time.sleep(DT)
    return True


def reset_to_neutral(v):
    """Snap back to neutral stance instantly."""
    data.qpos[:] = neutral_qpos
    data.qvel[:] = 0.0
    mujoco.mj_forward(model, data)
    v.sync()
    time.sleep(0.3)   # brief pause so the reset is visible


# ── Main loop ─────────────────────────────────────────────────────────────────
with viewer.launch_passive(model, data) as v:
    v.opt.geomgroup[1] = 0   # hide collision boxes

    cycle = 0
    while v.is_running():
        cycle += 1
        print(f"\n── Cycle {cycle} ──────────────────────────────────────────")

        for label, traj in trajectories:
            if not v.is_running():
                break
            print(f"  {label}")
            reset_to_neutral(v)
            if not play_traj(traj, v):
                break

        # Hold neutral briefly at end of full cycle before repeating
        if v.is_running():
            reset_to_neutral(v)
            time.sleep(0.5)
