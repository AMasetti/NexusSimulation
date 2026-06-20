#!/usr/bin/env python3
"""
Diagnostic script to validate assumptions about SpotMicro stiff-leg behavior.
Run this BEFORE training to understand what's happening.

Validates:
  1. Whether qpos=0 is a stable standing pose (height check)
  2. Ctrl ranges for each actuator (are knees/ankles ranged near zero?)
  3. How quickly the robot falls with zero action
  4. Whether the reward signal for leg movement is meaningful vs fall penalty
"""

import numpy as np
import mujoco
from spotmicro_loader import load_spotmicro_xml
from spotmicro_env import SpotMicroEnv, DEFAULT_BASE_HEIGHT, NUM_JOINTS, JOINT_NAMES

# ─────────────────────────────────────────────
# 1. CTRL RANGE DIAGNOSTIC
# ─────────────────────────────────────────────
print("\n" + "="*60)
print("DIAGNOSTIC 1: Actuator ctrl ranges")
print("="*60)
xml = load_spotmicro_xml()
model = mujoco.MjModel.from_xml_string(xml)
model.opt.timestep = 0.02

print(f"{'Actuator':<30} {'Low':>8} {'High':>8} {'Range':>8}")
print("-" * 58)
for i in range(model.nu):
    name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i) or f"actuator_{i}"
    lo, hi = model.actuator_ctrlrange[i]
    print(f"{name:<30} {lo:>8.3f} {hi:>8.3f} {hi-lo:>8.3f}")

# ─────────────────────────────────────────────
# 2. DEFAULT POSE STABILITY DIAGNOSTIC
# ─────────────────────────────────────────────
print("\n" + "="*60)
print("DIAGNOSTIC 2: Stability at qpos=0 (all joints zero)")
print("="*60)
data = mujoco.MjData(model)
data.qpos[:] = 0.0
data.qpos[2] = DEFAULT_BASE_HEIGHT
data.qpos[3:7] = [1.0, 0.0, 0.0, 0.0]
mujoco.mj_forward(model, data)

print(f"Initial height:      {data.qpos[2]:.4f} m  (target: {DEFAULT_BASE_HEIGHT})")
print(f"Termination thresh:  0.12 m")

# Step with zero action and record height over time
heights = []
for step in range(200):
    data.ctrl[:] = 0.0
    for _ in range(2):  # frame_skip=2
        mujoco.mj_step(model, data)
    heights.append(data.qpos[2])

heights = np.array(heights)
first_below = np.argmax(heights < 0.12)
if first_below == 0 and heights[0] >= 0.12:
    first_below = -1  # never fell

print(f"\nHeight at step 1:    {heights[0]:.4f} m")
print(f"Height at step 10:   {heights[min(9, len(heights)-1)]:.4f} m")
print(f"Height at step 50:   {heights[min(49, len(heights)-1)]:.4f} m")
print(f"Height at step 100:  {heights[min(99, len(heights)-1)]:.4f} m")
print(f"Height at step 200:  {heights[-1]:.4f} m")
if first_below > 0:
    print(f"\n⚠️  FALLS BELOW 0.12m at step {first_below}  ← EPISODE TERMINATES EARLY!")
else:
    print(f"\n✅ Never fell below 0.12m in 200 steps with zero action")

# ─────────────────────────────────────────────
# 3. REWARD SCALE DIAGNOSTIC
# ─────────────────────────────────────────────
print("\n" + "="*60)
print("DIAGNOSTIC 3: Reward component scales")
print("="*60)
env = SpotMicroEnv()

# Simulate what a typical upright step reward looks like
reward_survival = env.reward_survival           # 0.02 per step
reward_fall = env.reward_fall                   # -0.15 per fall
reward_leg = env.reward_leg_action              # 0.02 * mean_abs_leg_action
reward_vel = env.reward_vel                     # 1.0 * forward_displacement

# Estimate forward displacement per step at 0.1 m/s
vel_estimate = 0.1  # m/s
step_dt = model.opt.timestep * 2               # frame_skip=2
forward_per_step = vel_estimate * step_dt
forward_reward_est = env.reward_vel * forward_per_step

print(f"Survival reward (per step):      {reward_survival:+.4f}")
print(f"Fall penalty (per fall):         {reward_fall:+.4f}")
print(f"Forward reward @ 0.1m/s:        {forward_reward_est:+.4f}")
print(f"Leg action reward (max):         {env.reward_leg_action:+.4f}  (if mean|action|=1.0)")
print(f"Joint vel reward:                {env.reward_joint_vel:+.4f}  (if mean_sq_vel=1.0)")
print(f"\nFall penalty / survival ratio:   {abs(reward_fall)/reward_survival:.1f}x")
print(f"Need to survive N steps to offset one fall: {abs(reward_fall)/reward_survival:.0f} steps")

if abs(reward_fall) / reward_survival > 5:
    print("\n⚠️  Fall penalty is >> survival bonus → agent learns to FREEZE to avoid falls")
else:
    print("\n✅ Reward balance looks reasonable")

# ─────────────────────────────────────────────
# 4. JOINT ANGLE DIAGNOSTIC — are zeros a bad pose?
# ─────────────────────────────────────────────
print("\n" + "="*60)
print("DIAGNOSTIC 4: Joint range analysis (are zeros at limits?)")
print("="*60)
print(f"{'Joint':<30} {'Low':>8} {'High':>8} {'Zero pos':>10}")
print("-" * 60)
for i, name in enumerate(JOINT_NAMES):
    jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
    lo = model.jnt_range[jid, 0]
    hi = model.jnt_range[jid, 1]
    at_limit = " ← AT LIMIT!" if (abs(lo) < 0.01 or abs(hi) < 0.01) else ""
    print(f"{name:<30} {lo:>8.3f} {hi:>8.3f}  zero{at_limit}")

env.close()
print("\n" + "="*60)
print("DIAGNOSTIC COMPLETE — review ⚠️  warnings above")
print("="*60)