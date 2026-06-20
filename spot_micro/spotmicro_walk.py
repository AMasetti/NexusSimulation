#!/usr/bin/env python3
"""
Run the trained SpotMicro policy with the MuJoCo viewer.
On macOS: run with mjpython (e.g. mjpython spotmicro_walk.py).
Uses the same env (timestep 0.02) as training so episodes last many steps.
Playback: PLAYBACK_SPEED = 1 is real time; higher = faster (e.g. 25 = 25x, no slow-mo).
"""
import time
import gymnasium as gym
from stable_baselines3 import PPO

from spotmicro_env import SpotMicroEnv

# Must match the config used when training (and in the notebook)
SAVE_PATH = "./logs/spotmicro_ppo"
LOAD_PATH = SAVE_PATH
MAX_EPISODE_STEPS = 600_000
SEED = 0
# 1 = real time; >1 = faster playback. Use 1.0 to see actual motion speed.
PLAYBACK_SPEED = 1.0

model = PPO.load(LOAD_PATH)

env = SpotMicroEnv(
    max_episode_steps=MAX_EPISODE_STEPS,
    render_mode="human",
)
env = gym.wrappers.TimeLimit(env, max_episode_steps=MAX_EPISODE_STEPS)

# Sim time per env step (timestep * frame_skip) so we can throttle to real time
base_env = env.env if hasattr(env, "env") else env
step_sim_dt = base_env.model.opt.timestep * base_env.frame_skip  # e.g. 0.02 * 15 = 0.3 s

obs, info = env.reset(seed=SEED)
total_reward = 0.0
steps = 0
while True:
    t0 = time.perf_counter()
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, info = env.step(action)
    total_reward += reward
    steps += 1
    # Throttle: 1s sim = (1/PLAYBACK_SPEED) s wall; skip sleep if we're already behind
    elapsed = time.perf_counter() - t0
    target_wall = step_sim_dt / PLAYBACK_SPEED
    time.sleep(max(0.0, target_wall - elapsed))
    if terminated or truncated:
        break

env.close()
sim_time_s = steps * step_sim_dt
print(f"Episode finished: steps={steps}, sim_time={sim_time_s:.1f}s, total_reward={total_reward:.2f}")