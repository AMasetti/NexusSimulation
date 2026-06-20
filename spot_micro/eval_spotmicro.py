#!/usr/bin/env python3
"""
Evaluate a saved SpotMicro policy and report benchmark metrics.

Usage:
  # Pure RL model
  mjpython eval_spotmicro.py --model ./logs/spotmicro_ppo --episodes 20
  mjpython eval_spotmicro.py --model ./logs/spotmicro_ppo --episodes 5 --render --speed 0.5

  # Trajectory-tracking model
  mjpython eval_spotmicro.py --model ./logs/spotmicro_traj_ppo --traj --episodes 10 --render --speed 0.5
  mjpython eval_spotmicro.py --model logs/best_model --traj --render --speed 0.3
"""

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
from stable_baselines3 import PPO, SAC

from spotmicro_env import SpotMicroEnv
from spotmicro_traj_env import SpotMicroTrajEnv, TROT_KEYFRAMES


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Benchmark a saved SpotMicro policy")
    p.add_argument("--model", type=str, default="./logs/spotmicro_ppo")
    p.add_argument("--algo", type=str, default="ppo", choices=["ppo", "sac"])
    p.add_argument("--traj", action="store_true",
                   help="Use trajectory-tracking env (SpotMicroTrajEnv) instead of pure RL env")
    p.add_argument("--steps_per_frame", type=int, default=20,
                   help="[traj only] steps per keyframe, must match training value")
    p.add_argument("--episodes", type=int, default=20)
    p.add_argument("--max_episode_steps", type=int, default=2_000)
    p.add_argument("--render", action="store_true")
    p.add_argument("--speed", type=float, default=1.0,
                   help="Playback speed multiplier (1.0=real-time, 0.5=half speed, 0.1=very slow)")
    p.add_argument("--deterministic", action="store_true", default=True,
                   help="Use deterministic actions (no exploration noise)")
    return p.parse_args()


def run_episode(env, model, deterministic: bool, step_delay: float = 0.0) -> dict:
    obs, _ = env.reset()
    start_x = float(env.data.qpos[0])

    total_reward = 0.0
    steps = 0
    max_height = float(env.data.qpos[2])
    max_x_vel = 0.0
    fell = False
    cycles_completed = 0
    is_traj = isinstance(env, SpotMicroTrajEnv)
    cycle_len = (env.n_frames * env.steps_per_frame) if is_traj else None

    while True:
        action, _ = model.predict(obs, deterministic=deterministic)
        obs, reward, terminated, truncated, info = env.step(action)

        total_reward += reward
        steps += 1
        max_height = max(max_height, float(env.data.qpos[2]))
        max_x_vel = max(max_x_vel, float(env.data.qvel[0]))
        if is_traj and cycle_len and steps % cycle_len == 0:
            cycles_completed += 1

        if terminated:
            fell = True
        if step_delay > 0:
            time.sleep(step_delay)

        if terminated or truncated:
            break

    end_x = float(env.data.qpos[0])
    sim_time = steps * env.frame_skip * env.model.opt.timestep

    return {
        "total_reward": total_reward,
        "steps": steps,
        "sim_time_s": sim_time,
        "distance_m": end_x - start_x,
        "max_height_m": max_height,
        "max_x_vel_ms": max_x_vel,
        "fell": fell,
        "cycles_completed": cycles_completed,
    }


def main() -> None:
    args = parse_args()

    render_mode = "human" if args.render else None
    if args.traj:
        env = SpotMicroTrajEnv(
            keyframes=TROT_KEYFRAMES,
            steps_per_frame=args.steps_per_frame,
            max_episode_steps=args.max_episode_steps,
            render_mode=render_mode,
        )
        env_label = f"trajectory ({len(TROT_KEYFRAMES)} keyframes, {args.steps_per_frame} steps/frame)"
    else:
        env = SpotMicroEnv(max_episode_steps=args.max_episode_steps, render_mode=render_mode)
        env_label = "pure RL"

    cls = PPO if args.algo == "ppo" else SAC
    model = cls.load(args.model, env=env)
    print(f"Loaded {args.algo.upper()} model from {args.model}  [{env_label}]\n")

    # real-time step = frame_skip * timestep; divide by speed for slower playback
    step_delay = 0.0
    if args.render and args.speed > 0:
        step_delay = (env.frame_skip * env.model.opt.timestep) / args.speed

    results = []
    for ep in range(args.episodes):
        r = run_episode(env, model, deterministic=args.deterministic, step_delay=step_delay)
        results.append(r)
        status = "FELL " if r["fell"] else "TRUNC"
        extra = f"  cycles={r['cycles_completed']}" if args.traj else ""
        print(
            f"  ep {ep+1:3d} [{status}]  "
            f"steps={r['steps']:5d}  "
            f"time={r['sim_time_s']:5.2f}s  "
            f"dist={r['distance_m']:+6.3f}m  "
            f"max_h={r['max_height_m']:.3f}m  "
            f"max_vx={r['max_x_vel_ms']:.3f}m/s  "
            f"reward={r['total_reward']:8.1f}"
            f"{extra}"
        )

    env.close()

    # Aggregate stats
    print("\n" + "=" * 60)
    print(f"{'BENCHMARK SUMMARY':^60}")
    print("=" * 60)
    metrics = [
        ("Survival rate",        f"{sum(not r['fell'] for r in results)/len(results)*100:.0f}%"),
        ("Mean episode length",  f"{np.mean([r['steps'] for r in results]):.1f} steps"),
        ("Mean sim time",        f"{np.mean([r['sim_time_s'] for r in results]):.2f}s"),
        ("Mean forward dist",    f"{np.mean([r['distance_m'] for r in results]):+.3f}m"),
        ("Max forward dist",     f"{max(r['distance_m'] for r in results):+.3f}m"),
        ("Mean max height",      f"{np.mean([r['max_height_m'] for r in results]):.3f}m"),
        ("Mean max x velocity",  f"{np.mean([r['max_x_vel_ms'] for r in results]):.3f}m/s"),
        ("Max x velocity",       f"{max(r['max_x_vel_ms'] for r in results):.3f}m/s"),
        ("Mean total reward",    f"{np.mean([r['total_reward'] for r in results]):.1f}"),
    ]
    if args.traj:
        metrics.append(("Mean gait cycles",  f"{np.mean([r['cycles_completed'] for r in results]):.1f}"))
    for label, value in metrics:
        print(f"  {label:<24} {value}")
    print("=" * 60)

    # Qualitative assessment
    mean_dist  = np.mean([r["distance_m"] for r in results])
    mean_steps = np.mean([r["steps"] for r in results])
    print("\nAssessment:")
    if mean_dist < 0.05 and mean_steps < 50:
        print("  ❌ Robot falls immediately — likely learned to fall forward for velocity reward.")
        print("     Try: longer training (5M+ steps), lower reward_vel, or add survival bonus.")
    elif mean_dist < 0.3:
        print("  ⚠  Robot takes a few steps but doesn't sustain locomotion.")
        print("     Continue training or tune reward weights.")
    elif mean_dist < 1.0:
        print("  ✅ Robot shows early walking behaviour. Continue training for more stability.")
    else:
        print("  ✅✅ Robot walks well! Consider increasing terrain complexity.")


if __name__ == "__main__":
    main()
