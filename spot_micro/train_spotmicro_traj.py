#!/usr/bin/env python3
"""
Train SpotMicro to walk by tracking a reference trot gait (trajectory imitation).

The policy is rewarded for:
  1. Matching the current keyframe target pose  (tracking reward, DeepMimic-style)
  2. Moving forward                             (velocity reward)
  3. Staying upright                            (survival + tilt penalties)

Usage:
  mjpython train_spotmicro_traj.py
  mjpython train_spotmicro_traj.py --algo sac --total_timesteps 3000000
  mjpython train_spotmicro_traj.py --render      # show viewer while training
"""

import argparse
import os

import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO, SAC
from stable_baselines3.common.callbacks import EvalCallback

from spotmicro_traj_env import SpotMicroTrajEnv, WALK_5_KEYFRAMES, TROT_KEYFRAMES


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train SpotMicro trajectory tracking")
    p.add_argument("--algo",             type=str,   default="ppo", choices=["ppo", "sac"])
    p.add_argument("--total_timesteps",  type=float, default=3e6)
    p.add_argument("--save_path",        type=str,   default="./logs/spotmicro_traj_ppo")
    p.add_argument("--trot",             action="store_true",
                   help="Use 8-frame TROT_KEYFRAMES instead of the default 5-pose walk")
    p.add_argument("--seed",             type=int,   default=0)
    p.add_argument("--max_episode_steps",type=int,   default=4_000)
    p.add_argument("--steps_per_frame",  type=int,   default=20,
                   help="Env steps to hold each keyframe (lower = faster gait cycle)")
    p.add_argument("--render", action="store_true",  help="Show MuJoCo viewer (slower)")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    total_timesteps = int(args.total_timesteps)

    keyframes = TROT_KEYFRAMES if args.trot else WALK_5_KEYFRAMES
    render_mode = "human" if args.render else None
    env = SpotMicroTrajEnv(
        keyframes=keyframes,
        steps_per_frame=args.steps_per_frame,
        max_episode_steps=args.max_episode_steps,
        render_mode=render_mode,
    )
    env = gym.wrappers.TimeLimit(env, max_episode_steps=args.max_episode_steps)
    env = gym.wrappers.RecordEpisodeStatistics(env)

    # Separate eval env (no render) for EvalCallback
    eval_env = SpotMicroTrajEnv(
        keyframes=keyframes,
        steps_per_frame=args.steps_per_frame,
        max_episode_steps=args.max_episode_steps,
    )
    eval_env = gym.wrappers.TimeLimit(eval_env, max_episode_steps=args.max_episode_steps)

    os.makedirs(os.path.dirname(args.save_path) or ".", exist_ok=True)
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=os.path.dirname(args.save_path) or ".",
        log_path=os.path.dirname(args.save_path) or ".",
        eval_freq=50_000,
        n_eval_episodes=5,
        deterministic=True,
        verbose=1,
    )

    if args.algo == "ppo":
        model = PPO(
            "MlpPolicy",
            env,
            learning_rate=1e-4,
            n_steps=4096,
            batch_size=128,
            n_epochs=10,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.1,
            ent_coef=0.01,
            policy_kwargs=dict(
                net_arch=dict(pi=[256, 256], vf=[256, 256]),
                log_std_init=-1.0,
            ),
            verbose=1,
            seed=args.seed,
        )
    else:
        model = SAC(
            "MlpPolicy",
            env,
            learning_rate=3e-4,
            buffer_size=300_000,
            batch_size=256,
            gamma=0.99,
            policy_kwargs=dict(net_arch=[256, 256]),
            verbose=1,
            seed=args.seed,
        )

    gait_name = "trot (8-frame)" if args.trot else "walk-5 (5-pose)"
    print(f"\nTraining {args.algo.upper()} for {total_timesteps:,} steps  [{gait_name}]")
    print(f"Gait: {len(keyframes)} keyframes × {args.steps_per_frame} steps/frame "
          f"= {len(keyframes) * args.steps_per_frame * 0.025:.2f}s per cycle\n")

    model.learn(total_timesteps=total_timesteps, callback=eval_callback)
    model.save(args.save_path)
    env.close()
    eval_env.close()
    print(f"\nSaved model to {args.save_path}")
    print("Best model saved to best_model.zip (by EvalCallback)")


if __name__ == "__main__":
    main()
