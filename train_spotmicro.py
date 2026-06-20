#!/usr/bin/env python3
"""
Train an RL policy (PPO or SAC) for SpotMicro forward locomotion.
Usage:
  python train_spotmicro.py --algo ppo --total_timesteps 1000000 --save_path ./logs/spotmicro_ppo
  python train_spotmicro.py --render   # show MuJoCo viewer while training (slower)
"""

import argparse
import os

import gymnasium as gym
from stable_baselines3 import PPO, SAC

from spotmicro_env import SpotMicroEnv


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train SpotMicro locomotion")
    p.add_argument("--algo", type=str, default="ppo", choices=["ppo", "sac"])
    p.add_argument("--total_timesteps", type=float, default=1e6)
    p.add_argument("--save_path", type=str, default="./logs/spotmicro_ppo")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--max_episode_steps", type=int, default=2_000)
    p.add_argument("--render", action="store_true", help="Show MuJoCo viewer during training (slower)")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    total_timesteps = int(args.total_timesteps)

    render_mode = "human" if args.render else None
    env = SpotMicroEnv(
        max_episode_steps=args.max_episode_steps,
        render_mode=render_mode,
    )
    env = gym.wrappers.TimeLimit(env, max_episode_steps=args.max_episode_steps)
    env = gym.wrappers.RecordEpisodeStatistics(env)

    if args.algo == "ppo":
        model = PPO(
            "MlpPolicy",
            env,
            learning_rate=1e-4,       # lower lr: was causing large kl/clip_fraction
            n_steps=4096,
            batch_size=128,
            n_epochs=10,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.1,           # tighter clip: was 0.2, clip_fraction was 0.55-0.78
            ent_coef=0.01,            # more exploration early on
            policy_kwargs=dict(log_std_init=-1.0),
            verbose=1,
            seed=args.seed,
        )
    else:
        model = SAC(
            "MlpPolicy",
            env,
            learning_rate=3e-4,
            buffer_size=100_000,
            batch_size=256,
            gamma=0.99,
            verbose=1,
            seed=args.seed,
        )

    os.makedirs(os.path.dirname(args.save_path) or ".", exist_ok=True)
    model.learn(total_timesteps=total_timesteps)
    model.save(args.save_path)
    env.close()
    print(f"Saved model to {args.save_path}")


if __name__ == "__main__":
    main()
