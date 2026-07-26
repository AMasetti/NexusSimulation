"""
PPO training for Optimus walking.

Usage:
    python train.py                     # fresh training
    python train.py --resume            # resume from latest checkpoint
    python train.py --eval checkpoints/best_model  # watch a saved policy
"""
import argparse
import os
import numpy as np

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize
from stable_baselines3.common.callbacks import (
    CheckpointCallback, EvalCallback, BaseCallback
)

from optimus_env import OptimusEnv

CHECKPOINT_DIR = os.path.join(os.path.dirname(__file__), "checkpoints")
LOG_DIR        = os.path.join(os.path.dirname(__file__), "logs")
N_ENVS         = 8       # parallel envs (CPU-bound, not GPU)
TOTAL_STEPS    = 20_000_000


class ProgressCallback(BaseCallback):
    """Print mean reward every 100k steps."""
    def __init__(self, log_interval=100_000):
        super().__init__()
        self._interval = log_interval
        self._last = 0

    def _on_step(self):
        if self.num_timesteps - self._last >= self._interval:
            self._last = self.num_timesteps
            if len(self.model.ep_info_buffer) > 0:
                mean_r = np.mean([ep["r"] for ep in self.model.ep_info_buffer])
                mean_l = np.mean([ep["l"] for ep in self.model.ep_info_buffer])
                print(f"[{self.num_timesteps:>9,}]  mean_ep_reward={mean_r:+.2f}  mean_ep_len={mean_l:.0f}")
        return True


def make_env():
    return OptimusEnv()


def train(resume=False):
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)

    vec_env = make_vec_env(make_env, n_envs=N_ENVS, vec_env_cls=SubprocVecEnv)
    vec_env = VecNormalize(vec_env, norm_obs=True, norm_reward=True, clip_obs=10.0)

    latest_ckpt = os.path.join(CHECKPOINT_DIR, "latest_model.zip")
    vecnorm_path = os.path.join(CHECKPOINT_DIR, "vecnorm.pkl")

    if resume and os.path.exists(latest_ckpt):
        print(f"Resuming from {latest_ckpt}")
        model = PPO.load(latest_ckpt, env=vec_env, device="mps")
        if os.path.exists(vecnorm_path):
            vec_env = VecNormalize.load(vecnorm_path, vec_env)
    else:
        model = PPO(
            "MlpPolicy",
            vec_env,
            verbose=0,
            device="cpu",
            learning_rate=3e-4,
            n_steps=2048,
            batch_size=512,
            n_epochs=10,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=0.01,
            policy_kwargs=dict(net_arch=[512, 512, 256]),
            tensorboard_log=None,
        )

    # Eval env (single, no normalisation sharing)
    eval_env = VecNormalize(
        make_vec_env(make_env, n_envs=1),
        norm_obs=True, norm_reward=False, training=False
    )

    callbacks = [
        ProgressCallback(log_interval=100_000),
        CheckpointCallback(
            save_freq=250_000 // N_ENVS,
            save_path=CHECKPOINT_DIR,
            name_prefix="optimus_ppo",
            save_vecnormalize=True,
        ),
        EvalCallback(
            eval_env,
            best_model_save_path=os.path.join(CHECKPOINT_DIR, "best"),
            log_path=LOG_DIR,
            eval_freq=500_000 // N_ENVS,
            n_eval_episodes=5,
            deterministic=True,
        ),
    ]

    print(f"Training Optimus PPO — {TOTAL_STEPS:,} steps across {N_ENVS} envs")
    print(f"Checkpoints → {CHECKPOINT_DIR}")
    print()

    model.learn(
        total_timesteps=TOTAL_STEPS,
        callback=callbacks,
        reset_num_timesteps=not resume,
        tb_log_name="optimus_ppo",
    )

    model.save(latest_ckpt)
    vec_env.save(vecnorm_path)
    print(f"\nSaved → {latest_ckpt}")


def evaluate(model_path):
    env = OptimusEnv(render_mode="human")
    model = PPO.load(model_path, device="mps")

    obs, _ = env.reset()
    total_reward = 0.0
    steps = 0
    while True:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, _ = env.step(action)
        total_reward += reward
        steps += 1
        if terminated or truncated:
            print(f"Episode done — steps={steps}  total_reward={total_reward:.2f}")
            obs, _ = env.reset()
            total_reward = 0.0
            steps = 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--eval", type=str, default=None, help="Path to model to evaluate")
    args = parser.parse_args()

    if args.eval:
        evaluate(args.eval)
    else:
        train(resume=args.resume)
