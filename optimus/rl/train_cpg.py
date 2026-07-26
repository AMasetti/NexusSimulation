"""
PPO training on top of CPG gait for Optimus.

Usage:
    python train_cpg.py                     # fresh run
    python train_cpg.py --resume            # resume from latest checkpoint
    python train_cpg.py --eval <path>       # watch a saved policy
    python train_cpg.py --cpg-only          # visualise CPG without RL
"""

import argparse, os, time, numpy as np

CHECKPOINT_DIR = os.path.join(os.path.dirname(__file__), "checkpoints_cpg")
N_ENVS         = 8
TOTAL_STEPS    = 10_000_000   # CPG converges faster — 10M is enough


def train(resume=False):
    from stable_baselines3 import PPO
    from stable_baselines3.common.env_util import make_vec_env
    from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize
    from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback, BaseCallback
    from optimus_cpg_env import OptimusCPGEnv

    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    vec_env  = make_vec_env(OptimusCPGEnv, n_envs=N_ENVS, vec_env_cls=SubprocVecEnv)
    vec_env  = VecNormalize(vec_env, norm_obs=True, norm_reward=True, clip_obs=10.0)
    eval_env = VecNormalize(make_vec_env(OptimusCPGEnv, n_envs=1), norm_obs=True, norm_reward=False, training=False)

    latest = os.path.join(CHECKPOINT_DIR, "latest_model.zip")

    if resume and os.path.exists(latest):
        print(f"Resuming from {latest}")
        model = PPO.load(latest, env=vec_env, device="cpu")
        vn    = os.path.join(CHECKPOINT_DIR, "vecnorm.pkl")
        if os.path.exists(vn):
            vec_env = VecNormalize.load(vn, vec_env)
    else:
        model = PPO(
            "MlpPolicy", vec_env,
            verbose=0, device="cpu",
            learning_rate=3e-4, n_steps=2048, batch_size=512,
            n_epochs=10, gamma=0.99, gae_lambda=0.95,
            clip_range=0.2, ent_coef=0.005,
            policy_kwargs=dict(net_arch=[256, 256]),   # smaller net — residuals are simple
            tensorboard_log=None,
        )

    class ProgressCB(BaseCallback):
        def __init__(self): super().__init__(); self._last = 0
        def _on_step(self):
            if self.num_timesteps - self._last >= 100_000:
                self._last = self.num_timesteps
                if self.model.ep_info_buffer:
                    r = np.mean([e["r"] for e in self.model.ep_info_buffer])
                    l = np.mean([e["l"] for e in self.model.ep_info_buffer])
                    print(f"[{self.num_timesteps:>9,}]  mean_r={r:+.1f}  ep_len={l:.0f}")
            return True

    callbacks = [
        ProgressCB(),
        CheckpointCallback(save_freq=250_000 // N_ENVS, save_path=CHECKPOINT_DIR,
                           name_prefix="cpg_ppo", save_vecnormalize=True),
        EvalCallback(eval_env, best_model_save_path=os.path.join(CHECKPOINT_DIR, "best"),
                     log_path=os.path.join(CHECKPOINT_DIR, "logs"),
                     eval_freq=500_000 // N_ENVS, n_eval_episodes=5, deterministic=True),
    ]

    print(f"CPG+PPO training — {TOTAL_STEPS:,} steps across {N_ENVS} envs")
    model.learn(total_timesteps=TOTAL_STEPS, callback=callbacks, reset_num_timesteps=not resume)
    model.save(latest)
    vec_env.save(os.path.join(CHECKPOINT_DIR, "vecnorm.pkl"))
    print(f"Done → {latest}")


def evaluate(model_path):
    from stable_baselines3 import PPO
    from optimus_cpg_env import OptimusCPGEnv

    env   = OptimusCPGEnv(render_mode="human")
    model = PPO.load(model_path, device="cpu")
    obs, _ = env.reset()
    total_r, steps = 0.0, 0
    while True:
        t0 = time.perf_counter()
        action, _ = model.predict(obs, deterministic=True)
        obs, r, terminated, truncated, _ = env.step(action)
        total_r += r; steps += 1
        if terminated or truncated:
            print(f"Episode — steps={steps}  reward={total_r:.1f}")
            obs, _ = env.reset(); total_r = 0.0; steps = 0
        time.sleep(max(0.0, env.ctrl_dt - (time.perf_counter() - t0)))


def cpg_only():
    """Visualise the raw CPG gait without any RL policy."""
    import mujoco, mujoco.viewer
    from cpg import CPG

    PARA_KP, PARA_KD, PARA_FMAX = 400.0, 20.0, 5.0
    PARA_PAIRS = [
        ("Servo-Knee-L-Top",    "Unactuated-Knee-L-Top",      -1),
        ("Servo-Knee-L-Top",    "Unactuated-Tendon-L-Top",    -1),
        ("Servo-Knee-L-Bottom", "Unactuated-Knee-L-Bottom",   -1),
        ("Servo-Knee-L-Bottom", "Unactuated-Tendon-L-Bottom", +1),
        ("Servo-Knee-R-Top",    "Unactuated-Knee-R-Top",      -1),
        ("Servo-Knee-R-Top",    "Unactuated-Tendon-R-Top",    -1),
        ("Servo-Knee-R-Bottom", "Unactuated-Knee-R-Bottom",   -1),
        ("Servo-Knee-R-Bottom", "Unactuated-Tendon-R-Bottom", +1),
    ]
    URDF = os.path.join(os.path.dirname(__file__), "../urdf/full/optimus_mujoco_fixed.xml")

    m = mujoco.MjModel.from_xml_path(URDF)
    d = mujoco.MjData(m)

    jpos, jdof, amap = {}, {}, {}
    for i in range(m.njnt):
        name = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, i)
        if name and name != "root": jpos[name] = m.jnt_qposadr[i]; jdof[name] = m.jnt_dofadr[i]
    for i in range(m.nu):
        name = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
        if name: amap[name] = i

    KP, KD = 8.0, 0.4

    def apply_para():
        for dr, fl, sg in PARA_PAIRS:
            q_err = sg * d.qpos[jpos[dr]] - d.qpos[jpos[fl]]
            v_err = sg * d.qvel[jdof[dr]] - d.qvel[jdof[fl]]
            d.qfrc_applied[jdof[fl]] = np.clip(PARA_KP * q_err + PARA_KD * v_err, -PARA_FMAX, PARA_FMAX)

    # Warmup
    saved = d.qpos[:7].copy()
    for _ in range(1000):
        apply_para(); mujoco.mj_step(m, d)
        d.qpos[:7] = saved; d.qvel[:] = 0.0

    cpg = CPG()

    def imu_pitch():
        q = d.qpos[3:7]
        return float(2.0 * (q[0] * q[2] - q[3] * q[1]))

    with mujoco.viewer.launch_passive(m, d) as viewer:
        t_last = time.perf_counter()
        while viewer.is_running():
            now = time.perf_counter()
            dt  = now - t_last; t_last = now

            targets = cpg.step(dt)
            targets = cpg.stabilise(targets, imu_pitch())

            for name, target in targets.items():
                if name not in amap or name not in jpos: continue
                q  = d.qpos[jpos[name]]; dq = d.qvel[jdof[name]]
                torque = KP * (target - q) - KD * dq
                lo, hi = m.actuator_ctrlrange[amap[name]]
                d.ctrl[amap[name]] = np.clip(torque, lo, hi)

            apply_para()
            mujoco.mj_step(m, d)
            viewer.sync()
            time.sleep(max(0.0, 0.001 - (time.perf_counter() - now)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume",   action="store_true")
    parser.add_argument("--eval",     type=str, default=None)
    parser.add_argument("--cpg-only", action="store_true")
    args = parser.parse_args()

    if args.cpg_only:
        cpg_only()
    elif args.eval:
        evaluate(args.eval)
    else:
        train(resume=args.resume)
