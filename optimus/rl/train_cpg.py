"""
CPG + PPO training for Optimus.

Usage:
    python train_cpg.py                        # fresh run
    python train_cpg.py --resume <run_dir>     # resume from a timestamped run
    python train_cpg.py --eval <model.zip>     # evaluate a saved policy
    python train_cpg.py --cpg-only             # visualise CPG gait only (no RL)

Run directories: checkpoints/<timestamp>_run_N/
  best_model.zip     — best policy seen so far this run (eval reward)
  latest_model.zip   — checkpoint at last save (resumable)
  vecnorm.pkl        — VecNormalize stats
  progress.log       — per-step diagnostics, one line per 100k steps
  worker_<id>.log    — per-worker live episode diagnostics
"""

import argparse, os, time, datetime, glob, sys
import numpy as np

CKPT_ROOT  = os.path.join(os.path.dirname(__file__), "checkpoints")
N_ENVS     = 12
TOTAL_STEPS = 10_000_000


def _run_dir(label: str = "") -> str:
    ts  = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    # Count existing runs to assign N
    existing = glob.glob(os.path.join(CKPT_ROOT, "*_run_*"))
    n = len(existing) + 1
    suffix = f"_{label}" if label else ""
    return os.path.join(CKPT_ROOT, f"{ts}_run_{n}{suffix}")


def train(resume_dir: str | None = None):
    from stable_baselines3 import PPO
    from stable_baselines3.common.env_util import make_vec_env
    from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize
    from stable_baselines3.common.callbacks import EvalCallback, BaseCallback
    from stable_baselines3.common.monitor import Monitor
    from optimus_cpg_env import OptimusCPGEnv

    run_dir = resume_dir or _run_dir()
    os.makedirs(run_dir, exist_ok=True)

    progress_log = open(os.path.join(run_dir, "progress.log"), "a", buffering=1)
    progress_log.write(f"\n# Run started {datetime.datetime.now().isoformat()}\n")

    def make_env(rank):
        def _init():
            env = OptimusCPGEnv(worker_id=rank)
            log_path = os.path.join(run_dir, f"worker_{rank}.log")
            return Monitor(env, filename=log_path, info_keywords=())
        return _init

    env_fns  = [make_env(i) for i in range(N_ENVS)]
    vec_env  = SubprocVecEnv(env_fns)
    vec_env  = VecNormalize(vec_env, norm_obs=True, norm_reward=True, clip_obs=10.0)
    eval_env = VecNormalize(
        SubprocVecEnv([make_env(99)]),
        norm_obs=True, norm_reward=False, training=False
    )

    latest = os.path.join(run_dir, "latest_model.zip")
    vn_path = os.path.join(run_dir, "vecnorm.pkl")

    if resume_dir and os.path.exists(latest):
        print(f"[train] Resuming from {latest}")
        model = PPO.load(latest, env=vec_env, device="mps")
        if os.path.exists(vn_path):
            vec_env = VecNormalize.load(vn_path, vec_env)
    else:
        model = PPO(
            "MlpPolicy", vec_env,
            verbose=0, device="mps",
            learning_rate=3e-4,
            n_steps=4096,
            batch_size=512,
            n_epochs=10,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=0.005,
            policy_kwargs=dict(net_arch=[256, 256]),
        )

    class DiagCB(BaseCallback):
        """Prints per-step diagnostics, writes progress.log, saves latest checkpoint."""
        def __init__(self):
            super().__init__()
            self._last = 0

        def _on_step(self):
            if self.num_timesteps - self._last < 100_000:
                return True
            self._last = self.num_timesteps

            buf = self.model.ep_info_buffer
            mean_r   = float(np.mean([e["r"] for e in buf])) if buf else 0.0
            mean_len = float(np.mean([e["l"] for e in buf])) if buf else 0.0

            line = (f"[{self.num_timesteps:>9,}]  "
                    f"reward={mean_r:+.1f}  ep_len={mean_len:.0f}")
            print(line, flush=True)
            progress_log.write(line + "\n")

            # Always keep a loadable latest checkpoint for --eval
            self.model.save(latest)
            vec_env.save(vn_path)
            return True

    callbacks = [
        DiagCB(),
        EvalCallback(
            eval_env,
            best_model_save_path=run_dir,   # saves best_model.zip
            log_path=os.path.join(run_dir, "logs"),
            eval_freq=500_000 // N_ENVS,
            n_eval_episodes=5,
            deterministic=True,
            verbose=1,
        ),
    ]

    print(f"[train] Run dir: {run_dir}")
    print(f"[train] {TOTAL_STEPS:,} steps × {N_ENVS} envs  device=mps")
    model.learn(
        total_timesteps=TOTAL_STEPS,
        callback=callbacks,
        reset_num_timesteps=(resume_dir is None),
    )

    model.save(latest)
    vec_env.save(vn_path)
    progress_log.write(f"# Run finished {datetime.datetime.now().isoformat()}\n")
    progress_log.close()
    print(f"[train] Done → {run_dir}")


def evaluate(model_path: str):
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
    from optimus_cpg_env import OptimusCPGEnv

    # Load VecNormalize stats from same directory as model — required for correct inference.
    run_dir  = os.path.dirname(os.path.abspath(model_path))
    vn_path  = os.path.join(run_dir, "vecnorm.pkl")

    raw_env  = DummyVecEnv([lambda: OptimusCPGEnv(render_mode="human")])
    if os.path.exists(vn_path):
        env = VecNormalize.load(vn_path, raw_env)
        env.training = False   # freeze stats — don't update during eval
        env.norm_reward = False
        print(f"[eval] Loaded VecNormalize from {vn_path}")
    else:
        env = raw_env
        print(f"[eval] WARNING: no vecnorm.pkl found at {vn_path} — obs will be unnormalised")

    model = PPO.load(model_path, env=env, device="cpu")
    obs = env.reset()
    total_r, steps = 0.0, 0

    while True:
        t0 = time.perf_counter()
        action, _ = model.predict(obs, deterministic=True)
        obs, r, done, info = env.step(action)
        total_r += float(r[0])
        steps   += 1

        if done[0]:
            inner = env.envs[0] if hasattr(env, 'envs') else env.venv.envs[0]
            d = inner.diagnostics()
            print(f"Episode  steps={steps}  reward={total_r:.1f}  "
                  f"fwd={d['fwd_dist_m']}m  drift={d['lat_drift_m']}m  "
                  f"pitch={d['imu_pitch']}°  roll={d['imu_roll']}°  "
                  f"yaw_rate={d['yaw_rate']}°/s")
            obs = env.reset()
            total_r = 0.0
            steps   = 0

        elapsed = time.perf_counter() - t0
        time.sleep(max(0.0, 0.02 - elapsed))


def cpg_only():
    """Visualise raw CPG gait without any RL policy."""
    import mujoco, mujoco.viewer
    from cpg import CPG

    URDF = os.path.join(os.path.dirname(__file__), "../urdf/full/optimus_mujoco_fixed.xml")
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
    PARA_KP, PARA_KD, PARA_FMAX = 400.0, 20.0, 5.0
    KP, KD = 6.0, 0.4

    m = mujoco.MjModel.from_xml_path(URDF)
    d = mujoco.MjData(m)

    jpos, jdof, amap = {}, {}, {}
    for i in range(m.njnt):
        name = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, i)
        if name and name != "root":
            jpos[name] = m.jnt_qposadr[i]
            jdof[name] = m.jnt_dofadr[i]
    for i in range(m.nu):
        name = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
        if name:
            amap[name] = i

    def apply_para():
        for dr, fl, sg in PARA_PAIRS:
            if dr not in jpos or fl not in jpos: continue
            q_err = sg * d.qpos[jpos[dr]] - d.qpos[jpos[fl]]
            v_err = sg * d.qvel[jdof[dr]] - d.qvel[jdof[fl]]
            d.qfrc_applied[jdof[fl]] = np.clip(
                PARA_KP * q_err + PARA_KD * v_err, -PARA_FMAX, PARA_FMAX
            )

    # Warmup with standing crouch
    for name, angle in [("Servo-Knee-L-Top",    -np.radians(23.0)),
                         ("Servo-Knee-L-Bottom", -np.radians(23.0)),
                         ("Servo-Knee-R-Top",    -np.radians(23.0)),
                         ("Servo-Knee-R-Bottom", -np.radians(23.0))]:
        if name in jpos:
            d.qpos[jpos[name]] = angle

    saved = d.qpos[:7].copy()
    for _ in range(1000):
        apply_para(); mujoco.mj_step(m, d)
        d.qpos[:7] = saved; d.qvel[:] = 0.0

    cpg = CPG()
    CTRL_DT = 1.0 / 50   # 50 Hz — hardware cadence

    print("[cpg-only] Running at 50 Hz. Watch for forward motion (+Y).")
    print("           Pitch/roll printed every 2s.")
    last_print = time.perf_counter()

    with mujoco.viewer.launch_passive(m, d) as viewer:
        while viewer.is_running():
            t0 = time.perf_counter()

            targets = cpg.step(CTRL_DT)

            # Simple IMU from quaternion
            q = d.qpos[3:7]
            pitch = float(np.arctan2(2*(q[0]*q[1]+q[2]*q[3]), 1-2*(q[1]**2+q[2]**2)))
            roll  = float(np.arctan2(2*(q[0]*q[3]+q[1]*q[2]), 1-2*(q[2]**2+q[3]**2)))
            yaw_r = float(d.qvel[4])
            targets = cpg.stabilise(targets, pitch, roll, yaw_r)

            for name, target in targets.items():
                if name not in amap or name not in jpos: continue
                q_j = d.qpos[jpos[name]]
                dq  = d.qvel[jdof[name]]
                torque = KP * (target - q_j) - KD * dq
                lo, hi = m.actuator_ctrlrange[amap[name]]
                d.ctrl[amap[name]] = np.clip(torque, lo, hi)

            apply_para()
            mujoco.mj_step(m, d)
            viewer.sync()

            now = time.perf_counter()
            if now - last_print >= 2.0:
                print(f"  y={d.qpos[1]:.3f}m  x={d.qpos[0]:.3f}m  "
                      f"z={d.qpos[2]:.3f}m  pitch={np.degrees(pitch):.1f}°  "
                      f"roll={np.degrees(roll):.1f}°  yaw_rate={np.degrees(yaw_r):.1f}°/s",
                      flush=True)
                last_print = now

            time.sleep(max(0.0, CTRL_DT - (time.perf_counter() - t0)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume",   type=str, default=None,
                        help="Path to a timestamped run dir to resume from")
    parser.add_argument("--eval",     type=str, default=None,
                        help="Path to a .zip model to evaluate in viewer")
    parser.add_argument("--cpg-only", action="store_true",
                        help="Visualise CPG gait without RL")
    args = parser.parse_args()

    if args.cpg_only:
        cpg_only()
    elif args.eval:
        evaluate(args.eval)
    else:
        train(resume_dir=args.resume or None)
