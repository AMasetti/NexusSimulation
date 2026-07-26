"""
Render a frame of the current best policy and send it to Telegram.
Usage: python snapshot.py <timesteps> <mean_r> <ep_len> <prev_r> <total_steps>
"""
import sys, os, numpy as np, mujoco
from PIL import Image
import requests

BOT_TOKEN = os.environ["TELEGRAM_BOT_TOKEN"]
CHAT_ID   = os.environ.get("TELEGRAM_CHAT_ID", "212162466")
URDF      = os.path.join(os.path.dirname(__file__), "../urdf/full/optimus_mujoco_fixed.xml")
CKPT_DIR  = os.path.join(os.path.dirname(__file__), "checkpoints")

PARA_KP, PARA_KD, PARA_FMAX = 400.0, 20.0, 50.0
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


def send_photo(img_path, caption):
    with open(img_path, "rb") as f:
        requests.post(
            f"https://api.telegram.org/bot{BOT_TOKEN}/sendPhoto",
            data={"chat_id": CHAT_ID, "caption": caption, "parse_mode": "Markdown"},
            files={"photo": f},
        )


def render_snapshot(steps, mean_r, ep_len, prev_r, total_steps):
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import VecNormalize, DummyVecEnv
    from optimus_env import OptimusEnv

    best_zip = os.path.join(CKPT_DIR, "best", "best_model.zip")
    if not os.path.exists(best_zip):
        best_zip = os.path.join(CKPT_DIR, f"optimus_ppo_{steps}_steps.zip")

    model = PPO.load(best_zip, device="cpu")

    m = mujoco.MjModel.from_xml_path(URDF)
    d = mujoco.MjData(m)

    # Build joint/actuator maps
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
        for driven, follower, sign in PARA_PAIRS:
            q_err = sign * d.qpos[jpos[driven]] - d.qpos[jpos[follower]]
            v_err = sign * d.qvel[jdof[driven]] - d.qvel[jdof[follower]]
            torque = np.clip(PARA_KP * q_err + PARA_KD * v_err, -PARA_FMAX, PARA_FMAX)
            d.qfrc_applied[jdof[follower]] = torque

    # Warmup
    saved = d.qpos[:7].copy()
    for _ in range(1000):
        apply_para()
        mujoco.mj_step(m, d)
        d.qpos[:7] = saved
        d.qvel[:] = 0.0

    # Run policy for 2s (2000 steps) to get a meaningful pose
    obs_raw = np.concatenate([
        d.qpos[3:7], d.qvel[3:6], d.qvel[0:3], d.qpos[7:], d.qvel[6:]
    ]).astype(np.float32)

    for _ in range(100):  # 100 control steps @ 20 sim steps each
        action, _ = model.predict(obs_raw, deterministic=True)
        for i in range(m.nu):
            lo, hi = m.actuator_ctrlrange[i]
            d.ctrl[i] = float(action[i]) * (hi - lo) / 2.0
        for _ in range(20):
            apply_para()
            mujoco.mj_step(m, d)
        obs_raw = np.concatenate([
            d.qpos[3:7], d.qvel[3:6], d.qvel[0:3], d.qpos[7:], d.qvel[6:]
        ]).astype(np.float32)
        if not np.all(np.isfinite(obs_raw)):
            break

    # Render
    renderer = mujoco.Renderer(m, height=480, width=640)
    renderer.update_scene(d, camera=-1)
    pixels = renderer.render()
    img_path = "/tmp/optimus_snapshot.png"
    Image.fromarray(pixels).save(img_path)

    # Build caption
    pct = int(steps / total_steps * 100)
    filled = int(pct / 5)
    bar = "━" * filled + "░" * (20 - filled)
    delta = f"{mean_r - prev_r:+.1f}" if prev_r is not None else "—"
    caption = (
        f"🤖 *Optimus PPO — {steps/1e6:.1f}M steps*\n"
        f"{bar} {pct}%\n"
        f"mean\\_r: `{mean_r:+.2f}` ({delta})\n"
        f"ep\\_len: `{ep_len:.0f}` steps\n"
        f"best so far: `{mean_r:+.2f}` @ {steps/1e6:.1f}M"
    )
    send_photo(img_path, caption)
    print(f"Snapshot sent — {steps:,} steps, mean_r={mean_r:.2f}")


if __name__ == "__main__":
    steps    = int(sys.argv[1])
    mean_r   = float(sys.argv[2])
    ep_len   = float(sys.argv[3])
    prev_r   = float(sys.argv[4]) if sys.argv[4] != "None" else None
    total    = int(sys.argv[5])
    render_snapshot(steps, mean_r, ep_len, prev_r, total)
