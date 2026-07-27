"""
Real-time training monitor for Optimus CPG+RL.
Polls evaluations.npz, computes velocity from latest checkpoint,
and sends a Telegram update whenever a new eval point appears.

Usage: python monitor.py
"""

import os, time, numpy as np, requests, glob

BOT_TOKEN  = os.environ["TELEGRAM_BOT_TOKEN"]
CHAT_ID    = os.environ["TELEGRAM_CHAT_ID"]
CKPT_DIR   = os.path.join(os.path.dirname(__file__), "checkpoints_cpg")
EVAL_LOG   = os.path.join(CKPT_DIR, "logs", "evaluations.npz")
TOTAL      = 10_000_000
POLL_SEC   = 15


def send(text: str):
    requests.post(
        f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage",
        data={"chat_id": CHAT_ID, "parse_mode": "Markdown", "text": text},
        timeout=10,
    )


def send_photo(img_path: str, caption: str):
    with open(img_path, "rb") as f:
        requests.post(
            f"https://api.telegram.org/bot{BOT_TOKEN}/sendPhoto",
            data={"chat_id": CHAT_ID, "parse_mode": "Markdown", "caption": caption},
            files={"photo": f},
            timeout=15,
        )


def measure_velocity(ckpt_path: str) -> tuple:
    """Returns (mean_vel_ms, x_drift_m, survived_steps) from 5s rollout."""
    try:
        from stable_baselines3 import PPO
        from optimus_cpg_env import OptimusCPGEnv
        import warnings; warnings.filterwarnings("ignore")

        env = OptimusCPGEnv()
        model = PPO.load(ckpt_path, device="cpu")
        obs, _ = env.reset()
        ys, xs, steps = [], [], 0
        for _ in range(250):   # 5s at 50Hz
            a, _ = model.predict(obs, deterministic=True)
            obs, _, done, trunc, _ = env.step(a)
            ys.append(env.data.qpos[1])
            xs.append(env.data.qpos[0])
            steps += 1
            if done or trunc:
                break
        env.close()
        t = steps * 0.02
        vel  = (ys[-1] - ys[0]) / t if t > 0 else 0.0
        drift = abs(xs[-1] - xs[0])
        return round(vel, 3), round(drift, 3), steps
    except Exception as e:
        return 0.0, 0.0, 0


def render_frame(ckpt_path: str) -> str | None:
    """Render one frame of the policy, return path or None."""
    try:
        from stable_baselines3 import PPO
        from optimus_cpg_env import OptimusCPGEnv
        from PIL import Image
        import mujoco, warnings; warnings.filterwarnings("ignore")

        env = OptimusCPGEnv()
        model = PPO.load(ckpt_path, device="cpu")
        obs, _ = env.reset()
        for _ in range(100):   # 2s run-in
            a, _ = model.predict(obs, deterministic=True)
            obs, _, done, trunc, _ = env.step(a)
            if done or trunc: break

        renderer = mujoco.Renderer(env.model, height=480, width=640)
        renderer.update_scene(env.data, camera=-1)
        pixels = renderer.render()
        env.close()

        path = "/tmp/optimus_monitor_frame.png"
        Image.fromarray(pixels).save(path)
        return path
    except Exception:
        return None


def progress_bar(steps: int, total: int, width: int = 20) -> str:
    filled = int(steps / total * width)
    return "━" * filled + "░" * (width - filled)


def format_message(steps, mean_r, ep_len, vel, drift, survived,
                   prev_r, prev_vel, elapsed_min) -> str:
    pct  = steps / TOTAL * 100
    bar  = progress_bar(steps, TOTAL)
    dr   = f"{mean_r - prev_r:+.1f}" if prev_r is not None else "—"
    dv   = f"{vel - prev_vel:+.3f}" if prev_vel is not None else "—"
    eta  = (elapsed_min / pct * (100 - pct)) if pct > 0 else 0

    lines = [
        f"🤖 *Optimus Run 7 — {steps/1e6:.2f}M steps*",
        f"`{bar}` {pct:.1f}%",
        f"",
        f"📈 *Reward:*  `{mean_r:+.1f}` ({dr})",
        f"⏱ *Ep len:*  `{ep_len:.0f}` steps",
        f"",
        f"🏃 *Forward vel:* `{vel:+.3f} m/s` ({dv})",
        f"↔️ *Lateral drift:* `{drift:.3f} m` in 5s",
        f"🦵 *Survived:* `{survived}` steps",
        f"",
        f"⏳ *ETA:* ~{eta:.0f} min",
    ]
    return "\n".join(lines)


def main():
    print("Monitor started — polling every", POLL_SEC, "s")
    send("🟢 *Monitor started* — Optimus Run 7 (12 envs, MPS)\nWaiting for first eval...")

    seen_steps  = set()
    prev_r      = None
    prev_vel    = None
    start_time  = time.time()

    while True:
        time.sleep(POLL_SEC)

        if not os.path.exists(EVAL_LOG):
            continue

        try:
            d = np.load(EVAL_LOG)
        except Exception:
            continue

        timesteps = d["timesteps"]
        results   = d["results"].mean(axis=1)
        ep_lens   = d["ep_lengths"].mean(axis=1)

        for i, steps in enumerate(timesteps):
            steps = int(steps)
            if steps in seen_steps:
                continue
            seen_steps.add(steps)

            mean_r = float(results[i])
            ep_len = float(ep_lens[i])
            elapsed_min = (time.time() - start_time) / 60

            # Find matching checkpoint
            pattern = os.path.join(CKPT_DIR, f"cpg_ppo_{steps}_steps.zip")
            matches = glob.glob(pattern)
            ckpt    = matches[0] if matches else os.path.join(CKPT_DIR, "latest_model.zip")

            vel, drift, survived = measure_velocity(ckpt) if os.path.exists(ckpt) else (0, 0, 0)

            msg  = format_message(steps, mean_r, ep_len, vel, drift, survived,
                                  prev_r, prev_vel, elapsed_min)
            frame = render_frame(ckpt) if os.path.exists(ckpt) else None

            if frame:
                send_photo(frame, msg)
            else:
                send(msg)

            print(f"[{steps:,}] r={mean_r:.1f} vel={vel:.3f}m/s drift={drift:.3f}m")
            prev_r   = mean_r
            prev_vel = vel

            # Stop when done
            if steps >= TOTAL:
                send("✅ *Training complete!* Run `make sim-cpg-eval` to watch the policy.")
                return


if __name__ == "__main__":
    main()
