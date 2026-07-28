"""
Training monitor — polls the latest run for eval results, sends Telegram text only.

Sends when:
  - A new eval point appears (every ~500k steps)
  - Reward regresses >5 points from best (diagnostic alert)
  - Training completes

No images — they added no diagnostic value.

Usage:
    python monitor.py          # auto-detects latest run dir
    python monitor.py <dir>    # monitor a specific run dir
"""

import os, sys, time, glob, datetime
import numpy as np
import requests

BOT_TOKEN = os.environ["TELEGRAM_BOT_TOKEN"]
CHAT_ID   = os.environ["TELEGRAM_CHAT_ID"]
TOTAL     = 10_000_000
POLL_SEC  = 20


def send(text: str):
    try:
        requests.post(
            f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage",
            data={"chat_id": CHAT_ID, "parse_mode": "Markdown", "text": text},
            timeout=10,
        )
    except Exception as e:
        print(f"[monitor] Telegram error: {e}", flush=True)


def _bar(steps: int, total: int, width: int = 20) -> str:
    filled = int(steps / total * width)
    return "█" * filled + "░" * (width - filled)


def measure_velocity(ckpt_path: str) -> tuple[float, float, int]:
    """5-second rollout → (fwd_vel m/s, lat_drift m, survived_steps)."""
    try:
        from stable_baselines3 import PPO
        from optimus_cpg_env import OptimusCPGEnv
        import warnings; warnings.filterwarnings("ignore")

        env   = OptimusCPGEnv()
        model = PPO.load(ckpt_path, device="cpu")
        obs, _ = env.reset()
        ys, xs, steps = [], [], 0
        for _ in range(250):
            a, _ = model.predict(obs, deterministic=True)
            obs, _, done, trunc, _ = env.step(a)
            ys.append(env.data.qpos[1])
            xs.append(env.data.qpos[0])
            steps += 1
            if done or trunc:
                break
        env.close()
        t   = steps * env.ctrl_dt
        vel = (ys[-1] - ys[0]) / t if t > 0 else 0.0
        return round(vel, 3), round(abs(xs[-1] - xs[0]), 3), steps
    except Exception as e:
        print(f"[monitor] velocity measure failed: {e}", flush=True)
        return 0.0, 0.0, 0


def latest_run_dir() -> str | None:
    dirs = sorted(glob.glob(
        os.path.join(os.path.dirname(__file__), "checkpoints", "*_run_*")
    ))
    return dirs[-1] if dirs else None


def main():
    run_dir = sys.argv[1] if len(sys.argv) > 1 else latest_run_dir()
    if not run_dir:
        print("[monitor] No run dir found. Start training first.", flush=True)
        return

    eval_log = os.path.join(run_dir, "logs", "evaluations.npz")
    run_name = os.path.basename(run_dir)

    print(f"[monitor] Watching {run_dir}", flush=True)
    send(f"🟢 *Monitor* — `{run_name}`\nPolling every {POLL_SEC}s...")

    seen_steps = set()
    best_r     = None
    prev_vel   = None
    start_time = time.time()

    while True:
        time.sleep(POLL_SEC)

        if not os.path.exists(eval_log):
            continue

        try:
            d = np.load(eval_log)
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
            pct    = steps / TOTAL * 100
            elapsed_min = (time.time() - start_time) / 60
            eta_min = (elapsed_min / pct * (100 - pct)) if pct > 0 else 0

            # Find best model checkpoint
            best_zip = os.path.join(run_dir, "best_model.zip")
            ckpt     = best_zip if os.path.exists(best_zip) else None

            vel, drift, survived = measure_velocity(ckpt) if ckpt else (0.0, 0.0, 0)

            dv   = f"{vel - prev_vel:+.3f}" if prev_vel is not None else "—"
            dr   = f"{mean_r - best_r:+.1f}" if best_r is not None else "—"

            # Diagnose: what's the robot actually doing?
            if survived < 50:
                diag = "🔴 falling fast (<1s) — gait unstable"
            elif vel < 0.0:
                diag = "🟡 moving backward — CPG phase or knee sign issue"
            elif vel < 0.05:
                diag = "🟡 barely moving — reward shaping or action too small"
            elif drift > 0.3:
                diag = "🟡 lateral drift — yaw penalty needs tuning"
            else:
                diag = "🟢 forward progress"

            msg = (
                f"📊 `{run_name}`\n"
                f"`{_bar(steps, TOTAL)}` {pct:.1f}%  ({steps/1e6:.2f}M / {TOTAL/1e6:.0f}M)\n"
                f"\n"
                f"Reward:   `{mean_r:+.1f}` ({dr})\n"
                f"Ep len:   `{ep_len:.0f}` steps ({ep_len * env_dt():.1f}s)\n"
                f"\n"
                f"Fwd vel:  `{vel:+.3f} m/s` ({dv})\n"
                f"Lat drift:`{drift:.3f} m` in {survived * env_dt():.1f}s\n"
                f"Survived: `{survived}` steps\n"
                f"\n"
                f"{diag}\n"
                f"\n"
                f"ETA: ~{eta_min:.0f} min"
            )

            send(msg)
            print(f"[{steps:,}] r={mean_r:.1f} vel={vel:+.3f}m/s drift={drift:.3f}m survived={survived}",
                  flush=True)

            best_r   = max(mean_r, best_r) if best_r is not None else mean_r
            prev_vel = vel

            if steps >= TOTAL:
                send(f"✅ *Training complete* — `{run_name}`\n"
                     f"Best reward: `{best_r:.1f}`\n"
                     f"Run: `make sim-cpg-eval`")
                return


def env_dt() -> float:
    return 1.0 / 50   # 50 Hz control cadence


if __name__ == "__main__":
    main()
