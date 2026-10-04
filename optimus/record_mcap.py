#!/usr/bin/env python3
"""Record simulated Optimus episodes as rosbag2-style MCAP for the nexus-data pipeline.

Runs Optimus headless — the trained CPG+PPO policy (the newest run in rl/checkpoints,
like `make cpg-eval`), or the bare CPG gait with --model none — and writes, every
control step (50 Hz):
  /optimus/joint_states  sensor_msgs/JointState  servo angles named like the robot
  /optimus/imu/raw       sensor_msgs/Imu         the IMU site: gyro + specific force

Servo angles come from the URDF joints through futurespace's robot.json, inverting
`joint = scale * servo + offset` on each servo's pivot joint, so the episode has the
same shape as one recorded on the real robot. Episodes land in the same folder the
recorder uses (with nexus.json, source "sim", stamped at the source), so

    nexus-data ingest-landing

picks them up. Usage:
    python optimus/record_mcap.py --episodes 3 --seconds 10
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import mujoco
import numpy as np
from mcap_ros2.writer import Writer

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "rl"))
from optimus_cpg_env import OptimusCPGEnv  # noqa: E402

DEFAULT_ROBOT_JSON = HERE.parents[1] / "futurespace-ui" / "public" / "models" / "optimus" / "robot.json"

_SEP = "=" * 80 + "\n"
_TIME = "MSG: builtin_interfaces/Time\nint32 sec\nuint32 nanosec\n"
_HEADER = "MSG: std_msgs/Header\nbuiltin_interfaces/Time stamp\nstring frame_id\n"
_VEC3 = "MSG: geometry_msgs/Vector3\nfloat64 x\nfloat64 y\nfloat64 z\n"
_QUAT = "MSG: geometry_msgs/Quaternion\nfloat64 x\nfloat64 y\nfloat64 z\nfloat64 w\n"
JOINT_STATE = (
    "std_msgs/Header header\nstring[] name\nfloat64[] position\nfloat64[] velocity\n"
    "float64[] effort\n" + _SEP + _HEADER + _SEP + _TIME
)
IMU = (
    "std_msgs/Header header\ngeometry_msgs/Quaternion orientation\n"
    "float64[9] orientation_covariance\ngeometry_msgs/Vector3 angular_velocity\n"
    "float64[9] angular_velocity_covariance\ngeometry_msgs/Vector3 linear_acceleration\n"
    "float64[9] linear_acceleration_covariance\n"
    + _SEP + _HEADER + _SEP + _TIME + _SEP + _QUAT + _SEP + _VEC3
)


def servo_map(model: mujoco.MjModel, robot_json: Path) -> list[tuple[str, int, float, float]]:
    """(servo id, qpos address of its pivot joint, scale, offset rad) for every servo."""
    out = []
    for s in json.loads(robot_json.read_text())["servos"]:
        pivot = s.get("pivot") or s["joints"][0]["joint"]
        term = next(t for t in s["joints"] if t["joint"] == pivot)
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, pivot)
        if jid < 0:
            raise SystemExit(f"joint {pivot} (servo {s['id']}) is not in the MuJoCo model")
        out.append(
            (s["id"], int(model.jnt_qposadr[jid]), float(term.get("scale", 1)),
             math.radians(term.get("offsetDeg", 0)))
        )
    return out


def imu_reading(model: mujoco.MjModel, data: mujoco.MjData, site: int) -> tuple[np.ndarray, np.ndarray]:
    """Angular velocity and specific force (what an accelerometer reads) in the site frame."""
    mujoco.mj_rnePostConstraint(model, data)
    acc = np.zeros(6)
    mujoco.mj_objectAcceleration(model, data, mujoco.mjtObj.mjOBJ_SITE, site, acc, 1)
    vel = np.zeros(6)
    mujoco.mj_objectVelocity(model, data, mujoco.mjtObj.mjOBJ_SITE, site, vel, 1)
    rot = data.site_xmat[site].reshape(3, 3)
    specific_force = acc[3:] - rot.T @ model.opt.gravity
    return vel[:3], specific_force


class Policy:
    """The trained residual policy, or None for the bare CPG (zero residual)."""

    def __init__(self, model_path: Path | None) -> None:
        self.model = self.norm = None
        if model_path is None:
            return
        from stable_baselines3 import PPO
        from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

        self.model = PPO.load(str(model_path), device="cpu")
        stats = model_path.parent / "vecnorm.pkl"
        if stats.exists():
            self.norm = VecNormalize.load(str(stats), DummyVecEnv([OptimusCPGEnv]))
            self.norm.training = False

    def act(self, obs: np.ndarray, shape: tuple[int, ...]) -> np.ndarray:
        if self.model is None:
            return np.zeros(shape, dtype=np.float32)
        x = self.norm.normalize_obs(obs) if self.norm is not None else obs
        action, _ = self.model.predict(x, deterministic=True)
        return action


def newest_model() -> Path | None:
    runs = sorted((HERE / "rl" / "checkpoints").glob("*_run_*/best_model.zip"), reverse=True)
    return runs[0] if runs else None


def record(env: OptimusCPGEnv, policy: Policy, servos, seconds: float, out_dir: Path, name: str,
           labels: dict) -> Path:
    folder = out_dir / name
    folder.mkdir(parents=True, exist_ok=False)
    site = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_SITE, "imu")
    start_ns = int(datetime.now(timezone.utc).timestamp() * 1e9)
    names = [s[0] for s in servos]
    obs = env._get_obs()

    with open(folder / f"{name}_0.mcap", "wb") as f:
        w = Writer(f)
        js = w.register_msgdef("sensor_msgs/msg/JointState", JOINT_STATE)
        imu = w.register_msgdef("sensor_msgs/msg/Imu", IMU)
        steps = round(seconds / env.ctrl_dt)
        for k in range(steps + 1):
            if k:
                obs, _, terminated, truncated, _ = env.step(policy.act(obs, env.action_space.shape))
                if terminated or truncated:
                    print(f"  {name}: robot fell at {k * env.ctrl_dt:.2f} s, episode ends there")
                    break
            stamp = start_ns + round(k * env.ctrl_dt * 1e9)
            header = {"stamp": {"sec": stamp // 10**9, "nanosec": stamp % 10**9}, "frame_id": ""}
            q = env.data.qpos
            pos = [(float(q[adr]) - off) / scale for _, adr, scale, off in servos]
            w.write_message(
                "/optimus/joint_states", js,
                {"header": header, "name": names, "position": pos, "velocity": [], "effort": []},
                log_time=stamp, publish_time=stamp,
            )
            gyro, accel = imu_reading(env.model, env.data, site)
            w.write_message(
                "/optimus/imu/raw", imu,
                {"header": {**header, "frame_id": "imu_link"},
                 "orientation": {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0},
                 "orientation_covariance": [-1.0] + [0.0] * 8,
                 "angular_velocity": dict(zip("xyz", map(float, gyro))),
                 "angular_velocity_covariance": [0.0] * 9,
                 "linear_acceleration": dict(zip("xyz", map(float, accel))),
                 "linear_acceleration_covariance": [0.0] * 9},
                log_time=stamp, publish_time=stamp,
            )
        w.finish()

    # Written last, like the recorder: its presence marks the episode complete.
    (folder / "nexus.json").write_text(json.dumps({
        "name": name,
        "robot": "optimus",
        "source": "sim",
        "clock_source": "source",
        **labels,
        "started_at": datetime.fromtimestamp(start_ns / 1e9, timezone.utc).isoformat(),
    }, indent=2))
    return folder


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--episodes", type=int, default=1)
    p.add_argument("--seconds", type=float, default=10.0)
    p.add_argument("--task", default="walk")
    p.add_argument("--model", default="newest",
                   help='best_model.zip of a CPG+PPO run, "newest" (default) or "none" for the bare CPG')
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--robot-json", type=Path, default=DEFAULT_ROBOT_JSON)
    p.add_argument(
        "--out", type=Path,
        default=Path(os.environ.get("NEXUS_DATA_ROOT", Path.home() / "nexus-data")) / "landing",
        help="landing folder (default $NEXUS_DATA_ROOT/landing)",
    )
    args = p.parse_args()

    model = newest_model() if args.model == "newest" else None if args.model == "none" else Path(args.model)
    policy = Policy(model)
    print(f"policy: {model.relative_to(HERE) if model else 'bare CPG (zero residual)'}")
    env = OptimusCPGEnv()
    servos = servo_map(env.model, args.robot_json)
    args.out.mkdir(parents=True, exist_ok=True)
    for i in range(args.episodes):
        env.reset(seed=args.seed + i)
        stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
        name = f"optimus_{stamp}_sim-{args.task}-{i:02d}"
        how = model.parent.name if model else "bare CPG"
        labels = {"task": args.task, "operator": "mujoco",
                  "notes": f"OptimusCPGEnv seed {args.seed + i}, policy {how}"}
        folder = record(env, policy, servos, args.seconds, args.out, name, labels)
        print(f"wrote {folder}")


if __name__ == "__main__":
    main()
