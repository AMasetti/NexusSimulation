"""
Gymnasium environment for the SpotMicro quadruped.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import mujoco
import gymnasium as gym
from gymnasium import spaces

from spotmicro_loader import load_spotmicro_xml

JOINT_NAMES = [
    "front_left_shoulder", "front_left_leg", "front_left_foot",
    "front_right_shoulder", "front_right_leg", "front_right_foot",
    "rear_left_shoulder", "rear_left_leg", "rear_left_foot",
    "rear_right_shoulder", "rear_right_leg", "rear_right_foot",
]

DEFAULT_BASE_HEIGHT = 0.259   # standing height from XML
FREE_JOINT_QPOS_SIZE = 7
NUM_JOINTS = 12


def _quat_to_euler(quat: np.ndarray) -> np.ndarray:
    w, x, y, z = quat[0], quat[1], quat[2], quat[3]
    sinr_cosp = 2 * (w * x + y * z)
    cosr_cosp = 1 - 2 * (x * x + y * y)
    roll = np.arctan2(sinr_cosp, cosr_cosp)
    sinp = np.clip(2 * (w * y - z * x), -1.0, 1.0)
    pitch = np.arcsin(sinp)
    siny_cosp = 2 * (w * z + x * y)
    cosy_cosp = 1 - 2 * (y * y + z * z)
    yaw = np.arctan2(siny_cosp, cosy_cosp)
    return np.array([roll, pitch, yaw], dtype=np.float64)


class SpotMicroEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 50}

    def __init__(
        self,
        max_episode_steps: int = 60_000,
        frame_skip: int = 5,            # 5 × 0.005s = 0.025s per action (40 Hz)
        reward_vel: float = 6.0,       # forward velocity reward (only when upright)
        reward_tilt: float = -2.0,      # roll penalty
        reward_pitch: float = -4.0,     # pitch (nose-dive) penalized
        reward_pitch_back: float = -8.0, # falling backwards penalized harder
        reward_height: float = -3.0,    # penalize deviation from standing height
        reward_action: float = -0.05,
        reward_survival: float = 0.2,   # per-step bonus for staying upright (incentivises not falling)
        reward_joint_vel: float = -0.01,  # penalize rapid joint oscillation
        reward_lateral: float = -1.0,     # penalize sideways drift
        reward_yaw_rate: float = -0.5,    # penalize spinning in place
        reward_fall: float = -2.0,
        upright_height_min: float = 0.18,
        upright_tilt_sq_max: float = 0.15,
        height_threshold: float = 0.15, # terminate before touching ground
        tilt_threshold: float = 0.5,
        reset_noise_scale: float = 0.02,
        render_mode: str | None = None,
    ):
        super().__init__()
        self.max_episode_steps = max_episode_steps
        self.frame_skip = frame_skip
        self.reward_vel = reward_vel
        self.reward_tilt = reward_tilt
        self.reward_pitch = reward_pitch
        self.reward_pitch_back = reward_pitch_back
        self.reward_height = reward_height
        self.reward_action = reward_action
        self.reward_survival = reward_survival
        self.reward_joint_vel = reward_joint_vel
        self.reward_lateral = reward_lateral
        self.reward_yaw_rate = reward_yaw_rate
        self.reward_fall = reward_fall
        self._leg_foot_actuator_idx = np.array([1, 2, 4, 5, 7, 8, 10, 11], dtype=np.intp)
        self.upright_height_min = upright_height_min
        self.upright_tilt_sq_max = upright_tilt_sq_max
        self.height_threshold = height_threshold
        self.tilt_threshold = tilt_threshold
        self.reset_noise_scale = reset_noise_scale
        self.render_mode = render_mode

        xml_str = load_spotmicro_xml()
        self.model = mujoco.MjModel.from_xml_string(xml_str)
        self.data = mujoco.MjData(self.model)
        # timestep is set in the XML <option> block (0.005s); do not override here

        self._joint_qposadr = np.array([
            self.model.jnt_qposadr[mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, n)]
            for n in JOINT_NAMES], dtype=np.int32)
        self._joint_dofadr = np.array([
            self.model.jnt_dofadr[mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, n)]
            for n in JOINT_NAMES], dtype=np.int32)

        # Fix: treat [0,0] ctrlrange as unset (MuJoCo default when XML has none)
        ctrl_range = self.model.actuator_ctrlrange
        range_span = ctrl_range[:, 1] - ctrl_range[:, 0]
        if np.any(range_span > 1e-6):
            self._ctrl_high = ctrl_range[:, 1].copy()
        else:
            self._ctrl_high = np.full(self.model.nu, 3.0)  # 3 Nm — real servo range

        obs_dim = 1 + 4 + 3 + 3 + NUM_JOINTS + NUM_JOINTS  # 35
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float64)
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(self.model.nu,), dtype=np.float64)

        self._step_count = 0
        self._viewer = None
        self._last_x = 0.0

    def _get_obs(self) -> np.ndarray:
        return np.concatenate([
            [self.data.qpos[2]],          # height
            self.data.qpos[3:7],          # quaternion
            self.data.qvel[0:3],          # linear vel
            self.data.qvel[3:6],          # angular vel
            self.data.qpos[self._joint_qposadr],
            self.data.qvel[self._joint_dofadr],
        ]).astype(np.float64)

    def _get_reward(self, x_pos, x_vel, height, quat, action, joint_vel) -> float:
        roll, pitch, yaw = _quat_to_euler(quat)
        tilt_sq = roll**2 + pitch**2
        height_err = (height - DEFAULT_BASE_HEIGHT) ** 2  # target = standing tall

        is_upright = height >= self.upright_height_min and tilt_sq < self.upright_tilt_sq_max

        # Reward forward velocity + survival — only when upright
        vel_reward      = self.reward_vel * x_vel if is_upright else 0.0
        survival_reward = self.reward_survival if is_upright else 0.0
        fall_penalty    = self.reward_fall if not is_upright else 0.0
        # Penalize rapid joint oscillation (smooth motion preferred)
        joint_vel_penalty = self.reward_joint_vel * float(np.mean(joint_vel**2)) if is_upright else 0.0
        # Penalize sideways drift and spinning in place
        y_vel     = float(self.data.qvel[1])
        yaw_rate  = float(self.data.qvel[5])
        lateral_penalty  = self.reward_lateral * y_vel**2
        yaw_rate_penalty = self.reward_yaw_rate * yaw_rate**2

        # Roll penalty; pitch: nose-down and nose-up (falling backwards) both penalized, backward harder
        pitch_back = max(0.0, pitch)  # positive pitch = leaning back
        tilt_penalty = (
            self.reward_tilt * roll**2
            + self.reward_pitch * pitch**2
            + self.reward_pitch_back * pitch_back**2
        )

        return (
            vel_reward
            + survival_reward
            + fall_penalty
            + joint_vel_penalty
            + lateral_penalty
            + yaw_rate_penalty
            + tilt_penalty
            + self.reward_height * height_err
            + self.reward_action * np.sum(action**2)
        )

    def _is_terminated(self, height, quat) -> bool:
        if height < self.height_threshold:
            return True
        roll, pitch, _ = _quat_to_euler(quat)
        return abs(roll) > self.tilt_threshold or abs(pitch) > self.tilt_threshold

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self._step_count = 0

        # Start at proper standing height — NOT crouched
        self.data.qpos[:] = 0.0
        self.data.qpos[2] = DEFAULT_BASE_HEIGHT
        self.data.qpos[3:7] = [1.0, 0.0, 0.0, 0.0]
        self.data.qvel[:] = 0.0

        if self.reset_noise_scale > 0 and self.np_random is not None:
            self.data.qpos[2] += self.np_random.uniform(-self.reset_noise_scale, self.reset_noise_scale)
            self.data.qpos[self._joint_qposadr] += self.np_random.uniform(
                -self.reset_noise_scale, self.reset_noise_scale, size=NUM_JOINTS)
            yaw = self.np_random.uniform(-self.reset_noise_scale, self.reset_noise_scale)
            c, s = np.cos(yaw / 2), np.sin(yaw / 2)
            self.data.qpos[3:7] = [c, 0.0, 0.0, s]

        mujoco.mj_forward(self.model, self.data)
        self._last_x = float(self.data.qpos[0])
        return self._get_obs(), {"height": float(self.data.qpos[2])}

    def step(self, action):
        # Symmetric range: action [-1,1] * ctrl_high — no bias offset
        self.data.ctrl[:] = action * self._ctrl_high

        for _ in range(self.frame_skip):
            mujoco.mj_step(self.model, self.data)

        self._step_count += 1
        x_pos   = float(self.data.qpos[0])
        height  = float(self.data.qpos[2])
        quat    = self.data.qpos[3:7].copy()
        x_vel   = float(self.data.qvel[0])
        joint_vel = self.data.qvel[self._joint_dofadr].copy()

        reward     = self._get_reward(x_pos, x_vel, height, quat, action, joint_vel)
        self._last_x = x_pos
        terminated = self._is_terminated(height, quat)
        truncated  = self._step_count >= self.max_episode_steps
        obs        = self._get_obs()

        if self.render_mode == "human":
            self._render_frame()

        return obs, reward, terminated, truncated, {
            "height": height, "x_velocity": x_vel,
            "base_linvel": self.data.qvel[0:3].copy(),
        }

    def _render_frame(self) -> None:
        if self._viewer is None:
            from mujoco import viewer as mj_viewer
            self._viewer = mj_viewer.launch_passive(
                self.model, self.data, key_callback=None
            )
            # Hide collision geoms (group 1); show only visuals (group 2)
            self._viewer.opt.geomgroup[1] = 0
        self._viewer.sync()

    def render(self) -> np.ndarray | None:
        if self.render_mode == "human":
            self._render_frame()
            return None
        if self.render_mode == "rgb_array":
            import mujoco.renderer as mj_renderer
            if not hasattr(self, "_renderer"):
                self._renderer = mj_renderer.Renderer(self.model, height=480, width=640)
            self._renderer.update_scene(self.data)
            return self._renderer.render()
        return None

    def close(self) -> None:
        if self._viewer is not None:
            self._viewer.close()
            self._viewer = None
        if hasattr(self, "_renderer"):
            self._renderer.close()
            del self._renderer