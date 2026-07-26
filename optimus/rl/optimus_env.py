import numpy as np
import gymnasium as gym
from gymnasium import spaces
import mujoco
import os

URDF_PATH = os.path.join(os.path.dirname(__file__), "../urdf/full/optimus_mujoco_fixed.xml")

PARA_KP = 400.0
PARA_KD = 20.0
PARA_FMAX = 50.0  # ponytail: clamp prevents NaN from large velocity errors at episode start

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

# Observation: body orientation (quat 4) + body angvel (3) + body linvel (3)
#              + all joint pos (23 hinge) + all joint vel (23) = 56
OBS_DIM = 4 + 3 + 3 + 23 + 23  # = 56


class OptimusEnv(gym.Env):
    metadata = {"render_modes": ["human"]}

    def __init__(self, render_mode=None, ctrl_dt=0.02):
        super().__init__()

        self.model = mujoco.MjModel.from_xml_path(URDF_PATH)
        self.data  = mujoco.MjData(self.model)
        self.ctrl_dt = ctrl_dt  # policy runs at 50Hz, sim at 1kHz → 20 sim steps per action

        # Joint/actuator index maps
        self._jpos = {}
        self._jdof = {}
        for i in range(self.model.njnt):
            name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_JOINT, i)
            if name and name != "root":
                self._jpos[name] = self.model.jnt_qposadr[i]
                self._jdof[name] = self.model.jnt_dofadr[i]

        self._amap = {}
        for i in range(self.model.nu):
            name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
            if name:
                self._amap[name] = i

        self._root_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "root")

        # Sim steps per control step
        self._sim_steps = max(1, int(ctrl_dt / self.model.opt.timestep))

        # Action: torque targets for all 15 actuators, normalised to [-1, 1]
        # We scale by ctrlrange inside step()
        self.action_space = spaces.Box(-1.0, 1.0, shape=(self.model.nu,), dtype=np.float32)

        obs_high = np.full(OBS_DIM, np.inf, dtype=np.float32)
        self.observation_space = spaces.Box(-obs_high, obs_high, dtype=np.float32)

        self._render_mode = render_mode
        self._viewer = None

        # Saved spawn pose (set once in reset)
        self._spawn_qpos = None
        self._warmup_done = False

    # ------------------------------------------------------------------
    def _warmup(self):
        """Pre-torque joints for 1000 steps with frozen root, then record spawn pose."""
        mujoco.mj_resetData(self.model, self.data)
        saved = self.data.qpos[:7].copy()
        for _ in range(1000):
            self._apply_parallelogram()
            mujoco.mj_step(self.model, self.data)
            self.data.qpos[:7] = saved
            self.data.qvel[:] = 0.0
        self._spawn_qpos = self.data.qpos.copy()
        self._spawn_qvel = self.data.qvel.copy()

    def _apply_parallelogram(self):
        for driven, follower, sign in PARA_PAIRS:
            q_err = sign * self.data.qpos[self._jpos[driven]] - self.data.qpos[self._jpos[follower]]
            v_err = sign * self.data.qvel[self._jdof[driven]] - self.data.qvel[self._jdof[follower]]
            torque = PARA_KP * q_err + PARA_KD * v_err
            self.data.qfrc_applied[self._jdof[follower]] = np.clip(torque, -PARA_FMAX, PARA_FMAX)

    def _get_obs(self):
        d = self.data
        # Root orientation quaternion (w, x, y, z)
        quat = d.qpos[3:7].copy()
        # Root angular velocity (world frame)
        angvel = d.qvel[3:6].copy()
        # Root linear velocity (world frame)
        linvel = d.qvel[0:3].copy()
        # All hinge joint positions (qpos[7:30], 23 values)
        jpos = d.qpos[7:].copy()      # 23
        # All hinge joint velocities (qvel[6:29], 23 values)
        jvel = d.qvel[6:].copy()      # 23
        return np.concatenate([quat, angvel, linvel, jpos, jvel]).astype(np.float32)

    def _is_fallen(self):
        mat = self.data.xmat[self._root_id].reshape(3, 3)
        # z-column of rotation matrix = world-up in body frame
        # if body z-axis (up) has negative world-z component, robot is upside-down
        # allow up to ~60° tilt before calling it fallen
        return mat[2, 2] < 0.5

    # ------------------------------------------------------------------
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if self._spawn_qpos is None:
            self._warmup()

        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[:] = self._spawn_qpos
        self.data.qvel[:] = self._spawn_qvel

        # Small random perturbation to improve robustness
        noise = self.np_random.uniform(-0.02, 0.02, size=self.model.nq - 7)
        self.data.qpos[7:] += noise

        mujoco.mj_forward(self.model, self.data)

        self._steps = 0
        self._prev_x = self.data.qpos[0]  # track forward progress (X axis)
        self._prev_y = self.data.qpos[1]

        return self._get_obs(), {}

    def step(self, action):
        # Scale action [-1,1] → actual torque range per actuator
        for i in range(self.model.nu):
            lo, hi = self.model.actuator_ctrlrange[i]
            self.data.ctrl[i] = float(action[i]) * (hi - lo) / 2.0

        # Run sim_steps at 1kHz
        for _ in range(self._sim_steps):
            self._apply_parallelogram()
            mujoco.mj_step(self.model, self.data)

        obs = self._get_obs()
        fallen = self._is_fallen() or bool(np.any(~np.isfinite(obs)))

        # ---- Reward ----
        root_z   = self.data.qpos[2]          # height
        linvel_y = self.data.qvel[1]           # forward velocity (+Y = forward)
        angvel   = self.data.qvel[3:6]

        # Height reward: want z ≈ 0.25 (standing height)
        r_height = np.exp(-10.0 * (root_z - 0.251) ** 2)

        # Forward velocity reward
        r_forward = np.clip(linvel_y, -1.0, 2.0)

        # Stability: penalise angular velocity
        r_stable = -0.1 * float(np.sum(angvel ** 2))

        # Action smoothness penalty
        r_action = -0.005 * float(np.sum(action ** 2))

        reward = r_height + 0.5 * r_forward + r_stable + r_action

        if fallen:
            reward -= 5.0
            obs = np.nan_to_num(obs, nan=0.0, posinf=0.0, neginf=0.0)

        self._steps += 1
        terminated = fallen
        truncated  = self._steps >= 2000   # 40s episode max

        if self._render_mode == "human":
            self.render()

        return obs, float(reward), terminated, truncated, {}

    def render(self):
        if self._viewer is None:
            import mujoco.viewer as mjv
            self._viewer = mjv.launch_passive(self.model, self.data)
        self._viewer.sync()

    def close(self):
        if self._viewer is not None:
            self._viewer.close()
            self._viewer = None
