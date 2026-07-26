"""
Optimus CPG + Residual RL environment.

The policy outputs residual corrections on top of the CPG gait.
Action space: 15-dim corrections in [-1, 1], scaled to ±MAX_RESIDUAL rad.
Observation: CPG phase (sin/cos) + IMU (quat, angvel, linvel) + joint state.

This is strictly better than pure RL from scratch because:
  - CPG provides a stable rhythmic prior → policy starts near a walking gait
  - RL only needs to learn small corrections → converges in <5M steps
  - IMU pitch is explicitly in obs → stabilisation is learnable
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces
import mujoco
import os

from cpg import CPG

URDF_PATH  = os.path.join(os.path.dirname(__file__), "../urdf/full/optimus_mujoco_fixed.xml")
PARA_KP    = 400.0
PARA_KD    = 20.0
PARA_FMAX  = 5.0
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

MAX_RESIDUAL = 0.15   # rad — max correction the RL policy can add to CPG
KP           = 8.0    # position PD gains for CPG target tracking
KD           = 0.4

# Actuator name order (matches model.nu ordering)
ACTUATOR_NAMES = [
    "Servo-Hip-Body-Rotation",
    "Servo-Showlder-L-Front-Back", "Servo-Showlder-L-Inward-Outward", "Servo-Forearm-L",
    "Servo-Showlder-R-Front-Back", "Servo-Showlder-R-Inward-Outward", "Servo-Forearm-R",
    "Servo-Hip-L", "Servo-Knee-L-Top", "Servo-Knee-L-Bottom", "Servo-Ankle-L",
    "Servo-Hip-R", "Servo-Knee-R-Top", "Servo-Knee-R-Bottom", "Servo-Ankle-R",
]

# obs: sin+cos of CPG phase (2) + quat (4) + angvel (3) + linvel (3) + jpos (23) + jvel (23) = 58
OBS_DIM = 2 + 4 + 3 + 3 + 23 + 23


class OptimusCPGEnv(gym.Env):
    metadata = {"render_modes": ["human"]}

    def __init__(self, render_mode=None, ctrl_dt=0.02):
        super().__init__()
        self.model = mujoco.MjModel.from_xml_path(URDF_PATH)
        self.data  = mujoco.MjData(self.model)
        self.ctrl_dt = ctrl_dt
        self._sim_steps = max(1, int(ctrl_dt / self.model.opt.timestep))

        # Index maps
        self._jpos, self._jdof, self._amap = {}, {}, {}
        for i in range(self.model.njnt):
            name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_JOINT, i)
            if name and name != "root":
                self._jpos[name] = self.model.jnt_qposadr[i]
                self._jdof[name] = self.model.jnt_dofadr[i]
        for i in range(self.model.nu):
            name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
            if name:
                self._amap[name] = i

        self._root_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "root")

        self.action_space      = spaces.Box(-1.0, 1.0, shape=(self.model.nu,), dtype=np.float32)
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(OBS_DIM,), dtype=np.float32)

        self._render_mode = render_mode
        self._viewer      = None
        self._spawn_qpos  = None
        self._cpg         = CPG()

    # ── Sim helpers ──────────────────────────────────────────────────────────

    def _apply_parallelogram(self):
        for driven, follower, sign in PARA_PAIRS:
            q_err  = sign * self.data.qpos[self._jpos[driven]] - self.data.qpos[self._jpos[follower]]
            v_err  = sign * self.data.qvel[self._jdof[driven]] - self.data.qvel[self._jdof[follower]]
            self.data.qfrc_applied[self._jdof[follower]] = np.clip(
                PARA_KP * q_err + PARA_KD * v_err, -PARA_FMAX, PARA_FMAX
            )

    def _warmup(self):
        mujoco.mj_resetData(self.model, self.data)
        saved = self.data.qpos[:7].copy()
        for _ in range(1000):
            self._apply_parallelogram()
            mujoco.mj_step(self.model, self.data)
            self.data.qpos[:7] = saved
            self.data.qvel[:]  = 0.0
        self._spawn_qpos = self.data.qpos.copy()
        self._spawn_qvel = self.data.qvel.copy()

    def _set_ctrl_from_targets(self, cpg_targets: dict, residual: np.ndarray):
        """
        Convert CPG angle targets + residual corrections into motor torques
        using a simple PD controller tracking the desired joint angle.
        """
        for i, name in enumerate(ACTUATOR_NAMES):
            if name not in self._amap or name not in self._jpos:
                continue
            target = cpg_targets.get(name, 0.0) + MAX_RESIDUAL * float(residual[i])
            target = np.clip(target, self.model.jnt_range[self._jpos[name] - 7, 0],
                                     self.model.jnt_range[self._jpos[name] - 7, 1])
            q  = self.data.qpos[self._jpos[name]]
            dq = self.data.qvel[self._jdof[name]]
            torque = KP * (target - q) - KD * dq
            lo, hi = self.model.actuator_ctrlrange[self._amap[name]]
            self.data.ctrl[self._amap[name]] = np.clip(torque, lo, hi)

    def _get_obs(self) -> np.ndarray:
        d = self.data
        cpg_phase = np.array([np.sin(self._cpg.phase_L), np.cos(self._cpg.phase_L)], dtype=np.float32)
        quat   = d.qpos[3:7].astype(np.float32)
        angvel = d.qvel[3:6].astype(np.float32)
        linvel = d.qvel[0:3].astype(np.float32)
        jpos   = d.qpos[7:].astype(np.float32)
        jvel   = d.qvel[6:].astype(np.float32)
        return np.concatenate([cpg_phase, quat, angvel, linvel, jpos, jvel])

    def _imu_pitch(self) -> float:
        """Approximate pitch from quaternion (rotation around X axis)."""
        q = self.data.qpos[3:7]  # w, x, y, z
        return float(2.0 * (q[0] * q[2] - q[3] * q[1]))

    def _is_fallen(self) -> bool:
        if self.data.qpos[2] < 0.10:
            return True
        mat = self.data.xmat[self._root_id].reshape(3, 3)
        return mat[2, 2] < 0.5

    # ── Gym API ──────────────────────────────────────────────────────────────

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if self._spawn_qpos is None:
            self._warmup()

        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[:] = self._spawn_qpos
        self.data.qvel[:] = self._spawn_qvel
        noise = self.np_random.uniform(-0.02, 0.02, size=self.model.nq - 7)
        self.data.qpos[7:] += noise
        mujoco.mj_forward(self.model, self.data)

        # Randomise CPG phase so policy learns all phases
        self._cpg.phase_L = self.np_random.uniform(0, 2 * np.pi)
        self._cpg.phase_R = (self._cpg.phase_L + np.pi) % (2 * np.pi)

        self._steps  = 0
        self._prev_y = self.data.qpos[1]
        return self._get_obs(), {}

    def step(self, action: np.ndarray):
        # Advance CPG
        cpg_targets = self._cpg.step(self.ctrl_dt)

        # IMU stabilisation (reactive, not learned — always on)
        pitch = self._imu_pitch()
        cpg_targets = self._cpg.stabilise(cpg_targets, pitch)

        # Apply CPG + RL residual as torques
        self._set_ctrl_from_targets(cpg_targets, action)

        for _ in range(self._sim_steps):
            self._apply_parallelogram()
            mujoco.mj_step(self.model, self.data)

        obs    = self._get_obs()
        fallen = self._is_fallen() or bool(np.any(~np.isfinite(obs)))

        # ── Reward ───────────────────────────────────────────────────────────
        root_z = self.data.qpos[2]
        cur_y  = self.data.qpos[1]

        # Forward progress this step (m/step)
        r_forward  = (cur_y - self._prev_y) / self.ctrl_dt   # velocity m/s
        self._prev_y = cur_y

        # Height — linear, always provides gradient
        r_height   = np.clip(root_z / 0.251, 0.0, 1.0)

        # Survival scaled by height — reduced so forward motion dominates
        r_survive  = 0.1 * r_height

        # Stability — penalise angular velocity
        r_stable   = -0.05 * float(np.sum(self.data.qvel[3:6] ** 2))

        # Small action penalty — keep residuals small (trust the CPG)
        r_action   = -0.002 * float(np.sum(action ** 2))

        reward = r_height + r_survive + 3.0 * np.clip(r_forward, -0.5, 3.0) + r_stable + r_action

        if fallen:
            reward -= 1.0
            obs = np.nan_to_num(obs, nan=0.0, posinf=0.0, neginf=0.0)

        self._steps += 1
        terminated = fallen
        truncated  = self._steps >= 2000

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
