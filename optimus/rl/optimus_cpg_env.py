"""
Optimus CPG + Residual RL environment — hardware-faithful.

Hardware constraints applied:
  - Control loop: 50 Hz (PCA9685 servo PWM cadence)
  - IMU: MPU6050 complementary filter, α=0.98, pitch/roll/yaw_rate
  - Joint limits: from firmware config.h (knee ±70°, hip ±45°, ankle ±90°)
  - Servo torque: MG995 stall ~0.92 N·m (legs), Futaba S3003 ~0.31 N·m (arms)
  - Geometry: L1=L2=90 mm, L3=30 mm

Observation (hardware-available signals only):
  - CPG phase sin/cos (2)
  - IMU: pitch, roll, yaw_rate (3)  ← matches MPU6050 output
  - Joint positions (23)
  - Joint velocities (23)
  Total: 51 dims

Action: residual corrections on top of CPG, ±MAX_RESIDUAL rad, 15 joints.
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces
import mujoco
import os

from cpg import CPG, LIM_KNEE, LIM_HIP_ROLL, LIM_HIP_PITCH, LIM_ANKLE, LIM_SHOULDER, LIM_FOREARM

URDF_PATH = os.path.join(os.path.dirname(__file__), "../urdf/full/optimus_mujoco_fixed.xml")

# Parallelogram constraint (passive follower joints track driven joints)
PARA_KP   = 400.0
PARA_KD   = 20.0
PARA_FMAX = 5.0
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

# Hardware: PCA9685 at 50 Hz → 20 ms control period
CTRL_HZ      = 50
CTRL_DT      = 1.0 / CTRL_HZ

# MG995 stall torque 9.4 kg·cm = 0.92 N·m; use 90% to avoid rail saturation
KP           = 10.0   # stronger PD — knees now have 1.96 N·m, need higher gain to use it
KD           = 0.5
MAX_RESIDUAL = 0.20   # rad — RL needs room to discover hip coordination for foot clearance

ACTUATOR_NAMES = [
    "Servo-Hip-Body-Rotation",
    "Servo-Showlder-L-Front-Back", "Servo-Showlder-L-Inward-Outward", "Servo-Forearm-L",
    "Servo-Showlder-R-Front-Back", "Servo-Showlder-R-Inward-Outward", "Servo-Forearm-R",
    "Servo-Hip-L", "Servo-Knee-L-Top", "Servo-Knee-L-Bottom", "Servo-Ankle-L",
    "Servo-Hip-R", "Servo-Knee-R-Top", "Servo-Knee-R-Bottom", "Servo-Ankle-R",
]

# obs: 2 (CPG phase) + 3 (IMU: pitch, roll, yaw_rate) + 23 (jpos) + 23 (jvel) = 51
OBS_DIM = 2 + 3 + 23 + 23


class OptimusCPGEnv(gym.Env):
    """
    Optimus walking environment.
    Worker ID optionally passed for per-worker log files.
    """
    metadata = {"render_modes": ["human"]}

    def __init__(self, render_mode=None, worker_id: int = 0):
        super().__init__()
        self.model = mujoco.MjModel.from_xml_path(URDF_PATH)
        self.data  = mujoco.MjData(self.model)
        self.ctrl_dt = CTRL_DT
        self._sim_steps = max(1, int(CTRL_DT / self.model.opt.timestep))
        self._worker_id = worker_id

        # Build index maps
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
        self._spawn_qvel  = None
        self._cpg         = CPG()

        # IMU complementary filter state (mirrors firmware MPU6050)
        self._imu_pitch    = 0.0
        self._imu_roll     = 0.0
        self._imu_yaw_rate = 0.0
        self._imu_alpha    = 0.98   # firmware COMPLEMENTARY_ALPHA

        # Per-episode diagnostics
        self._ep_fwd_dist  = 0.0
        self._ep_lat_drift = 0.0
        self._ep_falls     = 0
        self._ep_steps     = 0

    # ── Sim helpers ──────────────────────────────────────────────────────────

    def _apply_parallelogram(self):
        for driven, follower, sign in PARA_PAIRS:
            if driven not in self._jpos or follower not in self._jpos:
                continue
            q_err = sign * self.data.qpos[self._jpos[driven]] - self.data.qpos[self._jpos[follower]]
            v_err = sign * self.data.qvel[self._jdof[driven]] - self.data.qvel[self._jdof[follower]]
            self.data.qfrc_applied[self._jdof[follower]] = np.clip(
                PARA_KP * q_err + PARA_KD * v_err, -PARA_FMAX, PARA_FMAX
            )

    def _warmup(self):
        mujoco.mj_resetData(self.model, self.data)
        # Spawn at t=0 where hip CPG base is 0 — no torque spike on first step.
        cpg_t0 = CPG()
        t0_targets = cpg_t0._angles(0.0)
        self._cpg_t0_offset = 0.0
        for name, angle in t0_targets.items():
            if name in self._jpos:
                self.data.qpos[self._jpos[name]] = angle

        saved = self.data.qpos[:7].copy()
        roll_samples = []
        for _ in range(1000):
            self._apply_parallelogram()
            mujoco.mj_step(self.model, self.data)
            self.data.qpos[:7] = saved
            self.data.qvel[:]  = 0.0
            # Collect roll samples for calibration (last 500 steps = settled motion)
            if _ >= 500:
                q = self.data.qpos[3:7]
                roll_accel = float(np.arctan2(2*(q[0]*q[3] + q[1]*q[2]),
                                              1 - 2*(q[2]**2 + q[3]**2)))
                roll_samples.append(roll_accel)

        self._cpg.calibrate(roll_samples)
        self._spawn_qpos = self.data.qpos.copy()
        self._spawn_qvel = self.data.qvel.copy()

    def _update_imu(self):
        """
        Complementary filter matching firmware MPU6050::update().
        Uses MuJoCo root body quaternion and angular velocity as ground truth.
        """
        q = self.data.qpos[3:7]   # w, x, y, z
        # Pitch: rotation around X (forward lean)
        pitch_accel = float(np.arctan2(2*(q[0]*q[1] + q[2]*q[3]),
                                       1 - 2*(q[1]**2 + q[2]**2)))
        # Roll: rotation around Z (lateral lean)  — firmware uses atan2(ax, ay)
        roll_accel  = float(np.arctan2(2*(q[0]*q[3] + q[1]*q[2]),
                                       1 - 2*(q[2]**2 + q[3]**2)))

        gx = float(self.data.qvel[3])   # angular velocity around X
        gz = float(self.data.qvel[5])   # angular velocity around Z
        gy = float(self.data.qvel[4])   # yaw rate around Y

        dt = self.ctrl_dt
        self._imu_pitch    = self._imu_alpha * (self._imu_pitch + gx * dt) + (1 - self._imu_alpha) * pitch_accel
        self._imu_roll     = self._imu_alpha * (self._imu_roll  + gz * dt) + (1 - self._imu_alpha) * roll_accel
        self._imu_yaw_rate = gy

    def _set_ctrl(self, cpg_targets: dict, residual: np.ndarray):
        for i, name in enumerate(ACTUATOR_NAMES):
            if name not in self._amap or name not in self._jpos:
                continue
            target = cpg_targets.get(name, 0.0) + MAX_RESIDUAL * float(residual[i])
            target = float(np.clip(target,
                           self.model.jnt_range[self._jpos[name] - 7, 0],
                           self.model.jnt_range[self._jpos[name] - 7, 1]))
            q      = self.data.qpos[self._jpos[name]]
            dq     = self.data.qvel[self._jdof[name]]
            torque = KP * (target - q) - KD * dq
            lo, hi = self.model.actuator_ctrlrange[self._amap[name]]
            self.data.ctrl[self._amap[name]] = np.clip(torque, lo, hi)

    def _get_obs(self) -> np.ndarray:
        cpg_phase = np.array([np.sin(self._cpg._t * self._cpg._omega),
                               np.cos(self._cpg._t * self._cpg._omega)], dtype=np.float32)
        imu   = np.array([self._imu_pitch, self._imu_roll, self._imu_yaw_rate], dtype=np.float32)
        jpos  = self.data.qpos[7:].astype(np.float32)
        jvel  = self.data.qvel[6:].astype(np.float32)
        return np.concatenate([cpg_phase, imu, jpos, jvel])

    def _is_fallen(self) -> bool:
        if self.data.qpos[2] < 0.10:
            return True
        mat = self.data.xmat[self._root_id].reshape(3, 3)
        return mat[2, 2] < 0.5   # torso tilt > ~60°

    def diagnostics(self) -> dict:
        """Return current episode diagnostics for live logging."""
        return {
            "worker":     self._worker_id,
            "steps":      self._ep_steps,
            "fwd_dist_m": round(self._ep_fwd_dist, 3),
            "lat_drift_m": round(abs(self._ep_lat_drift), 3),
            "imu_pitch":  round(np.degrees(self._imu_pitch), 1),
            "imu_roll":   round(np.degrees(self._imu_roll), 1),
            "yaw_rate":   round(np.degrees(self._imu_yaw_rate), 1),
            "height_m":   round(float(self.data.qpos[2]), 3),
        }

    # ── Gym API ──────────────────────────────────────────────────────────────

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if self._spawn_qpos is None:
            self._warmup()

        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[:] = self._spawn_qpos
        self.data.qvel[:] = self._spawn_qvel
        noise = self.np_random.uniform(-0.01, 0.01, size=self.model.nq - 7)
        self.data.qpos[7:] += noise
        mujoco.mj_forward(self.model, self.data)

        # Start CPG at the same phase used during warmup (hip=0 crossing).
        self._cpg.reset(getattr(self, '_cpg_t0_offset', 0.0))

        # Reset IMU filter
        self._imu_pitch = self._imu_roll = self._imu_yaw_rate = 0.0

        # Reset diagnostics
        self._ep_fwd_dist = self._ep_lat_drift = 0.0
        self._ep_steps = 0
        self._prev_y = float(self.data.qpos[1])
        self._start_x = float(self.data.qpos[0])

        return self._get_obs(), {}

    def step(self, action: np.ndarray):
        # Update IMU filter (50 Hz matches CPG task; firmware IMU runs at 100Hz
        # but CPG reads it at 100Hz — here we fuse at control cadence)
        self._update_imu()

        # CPG step — pass IMU roll so phase reset can modulate oscillator speed
        cpg_targets = self._cpg.step(self.ctrl_dt, roll=self._imu_roll)

        # IMU stabilisation (mirrors firmware Stabilizer::compute)
        cpg_targets = self._cpg.stabilise(
            cpg_targets, self._imu_pitch, self._imu_roll, self._imu_yaw_rate
        )

        # Apply CPG + RL residual
        self._set_ctrl(cpg_targets, action)

        for _ in range(self._sim_steps):
            self._apply_parallelogram()
            mujoco.mj_step(self.model, self.data)

        obs    = self._get_obs()
        fallen = self._is_fallen() or bool(np.any(~np.isfinite(obs)))

        # ── Reward ───────────────────────────────────────────────────────────
        cur_y  = float(self.data.qpos[1])
        cur_x  = float(self.data.qpos[0])
        root_z = float(self.data.qpos[2])

        fwd_vel    = (cur_y - self._prev_y) / self.ctrl_dt   # m/s forward
        self._prev_y = cur_y

        # Update episode diagnostics
        self._ep_fwd_dist  += max(0.0, cur_y - self._prev_y + fwd_vel * self.ctrl_dt)
        self._ep_lat_drift  = cur_x - self._start_x
        self._ep_steps     += 1

        r_height   = np.clip(root_z / 0.251, 0.0, 1.0)
        r_survive  = 0.1 * r_height
        # Forward velocity reward — penalise standing still (marching in place scores 0, costs -0.3)
        r_forward  = 4.0 * np.clip(fwd_vel, 0.0, 3.0) - 0.3
        r_stable   = -0.05  * float(np.sum(self.data.qvel[3:6] ** 2))
        r_straight = -2.0   * float(self.data.qvel[0] ** 2)   # penalise X velocity
        r_yaw      = -1.0   * float(self.data.qvel[5] ** 2)   # penalise Z spin
        r_action   = -0.005 * float(np.sum(action ** 2))

        reward = r_height + r_survive + r_forward + r_stable + r_straight + r_yaw + r_action

        if fallen:
            reward -= 1.0
            obs = np.nan_to_num(obs, nan=0.0, posinf=0.0, neginf=0.0)

        self._ep_steps += 1
        terminated = fallen
        truncated  = self._ep_steps >= 2000

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
