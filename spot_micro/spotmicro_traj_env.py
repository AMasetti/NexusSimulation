"""
Trajectory-tracking Gymnasium environment for SpotMicro.

Instead of learning motion from scratch, the policy tracks a reference gait cycle
(keyframes) using a DeepMimic-style tracking reward. Given N keyframe poses the
robot must follow them in sequence, cycling continuously.

The observation is augmented with the current target pose and the gait phase so
the policy knows where it is in the cycle.

Available keyframe sets:
    TROT_KEYFRAMES      — 8-frame diagonal trot gait
    WALK_5_KEYFRAMES    — 5-frame human-designed walk cycle (easier to understand/edit)

Usage:
    from spotmicro_traj_env import SpotMicroTrajEnv, WALK_5_KEYFRAMES, TROT_KEYFRAMES
    env = SpotMicroTrajEnv()                                  # default: WALK_5_KEYFRAMES
    env = SpotMicroTrajEnv(keyframes=TROT_KEYFRAMES)          # 8-frame trot
    env = SpotMicroTrajEnv(keyframes=my_keyframes)            # custom (N×12 array)
"""

import numpy as np
import mujoco
import gymnasium as gym
from gymnasium import spaces

from spotmicro_loader import load_spotmicro_xml

JOINT_NAMES = [
    "front_left_shoulder",  "front_left_leg",  "front_left_foot",
    "front_right_shoulder", "front_right_leg", "front_right_foot",
    "rear_left_shoulder",   "rear_left_leg",   "rear_left_foot",
    "rear_right_shoulder",  "rear_right_leg",  "rear_right_foot",
]

NUM_JOINTS        = 12
FREE_JOINT_QPOS_SIZE = 7

# Body height for the natural flexed standing pose (hip=0.65, knee=-1.40).
# Computed from XML geometry:
#   upper_leg_z=0.109m  lower_leg_z=0.130m  toe_radius=0.020m
#   h = 0.109*cos(0.65) + 0.130*cos(0.65-1.40) + 0.020 ≈ 0.202m
# This is ~78% of the fully-extended height (0.259m) — a natural crouched stance.
DEFAULT_BASE_HEIGHT = 0.202

# Natural standing joint pose — all four legs identical.
# Order per leg: [shoulder, hip, knee]; leg order: FL, FR, RL, RR
STAND_JOINTS = np.array([
    0.0,  0.65, -1.40,   # FL
    0.0,  0.65, -1.40,   # FR
    0.0,  0.65, -1.40,   # RL
    0.0,  0.65, -1.40,   # RR
], dtype=np.float64)

# ---------------------------------------------------------------------------
# Reference trot gait — 8 keyframes for one full cycle.
#
# Joint order per leg: [shoulder, hip, foot]
# Legs order: FL, FR, RL, RR
#
# At ref=0 all joints the robot stands at DEFAULT_BASE_HEIGHT.
# Trot: diagonal pairs FL+RR and FR+RL alternate swing/stance.
#
# shoulder: abduction axis (keep ~0 for straight walking)
# hip:      pitch forward (+) / back (-)    range [-2.67, 1.55]
# foot:     pitch, 0=straight, negative=bent range [-2.59, 0.10]
#
# These are starting-point values — RL will learn to execute them robustly.
# ---------------------------------------------------------------------------
_SH = 0.0    # shoulder neutral

# Stance phase joint angles
_HIP_STANCE  = -0.40   # hip slightly back (supporting)
_KNEE_STANCE = -1.00   # knee bent for ground contact

# Swing phase joint angles
_HIP_SWING   =  0.35   # hip forward (clearing ground)
_KNEE_SWING  = -0.55   # knee lifted

# Push-off (end of stance)
_HIP_PUSH    = -0.65
_KNEE_PUSH   = -1.05

def _leg(sh, hip, knee):
    """Helper: returns [shoulder, hip, foot] for one leg."""
    return [sh, hip, knee]

# ---------------------------------------------------------------------------
# WALK_5_KEYFRAMES — 5 hand-designed poses for one walking half-cycle.
#
# Each pose is a full snapshot of all 12 joints.  Edit these values to adjust
# the gait; the RL policy learns to reach each pose in sequence, cycling back
# to pose 0 after pose 4.
#
# Joint anatomy reminder:
#   shoulder : abduction left/right  (keep 0 for straight walking)
#   hip      : swing forward (+) / push back (-)    range [-2.67,  1.55] rad
#   foot     : knee straight (0) / bent (negative)  range [-2.59,  0.10] rad
#
# Leg column order: FL, FR, RL, RR
# ---------------------------------------------------------------------------

# Shared angle constants — tune these to change gait character
_SH_N   =  0.00   # shoulder: neutral (no abduction)

# Hip angles are relative to straight-down (ref=0).
# Natural stand = 0.65 rad (leg angled slightly forward, foot under CoM).
_HIP_N  =  0.65   # hip: natural standing (foot directly under shoulder)
_HIP_F  =  1.00   # hip: forward swing (reaching ahead, foot ~27mm off ground with _KNEE_U)
_HIP_B  =  0.35   # hip: trailing after push-off

# Knee angles (negative = bent; 0 = fully extended / straight).
# With hip=0.65 and knee=-1.40 the body sits at DEFAULT_BASE_HEIGHT=0.202m.
_KNEE_N = -1.40   # knee: natural standing flex
_KNEE_S = -1.40   # knee: stance (same as natural — stable under load)
_KNEE_U = -0.50   # knee: raised during swing (~27mm foot clearance at hip=1.0)

WALK_5_KEYFRAMES = np.array([
    # ── Pose 0 : STANDING ──────────────────────────────────────────────────
    # All legs at natural flex (hip=0.65, knee=-1.40).
    # Body at DEFAULT_BASE_HEIGHT=0.202m — matches the physics reset pose.
    #
    #           FL                      FR                      RL                      RR
    _leg(_SH_N, _HIP_N, _KNEE_N) + _leg(_SH_N, _HIP_N, _KNEE_N) + _leg(_SH_N, _HIP_N, _KNEE_N) + _leg(_SH_N, _HIP_N, _KNEE_N),

    # ── Pose 1 : FL SWING ──────────────────────────────────────────────────
    # Front-left lifts and reaches forward (hip=1.0, knee=-0.50 → foot ~27mm
    # off ground).  The other three legs hold neutral stance.
    #
    #           FL (swing fwd+up)        FR (stance)             RL (stance)             RR (stance)
    _leg(_SH_N, _HIP_F, _KNEE_U) + _leg(_SH_N, _HIP_N, _KNEE_S) + _leg(_SH_N, _HIP_N, _KNEE_S) + _leg(_SH_N, _HIP_N, _KNEE_S),

    # ── Pose 2 : FL LANDS, RR SWING ────────────────────────────────────────
    # FL touches down at hip=0.50 (just ahead of neutral, starting to push
    # back).  Diagonal partner RR lifts and reaches forward.
    #
    #           FL (just landed)         FR (stance)             RL (stance)             RR (swing fwd+up)
    _leg(_SH_N, 0.50,   _KNEE_S) + _leg(_SH_N, _HIP_N, _KNEE_S) + _leg(_SH_N, _HIP_N, _KNEE_S) + _leg(_SH_N, _HIP_F, _KNEE_U),

    # ── Pose 3 : RR LANDS, FR+RL SWING ────────────────────────────────────
    # RR touches down (hip=0.50).  FL is now trailing (hip=0.35, push-off).
    # FR and RL both swing forward — completes one full diagonal exchange.
    #
    #           FL (trailing/push-off)   FR (swing fwd+up)       RL (swing fwd+up)       RR (just landed)
    _leg(_SH_N, _HIP_B, _KNEE_S) + _leg(_SH_N, _HIP_F, _KNEE_U) + _leg(_SH_N, _HIP_F, _KNEE_U) + _leg(_SH_N, 0.50,   _KNEE_S),

    # ── Pose 4 : STANDING (return) ─────────────────────────────────────────
    # All legs return to natural stance — clean loop point.
    #
    #           FL                      FR                      RL                      RR
    _leg(_SH_N, _HIP_N, _KNEE_N) + _leg(_SH_N, _HIP_N, _KNEE_N) + _leg(_SH_N, _HIP_N, _KNEE_N) + _leg(_SH_N, _HIP_N, _KNEE_N),
], dtype=np.float64)


# ---------------------------------------------------------------------------
# TROT_KEYFRAMES — 8-frame diagonal trot gait (more dynamic, harder to learn)
# ---------------------------------------------------------------------------

# Columns: FL_sh FL_hip FL_knee  FR_sh FR_hip FR_knee  RL_sh RL_hip RL_knee  RR_sh RR_hip RR_knee
TROT_KEYFRAMES = np.array([
    # Phase 0: FL+RR lift off — FR+RL push off
    _leg(_SH, 0.0, -0.75)     + _leg(_SH, _HIP_PUSH, _KNEE_PUSH)  + _leg(_SH, _HIP_PUSH, _KNEE_PUSH)  + _leg(_SH, 0.0, -0.75),
    # Phase 1: FL+RR swing peak — FR+RL stance
    _leg(_SH, _HIP_SWING, _KNEE_SWING) + _leg(_SH, _HIP_STANCE, _KNEE_STANCE) + _leg(_SH, _HIP_STANCE, _KNEE_STANCE) + _leg(_SH, _HIP_SWING, _KNEE_SWING),
    # Phase 2: FL+RR touch down — FR+RL stance
    _leg(_SH, 0.2, -0.85)     + _leg(_SH, _HIP_STANCE, _KNEE_STANCE) + _leg(_SH, _HIP_STANCE, _KNEE_STANCE) + _leg(_SH, 0.2, -0.85),
    # Phase 3: FL+RR stance mid — FR+RL lift off
    _leg(_SH, _HIP_STANCE, _KNEE_STANCE) + _leg(_SH, 0.0, -0.75)  + _leg(_SH, 0.0, -0.75)  + _leg(_SH, _HIP_STANCE, _KNEE_STANCE),
    # Phase 4: FR+RL swing peak — FL+RR stance
    _leg(_SH, _HIP_STANCE, _KNEE_STANCE) + _leg(_SH, _HIP_SWING, _KNEE_SWING) + _leg(_SH, _HIP_SWING, _KNEE_SWING) + _leg(_SH, _HIP_STANCE, _KNEE_STANCE),
    # Phase 5: FR+RL touch down — FL+RR stance
    _leg(_SH, _HIP_STANCE, _KNEE_STANCE) + _leg(_SH, 0.2, -0.85)  + _leg(_SH, 0.2, -0.85)  + _leg(_SH, _HIP_STANCE, _KNEE_STANCE),
    # Phase 6: FL+RR push off — FR+RL stance mid
    _leg(_SH, _HIP_PUSH, _KNEE_PUSH)   + _leg(_SH, _HIP_STANCE, _KNEE_STANCE) + _leg(_SH, _HIP_STANCE, _KNEE_STANCE) + _leg(_SH, _HIP_PUSH, _KNEE_PUSH),
    # Phase 7: transition — all near stance, ready to repeat
    _leg(_SH, -0.20, -0.90)   + _leg(_SH, -0.20, -0.90) + _leg(_SH, -0.20, -0.90) + _leg(_SH, -0.20, -0.90),
], dtype=np.float64)


def _quat_to_euler(quat: np.ndarray) -> np.ndarray:
    w, x, y, z = quat
    sinr_cosp = 2 * (w * x + y * z)
    cosr_cosp = 1 - 2 * (x * x + y * y)
    roll = np.arctan2(sinr_cosp, cosr_cosp)
    sinp = np.clip(2 * (w * y - z * x), -1.0, 1.0)
    pitch = np.arcsin(sinp)
    siny_cosp = 2 * (w * z + x * y)
    cosy_cosp = 1 - 2 * (y * y + z * z)
    yaw = np.arctan2(siny_cosp, cosy_cosp)
    return np.array([roll, pitch, yaw], dtype=np.float64)


class SpotMicroTrajEnv(gym.Env):
    """
    SpotMicro environment with reference trajectory tracking.

    The gait cycle is defined by `keyframes` (shape N×12). Each keyframe is
    held for `steps_per_frame` env steps; the cycle repeats continuously.
    The policy is rewarded for matching the current target pose (tracking reward)
    plus forward velocity and staying upright.

    Observation (48-dim):
        height (1) + quaternion (4) + linear vel (3) + angular vel (3)
        + joint positions (12) + joint velocities (12)
        + target joint positions (12) + gait phase (1)
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 50}

    def __init__(
        self,
        keyframes: np.ndarray | None = None,   # (N, 12); uses WALK_5_KEYFRAMES if None
        steps_per_frame: int = 20,              # env steps to hold each keyframe
        max_episode_steps: int = 4_000,
        warmup_steps: int = 20,                 # hold neutral before gait starts (lets robot settle)
        frame_skip: int = 5,                    # 5 × 0.005s = 0.025s per action (40 Hz)
        reward_tracking: float = 1.5,           # guidance only — must not dominate velocity
        reward_vel: float = 12.0,               # forward velocity (primary objective)
        reward_survival: float = 0.3,           # per-step upright bonus
        reward_tilt: float = -2.0,
        reward_pitch: float = -4.0,
        reward_pitch_back: float = -8.0,
        reward_height: float = -3.0,
        reward_action: float = -0.05,
        reward_lateral: float = -1.0,
        reward_yaw_rate: float = -0.5,
        reward_fall: float = -2.0,
        tracking_sigma: float = 1.0,            # sharpness of tracking reward: exp(-sigma * err)
        upright_height_min: float = 0.14,
        upright_tilt_sq_max: float = 0.15,
        height_threshold: float = 0.10,
        tilt_threshold: float = 0.5,
        reset_noise_scale: float = 0.02,
        render_mode: str | None = None,
    ):
        super().__init__()

        self.keyframes       = np.array(keyframes if keyframes is not None else WALK_5_KEYFRAMES, dtype=np.float64)
        self.n_frames        = len(self.keyframes)
        self.steps_per_frame = steps_per_frame

        self.max_episode_steps  = max_episode_steps
        self.frame_skip         = frame_skip
        self.reward_tracking    = reward_tracking
        self.reward_vel         = reward_vel
        self.reward_survival    = reward_survival
        self.reward_tilt        = reward_tilt
        self.reward_pitch       = reward_pitch
        self.reward_pitch_back  = reward_pitch_back
        self.reward_height      = reward_height
        self.reward_action      = reward_action
        self.reward_lateral     = reward_lateral
        self.reward_yaw_rate    = reward_yaw_rate
        self.reward_fall        = reward_fall
        self.tracking_sigma     = tracking_sigma
        self.upright_height_min = upright_height_min
        self.upright_tilt_sq_max = upright_tilt_sq_max
        self.height_threshold   = height_threshold
        self.tilt_threshold     = tilt_threshold
        self.reset_noise_scale  = reset_noise_scale
        self.warmup_steps       = warmup_steps
        self.render_mode        = render_mode

        xml_str = load_spotmicro_xml()
        self.model = mujoco.MjModel.from_xml_string(xml_str)
        self.data  = mujoco.MjData(self.model)
        # timestep set in XML <option> block (0.005s)

        self._joint_qposadr = np.array([
            self.model.jnt_qposadr[mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, n)]
            for n in JOINT_NAMES], dtype=np.int32)
        self._joint_dofadr = np.array([
            self.model.jnt_dofadr[mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, n)]
            for n in JOINT_NAMES], dtype=np.int32)

        ctrl_range  = self.model.actuator_ctrlrange
        range_span  = ctrl_range[:, 1] - ctrl_range[:, 0]
        # XML motors have no ctrlrange set → range_span=0 → use 5 Nm fallback.
        # 5 Nm gives enough torque for a 4.4 kg robot to push off and walk.
        self._ctrl_high = ctrl_range[:, 1].copy() if np.any(range_span > 1e-6) \
                          else np.full(self.model.nu, 5.0)

        # obs = base (1+4+3+3+12+12=35) + target_pose (12) + phase (1) = 48
        obs_dim = 35 + NUM_JOINTS + 1
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float64)
        self.action_space      = spaces.Box(low=-1.0, high=1.0, shape=(self.model.nu,), dtype=np.float64)

        self._step_count  = 0
        self._viewer      = None

    # ------------------------------------------------------------------
    # Phase helpers
    # ------------------------------------------------------------------

    def _current_target(self) -> np.ndarray:
        """Return the target joint positions for the current gait phase.

        During warmup (first `warmup_steps` steps) the target is the neutral
        standing pose (all zeros) so the robot can settle before the gait starts.
        """
        if self._step_count < self.warmup_steps:
            return STAND_JOINTS.copy()
        adjusted  = self._step_count - self.warmup_steps
        frame_idx = (adjusted // self.steps_per_frame) % self.n_frames
        next_idx  = (frame_idx + 1) % self.n_frames
        t = (adjusted % self.steps_per_frame) / self.steps_per_frame
        return (1.0 - t) * self.keyframes[frame_idx] + t * self.keyframes[next_idx]

    def _gait_phase(self) -> float:
        """Normalised phase in [0, 1) over one full gait cycle (0 during warmup)."""
        if self._step_count < self.warmup_steps:
            return 0.0
        adjusted  = self._step_count - self.warmup_steps
        cycle_len = self.n_frames * self.steps_per_frame
        return (adjusted % cycle_len) / cycle_len

    # ------------------------------------------------------------------
    # Core gym interface
    # ------------------------------------------------------------------

    def _get_obs(self) -> np.ndarray:
        target = self._current_target()
        phase  = self._gait_phase()
        return np.concatenate([
            [self.data.qpos[2]],                          # height
            self.data.qpos[3:7],                          # quaternion
            self.data.qvel[0:3],                          # linear vel
            self.data.qvel[3:6],                          # angular vel
            self.data.qpos[self._joint_qposadr],          # joint positions
            self.data.qvel[self._joint_dofadr],           # joint velocities
            target,                                       # target pose (12)
            [phase],                                      # gait phase (1)
        ]).astype(np.float64)

    def _get_reward(self, x_vel: float, height: float, quat: np.ndarray,
                    action: np.ndarray, joint_pos: np.ndarray) -> float:
        roll, pitch, _ = _quat_to_euler(quat)
        tilt_sq    = roll**2 + pitch**2
        height_err = (height - DEFAULT_BASE_HEIGHT) ** 2
        is_upright = height >= self.upright_height_min and tilt_sq < self.upright_tilt_sq_max

        # --- Tracking reward (DeepMimic-style) ---
        target   = self._current_target()
        pose_err = float(np.mean((joint_pos - target) ** 2))
        tracking_reward = self.reward_tracking * np.exp(-self.tracking_sigma * pose_err)

        # --- Locomotion rewards ---
        vel_reward      = self.reward_vel * x_vel if is_upright else 0.0
        survival_reward = self.reward_survival    if is_upright else 0.0
        fall_penalty    = self.reward_fall        if not is_upright else 0.0

        # --- Stability penalties ---
        y_vel    = float(self.data.qvel[1])
        yaw_rate = float(self.data.qvel[5])
        pitch_back = max(0.0, pitch)
        tilt_penalty = (
            self.reward_tilt       * roll**2
            + self.reward_pitch    * pitch**2
            + self.reward_pitch_back * pitch_back**2
        )

        return (
            tracking_reward
            + vel_reward
            + survival_reward
            + fall_penalty
            + tilt_penalty
            + self.reward_height   * height_err
            + self.reward_lateral  * y_vel**2
            + self.reward_yaw_rate * yaw_rate**2
            + self.reward_action   * float(np.sum(action**2))
        )

    def _is_terminated(self, height: float, quat: np.ndarray) -> bool:
        if height < self.height_threshold:
            return True
        roll, pitch, _ = _quat_to_euler(quat)
        return abs(roll) > self.tilt_threshold or abs(pitch) > self.tilt_threshold

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self._step_count = 0

        # Reset to the natural flexed standing pose (hip=0.65, knee=-1.40).
        # DEFAULT_BASE_HEIGHT (0.202m) is computed to match this joint config so
        # the feet land exactly on the ground — no floating, no collision.
        self.data.qpos[:] = 0.0
        self.data.qpos[2] = DEFAULT_BASE_HEIGHT
        self.data.qpos[3:7] = [1.0, 0.0, 0.0, 0.0]
        self.data.qpos[self._joint_qposadr] = STAND_JOINTS.copy()
        self.data.qvel[:] = 0.0

        if self.reset_noise_scale > 0 and self.np_random is not None:
            self.data.qpos[2] += self.np_random.uniform(
                -self.reset_noise_scale, self.reset_noise_scale)
            self.data.qpos[self._joint_qposadr] += self.np_random.uniform(
                -self.reset_noise_scale, self.reset_noise_scale, size=NUM_JOINTS)

        mujoco.mj_forward(self.model, self.data)
        return self._get_obs(), {"height": float(self.data.qpos[2])}

    def step(self, action):
        self.data.ctrl[:] = action * self._ctrl_high

        for _ in range(self.frame_skip):
            mujoco.mj_step(self.model, self.data)

        self._step_count += 1
        height    = float(self.data.qpos[2])
        quat      = self.data.qpos[3:7].copy()
        x_vel     = float(self.data.qvel[0])
        joint_pos = self.data.qpos[self._joint_qposadr].copy()

        reward     = self._get_reward(x_vel, height, quat, action, joint_pos)
        terminated = self._is_terminated(height, quat)
        truncated  = self._step_count >= self.max_episode_steps
        obs        = self._get_obs()

        if self.render_mode == "human":
            self._render_frame()

        return obs, reward, terminated, truncated, {
            "height": height, "x_velocity": x_vel,
            "gait_phase": self._gait_phase(),
            "frame_idx": (self._step_count // self.steps_per_frame) % self.n_frames,
        }

    def _render_frame(self) -> None:
        if self._viewer is None:
            from mujoco import viewer as mj_viewer
            self._viewer = mj_viewer.launch_passive(self.model, self.data, key_callback=None)
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
