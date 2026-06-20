"""
SpotMicro Inverse Kinematics — body-CoG-based geometric solver.

Given a desired body pose (x, y, z, roll, pitch, yaw), computes the 12 joint
angles required for all four legs to accommodate that posture with feet on the
ground. Pure geometry — no MuJoCo dependency.

Robot geometry verified directly from:
  mujuco/Mujuco XML/SpotMicro/spotmicro.xml
"""

import numpy as np
from typing import Tuple

# ---------------------------------------------------------------------------
# Robot geometry constants (all values in metres / radians)
# ---------------------------------------------------------------------------

LEG_NAMES = ["front_left", "front_right", "rear_left", "rear_right"]

# Shoulder joint origins in the body frame
SHOULDER_ORIGINS = {
    "front_left":  np.array([ 0.093,  0.0395, 0.0]),
    "front_right": np.array([ 0.093, -0.0395, 0.0]),
    "rear_left":   np.array([-0.093,  0.0395, 0.0]),
    "rear_right":  np.array([-0.093, -0.0395, 0.0]),
}

# Lateral offset from shoulder joint to hip joint (shoulder → leg_link pos)
HIP_OFFSET_Y = {
    "front_left":   0.055,
    "front_right": -0.055,
    "rear_left":    0.055,
    "rear_right":  -0.055,
}

# Upper-leg link: foot_link pos relative to leg_link = (0.014, 0, -0.109)
_KNEE_X = 0.014     # forward offset introduces a lean angle
_KNEE_Z = 0.109     # vertical drop
L1 = np.sqrt(_KNEE_X**2 + _KNEE_Z**2)  # effective upper-leg length ≈ 0.10989 m
ALPHA = np.arctan2(_KNEE_X, _KNEE_Z)   # upper-leg forward lean ≈ 0.1284 rad

# Lower-leg link: toe_link pos relative to foot_link = (0, 0, -0.130)
L2 = 0.130  # m

# Joint limits [rad]
JOINT_LIMITS = {
    "shoulder": (-0.548,  0.548),
    "hip":      (-2.666,  1.548),
    "knee":     (-2.590,  0.100),
}

# Default standing height (body base_link initial z in XML)
DEFAULT_HEIGHT = 0.259


# ---------------------------------------------------------------------------
# SpotMicroKinematics
# ---------------------------------------------------------------------------

class SpotMicroKinematics:
    """
    Analytical body-CoG inverse kinematics for SpotMicro (12 DOF quadruped).

    Joint ordering throughout this class (matches spotmicro.xml actuator order):
        [FL_shoulder, FL_hip, FL_knee,
         FR_shoulder, FR_hip, FR_knee,
         RL_shoulder, RL_hip, RL_knee,
         RR_shoulder, RR_hip, RR_knee]

    Euler angles convention: ZYX (yaw applied first, then pitch, then roll),
    which matches the MuJoCo quat_to_euler convention used in spotmicro_env.py.
    """

    def __repr__(self) -> str:
        lines = ["SpotMicroKinematics — robot geometry:"]
        for leg in LEG_NAMES:
            o = SHOULDER_ORIGINS[leg]
            h = HIP_OFFSET_Y[leg]
            lines.append(
                f"  {leg:15s}  shoulder=({o[0]:+.4f}, {o[1]:+.4f}, {o[2]:+.4f})  "
                f"hip_offset_y={h:+.4f}"
            )
        lines.append(f"  L1={L1:.5f} m  L2={L2:.4f} m  alpha={np.degrees(ALPHA):.3f}°")
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def forward_kinematics(
        self,
        body_pos: np.ndarray,
        body_euler: np.ndarray,
        joint_angles: np.ndarray,
    ) -> np.ndarray:
        """
        Compute toe world positions given body pose and joint angles.

        Parameters
        ----------
        body_pos : (3,) world position of body origin [x, y, z]
        body_euler : (3,) [roll, pitch, yaw] radians
        joint_angles : (12,) joint angles in leg order FL/FR/RL/RR,
                       each as [shoulder, hip, knee]

        Returns
        -------
        (4, 3) toe positions in world frame, rows = [FL, FR, RL, RR]
        """
        body_pos = np.asarray(body_pos, dtype=float)
        body_euler = np.asarray(body_euler, dtype=float)
        joint_angles = np.asarray(joint_angles, dtype=float)

        T = self._build_body_transform(body_pos, body_euler)
        feet = np.zeros((4, 3))

        for i, leg in enumerate(LEG_NAMES):
            sh, hip, knee = joint_angles[i * 3: i * 3 + 3]
            feet[i] = self._fk_leg(T, leg, sh, hip, knee)

        return feet

    def inverse_kinematics(
        self,
        body_pos: np.ndarray,
        body_euler: np.ndarray,
        foot_positions: np.ndarray,
        knee_sign: float = -1.0,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute joint angles so each toe reaches the desired world position.

        Parameters
        ----------
        body_pos : (3,)
        body_euler : (3,) [roll, pitch, yaw]
        foot_positions : (4, 3) desired toe world positions [FL, FR, RL, RR]
        knee_sign : +1 for knee-up, -1 for knee-down (SpotMicro uses -1)

        Returns
        -------
        joint_angles : (12,)
        reachable : (4,) bool — False if IK had to clamp to joint limits
        """
        body_pos = np.asarray(body_pos, dtype=float)
        body_euler = np.asarray(body_euler, dtype=float)
        foot_positions = np.asarray(foot_positions, dtype=float)

        T = self._build_body_transform(body_pos, body_euler)
        joint_angles = np.zeros(12)
        reachable = np.ones(4, dtype=bool)

        for i, leg in enumerate(LEG_NAMES):
            angles, ok = self._solve_leg_ik(
                foot_positions[i], T, leg, knee_sign
            )
            joint_angles[i * 3: i * 3 + 3] = angles
            reachable[i] = ok

        return joint_angles, reachable

    def compute_stance_feet(
        self,
        body_pos: np.ndarray,
        body_euler: np.ndarray,
    ) -> np.ndarray:
        """
        Compute natural foot positions (z=0 ground plane) for a given body pose.

        Projects each shoulder position straight down to z=0 (accounting for
        body tilt), giving a stable flat-ground stance.

        Returns
        -------
        (4, 3) foot world positions with z=0
        """
        body_pos = np.asarray(body_pos, dtype=float)
        body_euler = np.asarray(body_euler, dtype=float)

        T = self._build_body_transform(body_pos, body_euler)
        feet = np.zeros((4, 3))

        for i, leg in enumerate(LEG_NAMES):
            # Hip joint world position (shoulder + lateral offset, θ1=0)
            hip_body = SHOULDER_ORIGINS[leg] + np.array([0.0, HIP_OFFSET_Y[leg], 0.0])
            hip_world = (T @ np.append(hip_body, 1.0))[:3]

            # Foot is directly below the hip joint at ground level (z=0).
            # Reachable whenever body height ≤ L1+L2 ≈ 0.240 m.
            feet[i] = [hip_world[0], hip_world[1], 0.0]

        return feet

    def interpolate_poses(
        self,
        start_body_pos: np.ndarray,
        start_body_euler: np.ndarray,
        end_body_pos: np.ndarray,
        end_body_euler: np.ndarray,
        n_steps: int = 100,
        keep_feet_fixed: bool = True,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Generate a kinematic trajectory between two body poses.

        Parameters
        ----------
        keep_feet_fixed : if True, feet stay at their start-stance world
                          positions (body adjusts over planted feet).
                          If False, stance is recomputed at every step.

        Returns
        -------
        joint_traj    : (N, 12)
        foot_traj     : (N, 4, 3)
        reachable_mask: (N, 4) bool
        """
        start_body_pos = np.asarray(start_body_pos, dtype=float)
        start_body_euler = np.asarray(start_body_euler, dtype=float)
        end_body_pos = np.asarray(end_body_pos, dtype=float)
        end_body_euler = np.asarray(end_body_euler, dtype=float)

        joint_traj = np.zeros((n_steps, 12))
        foot_traj = np.zeros((n_steps, 4, 3))
        reachable_mask = np.ones((n_steps, 4), dtype=bool)

        # Quaternions for SLERP
        q_start = self._euler_to_quat_wxyz(start_body_euler)
        q_end = self._euler_to_quat_wxyz(end_body_euler)

        # Fixed foot positions computed at start pose
        if keep_feet_fixed:
            fixed_feet = self.compute_stance_feet(start_body_pos, start_body_euler)

        for i in range(n_steps):
            t = i / max(n_steps - 1, 1)

            body_pos_t = (1.0 - t) * start_body_pos + t * end_body_pos
            q_t = self._slerp(q_start, q_end, t)
            body_euler_t = self._quat_wxyz_to_euler(q_t)

            if keep_feet_fixed:
                foot_pos_t = fixed_feet
            else:
                foot_pos_t = self.compute_stance_feet(body_pos_t, body_euler_t)

            angles, reachable = self.inverse_kinematics(
                body_pos_t, body_euler_t, foot_pos_t
            )
            joint_traj[i] = angles
            foot_traj[i] = foot_pos_t
            reachable_mask[i] = reachable

        return joint_traj, foot_traj, reachable_mask

    def build_qpos(
        self,
        body_pos: np.ndarray,
        body_euler: np.ndarray,
        joint_angles: np.ndarray,
    ) -> np.ndarray:
        """
        Build a MuJoCo qpos array (19,) from body pose and joint angles.

        Layout: [x, y, z, qw, qx, qy, qz, FL_sh, FL_hip, FL_knee,
                 FR_sh, FR_hip, FR_knee, RL_sh, RL_hip, RL_knee,
                 RR_sh, RR_hip, RR_knee]
        """
        body_pos = np.asarray(body_pos, dtype=float)
        body_euler = np.asarray(body_euler, dtype=float)
        joint_angles = np.asarray(joint_angles, dtype=float)

        qpos = np.zeros(19)
        qpos[0:3] = body_pos
        qpos[3:7] = self._euler_to_quat_wxyz(body_euler)
        qpos[7:19] = joint_angles
        return qpos

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _euler_to_rot(self, roll: float, pitch: float, yaw: float) -> np.ndarray:
        """ZYX Euler → 3×3 rotation matrix (world = R @ body_vec)."""
        cr, sr = np.cos(roll),  np.sin(roll)
        cp, sp = np.cos(pitch), np.sin(pitch)
        cy, sy = np.cos(yaw),   np.sin(yaw)

        Rx = np.array([[1,  0,   0 ],
                       [0,  cr, -sr],
                       [0,  sr,  cr]])
        Ry = np.array([[ cp, 0, sp],
                       [  0, 1,  0],
                       [-sp, 0, cp]])
        Rz = np.array([[cy, -sy, 0],
                       [sy,  cy, 0],
                       [ 0,   0, 1]])
        return Rz @ Ry @ Rx

    def _build_body_transform(
        self, body_pos: np.ndarray, body_euler: np.ndarray
    ) -> np.ndarray:
        """Return 4×4 homogeneous transform: body frame → world frame."""
        R = self._euler_to_rot(*body_euler)
        T = np.eye(4)
        T[:3, :3] = R
        T[:3,  3] = body_pos
        return T

    def _euler_to_quat_wxyz(self, euler: np.ndarray) -> np.ndarray:
        """ZYX Euler [roll, pitch, yaw] → unit quaternion [w, x, y, z]."""
        roll, pitch, yaw = euler
        cr, sr = np.cos(roll / 2),  np.sin(roll / 2)
        cp, sp = np.cos(pitch / 2), np.sin(pitch / 2)
        cy, sy = np.cos(yaw / 2),   np.sin(yaw / 2)

        w = cr * cp * cy + sr * sp * sy
        x = sr * cp * cy - cr * sp * sy
        y = cr * sp * cy + sr * cp * sy
        z = cr * cp * sy - sr * sp * cy

        q = np.array([w, x, y, z])
        return q / np.linalg.norm(q)

    def _quat_wxyz_to_euler(self, q: np.ndarray) -> np.ndarray:
        """Unit quaternion [w, x, y, z] → ZYX Euler [roll, pitch, yaw]."""
        w, x, y, z = q
        # Roll (x-axis rotation)
        sinr_cosp = 2.0 * (w * x + y * z)
        cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
        roll = np.arctan2(sinr_cosp, cosr_cosp)
        # Pitch (y-axis rotation)
        sinp = 2.0 * (w * y - z * x)
        sinp = np.clip(sinp, -1.0, 1.0)
        pitch = np.arcsin(sinp)
        # Yaw (z-axis rotation)
        siny_cosp = 2.0 * (w * z + x * y)
        cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
        yaw = np.arctan2(siny_cosp, cosy_cosp)
        return np.array([roll, pitch, yaw])

    def _slerp(self, q0: np.ndarray, q1: np.ndarray, t: float) -> np.ndarray:
        """Spherical linear interpolation between two unit quaternions."""
        q0 = q0 / np.linalg.norm(q0)
        q1 = q1 / np.linalg.norm(q1)
        dot = np.dot(q0, q1)
        # Ensure shortest path
        if dot < 0.0:
            q1 = -q1
            dot = -dot
        dot = np.clip(dot, -1.0, 1.0)
        if dot > 0.9995:
            # Nearly identical — linear interpolation
            result = q0 + t * (q1 - q0)
            return result / np.linalg.norm(result)
        theta_0 = np.arccos(dot)
        theta = theta_0 * t
        sin_theta = np.sin(theta)
        sin_theta_0 = np.sin(theta_0)
        s0 = np.cos(theta) - dot * sin_theta / sin_theta_0
        s1 = sin_theta / sin_theta_0
        return (s0 * q0 + s1 * q1) / np.linalg.norm(s0 * q0 + s1 * q1)

    def _fk_leg(
        self,
        T_body_world: np.ndarray,
        leg: str,
        theta1: float,
        theta2: float,
        theta3: float,
    ) -> np.ndarray:
        """Forward kinematics for one leg → toe world position."""
        d = HIP_OFFSET_Y[leg]

        # Shoulder frame: T_body_world @ translate(shoulder_origin)
        p_sh = SHOULDER_ORIGINS[leg]
        T_sh = T_body_world.copy()
        T_sh[:3, 3] = T_body_world[:3, :3] @ p_sh + T_body_world[:3, 3]

        # After shoulder rotation (Rx theta1): hip is at local (0, d, 0)
        Rx1 = self._rx(theta1)
        p_hip_in_sh = np.array([0.0, d, 0.0])
        p_hip_world = T_sh[:3, :3] @ (Rx1 @ p_hip_in_sh) + T_sh[:3, 3]
        R_hip = T_sh[:3, :3] @ Rx1

        # After hip rotation (Ry theta2): knee at local (0.014, 0, -0.109)
        Ry2 = self._ry(theta2)
        p_knee_in_hip = np.array([_KNEE_X, 0.0, -_KNEE_Z])
        p_knee_world = R_hip @ (Ry2 @ p_knee_in_hip) + p_hip_world
        R_knee = R_hip @ Ry2

        # After knee rotation (Ry theta3): toe at local (0, 0, -0.130)
        Ry3 = self._ry(theta3)
        p_toe_in_knee = np.array([0.0, 0.0, -L2])
        toe_world = R_knee @ (Ry3 @ p_toe_in_knee) + p_knee_world

        return toe_world

    def _solve_leg_ik(
        self,
        foot_world: np.ndarray,
        T_body_world: np.ndarray,
        leg: str,
        knee_sign: float,
    ) -> Tuple[np.ndarray, bool]:
        """
        Analytical IK for one 3-DOF leg.

        Returns (angles [sh, hip, knee], reachable).
        """
        reachable = True
        d = HIP_OFFSET_Y[leg]

        # --- Step 1: transform foot to shoulder frame ---
        T_world_body = np.linalg.inv(T_body_world)
        p_sh_origin = SHOULDER_ORIGINS[leg]

        foot_body_h = T_world_body @ np.append(foot_world, 1.0)
        foot_body = foot_body_h[:3]
        p_sh = foot_body - p_sh_origin  # foot in shoulder frame

        px, py, pz = p_sh

        # --- Step 2: shoulder abduction θ1 ---
        R_lat = np.sqrt(py**2 + pz**2)
        if R_lat < abs(d) - 1e-6:
            reachable = False
            R_lat = abs(d)  # clamp to avoid sqrt of negative

        discriminant = max(R_lat**2 - d**2, 0.0)
        theta1 = np.arctan2(-pz, py) - np.arctan2(np.sqrt(discriminant), d)

        # --- Step 3: project foot into hip-knee sagittal plane ---
        # Hip joint position in shoulder frame after applying θ1
        hip_y = d * np.cos(theta1)
        hip_z = d * np.sin(theta1)

        # Foot relative to hip joint (in shoulder frame)
        foot_from_hip = np.array([px, py - hip_y, pz - hip_z])

        hip_plane_x = foot_from_hip[0]
        hip_plane_z = -np.sqrt(foot_from_hip[1]**2 + foot_from_hip[2]**2)

        # --- Step 4: 2-link planar IK for hip (θ2) and knee (θ3) ---
        reach = np.sqrt(hip_plane_x**2 + hip_plane_z**2)

        max_reach = L1 + L2
        min_reach = abs(L1 - L2)

        if reach > max_reach - 1e-6 or reach < min_reach + 1e-6:
            reachable = False
            reach = np.clip(reach, min_reach + 1e-6, max_reach - 1e-6)

        cos_theta3 = (reach**2 - L1**2 - L2**2) / (2.0 * L1 * L2)
        cos_theta3 = np.clip(cos_theta3, -1.0, 1.0)
        # Correct formula: the law-of-cosines gives cos(α + θ3) not cos(θ3),
        # because the upper leg has a forward lean α baked into its geometry.
        theta3 = knee_sign * np.arccos(cos_theta3) - ALPHA

        # θ2: rotate the combined lower-leg vector [vx, vz] to match target [px, pz]
        vx = _KNEE_X - L2 * np.sin(theta3)
        vz = -_KNEE_Z - L2 * np.cos(theta3)
        theta2 = np.arctan2(vz, vx) - np.arctan2(hip_plane_z, hip_plane_x)

        # --- Step 5: clamp to joint limits ---
        def clamp(val, limits, name):
            lo, hi = limits
            clamped = np.clip(val, lo, hi)
            return clamped, abs(clamped - val) < 0.01

        theta1, ok1 = clamp(theta1, JOINT_LIMITS["shoulder"], "shoulder")
        theta2, ok2 = clamp(theta2, JOINT_LIMITS["hip"],      "hip")
        theta3, ok3 = clamp(theta3, JOINT_LIMITS["knee"],     "knee")

        if not (ok1 and ok2 and ok3):
            reachable = False

        return np.array([theta1, theta2, theta3]), reachable

    # ------------------------------------------------------------------
    # Rotation matrix helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _rx(angle: float) -> np.ndarray:
        c, s = np.cos(angle), np.sin(angle)
        return np.array([[1, 0,  0],
                         [0, c, -s],
                         [0, s,  c]])

    @staticmethod
    def _ry(angle: float) -> np.ndarray:
        c, s = np.cos(angle), np.sin(angle)
        return np.array([[ c, 0, s],
                         [ 0, 1, 0],
                         [-s, 0, c]])
