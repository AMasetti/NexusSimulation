"""
Host-side robot controller.

Bridges the SpotMicroKinematics solver to the physical robot:
    kinematics (joint angles) → serial → ESP32 → PCA9685 → MG995 servos

── Quickstart ────────────────────────────────────────────────────────
    python robot_controller.py                    # interactive REPL
    python robot_controller.py --port /dev/ttyUSB0

── Programmatic use ──────────────────────────────────────────────────
    from firmware.robot_controller import RobotController
    import numpy as np

    with RobotController(port="/dev/ttyUSB0") as robot:
        robot.centre()
        robot.set_body_pose(pos=[0, 0, 0.200], euler=[0, 0, 0])
        robot.move_to_pose(
            start_pos=[0, 0, 0.200], start_euler=[0, 0, 0],
            end_pos=[0, 0, 0.160],   end_euler=[0.3, 0, 0],
            n_steps=60, dt=0.04,
        )
"""

import argparse
import json
import sys
import time
import os

import numpy as np
import serial

# ── Path setup (works whether run directly or imported) ───────────────────────
_HERE    = os.path.dirname(os.path.abspath(__file__))
_KIN_DIR = os.path.dirname(_HERE)          # kinematics/
_MUJ_DIR = os.path.dirname(_KIN_DIR)       # mujuco/
for _p in [_KIN_DIR, _MUJ_DIR]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from spotmicro_kinematics import SpotMicroKinematics
from firmware.servo_config import SERVO_MAP, angle_to_pulse_us

# ── Default serial settings ───────────────────────────────────────────────────
DEFAULT_PORT     = "/dev/ttyUSB0"
DEFAULT_BAUDRATE = 115200
DEFAULT_TIMEOUT  = 2.0   # seconds

NEUTRAL_POS   = np.array([0.0, 0.0, 0.200])
NEUTRAL_EULER = np.zeros(3)


class RobotController:
    """
    High-level controller: body pose → IK → servo angles → ESP32.

    Can be used as a context manager:
        with RobotController("/dev/ttyUSB0") as robot:
            robot.set_body_pose([0, 0, 0.18], [0.2, 0, 0])
    """

    def __init__(self, port: str = DEFAULT_PORT,
                 baudrate: int = DEFAULT_BAUDRATE,
                 timeout: float = DEFAULT_TIMEOUT):
        self._port     = port
        self._baudrate = baudrate
        self._timeout  = timeout
        self._serial   = None
        self._kin      = SpotMicroKinematics()

    # ------------------------------------------------------------------
    # Connection management
    # ------------------------------------------------------------------

    def connect(self):
        self._serial = serial.Serial(
            self._port, self._baudrate,
            timeout=self._timeout,
        )
        time.sleep(1.5)   # let ESP32 boot / reset after DTR pulse
        self._serial.reset_input_buffer()
        resp = self.ping()
        if not resp.get("ok"):
            raise ConnectionError(f"ESP32 ping failed: {resp}")
        print(f"[robot] connected on {self._port}  ({resp.get('msg', '')})")

    def disconnect(self):
        if self._serial and self._serial.is_open:
            try:
                self.halt()
            except Exception:
                pass
            self._serial.close()
        self._serial = None

    def __enter__(self):
        self.connect()
        return self

    def __exit__(self, *_):
        self.disconnect()

    # ------------------------------------------------------------------
    # Low-level serial commands
    # ------------------------------------------------------------------

    def _send(self, msg: dict) -> dict:
        if not self._serial or not self._serial.is_open:
            raise ConnectionError("Not connected — call connect() first")
        line = json.dumps(msg) + "\n"
        self._serial.write(line.encode())
        raw = self._serial.readline()
        if not raw:
            raise TimeoutError("No response from ESP32")
        return json.loads(raw.decode().strip())

    def ping(self) -> dict:
        return self._send({"cmd": "ping"})

    def halt(self) -> dict:
        """Hold current servo positions (no movement)."""
        return self._send({"cmd": "halt"})

    def centre(self) -> dict:
        """Move all servos to their 1500 µs centre position."""
        return self._send({"cmd": "centre"})

    def send_angles(self, angles: np.ndarray) -> dict:
        """
        Send 12 joint angles (rad) directly to the robot.
        Order: [FL_sh, FL_hip, FL_knee, FR..., RL..., RR...]
        """
        return self._send({
            "cmd":  "angles",
            "data": angles[:12].tolist(),
        })

    # ------------------------------------------------------------------
    # Kinematics-based commands
    # ------------------------------------------------------------------

    def set_body_pose(self, pos, euler) -> dict:
        """
        Move the robot to a body pose by solving IK.

        pos   : [x, y, z] metres
        euler : [roll, pitch, yaw] radians
        """
        pos   = np.asarray(pos,   dtype=float)
        euler = np.asarray(euler, dtype=float)
        feet  = self._kin.compute_stance_feet(pos, euler)
        angles, reachable = self._kin.inverse_kinematics(pos, euler, feet)
        if not reachable.all():
            bad = [i for i, r in enumerate(reachable) if not r]
            print(f"[warn] legs {bad} hit joint limits")
        return self.send_angles(angles)

    def move_to_pose(self,
                     start_pos, start_euler,
                     end_pos,   end_euler,
                     n_steps: int = 60,
                     dt: float    = 0.04):
        """
        Smoothly interpolate from start to end body pose.

        n_steps : number of intermediate frames
        dt      : seconds per frame
        """
        start_pos   = np.asarray(start_pos,   dtype=float)
        start_euler = np.asarray(start_euler, dtype=float)
        end_pos     = np.asarray(end_pos,     dtype=float)
        end_euler   = np.asarray(end_euler,   dtype=float)

        joint_traj, _, reachable_mask = self._kin.interpolate_poses(
            start_pos, start_euler,
            end_pos,   end_euler,
            n_steps=n_steps,
            keep_feet_fixed=True,
        )
        bad = int((~reachable_mask.all(axis=1)).sum())
        if bad:
            print(f"[warn] {bad}/{n_steps} frames have joint-limit clamping")

        for angles in joint_traj:
            self.send_angles(angles)
            time.sleep(dt)

    def reset_to_neutral(self, n_steps: int = 40, dt: float = 0.04):
        """Return to the default standing pose."""
        neutral_feet   = self._kin.compute_stance_feet(NEUTRAL_POS, NEUTRAL_EULER)
        neutral_angles, _ = self._kin.inverse_kinematics(NEUTRAL_POS, NEUTRAL_EULER, neutral_feet)
        # Best-effort interpolation from wherever the robot currently is —
        # just send the target directly if we don't know the current pose.
        self.send_angles(neutral_angles)

    # ------------------------------------------------------------------
    # Full DOF sweep (mirrors run_kinematics.py)
    # ------------------------------------------------------------------

    MOVEMENTS = [
        ("1 · Squat/Rise         [Z — hip + knee]",
         NEUTRAL_POS, NEUTRAL_EULER,
         np.array([0.0,  0.0, 0.120]), np.zeros(3)),
        ("2 · Forward Shift      [X — hip front/rear]",
         NEUTRAL_POS, NEUTRAL_EULER,
         np.array([0.06, 0.0, 0.200]), np.zeros(3)),
        ("3 · Lateral Shift      [Y — shoulder L/R]",
         NEUTRAL_POS, NEUTRAL_EULER,
         np.array([0.0, 0.05, 0.200]), np.zeros(3)),
        ("4 · Roll               [shoulder max range]",
         NEUTRAL_POS, NEUTRAL_EULER,
         NEUTRAL_POS.copy(), np.array([0.4, 0.0, 0.0])),
        ("5 · Pitch              [hip front vs rear]",
         NEUTRAL_POS, NEUTRAL_EULER,
         NEUTRAL_POS.copy(), np.array([0.0, 0.3, 0.0])),
        ("6 · Yaw                [all shoulders + hip]",
         NEUTRAL_POS, NEUTRAL_EULER,
         NEUTRAL_POS.copy(), np.array([0.0, 0.0, 0.5])),
        ("7 · Roll + Squat       [shoulder + hip/knee]",
         NEUTRAL_POS, NEUTRAL_EULER,
         np.array([0.0,  0.0, 0.150]), np.array([0.35, 0.0, 0.0])),
        ("8 · Pitch + Fwd Lean   [front compression]",
         NEUTRAL_POS, NEUTRAL_EULER,
         np.array([0.05, 0.0, 0.190]), np.array([0.0, 0.25, 0.0])),
        ("9 · Full 6-DOF         [all joints]",
         NEUTRAL_POS, NEUTRAL_EULER,
         np.array([0.04, 0.03, 0.160]), np.array([0.25, 0.15, 0.3])),
    ]

    def run_dof_sweep(self, n_steps: int = 60, dt: float = 0.04,
                      cycles: int = 0):
        """
        Run all 9 DOF test movements on the physical robot.

        cycles : number of full cycles (0 = loop forever until Ctrl-C)
        """
        cycle = 0
        try:
            while cycles == 0 or cycle < cycles:
                cycle += 1
                print(f"\n── Cycle {cycle} {'(∞)' if cycles == 0 else f'/{cycles}'} ──")
                for label, sp, se, tp, te in self.MOVEMENTS:
                    print(f"  {label}")
                    self.reset_to_neutral()
                    time.sleep(0.3)
                    # Forward
                    self.move_to_pose(sp, se, tp, te, n_steps=n_steps, dt=dt)
                    # Return
                    self.move_to_pose(tp, te, sp, se, n_steps=n_steps, dt=dt)
                self.reset_to_neutral()
                time.sleep(0.5)
        except KeyboardInterrupt:
            print("\n[robot] sweep interrupted — halting")
            self.halt()


# ── CLI entry point ───────────────────────────────────────────────────────────

def _interactive(robot: RobotController):
    print("\nRobot interactive REPL — type 'help' for commands.")
    CMDS = """
  centre                        → all servos to centre
  neutral                       → standing neutral pose
  pose x y z roll pitch yaw     → set body pose (metres / radians)
  sweep                         → run full 9-movement DOF sweep
  ping                          → connectivity check
  quit / exit                   → disconnect and exit
"""
    print(CMDS)
    while True:
        try:
            line = input("robot> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not line:
            continue
        parts = line.split()
        cmd   = parts[0].lower()

        if cmd in ("quit", "exit"):
            break
        elif cmd == "help":
            print(CMDS)
        elif cmd == "ping":
            print(robot.ping())
        elif cmd == "centre":
            print(robot.centre())
        elif cmd == "neutral":
            robot.reset_to_neutral()
        elif cmd == "pose" and len(parts) == 7:
            vals = list(map(float, parts[1:]))
            print(robot.set_body_pose(vals[:3], vals[3:]))
        elif cmd == "sweep":
            robot.run_dof_sweep()
        else:
            print("  unknown command — type 'help'")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SpotMicro robot controller")
    parser.add_argument("--port",     default=DEFAULT_PORT,     help="Serial port")
    parser.add_argument("--baudrate", default=DEFAULT_BAUDRATE, type=int)
    parser.add_argument("--sweep",    action="store_true",      help="Run DOF sweep immediately")
    parser.add_argument("--cycles",   default=0, type=int,      help="Sweep cycles (0=∞)")
    args = parser.parse_args()

    with RobotController(port=args.port, baudrate=args.baudrate) as robot:
        if args.sweep:
            robot.run_dof_sweep(cycles=args.cycles)
        else:
            _interactive(robot)
