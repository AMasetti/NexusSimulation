import mujoco
import mujoco.viewer
import time
import os
import numpy as np

os.chdir(os.path.dirname(os.path.abspath(__file__)))

model = mujoco.MjModel.from_xml_path("optimus_mujoco_fixed.xml")
data = mujoco.MjData(model)

jpos = {}
jdof = {}
for i in range(model.njnt):
    name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i)
    if name and name != "root":
        jpos[name] = model.jnt_qposadr[i]
        jdof[name] = model.jnt_dofadr[i]

amap = {}
for i in range(model.nu):
    name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
    if name:
        amap[name] = i

KP = 80.0
KD = 5.0

KNEE = 0.0

TARGET = {
    "Servo-Hip-Body-Rotation":          0.0,
    "Servo-Hip-L":                      0.0,
    "Servo-Hip-R":                      0.0,
    "Servo-Knee-L-Top":                 KNEE,
    "Servo-Knee-L-Bottom":              KNEE,
    "Servo-Knee-R-Top":                 KNEE,
    "Servo-Knee-R-Bottom":              KNEE,
    "Servo-Ankle-L":                    0.0,
    "Servo-Ankle-R":                    0.0,
    "Servo-Showlder-L-Front-Back":      0.0,
    "Servo-Showlder-R-Front-Back":      0.0,
    "Servo-Showlder-L-Inward-Outward":  0.0,
    "Servo-Showlder-R-Inward-Outward":  0.0,
    "Servo-Forearm-L":                  0.0,
    "Servo-Forearm-R":                  0.0,
}


PARA_KP = 400.0   # stiff spring to enforce parallelogram constraint
PARA_KD = 20.0

def apply_parallelogram():
    """
    Enforce parallelogram linkage via torque on unactuated joints instead of
    writing qpos/qvel directly. Direct qpos/qvel writes create velocity
    discontinuities that produce spurious internal forces and body drift.
    """
    pairs = [
        ("Servo-Knee-L-Top",    "Unactuated-Knee-L-Top",    -1),
        ("Servo-Knee-L-Top",    "Unactuated-Tendon-L-Top",  -1),
        ("Servo-Knee-L-Bottom", "Unactuated-Knee-L-Bottom", -1),
        ("Servo-Knee-L-Bottom", "Unactuated-Tendon-L-Bottom", +1),
        ("Servo-Knee-R-Top",    "Unactuated-Knee-R-Top",    -1),
        ("Servo-Knee-R-Top",    "Unactuated-Tendon-R-Top",  -1),
        ("Servo-Knee-R-Bottom", "Unactuated-Knee-R-Bottom", -1),
        ("Servo-Knee-R-Bottom", "Unactuated-Tendon-R-Bottom", +1),
    ]
    for driven, follower, sign in pairs:
        q_target = sign * data.qpos[jpos[driven]]
        v_target = sign * data.qvel[jdof[driven]]
        q_err = q_target - data.qpos[jpos[follower]]
        v_err = v_target - data.qvel[jdof[follower]]
        data.qfrc_applied[jdof[follower]] = PARA_KP * q_err + PARA_KD * v_err


def apply_pd():
    for jname, target in TARGET.items():
        if jname not in amap:
            continue
        q  = data.qpos[jpos[jname]]
        dq = data.qvel[jdof[jname]]
        data.ctrl[amap[jname]] = KP * (target - q) - KD * dq


WARMUP_STEPS = 1000

print(f"Optimus loaded — {model.nbody} bodies, {model.nu} actuators")
print(f"Pose: knee={np.degrees(KNEE):.1f}°  total mass: {sum(model.body_mass[i] for i in range(1,model.nbody)):.2f} kg")
print(f"Warming up {WARMUP_STEPS} steps, then releasing freejoint...")
print("Close the viewer window to exit.")

saved_qpos0 = data.qpos[:7].copy()

for _ in range(WARMUP_STEPS):
    apply_parallelogram()
    apply_pd()
    mujoco.mj_step(model, data)
    data.qpos[:7] = saved_qpos0
    data.qvel[:] = 0.0

print("Freejoint released — robot is live.")

with mujoco.viewer.launch_passive(model, data) as viewer:
    while viewer.is_running():
        apply_parallelogram()
        apply_pd()
        mujoco.mj_step(model, data)
        viewer.sync()
        time.sleep(0.001)
