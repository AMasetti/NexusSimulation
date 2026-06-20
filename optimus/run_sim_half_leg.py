#!/usr/bin/env python3
"""
MuJoCo simulation for the half-leg robot.
- Loads robot.mujoco.xml from simulation/optimus/mjcf/half-leg/
- Resolves mesh paths against simulation/optimus/urdf/half-leg/meshes/
- MuJoCo does not support STEP meshes; the script expects .stl files (same base name as
  .step, e.g. Knee_Left.stl). Convert STEP to STL via step_to_stl.py.
- Loads constraints from simulation/optimus/urdf/half-leg/constraints.json
- Applies joint constraints each step; drives one joint with a sine wave. Close viewer to exit.
"""

import json
import os
import re
import time

import numpy as np
import mujoco
from mujoco import viewer

MUJUCO_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MJCF_PATH = os.path.join(MUJUCO_DIR, "optimus", "mjcf", "half-leg", "robot.mujoco.xml")
URDF_HALF_LEG_DIR = os.path.join(MUJUCO_DIR, "optimus", "urdf", "half-leg")
CONSTRAINTS_PATH = os.path.join(URDF_HALF_LEG_DIR, "constraints.json")
MESHES_DIR = os.path.join(URDF_HALF_LEG_DIR, "meshes")


def load_constraints(path: str) -> list[dict]:
    """Load constraints.json; each item has sourceJointId, targetJointId, factor, offset."""
    with open(path) as f:
        data = json.load(f)
    return data if isinstance(data, list) else []


def _mesh_file_for_mujoco(meshes_abs: str, filename: str) -> str:
    """Resolve mesh path: use .stl (MuJoCo does not support .step)."""
    base, ext = os.path.splitext(filename)
    if ext.lower() in (".step", ".stp"):
        filename = base + ".stl"
    return os.path.join(meshes_abs, filename).replace(os.sep, "/")


def load_mjcf_with_urdf_meshes(mjcf_path: str, meshes_dir: str) -> str:
    """
    Read MJCF XML and rewrite mesh file= paths to meshes_dir, using .stl instead of .step.
    MuJoCo from_xml_string() has no directory context, so we use absolute paths.
    """
    meshes_abs = os.path.abspath(meshes_dir)
    with open(mjcf_path, encoding="utf-8") as f:
        xml = f.read()
    # Replace file="meshes/XXX" with file="<abs>/meshes_dir/XXX" and .step -> .stl
    def replace_mesh(match: re.Match) -> str:
        path = _mesh_file_for_mujoco(meshes_abs, match.group(1))
        return f'file="{path}"'
    xml = re.sub(r'file="meshes/([^"]+)"', replace_mesh, xml)
    # MuJoCo schema does not allow scale on geom; remove it (1 1 1 = no scaling)
    xml = re.sub(r'\s+scale="[^"]*"', "", xml)
    return xml


def main() -> None:
    if not os.path.isfile(MJCF_PATH):
        raise FileNotFoundError(f"MJCF not found: {MJCF_PATH}")
    if not os.path.isdir(URDF_HALF_LEG_DIR):
        raise FileNotFoundError(f"URDF half-leg dir not found: {URDF_HALF_LEG_DIR}")
    if not os.path.isfile(CONSTRAINTS_PATH):
        raise FileNotFoundError(f"Constraints not found: {CONSTRAINTS_PATH}")
    if not os.path.isdir(MESHES_DIR):
        raise FileNotFoundError(f"Meshes dir not found: {MESHES_DIR}")

    # Require STL meshes (MuJoCo does not load .step)
    meshes_abs = os.path.abspath(MESHES_DIR)
    with open(MJCF_PATH, encoding="utf-8") as f:
        mjcf_text = f.read()
    mesh_refs = re.findall(r'file="meshes/([^"]+)"', mjcf_text)
    required_stl = set()
    for ref in mesh_refs:
        base, ext = os.path.splitext(ref)
        if ext.lower() in (".step", ".stp"):
            required_stl.add(base + ".stl")
        else:
            required_stl.add(ref)
    missing = sorted(f for f in required_stl if not os.path.isfile(os.path.join(meshes_abs, f)))
    if missing:
        raise FileNotFoundError(
            f"MuJoCo requires STL meshes. Missing in {MESHES_DIR}:\n  " + ", ".join(missing)
            + "\nConvert STEP to STL (e.g. in CAD or with a converter) and place the .stl files there."
        )

    xml_str = load_mjcf_with_urdf_meshes(MJCF_PATH, MESHES_DIR)
    print(f"Loading MJCF from {MJCF_PATH} (meshes from {MESHES_DIR})")
    model = mujoco.MjModel.from_xml_string(xml_str)
    data = mujoco.MjData(model)

    constraints = load_constraints(CONSTRAINTS_PATH)

    # Joint name -> qpos address
    joint_name_to_qposadr: dict[str, int] = {}
    for i in range(model.njnt):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i)
        if name:
            joint_name_to_qposadr[name] = model.jnt_qposadr[i]

    # Actuator name (e.g. "joint_Hip_Left_ctrl") -> ctrl index (for matching ctrl to qpos when driving)
    actuator_name_to_idx: dict[str, int] = {}
    for i in range(model.nu):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
        if name:
            actuator_name_to_idx[name] = i

    # Parse constraints: use sourceJointId / targetJointId as joint names (half-leg uses joint_X directly)
    parsed: list[tuple[int, int, float, float, str | None]] = []
    for c in constraints:
        src_id = c.get("sourceJointId") or c.get("sourceJoint")
        tgt_id = c.get("targetJointId") or c.get("targetJoint")
        factor = float(c.get("factor", 1.0))
        offset = float(c.get("offset", 0.0))
        src_name = str(src_id) if src_id is not None else None
        tgt_name = str(tgt_id) if tgt_id is not None else None
        if src_name not in joint_name_to_qposadr or (tgt_name and tgt_name not in joint_name_to_qposadr):
            continue
        parsed.append(
            (
                joint_name_to_qposadr[src_name],
                joint_name_to_qposadr[tgt_name],
                factor,
                offset,
                tgt_name,
            )
        )

    # Joint to drive with sine: prefer first constraint source, else first joint in model
    drive_joint_name: str | None = None
    if constraints:
        src_id = constraints[0].get("sourceJointId") or constraints[0].get("sourceJoint")
        if src_id and str(src_id) in joint_name_to_qposadr:
            drive_joint_name = str(src_id)
    if drive_joint_name is None and joint_name_to_qposadr:
        drive_joint_name = next(iter(joint_name_to_qposadr))
        print(f"Note: Driving first joint: {drive_joint_name}")
    if drive_joint_name is None:
        raise RuntimeError("Model has no joints.")
    drive_qposadr = joint_name_to_qposadr[drive_joint_name]
    drive_ctrl_idx = actuator_name_to_idx.get(f"{drive_joint_name}_ctrl")

    launch = viewer.launch_passive(model, data, key_callback=None)
    print("Half-leg model loaded. Close viewer window to exit.")

    t0 = time.time()
    try:
        while launch.is_running():
            t = time.time() - t0
            drive_val = 0.5 * np.sin(0.5 * t)

            # Drive joint and apply constraints via qpos (and ctrl so motors track)
            data.qpos[drive_qposadr] = drive_val
            if drive_ctrl_idx is not None:
                data.ctrl[drive_ctrl_idx] = drive_val

            for source_qposadr, target_qposadr, factor, offset, tgt_name in parsed:
                data.qpos[target_qposadr] = factor * data.qpos[source_qposadr] + offset
                ctrl_idx = actuator_name_to_idx.get(f"{tgt_name}_ctrl") if tgt_name else None
                if ctrl_idx is not None:
                    data.ctrl[ctrl_idx] = data.qpos[target_qposadr]

            mujoco.mj_step(model, data)
            launch.sync()
            time.sleep(0.001)
    except Exception as e:
        print(f"Simulation error: {e}")
        raise
    finally:
        launch.close()


if __name__ == "__main__":
    main()
