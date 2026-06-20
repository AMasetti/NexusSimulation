#!/usr/bin/env python3
"""
Simple MuJoCo simulation for the Servo Forge constrained robot.
- Prefers myrobot.mujoco.xml when present in urdf/servo-forge-export-constrained; else loads robot.urdf and converts to MJCF.
- Loads constraints.json from the same directory.
- Applies joint constraints each step (target_angle = factor * source_angle + offset).
- Drives the source joint with a slow sine wave so you see movement.
- Launches the passive viewer; close the window to exit.
"""

import json
import os
import re
import time
import xml.etree.ElementTree as ET

import numpy as np
import mujoco
from mujoco import viewer

# Paths: prefer myrobot.mujoco.xml, else URDF + meshes in mujuco/urdf/... or repo urdf/...
MUJUCO_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(MUJUCO_DIR, ".."))
_MODEL_CANDIDATES = (
    os.path.join(MUJUCO_DIR, "urdf", "servo-forge-export-constrained"),
    os.path.join(REPO_ROOT, "urdf", "servo-forge-export-constrained"),
    os.path.join(MUJUCO_DIR, "servo-forge-export-constrained"),
)
_MJCF_FILENAME = "myrobot.mujoco.xml"
_MJCF_CANDIDATES = tuple(os.path.join(d, _MJCF_FILENAME) for d in _MODEL_CANDIDATES)
MJCF_PATH = next((p for p in _MJCF_CANDIDATES if os.path.isfile(p)), None)
MODEL_DIR = next((d for d in _MODEL_CANDIDATES if os.path.isdir(d)), _MODEL_CANDIDATES[0])
MODEL_PATH = os.path.join(MODEL_DIR, "robot.urdf")
CONSTRAINTS_PATH = os.path.join(MODEL_DIR, "constraints.json")


def load_constraints(path: str) -> list[dict]:
    """Load constraints.json; each item has sourceJointId, targetJointId, factor, offset."""
    with open(path) as f:
        data = json.load(f)
    return data if isinstance(data, list) else []


def _parse_xyz(s: str | None, scale_mm: bool = True, flip_y: bool = False) -> str:
    """Parse 'x y z' string; if scale_mm and values look like mm (any |v|>=1), scale by 0.001.
    flip_y: negate Y (use when URDF is ROS/servo-forge and pivot appears on wrong side of parent).
    """
    if not s:
        return "0 0 0"
    parts = s.strip().split()
    if len(parts) != 3:
        return "0 0 0"
    try:
        x, y, z = float(parts[0]), float(parts[1]), float(parts[2])
    except ValueError:
        return "0 0 0"
    if scale_mm and (abs(x) >= 1 or abs(y) >= 1 or abs(z) >= 1):
        x, y, z = x * 0.001, y * 0.001, z * 0.001
    if flip_y:
        y = -y
    return f"{x} {y} {z}"


def _parse_rpy(s: str | None) -> str:
    """Parse URDF 'rpy' (roll pitch yaw in radians) to 'r p y' for MuJoCo euler (xyz order)."""
    if not s:
        return "0 0 0"
    parts = s.strip().split()
    if len(parts) != 3:
        return "0 0 0"
    try:
        r, p, y = float(parts[0]), float(parts[1]), float(parts[2])
    except ValueError:
        return "0 0 0"
    return f"{r} {p} {y}"


def _get_rgba(visual_elem: ET.Element) -> str:
    """Extract rgba from visual/material/color or visual/color; default grey."""
    color = visual_elem.find(".//color")
    if color is not None and color.get("rgba"):
        return color.get("rgba", "0.5 0.5 0.5 1")
    return "0.5 0.5 0.5 1"


def _mesh_name_from_file(filename: str) -> str:
    """MuJoCo-safe asset name from mesh filename (e.g. meshes/foo.stl -> foo_mesh)."""
    base = os.path.basename(filename).rsplit(".", 1)[0]
    safe = re.sub(r"[^a-zA-Z0-9_]", "_", base)
    return f"{safe}_mesh" if safe else "mesh"


def _urdf_find(elem: ET.Element, tag: str) -> ET.Element | None:
    """Find child by tag, with or without XML namespace (e.g. ROS URDF xmlns)."""
    found = elem.find(tag)
    if found is not None:
        return found
    if elem.tag.startswith("{"):
        ns = elem.tag[1 : elem.tag.index("}")]
        return elem.find(f"{{{ns}}}{tag}")
    return None


def _urdf_findall(elem: ET.Element, tag: str) -> list[ET.Element]:
    """Find all children by tag, with or without XML namespace."""
    found = elem.findall(tag)
    if found:
        return found
    if elem.tag.startswith("{"):
        ns = elem.tag[1 : elem.tag.index("}")]
        return elem.findall(f"{{{ns}}}{tag}")
    return []


def urdf_to_mjcf(urdf_path: str, meshes_dir: str, mesh_path_prefix: str = "") -> str:
    """Convert URDF to MJCF string so MuJoCo loads mesh visuals. Writes for loading from meshes_dir so relative mesh paths work.
    Set env MUJOCO_URDF_FLIP_Y=1 to negate Y in positions (pivot on other side) if axis appears offset.
    """
    flip_y = os.environ.get("MUJOCO_URDF_FLIP_Y", "").strip() in ("1", "true", "yes")
    tree = ET.parse(urdf_path)
    root = tree.getroot()
    links = {el.get("name"): el for el in _urdf_findall(root, "link")}
    joints = _urdf_findall(root, "joint")
    # Root is the link that is never a child of any joint (base_link in typical URDF).
    child_links = set()
    for j in joints:
        c = _urdf_find(j, "child")
        if c is not None and c.get("link"):
            child_links.add(c.get("link"))
    root_link_name = next((n for n in links if n not in child_links), None)
    if not root_link_name:
        root_link_name = list(links)[0]

    mesh_assets: list[tuple[str, str]] = []  # (asset_name, file_path)
    link_meshes: dict[str, list[tuple[str, str, str, str]]] = {}  # link_name -> [(mesh_asset_name, pos_xyz, rgba, rpy)]

    for link_name, link_el in links.items():
        link_meshes[link_name] = []
        for visual in _urdf_findall(link_el, "visual"):
            geom = _urdf_find(visual, "geometry")
            if geom is None:
                continue
            mesh_el = _urdf_find(geom, "mesh")
            if mesh_el is None:
                continue
            filename = mesh_el.get("filename") or ""
            if not filename:
                continue
            base = os.path.basename(filename)
            if not base:
                continue
            asset_name = _mesh_name_from_file(filename)
            file_path = mesh_path_prefix + base
            if not any(a[0] == asset_name for a in mesh_assets):
                mesh_assets.append((asset_name, file_path))
            origin = _urdf_find(visual, "origin")
            xyz = _parse_xyz(origin.get("xyz") if origin is not None else None, flip_y=flip_y)
            rpy = _parse_rpy(origin.get("rpy") if origin is not None else None)
            rgba = _get_rgba(visual)
            link_meshes[link_name].append((asset_name, xyz, rgba, rpy))

    asset_lines = "\n".join(f'    <mesh name="{name}" file="{path}"/>' for name, path in mesh_assets)
    if not asset_lines:
        asset_lines = "    <!-- no meshes -->"

    def emit_body(link_name: str, indent: str) -> str:
        link_el = links.get(link_name)
        lines = []
        body_indent = indent + "  "
        # Joints that have this link as parent -> child bodies
        child_joints = [j for j in joints if _urdf_find(j, "parent") is not None and _urdf_find(j, "parent").get("link") == link_name]
        pos = "0 0 0"
        for j in child_joints:
            origin = _urdf_find(j, "origin")
            xyz = _parse_xyz(origin.get("xyz") if origin is not None else None, flip_y=flip_y)
            rpy = _parse_rpy(origin.get("rpy") if origin is not None else None)
            axis_el = _urdf_find(j, "axis")
            axis = (axis_el.get("xyz") or "0 0 1").strip()
            lim = _urdf_find(j, "limit")
            lower = float(lim.get("lower", -3.14159)) if lim is not None else -3.14159
            upper = float(lim.get("upper", 3.14159)) if lim is not None else 3.14159
            jname = j.get("name", "joint")
            jtype = j.get("type", "revolute")
            child_link = _urdf_find(j, "child")
            if child_link is None:
                continue
            cname = child_link.get("link", "")
            if not cname or cname not in links:
                continue
            # Body at joint origin; hinge pivot at body origin (joint pos 0 0 0) so axis is fixed to parent as in URDF.
            body_attrs = f'name="{cname}" pos="{xyz}"'
            if rpy and rpy != "0 0 0":
                body_attrs += f' euler="{rpy}"'
            lines.append(f'{indent}<body {body_attrs}>')
            if jtype == "revolute" or jtype == "continuous":
                lines.append(f'{body_indent}<joint name="{jname}" type="hinge" pos="0 0 0" axis="{axis}" range="{lower} {upper}"/>')
            for i, (mesh_name, gpos, rgba, geom_rpy) in enumerate(link_meshes.get(cname, [])):
                geom_name = f"{cname}_geom_{i}"
                geom_attrs = f'name="{geom_name}" type="mesh" mesh="{mesh_name}" pos="{gpos}" rgba="{rgba}" contype="0" conaffinity="0"'
                if geom_rpy and geom_rpy != "0 0 0":
                    geom_attrs += f' euler="{geom_rpy}"'
                lines.append(f'{body_indent}<geom {geom_attrs}/>')
            lines.append(emit_body(cname, body_indent))
            lines.append(f"{indent}</body>")
        return "\n".join(lines)

    root_geoms = []
    for i, (mesh_name, gpos, rgba, geom_rpy) in enumerate(link_meshes.get(root_link_name, [])):
        geom_attrs = f'name="{root_link_name}_geom_{i}" type="mesh" mesh="{mesh_name}" pos="{gpos}" rgba="{rgba}" contype="0" conaffinity="0"'
        if geom_rpy and geom_rpy != "0 0 0":
            geom_attrs += f' euler="{geom_rpy}"'
        root_geoms.append(f'      <geom {geom_attrs}/>')
    root_geom_block = "\n".join(root_geoms) if root_geoms else "      <!-- no mesh -->"
    child_bodies = emit_body(root_link_name, "    ")

    return f'''<mujoco model="robot">
  <compiler angle="radian"/>
  <option gravity="0 0 -9.81"/>
  <asset>
{asset_lines}
  </asset>
  <worldbody>
    <body name="{root_link_name}" pos="0 0 0">
{root_geom_block}
{child_bodies}
    </body>
  </worldbody>
</mujoco>
'''


def main() -> None:
    # Prefer existing myrobot.mujoco.xml (e.g. from repo urdf/servo-forge-export-constrained)
    if MJCF_PATH is not None and os.path.isfile(MJCF_PATH):
        model_dir = os.path.dirname(MJCF_PATH)
        constraints_path = os.path.join(model_dir, "constraints.json")
        if not os.path.isfile(constraints_path):
            raise FileNotFoundError(f"Constraints not found: {constraints_path}")
        print(f"Loading MJCF from {MJCF_PATH}")
        model = mujoco.MjModel.from_xml_path(MJCF_PATH)
        data = mujoco.MjData(model)
        print("Model loaded (myrobot.mujoco.xml). Close viewer window to exit.")
    else:
        if not os.path.isfile(MODEL_PATH):
            raise FileNotFoundError(f"Model not found: {MODEL_PATH}")
        if not os.path.isfile(CONSTRAINTS_PATH):
            raise FileNotFoundError(f"Constraints not found: {CONSTRAINTS_PATH}")
        model_dir = MODEL_DIR
        constraints_path = CONSTRAINTS_PATH
        meshes_dir = os.path.join(model_dir, "meshes")
        # Discover required meshes from URDF (namespace-aware)
        tree = ET.parse(MODEL_PATH)
        urdf_root = tree.getroot()
        if urdf_root.tag.startswith("{"):
            _ns = urdf_root.tag[1 : urdf_root.tag.index("}")]
            _mesh_el = urdf_root.findall(f".//{{{_ns}}}mesh")
        else:
            _mesh_el = urdf_root.findall(".//mesh")
        required_meshes = []
        for mesh in _mesh_el:
            fn = mesh.get("filename") or ""
            if fn:
                base = os.path.basename(fn)
                if base and base not in required_meshes:
                    required_meshes.append(base)
        missing = [m for m in required_meshes if not os.path.isfile(os.path.join(meshes_dir, m))]
        if missing:
            raise FileNotFoundError(
                f"Meshes not found in {meshes_dir}\nMissing: {missing}\n"
                "Add the STL files there (e.g. from Servo Forge export) and run again."
            )
        # Convert URDF -> MJCF programmatically; write XML into meshes_dir so relative mesh paths resolve.
        mjcf_str = urdf_to_mjcf(MODEL_PATH, meshes_dir, mesh_path_prefix="")
        mjcf_path = os.path.join(meshes_dir, "_robot_mjcf.xml")
        with open(mjcf_path, "w", encoding="utf-8") as f:
            f.write(mjcf_str)
        try:
            print(f"Loading model with STL meshes from {model_dir}")
            model = mujoco.MjModel.from_xml_path(mjcf_path)
        finally:
            try:
                os.remove(mjcf_path)
            except OSError:
                pass
        data = mujoco.MjData(model)
        print("Model loaded (STL meshes). Close viewer window to exit.")

    constraints = load_constraints(constraints_path)

    # Map joint name -> qpos address
    joint_name_to_qposadr: dict[str, int] = {}
    for i in range(model.njnt):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i)
        if name:
            jnt_id = i
            qposadr = model.jnt_qposadr[jnt_id]
            joint_name_to_qposadr[name] = qposadr

    # URDF has no actuators (model.nu == 0); we drive by setting qpos each step.

    # Constraint: target joint angle = factor * source joint angle + offset.
    # Our JSON uses "joint_<linkname>"; URDF joint names are HipYAW_L, HipYAW_R.
    def constraint_joint_id_to_name(cid: str) -> str | None:
        if "Hip_Assembly_Servo_mount_1771209237471" in cid:
            return "HipYAW_R"
        if "Hip_Assembly_Servo_mount" in cid and "1771209237471" not in cid:
            return "HipYAW_L"
        return None

    # (source_qposadr, target_qposadr, factor, offset) for qpos-based; target is always qpos
    parsed: list[tuple[int, int, float, float]] = []
    for c in constraints:
        src_id = c.get("sourceJointId") or c.get("sourceJoint")
        tgt_id = c.get("targetJointId") or c.get("targetJoint")
        factor = float(c.get("factor", 1.0))
        offset = float(c.get("offset", 0.0))
        src_name = constraint_joint_id_to_name(src_id) if isinstance(src_id, str) else None
        tgt_name = constraint_joint_id_to_name(tgt_id) if isinstance(tgt_id, str) else None
        if src_name not in joint_name_to_qposadr or (tgt_name and tgt_name not in joint_name_to_qposadr):
            continue
        parsed.append(
            (
                joint_name_to_qposadr[src_name],
                joint_name_to_qposadr[tgt_name],
                factor,
                offset,
            )
        )

    # Joint we drive with a sine: prefer HipYAW_R, else first constraint source joint, else first joint in model
    drive_qposadr = joint_name_to_qposadr.get("HipYAW_R")
    if drive_qposadr is None:
        for c in constraints:
            src_id = c.get("sourceJointId") or c.get("sourceJoint")
            src_name = constraint_joint_id_to_name(src_id) if isinstance(src_id, str) else None
            if src_name and src_name in joint_name_to_qposadr:
                drive_qposadr = joint_name_to_qposadr[src_name]
                break
    if drive_qposadr is None and joint_name_to_qposadr:
        drive_qposadr = next(iter(joint_name_to_qposadr.values()))
        print(f"Note: Using first available joint (model joints: {list(joint_name_to_qposadr)})")
    if drive_qposadr is None:
        raise RuntimeError(
            "Model has no joints. Generated MJCF may have no hinge joints; check URDF and converter."
        )

    launch = viewer.launch_passive(model, data, key_callback=None)

    t0 = time.time()
    try:
        while launch.is_running():
            t = time.time() - t0
            drive_val = 0.5 * np.sin(0.5 * t)

            # Drive and constraints via qpos (URDF has no actuators)
            data.qpos[drive_qposadr] = drive_val
            for source_qposadr, target_qposadr, factor, offset in parsed:
                data.qpos[target_qposadr] = factor * data.qpos[source_qposadr] + offset

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
