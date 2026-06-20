"""
Shared loader for the SpotMicro MuJoCo model.
Returns XML string with meshes resolved, scale/inertial/floor fixes applied.
Used by run_sim_spotmicro.py and spotmicro_env.py.
"""

import os
import re

MUJUCO_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MJCF_PATH = os.path.join(MUJUCO_DIR, "spot_micro", "mjcf", "spotmicro.xml")
MESHES_DIR = os.path.join(MUJUCO_DIR, "spot_micro", "urdf", "spotmicro_description", "meshes", "stl")


def load_spotmicro_xml() -> str:
    """
    Load SpotMicro MJCF and return XML string ready for MjModel.from_xml_string().
    - Resolves package:// mesh paths to MESHES_DIR
    - Strips scale from geom, adds scale to mesh assets
    - Adds pos to inertial elements
    - Injects ground plane
    """
    if not os.path.isfile(MJCF_PATH):
        raise FileNotFoundError(f"MJCF not found: {MJCF_PATH}")
    if not os.path.isdir(MESHES_DIR):
        raise FileNotFoundError(f"Meshes dir not found: {MESHES_DIR}")

    meshes_abs = os.path.abspath(MESHES_DIR)
    with open(MJCF_PATH, encoding="utf-8") as f:
        xml = f.read()

    # package://spotmicro_description/meshes/stl/XXX -> absolute path
    pattern = re.compile(
        r'file="package://spotmicro_description/meshes/stl/([^"]+)"'
    )
    xml = pattern.sub(
        lambda m: f'file="{os.path.join(meshes_abs, m.group(1)).replace(os.sep, "/")}"',
        xml,
    )
    # Strip scale from geom (MuJoCo schema does not allow scale on geom in this version)
    xml = re.sub(r'\s+scale="[^"]*"', "", xml)
    # Add scale to mesh assets (STL are in mm; 0.001 -> meters)
    xml = re.sub(
        r'(<mesh\s+name="[^"]+"\s+file="[^"]+")\s*/>',
        r'\1 scale="0.001 0.001 0.001" />',
        xml,
    )
    # MuJoCo requires pos on inertial; add if missing
    xml = re.sub(r'<inertial\s+mass=', r'<inertial pos="0 0 0" mass=', xml)
    # Add ground plane (model has no floor)
    # High-friction floor: condim=6 (full contact with torsional grip),
    # friction=[1.5, 0.5, 0.5] so the toe's priority=1 friction wins on the
    # robot side but the floor provides solid traction regardless.
    xml = xml.replace(
        "<worldbody>",
        '<worldbody>\n    <geom name="floor" type="plane" pos="0 0 0" size="10 10 0.1"'
        ' condim="6" friction="1.5 0.5 0.5" />',
    )
    return xml
