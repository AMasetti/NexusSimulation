#!/usr/bin/env python3
"""
MuJoCo simulation for the SpotMicro quadruped.
- Loads spotmicro.xml via spotmicro_loader
- Launches the MuJoCo viewer. Close the viewer window to exit.
"""

import time

import mujoco
from mujoco import viewer

from spotmicro_loader import MJCF_PATH, MESHES_DIR, load_spotmicro_xml


def main() -> None:
    xml_str = load_spotmicro_xml()
    print(f"Loading SpotMicro from {MJCF_PATH} (meshes from {MESHES_DIR})")
    model = mujoco.MjModel.from_xml_string(xml_str)
    data = mujoco.MjData(model)
    model.opt.timestep = 0.02  # 0.02 s per step (0.2 caused instant fall)

    launch = viewer.launch_passive(model, data, key_callback=None)
    launch.opt.geomgroup[1] = 0  # Hide collision boxes; show only visuals (group 2)
    print("SpotMicro model loaded. Close viewer window to exit.")

    try:
        while launch.is_running():
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
