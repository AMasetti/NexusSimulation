# MuJoCo

MuJoCo experiments and tests for the biped model.

## Quick run

```bash
cd mujuco
pip install -r requirements.txt
# On macOS the viewer requires mjpython:
mjpython run_sim.py
# On Linux you can use: python run_sim.py
```

This loads `urdf/servo-forge-export-constrained/robot.urdf` (with `meshes/` and `constraints.json`), drives one hip joint with a slow sine wave, applies the constraint so the other hip mirrors it, and opens the MuJoCo viewer. Close the viewer window to exit.

## URDF to MuJoCo XML

To convert a URDF to MuJoCo’s native MJCF XML (e.g. for editing or to use with `run_sim.py`’s preferred `myrobot.mujoco.xml`):

```bash
urdf2mjcf robot.urdf --output myrobot.mujoco.xml
```

Run this from the directory that contains `robot.urdf` and your `meshes/` folder so relative mesh paths in the URDF resolve. The generated `myrobot.mujoco.xml` can be placed in `urdf/servo-forge-export-constrained/`; the sim will load it automatically when present.

Install `urdf2mjcf` if needed: `pip install urdf2mjcf`.

## urdf/servo-forge-export-constrained

- **robot.urdf** – URDF used by the sim; mesh paths are relative to this folder (`meshes/*.stl`).
- **myrobot.mujoco.xml** – (Optional) MuJoCo XML; if present, the sim loads this instead of converting the URDF.
- **constraints.json** – Constraint definitions (source/target joints, factor, offset).

Constraints are applied in Python each step: target joint’s actuator control = factor × source joint qpos + offset, so you can simulate movement with the mirroring constraint.

## Half-leg sim (robot.mujoco.xml + URDF/half-leg)

```bash
# MuJoCo does not load STEP meshes. Convert STEP → STL first:
pip install -r requirements-step2stl.txt   # or: pip install cadquery
python step_to_stl.py                        # converts URDF/half-leg/meshes/*.step → *.stl

mjpython run_sim_half_leg.py
```

`run_sim_half_leg.py` loads `Mujuco XML/half-leg/robot.mujoco.xml`, resolves meshes from `URDF/half-leg/meshes/`, and applies `URDF/half-leg/constraints.json`. The converter writes STL files next to each STEP (same base name).

## SpotMicro sim (spotmicro.xml)

```bash
cd mujuco
pip install -r requirements.txt
mjpython run_sim_spotmicro.py   # macOS; on Linux: python run_sim_spotmicro.py
```

Loads `Mujuco XML/SpotMicro/spotmicro.xml`, resolves mesh paths from `URDF/spotmicro_description/meshes/stl/`, and opens the MuJoCo viewer. Close the viewer window to exit.

## RL locomotion (SpotMicro)

Train a policy to move the SpotMicro forward using reinforcement learning (PPO or SAC from Stable-Baselines3).

**Install dependencies** (includes gymnasium, stable-baselines3, torch):

```bash
cd mujuco
pip install -r requirements.txt
```

**Train** (example: PPO for 1M steps, save to `./logs/spotmicro_ppo`):

```bash
python train_spotmicro.py --algo ppo --total_timesteps 1000000 --save_path ./logs/spotmicro_ppo
```

Options: `--algo sac`, `--max_episode_steps 60000`, `--seed 0`. The Gymnasium env is in `spotmicro_env.py`; reward is forward velocity with small penalties for tilt, height error, and action magnitude. Episodes terminate when the base falls (low height or large tilt) or after `max_episode_steps`.

**Run a saved policy with the viewer**: load the model with Stable-Baselines3 (`PPO.load("path")`), create `SpotMicroEnv(render_mode="human")`, and step with `model.predict(obs)` in a loop until the episode ends. A **Jupyter notebook** that trains then runs the walking spot is in `spotmicro_train_and_run.ipynb`—run all cells to train and then watch the policy in the MuJoCo viewer.

## Viewer: only seeing boxes instead of STL meshes?

MuJoCo loads both **visual** (mesh) and **collision** (e.g. box) geometry from the URDF and puts them in different geom groups. If you only see simple shapes, press **1** or **2** in the viewer window to toggle geom groups—the STL visuals may be in another group. The script also injects `compiler meshdir` and `discardvisual="false"` so mesh paths resolve and visual geometry is kept (see [MuJoCo URDF issues](https://github.com/google-deepmind/mujoco/issues/1569)).
