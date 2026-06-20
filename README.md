# NexusSimulation

MuJoCo simulation environment for the Nexus Robotics project. Covers two robots:

- **Optimus** — 8-DOF biped (primary development target)
- **SpotMicro** — quadruped (RL locomotion experiments)

Part of the [NexusRobotics](https://github.com/AMasetti/NexusRobotics) monorepo.

---

## Setup

```bash
pip install -r requirements.txt
```

On macOS, use `mjpython` instead of `python` for any script that opens the viewer.

---

## Simulations

### Optimus — full biped

```bash
mjpython run_sim.py          # macOS
python   run_sim.py          # Linux
```

Loads `optimus/mjcf/optimus.mujoco.xml` and opens the MuJoCo interactive viewer.

### Optimus — half-leg prototype

The half-leg model uses STEP meshes that must be converted to STL before first run:

```bash
pip install -r requirements-step2stl.txt   # installs cadquery
python step_to_stl.py                       # converts optimus/urdf/half-leg/meshes/*.step → *.stl

mjpython run_sim_half_leg.py
```

### SpotMicro — viewer

```bash
mjpython run_sim_spotmicro.py   # macOS
python   run_sim_spotmicro.py   # Linux
```

Loads `spot_micro/mjcf/spotmicro.xml` with STL meshes from `spot_micro/urdf/spotmicro_description/meshes/stl/`.

### SpotMicro — RL training (PPO / SAC)

```bash
# Train — PPO for 1M steps
python train_spotmicro.py --algo ppo --total_timesteps 1000000 --save_path ./logs/spotmicro_ppo

# Train — SAC
python train_spotmicro.py --algo sac --total_timesteps 1000000 --save_path ./logs/spotmicro_sac
```

The Gym environment is in `spotmicro_env.py`. Reward: forward velocity with penalties for tilt, height error, and action magnitude. Episodes end on fall or `max_episode_steps`.

A notebook that trains and then runs the policy in the viewer is available at `spotmicro_train_and_run.ipynb`.

---

## File structure

```
NexusSimulation/
├── optimus/
│   ├── mjcf/
│   │   ├── optimus.mujoco.xml        # MJCF model — main Optimus sim
│   │   ├── optimus_mujoco_fixed.xml  # Fixed-base variant
│   │   └── half-leg/
│   │       └── robot.mujoco.xml      # Half-leg prototype MJCF
│   ├── meshes/                       # STL meshes for Optimus body parts
│   └── urdf/
│       ├── full/                     # URDF + meshes for full Optimus model
│       └── half-leg/                 # URDF + STEP meshes for half-leg prototype
├── spot_micro/
│   ├── mjcf/
│   │   └── spotmicro.xml            # SpotMicro MJCF
│   └── urdf/
│       └── spotmicro_description/   # URDF + STL meshes for SpotMicro
├── kinematics/                      # SpotMicro kinematics demos and firmware prototypes
├── logs/                            # Trained model checkpoints
├── run_sim.py                       # Optimus full-body viewer
├── run_sim_half_leg.py              # Optimus half-leg viewer
├── run_sim_spotmicro.py             # SpotMicro viewer
├── spotmicro_env.py                 # Gymnasium environment for SpotMicro RL
├── spotmicro_traj_env.py            # Trajectory-following variant
├── train_spotmicro.py               # PPO / SAC training script
├── spotmicro_loader.py              # Shared SpotMicro model loader
├── step_to_stl.py                   # STEP → STL converter (half-leg meshes)
└── requirements.txt
```

---

## MuJoCo viewer tips

- Press **1** / **2** to toggle geom groups if you only see boxes instead of STL meshes (visual and collision geometry are in separate groups).
- Press **Space** to pause/resume simulation.
- Drag with right mouse button to rotate the camera.

---

## URDF → MJCF conversion

To regenerate a `.mujoco.xml` from a URDF:

```bash
pip install urdf2mjcf
urdf2mjcf robot.urdf --output robot.mujoco.xml
```

Run from the directory containing `robot.urdf` and its `meshes/` folder so relative paths resolve correctly.
