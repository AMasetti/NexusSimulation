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
mjpython optimus/run_sim.py          # macOS
python   optimus/run_sim.py          # Linux
```

Loads `optimus/mjcf/optimus.mujoco.xml` and opens the MuJoCo interactive viewer.

### Optimus — half-leg prototype

The half-leg model uses STEP meshes that must be converted to STL before first run:

```bash
pip install -r optimus/requirements-step2stl.txt   # installs cadquery
python optimus/step_to_stl.py                       # converts optimus/urdf/half-leg/meshes/*.step → *.stl

mjpython optimus/run_sim_half_leg.py
```

### SpotMicro — viewer

```bash
mjpython spot_micro/run_sim_spotmicro.py   # macOS
python   spot_micro/run_sim_spotmicro.py   # Linux
```

Loads `spot_micro/mjcf/spotmicro.xml` with STL meshes from `spot_micro/urdf/spotmicro_description/meshes/stl/`.

### SpotMicro — RL training (PPO / SAC)

```bash
# Train — PPO for 1M steps
python spot_micro/train_spotmicro.py --algo ppo --total_timesteps 1000000 --save_path ./logs/spotmicro_ppo

# Train — SAC
python spot_micro/train_spotmicro.py --algo sac --total_timesteps 1000000 --save_path ./logs/spotmicro_sac
```

The Gym environment is in `spot_micro/spotmicro_env.py`. Reward: forward velocity with penalties for tilt, height error, and action magnitude. Episodes end on fall or `max_episode_steps`.

A notebook that trains and then runs the policy in the viewer is available at `spot_micro/spotmicro_train_and_run.ipynb`.

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
│   ├── urdf/
│   │   ├── full/                     # URDF + meshes for full Optimus model
│   │   └── half-leg/                 # URDF + STEP meshes for half-leg prototype
│   ├── run_sim.py                    # Optimus full-body viewer
│   ├── run_sim_half_leg.py           # Optimus half-leg viewer
│   ├── step_to_stl.py                # STEP → STL converter (half-leg meshes)
│   └── requirements-step2stl.txt
├── spot_micro/
│   ├── mjcf/
│   │   └── spotmicro.xml            # SpotMicro MJCF
│   ├── urdf/
│   │   └── spotmicro_description/   # URDF + STL meshes for SpotMicro
│   ├── run_sim_spotmicro.py         # SpotMicro viewer
│   ├── spotmicro_loader.py          # Shared SpotMicro model loader
│   ├── spotmicro_env.py             # Gymnasium environment for SpotMicro RL
│   ├── spotmicro_traj_env.py        # Trajectory-following variant
│   ├── spotmicro_walk.py            # Run trained policy in viewer
│   ├── train_spotmicro.py           # PPO / SAC training script
│   ├── train_spotmicro_traj.py      # Trajectory-imitation training
│   ├── eval_spotmicro.py            # Policy evaluation / benchmarking
│   ├── test_spot_micro.py           # Diagnostic / sanity checks
│   └── spotmicro_train_and_run.ipynb
├── kinematics/                      # SpotMicro kinematics demos and firmware prototypes
├── logs/                            # Trained model checkpoints
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
