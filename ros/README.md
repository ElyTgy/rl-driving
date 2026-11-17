# Driving the BC model in the simulator

`bc_drive_node.py` is a ROS 2 node that drives the car with the trained steering MLP.
It runs against `f1tenth_gym_ros` (the sim) and you watch it in Foxglove.

```
/scan ──────────┐
                ├─► make_observation ─► MLP ─► steering ─► /drive (steering + fixed 1.0 m/s)
/ego_racecar/odom (speed, yaw rate) ──┘
```

| file | what it is |
|---|---|
| `bc_drive_node.py` | the node: every 26 ms builds the observation with `rl_driving.preprocess.make_observation` and publishes the MLP's steering |
| `sim_ims_small.yaml` | sim config: ims_small oval, lidar set to match the real car (1081 beams, ±135°, 30 m) |
| `sim_levine.yaml` | sim config: levine corridor loop (harder: 90° corners, side openings) |

The node loads `checkpoints/bc_v1.npz`, a numpy copy of the trained model. The ROS Python has no
torch, so the weights are exported once from the `.pt` in the project venv:

```bash
cd ~/sim_ws/src/rl-driving
.venv/bin/python scripts/export_policy.py      # checkpoints/bc_v1.pt -> bc_v1.npz, checks numpy == torch
```

Re-run this after every retraining.

## Run it (two terminals)

Both terminals need the ROS environment:

```bash
source /opt/ros/humble/setup.bash
source ~/sim_ws/install/local_setup.bash
source ~/sim_ws/.venv/bin/activate
cd ~/sim_ws/src/rl-driving
```

**Terminal 1, the simulator** (Foxglove opens in the browser):
```bash
ros2 launch f1tenth_gym_ros gym_bridge_launch.py config:=$PWD/ros/sim_ims_small.yaml
```
If the browser does not open, go to
<https://app.foxglove.dev/?ds=foxglove-websocket&ds.url=ws://localhost:8765>.
The first time, import the layout (Layout menu → Import from file):
`~/sim_ws/src/f1tenth_gym_ros/config/foxglove/gym_bridge_foxglove.json`. It shows the map, the
car and the lidar scan. Wait until the terminal prints `Running in asynchronous mode`.

**Terminal 2, the BC driver:**
```bash
python3 ros/bc_drive_node.py
```
The car starts driving. Every 2 s the node prints its position, speed, steering, distance driven
and number of collisions. Ctrl-C stops the node and sends a final "stop" command to the car.

Options (all ROS parameters):
```bash
python3 ros/bc_drive_node.py --ros-args -p speed:=1.0 -p policy_path:=/abs/path/model.npz
```

**Restarting after a crash:** the sim does not reset by itself; the car stays stuck at the wall.
Either restart terminal 1, or use the 3D panel's publish tool in "pose estimate" mode
(it publishes on `/initialpose`): click and drag on the map to place the car and its heading.

## What to expect with bc_v1 (trained on wall_follow data, checked 2026-11-16)

| config | result |
|---|---|
| `sim_ims_small.yaml` (counterclockwise, outer wall on the right) | **drives cleanly**: 3+ laps in 75 s, no collisions, ~0.6 m from the right wall like the expert |
| ims_small clockwise (`stheta: 3.14159`, inner wall on the right) | crashes at the sharp tip of the inner wall after ~10 m |
| `sim_levine.yaml` (clockwise) | straight corridors are fine; crashes at the first sharp right turn around a corner, and later turns into a dead end on the right |

Why the right turns fail: the real track was driven as a loop of left turns. In the training
data 23.7% of rows are hard left turns (yaw rate > 0.5 rad/s) and 0.4% are hard right turns, and
the -40° beam the expert relies on never read more than 1.27 m. At a sharp convex corner that beam
suddenly reads several metres, far outside anything the model saw, so its output is unreliable.
This is a data-coverage limit of BC, not a bug in the node: more varied data (or RL fine-tuning
in the sim) is what fixes it.

## Notes
- **Lidar config matters.** The model needs 1081 beams; the default `f1tenth_gym_ros/config/sim.yaml`
  uses 819. With the wrong count the node logs an error and does not drive.
- **Why a 26 ms timer** instead of reacting to each scan: the sim publishes scans every 4 ms
  (repeating the same scan between 10 ms physics steps). The model was trained on scan pairs ~26 ms
  apart (the real lidar's rate), and the timer keeps "previous scan" meaning the same thing.
- **Sim only for now.** Sim odometry has standard signs and rad/s. On the real car `vx` and
  `wz_odom` are sign-flipped and the IMU yaw rate is in deg/s (see CLAUDE.md), so the node's
  speed and yaw-rate inputs need converting before it can run on the car.
- The node runs with the ROS Python (`~/sim_ws/.venv`, Python 3.10) and imports `rl_driving` from
  this repo's `src/` folder; only numpy is needed.
