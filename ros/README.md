# Driving with the BC model: simulator and real car

`bc_drive_node.py` is a ROS 2 node that drives the car with the trained steering MLP.
It runs against `f1tenth_gym_ros` (the sim, watched in Foxglove) or on the real car, where
`bc_drive_launch.py` puts it behind the milestone 4 safety stack (AEB).

```
/scan ──────────────────┐
                        ├─► make_observation ─► MLP ─► steering ─► drive command (steering + fixed 1.0 m/s)
speed, yaw rate ────────┘
   sim: /ego_racecar/odom
   car: -(/odom linear.x), /sensors/imu/raw z × π/180   (rl_driving/car_signals.py)
```

| file | what it is |
|---|---|
| `bc_drive_node.py` | the node: every 26 ms builds the observation with `rl_driving.preprocess.make_observation` and publishes the MLP's steering. Parameter `mode`: `sim` or `car` |
| `bc_drive_launch.py` | **for the car** (and sim dry runs): milestone4 `safety_node` + `drive_arbiter` (+ `lap_count` on the car), the BC node, and a rosbag recording |
| `sim_ims_small.yaml` | sim config: ims_small oval, lidar set to match the real car (1081 beams, ±135°, 30 m) |
| `sim_left_loop.yaml` | sim config: `maps/left_loop`, a generated track with only left turns (5 corners of different angles, 1.8 m wide, 58 m lap) |
| `sim_ims.yaml` | sim config: IMS_map, a large oval (left turns only, gentle corners, ~5 min per lap) |
| `sim_levine.yaml` | sim config: levine corridor loop (harder: 90° corners, side openings, right turns) |
| `maps/left_loop.png/.yaml` | made by `scripts/make_left_loop_map.py` (edit the polygon there to make other left-turn tracks) |

The node loads `checkpoints/bc_v1.npz`, a numpy copy of the trained model. The ROS Python has no
torch, so the weights are exported once from the `.pt` in the project venv:

```bash
cd ~/sim_ws/src/rl-driving
.venv/bin/python scripts/export_policy.py      # checkpoints/bc_v1.pt -> bc_v1.npz, checks numpy == torch
```

Re-run this after every retraining.

## Run it in the sim (two terminals)

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

**Terminal 2, the BC driver.** Either the node alone (publishes straight to `/drive`, no AEB):
```bash
python3 ros/bc_drive_node.py
```
or the full car stack against the sim, as a dry run of the real car (AEB on):
```bash
ros2 launch ros/bc_drive_launch.py mode:=sim record:=false
```
The car starts driving. Every 2 s the node prints speed, yaw rate, steering, distance driven
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
| `sim_left_loop.yaml` (new track, left turns only) | **drives cleanly**: 127 m (~2.2 laps) in 130 s, no collisions, 0.51–0.6 m from the wall |
| `sim_ims.yaml` (large oval, left turns only) | **drives cleanly**: 146 m (half a lap) in 150 s, no collisions, ~0.54 m from the wall |
| ims_small clockwise (`stheta: 3.14159`, inner wall on the right) | crashes at the sharp tip of the inner wall after ~10 m |
| `sim_levine.yaml` (clockwise) | straight corridors are fine; crashes at the first sharp right turn around a corner, and later turns into a dead end on the right |

Why the right turns fail: the real track was driven as a loop of left turns. In the training
data 23.7% of rows are hard left turns (yaw rate > 0.5 rad/s) and 0.4% are hard right turns, and
the -40° beam the expert relies on never read more than 1.27 m. At a sharp convex corner that beam
suddenly reads several metres, far outside anything the model saw, so its output is unreliable.
This is a data-coverage limit of BC, not a bug in the node: more varied data (or RL fine-tuning
in the sim) is what fixes it. On new tracks with only left turns (left_loop, IMS) it drives
cleanly, so it generalises to new track shapes as long as the turns are the kind it was trained on.

With the full stack (`bc_drive_launch.py mode:=sim`, checked 2026-11-16):

| config | result |
|---|---|
| `sim_left_loop.yaml` | drives cleanly, AEB never fires during normal driving |
| `sim_levine.yaml` | **AEB stops the car** at the first right turn (min TTC 0.697 s), 0.4 m before the spot where it crashes without AEB; no collision |
| `sim_left_loop.yaml`, `mode:=car` with a test relay that fakes the car's sensors (odometry sign-flipped, IMU in deg/s) | drives cleanly (86 m, no collisions); the node reads +1.00 m/s, so the conversions are right |

## Driving the real car

The car is driven through the milestone 4 safety stack, exactly like `milestone4_wf_py.py`,
with the BC node in wall_follow's place:

```
bc_drive_node ──/wall_follow_drive──► drive_arbiter ──/drive──► car
safety_node ────/safety_drive──────►      ▲           (once AEB fires, the arbiter only sends stops)
lap_count ──────/safety_drive─────────────┘           (stops the car after target_laps)
```

The node publishes on `/wall_follow_drive` because the milestone 4 arbiter already forwards that
topic, so the milestone code needs no change. **Never run wall_follow at the same time.**
In `mode:=car` the node refuses to publish straight to `/drive` (that would bypass the AEB).

### Once: set up the Jetson
1. Get this repo onto the Jetson (`git pull` on the `real-driving` branch, or copy it), at
   `~/sim_ws/src/rl-driving`. The launch file finds everything relative to the repo.
2. Copy the model. `checkpoints/` is gitignored, so it is not in git:
   ```bash
   # on the laptop
   scp checkpoints/bc_v1.npz <jetson-user>@<jetson-ip>:~/sim_ws/src/rl-driving/checkpoints/
   ```
3. Build milestone 4 if it is not built yet: `cd ~/sim_ws && colcon build --packages-select milestone4`.
4. Check that the ROS Python has numpy: `python3 -c "import numpy; print(numpy.__version__)"`.
   (The node is tested with numpy 1.21 and 2.2; torch is not needed.)

### Every time
1. Start the car's drivers (lidar, VESC/odometry, IMU) the way you do for milestone 4.
   Check the topics exist: `ros2 topic list | grep -E "^/scan|^/odom|/sensors/imu/raw"`.
2. Source ROS and the workspace, then from `~/sim_ws/src/rl-driving`:
   ```bash
   ros2 launch ros/bc_drive_launch.py mode:=car
   ```
   Options: `speed:=1.0` (keep 1.0: the model was trained at that speed), `target_laps:=1`,
   `record:=false`, `log_dir:=~/bc_logs`.
3. The node waits until scan, odometry and IMU have all arrived (it logs `waiting for: ...`),
   then drives. A bag of `/scan /odom /sensors/imu/raw /wall_follow_drive /drive /safety_drive`
   goes to `~/bc_logs/bc_run_<time>/`.
4. After an AEB stop the arbiter keeps the car stopped; Ctrl-C and relaunch to drive again.

### Checks before the first drive (wheels off the ground)
- `ros2 topic hz /wall_follow_drive` shows ~38 Hz.
- Hold a board close on the car's **right**: the logged steer goes **positive** (left, away from it).
- Turn the car **left** by hand: the logged yaw rate is **positive**, roughly in rad/s (not 50+).
- On the first drive: the logged speed is about **+1.00** m/s. If it reads -1.00, the odometry
  sign is not what the data showed; stop and check before trusting the model.

### Risks
- **Only left turns are safe.** The model was trained on a left-turn loop and crashes at sharp
  right turns in the sim. Drive the track the data was recorded on, in the same direction.
- **No obstacles.** The model never learned to avoid them; the AEB is the only protection.
- **Actuation delay.** The real car reacts ~100 ms later than the sim, so expect more weaving.
- **The AEB only looks ahead.** It works on time-to-collision, so a slow drift into a side wall
  may not trigger it. Keep the physical kill switch / deadman in hand.

## Notes
- **Lidar config matters.** The model needs 1081 beams; the default `f1tenth_gym_ros/config/sim.yaml`
  uses 819. With the wrong count the node logs an error and does not drive.
- **Why a 26 ms timer** instead of reacting to each scan: the sim publishes scans every 4 ms
  (repeating the same scan between 10 ms physics steps). The model was trained on scan pairs ~26 ms
  apart (the real lidar's rate), and the timer keeps "previous scan" meaning the same thing.
- **Sim vs car sensors.** Sim odometry has standard signs and rad/s. On the real car odometry
  `linear.x` is sign-flipped and the IMU yaw rate is in deg/s (see CLAUDE.md). `mode:=car` converts
  both with `rl_driving/car_signals.py`, the same functions `cleaning.py` used to build the
  training data.
- The node runs with the ROS Python (`~/sim_ws/.venv`, Python 3.10) and imports `rl_driving` from
  this repo's `src/` folder; only numpy is needed.
