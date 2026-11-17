# Dataset columns

Each run's `dataset/chunk_*.npz` has **one row per lidar scan** (~38 Hz, ~26 ms apart).
The recorder is `../rl-data/milestone3/data_tools/data_recorder.py`. The first rows are NaN until every topic has published.

Load it with:
```python
z = np.load('raw-data/<run>/dataset/chunk_000.npz')
d = {k: z[k] for k in z.files}
```

## Time
| Column | Meaning |
|---|---|
| `t` | Scan time in Unix seconds. Use `t - t[0]` for seconds since start. |

## Lidar (measured)
| Column | Meaning |
|---|---|
| `ranges` | Shape (N, 1081). Distances in m, from -135° (right, beam 0) to +135° (left, beam 1080) in 0.25° steps. Beam 540 = straight ahead. Valid range 0.02–30 m. **65.533 = no return** (treat as far, not an obstacle). |

## State
Two sources: **odometry** is *computed* from the commands with a bicycle model (speed comes from the real motor RPM). The **IMU** is *measured* by a physical sensor.

| Column | Source | Meaning |
|---|---|---|
| `x`, `y` | odom | Position in m. Drifts, and runs backwards (sign-flipped). Not a network input: it's track-specific. |
| `yaw` | odom | Heading (direction faced) in rad. Track-specific, not an input. |
| `vx` | odom | Forward speed in m/s. **Sign-flipped: use `-vx`.** |
| `vy` | odom | Sideways speed. Always 0 (the model assumes no slip). Ignore. |
| `wz_odom` | odom | Yaw rate (how fast the heading changes) in rad/s. Sign-flipped and has a ~0.23 rad/s servo offset. Prefer `wz_imu`. |
| `wz_imu` | IMU | Yaw rate, measured. **Units are deg/s: multiply by π/180.** The sign is correct (+ = turning left). Lags odom by ~100 ms (a hint at the actuation delay). |
| `ax`, `ay` | IMU | Forward and sideways acceleration (probably in g). Checks only. |

`yaw` is an angle. `wz_*` is its rate of change. Don't compute the yaw rate by differencing `yaw`: that just gives back `wz_odom` with extra noise and ±π wrap jumps.

## Commands (all *computed*, not measured)
Steering is an **absolute** wheel angle in rad (+ = left), clipped to ±0.3 by wall_follow. Speed is a target in m/s, not the actual speed.

```
wall_follow ──► + drift ──► arbiter (AEB) ──► /drive ──► VESC ──► wheels
expert_*        applied_*                     drive_*
```

| Column | Meaning |
|---|---|
| `expert_steer` | What the expert (wall_follow) wanted, before drift. **The training label.** During drift it holds the expert's correction. |
| `expert_speed` | The expert's target speed. Always 1.0 for wall_follow, so it's not a label yet. |
| `applied_steer` | `expert_steer + drift_offset`, i.e. what was sent toward the car. Equals expert outside drift. Don't use it as a label: it would teach the network to copy the drift. |
| `applied_speed` | Drift only changes steering, so this always equals `expert_speed`. |
| `drive_steer`, `drive_speed` | The arbiter's output on `/drive`. Identical to `applied_*` until AEB fires, then 0. Not a label. |

## Flags
| Column | Meaning |
|---|---|
| `drift_offset` | Steering offset (rad) added by the `steering_drift` node. 0 most of the time. During a ~1 s burst it ramps as a triangle up to about ±0.1 rad and back. |
| `drift_active` | 1 while `drift_offset != 0` (computed by the recorder). **Keep these rows**: they hold the recovery behaviour. |
| `aeb_active` | **Latches**: becomes 1 on the first `/safety_drive` or `/camera_safety_drive` message and stays 1 for the rest of the run. **Drop these rows.** |

The expert doesn't know about the drift. `applied_steer` splits from `expert_steer` the moment drift starts, but the expert only corrects a few hundred ms later, once the car has moved. The correction can continue after `drift_active` returns to 0.

## Used for BC training
- **Inputs (X):** current + previous scan (reduced to ~108 bins), speed `-vx`, yaw rate `wz_imu × π/180`
- **Label (y):** `expert_steer / 0.4189` (normalised to [-1, 1])
- **Drop rows with:** NaN, `aeb_active == 1`, speed ≈ 0, no valid previous scan
- **Everything else** is for checks and plots only.
