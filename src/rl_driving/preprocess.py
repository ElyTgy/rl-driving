"""Observation preprocessing shared by BC training, the RL sim wrapper and the ROS2 node on the car.

Every place that feeds the network must call `make_observation` with the same raw inputs,
so the sim and the car produce identical values. Changing any constant here changes what the
network sees: bump PREPROC_VERSION and retrain. Save the version with every dataset and model.

Observation layout (length OBS_DIM = 2 * N_BINS + 2):
    [scan bins at t (N_BINS), scan bins at t-1 (N_BINS), speed, yaw rate]
"""
import numpy as np

PREPROC_VERSION = 1

# Lidar, matching the real car (meta.json) and the sim config
N_BEAMS = 1081            # -135° .. +135°, 0.25° apart
RANGE_MIN = 0.02          # m; readings below this are invalid
RANGE_MAX = 30.0          # m; readings above this are invalid (65.533 = no return)

# Scan reduction
BEAMS_PER_BIN = 10        # 1081 beams -> 108 bins (the last bin holds 11 beams)
HORIZON = 10.0            # m; ranges are clipped here, invalid readings become HORIZON ("far")
_BIN_STARTS = np.arange(0, N_BEAMS - 1, BEAMS_PER_BIN)   # 0, 10, ..., 1070
N_BINS = len(_BIN_STARTS)                                # 108

# Fixed scales (never dataset statistics, so the sim and the car scale identically)
SPEED_SCALE = 3.0         # m/s
YAW_RATE_SCALE = 2.0      # rad/s; the wall follower peaks around 1.45 rad/s
STEER_MAX = 0.4189        # rad; the sim's steering limit, used to normalise the label

OBS_DIM = 2 * N_BINS + 2


def reduce_scan(ranges):
    """Turn raw ranges (..., 1081) in metres into bins (..., N_BINS) in [0, 1].

    1. Invalid readings (NaN, inf, < RANGE_MIN, > RANGE_MAX) become HORIZON, i.e. "far".
       This must happen before the min, or a 0.0 or NaN reading would win the min.
    2. Clip to HORIZON.
    3. Min over each group of BEAMS_PER_BIN beams, so a thin obstacle seen by one beam is kept.
    4. Divide by HORIZON.
    """
    r = np.asarray(ranges, dtype=np.float32)
    if r.shape[-1] != N_BEAMS:
        raise ValueError(f'expected {N_BEAMS} beams, got {r.shape[-1]}')
    with np.errstate(invalid='ignore'):
        invalid = ~np.isfinite(r) | (r < RANGE_MIN) | (r > RANGE_MAX)
    r = np.where(invalid, HORIZON, np.minimum(r, HORIZON))
    bins = np.minimum.reduceat(r, _BIN_STARTS, axis=-1)
    return bins / HORIZON


def make_observation(ranges, ranges_prev, speed, yaw_rate):
    """Build the network input from raw, calibrated signals.

    ranges, ranges_prev: raw scans (..., 1081) in m, the current one and the one just before it
    speed: forward speed in m/s (positive forward; on the real car that is -vx)
    yaw_rate: in rad/s (positive = turning left; on the real car that is wz_imu * pi / 180)
    Works on a single sample or a batch. Returns float32 (..., OBS_DIM).
    """
    speed = np.asarray(speed, dtype=np.float32)[..., None] / SPEED_SCALE
    yaw_rate = np.asarray(yaw_rate, dtype=np.float32)[..., None] / YAW_RATE_SCALE
    return np.concatenate(
        [reduce_scan(ranges), reduce_scan(ranges_prev), speed, yaw_rate], axis=-1
    ).astype(np.float32)


def steer_to_action(steer):
    """Steering in rad -> network output in [-1, 1]."""
    return np.asarray(steer, dtype=np.float32) / STEER_MAX


def action_to_steer(action):
    """Network output in [-1, 1] -> steering in rad."""
    return np.clip(np.asarray(action, dtype=np.float32), -1.0, 1.0) * STEER_MAX
