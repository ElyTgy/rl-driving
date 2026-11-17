"""Cleaning one recorded run: load it, calibrate the sensors, decide which rows are usable.

Used by scripts/build_dataset.py (and can be used by scripts/view_run.py), so every tool drops
exactly the same rows. Column meanings are in docs.md.

Pipeline for one run (see `clean_run`):
    load_raw_run -> calibrate_signals -> build_drop_masks -> select_usable_rows
"""
import glob
import json
import os
from dataclasses import dataclass

import numpy as np

from .car_signals import speed_from_car_odom, yaw_rate_from_car_imu

# A row is "stopped" below this speed. Stopped rows are near-duplicate scans where the expert isn't driving.
STOP_SPEED_MPS = 0.1

# Longest allowed time between a scan and the scan before it. Normal spacing is ~0.027 s;
# 0.04 s allows jitter but rejects a missed scan (~0.05 s) or a longer recording gap.
MAX_SCAN_GAP_S = 0.04

# Drop reasons, in the order they are counted in the report.
DROP_REASONS = ('missing_data', 'safety_stop', 'stopped', 'after_cut', 'no_previous_scan')


@dataclass
class RunConfig:
    """How to clean one run."""
    run_id: str                       # YYYYMMDD_HHMMSS, the end of the run folder name
    cut_time_s: float | None = None   # drop everything after this time (s since start); None = no cut
    note: str = ''


@dataclass
class CleanedRun:
    """Result of cleaning one run."""
    config: RunConfig
    usable_rows: dict      # only the kept rows, only the columns training needs (see select_usable_rows)
    drop_masks: dict       # per-row True/False masks for every row of the raw run (see build_drop_masks)
    signals: dict          # calibrated signals for every row of the raw run (see calibrate_signals)


# ---------------------------------------------------------------- loading

def find_run_folder(raw_dir, run_id):
    """Return the folder of a run, looked up by its id (the folder's label prefix may change)."""
    matches = glob.glob(os.path.join(raw_dir, f'*{run_id}'))
    if len(matches) != 1:
        raise FileNotFoundError(f'expected 1 folder for run {run_id} in {raw_dir}, found {matches}')
    return matches[0]


def load_raw_run(raw_dir, run_id):
    """Load every chunk_*.npz of a run, joined in order. Returns (raw_rows, meta).

    raw_rows: dict column name -> array, one row per lidar scan.
    meta: the run's meta.json (lidar setup, topics, drift fraction).
    """
    dataset_dir = os.path.join(find_run_folder(raw_dir, run_id), 'dataset')
    chunk_paths = sorted(glob.glob(os.path.join(dataset_dir, 'chunk_*.npz')))
    if not chunk_paths:
        raise FileNotFoundError(f'no chunk files in {dataset_dir}')

    chunks = [np.load(path) for path in chunk_paths]
    raw_rows = {column: np.concatenate([chunk[column] for chunk in chunks]) for column in chunks[0].files}
    with open(os.path.join(dataset_dir, 'meta.json')) as f:
        meta = json.load(f)
    return raw_rows, meta


# ---------------------------------------------------------------- calibration

def calibrate_signals(raw_rows):
    """Apply the real car's sensor fixes once, so later steps never use the raw columns by mistake.

    - odometry speed is sign-flipped:   speed = -vx
    - the IMU yaw rate is in deg/s:     yaw_rate = wz_imu * pi / 180
    - the label is the expert's steering BEFORE drift (during drift it holds the expert's correction)
    The two sensor fixes come from car_signals, which the ROS node on the car uses too.
    """
    return {
        'time_s':          raw_rows['t'] - raw_rows['t'][0],
        'speed_mps':       speed_from_car_odom(raw_rows['vx']),
        'yaw_rate_radps':  yaw_rate_from_car_imu(raw_rows['wz_imu']),
        'steer_label_rad': raw_rows['expert_steer'],
    }


# ---------------------------------------------------------------- deciding which rows to keep

def build_drop_masks(raw_rows, signals, cut_time_s=None):
    """One True/False mask per drop reason (True = drop that row), plus the combined result.

    missing_data      the label, speed, yaw rate or any lidar beam is NaN (start of run, topics not ready)
    safety_stop       aeb_active is 1. It latches, so this drops everything after the first safety stop
    stopped           speed below STOP_SPEED_MPS
    after_cut         time after cut_time_s (hand-chosen end of usable data)
    no_previous_scan  the row before is not kept, or is more than MAX_SCAN_GAP_S earlier
                      (every sample also uses the previous scan)

    Also returns:
    passes_row_checks  not dropped by the first four reasons
    usable             passes_row_checks and has a valid previous scan: the rows used for training
    """
    time_s = signals['time_s']
    n_rows = len(time_s)
    masks = {
        'missing_data': (np.isnan(signals['steer_label_rad']) | np.isnan(signals['speed_mps'])
                         | np.isnan(signals['yaw_rate_radps']) | np.isnan(raw_rows['ranges']).any(axis=1)),
        'safety_stop': raw_rows['aeb_active'] == 1,
        'stopped': signals['speed_mps'] < STOP_SPEED_MPS,     # NaN < x is False; NaN rows are in missing_data
        'after_cut': time_s > cut_time_s if cut_time_s is not None else np.zeros(n_rows, dtype=bool),
    }
    passes_row_checks = ~(masks['missing_data'] | masks['safety_stop'] | masks['stopped'] | masks['after_cut'])

    # Row i has a valid previous scan if row i-1 passed and is close enough in time.
    # Shifting by one: has_previous_scan[1:] looks at passes_row_checks[:-1].
    has_previous_scan = np.zeros(n_rows, dtype=bool)
    has_previous_scan[1:] = passes_row_checks[:-1] & (np.diff(time_s) < MAX_SCAN_GAP_S)
    masks['no_previous_scan'] = ~has_previous_scan

    masks['passes_row_checks'] = passes_row_checks
    masks['usable'] = passes_row_checks & has_previous_scan
    return masks


def find_true_blocks(mask):
    """Return (start, end) row indices of every run of consecutive True values. `end` is exclusive.

    A block starts where a row is True and the row before is False (mask[1:] & ~mask[:-1]),
    and ends at the reverse. The first and last rows are handled separately because they have no neighbour.
    """
    block_starts = np.where(mask[1:] & ~mask[:-1])[0] + 1
    block_ends = np.where(~mask[1:] & mask[:-1])[0] + 1
    if mask[0]:
        block_starts = np.r_[0, block_starts]
    if mask[-1]:
        block_ends = np.r_[block_ends, len(mask)]
    return list(zip(block_starts, block_ends))


def describe_dropped_rows(drop_masks, time_s):
    """Return a text report: rows dropped per reason, and every block of consecutive dropped rows.

    Reasons often overlap (e.g. safety-stop rows are also stopped), so each row is counted once,
    under the first reason in DROP_REASONS that matches. Blocks in the middle of a run are worth a look.
    """
    n_rows = len(time_s)
    lines = []
    already_counted = np.zeros(n_rows, dtype=bool)
    for reason in DROP_REASONS:
        newly_dropped = drop_masks[reason] & ~already_counted
        lines.append(f'  {reason:17s} drops {newly_dropped.sum():5d}')
        already_counted |= drop_masks[reason]
    lines.append(f'  {"kept":17s}       {drop_masks["usable"].sum():5d} of {n_rows}')

    dropped_blocks = find_true_blocks(~drop_masks['usable'])
    lines.append(f'  dropped blocks: {len(dropped_blocks)}')
    for start, end in dropped_blocks:
        lines.append(f'    rows {start:5d}-{end - 1:5d}: {end - start:5d} rows  '
                     f'({time_s[start]:6.2f}-{time_s[end - 1]:6.2f} s)')
    return '\n'.join(lines)


# ---------------------------------------------------------------- selecting what training needs

def select_usable_rows(raw_rows, signals, drop_masks):
    """Keep only the usable rows and only the columns training needs.

    Inputs:  ranges (current scan), previous_ranges (the scan just before), speed_mps, yaw_rate_radps
    Label:   steer_label_rad (expert_steer, in rad, not normalised yet)
    Extras:  time_s and drift_active, for plots, stats and splitting (not network inputs)
    previous_ranges takes row i-1 for every kept row i; the no_previous_scan rule guarantees it is valid.
    """
    kept_index = np.where(drop_masks['usable'])[0]
    return {
        'ranges':          raw_rows['ranges'][kept_index],
        'previous_ranges': raw_rows['ranges'][kept_index - 1],
        'speed_mps':       signals['speed_mps'][kept_index],
        'yaw_rate_radps':  signals['yaw_rate_radps'][kept_index],
        'steer_label_rad': signals['steer_label_rad'][kept_index],
        'time_s':          signals['time_s'][kept_index],
        'drift_active':    raw_rows['drift_active'][kept_index],
    }


def clean_run(raw_dir, config):
    """Load, calibrate and mask one run, and keep its usable rows."""
    raw_rows, _meta = load_raw_run(raw_dir, config.run_id)
    signals = calibrate_signals(raw_rows)
    drop_masks = build_drop_masks(raw_rows, signals, config.cut_time_s)
    usable_rows = select_usable_rows(raw_rows, signals, drop_masks)
    return CleanedRun(config, usable_rows, drop_masks, signals)
