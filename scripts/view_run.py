"""Interactive viewer for one recorded run: what the BC model sees, row by row.

Usage (from the repo root):
    .venv/bin/python scripts/view_run.py 20261106_094614          # by run id
    .venv/bin/python scripts/view_run.py half-run                  # or any unique part of the folder name
    .venv/bin/python scripts/view_run.py 094720 --kept-only        # slider steps only over training rows
    .venv/bin/python scripts/view_run.py 095949 --cut 40           # override the tail cut (s)

Window:
  left:   top-down lidar view in the car frame (forward = up, left = left).
          grey = raw beams (clipped to the horizon), orange x = invalid beams (shown as "far"),
          blue = the 108 min-pooled bins of the current scan, light blue = previous scan bins
          (exactly what make_observation produces). Arrows = expert (blue) and applied (orange) steering.
  right:  speed, yaw rate, steering and a lidar heat map (bins over time).
          grey bands = drift active, red bands = rows dropped from training.
  bottom: slider (moves every panel), Play/Pause button.
Keys: left/right = 1 row, up/down = 10 rows, space = play/pause.

The drop rules below repeat the ones in notebooks/data-cleaning.ipynb (make_masks). Keep them in sync,
or better, move the notebook's cleaning functions into src/rl_driving and import them here.
"""
import argparse
import glob
import json
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.widgets import Button, Slider

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, 'src'))
from rl_driving.preprocess import (BEAMS_PER_BIN, HORIZON, N_BEAMS, N_BINS, PREPROC_VERSION,  # noqa: E402
                                   RANGE_MAX, RANGE_MIN, make_observation, reduce_scan, steer_to_action)

RAW_DIR = os.path.join(REPO, 'raw-data')

# Same settings as the notebook
STOP_SPEED = 0.1    # m/s
MAX_GAP = 0.04      # s
DEFAULT_CUTS = {'20261106_094614': 20.0}   # run id -> cut time (s); other runs: no cut
DROP_REASONS = ['nan', 'aeb', 'stopped', 'too_late', 'no_prev']

PLAY_INTERVAL_MS = 50


# ---------- loading and cleaning ----------

def find_run_dir(key):
    matches = [p for p in glob.glob(os.path.join(RAW_DIR, '*run_*')) if key in os.path.basename(p)]
    if len(matches) != 1:
        runs = '\n  '.join(sorted(os.path.basename(p) for p in glob.glob(os.path.join(RAW_DIR, '*run_*'))))
        sys.exit(f'"{key}" matches {len(matches)} runs. Available:\n  {runs}')
    return matches[0]


def load_run(run_dir):
    files = sorted(glob.glob(os.path.join(run_dir, 'dataset', 'chunk_*.npz')))
    if not files:
        sys.exit(f'no chunk files in {run_dir}/dataset')
    chunks = [np.load(f) for f in files]
    d = {k: np.concatenate([c[k] for c in chunks]) for k in chunks[0].files}
    with open(os.path.join(run_dir, 'dataset', 'meta.json')) as f:
        meta = json.load(f)
    if len(d['t']) == 0:
        sys.exit(f'{os.path.basename(run_dir)} is empty (0 rows)')
    return d, meta


def make_masks(d, sig, cut):
    n = len(sig['t'])
    m = {
        'nan': (np.isnan(sig['label']) | np.isnan(sig['speed']) | np.isnan(sig['yaw_rate'])
                | np.isnan(d['ranges']).any(axis=1)),
        'aeb': d['aeb_active'] == 1,
        'stopped': sig['speed'] < STOP_SPEED,
        'too_late': (sig['t'] > cut) if cut is not None else np.zeros(n, dtype=bool),
    }
    keep = ~(m['nan'] | m['aeb'] | m['stopped'] | m['too_late'])
    has_prev = np.zeros(n, dtype=bool)
    has_prev[1:] = keep[:-1] & (np.diff(sig['t']) < MAX_GAP)
    m['no_prev'] = ~has_prev
    m['final'] = keep & has_prev
    return m


def first_reason(masks):
    """Per row: the first drop reason that applies, or '' if the row is kept."""
    reason = np.full(len(masks['final']), '', dtype=object)
    for name in reversed(DROP_REASONS):     # reversed so the first reason in the list wins
        reason[masks[name]] = name
    reason[masks['final']] = ''
    return reason


def find_blocks(mask):
    starts = np.where(mask[1:] & ~mask[:-1])[0] + 1
    ends = np.where(~mask[1:] & mask[:-1])[0] + 1
    if mask[0]:
        starts = np.r_[0, starts]
    if mask[-1]:
        ends = np.r_[ends, len(mask)]
    return list(zip(starts, ends))


# ---------- viewer ----------

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('run', help='run id (YYYYMMDD_HHMMSS) or any unique part of the folder name')
    ap.add_argument('--cut', type=float, default=None, help='drop rows after this time (s); default from DEFAULT_CUTS')
    ap.add_argument('--kept-only', action='store_true', help='slider only visits rows used for training')
    args = ap.parse_args()

    run_dir = find_run_dir(args.run)
    run_name = os.path.basename(run_dir)
    run_id = run_name.split('run_')[-1]
    cut = args.cut if args.cut is not None else DEFAULT_CUTS.get(run_id)

    d, meta = load_run(run_dir)
    sig = {
        't': d['t'] - d['t'][0],
        'speed': -d['vx'],                          # odometry is sign-flipped
        'yaw_rate': d['wz_imu'] * np.pi / 180,      # IMU reports deg/s
        'label': d['expert_steer'],
    }
    t = sig['t']
    masks = make_masks(d, sig, cut)
    reason = first_reason(masks)
    drift = d['drift_active'] == 1
    n = len(t)

    # Model inputs for every row (rows with NaN still get bins; they are just marked as dropped)
    bins = reduce_scan(d['ranges'])                         # (n, N_BINS) in [0, 1]
    label_norm = steer_to_action(sig['label'])

    # Beam and bin angles (rad, 0 = forward, + = left)
    sc = meta['scan']
    beam_ang = sc['angle_min'] + np.arange(N_BEAMS) * sc['angle_increment']
    starts = np.arange(0, N_BEAMS - 1, BEAMS_PER_BIN)
    ends = np.r_[starts[1:], N_BEAMS]
    bin_ang = np.array([beam_ang[s:e].mean() for s, e in zip(starts, ends)])

    rows = np.where(masks['final'])[0] if args.kept_only else np.arange(n)
    if len(rows) == 0:
        sys.exit('no rows to show')

    def to_xy(r, ang):
        # car frame: x forward, y left  ->  screen: up = forward, left = left
        return np.column_stack([-r * np.sin(ang), r * np.cos(ang)])

    # ---- layout ----
    fig = plt.figure(figsize=(16, 9))
    fig.suptitle(f'{run_name}   ({n} rows, {t[-1]:.1f} s, cut={cut}, '
                 f'{masks["final"].sum()} kept, preproc v{PREPROC_VERSION})', fontsize=11)
    gs = fig.add_gridspec(4, 2, width_ratios=[1, 1.6], left=0.05, right=0.98, top=0.93, bottom=0.12,
                          hspace=0.25, wspace=0.15)
    ax_lidar = fig.add_subplot(gs[:3, 0])
    ax_info = fig.add_subplot(gs[3, 0])
    ax_speed = fig.add_subplot(gs[0, 1])
    ax_yaw = fig.add_subplot(gs[1, 1], sharex=ax_speed)
    ax_steer = fig.add_subplot(gs[2, 1], sharex=ax_speed)
    ax_heat = fig.add_subplot(gs[3, 1], sharex=ax_speed)
    ts_axes = [ax_speed, ax_yaw, ax_steer, ax_heat]

    # ---- time series ----
    drop_blocks = find_blocks(~masks['final'])
    drift_blocks = find_blocks(drift)
    for ax in ts_axes[:3]:
        for s, e in drift_blocks:
            ax.axvspan(t[s], t[e - 1], color='#888888', alpha=0.2, lw=0)
        for s, e in drop_blocks:
            ax.axvspan(t[s], t[e - 1], color='#d03b3b', alpha=0.12, lw=0)
        ax.grid(alpha=0.25)
        ax.tick_params(labelbottom=False)

    ax_speed.plot(t, sig['speed'], color='#2a78d6', lw=1.2)
    ax_speed.axhline(STOP_SPEED, color='#d03b3b', lw=0.8, ls='--')
    ax_speed.set_ylabel('speed -vx\n[m/s]')

    ax_yaw.plot(t, sig['yaw_rate'], color='#2a78d6', lw=1.2)
    ax_yaw.set_ylabel('yaw rate wz_imu\n[rad/s]')

    ax_steer.plot(t, d['applied_steer'], color='#eb6834', lw=1.0, label='applied (expert + drift)')
    ax_steer.plot(t, d['expert_steer'], color='#2a78d6', lw=1.3, label='expert = label')
    ax_steer.set_ylabel('steering\n[rad, + = left]')
    ax_steer.legend(loc='upper left', frameon=False, fontsize=8, ncol=2)

    finite = np.isfinite(t)
    ax_heat.imshow(bins.T * HORIZON, aspect='auto', origin='lower', cmap='viridis', vmin=0, vmax=HORIZON,
                   extent=[t[finite][0], t[finite][-1], np.degrees(bin_ang[0]), np.degrees(bin_ang[-1])],
                   interpolation='nearest')
    ax_heat.set_ylabel('bin angle [deg]\n(model scan, m)')
    ax_heat.set_xlabel('time [s]')

    cursors = [ax.axvline(t[rows[0]], color='k', lw=1) for ax in ts_axes]

    # ---- lidar view ----
    lim = HORIZON + 0.5
    ax_lidar.set_xlim(-lim, lim)
    ax_lidar.set_ylim(-lim * 0.75, lim)
    ax_lidar.set_aspect('equal')
    ax_lidar.grid(alpha=0.25)
    ax_lidar.set_xlabel('left  <-  [m]  ->  right')
    ax_lidar.set_ylabel('forward [m]')
    ax_lidar.add_patch(plt.Circle((0, 0), HORIZON, fill=False, ls=':', color='#888888'))
    ax_lidar.plot([0], [0], marker='^', color='k', ms=9)

    raw_pts = ax_lidar.scatter([], [], s=2, color='#999999', label='raw beams')
    bad_pts = ax_lidar.scatter([], [], s=10, marker='x', color='#eb6834', lw=0.8, label='invalid -> far')
    prev_pts = ax_lidar.scatter([], [], s=14, color='#9cc3ee', label='prev scan bins (model)')
    bin_pts = ax_lidar.scatter([], [], s=14, color='#2a78d6', label='current scan bins (model)')
    expert_arrow, = ax_lidar.plot([], [], color='#2a78d6', lw=2.5)
    applied_arrow, = ax_lidar.plot([], [], color='#eb6834', lw=1.5)
    ax_lidar.legend(loc='lower left', fontsize=8, frameon=True, markerscale=2)

    ax_info.axis('off')
    info = ax_info.text(0, 1, '', va='top', ha='left', family='monospace', fontsize=9, transform=ax_info.transAxes)

    def steer_line(steer, length=2.0):
        a = 0.0 if not np.isfinite(steer) else steer
        return [0, -length * np.sin(a)], [0, length * np.cos(a)]

    def draw(i):
        r = d['ranges'][i]
        with np.errstate(invalid='ignore'):
            invalid = ~np.isfinite(r) | (r < RANGE_MIN) | (r > RANGE_MAX)
        raw_pts.set_offsets(to_xy(np.minimum(r[~invalid], HORIZON), beam_ang[~invalid]))
        bad_pts.set_offsets(to_xy(np.full(invalid.sum(), HORIZON), beam_ang[invalid]))
        bin_pts.set_offsets(to_xy(bins[i] * HORIZON, bin_ang))
        prev_pts.set_offsets(to_xy(bins[i - 1] * HORIZON, bin_ang) if i > 0 else np.empty((0, 2)))
        expert_arrow.set_data(*steer_line(d['expert_steer'][i]))
        applied_arrow.set_data(*steer_line(d['applied_steer'][i]))
        for c in cursors:
            c.set_xdata([t[i], t[i]])

        kept = masks['final'][i]
        status = 'KEPT for training' if kept else f'DROPPED ({reason[i]})'
        ax_lidar.set_title(f't = {t[i]:6.2f} s   row {i}   {status}', color='k' if kept else '#d03b3b')
        prev_i = max(i - 1, 0)
        obs = make_observation(d['ranges'][i], d['ranges'][prev_i], sig['speed'][i], sig['yaw_rate'][i])
        info.set_text(
            f'raw / calibrated              model input / label\n'
            f'speed    {sig["speed"][i]:+7.3f} m/s       obs[-2] = {obs[-2]:+.3f}  (speed/3)\n'
            f'yaw rate {sig["yaw_rate"][i]:+7.3f} rad/s     obs[-1] = {obs[-1]:+.3f}  (yaw/2)\n'
            f'expert   {d["expert_steer"][i]:+7.3f} rad       label   = {label_norm[i]:+.3f}  (/0.4189)\n'
            f'applied  {d["applied_steer"][i]:+7.3f} rad       drift {"ON " if drift[i] else "off"} '
            f'({d["drift_offset"][i]:+.3f})   aeb {int(d["aeb_active"][i]) if np.isfinite(d["aeb_active"][i]) else "?"}\n'
            f'invalid beams {invalid.sum():4d}/{N_BEAMS}     closest bin {bins[i].min() * HORIZON:5.2f} m\n'
            f'dt to prev scan {t[i] - t[prev_i]:.3f} s   obs length {obs.size}'
        )
        fig.canvas.draw_idle()

    # ---- controls ----
    ax_slider = fig.add_axes([0.17, 0.03, 0.75, 0.03])
    slider = Slider(ax_slider, 'kept row #' if args.kept_only else 'row', 0, len(rows) - 1,
                    valinit=0, valstep=1)
    slider.on_changed(lambda v: draw(rows[int(v)]))

    ax_btn = fig.add_axes([0.03, 0.025, 0.06, 0.04])
    btn = Button(ax_btn, 'Play')
    timer = fig.canvas.new_timer(interval=PLAY_INTERVAL_MS)
    step = max(1, round(PLAY_INTERVAL_MS / 1000 / 0.026)) if not args.kept_only else 2   # ~real time

    def tick():
        k = int(slider.val) + step
        if k >= len(rows):
            toggle()
            return
        slider.set_val(k)

    timer.add_callback(tick)
    playing = [False]

    def toggle(_=None):
        playing[0] = not playing[0]
        (timer.start if playing[0] else timer.stop)()
        btn.label.set_text('Pause' if playing[0] else 'Play')
        fig.canvas.draw_idle()

    btn.on_clicked(toggle)

    def on_key(ev):
        moves = {'right': 1, 'left': -1, 'up': 10, 'down': -10}
        if ev.key in moves:
            slider.set_val(int(np.clip(slider.val + moves[ev.key], 0, len(rows) - 1)))
        elif ev.key == ' ':
            toggle()

    fig.canvas.mpl_connect('key_press_event', on_key)

    def on_click(ev):   # click on a time-series panel to jump there
        if ev.inaxes in ts_axes and ev.xdata is not None and (fig.canvas.toolbar is None or fig.canvas.toolbar.mode == ''):
            k = int(np.argmin(np.abs(t[rows] - ev.xdata)))
            slider.set_val(k)

    fig.canvas.mpl_connect('button_press_event', on_click)

    draw(rows[0])
    plt.show()


if __name__ == '__main__':
    main()
