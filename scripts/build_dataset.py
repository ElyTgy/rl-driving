"""Build the BC training datasets from the recorded runs (script version of notebooks/data-cleaning.ipynb).

Steps:
  1. Clean every run in RUNS (src/rl_driving/cleaning.py): drop rows with missing data, after a safety
     stop, while stopped, after the run's cut time, or without a valid previous scan.
  2. Turn the kept rows into network inputs X and labels y (src/rl_driving/preprocess.py).
  3. Make two splits and save both to processed/:
       bc_v<version>.npz        train / val by whole run
       bc_v<version>_folds.npz  blocked k-fold: time chunks shuffled into folds, with a gap around val chunks
                                (this is what notebooks/02_train_bc.ipynb trains on)
  4. Print a summary, and optionally save check plots.

Usage (from the repo root):
    .venv/bin/python scripts/build_dataset.py
    .venv/bin/python scripts/build_dataset.py --use-fail-run          # include the fail-run (095949)
    .venv/bin/python scripts/build_dataset.py --plots processed/plots # also save check plots as PNGs
"""
import argparse
import os

import numpy as np

from rl_driving.cleaning import RunConfig, clean_run, describe_dropped_rows
from rl_driving.preprocess import N_BINS, OBS_DIM, PREPROC_VERSION, make_observation, steer_to_action

REPO_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DIR = os.path.join(REPO_DIR, 'raw-data')
DEFAULT_OUT_DIR = os.path.join(REPO_DIR, 'processed')

HALF_RUN = '20261106_094614'
ONE_RUN = '20261106_094720'
FAIL_RUN = '20261106_095949'

# Every run we may use. no-cam-run (094711) is left out because it is empty.
RUNS = [
    RunConfig(HALF_RUN, cut_time_s=20.0, note='half-run: obstacle placed too early, AEB at 21.7 s'),
    RunConfig(ONE_RUN, cut_time_s=None, note='1-run: AEB at 74.2 s cuts the tail'),
    RunConfig(FAIL_RUN, cut_time_s=None, note='fail-run: unknown why "fail", recorded with 30% drift'),
]

# Train / val by whole run (the simple split)
RUN_SPLIT_TRAIN = [ONE_RUN]
RUN_SPLIT_VAL = [HALF_RUN]

# Blocked k-fold
CHUNK_LENGTH_S = 8.0   # each run is cut into ~8 s pieces; a leftover under half that joins the piece before
NUM_FOLDS = 5          # each fold is val once (~20%)
GAP_S = 1.0            # train rows this close to a val row (same run) are left out of that fold
SHUFFLE_SEED = 0


# ---------------------------------------------------------------- features and labels

def build_features_and_labels(usable_rows):
    """Turn one cleaned run into (features X, labels y, drift flags).

    X: (rows, OBS_DIM) from make_observation: current scan bins, previous scan bins, speed, yaw rate
    y: (rows,) expert steering normalised to [-1, 1]
    drift: (rows,) 1 while drift was active; not an input, used for stats
    """
    features = make_observation(usable_rows['ranges'], usable_rows['previous_ranges'],
                                usable_rows['speed_mps'], usable_rows['yaw_rate_radps'])
    labels = steer_to_action(usable_rows['steer_label_rad'])
    drift = usable_rows['drift_active'].astype(np.float32)
    return features, labels, drift


def concatenate_runs(dataset_by_run, run_ids):
    """Join (X, y, drift) of several runs. Safe, because every sample already holds its own previous scan."""
    features = np.concatenate([dataset_by_run[run_id][0] for run_id in run_ids])
    labels = np.concatenate([dataset_by_run[run_id][1] for run_id in run_ids])
    drift = np.concatenate([dataset_by_run[run_id][2] for run_id in run_ids])
    return features, labels, drift


# ---------------------------------------------------------------- blocked k-fold

def assign_time_chunks(run_index, time_s, chunk_length_s=CHUNK_LENGTH_S):
    """Give every row a chunk id: consecutive ~chunk_length_s pieces of time within each run.

    Ids are numbered per run, so a chunk never spans two runs.
    """
    chunk_id = np.zeros(len(time_s), dtype=int)
    next_free_id = 0
    for run in np.unique(run_index):
        rows = np.where(run_index == run)[0]
        time_in_run = time_s[rows] - time_s[rows][0]
        local_chunk = (time_in_run // chunk_length_s).astype(int)
        num_local_chunks = local_chunk.max() + 1
        last_chunk_length = time_in_run[-1] - (num_local_chunks - 1) * chunk_length_s
        if num_local_chunks > 1 and last_chunk_length < chunk_length_s / 2:
            local_chunk[local_chunk == num_local_chunks - 1] = num_local_chunks - 2
        chunk_id[rows] = local_chunk + next_free_id
        next_free_id += local_chunk.max() + 1
    return chunk_id


def assign_chunks_to_folds(chunk_id, num_folds=NUM_FOLDS, seed=SHUFFLE_SEED):
    """Shuffle the chunk ids and deal them into folds in turn. Returns the fold of every row."""
    chunk_ids = np.unique(chunk_id)
    np.random.default_rng(seed).shuffle(chunk_ids)
    fold_of_chunk = {chunk: position % num_folds for position, chunk in enumerate(chunk_ids)}
    return np.array([fold_of_chunk[chunk] for chunk in chunk_id])


def fold_train_val_masks(fold, run_index, time_s, val_fold, gap_s=GAP_S):
    """Train and val row masks for one fold.

    Val = rows in val_fold. Train = every other row, except rows within gap_s of a val row of the
    same run: scans on either side of a chunk boundary are near-copies and would leak into val.
    """
    is_val = fold == val_fold
    is_train = ~is_val
    for run in np.unique(run_index):
        in_run = run_index == run
        val_times = np.sort(time_s[in_run & is_val])
        if len(val_times) == 0:
            continue
        train_rows = np.where(in_run & is_train)[0]
        # distance from each train row to the nearest val row: check the val times just before and after it
        insert_at = np.searchsorted(val_times, time_s[train_rows])
        before = val_times[np.clip(insert_at - 1, 0, len(val_times) - 1)]
        after = val_times[np.clip(insert_at, 0, len(val_times) - 1)]
        distance_to_val = np.minimum(np.abs(time_s[train_rows] - before), np.abs(time_s[train_rows] - after))
        is_train[train_rows[distance_to_val < gap_s]] = False
    return is_train, is_val


def build_kfold_split(dataset_by_run, cleaned_by_run, run_ids):
    """Stack the runs and build every fold's train / val masks. Returns a dict ready to save."""
    features, labels, drift = concatenate_runs(dataset_by_run, run_ids)
    time_s = np.concatenate([cleaned_by_run[run_id].usable_rows['time_s'] for run_id in run_ids])
    run_index = np.concatenate([np.full(len(dataset_by_run[run_id][1]), i) for i, run_id in enumerate(run_ids)])

    chunk_id = assign_time_chunks(run_index, time_s)
    fold = assign_chunks_to_folds(chunk_id)
    train_masks = np.zeros((NUM_FOLDS, len(labels)), dtype=bool)
    val_masks = np.zeros((NUM_FOLDS, len(labels)), dtype=bool)
    for k in range(NUM_FOLDS):
        train_masks[k], val_masks[k] = fold_train_val_masks(fold, run_index, time_s, k)

    # Key names match what notebooks/02_train_bc.ipynb reads.
    return {
        'X': features, 'y': labels, 'drift': drift, 't': time_s, 'run': run_index,
        'chunk': chunk_id, 'fold': fold, 'train_masks': train_masks, 'val_masks': val_masks,
        'runs': np.array(run_ids), 'preproc_version': PREPROC_VERSION,
        'chunk_len': CHUNK_LENGTH_S, 'gap': GAP_S, 'seed': SHUFFLE_SEED,
    }


# ---------------------------------------------------------------- reports

def print_run_summary(cleaned_by_run):
    print(f'{"run":16s} {"rows":>6s} {"drift":>6s} {"speed":>6s}  time kept')
    for run_id, cleaned in cleaned_by_run.items():
        rows = cleaned.usable_rows
        print(f'{run_id:16s} {len(rows["steer_label_rad"]):6d} {rows["drift_active"].mean():6.1%} '
              f'{np.median(rows["speed_mps"]):6.2f}  {rows["time_s"][0]:.1f}-{rows["time_s"][-1]:.1f} s')


def print_feature_summary(name, features, labels, drift):
    """Shape, NaN count and value range of every part of X, and of y."""
    feature_parts = [('current scan', slice(0, N_BINS)), ('previous scan', slice(N_BINS, 2 * N_BINS)),
                     ('speed', slice(2 * N_BINS, 2 * N_BINS + 1)), ('yaw rate', slice(2 * N_BINS + 1, OBS_DIM))]
    num_nan = int(np.isnan(features).sum() + np.isnan(labels).sum())
    print(f'{name}: X {features.shape}  y {labels.shape}  drift {drift.mean():.1%}  NaN {num_nan}')
    for part_name, columns in feature_parts:
        values = features[:, columns]
        print(f'  {part_name:14s} min {values.min():+.3f}  mean {values.mean():+.3f}  max {values.max():+.3f}')
    print(f'  {"label y":14s} min {labels.min():+.3f}  mean {labels.mean():+.3f}  max {labels.max():+.3f}')


def print_fold_summary(kfold):
    num_samples = len(kfold['y'])
    print(f'{num_samples} samples, {kfold["chunk"].max() + 1} chunks, {NUM_FOLDS} folds')
    print(f'{"fold":>4} {"chunks":>6} {"train":>6} {"val":>5} {"gap":>5} {"val %":>6} {"drift tr":>8} {"drift val":>9}')
    for k in range(NUM_FOLDS):
        is_train, is_val = kfold['train_masks'][k], kfold['val_masks'][k]
        num_gap = num_samples - is_train.sum() - is_val.sum()
        print(f'{k:4d} {len(np.unique(kfold["chunk"][is_val])):6d} {is_train.sum():6d} {is_val.sum():5d} '
              f'{num_gap:5d} {is_val.sum() / (is_train.sum() + is_val.sum()):6.1%} '
              f'{kfold["drift"][is_train].mean():8.1%} {kfold["drift"][is_val].mean():9.1%}')


# ---------------------------------------------------------------- check plots (optional)

def save_check_plots(plot_dir, cleaned_by_run, kfold):
    """Save the notebook's check plots as PNGs: kept/dropped rows per run, raw vs reduced scan, fold layout."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    from rl_driving.preprocess import BEAMS_PER_BIN, HORIZON, N_BEAMS, reduce_scan

    os.makedirs(plot_dir, exist_ok=True)

    for run_id, cleaned in cleaned_by_run.items():
        time_s, dropped = cleaned.signals['time_s'], ~cleaned.drop_masks['usable']
        fig, axes = plt.subplots(2, 1, figsize=(13, 4.5), sharex=True)
        for ax, values, label in [(axes[0], cleaned.signals['speed_mps'], 'speed [m/s]'),
                                  (axes[1], cleaned.signals['steer_label_rad'], 'expert_steer [rad]')]:
            ax.plot(time_s, values, color='#888888', lw=1)
            ax.plot(time_s[dropped], values[dropped], '.', color='#d03b3b', ms=3, label='dropped')
            ax.set_ylabel(label)
            ax.grid(alpha=0.25)
        axes[0].set_title(f'run {run_id}: {cleaned.drop_masks["usable"].sum()} rows kept')
        axes[0].legend(frameon=False)
        axes[1].set_xlabel('time [s]')
        fig.tight_layout()
        fig.savefig(os.path.join(plot_dir, f'kept_rows_{run_id}.png'), dpi=90)
        plt.close(fig)

    example_scan = cleaned_by_run[HALF_RUN].usable_rows['ranges'][300]
    beam_angles_deg = np.linspace(-135, 135, N_BEAMS)
    bin_centres_deg = beam_angles_deg[np.arange(0, N_BEAMS - 1, BEAMS_PER_BIN)] + BEAMS_PER_BIN * 0.25 / 2
    fig, ax = plt.subplots(figsize=(13, 3.5))
    ax.plot(beam_angles_deg, np.clip(example_scan, 0, HORIZON), '.', color='#888888', ms=2, label='raw beams (clipped)')
    ax.step(bin_centres_deg, reduce_scan(example_scan) * HORIZON, where='mid', color='#2a78d6', lw=1.6,
            label='reduced (min of 10 beams)')
    ax.set_xlabel('beam angle [deg]  (- = right, + = left)')
    ax.set_ylabel('range [m]')
    ax.legend(frameon=False)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(os.path.join(plot_dir, 'raw_vs_reduced_scan.png'), dpi=90)
    plt.close(fig)

    run_ids = list(kfold['runs'])
    run_lengths = [kfold['t'][kfold['run'] == i].max() for i in range(len(run_ids))]
    fig, axes = plt.subplots(1, len(run_ids), figsize=(13, 2.8), sharey=True,
                             gridspec_kw={'width_ratios': run_lengths})
    for i, ax in enumerate(np.atleast_1d(axes)):
        in_run = kfold['run'] == i
        for k in range(NUM_FOLDS):
            for mask, color in [(kfold['train_masks'][k], '#2a78d6'), (kfold['val_masks'][k], '#eb6834')]:
                rows = in_run & mask
                ax.scatter(kfold['t'][rows], np.full(rows.sum(), k), s=2, color=color)
        ax.set_title(f'run {run_ids[i]}  (blue = train, orange = val)', fontsize=9)
        ax.set_xlabel('time [s]')
    np.atleast_1d(axes)[0].set_ylabel('fold')
    np.atleast_1d(axes)[0].invert_yaxis()
    fig.tight_layout()
    fig.savefig(os.path.join(plot_dir, 'fold_layout.png'), dpi=90)
    plt.close(fig)
    print(f'saved check plots to {plot_dir}')


# ---------------------------------------------------------------- main

def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--out-dir', default=DEFAULT_OUT_DIR, help='where to save the .npz files')
    parser.add_argument('--use-fail-run', action='store_true', help='include the fail-run (095949) in training data')
    parser.add_argument('--plots', metavar='DIR', help='also save check plots as PNGs in DIR')
    args = parser.parse_args()

    runs_to_use = [run for run in RUNS if run.run_id != FAIL_RUN or args.use_fail_run]

    print('=== cleaning')
    cleaned_by_run = {}
    for run in runs_to_use:
        cleaned = clean_run(RAW_DIR, run)
        cleaned_by_run[run.run_id] = cleaned
        print(f'run {run.run_id} ({run.note}), cut = {run.cut_time_s}')
        print(describe_dropped_rows(cleaned.drop_masks, cleaned.signals['time_s']))
    print()
    print_run_summary(cleaned_by_run)

    dataset_by_run = {run_id: build_features_and_labels(cleaned.usable_rows)
                      for run_id, cleaned in cleaned_by_run.items()}

    print('\n=== split by run')
    train_runs = RUN_SPLIT_TRAIN + ([FAIL_RUN] if args.use_fail_run else [])
    train_features, train_labels, train_drift = concatenate_runs(dataset_by_run, train_runs)
    val_features, val_labels, val_drift = concatenate_runs(dataset_by_run, RUN_SPLIT_VAL)
    print_feature_summary(f'train {train_runs}', train_features, train_labels, train_drift)
    print_feature_summary(f'val {RUN_SPLIT_VAL}', val_features, val_labels, val_drift)

    print('\n=== blocked k-fold')
    kfold = build_kfold_split(dataset_by_run, cleaned_by_run, list(dataset_by_run))
    print_fold_summary(kfold)

    os.makedirs(args.out_dir, exist_ok=True)
    run_split_path = os.path.join(args.out_dir, f'bc_v{PREPROC_VERSION}.npz')
    np.savez_compressed(
        run_split_path,
        X_train=train_features, y_train=train_labels, drift_train=train_drift,
        X_val=val_features, y_val=val_labels, drift_val=val_drift,
        preproc_version=PREPROC_VERSION, train_runs=np.array(train_runs), val_runs=np.array(RUN_SPLIT_VAL),
    )
    kfold_path = os.path.join(args.out_dir, f'bc_v{PREPROC_VERSION}_folds.npz')
    np.savez_compressed(kfold_path, **kfold)
    print(f'\nsaved {run_split_path}\nsaved {kfold_path}')

    if args.plots:
        save_check_plots(args.plots, cleaned_by_run, kfold)


if __name__ == '__main__':
    main()
