"""Train the BC MLP: steering and speed (the v1 steering-only version is notebooks/02_train_bc.ipynb).

Steps:
  1. Load the k-fold dataset made by scripts/build_dataset.py (processed/bc_v<version>_folds.npz).
  2. K-fold: per fold, train the MLP with early stopping on the val rows, and score it next to
     three baselines (linear = ridge regression, mean, zero). Print mean ± std over the folds.
  3. Train the final model on all rows for the median best epoch of the folds (+1), and save it with
     the settings and k-fold scores to checkpoints/bc_v<version>.pt, plus a JSON of the results.

Usage (from the repo root; the whole run takes ~15 min on this 2-core machine):
    nice .venv/bin/python scripts/train_bc.py
    nice .venv/bin/python scripts/train_bc.py --plots checkpoints/plots   # also save plots as PNGs
    nice .venv/bin/python scripts/train_bc.py --final-epochs 120          # skip k-fold, final model only
"""
import argparse
import json
import os

import numpy as np
import torch

from rl_driving.bc import predict, save_model, train_bc
from rl_driving.evaluate import fit_ridge, metrics
from rl_driving.preprocess import PREPROC_VERSION, STEER_MAX, action_to_speed

REPO_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_DATA = os.path.join(REPO_DIR, 'processed', f'bc_v{PREPROC_VERSION}_folds.npz')
DEFAULT_OUT = os.path.join(REPO_DIR, 'checkpoints', f'bc_v{PREPROC_VERSION}.pt')
MODELS = ['mlp', 'linear', 'mean', 'zero']


def run_kfold(X, y, drift, train_masks, val_masks, settings):
    """Train and score one MLP per fold. Returns (results rows, histories, val predictions per fold)."""
    results, histories, val_preds = [], [], {}
    for k in range(len(train_masks)):
        tr, va = train_masks[k], val_masks[k]
        model, hist = train_bc(X[tr], y[tr], X[va], y[va], seed=k, **settings)
        pred = predict(model, X[va])
        val_preds[k] = pred
        histories.append(hist)

        ridge = fit_ridge(X[tr], y[tr])
        for name, p in [('mlp', pred), ('linear', ridge(X[va])),
                        ('mean', np.broadcast_to(y[tr].mean(axis=0), y[va].shape)), ('zero', np.zeros_like(y[va]))]:
            results.append({'fold': k, 'model': name, **metrics(p, y[va], drift[va])})
        mlp = results[-4]
        print(f'fold {k}: best epoch {hist["best_epoch"]:3d}, steer RMSE {mlp["rmse"]:.4f} rad (R² {mlp["r2"]:.3f}), '
              f'speed RMSE {mlp["speed_rmse"]:.4f} m/s (R² {mlp["speed_r2"]:.3f})', flush=True)
    return results, histories, val_preds


def summarise(results):
    """Print mean ± std over folds per model, and return the means as a dict."""
    keys = ['rmse', 'r2', 'speed_rmse', 'speed_r2']
    print(f'\n{"model":8s} {"steer RMSE [rad]":>18s} {"steer R²":>16s} {"speed RMSE [m/s]":>18s} {"speed R²":>16s}')
    summary = {}
    for name in MODELS:
        rows = [r for r in results if r['model'] == name]
        values = {key: np.array([r[key] for r in rows], dtype=float) for key in rows[0] if key not in ('fold', 'model')}
        summary[name] = {key: float(np.nanmean(v)) for key, v in values.items()}
        cols = [f'{np.nanmean(values[key]):.4f} ± {np.nanstd(values[key]):.4f}' for key in keys]
        print(f'{name:8s} ' + ' '.join(f'{c:>18s}' if i % 2 == 0 else f'{c:>16s}' for i, c in enumerate(cols)))
    return summary


def save_plots(plot_dir, histories, val_preds, ds, y, drift, val_masks):
    import matplotlib
    matplotlib.use('Agg')                              # no window, just files
    import matplotlib.pyplot as plt
    os.makedirs(plot_dir, exist_ok=True)

    # Learning curves: dashed = train, solid = val, dot = best val epoch.
    fig, ax = plt.subplots(figsize=(10, 4))
    for k, h in enumerate(histories):
        color = plt.cm.tab10(k)
        ax.plot(h['train'], color=color, lw=1, ls='--', alpha=0.7)
        ax.plot(h['val'], color=color, lw=1.6, label=f'fold {k}')
        ax.plot(h['best_epoch'], h['val'][h['best_epoch']], 'o', color=color)
    ax.set_yscale('log'); ax.set_xlabel('epoch'); ax.set_ylabel('MSE (normalised steer + speed)')
    ax.set_title('dashed = train, solid = val'); ax.legend(frameon=False, ncol=5, fontsize=8); ax.grid(alpha=0.25)
    fig.tight_layout(); fig.savefig(os.path.join(plot_dir, 'learning_curves.png'), dpi=120); plt.close(fig)

    # Predictions vs expert on fold 0's val rows: per run, a steering panel and a speed panel.
    k, va = 0, val_masks[0]
    t_va, run_va = ds['t'][va], ds['run'][va]
    runs_in_fold = np.unique(run_va)
    fig, axes = plt.subplots(2 * len(runs_in_fold), 1, figsize=(13, 5 * len(runs_in_fold)), squeeze=False)
    for i, r in enumerate(runs_in_fold):
        m = run_va == r
        tt = t_va[m]
        for ax, j, to_unit, unit in [(axes[2 * i, 0], 0, lambda a: a * STEER_MAX, 'steering [rad]'),
                                     (axes[2 * i + 1, 0], 1, action_to_speed, 'speed [m/s]')]:
            ax.plot(tt, to_unit(y[va][m, j]), '.', ms=2.5, color='#2a78d6', label='expert (label)')
            ax.plot(tt, to_unit(val_preds[k][m, j]), '.', ms=2.5, color='#eb6834', label='MLP prediction')
            ax.set_ylabel(unit); ax.grid(alpha=0.25); ax.legend(frameon=False, fontsize=8, ncol=2)
        axes[2 * i, 0].set_title(f'fold {k}, run {ds["runs"][r]} (val rows only)', fontsize=10)
    axes[-1, 0].set_xlabel('time [s]')
    fig.tight_layout(); fig.savefig(os.path.join(plot_dir, 'predictions_fold0.png'), dpi=120); plt.close(fig)
    print(f'plots saved in {plot_dir}')


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--data', default=DEFAULT_DATA, help='k-fold .npz from build_dataset.py')
    parser.add_argument('--out', default=DEFAULT_OUT, help='checkpoint path (a .json with the results goes next to it)')
    parser.add_argument('--epochs', type=int, default=300, help='max epochs per fold')
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--weight-decay', type=float, default=1e-4)
    parser.add_argument('--batch', type=int, default=256)
    parser.add_argument('--patience', type=int, default=60, help='stop a fold after this many epochs without val improvement')
    parser.add_argument('--final-epochs', type=int, help='skip k-fold and train the final model for this many epochs')
    parser.add_argument('--plots', metavar='DIR', help='also save learning curves and predictions as PNGs in DIR')
    args = parser.parse_args()

    torch.set_num_threads(2)                           # this machine has 2 cores

    ds = np.load(args.data)
    X, y, drift = ds['X'], ds['y'], ds['drift']
    train_masks, val_masks = ds['train_masks'], ds['val_masks']
    print(f'{len(y)} samples, X {X.shape}, runs {list(ds["runs"])}, {len(train_masks)} folds, '
          f'chunks of {float(ds["chunk_len"])} s, gap {float(ds["gap"])} s')

    settings = dict(epochs=args.epochs, lr=args.lr, weight_decay=args.weight_decay,
                    batch_size=args.batch, patience=args.patience)
    report = {'data': os.path.abspath(args.data), 'preproc_version': PREPROC_VERSION,
              'n_samples': int(len(y)), 'settings': settings}

    if args.final_epochs is None:
        results, histories, val_preds = run_kfold(X, y, drift, train_masks, val_masks, settings)
        report['kfold_mean'] = summarise(results)
        report['kfold_per_fold'] = results
        report['best_epochs'] = [h['best_epoch'] for h in histories]
        final_epochs = int(np.median(report['best_epochs'])) + 1
        if args.plots:
            save_plots(args.plots, histories, val_preds, ds, y, drift, val_masks)
    else:
        final_epochs = args.final_epochs
    report['final_epochs'] = final_epochs

    # Final model on all rows. No val set is left, so it trains for a fixed number of epochs.
    print(f'\ntraining final model on all {len(y)} samples for {final_epochs} epochs ...', flush=True)
    final_model, _ = train_bc(X, y, epochs=final_epochs, lr=args.lr, weight_decay=args.weight_decay,
                              batch_size=args.batch, seed=0)
    report['final_train_fit'] = metrics(predict(final_model, X), y, drift)

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    extra = {'epochs': final_epochs, 'runs': list(ds['runs']), 'settings': settings}
    if 'kfold_mean' in report:
        extra['kfold_rmse_rad'] = report['kfold_mean']['mlp']['rmse']
        extra['kfold_r2'] = report['kfold_mean']['mlp']['r2']
        extra['kfold_speed_rmse_mps'] = report['kfold_mean']['mlp']['speed_rmse']
    save_model(final_model, args.out, **extra)
    json_path = os.path.splitext(args.out)[0] + '.json'
    with open(json_path, 'w') as f:
        json.dump(report, f, indent=2, default=str)
    print(f'fit on its own training data: {report["final_train_fit"]}')
    print(f'saved {args.out} and {json_path}')


if __name__ == '__main__':
    main()
