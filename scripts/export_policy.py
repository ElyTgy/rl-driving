"""Export a trained BC checkpoint (.pt, needs torch) to a numpy policy (.npz, no torch needed).

The .npz is what the ROS 2 node loads (ros/bc_drive_node.py), because the ROS Python has no torch.
After exporting, the script runs both versions on the training observations and checks that
they agree, so a wrong layer order or a missing transpose cannot slip through.

Usage (from the repo root, in the project venv):
    .venv/bin/python scripts/export_policy.py                          # checkpoints/bc_v2.pt -> .npz
    .venv/bin/python scripts/export_policy.py --checkpoint path/to/model.pt
"""
import argparse
import os

import numpy as np
import torch
from torch import nn

from rl_driving.bc import load_model, predict
from rl_driving.numpy_policy import NumpyPolicy
from rl_driving.preprocess import PREPROC_VERSION

REPO_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_CHECKPOINT = os.path.join(REPO_DIR, 'checkpoints', f'bc_v{PREPROC_VERSION}.pt')
DEFAULT_DATA = os.path.join(REPO_DIR, 'processed', f'bc_v{PREPROC_VERSION}_folds.npz')
MAX_ALLOWED_DIFF = 1e-4      # normalised actions; float32 rounding is ~1e-6


def to_numpy_policy(model):
    """Copy the Linear layers of a SteeringMLP, in order (body layers, then the head)."""
    linears = [m for m in model.body if isinstance(m, nn.Linear)] + [model.head]
    return NumpyPolicy([m.weight.detach().numpy() for m in linears],
                       [m.bias.detach().numpy() for m in linears])


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--checkpoint', default=DEFAULT_CHECKPOINT, help='.pt saved by train_bc.py')
    parser.add_argument('--data', default=DEFAULT_DATA, help='dataset whose observations are used for the check')
    args = parser.parse_args()
    torch.set_num_threads(2)

    model, _ = load_model(args.checkpoint)
    policy = to_numpy_policy(model)
    out_path = os.path.splitext(args.checkpoint)[0] + '.npz'
    policy.save(out_path)

    # Check: the reloaded numpy policy must give the same outputs as torch on real observations.
    X = np.load(args.data)['X']
    diff = np.abs(NumpyPolicy.load(out_path)(X) - predict(model, X)).max()
    print(f'saved {out_path}  (layers: {[W.shape for W in policy.weights]})')
    print(f'numpy vs torch on {len(X)} observations: max difference {diff:.2e}')
    if diff > MAX_ALLOWED_DIFF:
        raise SystemExit(f'numpy and torch outputs differ by more than {MAX_ALLOWED_DIFF}; export is wrong')


if __name__ == '__main__':
    main()
