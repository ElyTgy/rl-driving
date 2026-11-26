"""The trained BC MLP, run with numpy only (no torch).

Why this exists: the ROS 2 Python on this machine (sim_ws/.venv) and, later, the Jetson do not
have torch. The network is small (two tanh layers and a linear head, see bc.SteeringMLP), so its
forward pass is just three matrix multiplies, which numpy does easily.

Workflow:
    1. Train with torch (scripts/train_bc.py) -> checkpoints/bc_v1.pt
    2. Export the weights (scripts/export_policy.py) -> checkpoints/bc_v1.npz
    3. Anywhere without torch: policy = NumpyPolicy.load('checkpoints/bc_v1.npz')

The .npz stores, for each Linear layer i (0, 1, ..., last = head):
    W{i}: weight matrix (out_features, in_features), exactly as torch stores it
    b{i}: bias vector (out_features,)
plus 'preproc_version', so a policy is never fed observations from a different preprocessing.
"""
import numpy as np

from .preprocess import PREPROC_VERSION


class NumpyPolicy:
    """obs (OBS_DIM,) or (B, OBS_DIM) -> normalised actions (out_dim,) or (B, out_dim), same as SteeringMLP
    (squeezed like SteeringMLP when out_dim is 1)."""

    def __init__(self, weights, biases):
        self.weights = [np.asarray(W, dtype=np.float32) for W in weights]
        self.biases = [np.asarray(b, dtype=np.float32) for b in biases]

    @property
    def obs_dim(self):
        return self.weights[0].shape[1]

    @classmethod
    def load(cls, path):
        """Load an exported policy. Refuses one made for another preprocessing version."""
        data = np.load(path)
        version = int(data['preproc_version'])
        if version != PREPROC_VERSION:
            raise ValueError(f'policy uses preprocessing v{version}, code is v{PREPROC_VERSION}')
        n_layers = int(data['n_layers'])
        return cls([data[f'W{i}'] for i in range(n_layers)],
                   [data[f'b{i}'] for i in range(n_layers)])

    def save(self, path):
        arrays = {'preproc_version': PREPROC_VERSION, 'n_layers': len(self.weights)}
        for i, (W, b) in enumerate(zip(self.weights, self.biases)):
            arrays[f'W{i}'], arrays[f'b{i}'] = W, b
        np.savez(path, **arrays)

    def __call__(self, obs):
        x = np.asarray(obs, dtype=np.float32)
        *hidden, last = range(len(self.weights))
        for i in hidden:                                   # body: Linear -> tanh
            x = np.tanh(x @ self.weights[i].T + self.biases[i])
        out = x @ self.weights[last].T + self.biases[last]  # head: Linear, no activation
        return out[..., 0] if out.shape[-1] == 1 else out  # like SteeringMLP.forward
