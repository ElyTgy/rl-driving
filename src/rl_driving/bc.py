"""Behavioural cloning: the steering network and its training loop.

The network maps an observation from `preprocess.make_observation` (OBS_DIM numbers) to one
normalised steering value (use `preprocess.action_to_steer` to get radians).
It is a plain MLP with tanh hidden layers, the same shape SB3's PPO actor uses with
net_arch=[256, 256] and activation_fn=nn.Tanh, so the weights can be copied into PPO later.
"""
import copy

import numpy as np
import torch
from torch import nn

from .preprocess import OBS_DIM, PREPROC_VERSION

HIDDEN = (256, 256)


class SteeringMLP(nn.Module):
    """obs (B, OBS_DIM) -> normalised steering (B,). The output is linear, not squashed (like PPO's action mean)."""

    def __init__(self, obs_dim=OBS_DIM, hidden=HIDDEN):
        super().__init__()
        layers, last = [], obs_dim
        for h in hidden:
            layers += [nn.Linear(last, h), nn.Tanh()]
            last = h
        self.body = nn.Sequential(*layers)
        self.head = nn.Linear(last, 1)

    def forward(self, obs):
        return self.head(self.body(obs)).squeeze(-1)


def predict(model, X, batch_size=4096):
    """Run the model on a numpy array, return a numpy array of normalised steering."""
    model.eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(X), batch_size):
            out.append(model(torch.as_tensor(X[i:i + batch_size], dtype=torch.float32)).numpy())
    return np.concatenate(out)


def train_bc(X_train, y_train, X_val=None, y_val=None, epochs=200, lr=1e-3, weight_decay=1e-4,
             batch_size=256, patience=30, seed=0, hidden=HIDDEN, verbose=False):
    """Train a SteeringMLP with MSE loss and Adam.

    If validation data is given, the weights from the epoch with the lowest val loss are returned
    (early stopping: training stops after `patience` epochs without improvement).
    Without validation data it trains for exactly `epochs` epochs (used for the final model).
    Returns (model, history) where history has per-epoch 'train' and 'val' MSE and 'best_epoch'.
    """
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    model = SteeringMLP(X_train.shape[1], hidden)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    # Cosine decay: big steps early, tiny steps at the end so the weights settle instead of bouncing.
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=lr * 0.01)
    loss_fn = nn.MSELoss()

    Xt = torch.as_tensor(X_train, dtype=torch.float32)
    yt = torch.as_tensor(y_train, dtype=torch.float32)
    has_val = X_val is not None
    history = {'train': [], 'val': [], 'best_epoch': epochs - 1}
    best_val, best_state, since_best = np.inf, None, 0

    for epoch in range(epochs):
        model.train()
        order = rng.permutation(len(Xt))             # reshuffle rows every epoch
        for i in range(0, len(order), batch_size):
            idx = order[i:i + batch_size]
            opt.zero_grad()
            loss = loss_fn(model(Xt[idx]), yt[idx])
            loss.backward()
            opt.step()
        sched.step()

        history['train'].append(float(np.mean((predict(model, X_train) - y_train) ** 2)))
        if has_val:
            val_mse = float(np.mean((predict(model, X_val) - y_val) ** 2))
            history['val'].append(val_mse)
            if val_mse < best_val:
                best_val, best_state, since_best = val_mse, copy.deepcopy(model.state_dict()), 0
                history['best_epoch'] = epoch
            else:
                since_best += 1
                if since_best >= patience:
                    break
        if verbose and epoch % 20 == 0:
            print(f'epoch {epoch:3d}  train {history["train"][-1]:.4f}'
                  + (f'  val {history["val"][-1]:.4f}' if has_val else ''))

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, history


def save_model(model, path, **extra):
    """Save weights plus what is needed to rebuild and use the model correctly."""
    torch.save({
        'state_dict': model.state_dict(),
        'obs_dim': model.body[0].in_features,
        'hidden': tuple(m.out_features for m in model.body if isinstance(m, nn.Linear)),
        'preproc_version': PREPROC_VERSION,
        **extra,
    }, path)


def load_model(path):
    """Load a model saved with save_model. Refuses a model built for another preprocessing version."""
    ckpt = torch.load(path, weights_only=False)
    if ckpt['preproc_version'] != PREPROC_VERSION:
        raise ValueError(f'model uses preprocessing v{ckpt["preproc_version"]}, code is v{PREPROC_VERSION}')
    model = SteeringMLP(ckpt['obs_dim'], ckpt['hidden'])
    model.load_state_dict(ckpt['state_dict'])
    model.eval()
    return model, ckpt
