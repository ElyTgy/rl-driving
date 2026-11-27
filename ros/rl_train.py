#!/usr/bin/env python3
"""Train the steering policy with PPO (Stable-Baselines3) in the ROS 2 F1TENTH sim.

Run with the ROS Python, with the sim already running (no safety node / arbiter / bc_drive):
    python3 ros/rl_train.py --start-poses poses.csv                    # smoke test, from scratch
    python3 ros/rl_train.py --start-poses poses.csv --centerline cl.csv \
        --init-policy checkpoints/bc_v2.npz --timesteps 200000          # warm start from BC

One-time install into the ROS venv:
    sim_ws/.venv/bin/pip install gymnasium stable-baselines3 torch --index-url https://download.pytorch.org/whl/cpu
(torch from that index, the rest from PyPI: install torch first, then `pip install gymnasium stable-baselines3`.)
"""
import argparse
import os
import sys

import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rl_env import F1TenthROSEnv, OBS_DIM  # noqa: E402


def load_table(path):
    """Read a .npy or .csv file into an array."""
    return np.load(path) if path.endswith('.npy') else np.loadtxt(path, delimiter=',')


def read_bc_layers(path):
    """Return [(W (in,out), b (out,)), ...] from the BC .npz, assuming arrays are stored in layer
    order (W0, b0, W1, b1, ...). Adjust here if your export_policy.py uses another layout."""
    d = np.load(path, allow_pickle=False)
    print('keys in', path, ':', [(k, d[k].shape) for k in d.files])
    mats = [d[k] for k in d.files if d[k].ndim == 2]
    biases = [d[k] for k in d.files if d[k].ndim == 1]
    if len(mats) != len(biases):
        raise SystemExit('could not pair weights and biases; edit read_bc_layers() for your npz layout')
    layers, in_dim = [], OBS_DIM
    for W, b in zip(mats, biases):
        W = W if W.shape[0] == in_dim else W.T                  # make it (in, out)
        if W.shape[0] != in_dim or W.shape[1] != b.shape[0]:
            raise SystemExit(f'layer shapes do not chain: {W.shape}, bias {b.shape}, expected in={in_dim}')
        layers.append((W.astype(np.float32), b.astype(np.float32)))
        in_dim = W.shape[1]
    if in_dim != 1:
        raise SystemExit(f'BC output size is {in_dim}, this env expects 1 (steering only)')
    return layers


def copy_bc_into_ppo(model, layers):
    """Copy BC weights into the PPO actor: hidden layers -> policy_net, last layer -> action_net."""
    linears = [m for m in model.policy.mlp_extractor.policy_net if isinstance(m, torch.nn.Linear)]
    linears.append(model.policy.action_net)
    assert len(linears) == len(layers), (len(linears), len(layers))
    with torch.no_grad():
        for lin, (W, b) in zip(linears, layers):
            lin.weight.copy_(torch.from_numpy(W.T))             # torch stores (out, in)
            lin.bias.copy_(torch.from_numpy(b))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--start-poses', required=True, help='csv/npy, rows of x,y,yaw')
    p.add_argument('--centerline', default=None, help='csv/npy, rows of x,y in track order')
    p.add_argument('--init-policy', default=None, help='BC .npz to warm-start the actor')
    p.add_argument('--activation', default='tanh', choices=['tanh', 'relu'], help='must match the BC net')
    p.add_argument('--timesteps', type=int, default=200_000)
    p.add_argument('--speed', type=float, default=1.0)
    p.add_argument('--lr', type=float, default=1e-4)
    p.add_argument('--out-dir', default=os.path.expanduser('~/rl_runs/run1'))
    args = p.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    env = F1TenthROSEnv(load_table(args.start_poses),
                        None if args.centerline is None else load_table(args.centerline),
                        speed=args.speed)

    policy_kwargs = dict(activation_fn=torch.nn.Tanh if args.activation == 'tanh' else torch.nn.ReLU,
                         log_std_init=-2.5)                     # small exploration noise
    layers = None
    if args.init_policy:
        layers = read_bc_layers(args.init_policy)
        hidden = [W.shape[1] for W, _ in layers[:-1]]
        policy_kwargs['net_arch'] = dict(pi=hidden, vf=hidden)  # actor must match the BC MLP

    model = PPO('MlpPolicy', env, learning_rate=args.lr, clip_range=0.1, n_steps=1024, batch_size=128,
                policy_kwargs=policy_kwargs, tensorboard_log=None, verbose=1)
    if layers:
        copy_bc_into_ppo(model, layers)
        print('copied BC weights into the PPO actor')

    ckpt = CheckpointCallback(save_freq=20_000, save_path=args.out_dir, name_prefix='ppo')
    try:
        model.learn(total_timesteps=args.timesteps, callback=ckpt)
    except KeyboardInterrupt:
        print('interrupted, saving')
    finally:
        model.save(os.path.join(args.out_dir, 'ppo_final'))
        env.close()


if __name__ == '__main__':
    main()