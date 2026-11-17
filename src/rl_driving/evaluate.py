"""Scoring steering predictions, and the linear (ridge) baseline. numpy only, so it is cheap to import.

Predictions and labels are normalised steering (rad / STEER_MAX); errors are reported in radians.
"""
import numpy as np

from .preprocess import STEER_MAX


def metrics(pred, true, drift_rows):
    """RMSE and MAE in rad, R², and MAE on the rows where drift was active (the expert's corrections)."""
    err = (pred - true) * STEER_MAX                   # back to radians
    ss_res = np.sum((pred - true) ** 2)
    ss_tot = np.sum((true - true.mean()) ** 2)
    return {
        'rmse': float(np.sqrt(np.mean(err ** 2))),
        'mae': float(np.mean(np.abs(err))),
        'r2': float(1 - ss_res / ss_tot),
        'mae_drift': float(np.mean(np.abs(err[drift_rows == 1]))) if drift_rows.any() else np.nan,
    }


def fit_ridge(X_tr, y_tr, lam=1e-2):
    """Ridge regression with a bias term. Returns a function X -> predictions."""
    A = np.c_[X_tr, np.ones(len(X_tr))]
    w = np.linalg.solve(A.T @ A + lam * np.eye(A.shape[1]), A.T @ y_tr)
    return lambda X_: np.c_[X_, np.ones(len(X_))] @ w
