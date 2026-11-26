"""Scoring BC predictions, and the linear (ridge) baseline. numpy only, so it is cheap to import.

Predictions and labels are normalised actions: column 0 steering (rad / STEER_MAX), column 1 speed
(see preprocess.speed_to_action). Errors are reported in rad for steering and m/s for speed.
"""
import numpy as np

from .preprocess import SPEED_MAX, SPEED_MIN, STEER_MAX

# One action unit in physical units: steering in rad, speed in m/s.
STEER_UNIT = STEER_MAX
SPEED_UNIT = (SPEED_MAX - SPEED_MIN) / 2


def _scores(pred, true, unit):
    err = (pred - true) * unit
    return np.sqrt(np.mean(err ** 2)), np.mean(np.abs(err)), 1 - np.sum((pred - true) ** 2) / np.sum((true - true.mean()) ** 2)


def metrics(pred, true, drift_rows):
    """Steering: RMSE and MAE in rad, R², and MAE on drift rows (the expert's corrections).
    Speed (when there is a second column): speed_rmse and speed_mae in m/s, speed_r2."""
    pred, true = pred.reshape(len(pred), -1), true.reshape(len(true), -1)
    rmse, mae, r2 = _scores(pred[:, 0], true[:, 0], STEER_UNIT)
    drift = drift_rows == 1
    result = {
        'rmse': float(rmse), 'mae': float(mae), 'r2': float(r2),
        'mae_drift': float(np.mean(np.abs(pred[drift, 0] - true[drift, 0])) * STEER_UNIT) if drift.any() else np.nan,
    }
    if true.shape[1] > 1:
        rmse, mae, r2 = _scores(pred[:, 1], true[:, 1], SPEED_UNIT)
        result.update(speed_rmse=float(rmse), speed_mae=float(mae), speed_r2=float(r2))
    return result


def fit_ridge(X_tr, y_tr, lam=1e-2):
    """Ridge regression with a bias term. Returns a function X -> predictions."""
    A = np.c_[X_tr, np.ones(len(X_tr))]
    w = np.linalg.solve(A.T @ A + lam * np.eye(A.shape[1]), A.T @ y_tr)
    return lambda X_: np.c_[X_, np.ones(len(X_))] @ w
