"""
Per-patient training loop for the v0 CNN-LSTM baseline: plain MSE loss, no
region weighting yet (that's v1). Mirrors the paper's per-patient
personalization -- one model per subject, not one pooled model.

Assumptions not stated in the paper: Adam optimizer, MSE training loss (RMSE/MAE
are only the *evaluation* metrics the paper names), learning rate, batch size,
epoch count, and early-stopping patience are all ours, plus z-score scaling of
the input features (fit on each subject's training split only). Revisit these
once the real dataset is in and v0's numbers can be sanity-checked.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from torch import nn

from . import features as feat
from . import sequences as seq
from .model import CNNLSTM, count_parameters

DEFAULT_EPOCHS = 30
DEFAULT_BATCH_SIZE = 64
DEFAULT_LR = 1e-3
DEFAULT_PATIENCE = 5


def get_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


@dataclass
class SubjectResult:
    subject_id: int
    rmse: float
    mae: float
    n_train: int
    n_val: int
    n_test: int
    n_params: int
    history: list[tuple[float, float]]  # (train_loss, val_loss) per epoch
    y_test: np.ndarray
    y_pred: np.ndarray


def _iter_batches(X: np.ndarray, y: np.ndarray, batch_size: int, shuffle: bool, rng: np.random.Generator):
    n = len(X)
    idx = rng.permutation(n) if shuffle else np.arange(n)
    for start in range(0, n, batch_size):
        sel = idx[start : start + batch_size]
        yield X[sel], y[sel]


def train_subject_model(
    df_subject: pd.DataFrame,
    lookback: int = seq.LOOKBACK,
    horizon: int = seq.HORIZON,
    epochs: int = DEFAULT_EPOCHS,
    batch_size: int = DEFAULT_BATCH_SIZE,
    lr: float = DEFAULT_LR,
    patience: int = DEFAULT_PATIENCE,
    device: torch.device | None = None,
    seed: int = 0,
) -> SubjectResult:
    device = device or get_device()
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)

    enriched = feat.add_engineered_features(df_subject)
    X, y = seq.make_windows(enriched, feat.FEATURE_COLUMNS, feat.TARGET_COLUMN, lookback, horizon)
    splits = seq.chronological_split(X, y)

    n_features = X.shape[-1]
    scaler = StandardScaler().fit(splits["X_train"].reshape(-1, n_features))

    def scale(arr: np.ndarray) -> np.ndarray:
        return scaler.transform(arr.reshape(-1, n_features)).reshape(arr.shape).astype(np.float32)

    X_train, X_val, X_test = (scale(splits[k]) for k in ("X_train", "X_val", "X_test"))
    y_train, y_val, y_test = splits["y_train"], splits["y_val"], splits["y_test"]

    # Targets are raw mg/dL (~40-400): with the output layer starting near 0,
    # unscaled targets gave gradients too small relative to the loss surface to
    # move the bias off ~0 within a normal epoch budget (observed: the model
    # collapsed to predicting a single near-constant value). z-score the target
    # for training and invert it only for reporting, so RMSE/MAE stay in mg/dL.
    y_mean, y_std = float(y_train.mean()), float(y_train.std())
    y_train_z = (y_train - y_mean) / y_std
    y_val_z = (y_val - y_mean) / y_std

    model = CNNLSTM(n_features=n_features).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    X_val_t = torch.from_numpy(X_val).to(device)
    y_val_t = torch.from_numpy(y_val_z).to(device)

    best_val = float("inf")
    best_state = copy.deepcopy(model.state_dict())
    stale_epochs = 0
    history: list[tuple[float, float]] = []

    for _ in range(epochs):
        model.train()
        train_losses = []
        for xb, yb in _iter_batches(X_train, y_train_z, batch_size, shuffle=True, rng=rng):
            optimizer.zero_grad()
            pred = model(torch.from_numpy(xb).to(device))
            loss = loss_fn(pred, torch.from_numpy(yb).to(device))
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())

        model.eval()
        with torch.no_grad():
            val_loss = loss_fn(model(X_val_t), y_val_t).item()
        history.append((float(np.mean(train_losses)), val_loss))

        if val_loss < best_val:
            best_val = val_loss
            best_state = copy.deepcopy(model.state_dict())
            stale_epochs = 0
        else:
            stale_epochs += 1
            if stale_epochs >= patience:
                break

    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        y_pred_z = model(torch.from_numpy(X_test).to(device)).cpu().numpy()
    y_pred = y_pred_z * y_std + y_mean

    rmse = float(np.sqrt(np.mean((y_test - y_pred) ** 2)))
    mae = float(np.mean(np.abs(y_test - y_pred)))

    return SubjectResult(
        subject_id=int(df_subject["subject_id"].iloc[0]),
        rmse=rmse,
        mae=mae,
        n_train=len(y_train),
        n_val=len(y_val),
        n_test=len(y_test),
        n_params=count_parameters(model),
        history=history,
        y_test=y_test,
        y_pred=y_pred,
    )


def run_baseline_v0(
    df_all: pd.DataFrame, subject_ids: list[int] | None = None, **kwargs
) -> tuple[pd.DataFrame, dict[int, SubjectResult]]:
    """Train one v0 CNN-LSTM per subject; return a Table-4-shaped summary plus raw results."""
    if subject_ids is None:
        subject_ids = sorted(df_all["subject_id"].unique())

    results: dict[int, SubjectResult] = {}
    rows = []
    for sid in subject_ids:
        df_subject = df_all[df_all["subject_id"] == sid].reset_index(drop=True)
        try:
            result = train_subject_model(df_subject, **kwargs)
        except ValueError as exc:
            print(f"Subject {sid}: skipped ({exc})")
            continue
        results[sid] = result
        rows.append(
            {"subject_id": sid, "rmse": result.rmse, "mae": result.mae, "n_test": result.n_test}
        )

    summary = pd.DataFrame(rows).set_index("subject_id")
    return summary, results
