"""
Per-patient training loop for the CNN-LSTM forecaster. Mirrors the paper's
per-patient personalization -- one model per subject, not one pooled model.

Plain MSE loss (v0, region_weights=None) and the paper's region-aware weighted
loss (v1, region_weights given -- see azt1d.glimmer.losses) share this same
function, since everything else about training is identical between the two.

Assumptions not stated in the paper: Adam optimizer, MSE as the v0 training
loss (RMSE/MAE are only the *evaluation* metrics the paper names), learning
rate, batch size, epoch count, and early-stopping patience are all ours, plus
z-score scaling of the input features and target (fit on each subject's
training split only).
"""

from __future__ import annotations

import copy
from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from torch import nn

from .. import reference as ref
from . import features as feat
from . import sequences as seq
from .losses import region_weighted_loss
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


def region_errors(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, dict[str, float]]:
    """
    RMSE/MAE broken out by clinical region (hypo/normal/hyper), computed on the
    *true* value. This is the metric that actually tests GLIMMER's point: does
    the weighted loss reduce error specifically where it's dangerous, not just
    on average?
    """
    regions = {
        "hypo": y_true < ref.HYPO_THRESHOLD,
        "normal": (y_true >= ref.HYPO_THRESHOLD) & (y_true <= ref.HYPER_THRESHOLD),
        "hyper": y_true > ref.HYPER_THRESHOLD,
    }
    out = {}
    for name, mask in regions.items():
        if mask.sum() == 0:
            out[name] = {"rmse": float("nan"), "mae": float("nan"), "n": 0}
            continue
        err = y_true[mask] - y_pred[mask]
        out[name] = {"rmse": float(np.sqrt(np.mean(err**2))), "mae": float(np.mean(np.abs(err))), "n": int(mask.sum())}
    return out


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
    region_weights: dict[str, float] | None = None,
) -> SubjectResult:
    """
    region_weights=None trains on plain MSE (v0). Passing a dict with
    w_hypo/w_normal/w_hyper (e.g. azt1d.reference.GLIMMER_PAPER_WEIGHTS)
    switches to the paper's region-weighted loss (v1).
    """
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

    if region_weights is None:
        loss_fn = nn.MSELoss()
    else:
        hypo_z = (ref.HYPO_THRESHOLD - y_mean) / y_std
        hyper_z = (ref.HYPER_THRESHOLD - y_mean) / y_std

        def loss_fn(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
            return region_weighted_loss(
                pred, target, hypo_z, hyper_z,
                region_weights["w_hypo"], region_weights["w_normal"], region_weights["w_hyper"],
            )

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
            val_pred_z = model(X_val_t)
            val_loss = loss_fn(val_pred_z, y_val_t).item()
            # Checkpoint selection always uses validation RMSE, not val_loss
            # directly. For v0 (plain MSE) these give identical epoch ordering,
            # since RMSE is just sqrt(MSE). For v1 (region-weighted loss) they
            # don't: with hypo/hyper samples rare in a small validation split,
            # the weighted loss swings a lot epoch to epoch depending on how
            # the model happens to do on that handful of points, and picking
            # checkpoints on it is close to picking at random. RMSE is smooth
            # and matches what the paper itself uses as its fitness metric for
            # weight search, so it's the more stable stand-in here too.
            val_rmse = torch.sqrt(torch.mean((val_pred_z - y_val_t) ** 2)).item()
        history.append((float(np.mean(train_losses)), val_loss))

        if val_rmse < best_val:
            best_val = val_rmse
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
    """
    Train one CNN-LSTM per subject; return a Table-4-shaped summary plus raw
    results. Pass region_weights=None (default) for v0, or a weights dict for
    v1 -- see train_subject_model.
    """
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
