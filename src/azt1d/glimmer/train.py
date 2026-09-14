"""
Per-patient training loop, shared across both architectures and both loss
variants. Mirrors the paper's per-patient personalization -- one model per
subject, not one pooled model.

Plain MSE loss (v0, region_weights=None) and the paper's region-aware weighted
loss (v1, region_weights given -- see azt1d.glimmer.losses) share this same
function, since everything else about training is identical between the two.
Same for architecture: architecture="cnn_lstm" (default) or "cnn_transformer"
(see azt1d.glimmer.model) selects the backbone; nothing else changes.

Data prep (windowing, scaling, train/val/test split) is split out into
prepare_subject_data() so it only runs once per subject even when a caller
needs many trainings for the same subject, e.g. the genetic algorithm search
in azt1d.glimmer.ga, which trains hundreds of candidate models per patient and
would otherwise redo the same windowing/scaling work every single time.

Assumptions not stated in the paper: Adam optimizer, MSE as the v0 training
loss (RMSE/MAE are only the *evaluation* metrics the paper names), learning
rate, batch size, epoch count, and early-stopping patience are all ours, plus
z-score scaling of the input features and target (fit on each subject's
training split only).
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from torch import nn

from .. import reference as ref
from . import features as feat
from . import sequences as seq
from .losses import region_weighted_loss
from .model import MODEL_CLASSES, count_parameters

DEFAULT_EPOCHS = 30
DEFAULT_BATCH_SIZE = 64
DEFAULT_LR = 1e-3
DEFAULT_PATIENCE = 5


def get_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


@dataclass
class PreparedSubjectData:
    """Windowed, split, and scaled data for one subject -- reusable across many trainings."""

    subject_id: int
    X_train: np.ndarray
    X_val: np.ndarray
    X_test: np.ndarray
    y_train: np.ndarray  # raw mg/dL
    y_val: np.ndarray
    y_test: np.ndarray
    y_mean: float
    y_std: float
    n_features: int


@dataclass
class SubjectResult:
    subject_id: int
    architecture: str
    region_weights: dict[str, float] | None
    rmse: float
    mae: float
    val_rmse: float  # mg/dL, the metric checkpoints/GA fitness are selected on
    n_train: int
    n_val: int
    n_test: int
    n_params: int
    n_features: int  # needed alongside architecture to reconstruct the model from model_state_dict
    y_mean: float  # needed to reproduce the same z-scoring if reusing the model for fresh inference
    y_std: float
    history: list[tuple[float, float]]  # (train_loss, val_loss) per epoch
    y_test: np.ndarray
    y_pred: np.ndarray
    model_state_dict: dict  # the actual trained weights -- see azt1d.glimmer.checkpoint.load_model


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


def prepare_subject_data(
    df_subject: pd.DataFrame, lookback: int = seq.LOOKBACK, horizon: int = seq.HORIZON
) -> PreparedSubjectData:
    enriched = feat.add_engineered_features(df_subject)
    X, y = seq.make_windows(enriched, feat.FEATURE_COLUMNS, feat.TARGET_COLUMN, lookback, horizon)
    splits = seq.chronological_split(X, y)

    n_features = X.shape[-1]
    scaler = StandardScaler().fit(splits["X_train"].reshape(-1, n_features))

    def scale(arr: np.ndarray) -> np.ndarray:
        return scaler.transform(arr.reshape(-1, n_features)).reshape(arr.shape).astype(np.float32)

    X_train, X_val, X_test = (scale(splits[k]) for k in ("X_train", "X_val", "X_test"))

    return PreparedSubjectData(
        subject_id=int(df_subject["subject_id"].iloc[0]),
        X_train=X_train,
        X_val=X_val,
        X_test=X_test,
        y_train=splits["y_train"],
        y_val=splits["y_val"],
        y_test=splits["y_test"],
        y_mean=float(splits["y_train"].mean()),
        y_std=float(splits["y_train"].std()),
        n_features=n_features,
    )


def train_prepared_model(
    data: PreparedSubjectData,
    architecture: str = "cnn_lstm",
    epochs: int = DEFAULT_EPOCHS,
    batch_size: int = DEFAULT_BATCH_SIZE,
    lr: float = DEFAULT_LR,
    patience: int = DEFAULT_PATIENCE,
    device: torch.device | None = None,
    seed: int = 0,
    region_weights: dict[str, float] | None = None,
) -> SubjectResult:
    """
    Train one model on already-prepared data (see prepare_subject_data).
    architecture: "cnn_lstm" or "cnn_transformer" (azt1d.glimmer.model).
    region_weights=None trains on plain MSE (v0). Passing a dict with
    w_hypo/w_normal/w_hyper (e.g. azt1d.reference.GLIMMER_PAPER_WEIGHTS)
    switches to the paper's region-weighted loss (v1).
    """
    device = device or get_device()
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)

    y_mean, y_std = data.y_mean, data.y_std
    y_train_z = (data.y_train - y_mean) / y_std
    y_val_z = (data.y_val - y_mean) / y_std

    model_class = MODEL_CLASSES[architecture]
    model = model_class(n_features=data.n_features).to(device)
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

    X_val_t = torch.from_numpy(data.X_val).to(device)
    y_val_t = torch.from_numpy(y_val_z).to(device)

    best_val = float("inf")
    best_state = copy.deepcopy(model.state_dict())
    stale_epochs = 0
    history: list[tuple[float, float]] = []

    for _ in range(epochs):
        model.train()
        train_losses = []
        for xb, yb in _iter_batches(data.X_train, y_train_z, batch_size, shuffle=True, rng=rng):
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
            val_rmse_z = torch.sqrt(torch.mean((val_pred_z - y_val_t) ** 2)).item()
        history.append((float(np.mean(train_losses)), val_loss))

        if val_rmse_z < best_val:
            best_val = val_rmse_z
            best_state = copy.deepcopy(model.state_dict())
            stale_epochs = 0
        else:
            stale_epochs += 1
            if stale_epochs >= patience:
                break

    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        y_pred_z = model(torch.from_numpy(data.X_test).to(device)).cpu().numpy()
    y_pred = y_pred_z * y_std + y_mean

    rmse = float(np.sqrt(np.mean((data.y_test - y_pred) ** 2)))
    mae = float(np.mean(np.abs(data.y_test - y_pred)))

    # Move to CPU before storing: checkpoints should load fine on a machine
    # (or a session) without the same GPU/MPS device available.
    cpu_state = {k: v.cpu() for k, v in best_state.items()}

    return SubjectResult(
        subject_id=data.subject_id,
        architecture=architecture,
        region_weights=region_weights,
        rmse=rmse,
        mae=mae,
        val_rmse=best_val * y_std,
        n_train=len(data.y_train),
        n_val=len(data.y_val),
        n_test=len(data.y_test),
        n_params=count_parameters(model),
        n_features=data.n_features,
        y_mean=y_mean,
        y_std=y_std,
        history=history,
        y_test=data.y_test,
        y_pred=y_pred,
        model_state_dict=cpu_state,
    )


def train_subject_model(
    df_subject: pd.DataFrame,
    lookback: int = seq.LOOKBACK,
    horizon: int = seq.HORIZON,
    architecture: str = "cnn_lstm",
    epochs: int = DEFAULT_EPOCHS,
    batch_size: int = DEFAULT_BATCH_SIZE,
    lr: float = DEFAULT_LR,
    patience: int = DEFAULT_PATIENCE,
    device: torch.device | None = None,
    seed: int = 0,
    region_weights: dict[str, float] | None = None,
) -> SubjectResult:
    """Convenience wrapper: prepare_subject_data() + train_prepared_model() in one call."""
    data = prepare_subject_data(df_subject, lookback, horizon)
    return train_prepared_model(
        data,
        architecture=architecture,
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
        patience=patience,
        device=device,
        seed=seed,
        region_weights=region_weights,
    )


def run_baseline_v0(
    df_all: pd.DataFrame, subject_ids: list[int] | None = None, **kwargs
) -> tuple[pd.DataFrame, dict[int, SubjectResult]]:
    """
    Train one model per subject; return a Table-4-shaped summary plus raw
    results. Pass region_weights=None (default) for v0, or a weights dict for
    v1; pass architecture="cnn_transformer" for the second backbone -- see
    train_subject_model.
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


def run_with_checkpoints(
    df_all: pd.DataFrame,
    checkpoint_dir: Path | str,
    subject_ids: list[int] | None = None,
    force: bool = False,
    **kwargs,
) -> tuple[pd.DataFrame, dict[int, SubjectResult]]:
    """
    Same as run_baseline_v0, but checkpointed: each subject's trained model
    and predictions are saved to checkpoint_dir right after that subject
    finishes, and loaded from there instead of retrained on a later call if
    already present. Safe to interrupt (a crash, a stopped kernel) and re-run
    the same cell -- already-finished subjects are skipped, not redone.

    Pass force=True to retrain everyone and overwrite existing checkpoints
    (e.g. after a code change that should invalidate old results).
    """
    from . import checkpoint  # local import: checkpoint.py -> model.py only, no cycle, but keeps it optional

    checkpoint_dir = Path(checkpoint_dir)
    if subject_ids is None:
        subject_ids = sorted(df_all["subject_id"].unique())

    results: dict[int, SubjectResult] = {}
    rows = []
    for sid in subject_ids:
        cached = None if force else checkpoint.load_result(checkpoint_dir, sid)
        if cached is not None:
            result = cached
            print(f"Subject {sid}: loaded from checkpoint (RMSE={result.rmse:.2f} mg/dL)")
        else:
            df_subject = df_all[df_all["subject_id"] == sid].reset_index(drop=True)
            try:
                result = train_subject_model(df_subject, **kwargs)
            except ValueError as exc:
                print(f"Subject {sid}: skipped ({exc})")
                continue
            checkpoint.save_result(checkpoint_dir, result)
            print(f"Subject {sid}: trained and checkpointed (RMSE={result.rmse:.2f} mg/dL)")

        results[sid] = result
        rows.append({"subject_id": sid, "rmse": result.rmse, "mae": result.mae, "n_test": result.n_test})

    summary = pd.DataFrame(rows).set_index("subject_id")
    return summary, results
