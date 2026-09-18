"""
Sliding-window sequence construction and the paper's chronological split.

Assumption: the paper never states the input lookback window length (only that
the prediction horizon is 60 minutes). We default to 24 steps = 2 hours of
history at 5-minute sampling; treat LOOKBACK as a knob to revisit once the real
data is in and we can see how sensitive results are to it.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

LOOKBACK = 24  # 2 hours of history -- our assumption, not stated in the paper.
HORIZON = 12  # 60 minutes ahead at 5-minute sampling, as stated in the paper.


def make_windows(
    df: pd.DataFrame,
    feature_columns: list[str],
    target_column: str,
    lookback: int = LOOKBACK,
    horizon: int = HORIZON,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Build (X, y) for one contiguous subject's timeline: X[i] is `lookback` past
    timesteps of `feature_columns`, y[i] is `target_column` `horizon` steps after
    the end of that window.

    Does not attempt to detect or exclude sensor gaps within a subject's stream
    -- a known v0 simplification, since the paper doesn't detail its own
    handling of missing CGM readings.
    """
    features = df[feature_columns].to_numpy(dtype=np.float32)
    target = df[target_column].to_numpy(dtype=np.float32)

    n = len(df) - lookback - horizon + 1
    if n <= 0:
        raise ValueError(f"Not enough rows ({len(df)}) for lookback={lookback} + horizon={horizon}")

    X = np.stack([features[i : i + lookback] for i in range(n)])
    y = target[lookback + horizon - 1 : lookback + horizon - 1 + n]
    return X, y


def make_multi_output_windows(
    df: pd.DataFrame,
    feature_columns: list[str],
    target_columns: list[str],
    lookback: int = LOOKBACK,
    horizon: int = 1,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Like make_windows, but Y has one column per entry in target_columns
    instead of a single target -- for a model predicting several quantities
    at once (e.g. every recursive-rollout input, not just CGM) rather than
    one 60-minutes-ahead value.
    """
    features = df[feature_columns].to_numpy(dtype=np.float32)
    targets = df[target_columns].to_numpy(dtype=np.float32)

    n = len(df) - lookback - horizon + 1
    if n <= 0:
        raise ValueError(f"Not enough rows ({len(df)}) for lookback={lookback} + horizon={horizon}")

    X = np.stack([features[i : i + lookback] for i in range(n)])
    Y = targets[lookback + horizon - 1 : lookback + horizon - 1 + n]
    return X, Y


def chronological_split(
    X: np.ndarray, y: np.ndarray, test_size: float = 0.2, val_size: float = 0.2
) -> dict[str, np.ndarray]:
    """
    Paper's split (Section 4.3): chronologically hold out the last `test_size`
    fraction as test, then split the remaining training portion the same way
    into train/val. No shuffling anywhere -- these are time series.
    """
    n = len(X)
    n_test = int(round(n * test_size))
    n_trainval = n - n_test

    X_trainval, y_trainval = X[:n_trainval], y[:n_trainval]
    X_test, y_test = X[n_trainval:], y[n_trainval:]

    n_val = int(round(n_trainval * val_size))
    n_train = n_trainval - n_val

    return {
        "X_train": X_trainval[:n_train],
        "y_train": y_trainval[:n_train],
        "X_val": X_trainval[n_train:],
        "y_val": y_trainval[n_train:],
        "X_test": X_test,
        "y_test": y_test,
    }
