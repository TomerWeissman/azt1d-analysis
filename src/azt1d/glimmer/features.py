"""
The six input features GLIMMER's paper names (Section 4.3): CGM, basal insulin,
bolus insulin, carb intake, a 200-point moving average of CGM, and a glucose
region label (hypo/normal/hyper). The first four already exist in the loaded
schema (azt1d.reference); this module adds the two engineered ones.

Assumption: the paper doesn't say how the region label is encoded as a model
input. We use a single integer channel (0=hypo, 1=normal, 2=hyper) rather than
one-hot, since their Fig. 2 diagram shows it as one feature slot alongside the
five scalar features (six boxes total, not eight).
"""

from __future__ import annotations

import pandas as pd

from .. import reference as ref

MA_COLUMN = "cgm_ma200"
REGION_COLUMN = "region_label"
MA_WINDOW = 200  # ~16-17 hours at 5-minute sampling, per the paper.

FEATURE_COLUMNS = [
    ref.CGM,
    ref.BASAL,
    ref.TOTAL_BOLUS_INSULIN_DELIVERED,
    ref.CARB_SIZE,
    MA_COLUMN,
    REGION_COLUMN,
]
TARGET_COLUMN = ref.CGM


def add_engineered_features(df: pd.DataFrame, ma_window: int = MA_WINDOW) -> pd.DataFrame:
    """Add the moving-average and region-label columns to one subject's data.

    Expects `df` to already be sorted by EventDateTime (azt1d.loading does this).
    """
    out = df.copy()
    out[MA_COLUMN] = out[ref.CGM].rolling(window=ma_window, min_periods=1).mean()
    out[REGION_COLUMN] = pd.cut(
        out[ref.CGM],
        bins=[-float("inf"), ref.HYPO_THRESHOLD, ref.HYPER_THRESHOLD, float("inf")],
        labels=[0, 1, 2],
    ).astype(int)
    return out
