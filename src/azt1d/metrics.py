"""Standard glycemic metrics computed from a loaded AZT1D-shaped DataFrame."""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import reference as ref


def time_in_range(df: pd.DataFrame, low: int = ref.HYPO_THRESHOLD, high: int = ref.HYPER_THRESHOLD) -> pd.Series:
    """Percent of CGM readings below/within/above [low, high], per subject_id."""
    cgm = df[ref.CGM]
    band = pd.cut(cgm, bins=[-np.inf, low - 1e-9, high, np.inf], labels=["hypo", "in_range", "hyper"])
    counts = pd.crosstab(df["subject_id"], band, normalize="index") * 100
    return counts.reindex(columns=["hypo", "in_range", "hyper"], fill_value=0)


def glucose_management_indicator(df: pd.DataFrame) -> pd.Series:
    """GMI (%): an estimated A1c derived from mean CGM (Bergenstal et al. 2018)."""
    mean_cgm = df.groupby("subject_id")[ref.CGM].mean()
    return 3.31 + 0.02392 * mean_cgm


def coefficient_of_variation(df: pd.DataFrame) -> pd.Series:
    """CGM coefficient of variation (%) per subject; >36% flags high glycemic variability."""
    grouped = df.groupby("subject_id")[ref.CGM]
    return (grouped.std() / grouped.mean()) * 100


def daily_carb_and_bolus(df: pd.DataFrame) -> pd.DataFrame:
    """Per-subject, per-day totals for carbs and bolus insulin."""
    daily = df.copy()
    daily["date"] = daily[ref.EVENT_DATETIME].dt.date
    return daily.groupby(["subject_id", "date"]).agg(
        total_carbs_g=(ref.CARB_SIZE, "sum"),
        total_bolus_u=(ref.TOTAL_BOLUS_INSULIN_DELIVERED, "sum"),
        total_basal_u=(ref.BASAL, lambda s: s.sum() * ref.CGM_INTERVAL_MINUTES / 60),
        mean_cgm=(ref.CGM, "mean"),
    ).reset_index()


def device_mode_distribution(df: pd.DataFrame) -> pd.Series:
    """
    Percent of time spent in each device mode, across the whole dataset.

    The real files only log DeviceMode explicitly for "sleep"/"exercise" and
    leave it blank the rest of the time (~78% of rows) -- blank is treated here
    as "regular", so percentages are of the full dataset, not just logged rows.
    """
    filled = df[ref.DEVICE_MODE].fillna("regular")
    return filled.value_counts(normalize=True) * 100


def summary_table(df: pd.DataFrame) -> pd.DataFrame:
    """One row per subject: TIR bands, GMI, CV, and reading count."""
    tir = time_in_range(df)
    out = tir.copy()
    out["gmi_pct"] = glucose_management_indicator(df)
    out["cv_pct"] = coefficient_of_variation(df)
    out["n_readings"] = df.groupby("subject_id")[ref.CGM].size()
    return out.round(1)
