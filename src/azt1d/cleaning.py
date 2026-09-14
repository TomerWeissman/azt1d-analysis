"""
Cleaning steps that apply to any canonical-schema DataFrame, regardless of
which loader produced it (azt1d.loading for AZT1D's own raw files, or
azt1d.metabonet for any of the 14 datasets it bundles). Extracted out of
azt1d.loading so the GLIMMER training pipeline doesn't need to special-case
which loader a given DataFrame came from -- it just needs the canonical
schema, cleaned the same way every time.

Two things basically every insulin-pump dataset needs, for the same
underlying reason: basal is delivered continuously but only logged when it
changes (so it needs forward-filling to be usable as a per-timestep feature),
and "no event" bolus/carb readings come through as missing rather than zero
(so they need zero-filling to mean what they actually mean: nothing happened).
"""

from __future__ import annotations

import pandas as pd

from . import reference as ref

# Basal is a rate (U/hr); values far above what any real pump delivers are
# export glitches, not real doses. Same threshold loading.py used for AZT1D's
# raw files -- a physiological bound, not something specific to that dataset.
BASAL_MAX_PLAUSIBLE_U_PER_HR = 15.0


def clean_canonical_frame(df: pd.DataFrame) -> pd.DataFrame:
    """
    df must already have azt1d.reference.CANONICAL_COLUMNS plus subject_id.
    Safe to call on a multi-subject DataFrame -- basal forward-fill is done
    per subject and never crosses a subject boundary.
    """
    df = df.copy()

    # A missing CGM reading can't be trained on or evaluated against; AZT1D's
    # own raw files never have one, but a harmonized source (MetaboNet) can
    # carry a placeholder row (kept for grid regularity) with no real reading.
    # Dropping rather than interpolating: we don't fabricate a glucose value
    # that was never measured.
    df = df.dropna(subset=[ref.CGM])

    for col in ref.ZERO_FILLED_COLUMNS:
        df[col] = df[col].fillna(0)

    df.loc[df[ref.BASAL] > BASAL_MAX_PLAUSIBLE_U_PER_HR, ref.BASAL] = float("nan")

    df = df.sort_values(["subject_id", ref.EVENT_DATETIME])
    df[ref.BASAL] = df.groupby("subject_id")[ref.BASAL].transform(lambda s: s.ffill().bfill())

    return df.reset_index(drop=True)
