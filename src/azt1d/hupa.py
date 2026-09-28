"""
Loader for the HUPA-UCM Diabetes Dataset's `Preprocessed/` folder: one CSV per
patient, semicolon separated, already on a regular 5-minute grid.

Maps a patient's file onto the canonical schema (azt1d.reference.CANONICAL_COLUMNS)
and cleans it the same way as every other loader (azt1d.cleaning), so the GLIMMER
training and uncertainty code can use it unchanged.

The mapping was checked against MetaboNet's copy of HUPA-UCM (patient 27):
  - glucose, bolus: exact match on every row.
  - basal_rate: exactly equal to MetaboNet's `basal`, which is units delivered in
    that 5-minute row, not a rate. Multiplied by 12 here so Basal means U/hr like
    it does for AZT1D (same conversion azt1d.metabonet applies).
  - carb_input: matches on 99.1% of rows; the raw file is used as is.
The dataset's other columns (calories, heart rate, steps) are not used by GLIMMER.
"""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

from . import cleaning
from . import reference as ref

DEFAULT_DIR = Path("data") / "raw" / "hupa_ucm" / "HUPA-UCM Diabetes Dataset" / "Preprocessed"

_BASAL_TO_RATE = 60 / ref.CGM_INTERVAL_MINUTES  # raw basal is per 5-minute row; canonical is per hour


def subject_file(subject_id: int, directory: Path = DEFAULT_DIR) -> Path:
    return directory / f"HUPA{subject_id:04d}P.csv"


def load_subject(subject_id: int, directory: Path = DEFAULT_DIR) -> pd.DataFrame:
    """One HUPA-UCM patient on the canonical schema, cleaned. subject_id is the number in
    the file name (HUPA0027P.csv -> 27)."""
    raw = pd.read_csv(subject_file(subject_id, directory), sep=";", parse_dates=["time"])
    out = pd.DataFrame(
        {
            "subject_id": subject_id,
            ref.EVENT_DATETIME: raw["time"],
            ref.CGM: raw["glucose"],
            ref.BASAL: raw["basal_rate"] * _BASAL_TO_RATE,
            ref.TOTAL_BOLUS_INSULIN_DELIVERED: raw["bolus_volume_delivered"],
            ref.CARB_SIZE: raw["carb_input"],
        }
    )
    for col in (ref.DEVICE_MODE, ref.BOLUS_TYPE, ref.CORRECTION_DELIVERED, ref.FOOD_DELIVERED):
        out[col] = pd.NA
    return cleaning.clean_canonical_frame(out[["subject_id", *ref.CANONICAL_COLUMNS]])


def list_subject_ids(directory: Path = DEFAULT_DIR) -> list[int]:
    return sorted(int(re.search(r"HUPA(\d+)P", p.stem).group(1)) for p in directory.glob("HUPA*P.csv"))
