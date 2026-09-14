"""
Loading utilities for the real AZT1D dataset.

Confirmed layout (from the actual download): a top-level "CGM Records" folder
with one subfolder per subject ("Subject 1" .. "Subject 25"), each containing a
single "Subject N.csv" with exactly the column names the paper describes. The
Mendeley download is a single zip, but a user may instead hand us an
already-extracted folder (as happened here) -- discovery and load_real_dataset
both handle either case.

Several things the real CSVs do that the paper's text doesn't fully prepare you
for, discovered by inspecting the actual files across all 25 subjects:
- `DeviceMode` is only ever explicitly logged as "sleep" or "exercise"; blank
  (~78% of rows dataset-wide, and 100% for some subjects) means "regular" by
  omission rather than being logged as such. Downstream code that wants a
  percent-of-time breakdown should fillna("regular") first (see
  azt1d.metrics.device_mode_distribution).
- `Basal` is logged sparsely (~30-100% of rows depending on subject, irregular
  gaps, not a clean hourly cadence) rather than already forward-filled onto the
  5-minute CGM grid as the paper's preprocessing section describes. Forward-
  filled in azt1d.cleaning (shared with azt1d.metabonet) so it reads as "the
  rate in effect at this timestamp," matching the paper's stated intent even
  though the raw file doesn't ship it that way.
"""

from __future__ import annotations

import re
import zipfile
from pathlib import Path

import pandas as pd

from . import cleaning
from . import reference as ref

# Common spellings a real-world export might use for each canonical column.
# Matching is case-insensitive and ignores spaces/underscores.
NAME_VARIANTS: dict[str, list[str]] = {
    ref.EVENT_DATETIME: ["eventdatetime", "datetime", "timestamp", "date time"],
    ref.DEVICE_MODE: ["devicemode", "device mode", "mode"],
    ref.BOLUS_TYPE: ["bolustype", "bolus type"],
    ref.BASAL: ["basal", "basalrate", "basal rate", "basaltotaldelivered"],
    ref.CORRECTION_DELIVERED: ["correctiondelivered", "correction delivered", "correction"],
    ref.TOTAL_BOLUS_INSULIN_DELIVERED: [
        "totalbolusinsulindelivered",
        "total bolus insulin delivered",
        "bolus",
        "totalbolus",
    ],
    ref.FOOD_DELIVERED: ["fooddelivered", "food delivered", "food"],
    ref.CARB_SIZE: ["carbsize", "carb size", "carbs", "carbohydrates"],
    ref.CGM: ["cgm", "glucose", "cgm (mg/dl)", "cgm_mg_dl", "readings (cgm / bgm)", "readings (cgm/bgm)"],
}

_NUMERIC_AGG_COLUMNS = [
    ref.BASAL,
    ref.CORRECTION_DELIVERED,
    ref.TOTAL_BOLUS_INSULIN_DELIVERED,
    ref.FOOD_DELIVERED,
    ref.CARB_SIZE,
    ref.CGM,
]
_CATEGORICAL_AGG_COLUMNS = [ref.DEVICE_MODE, ref.BOLUS_TYPE]


def _first_non_null(series: pd.Series):
    valid = series.dropna()
    return valid.iloc[0] if len(valid) else float("nan")


def _collapse_duplicate_timestamps(df: pd.DataFrame) -> pd.DataFrame:
    """
    A handful of timestamps within a subject carry >1 row after exact-duplicate
    removal (genuinely differing values, e.g. two CGM readings a few seconds
    apart both rounded to the same minute). Average the numeric fields and keep
    the first non-null categorical value so the series has one row per instant.
    """
    if not df.duplicated(subset=[ref.EVENT_DATETIME]).any():
        return df
    agg = {c: "mean" for c in _NUMERIC_AGG_COLUMNS}
    agg.update({c: _first_non_null for c in _CATEGORICAL_AGG_COLUMNS})
    return df.groupby(ref.EVENT_DATETIME, as_index=False).agg(agg)


def _clean_categoricals(df: pd.DataFrame) -> pd.DataFrame:
    """Real files use a stray literal '0' as a null placeholder in text columns,
    and at least one OCR-glitch repeat ('sleepsleep') was found by inspection."""
    for col in _CATEGORICAL_AGG_COLUMNS:
        df[col] = df[col].replace("0", float("nan")).replace("sleepsleep", "sleep")
    return df


def _normalize(name: str) -> str:
    return re.sub(r"[\s_]+", "", name.strip().lower())


def _build_rename_map(columns: list[str]) -> dict[str, str]:
    lookup = {}
    for canonical, variants in NAME_VARIANTS.items():
        lookup[_normalize(canonical)] = canonical
        for v in variants:
            lookup[_normalize(v)] = canonical

    rename_map = {}
    for col in columns:
        key = _normalize(col)
        if key in lookup:
            rename_map[col] = lookup[key]
    return rename_map


def find_dataset_zip(raw_dir: Path) -> Path | None:
    """Look for the downloaded Mendeley archive in data/raw/."""
    candidates = sorted(raw_dir.glob("*.zip"))
    return candidates[0] if candidates else None


def ensure_extracted(zip_path: Path, extract_to: Path) -> Path:
    """Extract the archive once; reuse the extracted folder on later runs."""
    marker = extract_to / ".extracted"
    if marker.exists():
        return extract_to
    extract_to.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(extract_to)
    marker.write_text("ok")
    return extract_to


def discover_subject_files(extracted_root: Path) -> dict[int, Path]:
    """
    Recursively find per-subject CSVs and map them to a subject ID parsed out
    of the filename or an enclosing folder name (first integer found).
    """
    subject_files: dict[int, Path] = {}
    for csv_path in sorted(extracted_root.rglob("*.csv")):
        match = re.search(r"(\d+)", csv_path.stem) or re.search(r"(\d+)", csv_path.parent.name)
        if not match:
            continue
        subject_id = int(match.group(1))
        subject_files[subject_id] = csv_path
    return subject_files


def load_subject_csv(path: Path, subject_id: int) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df.rename(columns=_build_rename_map(list(df.columns)))

    missing = [c for c in ref.CANONICAL_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(
            f"{path.name}: missing expected columns {missing} after normalization "
            f"(found columns: {list(df.columns)}). Add the real spelling to "
            f"NAME_VARIANTS in azt1d/loading.py."
        )

    df = df[ref.CANONICAL_COLUMNS].copy()
    df[ref.EVENT_DATETIME] = pd.to_datetime(df[ref.EVENT_DATETIME])

    # Exact-duplicate rows show up in every file (up to ~1900 identical copies
    # of a single reading in one case) -- drop before sorting/aggregating.
    df = df.drop_duplicates().sort_values(ref.EVENT_DATETIME).reset_index(drop=True)
    df = _collapse_duplicate_timestamps(df)
    df = _clean_categoricals(df)

    # Zero-filling missing bolus/carb readings, dropping implausible Basal
    # spikes, and forward-filling Basal to a per-timestep rate all happen in
    # azt1d.cleaning, shared with azt1d.metabonet so every dataset gets the
    # same treatment regardless of which loader produced it.
    df.insert(0, "subject_id", subject_id)
    return cleaning.clean_canonical_frame(df)


def load_all_subjects(extracted_root: Path) -> pd.DataFrame:
    subject_files = discover_subject_files(extracted_root)
    if not subject_files:
        raise FileNotFoundError(f"No per-subject CSVs found under {extracted_root}")
    frames = [load_subject_csv(path, sid) for sid, path in sorted(subject_files.items())]
    return pd.concat(frames, ignore_index=True)


def load_real_dataset(raw_dir: Path, processed_dir: Path) -> pd.DataFrame:
    """
    Full pipeline: find the zip -> extract -> discover -> load -> concat. If
    raw_dir already contains per-subject CSVs directly (e.g. someone dropped in
    an already-extracted folder instead of the zip), skip extraction and read
    straight from raw_dir.
    """
    zip_path = find_dataset_zip(raw_dir)
    if zip_path is not None:
        extracted_root = ensure_extracted(zip_path, processed_dir / "extracted")
    elif discover_subject_files(raw_dir):
        extracted_root = raw_dir
    else:
        raise FileNotFoundError(
            f"No .zip and no per-subject CSVs found in {raw_dir}. Download 'AZT1D 2025.zip' "
            f"from https://doi.org/{ref.DATASET_SUMMARY['doi']} and place it there."
        )
    return load_all_subjects(extracted_root)
