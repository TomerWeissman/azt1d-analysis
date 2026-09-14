"""
Loader for MetaboNet (Khamesian et al., 2026, arXiv:2601.11505), a 154.8M-row
consolidation of 14 public T1D datasets (including AZT1D and OhioT1DM) into
one harmonized schema, shipped as a single parquet file.

This is the reverse of loading.py: that module is AZT1D-specific and reads
the dataset's own raw per-subject CSVs directly. This module reads any of the
14 datasets MetaboNet bundles through one generic path, mapped onto the same
canonical schema (azt1d.reference.CANONICAL_COLUMNS) so the rest of this
codebase doesn't need to care which loader a given DataFrame came from.

The mapping below is not a guess -- it was checked row-by-row against
loading.py's own AZT1D output before being written:
  - CGM: exact match on 98.2-100% of readings across all 23 subjects AZT1D
    and MetaboNet share (differences, where they occur, are calendar-second
    rounding collisions when two raw readings land in the same 5-minute
    bucket after MetaboNet's timestamp re-gridding, not real data
    disagreement).
  - bolus, carbs: exact match, 0 difference, on every bolus event checked.
  - basal: MetaboNet stores units delivered *in that row's interval*, not a
    rate. meta_basal * 12 == our own Basal (U/hr) to full precision. This
    loader applies that conversion so Basal means the same thing regardless
    of which loader produced it.
  - MetaboNet's `id` field is a plain per-dataset subject identifier, not
    globally unique across datasets (AZT1D and OhioT1DM both have a subject
    "1"-shaped id in their own numbering) -- always pair it with the source
    dataset name, never assume ids are comparable across sources.

Two AZT1D subjects (10 and 23) that exist in the raw files loading.py reads
are absent from MetaboNet's AZT1D subset -- a real gap in MetaboNet's own
harmonization, not a bug here.

Columns AZT1D's own schema has that MetaboNet's harmonized one doesn't
(DeviceMode, BolusType, CorrectionDelivered, FoodDelivered) come back as
NaN/pd.NA rather than being invented.
"""

from __future__ import annotations

from pathlib import Path

import duckdb
import pandas as pd

from . import cleaning
from . import reference as ref

DEFAULT_METABONET_PATH = Path.home() / "Downloads" / "metabonet_public.parquet"

# Every source dataset MetaboNet bundles, largest first (row count), as of
# the file this loader was validated against. Re-run list_sources() to check
# this is still accurate if the file gets updated.
KNOWN_SOURCES = [
    "Loop", "ReplaceBG", "IOBP2", "Flair", "DCLP5", "DCLP3", "PEDAP", "CTR3",
    "BrisT1D", "T1D-UOM", "HUPA-UCM", "AZT1D", "OhioT1DM", "ShanghaiT1DM",
]

# MetaboNet's basal is delivered units for that row's 5-minute interval; the
# rest of this codebase (and AZT1D's own raw files) express it as a rate.
ROW_INTERVAL_MINUTES = 5
_BASAL_RATE_SCALE = 60 / ROW_INTERVAL_MINUTES


def list_sources(path: Path = DEFAULT_METABONET_PATH) -> pd.DataFrame:
    """Row/subject counts per source dataset, read straight from the file."""
    con = duckdb.connect()
    return con.execute(
        f"""
        SELECT source_file, COUNT(*) AS n_rows, COUNT(DISTINCT id) AS n_subjects
        FROM read_parquet('{path}')
        GROUP BY source_file
        ORDER BY n_rows DESC
        """
    ).fetchdf()


def load_source(
    source_file: str,
    path: Path = DEFAULT_METABONET_PATH,
    subject_ids: list[str] | None = None,
    clean: bool = True,
) -> pd.DataFrame:
    """
    Load one MetaboNet source dataset onto azt1d's canonical schema, cleaned
    the same way azt1d.loading cleans AZT1D's own raw files (zero-filled
    bolus/carbs, forward-filled basal, dropped missing-CGM placeholder rows)
    -- see azt1d.cleaning. Pass clean=False to get MetaboNet's values as-is.

    subject_id comes back as a native int when every id for this source
    parses as one (true for AZT1D and OhioT1DM, checked directly), otherwise
    as MetaboNet's own string id -- not every source's ids are guaranteed
    numeric, so this doesn't force a cast that would fail for some of them.
    """
    con = duckdb.connect()
    subject_filter = ""
    if subject_ids is not None:
        ids = ", ".join(f"'{s}'" for s in subject_ids)
        subject_filter = f"AND id IN ({ids})"

    raw = con.execute(
        f"""
        SELECT id, date, CGM, basal, bolus, carbs
        FROM read_parquet('{path}')
        WHERE source_file = '{source_file}' {subject_filter}
        ORDER BY id, date
        """
    ).fetchdf()

    if raw.empty:
        raise ValueError(f"No rows found for source_file={source_file!r}. Known sources: {KNOWN_SOURCES}")

    out = pd.DataFrame(
        {
            "subject_id": _coerce_subject_ids(raw["id"]),
            ref.EVENT_DATETIME: raw["date"],
            ref.CGM: raw["CGM"],
            ref.BASAL: raw["basal"] * _BASAL_RATE_SCALE,
            ref.TOTAL_BOLUS_INSULIN_DELIVERED: raw["bolus"],
            ref.CARB_SIZE: raw["carbs"],
        }
    )
    for col in (ref.DEVICE_MODE, ref.BOLUS_TYPE, ref.CORRECTION_DELIVERED, ref.FOOD_DELIVERED):
        out[col] = pd.NA

    out = out[["subject_id", *ref.CANONICAL_COLUMNS]]
    return cleaning.clean_canonical_frame(out) if clean else out


def _coerce_subject_ids(ids: pd.Series) -> pd.Series:
    try:
        return ids.astype(int)
    except (ValueError, TypeError):
        return ids
