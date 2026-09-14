"""
Reference metadata for the AZT1D dataset, transcribed from the data descriptor:

Khamesian, S., Arefeen, A., Thompson, B. M., Grando, M. A., & Ghasemzadeh, H. (2025).
AZT1D: A Real-World Dataset for Type 1 Diabetes. arXiv:2506.14789.
Dataset: Mendeley Data, https://doi.org/10.17632/gk9m674wcx.1 (CC BY 4.0)

This module holds facts we know about the dataset *before* the raw files are
in hand: the record schema (Section III-B of the paper) and the per-patient
demographics (Table I). Nothing here depends on the actual CSVs being present.
"""

from __future__ import annotations

import pandas as pd

# Column names exactly as described in the paper's "Dataset Description" (III-B).
# Each subject's file is a single time-aligned record stream with these fields.
EVENT_DATETIME = "EventDateTime"
DEVICE_MODE = "DeviceMode"
BOLUS_TYPE = "BolusType"
BASAL = "Basal"
CORRECTION_DELIVERED = "CorrectionDelivered"
TOTAL_BOLUS_INSULIN_DELIVERED = "TotalBolusInsulinDelivered"
FOOD_DELIVERED = "FoodDelivered"
CARB_SIZE = "CarbSize"
CGM = "CGM"

CANONICAL_COLUMNS = [
    EVENT_DATETIME,
    DEVICE_MODE,
    BOLUS_TYPE,
    BASAL,
    CORRECTION_DELIVERED,
    TOTAL_BOLUS_INSULIN_DELIVERED,
    FOOD_DELIVERED,
    CARB_SIZE,
    CGM,
]

# Columns that are 0-filled by the source pipeline when no event occurred,
# rather than left missing (paper, Section III-A, last paragraph).
ZERO_FILLED_COLUMNS = [CARB_SIZE, CORRECTION_DELIVERED, TOTAL_BOLUS_INSULIN_DELIVERED, FOOD_DELIVERED]

DEVICE_MODES = ["regular", "sleep", "exercise"]
BOLUS_TYPES = ["standard", "correction", "automatic"]

# CGM sampling interval; basal is recorded hourly and forward-filled onto the
# 5-minute CGM grid by the source pipeline (paper, Section III-A).
CGM_INTERVAL_MINUTES = 5
BASAL_INTERVAL_MINUTES = 60

# Standard ambulatory glucose profile thresholds (mg/dL), used throughout for
# time-in-range / hypo- / hyperglycemia calculations.
HYPO_THRESHOLD = 70
HYPER_THRESHOLD = 180

# Table I: demographic and clinical data of the 25 patients, transcribed
# verbatim from the paper. "No." is the paper's patient index — treat it as a
# provisional join key to a subject's file-derived ID until confirmed against
# the actual filenames/folders once the dataset is extracted.
DEMOGRAPHICS_TABLE1 = pd.DataFrame(
    [
        (1, 7.2, "Male", 65),
        (2, 6.6, "Female", 67),
        (3, 5.1, "Male", 65),
        (4, 7.2, "Female", 69),
        (5, 6.5, "Male", 80),
        (6, 6.6, "Female", 77),
        (7, 6.8, "Male", 36),
        (8, 7.2, "Female", 66),
        (9, 8.2, "Female", 54),
        (10, 7.7, "Female", 71),
        (11, 7.3, "Male", 59),
        (12, 6.6, "Male", 43),
        (13, 6.7, "Male", 80),
        (14, 5.0, "Female", 32),
        (15, 5.9, "Female", 52),
        (16, 6.3, "Male", 40),
        (17, 7.1, "Female", 66),
        (18, 6.9, "Male", 65),
        (19, 6.2, "Male", 27),
        (20, 6.7, "Female", 61),
        (21, 6.4, "Female", 46),
        (22, 6.5, "Female", 46),
        (23, 5.7, "Female", 67),
        (24, 6.7, "Male", 74),
        (25, 6.8, "Male", 72),
    ],
    columns=["subject_id", "a1c_pct", "sex", "age"],
)

# GLIMMER paper (Khamesian et al.), Table 3: average per-patient region-loss
# weights found by their genetic algorithm on OhioT1DM, one set per architecture.
# w_normal is fixed at 1 in the paper; hypo and hyper are free.
GLIMMER_PAPER_WEIGHTS = {
    "cnn_lstm": {"w_hypo": 3.29, "w_normal": 1.0, "w_hyper": 2.38},
    "transformer": {"w_hypo": 4.67, "w_normal": 1.0, "w_hyper": 1.71},
}

DATASET_SUMMARY = {
    "name": "AZT1D",
    "doi": "10.17632/gk9m674wcx.1",
    "license": "CC BY 4.0",
    "n_subjects": 25,
    "avg_duration_days": 26,
    "total_monitoring_hours": 26707,
    "total_cgm_entries": 320488,
    "cgm_device": "Dexcom G6 Pro",
    "pump": "Tandem t:slim X2 (Control-IQ)",
    "collection_site": "Mayo Clinic Arizona, Scottsdale",
    "collection_window": "December 2023 - April 2024",
}
