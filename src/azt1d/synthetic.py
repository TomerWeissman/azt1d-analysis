"""
Synthetic AZT1D-shaped data, generated so the loading/EDA pipeline can be
built and tested *before* the real 776 MB dataset finishes downloading.

This is placeholder data only — glucose dynamics are a simple mean-reverting
random walk with meal/bolus effects, not a physiological model. It exists to
exercise the same schema (see azt1d.reference) and the same loader
(azt1d.loading) that the real per-subject CSVs will use, so switching from
synthetic to real data is a one-line path change in the notebook.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from . import reference as ref

CGM_MIN, CGM_MAX = 40, 400


def _daily_mealtimes(rng: np.random.Generator, day_start: pd.Timestamp) -> list[tuple[pd.Timestamp, float]]:
    """Return (time, carb_grams) for breakfast/lunch/dinner and maybe a snack."""
    meals = []
    for hour_center, carb_range in [(7.5, (20, 70)), (12.5, (30, 90)), (18.5, (30, 100))]:
        minutes = int(round(np.clip(rng.normal(hour_center, 0.75), 5, 22) * 60))
        carbs = rng.uniform(*carb_range)
        meals.append((day_start + pd.Timedelta(minutes=minutes), round(carbs)))
    if rng.random() < 0.4:
        minutes = int(round(rng.uniform(14, 16) * 60))
        meals.append((day_start + pd.Timedelta(minutes=minutes), round(rng.uniform(10, 30))))
    return meals


def _device_mode_for(ts: pd.Timestamp, exercise_windows: list[tuple[pd.Timestamp, pd.Timestamp]]) -> str:
    for start, end in exercise_windows:
        if start <= ts < end:
            return "exercise"
    hour = ts.hour + ts.minute / 60
    if hour >= 23 or hour < 6.5:
        return "sleep"
    return "regular"


def generate_subject_series(
    subject_id: int,
    rng: np.random.Generator,
    start_date: str = "2024-01-01",
    days: int = 26,
) -> pd.DataFrame:
    start = pd.Timestamp(start_date)
    timestamps = pd.date_range(start, periods=days * 24 * 60 // ref.CGM_INTERVAL_MINUTES, freq="5min")

    # Per-subject baseline traits, so subjects are visibly different from one another.
    baseline_glucose = rng.normal(140, 15)
    basal_rate = np.clip(rng.normal(0.8, 0.2), 0.3, 1.8)
    carb_ratio = rng.uniform(8, 14)  # grams of carb covered per unit of insulin

    n = len(timestamps)
    glucose = np.empty(n)
    glucose[0] = baseline_glucose
    basal = np.full(n, basal_rate)
    carb_size = np.zeros(n)
    food_delivered = np.zeros(n)
    correction_delivered = np.zeros(n)
    total_bolus = np.zeros(n)
    bolus_type = np.array([""] * n, dtype=object)

    exercise_windows = []
    for day in range(days):
        day_start = start + pd.Timedelta(days=day)
        if rng.random() < 0.3:
            ex_minutes = int(round(rng.uniform(16, 19) * 60))
            ex_start = day_start + pd.Timedelta(minutes=ex_minutes)
            exercise_windows.append((ex_start, ex_start + pd.Timedelta(minutes=int(rng.uniform(30, 60)))))
        # Hourly Control-IQ-like basal modulation.
        for hour in range(24):
            basal_idx = int((day * 24 + hour) * 60 / ref.CGM_INTERVAL_MINUTES)
            end_idx = min(basal_idx + 60 // ref.CGM_INTERVAL_MINUTES, n)
            if basal_idx < n:
                modulation = np.clip(rng.normal(1.0, 0.15), 0.4, 1.6)
                basal[basal_idx:end_idx] = round(basal_rate * modulation, 3)

        for meal_time, carbs in _daily_mealtimes(rng, day_start):
            idx = timestamps.searchsorted(meal_time)
            if idx >= n:
                continue
            carb_size[idx] = carbs
            dose = round(carbs / carb_ratio, 2)
            food_delivered[idx] = dose
            total_bolus[idx] += dose
            bolus_type[idx] = "standard"

    device_mode = [_device_mode_for(ts, exercise_windows) for ts in timestamps]

    # Simple mean-reverting glucose walk with meal spikes, insulin drops, and
    # sleep/exercise effects layered on top.
    for i in range(1, n):
        drift = 0.06 * (baseline_glucose - glucose[i - 1])
        noise = rng.normal(0, 3.5)
        meal_effect = carb_size[i - 1] * 3.0
        insulin_effect = -(basal[i - 1] * 1.2 + total_bolus[i - 1] * 9.0)
        mode_effect = -4.0 if device_mode[i - 1] == "exercise" else (1.0 if device_mode[i - 1] == "sleep" else 0.0)
        glucose[i] = glucose[i - 1] + drift + noise + meal_effect + insulin_effect + mode_effect
        glucose[i] = np.clip(glucose[i], CGM_MIN, CGM_MAX)

        # Correction bolus roughly every few hours if running high.
        if glucose[i] > 200 and total_bolus[i] == 0 and rng.random() < 0.05:
            correction = round((glucose[i] - 140) / 40, 2)
            correction_delivered[i] = correction
            total_bolus[i] += correction
            bolus_type[i] = "correction" if not bolus_type[i] else bolus_type[i]

    df = pd.DataFrame(
        {
            ref.EVENT_DATETIME: timestamps,
            ref.DEVICE_MODE: device_mode,
            ref.BOLUS_TYPE: bolus_type,
            ref.BASAL: np.round(basal, 3),
            ref.CORRECTION_DELIVERED: np.round(correction_delivered, 2),
            ref.TOTAL_BOLUS_INSULIN_DELIVERED: np.round(total_bolus, 2),
            ref.FOOD_DELIVERED: np.round(food_delivered, 2),
            ref.CARB_SIZE: carb_size,
            ref.CGM: np.round(glucose).astype(int),
        }
    )
    df.insert(0, "subject_id", subject_id)
    return df


def generate_synthetic_dataset(
    n_subjects: int = ref.DATASET_SUMMARY["n_subjects"],
    days: int = ref.DATASET_SUMMARY["avg_duration_days"],
    seed: int = 7,
) -> pd.DataFrame:
    master_rng = np.random.default_rng(seed)
    frames = []
    for subject_id in range(1, n_subjects + 1):
        subject_rng = np.random.default_rng(master_rng.integers(0, 2**32 - 1))
        # Stagger start dates a little so the "collection window" looks realistic.
        start = pd.Timestamp("2023-12-04") + pd.Timedelta(days=int(subject_rng.integers(0, 90)))
        frames.append(generate_subject_series(subject_id, subject_rng, str(start.date()), days))
    return pd.concat(frames, ignore_index=True)


def write_synthetic_csvs(output_dir: Path, n_subjects: int = 25, days: int = 26, seed: int = 7) -> Path:
    """
    Write one CSV per subject in the same shape the real per-subject files are
    expected to have, so azt1d.loading.load_all_subjects() works unchanged on
    this synthetic folder.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    master_rng = np.random.default_rng(seed)
    for subject_id in range(1, n_subjects + 1):
        subject_rng = np.random.default_rng(master_rng.integers(0, 2**32 - 1))
        start = pd.Timestamp("2023-12-04") + pd.Timedelta(days=int(subject_rng.integers(0, 90)))
        df = generate_subject_series(subject_id, subject_rng, str(start.date()), days)
        df.drop(columns="subject_id").to_csv(output_dir / f"Subject {subject_id}.csv", index=False)
    return output_dir
