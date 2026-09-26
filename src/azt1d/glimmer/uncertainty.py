"""
Uncertainty bands around a trained point forecaster, without retraining it.

Three methods borrowed from other fields, all built on the same saved
checkpoint's 60-minute-ahead predictions:

  - conformal: split conformal on absolute residuals (model-agnostic, constant width)
  - garch: GARCH(1,1) on the residual series (finance; width follows recent volatility)
  - analog: Analog Ensemble (weather; find past situations similar to now and reuse
    how wrong the model was then)

Everything is calibrated on the validation period only. Training residuals are
optimistically small (the model was fit on them) and test data is never used to fit
anything. Checkpoints only store test predictions, so validation predictions are
rebuilt here from the saved weights (see forecast_frames).

Residuals are signed: residual = actual - prediction, so a band for the true value is
[prediction + lower_offset, prediction + upper_offset].
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
from scipy import stats

from .. import reference as ref
from . import checkpoint as ckpt
from . import sequences as seq
from .clinical import clarke_error_grid_zones, clarke_zone_percentages
from .train import get_device, prepare_subject_data

SENSOR_MIN, SENSOR_MAX = 40.0, 400.0  # Dexcom G6 reporting range, used to clip bands
LAG = seq.HORIZON  # a forecast issued at window i is only judged 12 steps later


@dataclass
class Frame:
    """One chronological slice (validation or test) of a subject's forecasts."""

    actual: np.ndarray  # true glucose at the target time
    pred: np.ndarray  # the model's 60-minute-ahead prediction
    resid: np.ndarray  # actual - pred
    X: np.ndarray  # scaled input windows, (n, lookback, n_features)
    issue_time: np.ndarray  # datetime64, when the forecast was made
    target_time: np.ndarray  # datetime64, the time being forecast


def _minute_of_day(times: np.ndarray) -> np.ndarray:
    ts = pd.DatetimeIndex(times)
    return (ts.hour * 60 + ts.minute).to_numpy()


def forecast_frames(result, df_subject: pd.DataFrame, device: torch.device | None = None) -> tuple[Frame, Frame]:
    """Rebuild validation and test forecasts from a checkpointed SubjectResult.

    Windows are consecutive 5-minute steps, so window i has its last input row at
    i + lookback - 1 and its target row at i + lookback + horizon - 1. Validation
    windows come first, then test windows, with no gap between them.
    """
    device = device or get_device()
    data = prepare_subject_data(df_subject)
    n_train, n_val = len(data.X_train), len(data.X_val)
    times = pd.to_datetime(df_subject[ref.EVENT_DATETIME]).to_numpy()

    model = ckpt.load_model(result, device)

    def predict(X: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            z = model(torch.from_numpy(X).to(device)).cpu().numpy()
        return z * result.y_std + result.y_mean

    def make(X: np.ndarray, y: np.ndarray, first_window: int) -> Frame:
        idx = np.arange(first_window, first_window + len(X))
        pred = predict(X)
        return Frame(
            actual=y.astype(float),
            pred=pred.astype(float),
            resid=y.astype(float) - pred.astype(float),
            X=X,
            issue_time=times[idx + seq.LOOKBACK - 1],
            target_time=times[idx + seq.LOOKBACK + seq.HORIZON - 1],
        )

    val = make(data.X_val, data.y_val, n_train)
    test = make(data.X_test, data.y_test, n_train + n_val)
    return val, test


def matches_stored_predictions(test: Frame, result, tol: float = 0.05) -> float:
    """Max absolute gap (mg/dL) between rebuilt test predictions and the ones saved at
    training time. Should be ~0; a large value means the rebuild is not trustworthy."""
    return float(np.max(np.abs(test.pred - np.asarray(result.y_pred, dtype=float))))


# ---------------------------------------------------------------------------
# Conformal
# ---------------------------------------------------------------------------


def conformal_halfwidth(cal_resid: np.ndarray, alpha: float) -> float:
    """Split-conformal half width: the ceil((n+1)(1-alpha))-th smallest absolute residual."""
    n = len(cal_resid)
    k = math.ceil((n + 1) * (1 - alpha))
    if k > n:
        return float("inf")
    return float(np.sort(np.abs(cal_resid))[k - 1])


def conformal_offsets(cal: Frame, test: Frame, levels: tuple[float, ...]) -> dict[float, tuple[np.ndarray, np.ndarray]]:
    out = {}
    for level in levels:
        q = conformal_halfwidth(cal.resid, 1 - level)
        out[level] = (np.full(len(test.pred), -q), np.full(len(test.pred), q))
    return out


# ---------------------------------------------------------------------------
# GARCH
# ---------------------------------------------------------------------------


@dataclass
class GarchParams:
    mu: float
    omega: float
    alpha: float
    beta: float


def fit_garch(cal_resid: np.ndarray) -> GarchParams:
    from arch import arch_model

    am = arch_model(cal_resid, mean="Constant", vol="GARCH", p=1, q=1, dist="normal", rescale=False)
    res = am.fit(disp="off", show_warning=False)
    p = res.params
    return GarchParams(mu=float(p["mu"]), omega=float(p["omega"]), alpha=float(p["alpha[1]"]), beta=float(p["beta[1]"]))


def garch_h_step_variance(resid: np.ndarray, params: GarchParams, init_var: float, h: int = LAG) -> np.ndarray:
    """Variance forecast for each position p using only residuals at positions <= p - h.

    A forecast is made h steps before its target, and the error of a forecast only
    becomes known at its target time, so at issue time the newest known residual is
    the one for target p - h. Positions p < h have no known residual: nan.

    init_var seeds the recursion and must come from data that is known before the
    series starts (the calibration residuals), never from the series being filtered.
    """
    e = resid - params.mu
    n = len(e)
    phi = params.alpha + params.beta
    s2 = np.empty(n + 1)  # s2[k] = one-step-ahead variance given residuals through k-1
    s2[0] = init_var
    for k in range(n):
        s2[k + 1] = params.omega + params.alpha * e[k] ** 2 + params.beta * s2[k]
    geo = (h - 1) if abs(1 - phi) < 1e-9 else (1 - phi ** (h - 1)) / (1 - phi)
    out = np.full(n, np.nan)
    p = np.arange(h, n)
    out[p] = params.omega * geo + phi ** (h - 1) * s2[p - h + 1]
    return out


def garch_offsets(cal: Frame, test: Frame, levels: tuple[float, ...]):
    params = fit_garch(cal.resid)
    all_resid = np.concatenate([cal.resid, test.resid])
    init_var = float(np.var(cal.resid - params.mu))  # calibration data only
    var = garch_h_step_variance(all_resid, params, init_var, LAG)[len(cal.resid):]
    sigma = np.sqrt(var)
    out = {}
    for level in levels:
        z = stats.norm.ppf(0.5 + level / 2)
        out[level] = (params.mu - z * sigma, params.mu + z * sigma)
    return out, params, sigma


# ---------------------------------------------------------------------------
# Analog Ensemble
# ---------------------------------------------------------------------------


def _analog_features(frame: Frame, tod_weight: float) -> np.ndarray:
    flat = frame.X.reshape(len(frame.X), -1).astype(np.float64)
    angle = 2 * np.pi * _minute_of_day(frame.issue_time) / 1440
    return np.concatenate([flat, tod_weight * np.sin(angle)[:, None], tod_weight * np.cos(angle)[:, None]], axis=1)


def select_analogs(dist_row: np.ndarray, k: int, min_gap: int, pool_cap: int) -> np.ndarray:
    """Indices of the k nearest calibration windows, no two closer than min_gap windows
    apart (adjacent windows are near-copies of each other and would otherwise fill the
    whole ensemble with one moment in time)."""
    n = len(dist_row)
    cand = np.argpartition(dist_row, min(pool_cap, n - 1))[:pool_cap]
    cand = cand[np.argsort(dist_row[cand])]
    blocked = np.zeros(n, dtype=bool)
    chosen = []
    for i in cand:
        if blocked[i]:
            continue
        chosen.append(i)
        blocked[max(0, i - min_gap + 1) : i + min_gap] = True
        if len(chosen) == k:
            break
    return np.array(chosen)


def analog_offsets(
    cal: Frame,
    test: Frame,
    levels: tuple[float, ...],
    k: int = 50,
    min_gap: int = LAG,
    tod_weight: float = 3.0,
    pool_cap: int = 600,
):
    cal_f = _analog_features(cal, tod_weight)
    test_f = _analog_features(test, tod_weight)
    d2 = (test_f**2).sum(1)[:, None] + (cal_f**2).sum(1)[None, :] - 2 * test_f @ cal_f.T
    lo = {lv: np.empty(len(test.pred)) for lv in levels}
    hi = {lv: np.empty(len(test.pred)) for lv in levels}
    for j in range(len(test.pred)):
        sel = select_analogs(d2[j], k, min_gap, pool_cap)
        r = cal.resid[sel]
        for lv in levels:
            a = 1 - lv
            lo[lv][j], hi[lv][j] = np.quantile(r, [a / 2, 1 - a / 2])
    return {lv: (lo[lv], hi[lv]) for lv in levels}


# ---------------------------------------------------------------------------
# Bands, metrics, Clarke
# ---------------------------------------------------------------------------

METHODS = ("Conformal", "GARCH", "Analog Ensemble")


def build_bands(cal: Frame, test: Frame, levels: tuple[float, ...] = (0.8, 0.95)):
    """{method: {level: (lower, upper)}} as absolute glucose values, clipped to the sensor range."""
    offsets = {
        "Conformal": conformal_offsets(cal, test, levels),
        "GARCH": garch_offsets(cal, test, levels)[0],
        "Analog Ensemble": analog_offsets(cal, test, levels),
    }
    bands = {}
    for method, per_level in offsets.items():
        bands[method] = {
            lv: (np.clip(test.pred + lo, SENSOR_MIN, SENSOR_MAX), np.clip(test.pred + hi, SENSOR_MIN, SENSOR_MAX))
            for lv, (lo, hi) in per_level.items()
        }
    return bands


def region_of(actual: np.ndarray) -> np.ndarray:
    return np.select(
        [actual < ref.HYPO_THRESHOLD, actual > ref.HYPER_THRESHOLD], ["hypo", "hyper"], default="normal"
    )


def interval_metrics(actual: np.ndarray, lo: np.ndarray, hi: np.ndarray, level: float) -> dict[str, float]:
    alpha = 1 - level
    inside = (actual >= lo) & (actual <= hi)
    width = hi - lo
    winkler = width + (2 / alpha) * np.maximum(lo - actual, 0) + (2 / alpha) * np.maximum(actual - hi, 0)
    out = {"coverage": float(inside.mean()), "width": float(width.mean()), "interval_score": float(winkler.mean())}
    regions = region_of(actual)
    for name in ("hypo", "normal", "hyper"):
        m = regions == name
        out[f"coverage_{name}"] = float(inside[m].mean()) if m.any() else float("nan")
    return out


def band_edge_zones(actual: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> dict[str, dict[str, float]]:
    """Grade each band edge as if it were the prediction, on the Clarke Error Grid."""
    return {"lower": clarke_zone_percentages(actual, lo), "upper": clarke_zone_percentages(actual, hi)}


def width_by_point_zone(actual: np.ndarray, pred: np.ndarray, width: np.ndarray) -> dict[str, float]:
    zones = clarke_error_grid_zones(actual, pred)
    return {z: float(width[zones == z].mean()) if (zones == z).any() else float("nan") for z in "ABCDE"}


def width_warning_auc(actual: np.ndarray, pred: np.ndarray, width: np.ndarray) -> float:
    """Probability that a randomly chosen clinically bad forecast (Clarke C/D/E) has a
    wider band than a randomly chosen fine one (A/B). 0.5 means the width carries no
    information; 1.0 means wider bands perfectly flag the bad forecasts."""
    zones = clarke_error_grid_zones(actual, pred)
    bad = np.isin(zones, ["C", "D", "E"])
    if bad.sum() == 0 or (~bad).sum() == 0:
        return float("nan")
    ranks = stats.rankdata(width)
    n_bad, n_ok = bad.sum(), (~bad).sum()
    return float((ranks[bad].sum() - n_bad * (n_bad + 1) / 2) / (n_bad * n_ok))
