"""
Uncertainty bands around a trained point forecaster, without retraining it.

Six methods borrowed from other fields, all built on the same saved checkpoint's
60-minute-ahead predictions:

  - conformal: split conformal on absolute residuals (statistics, model-agnostic,
    constant width)
  - garch: GARCH(1,1) on the residual series (finance; width follows recent volatility,
    with a fitted mean-reversion speed)
  - analog: Analog Ensemble (weather; find past situations similar to now and reuse
    how wrong the model was then)
  - ewma: exponentially weighted moving average volatility, the RiskMetrics model that
    predates GARCH (finance; same idea as GARCH but with no mean reversion -- today's
    variance is just yesterday's blended with the latest squared error)
  - normalized conformal: split conformal on residuals divided by a local scale
    estimate from nearby moments, so the band widens and narrows locally instead of
    staying one fixed width (statistics; "locally weighted" conformal prediction)
  - time-of-day quantiles: empirical residual quantiles binned by time of day, the
    simplest thing weather forecasting calls a "climatology" baseline

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


def _conformal_quantile(sorted_scores: np.ndarray, level: float) -> float:
    """Same rank as conformal_halfwidth (rank = ceil((n+1) * level), since level = 1 - alpha), but
    takes an already-sorted array so BandEngine does not re-sort it on every call."""
    n = len(sorted_scores)
    rank = math.ceil((n + 1) * level)
    return float("inf") if rank > n else float(sorted_scores[rank - 1])


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


def garch_sigma(cal: Frame, test: Frame) -> tuple[GarchParams, np.ndarray]:
    """GARCH fitted on calibration residuals, then filtered through the test period with the
    60-minute lag. Returns the parameters and the 12-step-ahead sigma for each test window."""
    params = fit_garch(cal.resid)
    all_resid = np.concatenate([cal.resid, test.resid])
    init_var = float(np.var(cal.resid - params.mu))  # calibration data only
    var = garch_h_step_variance(all_resid, params, init_var, LAG)[len(cal.resid):]
    return params, np.sqrt(var)


# ---------------------------------------------------------------------------
# EWMA (RiskMetrics-style volatility, finance)
# ---------------------------------------------------------------------------


def fit_ewma(cal_resid: np.ndarray, candidates: tuple[float, ...] = (0.85, 0.90, 0.94, 0.97, 0.99)) -> GarchParams:
    """RiskMetrics EWMA volatility, in use in finance since before GARCH: today's variance is
    yesterday's variance blended with yesterday's squared error, with no long-run mean to revert
    to. It is exactly IGARCH(1,1) with omega=0 and alpha+beta=1, so it reuses garch_h_step_variance
    for the forecast recursion. mu is the calibration mean (the same bias correction GARCH gets from
    its own fit); the blend factor lambda has no closed-form fit for EWMA, so it is chosen from a
    small grid by one-step-ahead Gaussian log-likelihood on the calibration residuals."""
    mu = float(np.mean(cal_resid))
    e = cal_resid - mu
    best_lambda, best_ll = candidates[0], -np.inf
    for lam in candidates:
        s2 = float(np.var(e))
        ll = 0.0
        for t in range(1, len(e)):
            ll += -0.5 * (np.log(2 * np.pi * s2) + e[t] ** 2 / s2)
            s2 = lam * s2 + (1 - lam) * e[t - 1] ** 2
        if ll > best_ll:
            best_ll, best_lambda = ll, lam
    return GarchParams(mu=mu, omega=0.0, alpha=1 - best_lambda, beta=best_lambda)


def ewma_sigma(cal: Frame, test: Frame) -> tuple[GarchParams, np.ndarray]:
    """Same lag-respecting recursion as garch_sigma, with EWMA's fixed-persistence parameters."""
    params = fit_ewma(cal.resid)
    all_resid = np.concatenate([cal.resid, test.resid])
    init_var = float(np.var(cal.resid - params.mu))
    var = garch_h_step_variance(all_resid, params, init_var, LAG)[len(cal.resid):]
    return params, np.sqrt(var)


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


def analog_residual_sets(
    cal: Frame,
    test: Frame,
    k: int = 50,
    min_gap: int = LAG,
    tod_weight: float = 3.0,
    pool_cap: int = 600,
) -> np.ndarray:
    """For every test window, the signed forecast errors of its k analogs, shape (n_test, k).
    Rows are padded with nan if fewer than k spread-out analogs exist."""
    cal_f = _analog_features(cal, tod_weight)
    test_f = _analog_features(test, tod_weight)
    d2 = (test_f**2).sum(1)[:, None] + (cal_f**2).sum(1)[None, :] - 2 * test_f @ cal_f.T
    sets = np.full((len(test.pred), k), np.nan)
    for j in range(len(test.pred)):
        sel = select_analogs(d2[j], k, min_gap, pool_cap)
        sets[j, : len(sel)] = cal.resid[sel]
    return sets


def row_quantile(sorted_rows: np.ndarray, counts: np.ndarray, q: float) -> np.ndarray:
    """Linear-interpolated quantile q of each row of an ascending-sorted matrix whose
    nan padding sits at the end, using only the first counts[i] entries of row i."""
    pos = q * (counts - 1)
    i0 = np.floor(pos).astype(int)
    i1 = np.ceil(pos).astype(int)
    rows = np.arange(len(sorted_rows))
    frac = pos - i0
    return sorted_rows[rows, i0] * (1 - frac) + sorted_rows[rows, i1] * frac


# ---------------------------------------------------------------------------
# Normalized (locally weighted) Conformal -- statistics, conformal prediction literature
# ---------------------------------------------------------------------------


def local_scale(base: Frame, query: Frame, k: int, min_gap: int, tod_weight: float, pool_cap: int,
                 exclude_self: bool) -> np.ndarray:
    """A robust local spread of base's residuals around each row of query, from the k nearest base
    windows (same distance features as the analog ensemble): the median absolute deviation of their
    residuals, scaled to be a normal-consistent estimate of a standard deviation.

    exclude_self=True is for calibrating on base against itself (leave-one-out): a row is never its
    own neighbor. exclude_self=False is for scoring a separate query frame (the test period) against
    base (calibration)."""
    base_f = _analog_features(base, tod_weight)
    query_f = _analog_features(query, tod_weight)
    d2 = (query_f**2).sum(1)[:, None] + (base_f**2).sum(1)[None, :] - 2 * query_f @ base_f.T
    out = np.full(len(query.pred), np.nan)
    for j in range(len(query.pred)):
        row = d2[j].copy()
        if exclude_self:
            row[j] = np.inf
        sel = select_analogs(row, k, min_gap, pool_cap)
        vals = base.resid[sel]
        if len(vals) == 0:
            continue
        out[j] = 1.4826 * np.median(np.abs(vals - np.median(vals)))
    return out


def fit_normalized_conformal(cal: Frame, k: int, min_gap: int, tod_weight: float,
                              pool_cap: int) -> tuple[np.ndarray, np.ndarray]:
    """Locally weighted split conformal: the nonconformity score is each calibration residual
    divided by the local scale around it, so residuals from an easy (low-volatility) moment and a
    hard (high-volatility) moment are put on a common footing before the conformal quantile is
    taken. A test point's band width is that quantile scaled back up by its own local scale, so the
    band moves with local difficulty instead of staying one fixed width like plain conformal."""
    cal_scale = local_scale(cal, cal, k, min_gap, tod_weight, pool_cap, exclude_self=True)
    norm = np.abs(cal.resid) / np.maximum(cal_scale, 1.0)  # 1 mg/dL floor avoids dividing by ~0
    return cal_scale, np.sort(norm)


# ---------------------------------------------------------------------------
# Time-of-day quantiles -- a "climatology" baseline, as weather forecasting calls it
# ---------------------------------------------------------------------------


def _tod_bin(times: np.ndarray, n_bins: int) -> np.ndarray:
    edges = np.linspace(0, 1440, n_bins + 1)
    return np.clip(np.digitize(_minute_of_day(times), edges[1:-1]), 0, n_bins - 1)


def tod_quantile_bins(cal: Frame, test: Frame, n_bins: int) -> tuple[list[np.ndarray], np.ndarray]:
    """Calibration residuals grouped by time-of-day bin (sorted, for quantile lookup), plus each
    test window's bin. No distance computation at all -- the only thing conditioned on is the clock,
    the simplest baseline weather forecasting has a name for."""
    cal_bins = _tod_bin(cal.issue_time, n_bins)
    bins = [np.sort(cal.resid[cal_bins == b]) for b in range(n_bins)]
    return bins, _tod_bin(test.issue_time, n_bins)


# ---------------------------------------------------------------------------
# Bands, metrics, Clarke
# ---------------------------------------------------------------------------

METHODS = ("Conformal", "GARCH", "Analog Ensemble")
EXTRA_METHODS = ("EWMA", "Normalized Conformal", "Time-of-day Quantiles")
ALL_METHODS = METHODS + EXTRA_METHODS


class BandEngine:
    """All three band methods for one subject, prepared once so a band of any size (any
    coverage level) is cheap to produce. Everything is fitted on the calibration frame;
    the test frame is only ever filtered through, never fitted on."""

    def __init__(self, cal: Frame, test: Frame, k: int = 50, min_gap: int = LAG, tod_weight: float = 3.0,
                 analogs: bool = True, n_tod_bins: int = 8):
        self.pred = test.pred
        self._abs_sorted = np.sort(np.abs(cal.resid))
        self.garch_params, self.garch_sigma = garch_sigma(cal, test)
        self.ewma_params, self.ewma_sigma = ewma_sigma(cal, test)
        self._tod_bins, self._tod_test_bin = tod_quantile_bins(cal, test, n_tod_bins)
        self._analog_sorted = None
        self._norm_sorted = None
        if analogs:  # skip for very long records: these build an (n_test x n_cal) or (n_cal x n_cal) distance matrix
            sets = analog_residual_sets(cal, test, k=k, min_gap=min_gap, tod_weight=tod_weight)
            self._analog_sorted = np.sort(sets, axis=1)  # nan padding sorts to the end
            self._analog_counts = np.isfinite(sets).sum(axis=1)
            _, self._norm_sorted = fit_normalized_conformal(cal, k, min_gap, tod_weight, 600)
            self._norm_test_scale = local_scale(cal, test, k, min_gap, tod_weight, 600, exclude_self=False)

    def offsets(self, method: str, level: float) -> tuple[np.ndarray, np.ndarray]:
        """(lower, upper) offsets from the prediction, before clipping."""
        n = len(self.pred)
        alpha = 1 - level
        if method == "Conformal":
            q = _conformal_quantile(self._abs_sorted, level)
            return np.full(n, -q), np.full(n, q)
        if method == "GARCH":
            z = stats.norm.ppf(0.5 + level / 2)
            return self.garch_params.mu - z * self.garch_sigma, self.garch_params.mu + z * self.garch_sigma
        if method == "EWMA":
            z = stats.norm.ppf(0.5 + level / 2)
            return self.ewma_params.mu - z * self.ewma_sigma, self.ewma_params.mu + z * self.ewma_sigma
        if method == "Analog Ensemble":
            if self._analog_sorted is None:
                raise ValueError("this BandEngine was built with analogs=False")
            lo = row_quantile(self._analog_sorted, self._analog_counts, alpha / 2)
            hi = row_quantile(self._analog_sorted, self._analog_counts, 1 - alpha / 2)
            return lo, hi
        if method == "Normalized Conformal":
            if self._norm_sorted is None:
                raise ValueError("this BandEngine was built with analogs=False")
            q = _conformal_quantile(self._norm_sorted, level)
            width = q * self._norm_test_scale
            return -width, width
        if method == "Time-of-day Quantiles":
            lo = np.array([np.quantile(self._tod_bins[b], alpha / 2) if len(self._tod_bins[b]) else -np.inf
                            for b in self._tod_test_bin])
            hi = np.array([np.quantile(self._tod_bins[b], 1 - alpha / 2) if len(self._tod_bins[b]) else np.inf
                            for b in self._tod_test_bin])
            return lo, hi
        raise ValueError(method)

    def bands(self, method: str, level: float) -> tuple[np.ndarray, np.ndarray]:
        """(lower, upper) band as absolute glucose, clipped to the sensor range."""
        lo, hi = self.offsets(method, level)
        return np.clip(self.pred + lo, SENSOR_MIN, SENSOR_MAX), np.clip(self.pred + hi, SENSOR_MIN, SENSOR_MAX)


# ---------------------------------------------------------------------------
# The trigger rule: alarm when either band edge crosses into a danger zone
# ---------------------------------------------------------------------------


def trigger_flags(lo: np.ndarray, hi: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(any trigger, low-edge trigger, high-edge trigger). For a point forecast pass it as
    both lo and hi."""
    low = lo < ref.HYPO_THRESHOLD
    high = hi > ref.HYPER_THRESHOLD
    return low | high, low, high


def trigger_metrics(actual: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> dict[str, float]:
    """Alarm quality under the either-edge rule. Danger = real glucose under 70 or over 180;
    a false trigger is an alarm while real glucose was safe (70 to 180)."""
    trig, low, high = trigger_flags(lo, hi)
    hypo = actual < ref.HYPO_THRESHOLD
    hyper = actual > ref.HYPER_THRESHOLD
    danger = hypo | hyper
    safe = ~danger
    return {
        "false_trigger_rate": float(trig[safe].mean()),  # share of safe readings that falsely trigger
        "danger_caught": float(trig[danger].mean()),
        "lows_caught": float(trig[hypo].mean()) if hypo.any() else float("nan"),
        "highs_caught": float(trig[hyper].mean()) if hyper.any() else float("nan"),
        "share_of_alarms_false": float((trig & safe).sum() / max(trig.sum(), 1)),
        "false_via_low_edge": float((low & safe).sum() / safe.sum()),
        "false_via_high_edge": float((high & safe).sum() / safe.sum()),
        "n_safe": int(safe.sum()),
        "n_danger": int(danger.sum()),
    }


def swapped_prediction(pred: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> np.ndarray:
    """What a trigger-following forecaster would report. If an edge crosses into a danger zone,
    report that edge instead of the model's forecast. If both cross, report whichever
    crosses its line by more. If neither crosses, keep the model's forecast."""
    low_over = np.maximum(ref.HYPO_THRESHOLD - lo, 0.0)
    high_over = np.maximum(hi - ref.HYPER_THRESHOLD, 0.0)
    out = pred.copy()
    use_low = (low_over > 0) & (low_over >= high_over)
    use_high = (high_over > 0) & (high_over > low_over)
    out[use_low] = lo[use_low]
    out[use_high] = hi[use_high]
    return out


def level_for_false_trigger_rate(engines: dict, actuals: dict, method: str, target: float, iters: int = 30) -> float:
    """Band level at which `method` falsely triggers on `target` of all safe readings, pooled
    across subjects. Wider bands only ever trigger more, so a bisection is enough."""
    def pooled_rate(level: float) -> float:
        false = safe = 0
        for sid, eng in engines.items():
            lo, hi = eng.bands(method, level)
            trig, _, _ = trigger_flags(lo, hi)
            is_safe = (actuals[sid] >= ref.HYPO_THRESHOLD) & (actuals[sid] <= ref.HYPER_THRESHOLD)
            false += int((trig & is_safe).sum())
            safe += int(is_safe.sum())
        return false / safe

    lo_lv, hi_lv = 0.01, 0.999
    for _ in range(iters):
        mid = (lo_lv + hi_lv) / 2
        if pooled_rate(mid) < target:
            lo_lv = mid
        else:
            hi_lv = mid
    return (lo_lv + hi_lv) / 2


DEFAULT_AUC_LEVELS = np.round(np.concatenate([[0.001], np.arange(0.05, 0.951, 0.05), [0.99, 0.999]]), 3)


def trigger_auc(engine: "BandEngine", actual: np.ndarray, method: str, levels: np.ndarray = DEFAULT_AUC_LEVELS) -> float:
    """Area under the danger-caught vs. false-trigger-rate curve as the band grows from near zero to
    near full width -- the same curve plotted in notebooks_2/01, but reduced to one number so models
    can be ranked by it instead of by RMSE. It is a ROC-style AUC: false trigger rate stands in for a
    false-positive rate (out of safe readings) and danger caught stands in for a true-positive rate
    (out of dangerous readings), with band width playing the role of the threshold. 1.0 is impossible
    to reach in practice; higher means a better trade-off between catching real danger and crying wolf.

    Extended with (0, 0) at the low end and (1, 1) at the high end so the score is comparable across
    models even though no achievable band level gives a false trigger rate of exactly 0 or 1. This
    makes it an approximation, not an exact integral, when the curve is far from those anchors."""
    xs, ys = [0.0], [0.0]
    for level in levels:
        lo, hi = engine.bands(method, float(level))
        m = trigger_metrics(actual, lo, hi)
        xs.append(m["false_trigger_rate"])
        ys.append(m["danger_caught"])
    xs.append(1.0)
    ys.append(1.0)
    xs_arr, ys_arr = np.array(xs), np.array(ys)
    order = np.argsort(xs_arr, kind="stable")
    return float(np.trapezoid(ys_arr[order], xs_arr[order]))


# ---------------------------------------------------------------------------
# Coverage and Clarke helpers
# ---------------------------------------------------------------------------


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
