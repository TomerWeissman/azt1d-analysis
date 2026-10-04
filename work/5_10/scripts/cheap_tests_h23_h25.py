"""Three cheap tests from the 5_10 plan. No training: they use the saved AZT1D CNN-LSTM
checkpoints (cnn_lstm_v0, the plain model) and the existing band code.

H23  Noise ceiling, upper bound. Pairs of test windows with nearly identical inputs should have
     similar true outcomes, so the spread between their outcomes measures noise plus the input
     difference. Half that spread is an upper bound on noise variance. Supported if the bound is
     at most 75% of the model's own mean squared error.

H24  Band alarm vs point alarm. For a hypoglycemia alarm (true glucose under 70), tune the
     threshold on validation data to reach 90% sensitivity, then score it on test data. The point
     alarm fires when the forecast is under a threshold. The band alarm fires when the lower band
     edge is under 70. Supported if the band alarm has at least 20% fewer false-alarm onsets per
     day, with both sensitivities at or above 85%.

H25  Conformal coverage. For each patient, build conformal 80% bands from validation residuals and
     count how often the true value lands inside the band on the test period. Supported if 20 or
     more of the 25 patients land between 75% and 85%.

Run:    python work/5_10/scripts/cheap_tests_h23_h25.py
Writes: work/5_10/results/h23_lookalike.txt, h24_alarms.csv, h25_conformal_coverage.csv,
        verdicts_h23_h25.txt
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = next(p for p in Path(__file__).resolve().parents if (p / "pyproject.toml").exists())
WEEK = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from azt1d import loading, reference as ref  # noqa: E402
from azt1d.glimmer import checkpoint as ckpt, uncertainty as unc  # noqa: E402

RUN = "cnn_lstm_v0"
HYPO = ref.HYPO_THRESHOLD
SENS_TARGET = 0.90
LOOK_ALIKE_SAMPLE = 4000
EXCLUDE_STEPS = 12  # same patient and within 1 hour counts as the same moment, not a look-alike
H23_LIMIT = 0.75
H24_SAVING = 0.20
H24_MIN_SENS = 0.85
H25_LO, H25_HI, H25_MIN_COUNT = 0.75, 0.85, 20
LEVELS = np.round(np.arange(0.01, 0.9991, 0.01), 3)
THRESHOLDS = np.arange(40.0, 201.0, 1.0)


def load_frames():
    df = loading.load_real_dataset(ROOT / "data" / "raw", ROOT / "data" / "processed")
    out = {}
    for sid in sorted(int(s) for s in df["subject_id"].unique()):
        res = ckpt.load_result(ROOT / "data" / "processed" / "checkpoints" / RUN, sid)
        dfs = df[df["subject_id"] == sid].reset_index(drop=True)
        val, test = unc.forecast_frames(res, dfs)
        out[sid] = (val, test)
    return out


# ------------------------------------------------------------------ H23

def h23(frames) -> str:
    rng = np.random.default_rng(0)
    X, y, pid, tix, pred, act = [], [], [], [], [], []
    for sid, (_, test) in frames.items():
        n = len(test.actual)
        X.append(test.X.reshape(n, -1)); y.append(test.actual); pred.append(test.pred); act.append(test.actual)
        pid.append(np.full(n, sid)); tix.append(np.arange(n))
    X = np.concatenate(X).astype(np.float32); y = np.concatenate(y); pred = np.concatenate(pred)
    pid = np.concatenate(pid); tix = np.concatenate(tix)
    model_mse = float(np.mean((y - pred) ** 2))
    q = rng.choice(len(y), min(LOOK_ALIKE_SAMPLE, len(y)), replace=False)
    sq = (X ** 2).sum(1)
    diffs = []
    for start in range(0, len(q), 500):
        idx = q[start:start + 500]
        D = sq[idx][:, None] + sq[None, :] - 2 * X[idx] @ X.T
        same_moment = (pid[idx][:, None] == pid[None, :]) & (np.abs(tix[idx][:, None] - tix[None, :]) <= EXCLUDE_STEPS)
        D[same_moment] = np.inf
        D[np.arange(len(idx)), idx] = np.inf
        nn = D.argmin(1)
        diffs.append(y[idx] - y[nn])
    d = np.concatenate(diffs)
    noise_upper = float(np.mean(d ** 2) / 2)
    share = noise_upper / model_mse
    verdict = "SUPPORTED" if share <= H23_LIMIT else "INCONCLUSIVE (the bound is above 75%, so nothing is shown)"
    text = "\n".join([
        f"H23 (at least 25% of error reducible, i.e. noise at most 75%): {verdict}",
        f"  look-alike pairs used: {len(d):,} (same-patient pairs within 1 hour excluded)",
        f"  model mean squared error (pooled test): {model_mse:.1f} mg2/dL2 (RMSE {np.sqrt(model_mse):.1f})",
        f"  noise variance upper bound (half the mean squared pair difference): {noise_upper:.1f} mg2/dL2",
        f"  upper bound as a share of model error: {share:.0%} (rule: at most {H23_LIMIT:.0%})",
        "  caveat: neighbours still differ in inputs, so this is an upper bound on noise, not the noise itself",
    ])
    return text


# ------------------------------------------------------------------ H24

def tune_point(val_pred, val_act):
    lows = val_act < HYPO
    if lows.sum() == 0:
        return None
    for t in THRESHOLDS:
        if np.mean(val_pred[lows] < t) >= SENS_TARGET:
            return float(t)
    return None


def tune_band(eng_factory, val_act):
    lows = val_act < HYPO
    if lows.sum() == 0:
        return None
    for lv in LEVELS:
        lo, _ = eng_factory(lv)
        if np.mean(lo[lows] < HYPO) >= SENS_TARGET:
            return float(lv)
    return None


def onsets(alarm: np.ndarray) -> int:
    return int(np.sum(alarm[1:] & ~alarm[:-1]) + int(alarm[0]))


def h24(frames) -> tuple[str, pd.DataFrame]:
    rows = []
    for sid, (val, test) in frames.items():
        t_point = tune_point(val.pred, val.actual)
        # Tune on validation only: bands built from validation residuals and scored on validation.
        eng_val = unc.BandEngine(val, val, analogs=False)
        lv = tune_band(lambda L: eng_val.bands("GARCH", L), val.actual)
        if t_point is None or lv is None:
            continue
        # Then apply the tuned level to the test period.
        eng = unc.BandEngine(val, test, analogs=False)
        lo, _ = eng.bands("GARCH", lv)
        alarm_point = test.pred < t_point
        alarm_band = lo < HYPO
        test_days = len(test.actual) * 5 / 1440
        lows = test.actual < HYPO
        rows.append({
            "subject_id": sid, "test_days": test_days,
            "point_threshold": t_point, "point_sensitivity": float(np.mean(alarm_point[lows])),
            "point_onsets": onsets(alarm_point),
            "band_level": lv, "band_sensitivity": float(np.mean(alarm_band[lows])),
            "band_onsets": onsets(alarm_band),
        })
    res = pd.DataFrame(rows)
    days = res.test_days.sum()
    point_rate = res.point_onsets.sum() / days
    band_rate = res.band_onsets.sum() / days
    saving = 1 - band_rate / point_rate
    sens_ok = res.point_sensitivity.mean() >= H24_MIN_SENS and res.band_sensitivity.mean() >= H24_MIN_SENS
    verdict = "SUPPORTED" if saving >= H24_SAVING and sens_ok else "NOT SUPPORTED"
    text = "\n".join([
        f"H24 (band alarm needs at least 20% fewer false alarms per day at 90% sensitivity): {verdict}",
        f"  patients with validation lows: {len(res)} of {len(frames)}",
        f"  pooled alarm onsets per day: point {point_rate:.3f}, band {band_rate:.3f}, band saving {saving:.1%}",
        f"  mean sensitivity on test: point {res.point_sensitivity.mean():.2f}, band {res.band_sensitivity.mean():.2f}",
        f"  patients where the band alarm has fewer onsets: {(res.band_onsets < res.point_onsets).sum()} of {len(res)}",
        "  caveat: sensitivity on validation is noisy when lows are rare, and the test numbers vary with it",
    ])
    return text, res


# ------------------------------------------------------------------ H25

def h25(frames) -> tuple[str, pd.DataFrame]:
    rows = []
    for sid, (val, test) in frames.items():
        eng = unc.BandEngine(val, test, analogs=False)
        lo, hi = eng.bands("Conformal", 0.80)
        cov = float(np.mean((test.actual >= lo) & (test.actual <= hi)))
        rows.append({"subject_id": sid, "coverage_80": cov,
                     "inside_75_85": H25_LO <= cov <= H25_HI,
                     "band_halfwidth": float(np.nanmedian((hi - lo) / 2))})
    res = pd.DataFrame(rows)
    count = int(res.inside_75_85.sum())
    verdict = "SUPPORTED" if count >= H25_MIN_COUNT else "NOT SUPPORTED"
    text = "\n".join([
        f"H25 (conformal 80% coverage between 75% and 85% for at least 20 of 25 patients): {verdict}",
        f"  patients inside 75% to 85%: {count} of {len(res)}",
        f"  coverage range across patients: {res.coverage_80.min():.1%} to {res.coverage_80.max():.1%}",
        f"  median coverage: {res.coverage_80.median():.1%}",
    ])
    return text, res


def main():
    t0 = time.time()
    out = WEEK / "results"
    out.mkdir(parents=True, exist_ok=True)
    frames = load_frames()
    print(f"loaded {len(frames)} patients ({time.time() - t0:.0f}s)", flush=True)

    t23 = h23(frames)
    (out / "h23_lookalike.txt").write_text(t23 + "\n")
    print(t23, flush=True)

    t24, r24 = h24(frames)
    r24.to_csv(out / "h24_alarms.csv", index=False)
    print(t24, flush=True)

    t25, r25 = h25(frames)
    r25.to_csv(out / "h25_conformal_coverage.csv", index=False)
    print(t25, flush=True)

    (out / "verdicts_h23_h25.txt").write_text("\n\n".join([t23, t24, t25]) + "\n")
    print(f"runtime {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
