"""H41 and H42: does a patient's compressibility predict the CNN-LSTM's forecast error beyond
glucose variability?

For each AZT1D patient, using only the first 80% of their data (the target is later):
  - glucose SD (the bar to beat)
  - sample entropy (H41): glucose with each day's average removed, m=2, r=0.2 x SD,
    non-overlapping 7-day blocks, averaged
  - bits per step (H42): a small per-patient MLP predicts the next 5-minute glucose change, mean
    and variance, from the last 2 hours (glucose minus daily average, basal, bolus, carbs; true
    history = teacher forced). Trained on the first 64%, early-stopped and scored on the next 16%.
    Average Gaussian surprise, in bits per step (glucose in mg/dL).
Target: plain CNN-LSTM (cnn_lstm_v0) test RMSE on the last 20%.

Rule: adding the measure to glucose SD cuts the leave-one-patient-out error by at least 15%.

Run:    python work/5_10/scripts/compressibility.py
Writes: work/5_10/results/compressibility_patients.csv, verdict_h41_h42.txt,
        work/5_10/figures/compressibility.png
"""
from __future__ import annotations

import copy
import math
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr
from torch import nn

ROOT = next(p for p in Path(__file__).resolve().parents if (p / "pyproject.toml").exists())
WEEK = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from azt1d import loading, plotting, reference as ref  # noqa: E402
from azt1d.glimmer import checkpoint as ckpt  # noqa: E402

LOOKBACK, BLOCK = 24, 7 * 288
CUT_EARLY, CUT_TRAIN = 0.80, 0.64
H_CUT = 0.15


def detrend_daily(d: pd.DataFrame) -> np.ndarray:
    day = pd.to_datetime(d[ref.EVENT_DATETIME]).dt.floor("D")
    g = d[ref.CGM].astype(float)
    return (g - g.groupby(day).transform("mean")).to_numpy()


def sampen(x: np.ndarray, m: int = 2, r_frac: float = 0.2) -> float:
    r = r_frac * np.std(x)
    def count(mm):
        t = np.lib.stride_tricks.sliding_window_view(x, mm)[: len(x) - m]
        d = np.max(np.abs(t[:, None, :] - t[None, :, :]), axis=2)
        return (np.sum(d <= r) - len(t)) / 2
    b, a = count(m), count(m + 1)
    return float(-np.log(a / b)) if a > 0 and b > 0 else float("nan")


class NextStep(nn.Module):
    def __init__(self, n_in):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(n_in, 32), nn.ReLU(), nn.Linear(32, 2))

    def forward(self, x):
        mu, logv = self.net(x).unbind(-1)
        return mu, logv


def gauss_nll(mu, logv, y):
    return 0.5 * (math.log(2 * math.pi) + logv + (y - mu) ** 2 / torch.exp(logv))


def bits_per_step(d: pd.DataFrame, seed: int) -> float:
    rel = detrend_daily(d)
    g = d[ref.CGM].to_numpy(float)
    other = d[[ref.BASAL, ref.TOTAL_BOLUS_INSULIN_DELIVERED, ref.CARB_SIZE]].to_numpy(float)
    feats = np.column_stack([rel, other])
    n = len(d) - LOOKBACK - 1
    X = np.stack([feats[i:i + LOOKBACK].ravel() for i in range(n)])
    y = g[np.arange(n) + LOOKBACK] - g[np.arange(n) + LOOKBACK - 1]  # next 5-minute change, mg/dL
    n_early = int(n * CUT_EARLY / 1.0)
    tr, va = slice(0, int(n * CUT_TRAIN)), slice(int(n * CUT_TRAIN), n_early)
    mx, sx = X[tr].mean(0), X[tr].std(0) + 1e-6
    t = lambda a: torch.tensor((a - mx) / sx, dtype=torch.float32)  # noqa: E731
    Xtr, Xva = t(X[tr]), t(X[va])
    ytr, yva = torch.tensor(y[tr], dtype=torch.float32), torch.tensor(y[va], dtype=torch.float32)
    torch.manual_seed(seed)
    m = NextStep(X.shape[1])
    opt = torch.optim.Adam(m.parameters(), lr=1e-3)
    best, best_state, stale = float("inf"), None, 0
    for _ in range(60):
        perm = torch.randperm(len(Xtr))
        for b in range(0, len(Xtr), 256):
            i = perm[b:b + 256]
            loss = gauss_nll(*m(Xtr[i]), ytr[i]).mean()
            opt.zero_grad(); loss.backward(); opt.step()
        with torch.no_grad():
            v = gauss_nll(*m(Xva), yva).mean().item()
        if v < best:
            best, best_state, stale = v, copy.deepcopy(m.state_dict()), 0
        else:
            stale += 1
            if stale >= 5:
                break
    return best / math.log(2)  # nats to bits


def loo_err(Xf, y):
    Xf, y = np.column_stack([np.ones(len(y)), Xf]), np.asarray(y, float)
    errs = []
    for k in range(len(y)):
        msk = np.arange(len(y)) != k
        beta = np.linalg.lstsq(Xf[msk], y[msk], rcond=None)[0]
        errs.append((Xf[k] @ beta - y[k]) ** 2)
    return float(np.sqrt(np.mean(errs)))


def main():
    t0 = time.time()
    df = loading.load_real_dataset(ROOT / "data" / "raw", ROOT / "data" / "processed")
    rows = []
    for sid in sorted(int(s) for s in df.subject_id.unique()):
        d = df[df.subject_id == sid].reset_index(drop=True)
        early = d.iloc[: int(len(d) * CUT_EARLY)].reset_index(drop=True)
        rel = detrend_daily(early)
        se = [sampen(rel[s:s + BLOCK]) for s in range(0, len(rel) - BLOCK + 1, BLOCK)]
        res = ckpt.load_result(ROOT / "data" / "processed" / "checkpoints" / "cnn_lstm_v0", sid)
        rows.append({"subject_id": sid, "glucose_sd": float(early[ref.CGM].std()),
                     "glucose_mean": float(early[ref.CGM].mean()),
                     "sample_entropy": float(np.nanmean(se)), "n_blocks": len(se),
                     "bits_per_step": bits_per_step(d, seed=sid), "forecast_rmse": float(res.rmse)})
        print(f"patient {sid} done ({time.time() - t0:.0f}s)", flush=True)
    t = pd.DataFrame(rows)
    t.to_csv(WEEK / "results" / "compressibility_patients.csv", index=False)

    base = loo_err(t[["glucose_sd"]].to_numpy(), t.forecast_rmse)
    lines = [f"Baseline: glucose SD alone, leave-one-patient-out error {base:.2f} mg/dL"]
    for hid, col, name in (("H41", "sample_entropy", "sample entropy"), ("H42", "bits_per_step", "bits per step")):
        both = loo_err(t[["glucose_sd", col]].to_numpy(), t.forecast_rmse)
        alone = loo_err(t[[col]].to_numpy(), t.forecast_rmse)
        cut = 1 - both / base
        v = "SUPPORTED" if cut >= H_CUT else "NOT SUPPORTED"
        lines.append(f"{hid} ({name} adds at least 15% beyond glucose SD): {v}")
        lines.append(f"  LOO error: SD alone {base:.2f}, {name} alone {alone:.2f}, SD + {name} {both:.2f} mg/dL; cut vs SD alone {cut:+.0%}")
        lines.append(f"  Spearman with forecast RMSE: {name} {spearmanr(t[col], t.forecast_rmse).statistic:+.2f}; "
                     f"{name} with glucose SD {spearmanr(t[col], t.glucose_sd).statistic:+.2f}")
    lines.append(f"  Spearman glucose SD with forecast RMSE: {spearmanr(t.glucose_sd, t.forecast_rmse).statistic:+.2f}")
    lines.append(f"  ranges: sample entropy {t.sample_entropy.min():.2f} to {t.sample_entropy.max():.2f}; bits per step "
                 f"{t.bits_per_step.min():.2f} to {t.bits_per_step.max():.2f}; forecast RMSE {t.forecast_rmse.min():.1f} to {t.forecast_rmse.max():.1f}")
    lines.append(f"  runtime {time.time() - t0:.0f}s")
    text = "\n".join(lines)
    (WEEK / "results" / "verdict_h41_h42.txt").write_text(text + "\n")
    print(text)

    # residual of forecast RMSE after glucose SD (what SD can't explain)
    X1 = np.column_stack([np.ones(len(t)), t.glucose_sd])
    beta = np.linalg.lstsq(X1, t.forecast_rmse, rcond=None)[0]
    t["rmse_left_over"] = t.forecast_rmse - X1 @ beta

    plotting.apply_style()
    C = plotting.CATEGORICAL
    fig, axes = plt.subplots(1, 3, figsize=(17, 5.2))
    a = axes[0]
    a.scatter(t.glucose_sd, t.forecast_rmse, color=C[0], s=45)
    for r in t.itertuples():
        a.annotate(str(r.subject_id), (r.glucose_sd, r.forecast_rmse), fontsize=7, xytext=(3, 2), textcoords="offset points")
    a.set_xlabel("Glucose variability (SD, mg/dL), earlier 80%")
    a.set_ylabel("CNN-LSTM forecast error (RMSE, mg/dL), test")
    a.set_title(f"The bar to beat: variability vs error (Spearman {spearmanr(t.glucose_sd, t.forecast_rmse).statistic:+.2f})",
                loc="left", fontsize=10)
    for ax, col, name in ((axes[1], "sample_entropy", "Sample entropy (higher = more irregular)"),
                          (axes[2], "bits_per_step", "Bits per step (higher = harder to compress)")):
        ax.axhline(0, ls="--", color=plotting.BASELINE)
        ax.scatter(t[col], t.rmse_left_over, color=C[1], s=45)
        for r in t.itertuples():
            ax.annotate(str(r.subject_id), (getattr(r, col), r.rmse_left_over), fontsize=7, xytext=(3, 2), textcoords="offset points")
        ax.set_xlabel(name + ", earlier 80%")
        ax.set_ylabel("Error left over after variability (mg/dL)")
        rr = spearmanr(t[col], t.rmse_left_over).statistic
        ax.set_title(f"Does it explain what variability can't? (Spearman {rr:+.2f})", loc="left", fontsize=10)
    fig.suptitle("Does how compressible a patient's glucose is predict how badly the CNN-LSTM forecasts them? (AZT1D, 25 patients)",
                 x=0.01, ha="left", fontsize=11)
    fig.tight_layout()
    fig.savefig(WEEK / "figures" / "compressibility.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
