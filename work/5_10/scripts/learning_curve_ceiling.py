"""H30: would more training data cut the plain model's error by at least 25%?

For each AZT1D patient, retrain the plain per-patient CNN-LSTM (same code and settings as the
saved cnn_lstm_v0 run) on random subsets of its training windows: 10%, 20%, 35%, 60%, and 100%,
two seeds each (each run gets its own seed). Validation and test windows are unchanged. Early
stopping uses validation only. The test period is only scored.

Fit pooled test MSE = a + b * fraction^(-c). The asymptote a is the error this model class would
reach with unlimited data like this. A 90% bootstrap interval over patients gives its uncertainty.

H30 supported if a is at most 75% of the full-data MSE and the whole interval is below 75%.
Not supported if the whole interval is above 75%. Inconclusive otherwise.

Note: input and target scaling use the full training split's statistics at every fraction. That
is training-set information only, never test.

Run:    python work/5_10/scripts/learning_curve_ceiling.py
Writes: work/5_10/results/learning_curve_runs.csv, work/5_10/results/verdict_h30.txt,
        work/5_10/figures/learning_curve_ceiling.png
"""
from __future__ import annotations

import dataclasses
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from scipy.optimize import curve_fit

ROOT = next(p for p in Path(__file__).resolve().parents if (p / "pyproject.toml").exists())
WEEK = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from azt1d import loading, plotting  # noqa: E402
from azt1d.glimmer.train import prepare_subject_data, train_prepared_model  # noqa: E402

FRACTIONS = (0.10, 0.20, 0.35, 0.60, 1.00)
SEEDS = 2
H30_LIMIT = 0.75
N_BOOT = 500


def curve(f, a, b, c):
    return a + b * np.power(f, -c)


def fit_asymptote(fracs: np.ndarray, mse: np.ndarray) -> tuple[float, np.ndarray]:
    p0 = [mse.min() * 0.8, max(mse[0] - mse.min(), 1.0) * 0.1, 0.5]
    params, _ = curve_fit(curve, fracs, mse, p0=p0, bounds=([0, 0, 0.01], [np.inf, np.inf, 3.0]), maxfev=20000)
    return float(params[0]), params


def pooled_mse(runs: pd.DataFrame, patients) -> np.ndarray:
    """Pooled test MSE per fraction (patients weighted by test size), averaged over seeds."""
    sub = runs[runs.subject_id.isin(patients)] if patients is not None else runs
    out = []
    for f in FRACTIONS:
        r = sub[sub.fraction == f]
        per_seed = r.groupby("seed").apply(lambda g: g.sse.sum() / g.n_test.sum(), include_groups=False)
        out.append(per_seed.mean())
    return np.array(out)


def run_trainings(out_csv: Path) -> pd.DataFrame:
    df = loading.load_real_dataset(ROOT / "data" / "raw", ROOT / "data" / "processed")
    device = torch.device("cpu")
    rows, t0 = [], time.time()
    for sid in sorted(int(s) for s in df["subject_id"].unique()):
        data = prepare_subject_data(df[df["subject_id"] == sid].reset_index(drop=True))
        n = len(data.X_train)
        for k, frac in enumerate(FRACTIONS):
            for s in range(SEEDS):
                seed = sid * 1000 + k * 10 + s
                idx = np.sort(np.random.default_rng(seed).choice(n, max(int(round(frac * n)), 64), replace=False))
                sub = dataclasses.replace(data, X_train=data.X_train[idx], y_train=data.y_train[idx])
                res = train_prepared_model(sub, device=device, seed=seed)
                err = data.y_test - res.y_pred
                rows.append({"subject_id": sid, "fraction": frac, "seed": s, "n_train": len(idx),
                             "n_test": len(err), "sse": float(np.sum(err ** 2)), "rmse": res.rmse})
        pd.DataFrame(rows).to_csv(out_csv, index=False)
        print(f"patient {sid} done ({time.time() - t0:.0f}s)", flush=True)
    return pd.DataFrame(rows)


def main():
    t0 = time.time()
    out, figs = WEEK / "results", WEEK / "figures"
    out.mkdir(parents=True, exist_ok=True)
    figs.mkdir(parents=True, exist_ok=True)
    runs = run_trainings(out / "learning_curve_runs.csv")

    fr = np.array(FRACTIONS)
    mse = pooled_mse(runs, None)
    a, params = fit_asymptote(fr, mse)
    full = mse[-1]
    ratio = a / full

    patients = runs.subject_id.unique()
    rng = np.random.default_rng(0)
    boot = []
    for _ in range(N_BOOT):
        pick = rng.choice(patients, len(patients), replace=True)
        rows = pd.concat([runs[runs.subject_id == p].assign(subject_id=i) for i, p in enumerate(pick)])
        m = pooled_mse(rows, None)
        try:
            ab, _ = fit_asymptote(fr, m)
            boot.append(ab / m[-1])
        except RuntimeError:
            pass
    lo, hi = np.percentile(boot, [5, 95])
    if ratio <= H30_LIMIT and hi < H30_LIMIT:
        v = "SUPPORTED"
    elif lo > H30_LIMIT:
        v = "NOT SUPPORTED"
    else:
        v = "INCONCLUSIVE"
    text = "\n".join([
        f"H30 (unlimited data would cut error at least 25%): {v}",
        f"  pooled test MSE by training fraction: " + ", ".join(f"{f:.0%}: {m:.0f}" for f, m in zip(fr, mse)),
        f"  full-data MSE {full:.0f} (RMSE {np.sqrt(full):.1f} mg/dL)",
        f"  fitted asymptote {a:.0f} (RMSE {np.sqrt(a):.1f} mg/dL) = {ratio:.0%} of full-data MSE",
        f"  90% bootstrap interval over patients: {lo:.0%} to {hi:.0%} ({len(boot)} successful fits of {N_BOOT})",
        f"  fit: a={params[0]:.1f}, b={params[1]:.2f}, c={params[2]:.2f}",
        "  caveat: the asymptote is for this model class and these inputs, and assumes more data looks like this data",
    ])
    (out / "verdict_h30.txt").write_text(text + "\n")
    print(text, flush=True)

    plotting.apply_style()
    C = plotting.CATEGORICAL
    fig, ax = plt.subplots(figsize=(9, 5.5))
    per_seed = [pooled_mse(runs[runs.seed == s], None) for s in range(SEEDS)]
    for s, m in enumerate(per_seed):
        ax.scatter(fr * 100, np.sqrt(m), color=C[0], s=30, alpha=0.6, label="measured (each seed)" if s == 0 else None)
    xs = np.linspace(0.08, 10, 400)
    ax.plot(xs * 100, np.sqrt(curve(xs, *params)), color=C[0], lw=1.8, label="fitted curve")
    ax.axhline(np.sqrt(a), color=C[1], ls="--", lw=1.5, label=f"where it levels off: {np.sqrt(a):.1f} mg/dL")
    ax.axhline(np.sqrt(H30_LIMIT * full), color=plotting.BASELINE, ls=":", lw=1.5,
               label=f"25% lower than today's error: {np.sqrt(H30_LIMIT * full):.1f} mg/dL")
    ax.axvline(100, color=plotting.INK_MUTED, lw=0.8)
    ax.text(102, ax.get_ylim()[1] * 0.98, "today's data", fontsize=8, va="top", color=plotting.INK_SECONDARY)
    ax.set_xscale("log")
    ax.set_xticks([10, 20, 35, 60, 100, 300, 1000], ["10%", "20%", "35%", "60%", "100%", "3x", "10x"])
    ax.set_xlabel("Training data used (log scale; right of the line is extrapolated)")
    ax.set_ylabel("Typical forecast error, RMSE (mg/dL)")
    ax.set_title("Would more data help? Forecast error vs amount of training data", loc="left", fontsize=11)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(figs / "learning_curve_ceiling.png", dpi=150)
    plt.close(fig)
    print(f"runtime {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
