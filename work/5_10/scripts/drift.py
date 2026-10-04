"""H29: does the plain model's error grow from the validation period to the test period?

AZT1D plain checkpoints (cnn_lstm_v0). For each patient, RMSE in three consecutive periods:
validation, first half of test, second half of test. Also each period's glucose variability
(SD of the true values), to see whether a rise in error comes with harder data.

H29 supported if the median per-patient ratio test RMSE / validation RMSE is at least 1.10,
and at least 15 of 25 patients exceed 1.10.

Run:    python work/5_10/scripts/drift.py
Writes: work/5_10/results/h29_drift.csv, verdict_h29.txt, work/5_10/figures/h29_drift.png
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = next(p for p in Path(__file__).resolve().parents if (p / "pyproject.toml").exists())
WEEK = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from azt1d import loading, plotting  # noqa: E402
from azt1d.glimmer import checkpoint as ckpt, uncertainty as unc  # noqa: E402

RUN = "cnn_lstm_v0"
RATIO, MIN_PATIENTS = 1.10, 15


def rmse(e):
    return float(np.sqrt(np.mean(np.asarray(e) ** 2)))


def main():
    out, figs = WEEK / "results", WEEK / "figures"
    df = loading.load_real_dataset(ROOT / "data" / "raw", ROOT / "data" / "processed")
    rows = []
    for sid in sorted(int(s) for s in df["subject_id"].unique()):
        res = ckpt.load_result(ROOT / "data" / "processed" / "checkpoints" / RUN, sid)
        val, test = unc.forecast_frames(res, df[df["subject_id"] == sid].reset_index(drop=True))
        h = len(test.resid) // 2
        rows.append({
            "subject_id": sid,
            "rmse_val": rmse(val.resid), "rmse_test1": rmse(test.resid[:h]), "rmse_test2": rmse(test.resid[h:]),
            "rmse_test": rmse(test.resid),
            "sd_val": float(np.std(val.actual)), "sd_test": float(np.std(test.actual)),
        })
    t = pd.DataFrame(rows)
    t["ratio"] = t.rmse_test / t.rmse_val
    t["sd_ratio"] = t.sd_test / t.sd_val
    t.to_csv(out / "h29_drift.csv", index=False)
    med, n_up = float(t.ratio.median()), int((t.ratio >= RATIO).sum())
    corr = float(np.corrcoef(np.log(t.sd_ratio), np.log(t.ratio))[0, 1])
    ok = med >= RATIO and n_up >= MIN_PATIENTS
    text = "\n".join([
        f"H29 (error grows from validation to test): {'SUPPORTED' if ok else 'NOT SUPPORTED'}",
        f"  median ratio test/validation RMSE: {med:.2f} (rule: at least {RATIO:.2f})",
        f"  patients with ratio at least {RATIO:.2f}: {n_up} of {len(t)} (rule: at least {MIN_PATIENTS}); patients with lower test error: {int((t.ratio < 1).sum())}",
        f"  ratio range: {t.ratio.min():.2f} to {t.ratio.max():.2f}",
        f"  median RMSE by period: validation {t.rmse_val.median():.1f}, test first half {t.rmse_test1.median():.1f}, test second half {t.rmse_test2.median():.1f} mg/dL",
        f"  correlation between change in glucose variability and change in error (log scale): {corr:.2f}",
    ])
    (out / "verdict_h29.txt").write_text(text + "\n")
    print(text)

    plotting.apply_style()
    C = plotting.CATEGORICAL
    fig, (a, b) = plt.subplots(1, 2, figsize=(13, 5.4))
    lim = [min(t.rmse_val.min(), t.rmse_test.min()) * 0.9, max(t.rmse_val.max(), t.rmse_test.max()) * 1.05]
    a.fill_between(lim, [l * RATIO for l in lim], [lim[1] * 2] * 2, color=C[1], alpha=0.08, label="at least 10% worse on test")
    a.plot(lim, lim, ls="--", color=plotting.BASELINE, label="same error")
    a.scatter(t.rmse_val, t.rmse_test, color=C[0], s=40, zorder=3)
    for _, r in t.iterrows():
        a.annotate(str(int(r.subject_id)), (r.rmse_val, r.rmse_test), fontsize=7, xytext=(3, 2), textcoords="offset points")
    a.set_xlim(lim); a.set_ylim(lim)
    a.set_xlabel("Error on the validation period (RMSE, mg/dL)")
    a.set_ylabel("Error on the later test period (RMSE, mg/dL)")
    a.set_title(f"Each dot is a patient: {n_up} of {len(t)} are at least 10% worse later", loc="left", fontsize=10)
    a.legend(frameon=False, fontsize=8, loc="upper left")
    b.axhline(1, ls="--", color=plotting.BASELINE)
    b.axvline(1, ls="--", color=plotting.BASELINE)
    b.scatter(t.sd_ratio, t.ratio, color=C[0], s=40)
    for _, r in t.iterrows():
        b.annotate(str(int(r.subject_id)), (r.sd_ratio, r.ratio), fontsize=7, xytext=(3, 2), textcoords="offset points")
    b.set_xlabel("Glucose swings later vs earlier (test SD / validation SD)")
    b.set_ylabel("Error later vs earlier (test RMSE / validation RMSE)")
    b.set_title(f"Do bigger swings explain bigger errors? correlation {corr:.2f}", loc="left", fontsize=10)
    fig.suptitle(f"Does the model get worse over time? Median patient: {med:.2f}x the earlier error",
                 x=0.01, ha="left", fontsize=11)
    fig.tight_layout()
    fig.savefig(figs / "h29_drift.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
