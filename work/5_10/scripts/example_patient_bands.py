"""One-patient example: GARCH 80% band vs situation-specific 80% band (OhioT1DM).

Window: the 24 hours of the patient's test period with the widest glucose range, chosen by that
rule (not by which method looks better). Coverage over the whole test period is printed too.

Run:    python work/5_10/scripts/example_patient_bands.py [subject_id]   (default 588)
Writes: work/5_10/figures/example_patient_<id>_bands.png
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
sys.path.insert(0, str(Path(__file__).resolve().parent))

from azt1d import plotting, reference as ref  # noqa: E402
from azt1d.metabonet import load_source  # noqa: E402
from azt1d.glimmer import checkpoint as ckpt, uncertainty as unc  # noqa: E402
import predictability_map as pm  # noqa: E402
from ohio_situation_bands import build, add_situation_bands, RUN  # noqa: E402

WINDOW = 288  # 24 hours of 5-minute steps


def main(sid: int):
    val, test = build()
    test, _ = add_situation_bands(val, test)
    t = test[test.subject_id == sid].reset_index(drop=True)

    # timestamps for this patient's kept test rows (same keep rule as build())
    df = load_source("OhioT1DM")
    dfs = df[df["subject_id"] == sid].reset_index(drop=True)
    res = ckpt.load_result(ROOT / "data" / "processed" / "checkpoints" / RUN, sid)
    _, tf = unc.forecast_frames(res, dfs)
    times = pd.to_datetime(dfs[ref.EVENT_DATETIME]).to_numpy()
    keep = np.searchsorted(times, tf.issue_time) >= pm.TREND_STEPS
    t["time"] = pd.to_datetime(tf.target_time[keep])

    rng = t.actual.rolling(WINDOW).max() - t.actual.rolling(WINDOW).min()
    end = int(rng.idxmax()) + 1
    w = t.iloc[end - WINDOW:end]

    plotting.apply_style()
    C = plotting.CATEGORICAL
    fig, axes = plt.subplots(2, 1, figsize=(13, 8.2), sharex=True, sharey=True)
    panels = [("garch_lo", "garch_hi", "in_garch", "Current: GARCH 80% band", plotting.INK_MUTED),
              ("sit_lo", "sit_hi", "in_sit", "New: situation-specific 80% band", C[0])]
    for ax, (lo, hi, inside, title, col) in zip(axes, panels):
        ax.fill_between(w.time, w[lo], w[hi], color=col, alpha=0.25, lw=0, label="80% band")
        ax.plot(w.time, w.pred, color=col, lw=1.2, label="forecast (60 min ahead)")
        ax.plot(w.time, w.actual, color=plotting.INK_PRIMARY, lw=1.4, label="actual glucose")
        miss = ~w[inside]
        ax.scatter(w.time[miss], w.actual[miss], color=C[1], s=14, zorder=4, label="actual fell outside the band")
        ax.axhline(ref.HYPO_THRESHOLD, ls="--", lw=0.8, color=plotting.GLUCOSE_BAND_COLORS["hypo"])
        ax.axhline(ref.HYPER_THRESHOLD, ls="--", lw=0.8, color=plotting.GLUCOSE_BAND_COLORS["hyper"])
        cov_w, cov_all = w[inside].mean(), t[inside].mean()
        width = (w[hi] - w[lo]).mean()
        ax.set_title(f"{title}: caught {cov_w:.0%} in this window, {cov_all:.0%} over the whole test period; "
                     f"average width {width:.0f} mg/dL", loc="left", fontsize=10)
        ax.set_ylabel("Glucose (mg/dL)")
    axes[0].legend(frameon=False, fontsize=8, ncol=4, loc="upper left")
    axes[1].set_xlabel("Time (24 hours with the widest glucose range in this patient's test period)")
    fig.suptitle(f"OhioT1DM patient {sid}: same forecast, two ways of drawing the 80% band (target: catch 80%)",
                 x=0.01, ha="left", fontsize=11)
    fig.tight_layout()
    path = WEEK / "figures" / f"example_patient_{sid}_bands.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"window {w.time.iloc[0]} to {w.time.iloc[-1]}")
    print(f"GARCH: window {w.in_garch.mean():.1%}, whole {t.in_garch.mean():.1%}; "
          f"situation: window {w.in_sit.mean():.1%}, whole {t.in_sit.mean():.1%}")
    print("wrote", path)


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 588)
