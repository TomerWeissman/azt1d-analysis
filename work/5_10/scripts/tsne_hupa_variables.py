"""Quick look: the HUPA-UCM window-level t-SNE of the encoder summary, coloured by each recorded
variable, summarized over the window's 2 hours (the encoder's input span).

Same model, windows, seed and t-SNE settings as tsne_bottleneck.py, so the layout matches the
earlier HUPA maps. The encoder's inputs were glucose, basal, bolus and carbs; heart rate, steps,
calories and time of day were not inputs.

Per window (rows i to i+23 of the patient's raw file):
  glucose: last reading | basal: mean rate | bolus, carbs: total | heart rate: mean |
  steps, calories: total | time of day: hour of the last reading

Run:    python work/5_10/scripts/tsne_hupa_variables.py
Writes: work/5_10/figures/tsne_hupa_variables.png
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LogNorm
from sklearn.manifold import TSNE

ROOT = next(p for p in Path(__file__).resolve().parents if (p / "pyproject.toml").exists())
WEEK = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from azt1d import hupa, plotting  # noqa: E402
from tsne_bottleneck import embed, SEED  # noqa: E402
from teacher_forcing_autoencoder import LOOKBACK  # noqa: E402

VARIABLES = [  # column, how summarized, label, input to the encoder?, colour map, log scale
    ("glucose", "last", "glucose, last reading (mg/dL)", True, "magma", False),
    ("basal_rate", "mean", "basal insulin, mean (U per 5 min)", True, "viridis", False),
    ("bolus_volume_delivered", "sum", "bolus insulin, 2-hour total (U)", True, "viridis", True),
    ("carb_input", "sum", "carbs, 2-hour total (g)", True, "viridis", True),
    ("heart_rate", "mean", "heart rate, mean (bpm)", False, "plasma", False),
    ("steps", "sum", "steps, 2-hour total", False, "plasma", True),
    ("calories", "sum", "calories, 2-hour total", False, "plasma", False),
    ("hour", "last", "time of day (hour of last reading)", False, "twilight", False),
]


def window_values(picks):
    out = {v[0]: [] for v in VARIABLES}
    for sid, starts in picks:
        raw = pd.read_csv(hupa.subject_file(sid, ROOT / hupa.DEFAULT_DIR), sep=";", parse_dates=["time"])
        raw = raw.sort_values("time").reset_index(drop=True)
        raw["hour"] = raw.time.dt.hour + raw.time.dt.minute / 60
        for col, how, *_ in VARIABLES:
            a = raw[col].to_numpy(float)
            w = np.stack([a[s:s + LOOKBACK] for s in starts])
            out[col].append(w[:, -1] if how == "last" else (w.mean(1) if how == "mean" else w.sum(1)))
    return {k: np.concatenate(v) for k, v in out.items()}


def main():
    t0 = time.time()
    Z, S, G, rmse, picks = embed("HUPA-UCM", return_picks=True)
    vals = window_values(picks)
    assert np.allclose(vals["glucose"], G), "raw file rows do not line up with the encoded windows"
    T = TSNE(n_components=2, perplexity=30, init="pca", random_state=SEED).fit_transform(Z)
    print(f"encoded and mapped ({time.time() - t0:.0f}s)", flush=True)

    plotting.apply_style()
    fig, axes = plt.subplots(2, 4, figsize=(22, 11))
    for ax, (col, _, label, was_input, cmap, log) in zip(axes.ravel(), VARIABLES):
        v = vals[col]
        if log:
            zero = v <= 0
            ax.scatter(T[zero, 0], T[zero, 1], color="#d9d9d9", s=3, alpha=0.5, label="zero")
            sc = ax.scatter(T[~zero, 0], T[~zero, 1], c=v[~zero], cmap=cmap, s=4, alpha=0.8,
                            norm=LogNorm(vmin=max(v[~zero].min(), 1e-3), vmax=np.percentile(v[~zero], 99)))
            ax.legend(frameon=False, fontsize=7, loc="lower left", markerscale=3)
            share = f"; nonzero in {(~zero).mean():.0%} of windows"
        else:
            lo, hi = np.percentile(v, [1, 99])
            sc = ax.scatter(T[:, 0], T[:, 1], c=v, cmap=cmap, s=3, alpha=0.6, vmin=lo, vmax=hi)
            share = ""
        fig.colorbar(sc, ax=ax, shrink=0.75)
        tag = "encoder input" if was_input else "NOT an encoder input"
        ax.set_title(f"{label}\n({tag}{share})", loc="left", fontsize=9)
        ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle("HUPA-UCM: the same t-SNE map of the encoder summary, coloured by each recorded variable "
                 "(each dot = one 2-hour window)", x=0.01, ha="left", fontsize=11)
    fig.tight_layout()
    fig.savefig(WEEK / "figures" / "tsne_hupa_variables.png", dpi=150)
    plt.close(fig)
    print(f"runtime {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
