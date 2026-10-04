"""Presentation figure: the strength of situation-aware bands, and what it means in real life.

Data: OhioT1DM, the fresh-data confirmation (H28), plain model, 80% bands.
Left: every glucose situation's actual catch rate, GARCH vs situation-aware.
Right: four everyday situations, what was promised (80%) vs what each band delivered.
Footnote states the trade-off found in H38.

Run:    python work/5_10/scripts/presentation_figure.py
Writes: work/5_10/figures/presentation_situation_aware.png
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.transforms import blended_transform_factory
import numpy as np
import pandas as pd

ROOT = next(p for p in Path(__file__).resolve().parents if (p / "pyproject.toml").exists())
WEEK = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from azt1d import plotting  # noqa: E402

EXAMPLES = [
    ("70 to 120", "fast rise", "Rising fast\nfrom normal"),
    ("70 to 120", "fast fall", "Falling fast\ntoward a low"),
    ("250 and over", "fast rise", "Very high and\nstill climbing"),
    ("120 to 180", "flat", "Steady\nin range"),
]


def main():
    d = pd.read_csv(WEEK / "results" / "h28_ohio_situations.csv")
    plotting.apply_style()
    C = plotting.CATEGORICAL
    GREY, BLUE = plotting.INK_MUTED, C[0]
    fig, (a, b) = plt.subplots(1, 2, figsize=(15, 6.6), gridspec_kw={"width_ratios": [1, 1.35]})

    # Left: spread of actual catch rates across situations
    a.axhspan(75, 85, color=C[2], alpha=0.12)
    a.axhline(80, ls="--", color=plotting.BASELINE)
    a.text(1.55, 80, "promised\n80%", va="center", fontsize=8, color=plotting.INK_SECONDARY)
    rng = np.random.default_rng(3)
    for x, col, colr in ((0, "coverage_garch", GREY), (1, "coverage_situation", BLUE)):
        jitter = rng.uniform(-0.12, 0.12, len(d))
        a.scatter(x + jitter, d[col] * 100, s=np.sqrt(d.n_test) * 2.2, color=colr, alpha=0.75, edgecolor="white", lw=0.5)
        lo, hi = d[col].min() * 100, d[col].max() * 100
        a.plot([x + 0.22, x + 0.22], [lo, hi], color=colr, lw=2)
        a.text(x + 0.27, (lo + hi) / 2, f"{lo:.0f}% to\n{hi:.0f}%", va="center", fontsize=9, color=colr)
    a.set_xticks([0, 1], ["Standard band\n(GARCH)", "Situation-aware\nband"], fontsize=10)
    a.set_xlim(-0.45, 1.8)
    a.set_ylim(55, 102)
    a.set_ylabel("Share of real glucose values the band actually caught (%)")
    a.set_title("Each dot is one glucose situation (bigger dot = more readings)", loc="left", fontsize=10)
    gap_g = (d.coverage_garch - 0.8).abs().mean() * 100
    gap_s = (d.coverage_situation - 0.8).abs().mean() * 100
    a.text(0.02, 0.02, f"Average distance from 80%: {gap_g:.1f} points (standard) vs {gap_s:.1f} points (situation-aware)",
           transform=a.transAxes, fontsize=8.5, color=plotting.INK_SECONDARY)

    # Right: everyday situations
    x = np.arange(len(EXAMPLES))
    w = 0.36
    for i, (lvl, trd, label) in enumerate(EXAMPLES):
        r = d[(d.level == lvl) & (d.trend == trd)].iloc[0]
        g, s = r.coverage_garch * 100, r.coverage_situation * 100
        b.bar(i - w / 2, g, w, color=GREY)
        b.bar(i + w / 2, s, w, color=BLUE)
        b.text(i - w / 2, g + 1, f"{g:.0f}%", ha="center", fontsize=9)
        b.text(i + w / 2, s + 1, f"{s:.0f}%", ha="center", fontsize=9)
        if g < 75:
            note = "Standard band too narrow:\nfalse reassurance, misses\n" + f"{100 - g:.0f} in 100 instead of 20"
        elif g > 85:
            note = "Standard band too wide:\nworries more than needed\n(alarm fatigue)"
        else:
            note = "Both close;\nnew band slightly narrow"
        b.text(i, -0.17, note, ha="center", va="top", fontsize=8, color=plotting.INK_SECONDARY,
               transform=blended_transform_factory(b.transData, b.transAxes), clip_on=False)
    b.axhline(80, ls="--", color=plotting.BASELINE)
    b.set_xticks(x, [e[2] for e in EXAMPLES], fontsize=9)
    b.set_ylim(0, 108)
    b.set_ylabel("Caught by the band (%), promised 80%")
    b.legend(handles=[Patch(color=GREY, label="Standard band (GARCH)"), Patch(color=BLUE, label="Situation-aware band")],
             frameon=False, fontsize=9, loc="upper right", ncol=2)
    b.set_title("In everyday situations: does '80% sure' really mean 80%?", loc="left", fontsize=10)

    fig.suptitle("Situation-aware uncertainty: an '80% band' that means 80% whether glucose is steady, rising, or falling",
                 x=0.01, ha="left", fontsize=12.5, weight="bold")
    fig.text(0.01, 0.01,
             "OhioT1DM (12 patients, data the method was never tuned on), plain CNN-LSTM, 60-minute forecasts, 80% bands. "
             "Trade-off: situation-aware bands catch fewer of the rare large forecast errors (Clarke zone D: 41% vs 62%); "
             "combining with GARCH recovers most of that.",
             fontsize=7.5, color=plotting.INK_SECONDARY)
    fig.tight_layout(rect=[0, 0.09, 1, 0.95])
    fig.savefig(WEEK / "figures" / "presentation_situation_aware.png", dpi=170)
    plt.close(fig)
    print("wrote presentation_situation_aware.png")


if __name__ == "__main__":
    main()
