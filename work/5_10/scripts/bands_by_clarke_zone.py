"""Band width per Clarke zone: where on the Clarke grid are the bands wide or narrow?

Plain model (cnn_lstm_v0 / ohiot1dm_cnn_lstm_v0). Every test point is placed on the Clarke grid
(actual vs forecast) and coloured by the width of its 80% situation (Mondrian) band. Per Clarke
zone: number of points, average width, and share caught, for both the situation band and GARCH.

Descriptive, not a hypothesis test.

Run:    python work/5_10/scripts/bands_by_clarke_zone.py
Writes: work/5_10/results/bands_by_clarke_zone.csv, work/5_10/figures/bands_by_clarke_zone.png
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

from azt1d import plotting  # noqa: E402
from azt1d.glimmer.clinical import clarke_error_grid_zones, draw_clarke_grid  # noqa: E402
from verification_checks import frames, mondrian, garch, clip  # noqa: E402

RUNS = {"ohiot1dm": ("ohiot1dm_cnn_lstm_v0", "OhioT1DM"), "azt1d": ("cnn_lstm_v0", "AZT1D")}
ZONES = ["A", "B", "C", "D", "E"]


def main():
    rows, data = [], {}
    for ds, (run, label) in RUNS.items():
        v, t = frames(ds, run)
        y, p = t.actual.to_numpy(), t.pred.to_numpy()
        lo_m, hi_m = (clip(a) for a in mondrian(v, t))
        lo_g, hi_g = (clip(a) for a in garch(t))
        zone = clarke_error_grid_zones(y, p)
        w_m, w_g = hi_m - lo_m, hi_g - lo_g
        in_m, in_g = (y >= lo_m) & (y <= hi_m), (y >= lo_g) & (y <= hi_g)
        for z in ZONES:
            m = zone == z
            if m.sum() == 0:
                continue
            rows.append({"dataset": label, "zone": z, "n": int(m.sum()), "share_of_points": float(m.mean()),
                         "width_situation": float(w_m[m].mean()), "width_garch": float(w_g[m].mean()),
                         "caught_situation": float(in_m[m].mean()), "caught_garch": float(in_g[m].mean())})
        data[ds] = (y, p, w_m, label)
    res = pd.DataFrame(rows)
    res.to_csv(WEEK / "results" / "bands_by_clarke_zone.csv", index=False)
    print(res.round(3).to_string(index=False))

    plotting.apply_style()
    C = plotting.CATEGORICAL
    fig, axes = plt.subplots(2, 2, figsize=(14, 13), gridspec_kw={"width_ratios": [1, 1.15]})
    for row, ds in enumerate(RUNS):
        y, p, w, label = data[ds]
        ax = axes[row, 0]
        draw_clarke_grid(ax, np.array([]), np.array([]), "black")
        order = np.argsort(w)
        sc = ax.scatter(y[order], p[order], c=w[order], cmap="viridis", s=4, alpha=0.6, vmin=40, vmax=200)
        for z, (x, yy) in {"A": (30, 15), "B": (360, 100), "C": (100, 350), "D": (30, 120), "E": (40, 360)}.items():
            ax.text(x, yy, z, fontsize=14, weight="bold", color=plotting.INK_SECONDARY)
        fig.colorbar(sc, ax=ax, shrink=0.75, label="80% situation band width (mg/dL)")
        ax.set_title(f"{label}: each point = one forecast, coloured by its band width", loc="left", fontsize=10)

        ax = axes[row, 1]
        d = res[res.dataset == label]
        x = np.arange(len(d)); bw = 0.38
        ax.bar(x - bw / 2, d.width_garch, bw, color=plotting.INK_MUTED, label="GARCH (current)")
        ax.bar(x + bw / 2, d.width_situation, bw, color=C[0], label="situation band (new)")
        for xi, r in zip(x, d.itertuples()):
            top = max(r.width_garch, r.width_situation)
            ax.text(xi, top + 3, f"n={r.n:,} ({r.share_of_points:.1%})\ncaught: GARCH {r.caught_garch:.0%}, new {r.caught_situation:.0%}",
                    ha="center", fontsize=7.5)
        ax.set_xticks(x, [f"Zone {z}" for z in d.zone])
        ax.set_ylabel("Average 80% band width (mg/dL)")
        ax.set_ylim(0, max(d.width_garch.max(), d.width_situation.max()) * 1.35)
        ax.set_title(f"{label}: average band width and share caught, by Clarke zone", loc="left", fontsize=10)
        ax.legend(frameon=False, fontsize=8, loc="upper left")
    fig.suptitle("Where on the Clarke grid are the bands wide or narrow? (plain model, 80% bands)",
                 x=0.01, ha="left", fontsize=12)
    fig.tight_layout()
    fig.savefig(WEEK / "figures" / "bands_by_clarke_zone.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
