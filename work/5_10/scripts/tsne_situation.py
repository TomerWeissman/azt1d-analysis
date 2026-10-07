"""Quick look: the window-level t-SNE of the teacher-forcing bottleneck, coloured by situation.

Same windows, model and t-SNE settings as tsne_bottleneck.py (so the layout matches). Each
window's situation uses the same grid as the situation-band study (predictability_map.py):
current glucose level (5 bands) and change over the last 30 minutes (5 bands).

Run:    python work/5_10/scripts/tsne_situation.py
Writes: work/5_10/figures/tsne_situation.png
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch
from sklearn.manifold import TSNE

ROOT = next(p for p in Path(__file__).resolve().parents if (p / "pyproject.toml").exists())
WEEK = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from azt1d import plotting  # noqa: E402
from tsne_bottleneck import embed, DATASETS, SEED  # noqa: E402
import predictability_map as pm  # noqa: E402

LEVEL_COLORS = ["#d03b3b", "#2a78d6", "#1baf7a", "#eda100", "#4a3aa7"]  # under 70 ... 250+
TREND_COLORS = ["#08306b", "#6baed6", "#bdbdbd", "#fc9272", "#a50f15"]  # fast fall ... fast rise


def main():
    t0 = time.time()
    plotting.apply_style()
    fig, axes = plt.subplots(2, 3, figsize=(19, 12))
    for row, name in enumerate(DATASETS):
        Z, S, G, _, G6 = embed(name, return_prev=True)
        T = TSNE(n_components=2, perplexity=30, init="pca", random_state=SEED).fit_transform(Z)
        lvl = np.digitize(G, pm.LEVEL_EDGES)
        trd = np.digitize(G - G6, pm.RATE_EDGES)
        cell = lvl * 5 + trd
        print(f"{name}: done ({time.time() - t0:.0f}s)", flush=True)

        for ax, codes, colors, names, title in (
                (axes[row, 0], lvl, LEVEL_COLORS, pm.LEVEL_NAMES, "current level"),
                (axes[row, 1], trd, TREND_COLORS, pm.RATE_NAMES, "30-minute trend")):
            ax.scatter(T[:, 0], T[:, 1], c=codes, cmap=ListedColormap(colors), vmin=-0.5, vmax=4.5, s=3, alpha=0.6)
            ax.legend(handles=[Patch(color=c, label=n) for c, n in zip(colors, names)], frameon=False, fontsize=8,
                      loc="lower left", markerscale=2)
            ax.set_title(f"{name}: coloured by {title}", loc="left", fontsize=10)
            ax.set_xticks([]); ax.set_yticks([])

        ax = axes[row, 2]
        # combined situation: hue from level, lightness from trend
        base = np.array([plotting_rgb(c) for c in LEVEL_COLORS])
        shade = np.linspace(0.45, 1.25, 5)
        rgb = np.clip(base[lvl] * shade[trd][:, None], 0, 1)
        ax.scatter(T[:, 0], T[:, 1], c=rgb, s=3, alpha=0.7)
        counts = np.bincount(cell, minlength=25)
        for c in np.argsort(-counts)[:10]:
            m = cell == c
            x, y = np.median(T[m], axis=0)
            ax.text(x, y, f"{pm.LEVEL_NAMES[c // 5]}\n{pm.RATE_NAMES[c % 5]}", fontsize=6.5, ha="center",
                    weight="bold", bbox=dict(facecolor="white", alpha=0.6, lw=0, pad=1))
        ax.set_title(f"{name}: full situation (colour = level, dark to light = falling to rising);\n"
                     "labels at the middle of the 10 most common situations", loc="left", fontsize=9)
        ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle("t-SNE of the 8-number bottleneck, coloured by glucose situation (each dot = one 2-hour window)",
                 x=0.01, ha="left", fontsize=11)
    fig.tight_layout()
    fig.savefig(WEEK / "figures" / "tsne_situation.png", dpi=150)
    plt.close(fig)
    print(f"runtime {time.time() - t0:.0f}s")


def plotting_rgb(hex_color):
    h = hex_color.lstrip("#")
    return np.array([int(h[i:i + 2], 16) / 255 for i in (0, 2, 4)])


if __name__ == "__main__":
    main()
