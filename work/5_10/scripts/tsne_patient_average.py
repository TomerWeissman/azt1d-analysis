"""Quick look: t-SNE of each patient's AVERAGE bottleneck (one point per patient).

Uses the same shared encoder-decoders as tsne_bottleneck.py (one per dataset, same settings and
seed). Each patient's 400 test-window summaries (8 numbers each) are averaged into one 8-number
patient summary; t-SNE places the 25 patients, coloured by the plain CNN-LSTM's forecast error.

With 25 points t-SNE is unstable, so it is drawn with two seeds (perplexity 5). A real pattern
should show in both. Exploratory only.

Run:    python work/5_10/scripts/tsne_patient_average.py
Writes: work/5_10/figures/tsne_patient_average.png
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.manifold import TSNE

ROOT = next(p for p in Path(__file__).resolve().parents if (p / "pyproject.toml").exists())
WEEK = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from azt1d import plotting  # noqa: E402
from tsne_bottleneck import embed, DATASETS  # noqa: E402

SEEDS = (0, 1)


def main():
    t0 = time.time()
    plotting.apply_style()
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    for row, name in enumerate(DATASETS):
        Z, S, _, rmse = embed(name)
        ids = np.unique(S)
        avg = np.stack([Z[S == s].mean(0) for s in ids])
        err = np.array([rmse[s] for s in ids])
        for ax, seed in zip(axes[row], SEEDS):
            T = TSNE(n_components=2, perplexity=5, init="random", random_state=seed).fit_transform(avg)
            sc = ax.scatter(T[:, 0], T[:, 1], c=err, cmap="viridis", s=140, edgecolor="white")
            for (x, y), s in zip(T, ids):
                ax.annotate(str(s), (x, y), fontsize=8, xytext=(6, 4), textcoords="offset points")
            fig.colorbar(sc, ax=ax, shrink=0.8, label="CNN-LSTM forecast error (RMSE, mg/dL)")
            ax.set_title(f"{name}, t-SNE seed {seed}: one dot per patient (average of 400 windows)", loc="left", fontsize=10)
            ax.set_xticks([]); ax.set_yticks([])
        print(f"{name} done ({time.time() - t0:.0f}s)", flush=True)
    fig.suptitle("t-SNE of each patient's average 8-number summary, coloured by forecast error "
                 "(two seeds: a real pattern should appear in both; axes have no meaning)", x=0.01, ha="left", fontsize=11)
    fig.tight_layout()
    fig.savefig(WEEK / "figures" / "tsne_patient_average.png", dpi=150)
    plt.close(fig)
    print(f"runtime {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
