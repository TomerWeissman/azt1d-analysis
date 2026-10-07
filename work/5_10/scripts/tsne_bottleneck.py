"""Quick look: t-SNE instead of PCA for the teacher-forcing encoder-decoder's 8-number bottleneck.

Retrains the same model as teacher_forcing_autoencoder.py (AZT1D) and hupa_autoencoder.py
(HUPA-UCM), with the same settings and seed, encodes 400 test windows per patient, and draws a
t-SNE map three ways: coloured by the patient's plain CNN-LSTM forecast error, by the window's
current glucose, and by patient.

t-SNE keeps neighbours together but distorts distances between groups, and its axes have no
meaning. Exploratory only; not a hypothesis test.

Run:    python work/5_10/scripts/tsne_bottleneck.py
Writes: work/5_10/figures/tsne_bottleneck.png
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.manifold import TSNE

ROOT = next(p for p in Path(__file__).resolve().parents if (p / "pyproject.toml").exists())
WEEK = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from azt1d import hupa, loading, plotting  # noqa: E402
from azt1d.glimmer import checkpoint as ckpt  # noqa: E402
from teacher_forcing_autoencoder import (TFEncoderDecoder, windows, FEATURES, TRAIN_SHARE, TEST_SHARE,  # noqa: E402
                                         TRAIN_STRIDE, TEST_PER_PATIENT, SEED)

DATASETS = {
    "AZT1D": ("cnn_lstm_v0", None),
    "HUPA-UCM": ("hupa_ucm_cnn_lstm_v0", 3000),
}


def patients(name):
    if name == "AZT1D":
        df = loading.load_real_dataset(ROOT / "data" / "raw", ROOT / "data" / "processed")
        for sid in sorted(int(s) for s in df.subject_id.unique()):
            yield sid, df[df.subject_id == sid].reset_index(drop=True)
    else:
        for sid in hupa.list_subject_ids(ROOT / hupa.DEFAULT_DIR):
            yield sid, hupa.load_subject(sid, ROOT / hupa.DEFAULT_DIR)


def embed(name, return_prev: bool = False, return_picks: bool = False):
    """Encode test windows. With return_prev=True also returns each window's glucose 30 minutes
    (6 steps) before its last reading, for the situation trend. With return_picks=True also
    returns [(subject_id, window start indices)] in encoding order."""
    run, cap = DATASETS[name]
    torch.manual_seed(SEED)
    rng = np.random.default_rng(SEED)
    tr, te, rmse, picks = ([], [], []), [], {}, []
    for sid, d in patients(name):
        X, Y, P = windows(d)
        n = len(X)
        cut_tr, cut_te = int(n * TRAIN_SHARE), int(n * (1 - TEST_SHARE))
        sel = np.arange(0, cut_tr, TRAIN_STRIDE)
        if cap and len(sel) > cap:
            sel = np.sort(rng.choice(sel, cap, replace=False))
        for lst, arr in zip(tr, (X, Y, P)):
            lst.append(arr[sel])
        te_idx = np.arange(cut_te, n)
        pick = np.sort(rng.choice(te_idx, min(TEST_PER_PATIENT, len(te_idx)), replace=False))
        te.append((sid, X[pick], Y[pick], P[pick]))
        picks.append((sid, pick))
        rmse[sid] = float(ckpt.load_result(ROOT / "data" / "processed" / "checkpoints" / run, sid).rmse)
    Xtr, Ytr, Ptr = (np.concatenate(l) for l in tr)
    xm, xs = Xtr.reshape(-1, len(FEATURES)).mean(0), Xtr.reshape(-1, len(FEATURES)).std(0) + 1e-6
    gm, gs = float(xm[0]), float(xs[0])
    sx = lambda a: torch.from_numpy(((a - xm) / xs).astype(np.float32))  # noqa: E731
    sg = lambda a: torch.from_numpy(((a - gm) / gs).astype(np.float32))  # noqa: E731
    Xt, Yt, Pt = sx(Xtr), sg(Ytr), sg(Ptr)
    model = TFEncoderDecoder(len(FEATURES))
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    for _ in range(8):
        perm = torch.randperm(len(Xt))
        for b in range(0, len(Xt), 512):
            i = perm[b:b + 512]
            pred, _ = model(Xt[i], Pt[i], Yt[i])
            loss = ((pred - Yt[i]) ** 2).mean()
            opt.zero_grad(); loss.backward(); opt.step()
    model.eval()
    Z, S, G, G6 = [], [], [], []
    with torch.no_grad():
        for sid, X, Y, P in te:
            _, z = model(sx(X), sg(P), sg(Y))
            Z.append(z.numpy()); S.append(np.full(len(z), sid)); G.append(X[:, -1, 0]); G6.append(X[:, -7, 0])
    out = (np.concatenate(Z), np.concatenate(S), np.concatenate(G), rmse)
    if return_prev:
        out = out + (np.concatenate(G6),)
    if return_picks:
        out = out + (picks,)
    return out


def main():
    t0 = time.time()
    plotting.apply_style()
    fig, axes = plt.subplots(2, 3, figsize=(18, 11.5))
    for row, name in enumerate(DATASETS):
        Z, S, G, rmse = embed(name)
        print(f"{name}: encoded {len(Z):,} windows ({time.time() - t0:.0f}s)", flush=True)
        T = TSNE(n_components=2, perplexity=30, init="pca", random_state=SEED).fit_transform(Z)
        print(f"{name}: t-SNE done ({time.time() - t0:.0f}s)", flush=True)
        err = np.array([rmse[s] for s in S])
        for ax, (vals, cmap, label) in zip(axes[row], (
                (err, "viridis", "patient's CNN-LSTM forecast error (RMSE, mg/dL)"),
                (G, "magma", "current glucose (mg/dL)"),
                (S, "tab20", "patient"))):
            sc = ax.scatter(T[:, 0], T[:, 1], c=vals, cmap=cmap, s=3, alpha=0.6)
            if label == "patient":
                for sid in np.unique(S):
                    c = T[S == sid].mean(0)
                    ax.text(c[0], c[1], str(sid), fontsize=7, weight="bold")
            else:
                fig.colorbar(sc, ax=ax, shrink=0.8, label=label)
            ax.set_title(f"{name}: coloured by {label}", loc="left", fontsize=10)
            ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle("t-SNE of the 8-number bottleneck (each dot = one 2-hour window; axes and distances between groups have no meaning)",
                 x=0.01, ha="left", fontsize=11)
    fig.tight_layout()
    fig.savefig(WEEK / "figures" / "tsne_bottleneck.png", dpi=150)
    plt.close(fig)
    print(f"runtime {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
