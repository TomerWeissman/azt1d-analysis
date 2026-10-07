"""Quick look: the teacher-forcing encoder-decoder retrained on HUPA-UCM with ALL recorded variables
as inputs (glucose, basal, bolus, carbs, heart rate, steps, calories), then a window-level t-SNE
coloured by each variable and by time of day.

Same architecture, splits, per-patient training cap (3,000), test windows (400 per patient), seed
and t-SNE settings as the 4-input version (tsne_bottleneck.py / tsne_hupa_variables.py). Also
reports the decoder's teacher-forced next-hour error with and without the bottleneck, for both the
4-input and 7-input models, to see whether the extra inputs help.

Run:    python work/5_10/scripts/tsne_hupa_all_inputs.py
Writes: work/5_10/results/hupa_all_inputs_summary.txt, work/5_10/figures/tsne_hupa_all_inputs.png
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from matplotlib.colors import LogNorm
from sklearn.manifold import TSNE

ROOT = next(p for p in Path(__file__).resolve().parents if (p / "pyproject.toml").exists())
WEEK = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from azt1d import hupa, plotting  # noqa: E402
from teacher_forcing_autoencoder import (TFEncoderDecoder, LOOKBACK, HORIZON, TRAIN_SHARE, TEST_SHARE,  # noqa: E402
                                         TRAIN_STRIDE, TEST_PER_PATIENT, SEED)
from tsne_hupa_variables import VARIABLES  # noqa: E402

ALL_INPUTS = ["glucose", "basal_rate", "bolus_volume_delivered", "carb_input", "heart_rate", "steps", "calories"]
FOUR_INPUTS = ALL_INPUTS[:4]
CAP = 3000


def raw_patient(sid):
    raw = pd.read_csv(hupa.subject_file(sid, ROOT / hupa.DEFAULT_DIR), sep=";", parse_dates=["time"])
    raw = raw.sort_values("time").reset_index(drop=True)
    raw[["bolus_volume_delivered", "carb_input"]] = raw[["bolus_volume_delivered", "carb_input"]].fillna(0)
    raw["hour"] = raw.time.dt.hour + raw.time.dt.minute / 60
    return raw


def windows(raw, cols):
    f = raw[cols].to_numpy(np.float32)
    g = raw["glucose"].to_numpy(np.float32)
    n = len(raw) - LOOKBACK - HORIZON + 1
    idx = np.arange(n)
    X = np.stack([f[i:i + LOOKBACK] for i in idx])
    Y = np.stack([g[i + LOOKBACK:i + LOOKBACK + HORIZON] for i in idx])
    return X, Y, g[idx + LOOKBACK - 1]


def run(cols, raws):
    torch.manual_seed(SEED)
    rng = np.random.default_rng(SEED)
    tr, te = ([], [], []), []
    for sid, raw in raws:
        X, Y, P = windows(raw, cols)
        n = len(X)
        cut_tr, cut_te = int(n * TRAIN_SHARE), int(n * (1 - TEST_SHARE))
        sel = np.arange(0, cut_tr, TRAIN_STRIDE)
        if len(sel) > CAP:
            sel = np.sort(rng.choice(sel, CAP, replace=False))
        for lst, arr in zip(tr, (X, Y, P)):
            lst.append(arr[sel])
        te_idx = np.arange(cut_te, n)
        pick = np.sort(rng.choice(te_idx, min(TEST_PER_PATIENT, len(te_idx)), replace=False))
        te.append((sid, pick, X[pick], Y[pick], P[pick]))
    Xtr, Ytr, Ptr = (np.concatenate(l) for l in tr)
    xm, xs = Xtr.reshape(-1, len(cols)).mean(0), Xtr.reshape(-1, len(cols)).std(0) + 1e-6
    gm, gs = float(xm[0]), float(xs[0])
    sx = lambda a: torch.from_numpy(((a - xm) / xs).astype(np.float32))  # noqa: E731
    sg = lambda a: torch.from_numpy(((a - gm) / gs).astype(np.float32))  # noqa: E731
    model = TFEncoderDecoder(len(cols))
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    Xt, Yt, Pt = sx(Xtr), sg(Ytr), sg(Ptr)
    for _ in range(8):
        perm = torch.randperm(len(Xt))
        for b in range(0, len(Xt), 512):
            i = perm[b:b + 512]
            pred, _ = model(Xt[i], Pt[i], Yt[i])
            loss = ((pred - Yt[i]) ** 2).mean()
            opt.zero_grad(); loss.backward(); opt.step()
    model.eval()
    Z, picks, er, ez = [], [], [], []
    with torch.no_grad():
        for sid, pick, X, Y, P in te:
            x, y, p = sx(X), sg(Y), sg(P)
            pr, z = model(x, p, y)
            pz, _ = model(x, p, y, zero_z=True)
            er.append(((pr - y) ** 2).mean().item() * gs ** 2)
            ez.append(((pz - y) ** 2).mean().item() * gs ** 2)
            Z.append(z.numpy()); picks.append((sid, pick))
    return np.concatenate(Z), picks, float(np.sqrt(np.mean(er))), float(np.sqrt(np.mean(ez)))


def window_values(raws, picks):
    by_sid = dict(raws)
    out = {v[0]: [] for v in VARIABLES}
    for sid, starts in picks:
        raw = by_sid[sid]
        for col, how, *_ in VARIABLES:
            a = raw[col].to_numpy(float)
            w = np.stack([a[s:s + LOOKBACK] for s in starts])
            out[col].append(w[:, -1] if how == "last" else (w.mean(1) if how == "mean" else w.sum(1)))
    return {k: np.concatenate(v) for k, v in out.items()}


def main():
    t0 = time.time()
    raws = [(sid, raw_patient(sid)) for sid in hupa.list_subject_ids(ROOT / hupa.DEFAULT_DIR)]
    Z7, picks, r7, z7 = run(ALL_INPUTS, raws)
    _, _, r4, z4 = run(FOUR_INPUTS, raws)
    print(f"both models trained ({time.time() - t0:.0f}s)", flush=True)
    text = "\n".join([
        "HUPA-UCM teacher-forcing encoder-decoder, 7 inputs vs 4 inputs (teacher-forced next-hour error, test windows)",
        f"  7 inputs: {r7:.1f} mg/dL with the bottleneck, {z7:.1f} with it zeroed",
        f"  4 inputs: {r4:.1f} mg/dL with the bottleneck, {z4:.1f} with it zeroed",
        f"  runtime {time.time() - t0:.0f}s (one seed; small differences may be noise)",
    ])
    (WEEK / "results" / "hupa_all_inputs_summary.txt").write_text(text + "\n")
    print(text)

    vals = window_values(raws, picks)
    T = TSNE(n_components=2, perplexity=30, init="pca", random_state=SEED).fit_transform(Z7)
    plotting.apply_style()
    fig, axes = plt.subplots(2, 4, figsize=(22, 11))
    for ax, (col, _, label, _, cmap, log) in zip(axes.ravel(), VARIABLES):
        v = vals[col]
        if log:
            zero = v <= 0
            ax.scatter(T[zero, 0], T[zero, 1], color="#d9d9d9", s=3, alpha=0.5, label="zero")
            sc = ax.scatter(T[~zero, 0], T[~zero, 1], c=v[~zero], cmap=cmap, s=4, alpha=0.8,
                            norm=LogNorm(vmin=max(v[~zero].min(), 1e-3), vmax=np.percentile(v[~zero], 99)))
            ax.legend(frameon=False, fontsize=7, loc="lower left", markerscale=3)
        else:
            lo, hi = np.percentile(v, [1, 99])
            sc = ax.scatter(T[:, 0], T[:, 1], c=v, cmap=cmap, s=3, alpha=0.6, vmin=lo, vmax=hi)
        fig.colorbar(sc, ax=ax, shrink=0.75)
        tag = "NOT an input" if col == "hour" else "encoder input"
        ax.set_title(f"{label}\n({tag})", loc="left", fontsize=9)
        ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle("HUPA-UCM: t-SNE of the encoder summary when ALL 7 recorded variables are inputs "
                 "(each dot = one 2-hour window)", x=0.01, ha="left", fontsize=11)
    fig.tight_layout()
    fig.savefig(WEEK / "figures" / "tsne_hupa_all_inputs.png", dpi=150)
    plt.close(fig)
    print(f"runtime {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
