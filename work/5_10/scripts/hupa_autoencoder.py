"""H40: the teacher-forcing encoder-decoder (same as H39) on HUPA-UCM, one point per patient,
coloured by the plain CNN-LSTM's forecast error for that patient.

Same model, inputs (glucose, basal, bolus, carbs), shared scaling and split shares as
teacher_forcing_autoencoder.py. Each patient's training windows are capped at 3,000 (after taking
every 3rd window) so patient 27, with 53% of HUPA's rows, does not dominate training.

Run:    python work/5_10/scripts/hupa_autoencoder.py
Writes: work/5_10/results/hupa_autoencoder_patients.csv, verdict_h40.txt,
        work/5_10/figures/hupa_autoencoder_pca.png
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr
from sklearn.decomposition import PCA

ROOT = next(p for p in Path(__file__).resolve().parents if (p / "pyproject.toml").exists())
WEEK = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from azt1d import hupa, plotting  # noqa: E402
from azt1d.glimmer import checkpoint as ckpt  # noqa: E402
from teacher_forcing_autoencoder import (TFEncoderDecoder, windows, FEATURES, TRAIN_SHARE, TEST_SHARE,  # noqa: E402
                                         TRAIN_STRIDE, TEST_PER_PATIENT, SEED)

EPOCHS, CAP = 8, 3000


def loo_err(Xf, y):
    Xf, y = np.column_stack([np.ones(len(y)), Xf]), np.asarray(y, float)
    errs = []
    for k in range(len(y)):
        m = np.arange(len(y)) != k
        beta = np.linalg.lstsq(Xf[m], y[m], rcond=None)[0]
        errs.append((Xf[k] @ beta - y[k]) ** 2)
    return float(np.sqrt(np.mean(errs)))


def main():
    t0 = time.time()
    torch.manual_seed(SEED)
    rng = np.random.default_rng(SEED)
    tr_X, tr_Y, tr_P, te, prow = [], [], [], [], []
    for sid in hupa.list_subject_ids(ROOT / hupa.DEFAULT_DIR):
        d = hupa.load_subject(sid, ROOT / hupa.DEFAULT_DIR)
        X, Y, P = windows(d)
        n = len(X)
        cut_tr, cut_te = int(n * TRAIN_SHARE), int(n * (1 - TEST_SHARE))
        sel = np.arange(0, cut_tr, TRAIN_STRIDE)
        if len(sel) > CAP:
            sel = np.sort(rng.choice(sel, CAP, replace=False))
        tr_X.append(X[sel]); tr_Y.append(Y[sel]); tr_P.append(P[sel])
        te_idx = np.arange(cut_te, n)
        pick = np.sort(rng.choice(te_idx, min(TEST_PER_PATIENT, len(te_idx)), replace=False))
        te.append((sid, X[pick], Y[pick], P[pick]))
        res = ckpt.load_result(ROOT / "data" / "processed" / "checkpoints" / "hupa_ucm_cnn_lstm_v0", sid)
        prow.append({"subject_id": sid, "forecast_rmse": float(res.rmse),
                     "glucose_mean": float(np.mean(Y[te_idx][:, 0])), "glucose_sd": float(np.std(Y[te_idx][:, 0])),
                     "days": len(d) * 5 / 1440})
    Xtr, Ytr, Ptr = np.concatenate(tr_X), np.concatenate(tr_Y), np.concatenate(tr_P)
    xm, xs = Xtr.reshape(-1, len(FEATURES)).mean(0), Xtr.reshape(-1, len(FEATURES)).std(0) + 1e-6
    gm, gs = float(xm[0]), float(xs[0])
    sx = lambda a: torch.from_numpy(((a - xm) / xs).astype(np.float32))  # noqa: E731
    sg = lambda a: torch.from_numpy(((a - gm) / gs).astype(np.float32))  # noqa: E731
    Xt, Yt, Pt = sx(Xtr), sg(Ytr), sg(Ptr)
    print(f"training windows {len(Xt):,} ({time.time() - t0:.0f}s)", flush=True)

    model = TFEncoderDecoder(len(FEATURES))
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    for ep in range(EPOCHS):
        perm = torch.randperm(len(Xt))
        for b in range(0, len(Xt), 512):
            i = perm[b:b + 512]
            pred, _ = model(Xt[i], Pt[i], Yt[i])
            loss = ((pred - Yt[i]) ** 2).mean()
            opt.zero_grad(); loss.backward(); opt.step()
    model.eval()
    zs, ids, er, ez = [], [], [], []
    with torch.no_grad():
        for sid, X, Y, P in te:
            x, y, p = sx(X), sg(Y), sg(P)
            pr, z = model(x, p, y)
            pz, _ = model(x, p, y, zero_z=True)
            er.append(((pr - y) ** 2).mean().item() * gs ** 2)
            ez.append(((pz - y) ** 2).mean().item() * gs ** 2)
            zs.append(z.numpy()); ids.append(np.full(len(z), sid))
    Z, S = np.concatenate(zs), np.concatenate(ids)
    pca = PCA(n_components=2, random_state=SEED).fit(Z)
    pc = pca.transform(Z)
    pts = pd.DataFrame({"subject_id": S, "pc1": pc[:, 0], "pc2": pc[:, 1]})
    pt = pd.DataFrame(prow).set_index("subject_id")
    cen = pts.groupby("subject_id")[["pc1", "pc2"]].mean().join(pt)
    cen.to_csv(WEEK / "results" / "hupa_autoencoder_patients.csv")

    e_lat = loo_err(cen[["pc1", "pc2"]].to_numpy(), cen.forecast_rmse)
    e_mean = loo_err(cen[["glucose_mean"]].to_numpy(), cen.forecast_rmse)
    e_sd = loo_err(cen[["glucose_sd"]].to_numpy(), cen.forecast_rmse)
    verdict = "SUPPORTED" if e_lat <= 0.85 * e_mean else "NOT SUPPORTED"
    text = "\n".join([
        f"H40 (HUPA bottleneck predicts patient CNN-LSTM error at least 15% better than mean glucose, LOO): {verdict}",
        f"  LOO error predicting forecast RMSE: bottleneck position {e_lat:.2f}, mean glucose {e_mean:.2f}, glucose SD {e_sd:.2f} mg/dL "
        f"(bottleneck vs mean glucose {1 - e_lat / e_mean:+.0%})",
        f"  PCA explained variance: PC1 {pca.explained_variance_ratio_[0]:.0%}, PC2 {pca.explained_variance_ratio_[1]:.0%}",
        f"  bottleneck check: decoder RMSE {np.sqrt(np.mean(er)):.1f} with bottleneck, {np.sqrt(np.mean(ez)):.1f} with it zeroed (mg/dL)",
        f"  patient centroids, Spearman with forecast RMSE: PC1 {spearmanr(cen.pc1, cen.forecast_rmse).statistic:+.2f}, "
        f"PC2 {spearmanr(cen.pc2, cen.forecast_rmse).statistic:+.2f}, mean glucose {spearmanr(cen.glucose_mean, cen.forecast_rmse).statistic:+.2f}, "
        f"glucose SD {spearmanr(cen.glucose_sd, cen.forecast_rmse).statistic:+.2f}",
        f"  PC1 vs mean glucose across patients: {spearmanr(cen.pc1, cen.glucose_mean).statistic:+.2f}",
        f"  forecast RMSE across patients: {cen.forecast_rmse.min():.1f} to {cen.forecast_rmse.max():.1f} mg/dL",
        f"  runtime {time.time() - t0:.0f}s",
    ])
    (WEEK / "results" / "verdict_h40.txt").write_text(text + "\n")
    print(text)

    plotting.apply_style()
    fig, (a, b) = plt.subplots(1, 2, figsize=(15, 6.2))
    ev = pca.explained_variance_ratio_
    pts["rmse"] = pts.subject_id.map(pt.forecast_rmse)
    sc = a.scatter(pts.pc1, pts.pc2, c=pts.rmse, cmap="viridis", s=4, alpha=0.5)
    fig.colorbar(sc, ax=a, shrink=0.8, label="patient's CNN-LSTM forecast error (RMSE, mg/dL)")
    a.set_title("Every test window, coloured by its patient's forecast error", loc="left", fontsize=10)
    a.set_xlabel(f"PC1 ({ev[0]:.0%} of variance)"); a.set_ylabel(f"PC2 ({ev[1]:.0%} of variance)")
    sc = b.scatter(cen.pc1, cen.pc2, c=cen.forecast_rmse, cmap="viridis", s=110, edgecolor="white")
    for sid, r in cen.iterrows():
        b.annotate(f"{sid}", (r.pc1, r.pc2), fontsize=8, xytext=(5, 3), textcoords="offset points")
    fig.colorbar(sc, ax=b, shrink=0.8, label="forecast error (RMSE, mg/dL)")
    b.set_title("One point per patient (average position), coloured by CNN-LSTM forecast error", loc="left", fontsize=10)
    b.set_xlabel("PC1"); b.set_ylabel("PC2")
    fig.suptitle("HUPA-UCM: teacher-forcing encoder-decoder summary of 2 hours of history, by patient forecast error",
                 x=0.01, ha="left", fontsize=11)
    fig.tight_layout()
    fig.savefig(WEEK / "figures" / "hupa_autoencoder_pca.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
