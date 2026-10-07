"""Exploratory: PCA of a teacher-forcing encoder-decoder's bottleneck, coloured by patient variance.

Model: a GRU encoder reads 2 hours of history (glucose, basal, bolus, carbs: 24 steps) and
compresses it to an 8-number bottleneck. A GRU decoder predicts the next hour of glucose (12
steps). During training the decoder is fed the true previous glucose value at each step (teacher
forcing), plus the bottleneck. So the bottleneck only needs to carry what the true previous values
don't already say.

Data: AZT1D, all 25 patients. Each patient's first 64% of time trains the model (same split share
as the rest of the project). Inputs are scaled with one shared scale across patients, so
differences between patients are kept (per-patient scaling would erase them).

Plot: PCA of the bottleneck for test windows (last 20% of each patient), coloured by
  (a) the patient's glucose variability (SD of test-period glucose)
  (b) the plain forecasting model's error for that patient (RMSE, saved cnn_lstm_v0 checkpoint)
plus one point per patient (centroid). Also a check of how much the bottleneck matters: decoder
error with the real bottleneck vs with it set to zero.

Exploratory, not a hypothesis test. With 25 patients, patterns in the centroid panel are hints.

Run:    python work/5_10/scripts/teacher_forcing_autoencoder.py [epochs]   (default 8)
Writes: work/5_10/results/autoencoder_pca.csv, autoencoder_summary.txt,
        work/5_10/figures/autoencoder_pca.png
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
from torch import nn

ROOT = next(p for p in Path(__file__).resolve().parents if (p / "pyproject.toml").exists())
WEEK = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from azt1d import loading, plotting, reference as ref  # noqa: E402
from azt1d.glimmer import checkpoint as ckpt  # noqa: E402

FEATURES = [ref.CGM, ref.BASAL, ref.TOTAL_BOLUS_INSULIN_DELIVERED, ref.CARB_SIZE]
LOOKBACK, HORIZON, LATENT, HIDDEN = 24, 12, 8, 64
TRAIN_SHARE, TEST_SHARE = 0.64, 0.20
TRAIN_STRIDE, TEST_PER_PATIENT = 3, 400
SEED = 0


def windows(df_p: pd.DataFrame):
    f = df_p[FEATURES].to_numpy(np.float32)
    g = df_p[ref.CGM].to_numpy(np.float32)
    n = len(df_p) - LOOKBACK - HORIZON + 1
    idx = np.arange(n)
    X = np.stack([f[i:i + LOOKBACK] for i in idx])
    Y = np.stack([g[i + LOOKBACK:i + LOOKBACK + HORIZON] for i in idx])
    prev = g[idx + LOOKBACK - 1]  # last known glucose, the decoder's first teacher-forced input
    return X, Y, prev


class TFEncoderDecoder(nn.Module):
    def __init__(self, n_feat: int):
        super().__init__()
        self.enc = nn.GRU(n_feat, HIDDEN, batch_first=True)
        self.to_z = nn.Linear(HIDDEN, LATENT)
        self.dec = nn.GRU(1 + LATENT, HIDDEN, batch_first=True)
        self.init_h = nn.Linear(LATENT, HIDDEN)
        self.out = nn.Linear(HIDDEN, 1)

    def encode(self, x):
        _, h = self.enc(x)
        return self.to_z(h[-1])

    def forward(self, x, prev, y_true, zero_z: bool = False):
        z = self.encode(x)
        if zero_z:
            z = torch.zeros_like(z)
        dec_in = torch.cat([prev[:, None], y_true[:, :-1]], dim=1)[..., None]  # teacher forcing
        zz = z[:, None, :].expand(-1, HORIZON, -1)
        h0 = torch.tanh(self.init_h(z))[None]
        o, _ = self.dec(torch.cat([dec_in, zz], dim=-1), h0)
        return self.out(o).squeeze(-1), z


def main(epochs: int):
    t0 = time.time()
    torch.manual_seed(SEED)
    rng = np.random.default_rng(SEED)
    df = loading.load_real_dataset(ROOT / "data" / "raw", ROOT / "data" / "processed")
    tr_X, tr_Y, tr_P, te = [], [], [], []
    patient_rows = []
    for sid in sorted(int(s) for s in df.subject_id.unique()):
        d = df[df.subject_id == sid].reset_index(drop=True)
        X, Y, P = windows(d)
        n = len(X)
        cut_tr, cut_te = int(n * TRAIN_SHARE), int(n * (1 - TEST_SHARE))
        sel = np.arange(0, cut_tr, TRAIN_STRIDE)
        tr_X.append(X[sel]); tr_Y.append(Y[sel]); tr_P.append(P[sel])
        te_idx = np.arange(cut_te, n)
        pick = np.sort(rng.choice(te_idx, min(TEST_PER_PATIENT, len(te_idx)), replace=False))
        te.append((sid, X[pick], Y[pick], P[pick]))
        res = ckpt.load_result(ROOT / "data" / "processed" / "checkpoints" / "cnn_lstm_v0", sid)
        patient_rows.append({"subject_id": sid, "glucose_sd": float(np.std(Y[te_idx][:, 0])),
                             "glucose_mean": float(np.mean(Y[te_idx][:, 0])),
                             "forecast_rmse": float(res.rmse)})
    Xtr, Ytr, Ptr = np.concatenate(tr_X), np.concatenate(tr_Y), np.concatenate(tr_P)
    x_mean, x_std = Xtr.reshape(-1, len(FEATURES)).mean(0), Xtr.reshape(-1, len(FEATURES)).std(0) + 1e-6
    g_mean, g_std = float(x_mean[0]), float(x_std[0])
    sx = lambda a: torch.from_numpy(((a - x_mean) / x_std).astype(np.float32))  # noqa: E731
    sg = lambda a: torch.from_numpy(((a - g_mean) / g_std).astype(np.float32))  # noqa: E731
    Xt, Yt, Pt = sx(Xtr), sg(Ytr), sg(Ptr)
    print(f"training windows {len(Xt):,} from 25 patients ({time.time() - t0:.0f}s)", flush=True)

    model = TFEncoderDecoder(len(FEATURES))
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    for ep in range(epochs):
        perm = torch.randperm(len(Xt))
        tot = 0.0
        for b in range(0, len(Xt), 512):
            i = perm[b:b + 512]
            pred, _ = model(Xt[i], Pt[i], Yt[i])
            loss = ((pred - Yt[i]) ** 2).mean()
            opt.zero_grad(); loss.backward(); opt.step()
            tot += loss.item() * len(i)
        print(f"  epoch {ep + 1}/{epochs}: train MSE {tot / len(Xt):.4f} (scaled units) ({time.time() - t0:.0f}s)", flush=True)

    model.eval()
    zs, rows, err_real, err_zero, gnow = [], [], [], [], []
    with torch.no_grad():
        for sid, X, Y, P in te:
            x, y, p = sx(X), sg(Y), sg(P)
            pr, z = model(x, p, y)
            pz, _ = model(x, p, y, zero_z=True)
            err_real.append(((pr - y) ** 2).mean().item() * g_std ** 2)
            err_zero.append(((pz - y) ** 2).mean().item() * g_std ** 2)
            zs.append(z.numpy())
            gnow.append(X[:, -1, 0])
            rows.append(np.full(len(z), sid))
    Z, S = np.concatenate(zs), np.concatenate(rows)
    pca = PCA(n_components=2, random_state=SEED).fit(Z)
    pc = pca.transform(Z)
    pt = pd.DataFrame(patient_rows).set_index("subject_id")
    pts = pd.DataFrame({"subject_id": S, "pc1": pc[:, 0], "pc2": pc[:, 1]})
    pts["glucose_sd"] = pts.subject_id.map(pt.glucose_sd)
    pts["forecast_rmse"] = pts.subject_id.map(pt.forecast_rmse)
    pts.to_csv(WEEK / "results" / "autoencoder_pca.csv", index=False)
    cen = pts.groupby("subject_id")[["pc1", "pc2"]].mean().join(pt)

    rmse_real, rmse_zero = np.sqrt(np.mean(err_real)), np.sqrt(np.mean(err_zero))
    lines = [
        "Teacher-forcing encoder-decoder, PCA of the 8-number bottleneck (exploratory)",
        f"  PCA explained variance: PC1 {pca.explained_variance_ratio_[0]:.0%}, PC2 {pca.explained_variance_ratio_[1]:.0%}",
        f"  bottleneck check (teacher-forced next-hour error, test windows): real bottleneck RMSE {rmse_real:.1f} mg/dL, "
        f"bottleneck set to zero {rmse_zero:.1f} mg/dL",
    ]
    for col, name in (("glucose_sd", "glucose SD"), ("forecast_rmse", "plain-model forecast RMSE")):
        for pcn in ("pc1", "pc2"):
            r = spearmanr(cen[pcn], cen[col]).statistic
            lines.append(f"  patient centroids: Spearman {pcn.upper()} vs {name}: {r:+.2f} (25 patients)")
    lines.append(f"  window level: Spearman PC1 vs current glucose: {spearmanr(pc[:, 0], np.concatenate(gnow)).statistic:+.2f}")
    lines.append(f"  patient centroids: Spearman PC1 vs mean glucose: {spearmanr(cen.pc1, cen.glucose_mean).statistic:+.2f}")
    def loo_err(Xf, y):
        Xf, y = np.column_stack([np.ones(len(y)), Xf]), np.asarray(y, float)
        errs = []
        for k in range(len(y)):
            m = np.arange(len(y)) != k
            beta = np.linalg.lstsq(Xf[m], y[m], rcond=None)[0]
            errs.append((Xf[k] @ beta - y[k]) ** 2)
        return float(np.sqrt(np.mean(errs)))
    verdict = None
    for col, name in (("glucose_sd", "glucose SD"), ("forecast_rmse", "forecast RMSE")):
        e_lat = loo_err(cen[["pc1", "pc2"]].to_numpy(), cen[col])
        e_mean = loo_err(cen[["glucose_mean"]].to_numpy(), cen[col])
        e_both = loo_err(cen[["glucose_mean", "pc1", "pc2"]].to_numpy(), cen[col])
        lines.append(f"  leave-one-patient-out error predicting {name}: bottleneck position {e_lat:.2f}, mean glucose alone {e_mean:.2f}, "
                     f"both {e_both:.2f} mg/dL; bottleneck vs mean-glucose cut {1 - e_lat / e_mean:+.0%}")
        if col == "glucose_sd":
            verdict = "SUPPORTED" if e_lat <= 0.85 * e_mean else "NOT SUPPORTED"
    lines.insert(0, f"H39 (bottleneck predicts patient glucose SD at least 15% better than mean glucose, LOO): {verdict}")
    lines.append(f"  Spearman glucose SD vs forecast RMSE across patients: {spearmanr(cen.glucose_sd, cen.forecast_rmse).statistic:+.2f}")
    lines.append(f"  runtime {time.time() - t0:.0f}s, {epochs} epochs")
    text = "\n".join(lines)
    (WEEK / "results" / "autoencoder_summary.txt").write_text(text + "\n")
    print(text)

    plotting.apply_style()
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.8))
    ev = pca.explained_variance_ratio_
    for ax, col, title in ((axes[0], "glucose_sd", "Coloured by patient's glucose variability (SD, mg/dL)"),
                           (axes[1], "forecast_rmse", "Coloured by patient's forecast error (plain model RMSE, mg/dL)")):
        sc = ax.scatter(pts.pc1, pts.pc2, c=pts[col], cmap="viridis", s=4, alpha=0.5)
        fig.colorbar(sc, ax=ax, shrink=0.8)
        ax.set_title(title, loc="left", fontsize=10)
        ax.set_xlabel(f"PC1 ({ev[0]:.0%} of variance)")
        ax.set_ylabel(f"PC2 ({ev[1]:.0%} of variance)")
    ax = axes[2]
    sc = ax.scatter(cen.pc1, cen.pc2, c=cen.glucose_sd, cmap="viridis", s=90, edgecolor="white")
    for sid, r in cen.iterrows():
        ax.annotate(str(sid), (r.pc1, r.pc2), fontsize=8, xytext=(4, 3), textcoords="offset points")
    fig.colorbar(sc, ax=ax, shrink=0.8, label="glucose SD (mg/dL)")
    ax.set_title("One point per patient (average position), coloured by glucose SD", loc="left", fontsize=10)
    ax.set_xlabel("PC1"); ax.set_ylabel("PC2")
    fig.suptitle(f"Teacher-forcing encoder-decoder: what the 8-number summary of 2 hours of history looks like "
                 f"(AZT1D test windows, {TEST_PER_PATIENT} per patient)", x=0.01, ha="left", fontsize=11)
    fig.tight_layout()
    fig.savefig(WEEK / "figures" / "autoencoder_pca.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 8)
