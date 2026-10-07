"""H43: size 80% bands by the encoder's 8-number summary (learned situation) instead of the
hand-drawn 25-situation grid.

Per dataset (AZT1D, OhioT1DM):
  1. Train the teacher-forcing encoder-decoder (same as teacher_forcing_autoencoder.py) on each
     patient's first 64% of windows. Validation and test windows are never seen by it.
  2. Summarize every validation and test window of the plain CNN-LSTM with the encoder. Windows
     line up one to one with the forecaster's (checked: the encoder window's 12th future value
     equals the forecaster's target).
  3. Bands, 80%, centred on the plain forecast, calibrated on validation (pooled over patients):
       GARCH (reference) | hand-drawn grid (Mondrian, 25 level x trend situations) |
       learned groups (Mondrian on 25 k-means groups of the summary) |
       learned neighbours (conformal on the 200 nearest validation windows in summary space)
  4. Scored on test: per-situation coverage error (hand-drawn situations), interval score, width,
     overall coverage, Clarke zone D catch rate.

Run:    python work/5_10/scripts/learned_situation_bands.py
Writes: work/5_10/results/h43_learned_situation.csv, verdict_h43.txt,
        work/5_10/figures/h43_learned_situation.png
"""
from __future__ import annotations

import math
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors

ROOT = next(p for p in Path(__file__).resolve().parents if (p / "pyproject.toml").exists())
WEEK = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from azt1d import loading, plotting, reference as ref  # noqa: E402
from azt1d.metabonet import load_source  # noqa: E402
from azt1d.glimmer import checkpoint as ckpt, uncertainty as unc  # noqa: E402
from azt1d.glimmer.clinical import clarke_error_grid_zones  # noqa: E402
from teacher_forcing_autoencoder import TFEncoderDecoder, windows, FEATURES, TRAIN_STRIDE, SEED  # noqa: E402
import predictability_map as pm  # noqa: E402
from situation_bands import conformal_halfwidth  # noqa: E402

TARGET, MIN_VAL, MIN_TEST, K_GROUPS, K_NN = 0.80, 100, 30, 25, 200
Z80 = 1.2815515655446004
DATASETS = {"AZT1D": "cnn_lstm_v0", "OhioT1DM": "ohiot1dm_cnn_lstm_v0"}


def load(name):
    if name == "AZT1D":
        return loading.load_real_dataset(ROOT / "data" / "raw", ROOT / "data" / "processed")
    return load_source("OhioT1DM")


def train_encoder(train_parts):
    Xtr, Ytr, Ptr = (np.concatenate(p) for p in zip(*train_parts))
    xm, xs = Xtr.reshape(-1, len(FEATURES)).mean(0), Xtr.reshape(-1, len(FEATURES)).std(0) + 1e-6
    gm, gs = float(xm[0]), float(xs[0])
    sx = lambda a: torch.from_numpy(((a - xm) / xs).astype(np.float32))  # noqa: E731
    sg = lambda a: torch.from_numpy(((a - gm) / gs).astype(np.float32))  # noqa: E731
    torch.manual_seed(SEED)
    model = TFEncoderDecoder(len(FEATURES))
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

    def encode(X):
        with torch.no_grad():
            return model.encode(sx(X)).numpy()
    return encode


def build(name, run):
    df = load(name)
    per = []
    train_parts = []
    for sid in sorted(int(s) for s in df.subject_id.unique()):
        d = df[df.subject_id == sid].reset_index(drop=True)
        res = ckpt.load_result(ROOT / "data" / "processed" / "checkpoints" / run, sid)
        val, test = unc.forecast_frames(res, d)
        X, Y, P = windows(d)
        n_train = len(X) - len(val.actual) - len(test.actual)
        assert np.allclose(Y[n_train:n_train + len(val.actual), -1], val.actual), "window alignment (val)"
        assert np.allclose(Y[n_train + len(val.actual):, -1], test.actual), "window alignment (test)"
        sel = np.arange(0, n_train, TRAIN_STRIDE)
        train_parts.append((X[sel], Y[sel], P[sel]))
        eng = unc.BandEngine(val, test, analogs=False)
        per.append((sid, d, val, test, X, n_train, eng))
    encode = train_encoder(train_parts)
    rows = {"val": [], "test": []}
    for sid, d, val, test, X, n_train, eng in per:
        times = pd.to_datetime(d[ref.EVENT_DATETIME]).to_numpy()
        g = d[ref.CGM].to_numpy(float)
        for split, fr, lo_i in (("val", val, n_train), ("test", test, n_train + len(val.actual))):
            i = np.searchsorted(times, fr.issue_time)
            z = encode(X[lo_i:lo_i + len(fr.actual)])
            out = pd.DataFrame({"subject_id": sid, "actual": fr.actual, "pred": fr.pred, "resid": fr.resid,
                                "g_now": g[i], "g_prev": g[np.maximum(i - pm.TREND_STEPS, 0)], "keep": i >= pm.TREND_STEPS})
            out["Z"] = list(z)
            if split == "test":
                out["sigma"], out["mu"] = eng.garch_sigma, eng.garch_params.mu
            rows[split].append(out[out.keep].reset_index(drop=True))
    v, t = pd.concat(rows["val"], ignore_index=True), pd.concat(rows["test"], ignore_index=True)
    for f in (v, t):
        f["cell"] = np.digitize(f.g_now.to_numpy(), pm.LEVEL_EDGES) * 5 + np.digitize((f.g_now - f.g_prev).to_numpy(), pm.RATE_EDGES)
    return v, t


def group_halfwidth(v_err, v_groups, t_groups, n_groups):
    pooled = conformal_halfwidth(v_err, TARGET)
    q = np.full(n_groups, pooled)
    for c in range(n_groups):
        m = v_groups == c
        if m.sum() >= MIN_VAL:
            q[c] = conformal_halfwidth(v_err[m], TARGET)
    return q[t_groups]


def knn_halfwidth(Zv, err, Zt):
    mu, sd = Zv.mean(0), Zv.std(0) + 1e-9
    nn = NearestNeighbors(n_neighbors=K_NN).fit((Zv - mu) / sd)
    _, idx = nn.kneighbors((Zt - mu) / sd)
    srt = np.sort(err[idx], axis=1)
    k = math.ceil((K_NN + 1) * TARGET)
    return srt[:, min(k, K_NN) - 1]


def score(t, lo, hi):
    y = t.actual.to_numpy()
    lo, hi = np.clip(lo, unc.SENSOR_MIN, unc.SENSOR_MAX), np.clip(hi, unc.SENSOR_MIN, unc.SENSOR_MAX)
    ins = (y >= lo) & (y <= hi)
    cells = pd.DataFrame({"cell": t.cell.to_numpy(), "in": ins}).groupby("cell").filter(lambda x: len(x) >= MIN_TEST)
    cov_err = float((cells.groupby("cell")["in"].mean() - TARGET).abs().mean()) * 100
    a = 1 - TARGET
    iscore = float(np.mean((hi - lo) + (2 / a) * np.maximum(lo - y, 0) + (2 / a) * np.maximum(y - hi, 0)))
    zone = clarke_error_grid_zones(y, t.pred.to_numpy())
    return {"coverage_error": cov_err, "interval_score": iscore, "mean_width": float(np.mean(hi - lo)),
            "overall_coverage": float(ins.mean()), "zone_d_caught": float(ins[zone == "D"].mean()),
            "zone_a_caught": float(ins[zone == "A"].mean()), "n_zone_d": int((zone == "D").sum())}


def main():
    t0 = time.time()
    rows = []
    for name, run in DATASETS.items():
        v, t = build(name, run)
        print(f"{name}: encoder trained, {len(v):,} validation and {len(t):,} test windows ({time.time() - t0:.0f}s)", flush=True)
        p, err = t.pred.to_numpy(), np.abs(v.resid.to_numpy())
        Zv, Zt = np.stack(v.Z.to_numpy()), np.stack(t.Z.to_numpy())
        c = (t.pred + t.mu).to_numpy()
        methods = {"GARCH": (c - Z80 * t.sigma.to_numpy(), c + Z80 * t.sigma.to_numpy())}
        q = group_halfwidth(err, v.cell.to_numpy(), t.cell.to_numpy(), 25)
        methods["hand-drawn grid"] = (p - q, p + q)
        km = KMeans(K_GROUPS, n_init=10, random_state=SEED).fit(Zv)
        q = group_halfwidth(err, km.labels_, km.predict(Zt), K_GROUPS)
        methods["learned groups (k-means)"] = (p - q, p + q)
        q = knn_halfwidth(Zv, err, Zt)
        methods["learned neighbours (kNN)"] = (p - q, p + q)
        for m, (lo, hi) in methods.items():
            rows.append({"dataset": name, "method": m, **score(t, lo, hi)})
        print(f"{name}: scored ({time.time() - t0:.0f}s)", flush=True)
    res = pd.DataFrame(rows)
    res.to_csv(WEEK / "results" / "h43_learned_situation.csv", index=False)

    lines, ok_any = [], False
    for m in ("learned groups (k-means)", "learned neighbours (kNN)"):
        ok_both = True
        for name in DATASETS:
            d = res[res.dataset == name].set_index("method")
            h, l_ = d.loc["hand-drawn grid"], d.loc[m]
            ok = (l_.interval_score < h.interval_score and l_.zone_d_caught >= h.zone_d_caught + 0.05
                  and l_.coverage_error <= h.coverage_error + 1.0)
            ok_both &= ok
            lines.append(f"  {name}, {m}: interval score {l_.interval_score:.0f} vs grid {h.interval_score:.0f}; zone D caught "
                         f"{l_.zone_d_caught:.0%} vs {h.zone_d_caught:.0%}; per-situation error {l_.coverage_error:.1f} vs "
                         f"{h.coverage_error:.1f} pts -> {'meets' if ok else 'fails'}")
        ok_any |= ok_both
    lines.insert(0, f"H43 (a learned version beats the hand-drawn grid on interval score and zone D, both datasets): "
                    f"{'SUPPORTED' if ok_any else 'NOT SUPPORTED'}")
    for name in DATASETS:
        lines.append(f"  {name} full table:")
        for r in res[res.dataset == name].itertuples():
            lines.append(f"    {r.method}: per-situation error {r.coverage_error:.1f} pts, interval score {r.interval_score:.0f}, "
                         f"width {r.mean_width:.0f}, overall {r.overall_coverage:.1%}, zone D caught {r.zone_d_caught:.0%} "
                         f"(n={r.n_zone_d}), zone A caught {r.zone_a_caught:.0%}")
    lines.append(f"  runtime {time.time() - t0:.0f}s")
    text = "\n".join(lines)
    (WEEK / "results" / "verdict_h43.txt").write_text(text + "\n")
    print(text)

    plotting.apply_style()
    C = plotting.CATEGORICAL
    cols = {"GARCH": plotting.INK_MUTED, "hand-drawn grid": C[0], "learned groups (k-means)": C[3], "learned neighbours (kNN)": C[2]}
    metrics = [("coverage_error", "Per-situation distance from 80% (points)\nlower is better", "{:.1f}"),
               ("interval_score", "Interval score (width + miss penalty)\nlower is better", "{:.0f}"),
               ("zone_d_caught", "Dangerous misses caught (zone D)\nhigher is better", "{:.0%}")]
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    for row, name in enumerate(DATASETS):
        d = res[res.dataset == name]
        for ax, (col, label, fmt) in zip(axes[row], metrics):
            vals = d[col].to_numpy()
            ax.bar(range(len(d)), vals, color=[cols[m] for m in d.method])
            for i, val in enumerate(vals):
                ax.text(i, val * 1.01, fmt.format(val), ha="center", fontsize=8)
            ax.set_xticks(range(len(d)), [m.replace(" (", "\n(") for m in d.method], fontsize=7.5)
            ax.set_ylabel(label, fontsize=9)
            ax.set_ylim(0, vals.max() * 1.18)
            ax.set_title(name, loc="left", fontsize=10)
    fig.suptitle("Hand-drawn situations vs situations learned by the encoder (80% bands, plain CNN-LSTM)", x=0.01, ha="left", fontsize=11)
    fig.tight_layout()
    fig.savefig(WEEK / "figures" / "h43_learned_situation.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
