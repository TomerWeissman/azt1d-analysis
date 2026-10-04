"""H36: situation-grouped (Mondrian) bands vs a well-tuned CQR with the full input window.

CQR (conformalized quantile regression, Romano et al. 2019):
  - target: the plain model's forecast error (actual - forecast)
  - features: the full scaled input window the forecaster saw (24 steps x 6 inputs = 144),
    plus the forecast and the GARCH error size
  - model: gradient-boosted quantile regression for the 10% and 90% quantiles
  - tuning: 4 settings, chosen by pinball loss on the last 20% of the CQR training part
  - data: each patient's validation period split by time, first 70% to train, last 30% to
    conformalize. Test period only scored.

Mondrian bands are calibrated on the full validation period, as in H28.

Run:    python work/5_10/scripts/strong_cqr.py
Writes: work/5_10/results/h36_strong_cqr.csv, verdict_h36.txt, work/5_10/figures/h36_strong_cqr.png
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

ROOT = next(p for p in Path(__file__).resolve().parents if (p / "pyproject.toml").exists())
WEEK = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from azt1d import plotting, reference as ref  # noqa: E402
from azt1d.glimmer import checkpoint as ckpt, uncertainty as unc  # noqa: E402
import hybrid_bands as hb  # noqa: E402
import predictability_map as pm  # noqa: E402
from verification_checks import cov_err, inside, mondrian, garch, clip, TARGET  # noqa: E402

RUNS = {"ohiot1dm": "ohiot1dm_cnn_lstm_v0", "azt1d": "cnn_lstm_v0"}
GRID = [dict(learning_rate=lr, max_leaf_nodes=ln) for lr in (0.05, 0.1) for ln in (15, 31)]
H36_CUT = 0.10


def build(ds):
    df = hb.load(ds)
    vals, tests = [], []
    for sid in sorted(int(s) for s in df["subject_id"].unique()):
        res = ckpt.load_result(ROOT / "data" / "processed" / "checkpoints" / RUNS[ds], sid)
        dfs = df[df["subject_id"] == sid].reset_index(drop=True)
        val, test = unc.forecast_frames(res, dfs)
        eng = unc.BandEngine(val, test, analogs=False)
        p = eng.garch_params
        s_val = np.sqrt(unc.garch_h_step_variance(val.resid, p, float(np.var(val.resid - p.mu)), unc.LAG))
        times = pd.to_datetime(dfs[ref.EVENT_DATETIME]).to_numpy()
        g = dfs[ref.CGM].to_numpy()
        for fr, sig, bucket in ((val, s_val, vals), (test, eng.garch_sigma, tests)):
            i = np.searchsorted(times, fr.issue_time)
            keep = (i >= pm.TREND_STEPS) & np.isfinite(sig)
            d = pd.DataFrame({"subject_id": sid, "actual": fr.actual, "pred": fr.pred, "resid": fr.resid,
                              "mu": p.mu, "sigma": sig, "g_now": g[i], "g_prev": g[np.maximum(i - pm.TREND_STEPS, 0)]})[keep]
            d = d.reset_index(drop=True)
            d["X"] = list(fr.X.reshape(len(fr.X), -1)[keep].astype(np.float32))
            bucket.append(d)
    v, t = pd.concat(vals, ignore_index=True), pd.concat(tests, ignore_index=True)
    for f in (v, t):
        f["cell"] = np.digitize(f.g_now.to_numpy(), pm.LEVEL_EDGES) * 5 + np.digitize((f.g_now - f.g_prev).to_numpy(), pm.RATE_EDGES)
    return v, t


def feats(d):
    return np.column_stack([np.stack(d.X.to_numpy()), d.pred.to_numpy(), d.sigma.to_numpy()])


def pinball(y, q, a):
    e = y - q
    return float(np.mean(np.maximum(a * e, (a - 1) * e)))


def split_by_time(d, frac):
    pos = d.groupby("subject_id").cumcount() / d.groupby("subject_id").subject_id.transform("size")
    return d[pos < frac], d[pos >= frac]


def strong_cqr(v, t):
    train, cal = split_by_time(v, 0.70)
    fit_part, tune_part = split_by_time(train, 0.80)
    Xf, Xt = feats(fit_part), feats(tune_part)
    chosen = {}
    for a in (0.10, 0.90):
        scores = []
        for cfg in GRID:
            m = HistGradientBoostingRegressor(loss="quantile", quantile=a, max_iter=400, early_stopping=True,
                                              random_state=0, **cfg).fit(Xf, fit_part.resid.to_numpy())
            scores.append(pinball(tune_part.resid.to_numpy(), m.predict(Xt), a))
        chosen[a] = GRID[int(np.argmin(scores))]
    Xtr = feats(train)
    models = {a: HistGradientBoostingRegressor(loss="quantile", quantile=a, max_iter=400, early_stopping=True,
                                               random_state=0, **chosen[a]).fit(Xtr, train.resid.to_numpy())
              for a in (0.10, 0.90)}
    Xc, Xte = feats(cal), feats(t)
    lo_c, hi_c = models[0.10].predict(Xc), models[0.90].predict(Xc)
    score = np.maximum(lo_c - cal.resid.to_numpy(), cal.resid.to_numpy() - hi_c)
    n = len(score)
    q = float(np.sort(score)[min(int(np.ceil((n + 1) * TARGET)), n) - 1])
    p = t.pred.to_numpy()
    return p + models[0.10].predict(Xte) - q, p + models[0.90].predict(Xte) + q, chosen


def main():
    t0 = time.time()
    out, figs = WEEK / "results", WEEK / "figures"
    rows, boot_info = [], {}
    rng = np.random.default_rng(0)
    for ds in RUNS:
        v, t = build(ds)
        lo_c, hi_c, chosen = strong_cqr(v, t)
        lo_m, hi_m = mondrian(v, t)
        lo_g, hi_g = garch(t)
        ins = {"GARCH": inside(t, lo_g, hi_g), "CQR (tuned, full window)": inside(t, lo_c, hi_c),
               "Mondrian (situation)": inside(t, lo_m, hi_m)}
        widths = {"GARCH": clip(hi_g) - clip(lo_g), "CQR (tuned, full window)": clip(hi_c) - clip(lo_c),
                  "Mondrian (situation)": clip(hi_m) - clip(lo_m)}
        for name in ins:
            rows.append({"dataset": ds, "method": name, "coverage_error": cov_err(t, ins[name]) * 100,
                         "overall_coverage": float(ins[name].mean()), "mean_width": float(widths[name].mean())})
        # bootstrap over patients with fitted models held fixed (bands' calibration not refit)
        pats = t.subject_id.unique()
        diffs = []
        for _ in range(1000):
            pick = rng.choice(pats, len(pats), replace=True)
            idx = np.concatenate([np.flatnonzero(t.subject_id.to_numpy() == p) for p in pick])
            tb = t.iloc[idx]
            diffs.append((cov_err(tb, ins["CQR (tuned, full window)"][idx]) - cov_err(tb, ins["Mondrian (situation)"][idx])) * 100)
        boot_info[ds] = (np.percentile(diffs, [5, 95]), chosen)
        print(f"{ds} done ({time.time() - t0:.0f}s)", flush=True)

    res = pd.DataFrame(rows)
    res.to_csv(out / "h36_strong_cqr.csv", index=False)
    o = res[res.dataset == "ohiot1dm"].set_index("method").coverage_error
    cut = 1 - o["Mondrian (situation)"] / o["CQR (tuned, full window)"]
    verdict = "SUPPORTED" if cut >= H36_CUT else "NOT SUPPORTED"
    lines = [f"H36 (Mondrian at least 10% better calibrated than tuned full-window CQR, OhioT1DM): {verdict}"]
    for ds in RUNS:
        s = res[res.dataset == ds]
        ci, chosen = boot_info[ds]
        lines.append(f"  {ds}: " + "; ".join(f"{r.method} {r.coverage_error:.1f} pts, width {r.mean_width:.0f}, overall {r.overall_coverage:.1%}"
                                              for r in s.itertuples()))
        lines.append(f"    CQR minus Mondrian, 90% patient-bootstrap interval: {ci[0]:.1f} to {ci[1]:.1f} pts (positive = Mondrian better)")
        lines.append(f"    CQR settings chosen: low {chosen[0.10]}, high {chosen[0.90]}")
    lines.append(f"  Mondrian vs CQR on OhioT1DM: {cut:.0%} lower coverage error (rule: at least 10%)")
    text = "\n".join(lines)
    (out / "verdict_h36.txt").write_text(text + "\n")
    print(text)

    plotting.apply_style()
    C = plotting.CATEGORICAL
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    cols = [plotting.INK_MUTED, C[3], C[0]]
    for ax, ds in zip(axes, RUNS):
        s = res[res.dataset == ds]
        ax.bar(range(3), s.coverage_error, color=cols)
        for i, r in enumerate(s.itertuples()):
            ax.text(i, r.coverage_error + 0.1, f"{r.coverage_error:.1f} pts\nwidth {r.mean_width:.0f}", ha="center", fontsize=8)
        ax.set_xticks(range(3), [m.replace(" (", "\n(") for m in s.method], fontsize=8)
        ax.set_ylim(0, s.coverage_error.max() * 1.3)
        ax.set_ylabel("Average distance from 80% target (points)\nlower is better")
        ci, _ = boot_info[ds]
        ax.set_title(f"{'OhioT1DM (rule applies)' if ds == 'ohiot1dm' else 'AZT1D (second look)'}: CQR minus Mondrian 90% CI {ci[0]:.1f} to {ci[1]:.1f}",
                     loc="left", fontsize=9)
    fig.suptitle("H36: simple situation groups vs a tuned learned method (CQR)", x=0.01, ha="left", fontsize=11)
    fig.tight_layout()
    fig.savefig(figs / "h36_strong_cqr.png", dpi=150)
    plt.close(fig)
    print(f"runtime {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
