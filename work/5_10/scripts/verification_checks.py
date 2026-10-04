"""Verification checks H32-H35 for situation-grouped (Mondrian) conformal bands.

H32  Mondrian vs standard alternatives (OhioT1DM; AZT1D reported):
     plain conformal pooled, plain conformal per patient, CQR (conformalized quantile regression).
H33  Shuffled situation labels, 20 runs (OhioT1DM).
H34  Other models: GLIMMER fixed-weight CNN-LSTM, plain CNN-Transformer, on OhioT1DM and AZT1D.
H35  Bootstrap over patients, 1,000 resamples (OhioT1DM plain model).

Metric everywhere: per-situation coverage error = mean over real situations with at least 30 test
windows of |coverage - 80%|. All bands clipped to 40-400 mg/dL. Rules are in work/HYPOTHESES.md.

Run:    python work/5_10/scripts/verification_checks.py
Writes: work/5_10/results/verify_*.csv, verdicts_h32_h35.txt; work/5_10/figures/verify_h32..h35.png
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

from azt1d import plotting  # noqa: E402
from azt1d.glimmer import uncertainty as unc  # noqa: E402
import hybrid_bands as hb  # noqa: E402
from situation_bands import conformal_halfwidth  # noqa: E402

TARGET, MIN_VAL, MIN_TEST = 0.80, 100, 30
LO, HI = unc.SENSOR_MIN, unc.SENSOR_MAX
OUT, FIGS = WEEK / "results", WEEK / "figures"


def frames(dataset: str, run: str):
    hb.DATASETS = {dataset: run}
    return hb.frames_with_sigma(dataset)


def clip(a):
    return np.clip(a, LO, HI)


def cov_err(t, inside, cells=None):
    """Mean |coverage - 80%| over real situations with at least MIN_TEST test windows."""
    d = pd.DataFrame({"cell": t.cell.to_numpy() if cells is None else cells, "in": np.asarray(inside)})
    g = d.groupby("cell").filter(lambda x: len(x) >= MIN_TEST).groupby("cell")["in"].mean()
    return float((g - TARGET).abs().mean())


def inside(t, lo, hi):
    return (t.actual.to_numpy() >= clip(lo)) & (t.actual.to_numpy() <= clip(hi))


def group_q(v_scores, v_groups, t_groups, n_groups=25):
    pooled = conformal_halfwidth(v_scores, TARGET)
    q = np.full(n_groups, pooled)
    for c in range(n_groups):
        m = v_groups == c
        if m.sum() >= MIN_VAL:
            q[c] = conformal_halfwidth(v_scores[m], TARGET)
    return q[t_groups]


def mondrian(v, t, v_cells=None, t_cells=None):
    vc = v.cell.to_numpy() if v_cells is None else v_cells
    tc = t.cell.to_numpy() if t_cells is None else t_cells
    q = group_q(np.abs(v.resid.to_numpy()), vc, tc)
    return t.pred.to_numpy() - q, t.pred.to_numpy() + q


def garch(t):
    c = t.pred + t.mu
    return (c - hb.Z80 * t.sigma).to_numpy(), (c + hb.Z80 * t.sigma).to_numpy()


def hybrid(v, t):
    norm = (np.abs(v.resid - v.mu) / v.sigma).to_numpy()
    q = group_q(norm, v.cell.to_numpy(), t.cell.to_numpy())
    c = (t.pred + t.mu).to_numpy()
    return c - q * t.sigma.to_numpy(), c + q * t.sigma.to_numpy()


def conformal_pooled(v, t):
    q = conformal_halfwidth(np.abs(v.resid.to_numpy()), TARGET)
    return t.pred.to_numpy() - q, t.pred.to_numpy() + q


def conformal_per_patient(v, t):
    q = t.subject_id.map({s: conformal_halfwidth(np.abs(g.resid.to_numpy()), TARGET)
                          for s, g in v.groupby("subject_id")}).to_numpy()
    return t.pred.to_numpy() - q, t.pred.to_numpy() + q


def cqr(v, t):
    feats = lambda d: np.column_stack([d.pred, d.g_now, d.g_now - d.g_prev, d.sigma])  # noqa: E731
    half = v.groupby("subject_id").cumcount() < v.groupby("subject_id").subject_id.transform("size") / 2
    a, b = v[half], v[~half]
    models = {}
    for qn, qv in (("lo", 0.10), ("hi", 0.90)):
        m = HistGradientBoostingRegressor(loss="quantile", quantile=qv, max_iter=200, random_state=0)
        models[qn] = m.fit(feats(a), a.resid.to_numpy())
    lo_b, hi_b = models["lo"].predict(feats(b)), models["hi"].predict(feats(b))
    score = np.maximum(lo_b - b.resid.to_numpy(), b.resid.to_numpy() - hi_b)
    n = len(score)
    k = int(np.ceil((n + 1) * TARGET))
    q = float(np.sort(score)[min(k, n) - 1])
    p = t.pred.to_numpy()
    return p + models["lo"].predict(feats(t)) - q, p + models["hi"].predict(feats(t)) + q


def bar_figure(path, title, labels, values, highlight, note):
    plotting.apply_style()
    C = plotting.CATEGORICAL
    fig, ax = plt.subplots(figsize=(10, 4.8))
    cols = [C[0] if i == highlight else plotting.INK_MUTED for i in range(len(labels))]
    ax.bar(range(len(labels)), values, color=cols)
    for i, val in enumerate(values):
        ax.text(i, val + 0.1, f"{val:.1f}", ha="center", fontsize=9)
    ax.set_xticks(range(len(labels)), labels, fontsize=8)
    ax.set_ylabel("Average distance from 80% target (points)\nlower is better")
    ax.set_title(title, loc="left", fontsize=11)
    ax.text(0.99, 0.95, note, transform=ax.transAxes, ha="right", va="top", fontsize=8, color=plotting.INK_SECONDARY)
    ax.set_ylim(0, max(values) * 1.25)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    t0 = time.time()
    lines = []
    rng = np.random.default_rng(0)

    # ---------- H32
    rows = []
    data = {}
    for ds, run in (("ohiot1dm", "ohiot1dm_cnn_lstm_v0"), ("azt1d", "cnn_lstm_v0")):
        v, t = frames(ds, run)
        data[ds] = (v, t)
        methods = {"GARCH": garch(t), "plain conformal\n(pooled)": conformal_pooled(v, t),
                   "plain conformal\n(per patient)": conformal_per_patient(v, t), "CQR": cqr(v, t),
                   "Mondrian\n(situation)": mondrian(v, t), "Mondrian x GARCH\n(hybrid)": hybrid(v, t)}
        for name, (lo, hi) in methods.items():
            ins = inside(t, lo, hi)
            rows.append({"dataset": ds, "method": name.replace("\n", " "), "coverage_error": cov_err(t, ins) * 100,
                         "overall_coverage": float(ins.mean()), "mean_width": float(np.mean(clip(hi) - clip(lo)))})
        print(f"H32 {ds} done ({time.time() - t0:.0f}s)", flush=True)
    h32 = pd.DataFrame(rows)
    h32.to_csv(OUT / "verify_h32_alternatives.csv", index=False)
    o = h32[h32.dataset == "ohiot1dm"].set_index("method").coverage_error
    m = o["Mondrian (situation)"]
    rivals = ["plain conformal (pooled)", "plain conformal (per patient)", "CQR"]
    cuts = {r: 1 - m / o[r] for r in rivals}
    ok32 = all(c >= 0.20 for c in cuts.values())
    lines.append(f"H32 (Mondrian beats the standard alternatives by at least 20%, OhioT1DM): {'SUPPORTED' if ok32 else 'NOT SUPPORTED'}")
    for ds in ("ohiot1dm", "azt1d"):
        s = h32[h32.dataset == ds]
        lines.append(f"  {ds}: " + "; ".join(f"{r.method} {r.coverage_error:.1f} pts (width {r.mean_width:.0f})" for r in s.itertuples()))
    lines.append("  Mondrian cut vs each rival (OhioT1DM): " + ", ".join(f"{k} {c:.0%}" for k, c in cuts.items()))
    for ds in ("ohiot1dm", "azt1d"):
        s = h32[h32.dataset == ds]
        bar_figure(FIGS / f"verify_h32_alternatives_{ds}.png",
                   f"H32 ({'OhioT1DM, rule applies' if ds == 'ohiot1dm' else 'AZT1D, second look'}): situation-grouped bands vs standard methods",
                   [x.replace(" (", "\n(") for x in s.method], s.coverage_error.to_numpy(), 4,
                   "blue = situation-grouped (Mondrian)")

    # ---------- H33
    v, t = data["ohiot1dm"]
    real = cov_err(t, inside(t, *mondrian(v, t)))
    shuffled = []
    for _ in range(20):
        lo, hi = mondrian(v, t, rng.permutation(v.cell.to_numpy()), rng.permutation(t.cell.to_numpy()))
        shuffled.append(cov_err(t, inside(t, lo, hi)))
    shuffled = np.array(shuffled)
    ok33 = real < shuffled.min() and real <= 0.8 * shuffled.mean()
    pd.DataFrame({"run": ["real"] + [f"shuffle_{i}" for i in range(20)],
                  "coverage_error": np.r_[real, shuffled] * 100}).to_csv(OUT / "verify_h33_shuffle.csv", index=False)
    lines.append(f"H33 (real situations beat shuffled groups): {'SUPPORTED' if ok33 else 'NOT SUPPORTED'}")
    lines.append(f"  real {real * 100:.1f} pts; shuffled mean {shuffled.mean() * 100:.1f}, range {shuffled.min() * 100:.1f} to {shuffled.max() * 100:.1f}; real/shuffled-mean {real / shuffled.mean():.0%}")
    plotting.apply_style()
    C = plotting.CATEGORICAL
    fig, ax = plt.subplots(figsize=(10, 4.4))
    ax.scatter(shuffled * 100, np.zeros(20) + np.random.default_rng(1).uniform(-0.15, 0.15, 20), color=plotting.INK_MUTED, s=40,
               label="random groups (20 shuffles)")
    ax.scatter([real * 100], [0], color=C[0], s=140, marker="D", label="real situations", zorder=3)
    ax.set_yticks([])
    ax.set_ylim(-1, 1)
    ax.set_xlabel("Average distance from 80% target on the real situations (points; lower is better)")
    ax.set_title("H33: do the situations matter, or just having many groups? (OhioT1DM)", loc="left", fontsize=11)
    ax.legend(frameon=False, fontsize=9, loc="upper right")
    fig.tight_layout()
    fig.savefig(FIGS / "verify_h33_shuffle.png", dpi=150)
    plt.close(fig)
    print(f"H33 done ({time.time() - t0:.0f}s)", flush=True)

    # ---------- H34
    combos = [("ohiot1dm", "ohiot1dm_cnn_lstm_v1", "OhioT1DM, GLIMMER fixed-weight CNN-LSTM"),
              ("ohiot1dm", "ohiot1dm_cnn_transformer_v0", "OhioT1DM, plain CNN-Transformer"),
              ("azt1d", "cnn_lstm_v1", "AZT1D, GLIMMER fixed-weight CNN-LSTM"),
              ("azt1d", "cnn_transformer_v0", "AZT1D, plain CNN-Transformer")]
    rows = []
    for ds, run, label in combos:
        v2, t2 = frames(ds, run)
        g = cov_err(t2, inside(t2, *garch(t2)))
        mo = cov_err(t2, inside(t2, *mondrian(v2, t2)))
        hy = cov_err(t2, inside(t2, *hybrid(v2, t2)))
        rows.append({"combo": label, "garch": g * 100, "mondrian": mo * 100, "hybrid": hy * 100, "cut": 1 - mo / g})
        print(f"H34 {label} done ({time.time() - t0:.0f}s)", flush=True)
    h34 = pd.DataFrame(rows)
    h34.to_csv(OUT / "verify_h34_models.csv", index=False)
    ok34 = bool((h34.cut >= 0.20).all())
    lines.append(f"H34 (gain holds on other models, at least 20% in all 4): {'SUPPORTED' if ok34 else 'NOT SUPPORTED'}")
    for r in h34.itertuples():
        lines.append(f"  {r.combo}: GARCH {r.garch:.1f}, Mondrian {r.mondrian:.1f}, hybrid {r.hybrid:.1f} pts; Mondrian cut {r.cut:.0%}")
    fig, ax = plt.subplots(figsize=(11, 4.8))
    x = np.arange(len(h34)); w = 0.27
    for j, (col, lab, colr) in enumerate((("garch", "GARCH (current)", plotting.INK_MUTED), ("mondrian", "Mondrian (situation)", C[0]),
                                          ("hybrid", "Mondrian x GARCH (hybrid)", C[3]))):
        ax.bar(x + (j - 1) * w, h34[col], w, color=colr, label=lab)
    ax.set_xticks(x, [c.replace(", ", "\n") for c in h34.combo], fontsize=8)
    ax.set_ylabel("Average distance from 80% target (points)\nlower is better")
    ax.set_title("H34: does the gain hold on other forecasting models?", loc="left", fontsize=11)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(FIGS / "verify_h34_models.png", dpi=150)
    plt.close(fig)

    # ---------- H35
    pats = v.subject_id.unique()
    gains_mg, gains_hm = [], []
    for _ in range(1000):
        pick = rng.choice(pats, len(pats), replace=True)
        vb = pd.concat([v[v.subject_id == p].assign(subject_id=i) for i, p in enumerate(pick)], ignore_index=True)
        tb = pd.concat([t[t.subject_id == p].assign(subject_id=i) for i, p in enumerate(pick)], ignore_index=True)
        g = cov_err(tb, inside(tb, *garch(tb)))
        mo = cov_err(tb, inside(tb, *mondrian(vb, tb)))
        hy = cov_err(tb, inside(tb, *hybrid(vb, tb)))
        gains_mg.append((g - mo) * 100)
        gains_hm.append((mo - hy) * 100)
    gains_mg, gains_hm = np.array(gains_mg), np.array(gains_hm)
    ci_mg, ci_hm = np.percentile(gains_mg, [5, 95]), np.percentile(gains_hm, [5, 95])
    ok35 = ci_mg[0] > 0
    pd.DataFrame({"mondrian_vs_garch": gains_mg, "hybrid_vs_mondrian": gains_hm}).to_csv(OUT / "verify_h35_bootstrap.csv", index=False)
    lines.append(f"H35 (Mondrian-vs-GARCH gain survives patient resampling): {'SUPPORTED' if ok35 else 'NOT SUPPORTED'}")
    lines.append(f"  Mondrian vs GARCH gain: median {np.median(gains_mg):.1f} pts, 90% interval {ci_mg[0]:.1f} to {ci_mg[1]:.1f}")
    lines.append(f"  hybrid vs Mondrian gain: median {np.median(gains_hm):.1f} pts, 90% interval {ci_hm[0]:.1f} to {ci_hm[1]:.1f} (not in the rule)")
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.4))
    for ax, gains, ci, title in ((axes[0], gains_mg, ci_mg, "Mondrian (situation) vs GARCH"),
                                 (axes[1], gains_hm, ci_hm, "Hybrid vs Mondrian")):
        ax.hist(gains, bins=40, color=C[0], alpha=0.8)
        ax.axvline(0, color=plotting.INK_PRIMARY, lw=1.2)
        for c in ci:
            ax.axvline(c, color=C[1], ls="--", lw=1.2)
        ax.set_xlabel("Gain in points (right of the black line = improvement)")
        ax.set_title(f"{title}: 90% interval {ci[0]:.1f} to {ci[1]:.1f}", loc="left", fontsize=10)
    axes[0].set_ylabel("Bootstrap resamples")
    fig.suptitle("H35: are the gains real, or luck in which patients were included? (1,000 resamples, OhioT1DM)",
                 x=0.01, ha="left", fontsize=11)
    fig.tight_layout()
    fig.savefig(FIGS / "verify_h35_bootstrap.png", dpi=150)
    plt.close(fig)

    text = "\n".join(lines)
    (OUT / "verdicts_h32_h35.txt").write_text(text + "\n")
    print(text)
    print(f"runtime {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
