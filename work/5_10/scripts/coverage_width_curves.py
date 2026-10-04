"""H37: level-free comparison. How many points does each method catch for a given band width?

Each method is drawn at levels 50, 60, 70, 80, 85, 90, 95%. For each level: overall share of test
points caught, and average band width (both after clipping to 40-400 mg/dL). One curve per method.
A curve that sits higher catches more points for the same width.

H37: for each GARCH point, read the Mondrian (situation) catch rate off its curve at the same width
(linear interpolation). Supported if Mondrian is higher at every GARCH point inside Mondrian's
width range, on both datasets.

Also reported: mean interval score at 80% and 95% (lower is better). Interval score =
width + (2/alpha) x distance by which a point falls outside the band, alpha = 1 - level.

Run:    python work/5_10/scripts/coverage_width_curves.py
Writes: work/5_10/results/h37_curves.csv, verdict_h37.txt, work/5_10/figures/h37_coverage_width.png
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.ensemble import HistGradientBoostingRegressor

ROOT = next(p for p in Path(__file__).resolve().parents if (p / "pyproject.toml").exists())
WEEK = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from azt1d import plotting  # noqa: E402
from situation_bands import conformal_halfwidth  # noqa: E402
from verification_checks import frames, clip, MIN_VAL  # noqa: E402

LEVELS = (0.50, 0.60, 0.70, 0.80, 0.85, 0.90, 0.95)
RUNS = {"ohiot1dm": "ohiot1dm_cnn_lstm_v0", "azt1d": "cnn_lstm_v0"}


def group_q(scores, v_groups, t_groups, level, n=25):
    pooled = conformal_halfwidth(scores, level)
    q = np.full(n, pooled)
    for c in range(n):
        m = v_groups == c
        if m.sum() >= MIN_VAL:
            q[c] = conformal_halfwidth(scores[m], level)
    return q[t_groups]


def garch(v, t, level, cache):
    z = norm.ppf(0.5 + level / 2)
    c = (t.pred + t.mu).to_numpy()
    return c - z * t.sigma.to_numpy(), c + z * t.sigma.to_numpy()


def conformal(v, t, level, cache):
    q = conformal_halfwidth(np.abs(v.resid.to_numpy()), level)
    return t.pred.to_numpy() - q, t.pred.to_numpy() + q


def mondrian(v, t, level, cache):
    q = group_q(np.abs(v.resid.to_numpy()), v.cell.to_numpy(), t.cell.to_numpy(), level)
    return t.pred.to_numpy() - q, t.pred.to_numpy() + q


def hybrid(v, t, level, cache):
    norm_s = (np.abs(v.resid - v.mu) / v.sigma).to_numpy()
    q = group_q(norm_s, v.cell.to_numpy(), t.cell.to_numpy(), level)
    c = (t.pred + t.mu).to_numpy()
    return c - q * t.sigma.to_numpy(), c + q * t.sigma.to_numpy()


def cqr(v, t, level, cache):
    feats = lambda d: np.column_stack([d.pred, d.g_now, d.g_now - d.g_prev, d.sigma])  # noqa: E731
    pos = v.groupby("subject_id").cumcount() / v.groupby("subject_id").subject_id.transform("size")
    a, b = v[pos < 0.5], v[pos >= 0.5]
    a_lo, a_hi = (1 - level) / 2, 1 - (1 - level) / 2
    ms = {}
    for qv in (a_lo, a_hi):
        key = round(qv, 4)
        if key not in cache:
            cache[key] = HistGradientBoostingRegressor(loss="quantile", quantile=qv, max_iter=200,
                                                       random_state=0).fit(feats(a), a.resid.to_numpy())
        ms[qv] = cache[key]
    lo_b, hi_b = ms[a_lo].predict(feats(b)), ms[a_hi].predict(feats(b))
    score = np.maximum(lo_b - b.resid.to_numpy(), b.resid.to_numpy() - hi_b)
    q = conformal_halfwidth(score, level)
    p = t.pred.to_numpy()
    return p + ms[a_lo].predict(feats(t)) - q, p + ms[a_hi].predict(feats(t)) + q


METHODS = {"GARCH": garch, "plain conformal": conformal, "CQR": cqr,
           "situation (Mondrian)": mondrian, "situation x GARCH": hybrid}


def interval_score(y, lo, hi, level):
    a = 1 - level
    return float(np.mean((hi - lo) + (2 / a) * np.maximum(lo - y, 0) + (2 / a) * np.maximum(y - hi, 0)))


def main():
    t0 = time.time()
    rows = []
    for ds, run in RUNS.items():
        v, t = frames(ds, run)
        y = t.actual.to_numpy()
        cache = {}
        for name, fn in METHODS.items():
            for lv in LEVELS:
                lo, hi = fn(v, t, lv, cache)
                lo, hi = clip(lo), clip(hi)
                rows.append({"dataset": ds, "method": name, "level": lv,
                             "caught": float(np.mean((y >= lo) & (y <= hi))),
                             "width": float(np.mean(hi - lo)),
                             "interval_score": interval_score(y, lo, hi, lv)})
        print(f"{ds} done ({time.time() - t0:.0f}s)", flush=True)
    res = pd.DataFrame(rows)
    res.to_csv(WEEK / "results" / "h37_curves.csv", index=False)

    lines, ok_all = [], True
    for ds in RUNS:
        d = res[res.dataset == ds]
        g = d[d.method == "GARCH"].sort_values("width")
        m = d[d.method == "situation (Mondrian)"].sort_values("width")
        comps = []
        for r in g.itertuples():
            if m.width.min() <= r.width <= m.width.max():
                mc = float(np.interp(r.width, m.width, m.caught))
                comps.append((r.level, r.width, r.caught, mc))
        ok = len(comps) > 0 and all(mc > gc for _, _, gc, mc in comps)
        ok_all &= ok
        lines.append(f"  {ds}: Mondrian higher at {sum(mc > gc for *_, gc, mc in comps)} of {len(comps)} GARCH points in range")
        for lv, w, gc, mc in comps:
            lines.append(f"    GARCH {lv:.0%}: width {w:.0f}, GARCH caught {gc:.1%}, Mondrian at same width {mc:.1%} ({(mc - gc) * 100:+.1f} pts)")
        for lv in (0.80, 0.95):
            s = d[d.level == lv].set_index("method").interval_score
            lines.append(f"    interval score at {lv:.0%} (lower is better): " + ", ".join(f"{k} {val:.0f}" for k, val in s.items()))
    text = f"H37 (at the same width, situation bands catch more than GARCH, both datasets): {'SUPPORTED' if ok_all else 'NOT SUPPORTED'}\n" + "\n".join(lines)
    (WEEK / "results" / "verdict_h37.txt").write_text(text + "\n")
    print(text)

    plotting.apply_style()
    C = plotting.CATEGORICAL
    colors = {"GARCH": plotting.INK_MUTED, "plain conformal": C[4], "CQR": C[3],
              "situation (Mondrian)": C[0], "situation x GARCH": C[2]}
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.4), sharey=True)
    for ax, ds in zip(axes, RUNS):
        d = res[res.dataset == ds]
        for name in METHODS:
            s = d[d.method == name].sort_values("width")
            lw = 2.4 if name == "situation (Mondrian)" else 1.4
            ax.plot(s.width, s.caught * 100, marker="o", ms=4, lw=lw, color=colors[name], label=name)
        ax.set_xlabel("Average band width (mg/dL)")
        ax.set_title(f"{'OhioT1DM' if ds == 'ohiot1dm' else 'AZT1D'}: higher = more points caught for the same width",
                     loc="left", fontsize=10)
    axes[0].set_ylabel("Share of true glucose values inside the band (%)")
    axes[0].legend(frameon=False, fontsize=8, loc="lower right")
    fig.suptitle("Points caught vs band width, every method at levels 50% to 95% (no single target level)",
                 x=0.01, ha="left", fontsize=11)
    fig.tight_layout()
    fig.savefig(WEEK / "figures" / "h37_coverage_width.png", dpi=150)
    plt.close(fig)
    print(f"runtime {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
