"""H28: do situation-specific bands beat GARCH on a fresh dataset (OhioT1DM)?

Same method as H27 (situation_bands.py), applied to the plain OhioT1DM checkpoints
(ohiot1dm_cnn_lstm_v0, 12 patients), which no situation-band work has touched.

Coverage error = mean, over situations with at least 30 test windows, of |coverage - 80%|.
H28 supported if pooled coverage error falls at least 20% vs GARCH 80% bands, and the
per-patient coverage error (situations with at least 30 of that patient's test windows) falls
for at least 7 of 12 patients.

Also reported, not part of the rule: mean band width for both methods.

Run:    python work/5_10/scripts/ohio_situation_bands.py
Writes: work/5_10/results/h28_ohio_situations.csv, h28_ohio_patients.csv, verdict_h28.txt,
        work/5_10/figures/h28_ohio_situation_bands.png
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = next(p for p in Path(__file__).resolve().parents if (p / "pyproject.toml").exists())
WEEK = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from azt1d import plotting, reference as ref  # noqa: E402
from azt1d.metabonet import load_source  # noqa: E402
from azt1d.glimmer import checkpoint as ckpt, uncertainty as unc  # noqa: E402
import predictability_map as pm  # noqa: E402
from situation_bands import conformal_halfwidth  # noqa: E402

RUN = "ohiot1dm_cnn_lstm_v0"
TARGET, MIN_VAL, MIN_TEST = 0.80, 100, 30
H28_CUT, H28_MIN_PATIENTS = 0.20, 7


def rows_for(sid, frame, times, g, split):
    i = np.searchsorted(times, frame.issue_time)
    assert np.all(times[i] == frame.issue_time), "issue times do not line up"
    keep = i >= pm.TREND_STEPS
    d = pd.DataFrame({"subject_id": sid, "split": split, "actual": frame.actual, "pred": frame.pred,
                      "resid": frame.resid, "g_now": g[i], "g_prev": g[np.maximum(i - pm.TREND_STEPS, 0)]})
    return d[keep].reset_index(drop=True), keep


def build():
    df = load_source("OhioT1DM")
    vals, tests = [], []
    for sid in sorted(int(s) for s in df["subject_id"].unique()):
        res = ckpt.load_result(ROOT / "data" / "processed" / "checkpoints" / RUN, sid)
        dfs = df[df["subject_id"] == sid].reset_index(drop=True)
        val, test = unc.forecast_frames(res, dfs)
        assert unc.matches_stored_predictions(test, res) < 0.05
        glo, ghi = unc.BandEngine(val, test, analogs=False).bands("GARCH", TARGET)
        times = pd.to_datetime(dfs[ref.EVENT_DATETIME]).to_numpy()
        g = dfs[ref.CGM].to_numpy()
        v, _ = rows_for(sid, val, times, g, "val")
        t, keep = rows_for(sid, test, times, g, "test")
        t["garch_lo"], t["garch_hi"] = glo[keep], ghi[keep]
        vals.append(v); tests.append(t)
    val_all, test_all = pd.concat(vals, ignore_index=True), pd.concat(tests, ignore_index=True)
    for f in (val_all, test_all):
        f["cell"] = np.digitize(f.g_now.to_numpy(), pm.LEVEL_EDGES) * 5 + np.digitize((f.g_now - f.g_prev).to_numpy(), pm.RATE_EDGES)
    return val_all, test_all


def add_situation_bands(val, test):
    pooled = conformal_halfwidth(np.abs(val.resid.to_numpy()), TARGET)
    half = {}
    for c in range(25):
        v = val[val.cell == c]
        half[c] = conformal_halfwidth(np.abs(v.resid.to_numpy()), TARGET) if len(v) >= MIN_VAL else pooled
    test = test.copy()
    q = test.cell.map(half).to_numpy()
    test["sit_lo"], test["sit_hi"] = test.pred - q, test.pred + q
    test["in_sit"] = (test.actual >= test.sit_lo) & (test.actual <= test.sit_hi)
    test["in_garch"] = (test.actual >= test.garch_lo) & (test.actual <= test.garch_hi)
    return test, sum(1 for c in range(25) if len(val[val.cell == c]) < MIN_VAL)


def coverage_error(t: pd.DataFrame):
    cells = t.groupby("cell").filter(lambda g: len(g) >= MIN_TEST).groupby("cell")
    cs = cells.in_sit.mean(); cg = cells.in_garch.mean()
    return float((cs - TARGET).abs().mean()), float((cg - TARGET).abs().mean()), cs, cg, cells.size()


def main():
    t0 = time.time()
    out, figs = WEEK / "results", WEEK / "figures"
    val, test = build()
    test, n_fallback = add_situation_bands(val, test)
    print(f"OhioT1DM: validation {len(val):,}, test {len(test):,} windows ({time.time() - t0:.0f}s)", flush=True)

    e_sit, e_garch, cs, cg, sizes = coverage_error(test)
    cut = 1 - e_sit / e_garch
    sit_tab = pd.DataFrame({"n_test": sizes, "coverage_situation": cs, "coverage_garch": cg})
    sit_tab["level"] = [pm.LEVEL_NAMES[c // 5] for c in sit_tab.index]
    sit_tab["trend"] = [pm.RATE_NAMES[c % 5] for c in sit_tab.index]
    sit_tab.to_csv(out / "h28_ohio_situations.csv")

    prow = []
    for sid, t in test.groupby("subject_id"):
        ps, pg, *_ = coverage_error(t)
        prow.append({"subject_id": sid, "coverage_error_situation": ps, "coverage_error_garch": pg,
                     "mean_width_situation": float((t.sit_hi - t.sit_lo).mean()),
                     "mean_width_garch": float((t.garch_hi - t.garch_lo).mean())})
    ptab = pd.DataFrame(prow)
    ptab.to_csv(out / "h28_ohio_patients.csv", index=False)
    better = int((ptab.coverage_error_situation < ptab.coverage_error_garch).sum())
    ok = cut >= H28_CUT and better >= H28_MIN_PATIENTS
    text = "\n".join([
        f"H28 (situation bands beat GARCH on fresh data, OhioT1DM): {'SUPPORTED' if ok else 'NOT SUPPORTED'}",
        f"  pooled coverage error: GARCH {e_garch * 100:.1f} points, situation-specific {e_sit * 100:.1f} points, cut {cut:.0%} (rule: at least 20%)",
        f"  situations scored (at least {MIN_TEST} test windows): {len(sit_tab)}; situations on pooled width: {n_fallback}",
        f"  patients where situation-specific is closer to 80%: {better} of {len(ptab)} (rule: at least 7)",
        f"  overall coverage: GARCH {test.in_garch.mean():.1%}, situation-specific {test.in_sit.mean():.1%}",
        f"  mean band width: GARCH {(test.garch_hi - test.garch_lo).mean():.1f}, situation-specific {(test.sit_hi - test.sit_lo).mean():.1f} mg/dL (not part of the rule)",
    ])
    (out / "verdict_h28.txt").write_text(text + "\n")
    print(text, flush=True)

    plotting.apply_style()
    C = plotting.CATEGORICAL
    fig, (a, b) = plt.subplots(1, 2, figsize=(14, 5.6), gridspec_kw={"width_ratios": [1.6, 1]})
    d = sit_tab.assign(eg=(sit_tab.coverage_garch - TARGET).abs() * 100,
                       es=(sit_tab.coverage_situation - TARGET).abs() * 100).sort_values("eg")
    y = np.arange(len(d))
    for yi, (g_, s_) in enumerate(zip(d.eg, d.es)):
        a.plot([g_, s_], [yi, yi], color=C[2] if s_ < g_ else C[1], lw=2, alpha=0.7)
    a.scatter(d.eg, y, color=plotting.INK_MUTED, s=36, label="GARCH (current)", zorder=3)
    a.scatter(d.es, y, color=C[0], s=36, label="situation-specific (new)", zorder=3)
    a.set_yticks(y, [f"{l}, {t}" for l, t in zip(d.level, d.trend)], fontsize=7)
    a.set_xlabel("Distance from the 80% target (percentage points; closer to 0 is better)")
    a.set_title("By situation: green line = the new band got closer to 80%", loc="left", fontsize=10)
    a.legend(frameon=False, fontsize=8, loc="lower right")
    p = ptab.sort_values("coverage_error_garch")
    x = np.arange(len(p)); w = 0.38
    b.bar(x - w / 2, p.coverage_error_garch * 100, w, color=plotting.INK_MUTED, label="GARCH (current)")
    b.bar(x + w / 2, p.coverage_error_situation * 100, w, color=C[0], label="situation-specific (new)")
    b.set_xticks(x, [str(s) for s in p.subject_id], fontsize=8)
    b.set_xlabel("OhioT1DM patient")
    b.set_ylabel("Distance from 80% target (points)")
    b.set_title(f"By patient: new band closer for {better} of {len(p)}", loc="left", fontsize=10)
    b.legend(frameon=False, fontsize=8)
    fig.suptitle(f"Fresh-data check (OhioT1DM): average distance from target {e_garch * 100:.1f} -> {e_sit * 100:.1f} points ({cut:.0%} better)",
                 x=0.01, ha="left", fontsize=11)
    fig.tight_layout()
    fig.savefig(figs / "h28_ohio_situation_bands.png", dpi=150)
    plt.close(fig)
    print(f"runtime {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
