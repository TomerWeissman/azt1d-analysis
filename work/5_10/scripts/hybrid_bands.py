"""H31: GARCH x situation hybrid bands vs GARCH and situation-specific bands.

Three 80% bands around the same plain-model forecast:
  - GARCH: centre pred + mu, half-width 1.28 x GARCH sigma (current method)
  - situation: centre pred, half-width = 80% conformal quantile of |validation error| in the situation
  - hybrid: centre pred + mu, half-width = q_situation x GARCH sigma, where q_situation is the 80%
    conformal quantile of |validation error - mu| / validation GARCH sigma in that situation

GARCH sigma on validation is filtered through the validation residuals with the same 60-minute
lag as on test, from the GARCH model fitted on validation. Situations with under 100 validation
windows use the pooled quantile.

H31 (OhioT1DM): supported if hybrid pooled coverage error is at least 10% lower than situation
alone, and lower for at least 7 of 12 patients. AZT1D is run too, as a second look.

Run:    python work/5_10/scripts/hybrid_bands.py [example_subject]   (default 588, OhioT1DM)
Writes: work/5_10/results/h31_hybrid_<dataset>.csv, verdict_h31.txt,
        work/5_10/figures/h31_hybrid_summary.png, example_patient_<id>_three_bands.png
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = next(p for p in Path(__file__).resolve().parents if (p / "pyproject.toml").exists())
WEEK = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from azt1d import loading, plotting, reference as ref  # noqa: E402
from azt1d.metabonet import load_source  # noqa: E402
from azt1d.glimmer import checkpoint as ckpt, uncertainty as unc  # noqa: E402
import predictability_map as pm  # noqa: E402
from situation_bands import conformal_halfwidth  # noqa: E402

TARGET, MIN_VAL, MIN_TEST, Z80 = 0.80, 100, 30, 1.2815515655446004
DATASETS = {"ohiot1dm": "ohiot1dm_cnn_lstm_v0", "azt1d": "cnn_lstm_v0"}
METHODS = {"GARCH (current)": "garch", "situation only": "sit", "GARCH x situation (hybrid)": "hyb"}


def load(name):
    if name == "ohiot1dm":
        return load_source("OhioT1DM")
    return loading.load_real_dataset(ROOT / "data" / "raw", ROOT / "data" / "processed")


def frames_with_sigma(name):
    df = load(name)
    vals, tests = [], []
    for sid in sorted(int(s) for s in df["subject_id"].unique()):
        res = ckpt.load_result(ROOT / "data" / "processed" / "checkpoints" / DATASETS[name], sid)
        dfs = df[df["subject_id"] == sid].reset_index(drop=True)
        val, test = unc.forecast_frames(res, dfs)
        eng = unc.BandEngine(val, test, analogs=False)
        p = eng.garch_params
        init = float(np.var(val.resid - p.mu))
        s_val = np.sqrt(unc.garch_h_step_variance(val.resid, p, init, unc.LAG))
        times = pd.to_datetime(dfs[ref.EVENT_DATETIME]).to_numpy()
        g = dfs[ref.CGM].to_numpy()
        for fr, sig, bucket, split in ((val, s_val, vals, "val"), (test, eng.garch_sigma, tests, "test")):
            i = np.searchsorted(times, fr.issue_time)
            assert np.all(times[i] == fr.issue_time)
            d = pd.DataFrame({"subject_id": sid, "time": pd.to_datetime(fr.target_time), "actual": fr.actual,
                              "pred": fr.pred, "resid": fr.resid, "mu": p.mu, "sigma": sig,
                              "g_now": g[i], "g_prev": g[np.maximum(i - pm.TREND_STEPS, 0)]})
            bucket.append(d[(i >= pm.TREND_STEPS) & np.isfinite(sig)].reset_index(drop=True))
    v, t = pd.concat(vals, ignore_index=True), pd.concat(tests, ignore_index=True)
    for f in (v, t):
        f["cell"] = np.digitize(f.g_now.to_numpy(), pm.LEVEL_EDGES) * 5 + np.digitize((f.g_now - f.g_prev).to_numpy(), pm.RATE_EDGES)
    return v, t


def add_bands(v, t):
    t = t.copy()
    abs_err = np.abs(v.resid.to_numpy())
    norm = (np.abs(v.resid - v.mu) / v.sigma).to_numpy()
    pooled_sit, pooled_hyb = conformal_halfwidth(abs_err, TARGET), conformal_halfwidth(norm, TARGET)
    q_sit, q_hyb = {}, {}
    for c in range(25):
        m = (v.cell == c).to_numpy()
        enough = m.sum() >= MIN_VAL
        q_sit[c] = conformal_halfwidth(abs_err[m], TARGET) if enough else pooled_sit
        q_hyb[c] = conformal_halfwidth(norm[m], TARGET) if enough else pooled_hyb
    c_ = t.pred + t.mu
    t["garch_lo"], t["garch_hi"] = c_ - Z80 * t.sigma, c_ + Z80 * t.sigma
    qs = t.cell.map(q_sit)
    t["sit_lo"], t["sit_hi"] = t.pred - qs, t.pred + qs
    qh = t.cell.map(q_hyb)
    t["hyb_lo"], t["hyb_hi"] = c_ - qh * t.sigma, c_ + qh * t.sigma
    for k in METHODS.values():
        # clip to the sensor range (40 to 400 mg/dL), the same as every earlier band in this project
        t[f"{k}_lo"] = t[f"{k}_lo"].clip(unc.SENSOR_MIN, unc.SENSOR_MAX)
        t[f"{k}_hi"] = t[f"{k}_hi"].clip(unc.SENSOR_MIN, unc.SENSOR_MAX)
        t[f"in_{k}"] = (t.actual >= t[f"{k}_lo"]) & (t.actual <= t[f"{k}_hi"])
        t[f"w_{k}"] = t[f"{k}_hi"] - t[f"{k}_lo"]
    return t


def cov_err(t, k):
    g = t.groupby("cell").filter(lambda x: len(x) >= MIN_TEST).groupby("cell")[f"in_{k}"].mean()
    return float((g - TARGET).abs().mean())


def summarize(t):
    rows = []
    for label, k in METHODS.items():
        rows.append({"method": label, "coverage_error_points": cov_err(t, k) * 100,
                     "overall_coverage": float(t[f"in_{k}"].mean()), "mean_width": float(t[f"w_{k}"].mean())})
    per_patient = pd.DataFrame([{"subject_id": s, **{k: cov_err(g, k) for k in METHODS.values()}}
                                for s, g in t.groupby("subject_id")])
    return pd.DataFrame(rows), per_patient


def main(example_sid: int):
    out, figs = WEEK / "results", WEEK / "figures"
    results, texts = {}, []
    for name in DATASETS:
        v, t = frames_with_sigma(name)
        t = add_bands(v, t)
        summ, pp = summarize(t)
        summ.to_csv(out / f"h31_hybrid_{name}.csv", index=False)
        results[name] = (t, summ, pp)
        e = summ.set_index("method").coverage_error_points
        cut = 1 - e["GARCH x situation (hybrid)"] / e["situation only"]
        better = int((pp.hyb < pp.sit).sum())
        line = (f"{name}: coverage error GARCH {e['GARCH (current)']:.1f}, situation {e['situation only']:.1f}, "
                f"hybrid {e['GARCH x situation (hybrid)']:.1f} points; hybrid vs situation {cut:+.0%} better; "
                f"hybrid better for {better} of {len(pp)} patients; widths "
                + ", ".join(f"{m} {w:.0f}" for m, w in zip(summ.method, summ.mean_width)) + " mg/dL")
        texts.append(line)
        if name == "ohiot1dm":
            verdict = "SUPPORTED" if cut >= 0.10 and better >= 7 else "NOT SUPPORTED"
    text = f"H31 (hybrid beats situation-only on OhioT1DM): {verdict}\n  " + "\n  ".join(texts) + \
        "\n  rule (OhioT1DM): at least 10% lower coverage error than situation-only, and lower for at least 7 of 12 patients"
    (out / "verdict_h31.txt").write_text(text + "\n")
    print(text)

    plotting.apply_style()
    C = plotting.CATEGORICAL
    cols = [plotting.INK_MUTED, C[0], C[3]]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    for ax, name in zip(axes, DATASETS):
        s = results[name][1]
        x = np.arange(len(s))
        ax.bar(x, s.coverage_error_points, color=cols)
        for xi, (e, w) in enumerate(zip(s.coverage_error_points, s.mean_width)):
            ax.text(xi, e + 0.1, f"{e:.1f} pts\nwidth {w:.0f}", ha="center", fontsize=8)
        ax.set_xticks(x, s.method, fontsize=8)
        ax.set_ylabel("Average distance from 80% target (points)")
        ax.set_title(f"{'OhioT1DM (rule applies)' if name == 'ohiot1dm' else 'AZT1D (second look)'}: lower is better",
                     loc="left", fontsize=10)
        ax.set_ylim(0, s.coverage_error_points.max() * 1.3)
    fig.suptitle("Three ways to draw the 80% band: how close each gets to catching 80%", x=0.01, ha="left", fontsize=11)
    fig.tight_layout()
    fig.savefig(figs / "h31_hybrid_summary.png", dpi=150)
    plt.close(fig)

    t = results["ohiot1dm"][0]
    p = t[t.subject_id == example_sid].reset_index(drop=True)
    rng = p.actual.rolling(288).max() - p.actual.rolling(288).min()
    end = int(rng.idxmax()) + 1
    w = p.iloc[end - 288:end]
    fig, axes = plt.subplots(3, 1, figsize=(13, 11), sharex=True, sharey=True)
    for ax, (label, k), col in zip(axes, METHODS.items(), cols):
        ax.fill_between(w.time, w[f"{k}_lo"], w[f"{k}_hi"], color=col, alpha=0.25, lw=0, label="80% band")
        ax.plot(w.time, w.pred, color=col, lw=1.2, label="forecast (60 min ahead)")
        ax.plot(w.time, w.actual, color=plotting.INK_PRIMARY, lw=1.4, label="actual glucose")
        miss = ~w[f"in_{k}"]
        ax.scatter(w.time[miss], w.actual[miss], color=C[1], s=14, zorder=4, label="actual fell outside the band")
        ax.axhline(ref.HYPO_THRESHOLD, ls="--", lw=0.8, color=plotting.GLUCOSE_BAND_COLORS["hypo"])
        ax.axhline(ref.HYPER_THRESHOLD, ls="--", lw=0.8, color=plotting.GLUCOSE_BAND_COLORS["hyper"])
        ax.set_title(f"{label}: caught {w[f'in_{k}'].mean():.0%} in this window, {p[f'in_{k}'].mean():.0%} over the whole "
                     f"test period; average width {w[f'w_{k}'].mean():.0f} mg/dL", loc="left", fontsize=10)
        ax.set_ylabel("Glucose (mg/dL)")
    axes[0].legend(frameon=False, fontsize=8, ncol=4, loc="upper left")
    axes[-1].set_xlabel("Time (24 hours with the widest glucose range in this patient's test period)")
    fig.suptitle(f"OhioT1DM patient {example_sid}: same forecast, three ways to draw the 80% band", x=0.01, ha="left", fontsize=11)
    fig.tight_layout()
    fig.savefig(figs / f"example_patient_{example_sid}_three_bands.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 588)
