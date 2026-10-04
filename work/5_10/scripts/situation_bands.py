"""H27: situation-specific band widths vs the current GARCH bands.

Each of the 25 situations from the predictability map (current level by 30-minute trend) gets its
own band width. The width covers 80% of that situation's validation errors (split conformal,
per situation). Where a situation has fewer than 100 validation windows, the pooled validation
width is used instead, and the fallback is reported.

Bands are centred on the plain model's forecast. Coverage is then checked on the test period, per
situation, and compared with the current 80% GARCH bands on the same windows.

H27 is supported if at least 20 of the 25 situations land between 75% and 85% coverage. Empty
situations count as not inside.

No training. Uses the saved plain-model checkpoints and the existing band code.

Run:    python work/5_10/scripts/situation_bands.py
Writes: work/5_10/results/situation_bands.csv, work/5_10/results/verdict_h27.txt,
        work/5_10/figures/situation_bands.png
"""
from __future__ import annotations

import math
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

from azt1d import loading, plotting, reference as ref  # noqa: E402
from azt1d.glimmer import checkpoint as ckpt, uncertainty as unc  # noqa: E402
import predictability_map as pm  # noqa: E402  (same week: situation grid and names)

RUN = "cnn_lstm_v0"
TARGET = 0.80
MIN_VAL = 100
H27_MIN_INSIDE = 20
LOW, HIGH = 0.75, 0.85


def split_rows(sid, val, test, times, g) -> tuple[pd.DataFrame, pd.DataFrame]:
    out = []
    for name, fr in (("val", val), ("test", test)):
        i = np.searchsorted(times, fr.issue_time)
        assert np.all(times[i] == fr.issue_time), "issue times do not line up"
        keep = i >= pm.TREND_STEPS
        d = pd.DataFrame({
            "subject_id": sid, "split": name, "actual": fr.actual, "pred": fr.pred,
            "resid": fr.resid, "g_now": g[i], "g_prev": g[np.maximum(i - pm.TREND_STEPS, 0)],
        })[keep]
        out.append(d.reset_index(drop=True))
    return out[0], out[1]


def build() -> tuple[pd.DataFrame, pd.DataFrame]:
    df = loading.load_real_dataset(ROOT / "data" / "raw", ROOT / "data" / "processed")
    vals, tests = [], []
    for sid in sorted(int(s) for s in df["subject_id"].unique()):
        res = ckpt.load_result(ROOT / "data" / "processed" / "checkpoints" / RUN, sid)
        dfs = df[df["subject_id"] == sid].reset_index(drop=True)
        val, test = unc.forecast_frames(res, dfs)
        eng = unc.BandEngine(val, test, analogs=False)
        glo, ghi = eng.bands("GARCH", TARGET)
        times = pd.to_datetime(dfs[ref.EVENT_DATETIME]).to_numpy()
        g = dfs[ref.CGM].to_numpy()
        v, t = split_rows(sid, val, test, times, g)
        # align the GARCH band to the kept test rows
        i_test = np.searchsorted(times, test.issue_time)
        keep = i_test >= pm.TREND_STEPS
        t["garch_lo"], t["garch_hi"] = glo[keep], ghi[keep]
        vals.append(v); tests.append(t)
    val_all, test_all = pd.concat(vals, ignore_index=True), pd.concat(tests, ignore_index=True)
    for frame in (val_all, test_all):
        frame["lvl"] = np.digitize(frame.g_now.to_numpy(), pm.LEVEL_EDGES)
        frame["trd"] = np.digitize((frame.g_now - frame.g_prev).to_numpy(), pm.RATE_EDGES)
        frame["cell"] = frame.lvl * 5 + frame.trd
    return val_all, test_all


def conformal_halfwidth(abs_resid: np.ndarray, level: float) -> float:
    n = len(abs_resid)
    k = math.ceil((n + 1) * level)
    if k > n:
        return float(np.max(abs_resid))
    return float(np.sort(abs_resid)[k - 1])


def situation_table(val: pd.DataFrame, test: pd.DataFrame) -> pd.DataFrame:
    pooled_q = conformal_halfwidth(np.abs(val.resid.to_numpy()), TARGET)
    rows = []
    for lvl in range(5):
        for trd in range(5):
            c = lvl * 5 + trd
            v = val[val.cell == c]
            t = test[test.cell == c]
            rec = {"level": pm.LEVEL_NAMES[lvl], "trend": pm.RATE_NAMES[trd],
                   "n_val": len(v), "n_test": len(t)}
            if len(v) >= MIN_VAL:
                q, source = conformal_halfwidth(np.abs(v.resid.to_numpy()), TARGET), "own"
            else:
                q, source = pooled_q, "pooled"
            rec["halfwidth_source"] = source
            rec["halfwidth"] = q
            if len(t) == 0:
                rec.update({"coverage_situation": np.nan, "coverage_garch": np.nan,
                            "rmse_test": np.nan, "width_ratio": np.nan})
                rows.append(rec); continue
            lo, hi = t.pred - q, t.pred + q
            rec["coverage_situation"] = float(np.mean((t.actual >= lo) & (t.actual <= hi)))
            rec["coverage_garch"] = float(np.mean((t.actual >= t.garch_lo) & (t.actual <= t.garch_hi)))
            rmse = float(np.sqrt(np.mean(t.resid ** 2)))
            rec["rmse_test"] = rmse
            rec["width_ratio"] = (2 * q) / rmse
            rows.append(rec)
    return pd.DataFrame(rows)


def verdict(tab: pd.DataFrame) -> str:
    own_inside = tab.coverage_situation.between(LOW, HIGH).sum()
    garch_inside = tab.coverage_garch.between(LOW, HIGH).sum()
    fallbacks = int((tab.halfwidth_source == "pooled").sum())
    ok = own_inside >= H27_MIN_INSIDE
    cov = tab.coverage_situation.dropna()
    gcov = tab.coverage_garch.dropna()
    return "\n".join([
        f"H27 (situation-specific widths put at least 20 of 25 situations at 75% to 85% coverage): "
        f"{'SUPPORTED' if ok else 'NOT SUPPORTED'}",
        f"  situation-specific: {own_inside} of 25 inside 75% to 85%; coverage range {cov.min():.0%} to {cov.max():.0%}",
        f"  current GARCH bands on the same windows: {garch_inside} of 25 inside; coverage range {gcov.min():.0%} to {gcov.max():.0%}",
        f"  situations that used the pooled width (fewer than {MIN_VAL} validation windows): {fallbacks}",
    ])


def plot(tab: pd.DataFrame, path: Path) -> None:
    plotting.apply_style()
    C = plotting.CATEGORICAL
    d = tab.reset_index(drop=True)
    x = np.arange(len(d))
    fig, ax = plt.subplots(figsize=(13, 5))
    ax.axhspan(LOW, HIGH, color=C[2], alpha=0.15, label="75% to 85% (rule)")
    ax.axhline(TARGET, ls="--", color=plotting.BASELINE)
    ax.scatter(x, d.coverage_garch, color=C[1], s=40, label="current GARCH bands", zorder=3)
    ax.scatter(x, d.coverage_situation, color=C[0], s=40, marker="s", label="situation-specific", zorder=3)
    labels = [f"{a[:6]}\n{b[:9]}" for a, b in zip(d.level, d.trend)]
    ax.set_xticks(x, labels, fontsize=6, rotation=90)
    ax.set_ylabel("Share of test readings inside the 80% band")
    ax.set_title("H27: coverage by situation (each column is one situation)", loc="left", fontsize=10)
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    t0 = time.time()
    out = WEEK / "results"
    figs = WEEK / "figures"
    out.mkdir(parents=True, exist_ok=True)
    figs.mkdir(parents=True, exist_ok=True)
    val, test = build()
    print(f"validation {len(val):,} windows, test {len(test):,} windows ({time.time() - t0:.0f}s)", flush=True)
    tab = situation_table(val, test)
    tab.to_csv(out / "situation_bands.csv", index=False)
    v = verdict(tab)
    (out / "verdict_h27.txt").write_text(v + "\n")
    print(tab.round(3).to_string(index=False), flush=True)
    print(v, flush=True)
    plot(tab, figs / "situation_bands.png")
    print(f"runtime {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
