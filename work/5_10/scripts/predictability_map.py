"""H26: a predictability map of glucose situations.

Splits every AZT1D test window into a cell by two things known at forecast time:
  - current glucose level (5 bands: under 70, 70-120, 120-180, 180-250, 250 and over)
  - 30-minute trend: change in glucose over the 30 minutes before the forecast
    (5 bands: fast fall, slow fall, flat, slow rise, fast rise)

For each cell it reports:
  - RMSE of the plain model (saved checkpoints, cnn_lstm_v0)
  - the 80% GARCH band's coverage and mean width
  - a noise upper bound from look-alike windows inside the same cell (nearly identical inputs,
    not from the same patient within 1 hour), as in H23

H26 is supported if the noise upper bound, as a share of the cell's model error, differs at
least twofold between cells. The lowest cell's bound must be at most half the highest cell's.
Cells with fewer than 300 windows are not scored.

No training. Uses the saved checkpoints and the existing band code.

Run:    python work/5_10/scripts/predictability_map.py
Writes: work/5_10/results/predictability_map.csv, work/5_10/results/verdict_h26.txt,
        work/5_10/figures/predictability_map.png
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

from azt1d import loading, plotting, reference as ref  # noqa: E402
from azt1d.glimmer import checkpoint as ckpt, uncertainty as unc  # noqa: E402

RUN = "cnn_lstm_v0"
LEVEL_EDGES = [70, 120, 180, 250]  # inner edges: 5 bins
RATE_EDGES = [-30, -10, 10, 30]  # change over 30 minutes (6 steps): 5 bins
LEVEL_NAMES = ["under 70", "70 to 120", "120 to 180", "180 to 250", "250 and over"]
RATE_NAMES = ["fast fall", "slow fall", "flat", "slow rise", "fast rise"]
TREND_STEPS = 6
MIN_CELL = 300
LOOK_QUERIES = 800
EXCLUDE_STEPS = 12
H26_RATIO = 2.0


def build_rows() -> pd.DataFrame:
    df = loading.load_real_dataset(ROOT / "data" / "raw", ROOT / "data" / "processed")
    rows = []
    for sid in sorted(int(s) for s in df["subject_id"].unique()):
        res = ckpt.load_result(ROOT / "data" / "processed" / "checkpoints" / RUN, sid)
        dfs = df[df["subject_id"] == sid].reset_index(drop=True)
        val, test = unc.forecast_frames(res, dfs)
        eng = unc.BandEngine(val, test, analogs=False)
        lo, hi = eng.bands("GARCH", 0.80)
        times = pd.to_datetime(dfs[ref.EVENT_DATETIME]).to_numpy()
        g = dfs[ref.CGM].to_numpy()
        i = np.searchsorted(times, test.issue_time)
        assert np.all(times[i] == test.issue_time), "issue times do not line up with the raw series"
        n = len(test.actual)
        rows.append(pd.DataFrame({
            "subject_id": sid, "tix": np.arange(n),
            "actual": test.actual, "resid": test.resid, "lo": lo, "hi": hi,
            "g_now": g[i], "g_prev": g[np.maximum(i - TREND_STEPS, 0)],
            "has_history": i >= TREND_STEPS,
            "X": list(test.X.reshape(n, -1).astype(np.float32)),
        }))
    out = pd.concat(rows, ignore_index=True)
    return out[out.has_history].reset_index(drop=True)


def look_alike_noise(X: np.ndarray, y: np.ndarray, pid: np.ndarray, tix: np.ndarray, rng) -> float:
    """Half the mean squared gap between nearest look-alike pairs, as an upper bound on noise variance."""
    sq = (X ** 2).sum(1)
    q = rng.choice(len(y), min(LOOK_QUERIES, len(y)), replace=False)
    D = sq[q][:, None] + sq[None, :] - 2 * X[q] @ X.T
    same = (pid[q][:, None] == pid[None, :]) & (np.abs(tix[q][:, None] - tix[None, :]) <= EXCLUDE_STEPS)
    D[same] = np.inf
    D[np.arange(len(q)), q] = np.inf
    nn = D.argmin(1)
    d = y[q] - y[nn]
    return float(np.mean(d ** 2) / 2)


def cell_table(rows: pd.DataFrame) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    rows = rows.copy()
    rows["lvl"] = np.digitize(rows.g_now.to_numpy(), LEVEL_EDGES)
    rows["trd"] = np.digitize((rows.g_now - rows.g_prev).to_numpy(), RATE_EDGES)
    rows["cell"] = rows.lvl * 5 + rows.trd
    out = []
    for lvl in range(5):
        for trd in range(5):
            c = lvl * 5 + trd
            sub = rows[rows.cell == c]
            n = len(sub)
            rec = {"level": LEVEL_NAMES[lvl], "trend": RATE_NAMES[trd], "n": n}
            if n == 0:
                out.append(rec); continue
            mse = float(np.mean(sub.resid ** 2))
            rec.update({
                "rmse": float(np.sqrt(mse)),
                "coverage_80_garch": float(np.mean((sub.actual >= sub.lo) & (sub.actual <= sub.hi))),
                "mean_band_width": float(np.mean(sub.hi - sub.lo)),
                "model_mse": mse,
            })
            if n >= MIN_CELL:
                X = np.stack(sub.X.to_numpy()).astype(np.float32)
                noise_up = look_alike_noise(X, sub.actual.to_numpy(), sub.subject_id.to_numpy(),
                                            sub.tix.to_numpy(), rng)
                rec["noise_upper_bound"] = noise_up
                rec["noise_share_upper"] = noise_up / mse
            out.append(rec)
    return pd.DataFrame(out)


def verdict(tab: pd.DataFrame) -> str:
    scored = tab.dropna(subset=["noise_share_upper"])
    if len(scored) < 2:
        return "H26: not scored (fewer than two cells with at least 300 windows)"
    lo_cell = scored.loc[scored.noise_share_upper.idxmin()]
    hi_cell = scored.loc[scored.noise_share_upper.idxmax()]
    ok = lo_cell.noise_share_upper <= hi_cell.noise_share_upper / H26_RATIO
    return "\n".join([
        f"H26 (noise share differs at least twofold between situations): {'SUPPORTED' if ok else 'NOT SUPPORTED'}",
        f"  scored cells: {len(scored)} of 25",
        f"  lowest noise upper bound: {lo_cell.noise_share_upper:.0%} ({lo_cell.level}, {lo_cell.trend}, n={lo_cell.n})",
        f"  highest noise upper bound: {hi_cell.noise_share_upper:.0%} ({hi_cell.level}, {hi_cell.trend}, n={hi_cell.n})",
        f"  rule: lowest at most half the highest. Upper bounds only: a low bound is real evidence of low noise,"
        f" a high bound is not evidence of high noise.",
    ])


def heatmap(tab: pd.DataFrame, path: Path) -> None:
    plotting.apply_style()
    rmse = np.full((5, 5), np.nan); share = np.full((5, 5), np.nan); ns = np.zeros((5, 5), int)
    for _, r in tab.iterrows():
        a, b = LEVEL_NAMES.index(r.level), RATE_NAMES.index(r.trend)
        ns[a, b] = r.n
        if pd.notna(r.get("rmse")):
            rmse[a, b] = r.rmse
        if pd.notna(r.get("noise_share_upper")):
            share[a, b] = r.noise_share_upper
    fig, (a, b) = plt.subplots(1, 2, figsize=(13, 5.4))
    for ax, data, title, cmap, fmt in [
        (a, rmse, "Model RMSE per situation (mg/dL)", "YlOrRd", "{:.0f}"),
        (b, share, "Noise upper bound as share of model error", "Blues", "{:.0%}"),
    ]:
        im = ax.imshow(np.ma.masked_invalid(data), cmap=cmap, aspect="auto")
        ax.set_xticks(range(5), RATE_NAMES, rotation=25, fontsize=8)
        ax.set_yticks(range(5), LEVEL_NAMES, fontsize=8)
        ax.set_xlabel("30-minute trend")
        ax.set_ylabel("Current glucose (mg/dL)")
        ax.set_title(title, loc="left", fontsize=10)
        for i in range(5):
            for j in range(5):
                if np.isnan(data[i, j]):
                    txt = f"n={ns[i, j]}\n(not scored)" if ns[i, j] else "no data"
                else:
                    txt = fmt.format(data[i, j]) + f"\nn={ns[i, j]}"
                mid = (np.nanmin(data) + np.nanmax(data)) / 2
                dark = (not np.isnan(data[i, j])) and data[i, j] > mid
                ax.text(j, i, txt, ha="center", va="center", fontsize=7, color="white" if dark else "black")
        fig.colorbar(im, ax=ax, shrink=0.8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    t0 = time.time()
    out = WEEK / "results"
    figs = WEEK / "figures"
    out.mkdir(parents=True, exist_ok=True)
    figs.mkdir(parents=True, exist_ok=True)
    rows = build_rows()
    print(f"{len(rows):,} test windows with 30-minute history ({time.time() - t0:.0f}s)", flush=True)
    tab = cell_table(rows)
    tab.to_csv(out / "predictability_map.csv", index=False)
    v = verdict(tab)
    (out / "verdict_h26.txt").write_text(v + "\n")
    print(tab.round(3).to_string(index=False), flush=True)
    print(v, flush=True)
    heatmap(tab, figs / "predictability_map.png")
    print(f"runtime {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
