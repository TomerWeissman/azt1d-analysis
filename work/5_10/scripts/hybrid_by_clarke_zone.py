"""H38: does the combined band (situation x GARCH) recover GARCH's catch rate in Clarke zone D?

Plain models, OhioT1DM and AZT1D, 80% bands: GARCH, situation (Mondrian), combined (situation x
GARCH). Per Clarke zone: share caught and average width. Per-situation coverage error per method.

Rule (both datasets): combined zone D catch >= GARCH zone D catch - 5 points, and combined
per-situation coverage error <= 80% of GARCH's.

Run:    python work/5_10/scripts/hybrid_by_clarke_zone.py
Writes: work/5_10/results/h38_zone_catch.csv, verdict_h38.txt, work/5_10/figures/h38_zone_catch.png
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

from azt1d import plotting  # noqa: E402
from azt1d.glimmer.clinical import clarke_error_grid_zones  # noqa: E402
from verification_checks import frames, mondrian, garch, hybrid, clip, cov_err  # noqa: E402

RUNS = {"OhioT1DM": ("ohiot1dm", "ohiot1dm_cnn_lstm_v0"), "AZT1D": ("azt1d", "cnn_lstm_v0")}
METHODS = {"GARCH": garch, "situation": mondrian, "situation x GARCH": hybrid}
ZONES = ["A", "B", "D"]  # C and E have too few points (under 125 on both datasets)


def main():
    rows, sit_err, verdicts = [], {}, []
    for label, (ds, run) in RUNS.items():
        v, t = frames(ds, run)
        y, p = t.actual.to_numpy(), t.pred.to_numpy()
        zone = clarke_error_grid_zones(y, p)
        for name, fn in METHODS.items():
            lo, hi = fn(t) if name == "GARCH" else fn(v, t)
            lo, hi = clip(lo), clip(hi)
            ins = (y >= lo) & (y <= hi)
            sit_err[(label, name)] = cov_err(t, ins) * 100
            for z in ZONES:
                m = zone == z
                rows.append({"dataset": label, "method": name, "zone": z, "n": int(m.sum()),
                             "caught": float(ins[m].mean()), "width": float((hi - lo)[m].mean())})
        d = pd.DataFrame(rows)
        d = d[d.dataset == label].set_index(["method", "zone"]).caught
        dg, dh = d[("GARCH", "D")], d[("situation x GARCH", "D")]
        ok = dh >= dg - 0.05 and sit_err[(label, "situation x GARCH")] <= 0.8 * sit_err[(label, "GARCH")]
        verdicts.append((label, ok, dg, d[("situation", "D")], dh))
    res = pd.DataFrame(rows)
    res.to_csv(WEEK / "results" / "h38_zone_catch.csv", index=False)
    ok_all = all(v[1] for v in verdicts)
    lines = [f"H38 (combined band recovers zone D catch within 5 points of GARCH, keeps situation honesty): {'SUPPORTED' if ok_all else 'NOT SUPPORTED'}"]
    for label, ok, dg, ds_, dh in verdicts:
        lines.append(f"  {label}: zone D caught GARCH {dg:.0%}, situation {ds_:.0%}, combined {dh:.0%} "
                     f"(gap to GARCH {(dh - dg) * 100:+.0f} pts); per-situation error GARCH {sit_err[(label, 'GARCH')]:.1f}, "
                     f"combined {sit_err[(label, 'situation x GARCH')]:.1f} pts -> {'meets' if ok else 'fails'} the rule")
    for label in RUNS:
        s = res[res.dataset == label]
        for z in ZONES:
            r = s[s.zone == z].set_index("method")
            lines.append(f"  {label} zone {z} (n={int(r.n.iloc[0]):,}): " + ", ".join(
                f"{m} caught {r.caught[m]:.0%} width {r.width[m]:.0f}" for m in METHODS))
    text = "\n".join(lines)
    (WEEK / "results" / "verdict_h38.txt").write_text(text + "\n")
    print(text)

    plotting.apply_style()
    C = plotting.CATEGORICAL
    cols = {"GARCH": plotting.INK_MUTED, "situation": C[0], "situation x GARCH": C[3]}
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
    for ax, label in zip(axes, RUNS):
        s = res[res.dataset == label]
        x = np.arange(len(ZONES)); bw = 0.27
        for j, m in enumerate(METHODS):
            vals = [s[(s.method == m) & (s.zone == z)].caught.iloc[0] * 100 for z in ZONES]
            ax.bar(x + (j - 1) * bw, vals, bw, color=cols[m], label=m)
            for xi, val in zip(x, vals):
                ax.text(xi + (j - 1) * bw, val + 1.5, f"{val:.0f}%", ha="center", fontsize=7.5)
        names = {"A": "Zone A\n(accurate)", "B": "Zone B\n(off, harmless)", "D": "Zone D\n(dangerous miss)"}
        ax.set_xticks(x, [f"{names[z]}\nn={int(s[s.zone == z].n.iloc[0]):,}" for z in ZONES], fontsize=8)
        ax.axhline(80, ls="--", color=plotting.BASELINE)
        ax.set_title(f"{label}: share of points caught by the 80% band, per Clarke zone", loc="left", fontsize=10)
        ax.set_ylim(0, 110)
    axes[0].set_ylabel("Share of true values inside the band (%)")
    axes[0].legend(frameon=False, fontsize=8, loc="upper right")
    fig.suptitle("Does adding GARCH back recover the dangerous misses? (zones C and E omitted: too few points)",
                 x=0.01, ha="left", fontsize=11)
    fig.tight_layout()
    fig.savefig(WEEK / "figures" / "h38_zone_catch.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
