"""Builds notebooks_2/01_uncertainty_bands.ipynb (stage 2 of the project).

Run this, then execute the result with:
  jupyter nbconvert --to notebook --execute --inplace notebooks_2/01_uncertainty_bands.ipynb

The conclusions are written after inspecting real output; edit CONCLUSION below and
rebuild + re-execute (the whole notebook runs in a couple of minutes).
"""
import uuid
from pathlib import Path

import nbformat
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook

ROOT = Path(__file__).resolve().parent.parent


def md(text):
    c = new_markdown_cell(text)
    c["id"] = uuid.uuid4().hex[:8]
    return c


def code(text):
    c = new_code_cell(text.strip("\n"))
    c["id"] = uuid.uuid4().hex[:8]
    return c


CONCLUSION = """## 7. What this shows

All numbers pool the 25 patients' test periods (about 60,000 predictions: 963 real lows, 11,525 real highs, 47,521 safe readings). Nothing was fitted on the test period. This is one model (the fixed-weight CNN-LSTM) on one dataset (AZT1D), so it is a first look, not a final answer.

**Under the rule you asked about, ordinary bands would cry wolf almost constantly.** Triggering whenever either edge of an 80% band leaves the 70 to 180 zone falsely triggers on 86% of safe readings for conformal, 80% for GARCH, and 68% for the analog ensemble. At 95% it is 100%, 94%, and 93%. Roughly three out of four alarms would be false. The reason is simple: the safe zone is only 110 mg/dL wide and these bands are about that wide (or wider) on their own, so they leave the zone whether or not anything dangerous is happening. Danger is caught almost every time (92% to 100%), but that is easy when nearly everything triggers.

**There is a floor under the false trigger rate that no band can get below.** The raw model forecast, used as an alarm with no band at all, already falsely triggers on 32% of safe readings (and catches 72% of real danger, including only 45% of real lows). A band only ever adds room on either side of its center, so conformal, which is centered on the raw forecast, cannot do better than 32%. GARCH and the analog ensemble also shift their center to correct for the model's usual bias, which lowers their floors to 22% and 14%, but at the cost of catching less danger at that floor (61% and 53%). This is why the methods are compared at 35% and 50% false triggers rather than 10% or 25%: lower rates are not reachable for all three.

**The false triggers come mostly from the top edge.** At a 35% false trigger rate, 26% to 30% of safe readings falsely trigger through the top edge (above 180) and only 6% to 9% through the bottom edge (below 70). The model tends to forecast high, so its forecast and its top edge sit above 180 while real glucose is still in range. Subject 1 shows it: the forecast hovers near 190 for hours while real glucose is 150 to 170, and the alarm fires the whole time.

**At the same false trigger rate, the three methods are close to indistinguishable.** At 35% false triggers they catch 74.0% (conformal), 75.5% (GARCH), and 76.5% (analog) of real danger. At 50% it is 84.2%, 84.6%, and 84.8%. The curves in section 4 lie almost on top of each other. The differences that exist are small and go different ways: conformal catches the most real lows at 35% (49% against 46% and 38%), and the analog ensemble catches the most real highs (80% against 78% and 76%). No method is a clear winner.

**A band adds little over just using the forecast.** The raw forecast catches 72.2% of danger at 32% false triggers. Conformal at 35% catches 74.0%. So about 3 extra points of false triggers buys about 2 extra points of danger caught. Getting to roughly 85% of danger caught costs a 50% false trigger rate, meaning half of all safe readings would set off the alarm.

**Following the alarms on the Clarke Error Grid.** Replacing the forecast with the edge that crossed the line makes the dangerous miss (zone D) rarer, and it costs accuracy elsewhere. The raw forecast has 1.33% in zone D. At a 35% false trigger rate this falls to 1.24% (conformal), 0.95% (GARCH), and 1.02% (analog), a small gain. At 50% it falls to 0.81%, 0.62%, and 0.70%, roughly a 40% to 55% cut. The price is zone C (a prediction far enough above the truth to prompt an unneeded correction), which rises from 0.25% to 1.6% to 3.1% at 50%, and zone A (clinically fine), which drops from 51.5% to about 43% to 44%. The worst-case zone E also rises, from 0.21% to 0.41% to 0.66%. Between 43% and 57% of predictions get replaced by an edge, depending on the setting.

**The same band size gives very different false trigger rates for different patients.** At the band size that gives 35% overall, individual patients range from about 0% to 87% (conformal), 7% to 70% (GARCH), and 8% to 70% (analog), and 3 to 6 of the 25 patients are above 50%. At the 50% setting, 11 to 13 of the 25 patients are above 50% and the worst patient is at 84% to 97%. A single band size for everyone would leave some patients with an alarm that fires nearly all the time.

**Caveats.** The bands are calibrated on each patient's validation period and judged on a later test period, so any drift shows up as missed danger or extra false triggers. Consecutive predictions overlap heavily, which means conformal's usual guarantee does not strictly apply. The analog settings (50 analogs, spacing, time-of-day weight) were not tuned. Danger here includes highs over 180, which are less urgent than lows; a stricter definition (lows only, or under 54) would change the numbers.

**Next step this points to.** Because the false triggers come mostly from the top edge and the real emergency is the low side, the obvious follow-up is to size the two edges separately: a wide bottom edge to catch lows, and a narrow (or no) top edge so the model's habit of forecasting high stops setting off the alarm."""

cells = []

cells.append(md('''# How often would an error band set off a false alarm?

Stage 1 of this project (the notebooks in `notebooks/`) compared glucose forecasters. This notebook takes one of them, the **GLIMMER fixed-weight model** (a CNN-LSTM trained to care more about dangerous highs and lows), and asks a practical question about its 60-minute-ahead predictions:

> Suppose we put an error band around every prediction, and raised an alarm whenever **either edge of the band went past the danger zone**: the bottom edge under 70 mg/dL, or the top edge over 180 mg/dL. How often would that alarm go off for no reason?

Some terms, so the numbers below are unambiguous:

- **Danger** means the real glucose was under 70 or over 180 mg/dL. **Safe** means it was between 70 and 180.
- A **trigger** is an alarm: the band's bottom edge is under 70, or its top edge is over 180.
- A **false trigger** is a trigger when the real glucose was actually safe.
- The headline number is the **false trigger rate**: of all the readings where real glucose was safe, the share that set off the alarm.
- The other side of the trade is **danger caught**: of all the readings where real glucose really was dangerous, the share that set off the alarm.

A band that is wide will trigger more, catching more real danger but also falsely triggering more. So the question is not a single number but a trade-off, and comparing methods fairly means comparing them at the same false trigger rate.

Three ways of building the band are compared, each borrowed from a different field:

- **Conformal (basic).** Look at how wrong the model was on data it was not trained on, find the error size that covers the chosen share of cases, and use that as a fixed plus-or-minus around every prediction. It works for any model and is the simplest, so it is the reference point.
- **GARCH (finance).** Built to track volatility. It widens the band when the model's recent errors have been large and narrows it when they have been small. A 60-minute forecast's error is only known 60 minutes later, so it can only learn from errors that have already played out.
- **Analog Ensemble (weather).** Find the 50 past moments that looked most like right now (similar last two hours of glucose, insulin, and carbs, at a similar time of day), look at how wrong the model was in each, and add that spread of errors to today's prediction. The band can lean to one side.

Each method is fitted only on a patient's **validation period** (data the model did not train on, from before the test period) and judged on the **test period**, which nothing was fitted on. Nothing is retrained. All 25 AZT1D patients are used.'''))

cells.append(code('''
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

PROJECT_ROOT = Path.cwd().parent if Path.cwd().name == "notebooks_2" else Path.cwd()
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from azt1d import loading, plotting
from azt1d import reference as ref
from azt1d.glimmer import checkpoint as ckpt
from azt1d.glimmer import uncertainty as unc
from azt1d.glimmer.clinical import clarke_error_grid_zones, clarke_zone_percentages, draw_clarke_grid

plotting.apply_style()
pd.set_option("display.max_columns", None)

METHODS = unc.METHODS
METHOD_COLOR = dict(zip(METHODS, plotting.CATEGORICAL[:3]))
CKPT_DIR = PROJECT_ROOT / "data" / "processed" / "checkpoints" / "cnn_lstm_v1"
TARGETS = (0.35, 0.50)  # false trigger rates to compare the methods at (see the note in section 4 on why not lower)
'''))

cells.append(md('''## 1. Rebuild the forecasts and check the setup

The checkpoints only saved each patient's test predictions, and the methods need validation-period predictions too, so both are rebuilt from the saved model weights. Then a few checks on things that would quietly ruin the comparison if they were wrong.'''))

cells.append(code('''
df = loading.load_real_dataset(PROJECT_ROOT / "data" / "raw", PROJECT_ROOT / "data" / "processed")
subject_ids = sorted(int(s) for s in df["subject_id"].unique())

frames, results, gaps = {}, {}, {}
for sid in subject_ids:
    res = ckpt.load_result(CKPT_DIR, sid)
    df_subject = df[df["subject_id"] == sid].reset_index(drop=True)
    val, test = unc.forecast_frames(res, df_subject)
    frames[sid], results[sid] = (val, test), res
    gaps[sid] = unc.matches_stored_predictions(test, res)

print(f"Rebuilt forecasts for {len(frames)} patients.")
print(f"Largest gap between rebuilt and originally saved test predictions: {max(gaps.values()):.6f} mg/dL")
print(f"Validation windows per patient: {min(len(f[0].pred) for f in frames.values())} to {max(len(f[0].pred) for f in frames.values())}")
print(f"Test windows per patient:       {min(len(f[1].pred) for f in frames.values())} to {max(len(f[1].pred) for f in frames.values())}")
'''))

cells.append(code('''
# 1. Rebuilt predictions match what was saved at training time.
assert max(gaps.values()) < 0.05, "rebuilt predictions do not match the checkpoints"

# 2. GARCH never peeks. Changing residuals that had not happened yet (target time after the
#    moment the forecast was issued) must not change the variance used for that forecast.
val1, test1 = frames[1]
params = unc.fit_garch(val1.resid)
e = np.concatenate([val1.resid, test1.resid])[:300]
e_changed = e.copy()
e_changed[150:] += 1000.0
init_var = float(np.var(val1.resid - params.mu))  # from calibration data only
v_a = unc.garch_h_step_variance(e, params, init_var)
v_b = unc.garch_h_step_variance(e_changed, params, init_var)
lag = unc.LAG
assert np.allclose(v_a[lag:150 + lag], v_b[lag:150 + lag]), "GARCH variance changed because of a future residual"
assert not np.allclose(v_a[150 + lag:], v_b[150 + lag:]), "changed residuals should matter once they are actually known"

# 3. Conformal quantile matches a hand calculation: residual sizes 1..100, 90% band -> exactly 91.
assert unc.conformal_halfwidth(np.arange(1, 101, dtype=float), alpha=0.1) == 91.0

# 4. Calibration data always comes before the test period, and analogs are spread out.
for sid, (v, t) in frames.items():
    assert v.target_time.max() < t.target_time.min(), "calibration period overlaps the test period"
picked = unc.select_analogs(np.random.default_rng(0).random(1735), k=50, min_gap=12, pool_cap=600)
assert len(picked) == 50 and np.diff(np.sort(picked)).min() >= 12

# 5. The trigger arithmetic, on a hand-made example.
#    readings: real 50 (danger), 100 (safe), 100 (safe), 200 (danger)
#    triggers: low edge 60<70 fires, low edge 60<70 fires, nothing fires, high edge 250>180 fires
a = np.array([50.0, 100.0, 100.0, 200.0])
lo = np.array([60.0, 60.0, 90.0, 150.0])
hi = np.array([100.0, 200.0, 120.0, 250.0])
m = unc.trigger_metrics(a, lo, hi)
assert m["false_trigger_rate"] == 0.5 and m["danger_caught"] == 1.0
#    when both edges cross, the bigger overshoot wins (30 vs 20 -> low edge; 10 vs 120 -> high edge)
sw = unc.swapped_prediction(np.array([100.0, 100.0]), np.array([40.0, 60.0]), np.array([200.0, 300.0]))
assert list(sw) == [40.0, 300.0]

print("All checks passed.")
'''))

cells.append(md('''## 2. Build the bands

Each patient gets one engine holding all three methods, so a band of any size can be produced quickly. The GARCH numbers below are a curiosity: alpha is how strongly one big error widens the next band, and beta is how long that widening lingers.'''))

cells.append(code('''
engines = {sid: unc.BandEngine(frames[sid][0], frames[sid][1]) for sid in subject_ids}
actuals = {sid: frames[sid][1].actual for sid in subject_ids}
actual_all = np.concatenate([actuals[s] for s in subject_ids])
pred_all = np.concatenate([frames[s][1].pred for s in subject_ids])


def pooled_bands(method, level):
    """One method's band at one size, all 25 patients' test periods joined end to end."""
    los, his = zip(*(engines[s].bands(method, level) for s in subject_ids))
    return np.concatenate(los), np.concatenate(his)


region_all = unc.region_of(actual_all)
print("Test readings by what real glucose was:", {r: int((region_all == r).sum()) for r in ("hypo", "normal", "hyper")})
garch_p = pd.DataFrame([{"alpha": e.garch_params.alpha, "beta": e.garch_params.beta} for e in engines.values()])
garch_p["alpha_plus_beta"] = garch_p.alpha + garch_p.beta
garch_p.describe().loc[["mean", "min", "max"]].round(3)
'''))

cells.append(md('''## 3. The rule, applied to ordinary 80% and 95% bands

First the obvious version: build the usual 80% and 95% bands and trigger whenever either edge leaves the safe zone. For comparison, the last row uses the model's point forecast alone as the alarm (no band: trigger when the forecast itself is under 70 or over 180).

"Share of alarms that were false" is a different way to look at the same thing: when an alarm goes off, how often is it wrong.'''))

cells.append(code('''
SWEEP = np.round(np.concatenate([np.arange(0.05, 0.5, 0.05), np.arange(0.5, 0.9901, 0.01)]), 2)

sweep_rows = []
for method in METHODS:
    for level in SWEEP:
        lo, hi = pooled_bands(method, level)
        sweep_rows.append({"method": method, "level": float(level), **unc.trigger_metrics(actual_all, lo, hi)})
sweep = pd.DataFrame(sweep_rows)

# Wider bands should only ever trigger more.
for method in METHODS:
    rates = sweep[sweep.method == method].sort_values("level")["false_trigger_rate"].to_numpy()
    assert (np.diff(rates) >= -1e-9).all(), f"{method}: false trigger rate fell as the band got wider"

point = unc.trigger_metrics(actual_all, pred_all, pred_all)
cols = {"false_trigger_rate": "false trigger rate (of safe readings)", "danger_caught": "danger caught",
        "lows_caught": "real lows caught", "highs_caught": "real highs caught", "share_of_alarms_false": "share of alarms that were false"}
rows = []
for method in METHODS:
    for level in (0.8, 0.95):
        r = sweep[(sweep.method == method) & np.isclose(sweep.level, level)].iloc[0]
        rows.append({"alarm source": method, "band": f"{int(level * 100)}%", **{cols[k]: r[k] for k in cols}})
rows.append({"alarm source": "Point forecast (no band)", "band": "-", **{cols[k]: point[k] for k in cols}})
pd.DataFrame(rows).set_index(["alarm source", "band"]).round(3)
'''))

cells.append(md('''## 4. The trade-off, and a fair comparison

Wider bands catch more real danger and falsely trigger more, so the three methods cannot be compared at the same band size. The curves below plot, for every band size from 5% to 99%, how much real danger is caught against how often the alarm falsely triggers. A curve that sits higher for the same false trigger rate is a genuinely better alarm. The black dot is the point forecast used as an alarm on its own.

The three panels split "danger caught" into all danger, real lows only, and real highs only. The horizontal axis is always the overall false trigger rate. The dashed lines mark the two false trigger rates compared in the rest of the notebook (35% and 50%). They are not lower because, as the next table shows, no band can push a method below the false trigger rate of its own forecast line.'''))

cells.append(code('''
fig, axes = plt.subplots(1, 3, figsize=(16, 4.8), sharex=True, sharey=True)
panels = [("danger_caught", "Real danger caught (lows and highs)"), ("lows_caught", "Real lows caught"), ("highs_caught", "Real highs caught")]
for ax, (col, title) in zip(axes, panels):
    for method in METHODS:
        s = sweep[sweep.method == method].sort_values("level")
        ax.plot(s.false_trigger_rate, s[col], color=METHOD_COLOR[method], linewidth=1.8, label=method)
        for lv in (0.8, 0.95):
            r = s[np.isclose(s.level, lv)].iloc[0]
            ax.scatter([r.false_trigger_rate], [r[col]], color=METHOD_COLOR[method], s=28, zorder=3)
    ax.scatter([point["false_trigger_rate"]], [point[col]], color=plotting.INK_PRIMARY, s=60, zorder=4, label="Point forecast alone")
    for t in TARGETS:
        ax.axvline(t, linestyle="--", color=plotting.BASELINE, linewidth=1)
    ax.set_title(title)
    ax.set_xlabel("False trigger rate (share of safe readings that set off the alarm)")
axes[0].set_ylabel("Share of real danger that set off the alarm")
axes[0].legend(frameon=False, fontsize=8, loc="lower right")
axes[0].annotate("dots on each curve mark the 80% and 95% bands", (0.02, 0.03), xycoords="axes fraction", fontsize=8, color=plotting.INK_MUTED)
fig.tight_layout()
plt.show()
'''))

cells.append(md('''**A floor on false triggers.** A band is centered on a forecast and only ever adds room on either side of it, so making a band wider can only add triggers, never remove them. That means every method has a lowest possible false trigger rate: the rate you get with a band of essentially zero width, where the alarm is just the method's own center line crossing 70 or 180. Conformal is centered on the raw model forecast, while GARCH and the analog ensemble also shift the center to correct for the model's typical bias, which is why their floors are lower.'''))

cells.append(code('''
floor_rows = [{"alarm source": "Point forecast (raw model)", "false trigger rate": point["false_trigger_rate"], "danger caught": point["danger_caught"]}]
for method in METHODS:
    lo, hi = pooled_bands(method, 0.001)  # a band of essentially zero width
    tm = unc.trigger_metrics(actual_all, lo, hi)
    floor_rows.append({"alarm source": f"{method}, zero-width band", "false trigger rate": tm["false_trigger_rate"], "danger caught": tm["danger_caught"]})
pd.DataFrame(floor_rows).set_index("alarm source").round(3)
'''))

cells.append(md('''Now the same comparison in numbers, at a matched false trigger rate. For each method, the band size is tuned until it falsely triggers on exactly 35% of safe readings (and again on 50%). Then everything else about the alarm is compared at that setting. "False via low edge" and "false via high edge" split the false triggers by which edge caused them.'''))

cells.append(code('''
matched = {(m, t): unc.level_for_false_trigger_rate(engines, actuals, m, t) for m in METHODS for t in TARGETS}

rows = []
for t in TARGETS:
    for method in METHODS:
        lvl = matched[(method, t)]
        lo, hi = pooled_bands(method, lvl)
        tm = unc.trigger_metrics(actual_all, lo, hi)
        assert abs(tm["false_trigger_rate"] - t) < 0.003, "matching missed its target"
        rows.append({
            "false trigger target": f"{int(t * 100)}%", "method": method, "band size needed": lvl,
            "false trigger rate": tm["false_trigger_rate"], "danger caught": tm["danger_caught"],
            "real lows caught": tm["lows_caught"], "real highs caught": tm["highs_caught"],
            "share of alarms that were false": tm["share_of_alarms_false"],
            "false via low edge": tm["false_via_low_edge"], "false via high edge": tm["false_via_high_edge"],
        })
matched_table = pd.DataFrame(rows).set_index(["false trigger target", "method"])
print("For reference, the point forecast alone falsely triggers on "
      f"{point['false_trigger_rate']:.1%} of safe readings and catches {point['danger_caught']:.1%} of real danger "
      f"({point['lows_caught']:.1%} of lows, {point['highs_caught']:.1%} of highs).")
matched_table.round(3)
'''))

cells.append(md('''## 5. Where the alarms fire

Each figure is a 2-day stretch of one patient's held-out test data, centered on that patient's lowest real glucose reading. The black line is real glucose, the colored line is the prediction, and the shaded band is the method's band at the size that gives a 35% false trigger rate. The dashed lines mark 70 and 180 mg/dL. Along the bottom, each tick is a moment the alarm went off: the upper row of ticks are **correct** triggers (real glucose was in danger) and the lower row are **false** triggers (real glucose was safe).

Subject 1 is the project's usual example. The best and worst patients are picked automatically from the model's own test error.'''))

cells.append(code('''
rmse = {sid: results[sid].rmse for sid in subject_ids}
best_sid, worst_sid = min(rmse, key=rmse.get), max(rmse, key=rmse.get)
show_sids = list(dict.fromkeys([1, best_sid, worst_sid]))
print(f"Subjects shown: 1 (usual example), {best_sid} (best, RMSE {rmse[best_sid]:.1f}), {worst_sid} (worst, RMSE {rmse[worst_sid]:.1f})")


def plot_triggers(sid, target=TARGETS[0], half_window=144):
    val, test = frames[sid]
    center = int(np.argmin(test.actual))
    a, b = max(0, center - half_window), min(len(test.pred), center + half_window)
    x = pd.to_datetime(test.target_time[a:b])
    danger = (test.actual < ref.HYPO_THRESHOLD) | (test.actual > ref.HYPER_THRESHOLD)
    fig, axes = plt.subplots(3, 1, figsize=(12, 9.5), sharex=True, sharey=True)
    for ax, method in zip(axes, METHODS):
        color = METHOD_COLOR[method]
        lo, hi = engines[sid].bands(method, matched[(method, target)])
        trig, _, _ = unc.trigger_flags(lo, hi)
        ax.fill_between(x, lo[a:b], hi[a:b], color=color, alpha=0.25, linewidth=0)
        ax.plot(x, test.pred[a:b], color=color, linewidth=1.2)
        ax.plot(x, test.actual[a:b], color=plotting.INK_PRIMARY, linewidth=1.3)
        ax.axhline(ref.HYPO_THRESHOLD, linestyle="--", color=plotting.GLUCOSE_BAND_COLORS["hypo"], linewidth=1)
        ax.axhline(ref.HYPER_THRESHOLD, linestyle="--", color=plotting.GLUCOSE_BAND_COLORS["hyper"], linewidth=1)
        good = trig[a:b] & danger[a:b]
        false = trig[a:b] & ~danger[a:b]
        ax.vlines(x[good], 60, 78, color=plotting.INK_PRIMARY, linewidth=1.2)
        ax.vlines(x[false], 36, 54, color=plotting.INK_SECONDARY, linewidth=1.2, alpha=0.6)
        ax.annotate("correct trigger", xy=(1.005, 69), xycoords=("axes fraction", "data"), fontsize=7, color=plotting.INK_SECONDARY, va="center", annotation_clip=False)
        ax.annotate("false trigger", xy=(1.005, 45), xycoords=("axes fraction", "data"), fontsize=7, color=plotting.INK_SECONDARY, va="center", annotation_clip=False)
        ax.set_ylim(30, 330)
        ax.set_ylabel("Glucose (mg/dL)")
        ax.set_title(f"{method}: band sized for a {int(target * 100)}% false trigger rate (a {matched[(method, target)]:.0%} band)", loc="left", fontsize=10)
    fig.suptitle(f"Subject {sid}: where the alarm would have gone off")
    fig.tight_layout(rect=[0, 0, 0.93, 0.97])
    plt.show()


for sid in show_sids:
    plot_triggers(sid)
'''))

cells.append(md('''## 6. What the Clarke Error Grid looks like if we follow the alarms

The Clarke Error Grid grades a prediction by what a person would do differently because of the error: zone A is clinically fine, zone B is a benign error, zone C means the prediction was wrong enough to prompt an unneeded correction, zone D means a real dangerous moment was predicted as safe, and zone E means the prediction pointed the opposite way from the truth.

To grade an alarm rule on that grid, each prediction is replaced by what the alarm would have reported:

- If neither band edge crosses into a danger zone, the model's own forecast is kept.
- If the bottom edge drops under 70, the reported value is that bottom edge.
- If the top edge goes over 180, the reported value is that top edge.
- If both cross, the edge that crosses its line by more is used.

The result is the grid you would get by acting on the alarms instead of on the raw forecast. It is shown at the two matched false trigger rates (35% and 50%), so the three methods are compared on equal terms. The first row of the table is the model's raw forecast with no alarms, for reference.'''))

cells.append(code('''
zones = ["A", "B", "C", "D", "E"]
clarke_rows = [{"scenario": "Model forecast alone", "false trigger target": "-", **clarke_zone_percentages(actual_all, pred_all), "replaced by an edge": 0.0}]
swapped = {}
for t in TARGETS:
    for method in METHODS:
        lo, hi = pooled_bands(method, matched[(method, t)])
        sw = unc.swapped_prediction(pred_all, lo, hi)
        swapped[(method, t)] = sw
        trig, _, _ = unc.trigger_flags(lo, hi)
        clarke_rows.append({"scenario": method, "false trigger target": f"{int(t * 100)}%", **clarke_zone_percentages(actual_all, sw), "replaced by an edge": float(trig.mean())})
clarke_table = pd.DataFrame(clarke_rows).set_index(["scenario", "false trigger target"])
clarke_table.round(2)
'''))

cells.append(code('''
base = clarke_zone_percentages(actual_all, pred_all)
fig, axes = plt.subplots(1, 2, figsize=(13, 4.4), sharey=True)
for ax, t in zip(axes, TARGETS):
    x = np.arange(len(zones))
    w = 0.8 / (len(METHODS) + 1)
    ax.bar(x - 1.5 * w, [base[z] for z in zones], w, color=plotting.INK_MUTED, label="Model forecast alone")
    for i, method in enumerate(METHODS):
        pct = clarke_zone_percentages(actual_all, swapped[(method, t)])
        ax.bar(x + (i - 0.5) * w, [pct[z] for z in zones], w, color=METHOD_COLOR[method], label=method)
    ax.set_xticks(x)
    ax.set_xticklabels([f"Zone {z}" for z in zones])
    ax.set_title(f"Following the alarms at a {int(t * 100)}% false trigger rate")
axes[0].set_ylabel("% of predictions")
axes[0].legend(frameon=False, fontsize=8)
fig.tight_layout()
plt.show()
'''))

cells.append(code('''
rng = np.random.default_rng(0)
sample = rng.choice(len(actual_all), 6000, replace=False)
fig, axes = plt.subplots(2, 4, figsize=(19, 10))
for r, t in enumerate(TARGETS):
    draw_clarke_grid(axes[r, 0], actual_all[sample], pred_all[sample], plotting.INK_MUTED)
    axes[r, 0].set_title("Model forecast alone")
    for c, method in enumerate(METHODS, start=1):
        draw_clarke_grid(axes[r, c], actual_all[sample], swapped[(method, t)][sample], METHOD_COLOR[method])
        axes[r, c].set_title(f"{method}, following alarms ({int(t * 100)}% false triggers)")
fig.tight_layout()
plt.show()
'''))

cells.append(md('''The false trigger rate above is pooled across all 25 patients. Below is the same thing for each patient separately, at the band size that gives the pooled 35% and 50% rates. If the dots are tightly grouped, one band size works for everyone. If they spread out, some patients would be getting far more false alarms than the pooled number suggests.'''))

cells.append(code('''
per_patient_rows = []
fig, axes = plt.subplots(1, 2, figsize=(13, 4.2), sharey=True)
for ax, t in zip(axes, TARGETS):
    for i, method in enumerate(METHODS):
        vals = []
        for sid in subject_ids:
            lo, hi = engines[sid].bands(method, matched[(method, t)])
            vals.append(unc.trigger_metrics(actuals[sid], lo, hi)["false_trigger_rate"])
        per_patient_rows.append({"false trigger target": f"{int(t * 100)}%", "method": method, "lowest patient": min(vals),
                                 "median patient": float(np.median(vals)), "highest patient": max(vals),
                                 "patients above 50%": int(sum(v > 0.5 for v in vals))})
        jitter = np.random.default_rng(i).normal(0, 0.05, len(vals))
        ax.scatter(np.full(len(vals), i) + jitter, vals, color=METHOD_COLOR[method], s=24, alpha=0.85)
        ax.hlines(np.median(vals), i - 0.25, i + 0.25, color=plotting.INK_PRIMARY, linewidth=1.5)
    ax.axhline(t, linestyle="--", color=plotting.INK_MUTED, linewidth=1)
    ax.set_xticks(range(len(METHODS)))
    ax.set_xticklabels(METHODS)
    ax.set_title(f"Target {int(t * 100)}%: false trigger rate for each of the 25 patients (line = median)")
axes[0].set_ylabel("False trigger rate")
fig.tight_layout()
plt.show()
display(pd.DataFrame(per_patient_rows).set_index(["false trigger target", "method"]).round(3))
'''))

cells.append(md(CONCLUSION))

cells.append(md('''## Appendix: the earlier coverage and Clarke material

These sections come from the first version of this notebook, which asked a different question: how well do the bands contain the real glucose, and how do the band edges look when graded as predictions. They use the ordinary 80% and 95% bands.

**Coverage** is the share of test readings where real glucose landed inside the band. A 95% band should contain the truth 95% of the time. The table splits it by what real glucose actually was. "Interval score" combines band width and misses into one number (lower is better).'''))

cells.append(code('''
LEVELS = (0.8, 0.95)
pooled = {(m, lv): (actual_all, *pooled_bands(m, lv)) for m in METHODS for lv in LEVELS}
rows = []
for (method, level), (actual, lo, hi) in pooled.items():
    m = unc.interval_metrics(actual, lo, hi, level)
    per_patient = [unc.interval_metrics(actuals[s], *engines[s].bands(method, level), level)["coverage"] for s in subject_ids]
    rows.append({"method": method, "band": f"{int(level * 100)}%", "coverage": m["coverage"],
                 "coverage_when_real_low": m["coverage_hypo"], "coverage_when_real_normal": m["coverage_normal"],
                 "coverage_when_real_high": m["coverage_hyper"], "mean_width": m["width"], "interval_score": m["interval_score"],
                 "worst_patient_coverage": min(per_patient), "best_patient_coverage": max(per_patient)})
pd.DataFrame(rows).set_index(["band", "method"]).sort_index().round(3)
'''))

cells.append(md('''**Band edges graded as predictions on the Clarke grid.** Each band's bottom edge and top edge is treated as if it were a prediction against the real glucose. A wide band pushes its edges away from the truth by design, so many edge predictions land outside zone A. Bottom edges below 40 mg/dL are clipped to 40, the sensor floor.'''))

cells.append(code('''
zone_rows = [{"what is graded": "Point forecast (no band)", "band": "-", **clarke_zone_percentages(actual_all, pred_all)}]
for (method, level), (actual, lo, hi) in pooled.items():
    z = unc.band_edge_zones(actual, lo, hi)
    for edge in ("lower", "upper"):
        zone_rows.append({"what is graded": f"{method}, {edge} edge", "band": f"{int(level * 100)}%", **z[edge]})
pd.DataFrame(zone_rows).set_index(["what is graded", "band"]).round(2)
'''))

cells.append(md('''**Does a wider band warn about a forecast that is about to be clinically wrong?** Below, the point forecast is graded on the Clarke grid as usual and the average band width is compared across zones. The warning score is the chance that a randomly picked clinically bad forecast (zone C, D, or E) has a wider band than a randomly picked fine one (zone A or B): 0.5 means the width tells you nothing, 1.0 means it flags every bad forecast. Conformal has a constant width, so it should sit near 0.5 (it drifts slightly because bands are clipped at the sensor limits).'''))

cells.append(code('''
width_rows, auc_rows = [], []
for method in METHODS:
    for level in LEVELS:
        actual, lo, hi = pooled[(method, level)]
        width_rows.append({"method": method, "band": f"{int(level * 100)}%", **unc.width_by_point_zone(actual_all, pred_all, hi - lo)})
        auc_rows.append({"method": method, "band": f"{int(level * 100)}%", "warning score": unc.width_warning_auc(actual_all, pred_all, hi - lo)})
print("Mean band width (mg/dL) by Clarke zone of the point forecast:")
display(pd.DataFrame(width_rows).set_index(["band", "method"]).sort_index().round(1))
print("Warning score (0.5 = no information, 1.0 = perfect):")
display(pd.DataFrame(auc_rows).set_index(["band", "method"]).sort_index().round(3))
'''))

nb = new_notebook()
nb["cells"] = cells
nb["metadata"] = {"kernelspec": {"display_name": "AZT1D (venv)", "language": "python", "name": "azt1d"}}

out_dir = ROOT / "notebooks_2"
out_dir.mkdir(exist_ok=True)
out_path = out_dir / "01_uncertainty_bands.ipynb"
nbformat.write(nb, out_path)
print(f"Wrote {out_path} ({len(cells)} cells)")
