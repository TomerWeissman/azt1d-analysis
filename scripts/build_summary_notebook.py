"""Builds notebooks/07_summary.ipynb: a presentation-style walkthrough of this
project's findings -- goal, data, data idiosyncrasies, then the models and
their results compared.

Two kinds of source material, both used without any retraining:

1. Data-exploration plots/tables are lifted directly (base64 image / HTML
   table) from already-computed cell output in notebooks 01 and 02.
2. Every model-related chart and table is rebuilt from the saved checkpoints
   in data/processed/checkpoints/ -- each checkpoint already holds a trained
   model's predictions (SubjectResult.y_test/y_pred) and the genetic
   algorithm's found weights, so "rebuild the chart" here means reading
   already-computed numbers back off disk and re-plotting them with clear
   labels, never re-training anything or re-running the GA search. This is
   what a v0/v1/v3-labeled chart is replaced with: the same underlying
   numbers, plotted fresh as "Standard" / "Fixed danger-weighted" /
   "Personalized danger-weighted" so the model comparison in this notebook
   doesn't require knowing the project's internal version jargon.

Almost the entire notebook is markdown cells with embedded images/tables --
no visible code, since the point of this pass was "graphs and explanations,
not code, unless the code itself is what needs explaining."
"""
import base64
import io
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import nbformat
import numpy as np
import pandas as pd
from nbformat.v4 import new_markdown_cell

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from azt1d import loading, plotting, reference as ref  # noqa: E402
from azt1d.glimmer import checkpoint as ckpt  # noqa: E402
from azt1d.glimmer.clinical import clarke_zone_percentages, dysglycemia_event_metrics  # noqa: E402
from azt1d.glimmer.train import region_errors  # noqa: E402

plotting.apply_style()

NB_DIR = ROOT / "notebooks"
CKPT_DIR = ROOT / "data/processed/checkpoints"

C = plotting.CATEGORICAL
APPROACH_COLOR = {"Standard": C[0], "Fixed danger-weighted": C[1], "Personalized danger-weighted": C[2]}
ARCH_COLOR = {"CNN-LSTM": C[0], "CNN-Transformer": C[1]}
APPROACHES = ["Standard", "Fixed danger-weighted", "Personalized danger-weighted"]
SUFFIX = {"Standard": "v0", "Fixed danger-weighted": "v1", "Personalized danger-weighted": "v3_tuned"}

# ---------------------------------------------------------------------------
# Checkpoint loading (read-only, no retraining)
# ---------------------------------------------------------------------------


def _subject_ids(run_dir):
    return sorted(int(p.stem.split("_")[1]) for p in run_dir.glob("subject_*.pt"))


def load_results(run_name):
    d = CKPT_DIR / run_name
    ids = _subject_ids(d)
    return {sid: ckpt.load_result(d, sid) for sid in ids}


def pooled(results):
    y_true = np.concatenate([r.y_test for r in results.values()])
    y_pred = np.concatenate([r.y_pred for r in results.values()])
    return y_true, y_pred


def mean_std(results, attr):
    vals = [getattr(r, attr) for r in results.values()]
    return float(np.mean(vals)), float(np.std(vals))


def dataset_runs(prefix, architecture):
    """{'Standard': results, 'Fixed danger-weighted': results, ...} for one architecture/dataset."""
    return {label: load_results(f"{prefix}{architecture}_{SUFFIX[label]}") for label in APPROACHES}


# ---------------------------------------------------------------------------
# Rendering helpers
# ---------------------------------------------------------------------------


def fig_html(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    plt.close(fig)
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    return f'<img src="data:image/png;base64,{b64}" style="max-width:100%">'


def table_html(df, **kwargs):
    return df.to_html(**kwargs)


def grouped_bar(categories, series, ylabel, title, colors, figsize=(8.5, 4.8), value_fmt="{:.1f}"):
    """series: {label: [value per category]}"""
    fig, ax = plt.subplots(figsize=figsize)
    x = np.arange(len(categories))
    n = len(series)
    width = 0.8 / n
    for i, (label, vals) in enumerate(series.items()):
        offset = (i - (n - 1) / 2) * width
        bars = ax.bar(x + offset, vals, width, label=label, color=colors[label])
        for b, v in zip(bars, vals):
            ax.annotate(value_fmt.format(v), (b.get_x() + b.get_width() / 2, v),
                        ha="center", va="bottom", fontsize=8, color=plotting.INK_SECONDARY)
    ax.set_xticks(x)
    ax.set_xticklabels(categories, fontsize=10)
    ymax = max(v for vals in series.values() for v in vals)
    ax.set_ylim(0, ymax * 1.15)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(frameon=False, loc="upper left", bbox_to_anchor=(1.0, 1.0))
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Reused data-exploration content (verbatim outputs, no jargon issue)
# ---------------------------------------------------------------------------

_src_cache = {}


def _src_nb(name):
    if name not in _src_cache:
        _src_cache[name] = nbformat.read(NB_DIR / name, as_version=4)
    return _src_cache[name]


def reused_image_only(nbfile, cid):
    """Like reused(), but keeps only the image output -- for cells whose
    stored output also includes a table not worth carrying over."""
    nb = _src_nb(nbfile)
    cell = next(c for c in nb.cells if c.get("id") == cid)
    for out in cell.get("outputs", []):
        data = out.get("data", {})
        if "image/png" in data:
            return f'<img src="data:image/png;base64,{data["image/png"]}" style="max-width:100%">'
    raise ValueError(f"no image output found in {nbfile}/{cid}")


def reused(nbfile, cid):
    """Pull a code cell's stored outputs (images/tables/text) out of an
    original notebook and render them as plain HTML -- no code shown."""
    nb = _src_nb(nbfile)
    cell = next(c for c in nb.cells if c.get("id") == cid)
    parts = []
    for out in cell.get("outputs", []):
        ot = out.get("output_type")
        data = out.get("data", {}) if ot in ("display_data", "execute_result") else None
        if data and "image/png" in data:
            b64 = data["image/png"]
            parts.append(f'<img src="data:image/png;base64,{b64}" style="max-width:100%">')
        elif data and "text/html" in data:
            html = data["text/html"]
            parts.append(html if isinstance(html, str) else "".join(html))
        elif data and "text/plain" in data:
            text = data["text/plain"]
            text = text if isinstance(text, str) else "".join(text)
            parts.append(f"<pre>{text}</pre>")
        elif ot == "stream":
            text = out.get("text", "")
            text = text if isinstance(text, str) else "".join(text)
            parts.append(f"<pre>{text}</pre>")
    return "\n\n".join(parts)


def md(text):
    return new_markdown_cell(text)


def slug(title):
    import re

    s = re.sub(r"[^\w\s-]", "", title).strip().lower()
    return re.sub(r"\s+", "-", s)


# ---------------------------------------------------------------------------
# Build every model-related chart/table up front
# ---------------------------------------------------------------------------

print("Loading raw AZT1D data for rebuilt exploration charts...")
_raw_df = loading.load_real_dataset(ROOT / "data/raw", ROOT / "data/processed")

# Time-of-day risk, rebuilt as two independently-scaled panels -- the
# original shared-axis version buried hypo's variation under hyper's much
# larger scale (2% vs 25%), which is exactly the pattern this chart is
# supposed to show.
_tagged = _raw_df.copy()
_tagged["hour"] = _tagged[ref.EVENT_DATETIME].dt.hour
_time_blocks = {"00:00-06:00": (0, 6), "06:00-12:00": (6, 12), "12:00-18:00": (12, 18), "18:00-24:00": (18, 24)}
_tagged["block"] = pd.cut(_tagged["hour"], bins=[0, 6, 12, 18, 24], labels=list(_time_blocks.keys()),
                           right=False, include_lowest=True)
_tagged["glycemic_state"] = pd.cut(_tagged[ref.CGM], bins=[-np.inf, ref.HYPO_THRESHOLD, ref.HYPER_THRESHOLD, np.inf],
                                    labels=["hypo", "in_range", "hyper"])
_block_pct = pd.crosstab(_tagged["block"], _tagged["glycemic_state"], normalize="index") * 100

fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
for ax, region, color, title in [
    (axes[0], "hypo", plotting.GLUCOSE_BAND_COLORS["hypo"], "Hypoglycemia (<70 mg/dL)"),
    (axes[1], "hyper", plotting.GLUCOSE_BAND_COLORS["hyper"], "Hyperglycemia (>180 mg/dL)"),
]:
    vals = _block_pct[region]
    bars = ax.bar(range(len(vals)), vals, color=color)
    for b, v in zip(bars, vals):
        ax.annotate(f"{v:.1f}%", (b.get_x() + b.get_width() / 2, v), ha="center", va="bottom", fontsize=8)
    ax.set_xticks(range(len(vals)))
    ax.set_xticklabels(vals.index, fontsize=9)
    ax.set_ylim(0, vals.max() * 1.3)
    ax.set_ylabel("% of readings")
    ax.set_title(title)
fig.suptitle("Dysglycemia risk by time of day, whole cohort")
fig.tight_layout()
time_of_day_chart = fig_html(fig)

_hyper_worst = _block_pct["hyper"].idxmax()
_hyper_best = _block_pct["hyper"].idxmin()
_hypo_worst = _block_pct["hypo"].idxmax()
TIME_OF_DAY_OBSERVATION = (
    f"Hyperglycemia climbs steadily through the day, from {_block_pct['hyper'].min():.1f}% "
    f"overnight ({_hyper_best}) to {_block_pct['hyper'].max():.1f}% in the evening "
    f"({_hyper_worst}) -- consistent with meals accumulating their effect over the day. "
    f"Hypoglycemia is a much smaller share throughout, but still highest in the evening "
    f"({_hypo_worst}, {_block_pct['hypo'].max():.1f}%). Both kinds of risk peak at the same "
    "time of day, not opposite ends of it."
)

# Daily carb/bolus/basal totals per subject, as a chart instead of a table --
# sorted so the outliers are immediately visible instead of buried in 25 rows.
from azt1d import metrics as _metrics  # noqa: E402

_daily = (
    _metrics.daily_carb_and_bolus(_raw_df)
    .groupby("subject_id")[["total_carbs_g", "total_bolus_u", "total_basal_u"]]
    .mean()
    .sort_values("total_carbs_g")
)
fig, ax = plt.subplots(figsize=(9, 5))
y = np.arange(len(_daily))
ax.barh(y, _daily["total_carbs_g"], color=C[3])
ax.set_yticks(y)
ax.set_yticklabels(_daily.index, fontsize=7)
ax.set_xlabel("Average carbs logged per day (grams)")
ax.set_ylabel("Subject")
ax.set_title("How much each patient logs eating, per day")
fig.tight_layout()
carb_outlier_chart = fig_html(fig)

print("Loading checkpoints and building charts...")

az_lstm = dataset_runs("", "cnn_lstm")
az_tf = dataset_runs("", "cnn_transformer")
oh_lstm = dataset_runs("ohiot1dm_", "cnn_lstm")
oh_tf = dataset_runs("ohiot1dm_", "cnn_transformer")

# 1. Example forecast plot (Standard CNN-LSTM, AZT1D, subject 1)
r1 = az_lstm["Standard"][1]
fig, ax = plt.subplots(figsize=(11, 3.5))
n = 700
ax.plot(r1.y_test[:n], color=plotting.INK_PRIMARY, linewidth=1.3, label="Actual glucose")
ax.plot(r1.y_pred[:n], color=C[0], linewidth=1.3, label="Model's prediction")
ax.axhline(ref.HYPO_THRESHOLD, linestyle="--", color=plotting.GLUCOSE_BAND_COLORS["hypo"], linewidth=1)
ax.axhline(ref.HYPER_THRESHOLD, linestyle="--", color=plotting.GLUCOSE_BAND_COLORS["hyper"], linewidth=1)
ax.set_ylabel("Glucose (mg/dL)")
ax.set_xlabel("Time (5-minute steps)")
ax.set_title("One patient, held-out test data -- predicting 60 minutes ahead")
ax.legend(frameon=False)
fig.tight_layout()
forecast_chart = fig_html(fig)

# 2. Region-error chart, AZT1D CNN-LSTM, all 3 approaches.
# Per-subject region MAE, then averaged across subjects (every patient counted
# equally) -- matching the methodology used throughout notebooks 03/05/06, not
# a pooled/sample-weighted average, so these numbers stay consistent with
# everything already reported from this project.
def per_subject_region_mae(results):
    rows = []
    for r in results.values():
        re_ = region_errors(r.y_test, r.y_pred)
        rows.append({reg: re_[reg]["mae"] for reg in ("hypo", "normal", "hyper")})
    df = pd.DataFrame(rows)
    return df.mean()


region_data = {}
for label in APPROACHES:
    m = per_subject_region_mae(az_lstm[label])
    region_data[label] = [m["hypo"], m["normal"], m["hyper"]]
region_fig = grouped_bar(
    ["Hypo (<70)", "Normal (70-180)", "Hyper (>180)"],
    region_data, "Mean absolute error (mg/dL)",
    "Error by glucose region -- AZT1D, CNN-LSTM", APPROACH_COLOR,
)
region_chart_html = fig_html(region_fig)

# 3. Clinical: event detection P/R/F1, AZT1D CNN-LSTM, all 3 approaches
event_data = {"Precision": [], "Recall": [], "F1": []}
for label in APPROACHES:
    y_true, y_pred = pooled(az_lstm[label])
    m = dysglycemia_event_metrics(y_true, y_pred)
    event_data["Precision"].append(m["precision"])
    event_data["Recall"].append(m["recall"])
    event_data["F1"].append(m["f1"])
fig, ax = plt.subplots(figsize=(7.5, 4.5))
metrics_order = ["Precision", "Recall", "F1"]
x = np.arange(len(metrics_order))
width = 0.8 / len(APPROACHES)
for i, label in enumerate(APPROACHES):
    vals = [event_data[m][i] for m in metrics_order]
    offset = (i - 1) * width
    bars = ax.bar(x + offset, vals, width, label=label, color=APPROACH_COLOR[label])
    for b, v in zip(bars, vals):
        ax.annotate(f"{v:.2f}", (b.get_x() + b.get_width() / 2, v), ha="center", va="bottom", fontsize=8)
ax.set_xticks(x)
ax.set_xticklabels(metrics_order)
ax.set_ylim(0, 1.0)
ax.set_ylabel("Score")
ax.set_title("Detecting real dysglycemic moments -- AZT1D, CNN-LSTM")
ax.legend(frameon=False, loc="upper left", bbox_to_anchor=(1.0, 1.0))
fig.tight_layout()
event_chart_html = fig_html(fig)

# 4. Clinical: Clarke Error Grid zones, AZT1D CNN-LSTM, all 3 approaches
zone_data = {}
for label in APPROACHES:
    y_true, y_pred = pooled(az_lstm[label])
    z = clarke_zone_percentages(y_true, y_pred)
    zone_data[label] = [z["A"], z["B"], z["C"], z["D"], z["E"]]
zone_fig = grouped_bar(
    ["Zone A", "Zone B", "Zone C", "Zone D", "Zone E"],
    zone_data, "% of predictions", "Clarke Error Grid zones -- AZT1D, CNN-LSTM", APPROACH_COLOR,
    figsize=(10, 4.8), value_fmt="{:.1f}",
)
zone_chart_html = fig_html(zone_fig)

# 6. Full comparison: RMSE by approach x architecture, AZT1D and OhioT1DM
def full_comparison(lstm_runs, tf_runs, title):
    series = {}
    for arch_label, runs in [("CNN-LSTM", lstm_runs), ("CNN-Transformer", tf_runs)]:
        series[arch_label] = [mean_std(runs[a], "rmse")[0] for a in APPROACHES]
    fig = grouped_bar(APPROACHES, series, "Mean RMSE (mg/dL)", title, ARCH_COLOR, figsize=(8.5, 4.8))
    return fig_html(fig)


az_full_chart = full_comparison(az_lstm, az_tf, "All approaches, both architectures -- AZT1D")
oh_full_chart = full_comparison(oh_lstm, oh_tf, "All approaches, both architectures -- OhioT1DM")


def full_comparison_table(lstm_runs, tf_runs):
    rows = []
    for arch_label, runs in [("CNN-LSTM", lstm_runs), ("CNN-Transformer", tf_runs)]:
        row = {"Architecture": arch_label}
        for a in APPROACHES:
            rmse_m, rmse_s = mean_std(runs[a], "rmse")
            row[f"{a} RMSE"] = f"{rmse_m:.2f} +/- {rmse_s:.2f}"
        rows.append(row)
    df = pd.DataFrame(rows).set_index("Architecture")
    return table_html(df)


az_full_table = full_comparison_table(az_lstm, az_tf)
oh_full_table = full_comparison_table(oh_lstm, oh_tf)

# 7. Comparison against the paper's own reported numbers (CNN-LSTM only --
# the paper doesn't report a directly comparable Transformer+GLIMMER number
# for AZT1D). Paper numbers are hardcoded here since they come from the
# paper itself, not from anything computed in this project.
PAPER_NUMBERS = {
    "AZT1D": {"baseline": (29.55, 6.49), "glimmer": (22.48, 3.57)},
    "OhioT1DM": {"baseline": (31.98, 4.15), "glimmer": (23.97, 3.77)},
}
_rows = []
for dataset, runs in [("AZT1D", az_lstm), ("OhioT1DM", oh_lstm)]:
    p = PAPER_NUMBERS[dataset]
    std_m, std_s = mean_std(runs["Standard"], "rmse")
    pers_m, pers_s = mean_std(runs["Personalized danger-weighted"], "rmse")
    _rows.append({
        "Dataset": dataset,
        "Paper's baseline": f"{p['baseline'][0]:.2f} +/- {p['baseline'][1]:.2f}",
        "Paper's GLIMMER (best)": f"{p['glimmer'][0]:.2f} +/- {p['glimmer'][1]:.2f}",
        "This project, standard": f"{std_m:.2f} +/- {std_s:.2f}",
        "This project, personalized (best)": f"{pers_m:.2f} +/- {pers_s:.2f}",
    })
paper_comparison_table = table_html(pd.DataFrame(_rows).set_index("Dataset"), index=True)

print("Charts built.")

# ---------------------------------------------------------------------------
# Assemble
# ---------------------------------------------------------------------------

sections = []

sections.append((1, "The goal", [
    md(
        "Predict a person's blood glucose 60 minutes into the future, using their own "
        "recent glucose readings, insulin doses, and carbs eaten. The point of the extra "
        "warning time is simple: a person (or their pump) can act on a predicted high or "
        "low before it actually happens, not just react once it does.\n\n"
        "The specific question this project tests: **does it help to train the model to "
        "care more about getting dangerous moments right (very high or very low glucose), "
        "even at some cost to its accuracy the rest of the time?** That's the core idea "
        "of the paper this project replicates "
        "([GLIMMER](https://arxiv.org/abs/2502.14183))."
    ),
    md(
        "To make sure any answer to that question is real and not a fluke of one setup, "
        "it gets tested three different ways at once:\n\n"
        "- **Two model designs** (\"architectures\"): **CNN-LSTM** and **CNN-Transformer** "
        "-- different internal designs, same job, same inputs.\n"
        "- **Three ways of training each one**, covered in detail in Part 4: an ordinary "
        "model, a model told to weight dangerous-zone errors more heavily (using one fixed "
        "setting), and a model where that weighting is tuned individually per patient.\n"
        "- **Two independent, real-world datasets**: AZT1D and OhioT1DM -- different "
        "clinics, different patients, different insulin pump hardware.\n\n"
        "If an effect only shows up on one architecture, one training approach, or one "
        "dataset, it's probably not real. If it shows up consistently across all of them, "
        "that's a much stronger result."
    ),
]))

sections.append((2, "The data", [
    md(
        "AZT1D is real-world data from 25 people with type 1 diabetes, each using an "
        "automated insulin delivery system: a continuous glucose monitor (Dexcom G6 Pro) "
        "paired with an insulin pump (Tandem t:slim X2) that automatically adjusts insulin "
        "based on glucose trends. Researchers at Mayo Clinic Arizona pulled about a month "
        "of data per person straight off each patient's own pump and sensor during routine "
        "care, with informed consent and ethics review approval. It's published openly on "
        "Mendeley Data by a credentialed academic team. Worth keeping in mind: it's a small, "
        "single-clinic sample (25 people) that skews older (ages 27-80, averaging around "
        "60), so results here describe this group, not necessarily everyone with T1D."
    ),
    md(
        "**What each 5-minute reading contains:**\n\n"
        "- **Glucose (CGM):** measured continuously by a sensor under the skin, in mg/dL. "
        "Roughly 70-180 mg/dL is the everyday target range: below 70 is dangerously low "
        "(hypoglycemia), above 180 is too high (hyperglycemia).\n"
        "- **Background (\"basal\") insulin:** a steady trickle the pump delivers around "
        "the clock, in units per hour.\n"
        "- **Meal-time (\"bolus\") insulin:** a bigger, one-time dose for a meal or to "
        "correct a high, in units.\n"
        "- **Carbs:** grams of carbohydrate eaten at a logged meal -- what pushes glucose "
        "up, and roughly what insulin doses are sized against."
    ),
    md("### Who's in the dataset"),
    md(
        "25 patients, ages 27-80 (averaging around 59), 13 female and 12 male, with an "
        "average A1c of 6.64% -- diabetic but reasonably well-controlled as a group "
        "(A1c under 7% is a common treatment target)."
    ),
    md(reused_image_only("01_data_exploration.ipynb", "f3037d49")),
    md("### What one patient's data actually looks like"),
    md(reused("02_glimmer_v0_baseline.ipynb", "7540dc83")),
    md(
        "The dashed lines mark the danger thresholds. This one patient spent about 14% of "
        "their time above the high line and only about 1% below the low one -- a pattern "
        "that holds across the whole group too, covered next."
    ),
]))

sections.append((3, "What's unusual about this data", [
    md(
        "A few patterns worth knowing before looking at any model results, since they "
        "shape what \"doing well\" even means here."
    ),
    md("### Glucose readings lean high, not low"),
    md(reused("02_glimmer_v0_baseline.ipynb", "78bd8e35")),
    md(
        "Across the whole group, readings are high (above 180) about 20% of the time but "
        "low (below 70) only about 2% of the time. That makes sense two ways: patients and "
        "their pumps actively try to avoid going low, since the immediate risk is worse "
        "there, and glucose has a hard floor at zero that the high end doesn't have. It "
        "also means the hypo cases a model needs to catch are rare to begin with."
    ),
    md("### Some patients are much harder to predict for than others"),
    md(reused_image_only("01_data_exploration.ipynb", "f9f55a84")),
    md(
        "Time spent in the healthy range (70-180 mg/dL) ranges from under 50% for the "
        "toughest-controlled patient to over 90% for the best-controlled one. Later on, "
        "prediction accuracy tracks this almost exactly -- patients with more erratic "
        "glucose are harder to predict for, not because the model is worse at them "
        "specifically, but because their own data is intrinsically harder to forecast."
    ),
    md("### A few patients are extreme outliers in how much they log"),
    md(carb_outlier_chart),
    md(
        "Most patients log a fairly typical amount of carbs per day, but a couple are far "
        "outside that range -- one logs almost none, one logs far more than anyone else. "
        "Worth remembering when a specific patient's model results look unusual later: "
        "sometimes it's the data, not the model."
    ),
    md("### Risk isn't spread evenly across the day"),
    md(time_of_day_chart),
    md(TIME_OF_DAY_OBSERVATION),
]))

sections.append((4, "The models", []))

sections.append((None, None, [
    md(
        "**How a model is tested:** each patient gets their own personal model (not one "
        "model shared across everyone), trained on their own history. It looks at the last "
        "2 hours of glucose/insulin/carb data and predicts glucose 1 hour ahead. The most "
        "recent stretch of each patient's data is held back and never shown to the model "
        "during training, so every result below is measured on data the model has never "
        "seen."
    ),
    md(
        "**Three ways of training that model get compared throughout this part:**\n\n"
        "- **Standard** -- trained the ordinary way, to be right on average across every "
        "prediction equally. No special attention to dangerous glucose zones.\n"
        "- **Fixed danger-weighted** -- errors during dangerous zones (very high or very "
        "low glucose) are counted as more costly than errors in the normal range, using "
        "one weighting setting applied to every patient the same way.\n"
        "- **Personalized danger-weighted** -- the same idea, except each patient gets "
        "their own weighting instead of one shared setting, found by trying many candidate "
        "settings per patient and keeping whichever works best for them.\n\n"
        "All three appear together in every comparison below, so what changes is only "
        "which of these three trained the model, never the data or the architecture."
    ),
]))

sections.append((None, "A first look: what does an ordinary model get wrong", [
    md(
        "Before comparing all three, here's what the **standard** model alone does on one "
        "patient's held-out data, to see the actual problem the other two approaches are "
        "trying to fix."
    ),
    md(forecast_chart),
    md(
        "The prediction tracks the real value well in the middle of the range, but lags "
        "and undershoots right at the sharp highs and lows -- exactly the moments that "
        "matter most clinically. That gap is what danger-weighted training, fixed or "
        "personalized, is meant to close."
    ),
]))

sections.append((None, "Comparing all three approaches", [
    md(region_chart_html),
    md(
        "This is the actual mechanism: danger-weighting (fixed or personalized) brings "
        "hypo and hyper error down, exactly as intended, but normal-zone error goes up "
        "enough that overall accuracy (averaged across everything) ends up worse than "
        "standard's. Personalizing the weights per patient recovers some of that loss but "
        "not all of it. A real trade-off, not a straightforward win."
    ),
    md("**Does that trade-off matter clinically?** Two more direct checks:"),
    md(event_chart_html),
    md(
        "Danger-weighting pushes recall (catching real dangerous moments) up substantially, "
        "at the cost of precision (more false alarms) -- fixed weighting more so, "
        "personalized a smaller version of the same trade."
    ),
    md(
        "The Clarke Error Grid is the standard clinical-accuracy tool for glucose "
        "predictions, grading every prediction into one of five zones by how much the "
        "error would actually matter: **zone A** is clinically accurate, **zone D** means "
        "a real dangerous moment got predicted as safe -- the worst kind of miss for a "
        "warning system -- and **zone E** is the worst case in the other direction."
    ),
    md(zone_chart_html),
    md(
        "Zone D drops meaningfully with danger-weighted training, fixed or personalized -- "
        "fewer of the misses that would actually matter, even though zone A (plain "
        "accuracy) drops too."
    ),
]))

sections.append((5, "Does this hold up across architectures and datasets", [
    md(
        "Everything above used one architecture (CNN-LSTM) on one dataset (AZT1D), to "
        "explain the idea with one running example. On its own, that's not enough to "
        "trust the result -- it could be specific to that one model design or that one "
        "group of patients. So here's the same check, repeated on a second architecture "
        "(CNN-Transformer) and a second, independent dataset (OhioT1DM)."
    ),
    md(
        "**What to look for in both charts below:** the same order every time -- standard "
        "most accurate, fixed danger-weighted least accurate, personalized in between, "
        "closer to standard. If that order holds on a different model design and a "
        "different real-world dataset, that's strong evidence it's a real pattern, not a "
        "coincidence of one setup."
    ),
    md("**AZT1D** -- both architectures, all three approaches:"),
    md(az_full_chart),
    md(az_full_table),
    md("**OhioT1DM** -- the same comparison, on a second, independent dataset:"),
    md(oh_full_chart),
    md(oh_full_table),
    md(
        "The order holds in all four cases (2 architectures x 2 datasets): standard is "
        "always most accurate, fixed danger-weighted always least accurate, personalized "
        "always in between. That's good evidence this is a real, general pattern rather "
        "than something specific to one model or one dataset.\n\n"
        "Two things worth flagging: the (3.29, 2.38) fixed weights used throughout "
        "actually came from a genetic algorithm search on OhioT1DM specifically (the "
        "paper's own choice, not this project's). Running the same personalized search on "
        "OhioT1DM directly -- the most direct check available -- found consistently "
        "gentler weights than that reference value, not stronger ones. And on OhioT1DM "
        "specifically, the fixed weighting actually beat personalized tuning in the hypo "
        "zone, the one place the two datasets disagreed."
    ),
]))

sections.append((6, "How does this compare to the original GLIMMER paper", [
    md(
        "Everything so far compares this project's own approaches against each other. "
        "The table below is the more important comparison: how do this project's numbers "
        "stack up against what the paper itself reported? (CNN-LSTM only -- the paper "
        "doesn't report a directly comparable Transformer number.)"
    ),
    md(paper_comparison_table),
    md(
        "Two things to take from this table. First, **this project's standard model is a "
        "reasonable replication of the paper's own baseline** -- close on both datasets, "
        "not off by some large, suspicious margin, which is a basic sanity check that this "
        "implementation is doing roughly the right thing before trusting anything built on "
        "top of it. Second, and more important: **the paper's GLIMMER (their best "
        "danger-weighted result) beats their own baseline** on both datasets, while **this "
        "project's personalized danger-weighted result does not beat its own standard "
        "model** on either dataset -- the opposite direction. That gap, not any of the "
        "internal comparisons above, is the real headline finding of this replication."
    ),
]))

sections.append((7, "Bottom line", [
    md(
        "Danger-weighted training is a real trade, not a free win: it makes the model "
        "less accurate on average but more likely to catch the moments that actually "
        "matter clinically -- more real hypo/hyper events detected, and meaningfully "
        "fewer dangerous misses on the Clarke Error Grid. Personalizing those weights per "
        "patient, instead of using one fixed setting, recovers some but not all of the "
        "accuracy that fixed weighting gives up, consistently across both model designs "
        "and both real-world datasets tested. What it doesn't do, on either dataset, is "
        "beat the paper's own reported improvement over its baseline.\n\n"
        "The gap between this project's numbers and the original paper's is most likely "
        "the size of the genetic algorithm search used here (deliberately scoped down for "
        "compute reasons, see `azt1d/glimmer/ga.py`) -- a larger search budget is the one "
        "concrete next step if this needs to go further."
    ),
]))

cells = [
    md(
        "# GLIMMER on AZT1D and OhioT1DM: findings\n\n"
        "*A presentation-style walkthrough of this project's findings. Every chart and "
        "table here is built from already-computed results (trained models' saved "
        "predictions, the genetic algorithm's saved search results) -- nothing is "
        "retrained or re-run to produce this notebook. For the full detailed record, "
        "including all code, see notebooks 01-06.*"
    )
]

toc_entries = []
for num, title, _ in sections:
    if num is not None:
        toc_entries.append((num, title))

toc_lines = ["## Table of contents", ""]
for num, title in toc_entries:
    heading = f"{num}. {title}"
    toc_lines.append(f"{num}. [{title}](#{slug(heading)})")
cells.append(md("\n".join(toc_lines)))

for num, title, piece_cells in sections:
    if num is not None:
        cells.append(md(f"## {num}. {title}"))
    elif title is not None:
        cells.append(md(f"### {title}"))
    cells.extend(piece_cells)

new_nb = nbformat.v4.new_notebook()
new_nb["cells"] = cells
new_nb["metadata"] = {}

out_path = NB_DIR / "07_summary.ipynb"
nbformat.write(new_nb, out_path)
print(f"Wrote {out_path} ({len(cells)} cells)")
