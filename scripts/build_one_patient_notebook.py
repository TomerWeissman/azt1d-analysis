"""Builds notebooks_2/02_one_patient_view.ipynb: a title-only walkthrough for one patient.

Execute with:
  jupyter nbconvert --to notebook --execute --inplace notebooks_2/02_one_patient_view.ipynb
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


cells = []

cells.append(md("# Subject 1: forecasts, GARCH error bands, and the Clarke Error Grid"))

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
from azt1d.glimmer.clinical import clarke_zone_percentages, draw_clarke_grid

plotting.apply_style()

SUBJECT_ID = 1
WINDOW = 700                  # first 700 test steps (about 2.4 days), as in the notebook 07 chart
GRID_LEVEL = 0.80             # GARCH band size used for the Clarke grids
TREND_LEVELS = np.round(np.arange(0.55, 0.9001, 0.05), 2)
CKPT_ROOT = PROJECT_ROOT / "data" / "processed" / "checkpoints"
MODELS = {
    "No error weighted": ("cnn_lstm_v0", plotting.CATEGORICAL[0]),
    "Standard error weighted": ("cnn_lstm_v1", plotting.CATEGORICAL[1]),
}

df = loading.load_real_dataset(PROJECT_ROOT / "data" / "raw", PROJECT_ROOT / "data" / "processed")
df_subject = df[df["subject_id"] == SUBJECT_ID].reset_index(drop=True)

frames, engines = {}, {}
for name, (run, _) in MODELS.items():
    res = ckpt.load_result(CKPT_ROOT / run, SUBJECT_ID)
    val, test = unc.forecast_frames(res, df_subject)
    assert unc.matches_stored_predictions(test, res) < 0.05
    frames[name] = test
    engines[name] = unc.BandEngine(val, test)

actual = frames["No error weighted"].actual
time = pd.to_datetime(frames["No error weighted"].target_time)
'''))

cells.append(md("## 1. Actual vs. no error weighted vs. standard error weighted"))

cells.append(code('''
fig, ax = plt.subplots(figsize=(13, 4.6))
ax.plot(actual[:WINDOW], color=plotting.INK_PRIMARY, linewidth=1.5, label="Actual glucose", zorder=5)
for name, (_, color) in MODELS.items():
    ax.plot(frames[name].pred[:WINDOW], color=color, linewidth=1.2, alpha=0.9, label=name)
ax.axhline(ref.HYPO_THRESHOLD, linestyle="--", color=plotting.GLUCOSE_BAND_COLORS["hypo"], linewidth=1)
ax.axhline(ref.HYPER_THRESHOLD, linestyle="--", color=plotting.GLUCOSE_BAND_COLORS["hyper"], linewidth=1)
ax.set_xlabel("Time (5-minute steps)")
ax.set_ylabel("Glucose (mg/dL)")
ax.legend(frameon=False, loc="upper left", bbox_to_anchor=(1.0, 1.0))
fig.tight_layout()
plt.show()
'''))

cells.append(md("## 2. GARCH error bands"))

cells.append(code('''
BAND_LEVELS = (0.90, 0.80, 0.55)          # widest first so the narrower ones draw on top
BAND_ALPHA = {0.90: 0.14, 0.80: 0.26, 0.55: 0.42}

fig, axes = plt.subplots(2, 1, figsize=(13, 8.4), sharex=True, sharey=True)
for ax, (name, (_, color)) in zip(axes, MODELS.items()):
    for level in BAND_LEVELS:
        lo, hi = engines[name].bands("GARCH", level)
        ax.fill_between(np.arange(WINDOW), lo[:WINDOW], hi[:WINDOW], color=color, alpha=BAND_ALPHA[level], linewidth=0,
                        label=f"{int(level * 100)}% band")
    ax.plot(frames[name].pred[:WINDOW], color=color, linewidth=1.1, label="Prediction")
    ax.plot(actual[:WINDOW], color=plotting.INK_PRIMARY, linewidth=1.3, label="Actual glucose")
    ax.axhline(ref.HYPO_THRESHOLD, linestyle="--", color=plotting.GLUCOSE_BAND_COLORS["hypo"], linewidth=1)
    ax.axhline(ref.HYPER_THRESHOLD, linestyle="--", color=plotting.GLUCOSE_BAND_COLORS["hyper"], linewidth=1)
    ax.set_ylabel("Glucose (mg/dL)")
    ax.set_title(f"GARCH bands: {name}", loc="left", fontsize=11)
axes[0].legend(frameon=False, ncol=5, loc="upper right", fontsize=8)
axes[1].set_xlabel("Time (5-minute steps)")
fig.tight_layout()
plt.show()
'''))

cells.append(md("## 3. Clarke Error Grid"))

cells.append(code('''
grid_predictions = {}
for name in MODELS:
    lo, hi = engines[name].bands("GARCH", GRID_LEVEL)
    grid_predictions[f"GARCH, {name.lower()}"] = unc.swapped_prediction(frames[name].pred, lo, hi)
panels = {name: frames[name].pred for name in MODELS} | grid_predictions
panel_colors = {**{n: c for n, (_, c) in MODELS.items()}, **{f"GARCH, {n.lower()}": c for n, (_, c) in MODELS.items()}}

fig, axes = plt.subplots(2, 2, figsize=(11, 11))
for ax, (title, pred) in zip(axes.ravel(), panels.items()):
    draw_clarke_grid(ax, actual, pred, panel_colors[title])
    ax.set_title(title)
fig.tight_layout()
plt.show()
'''))

cells.append(code('''
zones = ["A", "B", "C", "D", "E"]
zone_table = pd.DataFrame({title: clarke_zone_percentages(actual, pred) for title, pred in panels.items()}).T[zones]
zone_table.round(2)
'''))

cells.append(md("## 4. Clarke zones as the GARCH band grows: no error weighted"))

cells.append(code('''
def zone_trend(name):
    rows = []
    for level in TREND_LEVELS:
        lo, hi = engines[name].bands("GARCH", float(level))
        rows.append({"band": level, **clarke_zone_percentages(actual, unc.swapped_prediction(frames[name].pred, lo, hi))})
    return pd.DataFrame(rows).set_index("band")


def plot_trend(name):
    trend = zone_trend(name)
    base = clarke_zone_percentages(actual, frames[name].pred)
    zone_color = dict(zip("ABCDE", plotting.CATEGORICAL[:5]))
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.4), sharex=True)
    for ax, group, title in ((axes[0], "AB", "Zones A and B"), (axes[1], "CDE", "Zones C, D and E")):
        for z in group:
            ax.plot(trend.index * 100, trend[z], marker="o", markersize=4, color=zone_color[z], linewidth=1.8, label=f"Zone {z}")
            ax.axhline(base[z], linestyle=":", color=zone_color[z], linewidth=1)
        ax.set_title(title)
        ax.set_xlabel("GARCH band size (coverage, %)")
        ax.set_ylabel("% of predictions")
        ax.legend(frameon=False, fontsize=8)
    fig.suptitle(f"{name}: dotted lines are the raw forecast with no band")
    fig.tight_layout()
    plt.show()
    return trend


trend_none = plot_trend("No error weighted")
trend_none.round(2)
'''))

cells.append(md("## 5. Clarke zones as the GARCH band grows: standard error weighted"))

cells.append(code('''
trend_standard = plot_trend("Standard error weighted")
trend_standard.round(2)
'''))

nb = new_notebook()
nb["cells"] = cells
nb["metadata"] = {"kernelspec": {"display_name": "AZT1D (venv)", "language": "python", "name": "azt1d"}}
out_path = ROOT / "notebooks_2" / "02_one_patient_view.ipynb"
nbformat.write(nb, out_path)
print(f"Wrote {out_path} ({len(cells)} cells)")
