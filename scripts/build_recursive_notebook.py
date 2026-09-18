"""Builds notebooks/08_recursive_forecasting.ipynb as a real, executed
notebook (unlike scripts/build_summary_notebook.py, this one runs live code
cells the normal way -- nbconvert --execute fills in the outputs).
"""
import uuid

import nbformat
from nbformat.v4 import new_code_cell, new_markdown_cell


def md(text):
    c = new_markdown_cell(text)
    c["id"] = uuid.uuid4().hex[:8]
    return c


def code(text):
    c = new_code_cell(text)
    c["id"] = uuid.uuid4().hex[:8]
    return c


cells = []

cells.append(md(
    "# Recursive forecasting: how far can the model trust itself?\n\n"
    "Every other notebook in this project asks the model to look at 2 hours of "
    "history and predict one value, 60 minutes ahead, in a single shot. That's "
    "the paper's own setup, and it's a reasonable thing to ask a model to do -- "
    "but it also means the model never has to live with its own mistakes. This "
    "notebook asks a different question: what happens if the model has to keep "
    "predicting forward, using its own earlier predictions as if they were real "
    "data, instead of always getting a fresh window of ground truth?"
))

cells.append(md(
    "## Why this needs a different model, not just a different question\n\n"
    "The existing models take 6 inputs per timestep (glucose, background "
    "insulin, meal insulin, carbs, a glucose moving average, and a hypo/normal/"
    "hyper label) and output exactly one thing: glucose, 60 minutes later. To "
    "run recursively -- feed a prediction back in as input for the next "
    "prediction -- the model has to predict *all* the things it needs as input, "
    "not just glucose, or it has no way to keep going without secretly relying "
    "on real future data it wouldn't actually have.\n\n"
    "Two of those 6 inputs (the moving average, the hypo/normal/hyper label) "
    "are just simple calculations on top of the glucose history, so they don't "
    "need their own model. The other four -- glucose, background insulin, meal "
    "insulin, carbs -- all need to be predicted directly. This notebook trains "
    "one small model to do exactly that: given the last 2 hours, predict all "
    "four, 5 minutes ahead. Then it's run recursively: predict one step, treat "
    "that prediction as real, predict the next step from it, and so on.\n\n"
    "Worth flagging up front: background insulin changes slowly and predictably, "
    "but meal insulin and carbs are the opposite -- a person eats or doses at a "
    "handful of specific moments and does nothing the rest of the time. A model "
    "trained to minimize average error on that kind of sparse, spiky data will "
    "likely learn to predict something close to zero most of the time, since "
    "that's usually the safest guess. Whether that actually happens, and what it "
    "does to the recursive glucose forecast, is checked directly below rather "
    "than assumed."
))

cells.append(code(
    "import sys\n"
    "from pathlib import Path\n\n"
    "import numpy as np\n"
    "import pandas as pd\n"
    "import matplotlib.pyplot as plt\n\n"
    "PROJECT_ROOT = Path.cwd().parent if Path.cwd().name == \"notebooks\" else Path.cwd()\n"
    "sys.path.insert(0, str(PROJECT_ROOT / \"src\"))\n\n"
    "from azt1d import loading, plotting\n"
    "from azt1d import reference as ref\n"
    "from azt1d.glimmer import features as feat\n"
    "from azt1d.glimmer import sequences as seq\n"
    "from azt1d.glimmer import recursive as rec\n\n"
    "plotting.apply_style()\n"
    "pd.set_option(\"display.max_columns\", None)"
))

cells.append(md("## 1. Load data, pick one patient"))

cells.append(code(
    "RAW_DIR = PROJECT_ROOT / \"data\" / \"raw\"\n"
    "PROCESSED_DIR = PROJECT_ROOT / \"data\" / \"processed\"\n\n"
    "df = loading.load_real_dataset(RAW_DIR, PROCESSED_DIR)\n"
    "SUBJECT_ID = 1  # the same illustrative patient used throughout this project\n"
    "df_subject = df[df[\"subject_id\"] == SUBJECT_ID].reset_index(drop=True)\n"
    "print(f\"Subject {SUBJECT_ID}: {len(df_subject):,} readings.\")"
))

cells.append(md(
    "**Update after a first pass:** an earlier version of this notebook (no time-of-day "
    "feature) found the recursive glucose curve flatlining almost immediately, because the "
    "model had no way to know a meal might be coming and just predicted \"probably nothing\" "
    "for meal insulin and carbs the whole way through. Two extra input features are added "
    "below to give it that signal: sin/cos of time-of-day (meals cluster around particular "
    "times, already visible earlier in this project's own data exploration). Unlike glucose, "
    "insulin, and carbs, time doesn't need to be predicted during the rollout at all -- the "
    "clock is exactly known in advance -- so it's filled in directly rather than generated."
))

cells.append(md("## 2. Train the multi-output, next-step model"))

cells.append(code(
    "data = rec.prepare_multi_output_data(df_subject)\n"
    "print(f\"train/val/test windows: {len(data.X_train)}/{len(data.X_val)}/{len(data.X_test)}\")\n"
    "print(\"Target columns:\", rec.TARGET_COLUMNS)\n"
    "print(\"Target means (train set):\", dict(zip(rec.TARGET_COLUMNS, data.target_means.round(3))))\n"
    "print(\"Target stds (train set):\", dict(zip(rec.TARGET_COLUMNS, data.target_stds.round(3))))"
))

cells.append(code(
    "result = rec.train_multi_output_model(data, architecture=\"cnn_lstm\")\n"
    "print(f\"Trained. {result.n_params:,} parameters.\")\n"
    "print(\"Per-target validation RMSE (raw units):\")\n"
    "for col, val in result.per_target_val_rmse.items():\n"
    "    print(f\"  {col}: {val:.3f}\")"
))

cells.append(md(
    "**Checking the sparse-target concern directly:** compare each target's "
    "validation RMSE above against simply guessing that target's own mean value "
    "every time. If the model's RMSE is barely better than the guess-the-mean "
    "baseline for bolus/carbs specifically, that's the predicted sparse-target "
    "collapse actually happening -- worth knowing before trusting the recursive "
    "curve's insulin/carb behavior."
))

cells.append(code(
    "baseline_rmse = {}\n"
    "for j, col in enumerate(rec.TARGET_COLUMNS):\n"
    "    mean_guess = data.Y_train[:, j].mean()\n"
    "    baseline_rmse[col] = float(np.sqrt(np.mean((data.Y_val[:, j] - mean_guess) ** 2)))\n\n"
    "comparison = pd.DataFrame({\n"
    "    \"model_rmse\": result.per_target_val_rmse,\n"
    "    \"guess_the_mean_rmse\": baseline_rmse,\n"
    "})\n"
    "comparison[\"model_beats_mean_by\"] = comparison[\"guess_the_mean_rmse\"] - comparison[\"model_rmse\"]\n"
    "comparison.round(3)"
))

cells.append(md(
    "## 3. Roll the model forward\n\n"
    "Starting right at the beginning of this patient's held-out test data -- "
    "never seen during training -- and walking forward 100 steps (about 8.3 "
    "hours), feeding each step's own prediction back in as the next step's "
    "input."
))

cells.append(code(
    "model = rec.load_multi_output_model(result)\n"
    "enriched = rec.add_all_features(df_subject)\n\n"
    "N_STEPS = 100  # ~8.3 hours of 5-minute steps\n"
    "n_train_val_windows = len(data.X_train) + len(data.X_val)\n"
    "start_idx = n_train_val_windows + seq.LOOKBACK  # first row of the held-out test region\n\n"
    "trajectory = rec.recursive_forecast(\n"
    "    model, enriched, start_idx=start_idx, scaler=data.scaler,\n"
    "    target_means=data.target_means, target_stds=data.target_stds, n_steps=N_STEPS,\n"
    ")\n"
    "trajectory.head()"
))

cells.append(md(
    "## 4. The curve, and the diff\n\n"
    "Top: the actual glucose trace against what the model generated entirely "
    "from its own prior predictions. Bottom: the gap between them, plotted "
    "against how far into the recursive rollout each point is -- this is the "
    "direct answer to \"how much does error grow the further out you go.\""
))

cells.append(code(
    "minutes_ahead = (trajectory[\"step\"] + 1) * 5\n"
    "error = trajectory[\"pred_CGM\"] - trajectory[\"actual_CGM\"]\n\n"
    "fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 6.5), sharex=True, height_ratios=[2, 1])\n\n"
    "ax1.plot(minutes_ahead, trajectory[\"actual_CGM\"], color=plotting.INK_PRIMARY, linewidth=1.4, label=\"Actual glucose\")\n"
    "ax1.plot(minutes_ahead, trajectory[\"pred_CGM\"], color=plotting.CATEGORICAL[0], linewidth=1.4, label=\"Recursively predicted\")\n"
    "ax1.axhline(ref.HYPO_THRESHOLD, linestyle=\"--\", color=plotting.GLUCOSE_BAND_COLORS[\"hypo\"], linewidth=1)\n"
    "ax1.axhline(ref.HYPER_THRESHOLD, linestyle=\"--\", color=plotting.GLUCOSE_BAND_COLORS[\"hyper\"], linewidth=1)\n"
    "ax1.set_ylabel(\"Glucose (mg/dL)\")\n"
    "ax1.set_title(f\"Subject {SUBJECT_ID}: recursive forecast vs. actual, {N_STEPS} steps ({N_STEPS*5} minutes)\")\n"
    "ax1.legend(frameon=False)\n\n"
    "ax2.axhline(0, color=plotting.BASELINE, linewidth=1)\n"
    "ax2.plot(minutes_ahead, error, color=plotting.CATEGORICAL[3], linewidth=1.3)\n"
    "ax2.fill_between(minutes_ahead, error, 0, color=plotting.CATEGORICAL[3], alpha=0.15)\n"
    "ax2.set_ylabel(\"Predicted - actual\\n(mg/dL)\")\n"
    "ax2.set_xlabel(\"Minutes into the recursive rollout\")\n\n"
    "fig.tight_layout()\n"
    "plt.show()"
))

cells.append(code(
    "print(f\"Mean absolute error, first 30 minutes:   {error[minutes_ahead <= 30].abs().mean():.1f} mg/dL\")\n"
    "print(f\"Mean absolute error, last 30 minutes:     {error[minutes_ahead >= minutes_ahead.max() - 30].abs().mean():.1f} mg/dL\")\n"
    "print(f\"Mean absolute error, whole rollout:       {error.abs().mean():.1f} mg/dL\")\n"
    "print(f\"Max absolute error reached, and at what point:  {error.abs().max():.1f} mg/dL at minute {minutes_ahead[error.abs().idxmax()]}\")"
))

cells.append(md(
    "## 5. What happened to the other three predictions\n\n"
    "The recursive glucose curve above depends on what the model predicted for "
    "background insulin, meal insulin, and carbs at every step too, since those "
    "feed back in as input just like glucose does. Worth seeing directly, given "
    "the sparse-target concern raised at the start."
))

cells.append(code(
    "fig, axes = plt.subplots(3, 1, figsize=(11, 8), sharex=True)\n"
    "panels = [\n"
    "    (\"Basal\", \"Background insulin (U/hr)\"),\n"
    "    (\"TotalBolusInsulinDelivered\", \"Meal insulin (U)\"),\n"
    "    (\"CarbSize\", \"Carbs (g)\"),\n"
    "]\n"
    "for ax, (col, ylabel) in zip(axes, panels):\n"
    "    ax.plot(minutes_ahead, trajectory[f\"actual_{col}\"], color=plotting.INK_PRIMARY, linewidth=1.2, label=\"Actual\")\n"
    "    ax.plot(minutes_ahead, trajectory[f\"pred_{col}\"], color=plotting.CATEGORICAL[1], linewidth=1.2, label=\"Predicted\")\n"
    "    ax.set_ylabel(ylabel)\n"
    "    ax.legend(frameon=False, fontsize=8)\n"
    "axes[-1].set_xlabel(\"Minutes into the recursive rollout\")\n"
    "fig.suptitle(f\"Subject {SUBJECT_ID}: the other three recursively-predicted inputs\")\n"
    "fig.tight_layout()\n"
    "plt.show()"
))

cells.append(md(
    "## 6. What the first version showed (no time-of-day feature)\n\n"
    "*Kept here for comparison against the result below with time-of-day added.*\n\n"
    "The single-step model itself is genuinely good: 10.2 mg/dL RMSE predicting glucose "
    "one 5-minute step ahead, against a 38.0 mg/dL baseline for just guessing the average. "
    "That part works.\n\n"
    "The recursive rollout is a different story. Average error is 8.4 mg/dL over the first "
    "30 minutes, still close to the one-step accuracy, but by the last 30 minutes of the "
    "500-minute rollout it has grown to 49.4 mg/dL, almost 6 times worse, peaking at 53.4 "
    "mg/dL. Error does not creep up steadily. It tracks specific moments the model gets "
    "structurally wrong, not just accumulated small drift.\n\n"
    "**The direct cause is visible in the other three predicted values.** Background "
    "insulin, meal insulin, and carbs all feed back into the model as input at every step, "
    "exactly like glucose does. Background insulin's prediction stays roughly in the right "
    "neighborhood, since it changes slowly. Meal insulin and carbs do not: the real data has "
    "a sharp, single meal partway through the rollout (about 22 units of insulin and 54 "
    "grams of carbs around minute 230), and the recursive prediction never sees it coming. "
    "It stays essentially flat near zero for the entire 500 minutes. That matches what the "
    "per-target validation numbers already showed before the rollout even started: the "
    "model's error on meal insulin and carbs was no better than just guessing the average "
    "value every time, while its glucose error beat that same baseline by a wide margin. "
    "The model was never actually predicting meals. It was predicting the safe default of "
    "no meal, which is usually correct, right up until it is not.\n\n"
    "Because no meal ever gets simulated, the recursive glucose curve has nothing to react "
    "to. It settles into a flat line around 130-140 mg/dL within about an hour and stays "
    "there for the rest of the rollout, missing every real swing along the way: a drop to "
    "83, a rise to 165, a later drop to 85. The gap between the flat prediction and the "
    "real, swinging glucose trace is exactly where the growing error in the panel above "
    "comes from.\n\n"
    "This is not really a finding about this particular model being weak. Its one-step "
    "glucose accuracy is solid. It is a finding about what recursive forecasting actually "
    "requires that one-shot forecasting does not: some way to anticipate discrete, "
    "behavior-driven events (a meal, a correction dose) before they happen, not just react "
    "to them once logged. Nothing in a few hours of recent glucose and insulin history "
    "reliably signals that a meal is about to happen in the next 5 minutes, so a plain "
    "regression model has no real basis for predicting one, and defaults to predicting "
    "the common case instead. Closing that gap would need a genuinely different kind of "
    "model, most likely one that treats meals and doses as events to detect or plan around "
    "rather than a continuous quantity to regress on, not just a bigger or better-tuned "
    "version of this one."
))

cells.append(md(
    "## 7. Does adding time-of-day actually help?\n\n"
    "Short answer: it changes the model's behavior, but it does not fix the actual "
    "problem, and on this specific window it is arguably worse.\n\n"
    "**The core diagnosis is unchanged.** Meal insulin and carbs are still no better "
    "predicted than just guessing their own mean (-0.005 and +0.007 respectively -- "
    "both still effectively zero), against a real improvement of 27.5 mg/dL for glucose "
    "and 0.64 for background insulin. The predicted meal insulin and carb trajectories "
    "in the plot above still stay flat near zero straight through the real meal at "
    "minute 230. Knowing the time of day does not tell the model *this specific patient "
    "is about to eat right now* -- it is a weak population-level signal, not the kind of "
    "specific trigger a regression model can act on for a single 5-minute-ahead "
    "prediction.\n\n"
    "**What time-of-day did change is background insulin**, whose validation RMSE "
    "improved from 0.32 to 0.30 mg/dL and whose predicted trajectory now drifts and "
    "curves over time instead of sitting at a flat constant, loosely tracking the "
    "general shape (though not the sharp step-changes) of the real basal rate. That "
    "shift is what reshaped the glucose curve too: instead of flatlining at one level, "
    "the recursive prediction now follows a slower drifting curve, dipping down through "
    "the middle of the rollout and climbing back up near the end.\n\n"
    "**That drift happens to line up with reality better at the very end of this "
    "500-minute window, and clearly worse in the middle.** Error in the last 30 minutes "
    "dropped from 49.4 to 17.1 mg/dL, but error in the first 30 minutes roughly doubled "
    "(8.4 to 15.0 mg/dL), and the peak error got both larger (53.4 to 75.4 mg/dL) and "
    "earlier (minute 485 to minute 250). The predicted curve still completely misses the "
    "actual meal-driven spike up to 165 mg/dL around minute 250, drifting down to about "
    "80 while the real value climbs -- if anything, the gap during the part of the "
    "rollout that matters most (right after the missed meal) got wider, not narrower. "
    "The improved tail-end number looks like coincidental convergence for this "
    "particular patient and window, not a fix to the underlying issue.\n\n"
    "**Bottom line:** a cheap input feature was worth trying, and it did surface a real, "
    "if secondary, improvement (background insulin tracks time-of-day structure now "
    "instead of a flat constant). But the actual bottleneck identified in the first pass "
    "-- the model has no way to anticipate a discrete meal event before it happens -- is "
    "still there, confirmed by the same near-zero improvement over guessing the mean for "
    "bolus and carbs. Closing that gap for real would need one of the heavier approaches "
    "discussed earlier: treating meal/dose prediction as its own event-detection problem, "
    "or simulating plausible meals from this patient's historical eating pattern instead "
    "of relying on the regression model's own flat point-estimate."
))

out_path = "notebooks/08_recursive_forecasting.ipynb"
new_notebook = nbformat.v4.new_notebook()
new_notebook["cells"] = cells
new_notebook["metadata"] = {
    "kernelspec": {"display_name": "AZT1D (venv)", "language": "python", "name": "azt1d"},
}
nbformat.write(new_notebook, out_path)
print(f"Wrote {out_path} ({len(cells)} cells)")
