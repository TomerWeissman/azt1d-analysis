---
name: hypothesis_testing
description: Test one specific hypothesis about this glucose-forecasting project with the least training and code that can answer it. Asks what to test, proposes a minimal experiment that can prove or disprove the claim, reuses saved checkpoints and results first, makes 2 to 4 graphs, and gives a verdict. Use when the user says "test whether", "is it true that", "does X help", "validate this idea", "check my hypothesis", or runs /hypothesis_testing.
---

# Hypothesis testing

Goal: a supported, refuted, or inconclusive answer with 2 to 4 graphs, using the least training and code that can settle the question. Default budget: 30 minutes of compute and build time.

## 1. Ask first

Ask only what is missing, in one message:

1. **The claim**, in one sentence. Example: "Epistemic uncertainty shrinks as a patient's own data grows."
2. **What would prove it wrong**, as a number or a picture. Example: "epistemic falls by less than 20% from 0 to 7 days."
3. **Which models and patients.** Default: plain (`cnn_lstm_v0`), fixed weights (`cnn_lstm_v1`), personalized (`cnn_lstm_v3_tuned`), all 25 AZT1D patients.
4. **Time budget.** Default: 30 minutes.

Use AskUserQuestion when the answer is one of a few options. Use free text otherwise.

## 2. Check whether something already answers it

Before any training, look for an existing answer:

- Saved results: `work/*/results/`, `work/*/figures/`, `work/WORK_LOG.md`, and the week READMEs.
- Saved checkpoints in `data/processed/checkpoints/<run>/subject_<id>.pt`. They store each patient's test predictions. `forecast_frames` rebuilds the validation predictions from the weights.
- Uncertainty-band questions need no training if a checkpoint exists. `BandEngine` runs on it directly.

If an existing answer covers the question, say so and skip training.

## 3. Propose the minimal test

Write a plan under 10 lines. Say what it will not test.

- **Data:** one or a few patients first. Use all patients only if the first pass looks promising.
- **Training:** none if a checkpoint exists. Otherwise one seed, the fewest epochs that show a trend, one patient, timed on a small slice first. Extrapolate the runtime. Ask before anything over 10 minutes.
- **Graphs:** 2 to 4. Each answers one sub-question and marks the direction that would support the claim.
- **Threshold:** write the verdict threshold down before running, from step 1.

If the plan fits the time budget, go ahead. If it does not, stop and ask.

## 4. Build the graphs

- One script in the current week's `scripts/` folder, figures in its `figures/` folder. Use a notebook only if the user asks for one. Week naming and layout are in the `log-work` skill.
- Reuse the project code (list below). Do not rewrite loading, windowing, or training.
- Figure menu. Pick what fits:
  - **Method or model A vs B:** one dot or line per patient, paired, with the mean.
  - **Something changes with a budget or parameter:** metric vs x, one line per patient, mean highlighted.
  - **Calibration:** empirical coverage vs nominal level, with the diagonal.
  - **Uncertainty tracks error:** band width by error decile.
  - **Alarm trade-off:** danger caught vs false-trigger rate, at matched operating points.
- Title each figure with its sub-question. No prose inside the figure.
- Look at every figure before reporting on it.

Minimal starting skeleton, using a saved checkpoint:

```python
import sys
from pathlib import Path
ROOT = next(p for p in Path(__file__).resolve().parents if (p / "pyproject.toml").exists())
WEEK = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from azt1d import loading, plotting
from azt1d.glimmer import checkpoint as ckpt, uncertainty as unc
plotting.apply_style()
df = loading.load_real_dataset(ROOT / "data" / "raw", ROOT / "data" / "processed")
res = ckpt.load_result(ROOT / "data/processed/checkpoints/cnn_lstm_v0", 1)
val, test = unc.forecast_frames(res, df[df.subject_id == 1].reset_index(drop=True))
eng = unc.BandEngine(val, test, analogs=False)  # analogs=False skips the slow distance-based methods
```

## 5. Verdict

Report in this order:

1. **Supported, refuted, or inconclusive**, against the threshold from step 1.
2. **The numbers**, with the per-patient spread and how many patients moved in the predicted direction.
3. **What the test cannot show:** patient count, seeds, data period, and whether one patient drives a pooled result.
4. **The next test** that would settle what is left, as one line.

Plain language. No em dashes. Do not claim more than the data shows.

## 6. Record it

Run the `log-work` skill. Log the hypothesis, prediction, threshold, and verdict, with type `analysis` or `experiment`. Put the verdict under Outcomes and the open question under Open in the week README.

## Reusable project code

- `azt1d.loading`: `load_real_dataset(raw_dir, processed_dir)`
- `azt1d.hupa`: `load_subject(sid)`, `list_subject_ids(dir)`
- `azt1d.glimmer.checkpoint`: `load_result(dir, sid)`, `load_model(result)`
- `azt1d.glimmer.uncertainty`: `forecast_frames(result, df_subject)`, `BandEngine(cal, test, analogs=True|False)`, `.bands(method, level)`, `trigger_metrics`, `trigger_auc`, `level_for_false_trigger_rate`, `swapped_prediction`, `METHODS`, `ALL_METHODS`
- `azt1d.glimmer.clinical`: `clarke_zone_percentages`, `clarke_error_grid_zones`, `draw_clarke_grid`, `dysglycemia_event_metrics`
- `azt1d.glimmer.train`: `prepare_subject_data`, `train_prepared_model`, `train_subject_model`, `run_with_checkpoints`
- `azt1d.plotting`: `apply_style()`, `CATEGORICAL`

Checkpoint runs in `data/processed/checkpoints/`: `cnn_lstm_v0` (plain), `cnn_lstm_v1` (fixed weights), `cnn_lstm_v3_tuned` (personalized), `cnn_transformer_*`, `ohiot1dm_*`, `hupa_ucm_cnn_lstm_v0` and `_v1`.

## Pitfalls already hit in this project

- **Seeds:** each ensemble member and each patient needs its own seed. One shared seed made every patient's genetic-algorithm search identical.
- **Test data stays out** of fitting, calibration, and early stopping. Calibrate on validation only.
- **Wider bands only add alarms.** Compare methods at a matched false-trigger rate, not at one band size.
- **A pooled result can be one patient.** In HUPA-UCM, patient 27 is 53% of the rows. Check the per-patient view.
- **Five patients or one seed is preliminary.** Say so in the verdict.
- **HUPA-UCM has straight-line filler** (about 8% of pooled readings). Check it before trusting a patient.
