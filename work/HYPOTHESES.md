# Hypothesis register

Every claim the project tests, with its status and the evidence behind it. One row per hypothesis. Update the status when a week changes it.

Statuses:
- **Testing**: being tested now.
- **Confirmed**: the evidence supports it, on the data and setup listed.
- **Supported**: the evidence points that way, but the test is preliminary.
- **Not supported**: the test was run and did not show the effect.
- **Invalidated**: the test showed it is false, or a bug explained the earlier result.
- **Not yet tested**: planned, not run.
- **Dropped**: no longer pursued.

## Register

| ID | Hypothesis | Week | Status | Evidence | Next test |
|---|---|---|---|---|---|
| H1 | A plain per-patient CNN-LSTM can forecast 60-minute glucose on AZT1D (GLIMMER's baseline). | 7_9 | Confirmed | [7_9 notebook 02](7_9/notebooks/02_glimmer_v0_baseline.ipynb) | none |
| H2 | ML-Glucose has an unexplored gap to build on. | 7_9 | Dropped | Not checked. Switched to AZT1D on Sep 13. | none |
| H3 | Weighting the loss toward hypo and hyper readings (GLIMMER's fixed weights) reduces dangerous errors. | 14_9 | Confirmed | Event recall 0.507 to 0.722; Clarke zone D 2.37% to 1.33%. [14_9 notebook 04](14_9/notebooks/04_glimmer_v2_clinical_metrics.ipynb) | none |
| H4 | The weighted loss also improves overall accuracy (RMSE, MAE). | 14_9 | Invalidated | Weighted RMSE and MAE are worse than the plain loss. [14_9 notebook 03](14_9/notebooks/03_glimmer_v1_weighted_loss.ipynb) | none |
| H5 | Per-patient genetic-algorithm weights beat fixed weights on RMSE. | 14_9 | Invalidated | Earlier 37.44 vs 41.18 came from every patient sharing one seed. After the fix: 41.12 vs 41.18. [14_9 notebook 05](14_9/notebooks/05_glimmer_v3_full_replication.ipynb) | none |
| H6 | OhioT1DM shows the same pattern as AZT1D. | 14_9 | Confirmed | Same qualitative result. [14_9 notebook 06](14_9/notebooks/06_glimmer_ohiot1dm.ipynb) | none |
| H7 | A recursive forecaster that predicts every input, including meals, can trace the glucose curve over time. | 14_9 | Invalidated | Predicting every input gives a flat trajectory. [14_9 notebook 08](14_9/notebooks/08_recursive_forecasting.ipynb) | none |
| H8 | Using real meal inputs makes the recursive forecast usable. | 14_9 | Supported | Improves on the flat trajectory. Qualitative only. | Put a number on it. |
| H9 | Ordinary 80% and 95% uncertainty bands can raise danger alarms without many false alarms. | 21_9 | Invalidated | They falsely trigger on 68 to 100% of safe readings. [21_9 notebook 01](21_9/notebooks/01_uncertainty_bands.ipynb) | none |
| H10 | Of GARCH, analog ensemble, and conformal bands, one gives clearly better alarms. | 21_9 | Invalidated | At matched 35% or 50% false triggers, all three catch 74 to 76.5% of danger. [21_9 notebook 01](21_9/notebooks/01_uncertainty_bands.ipynb) | none |
| H11 | The plain model has better interval quality than the weighted variants. | 28_9 | Confirmed | Interval AUC 0.797 vs 0.731 (fixed) and 0.730 (personalized), p < 0.0001; best for 22 of 25 patients. [28_9 notebook 03](28_9/notebooks/03_interval_utility_ranking.ipynb) | Check on a second dataset. |
| H12 | Volatility-based bands (EWMA, GARCH) widen when a forecast is about to be wrong. Conformal does not. | 28_9 | Supported | Widest-to-narrowest band ratio: EWMA 1.67x, GARCH 1.39x, conformal 1.10x. The bands still under-track error (error grows about 36x across deciles, band under 2x). [28_9 notebook 04](28_9/notebooks/04_more_uncertainty_methods.ipynb) | Check whether any method tracks error closely. |
| H13 | After fine-tuning on a new patient's first days, epistemic uncertainty shrinks and aleatoric stays flat. | 28_9 | Not supported | Ensemble epistemic flat at about 1.6 mg/dL. Evidential epistemic rises at 1 day in all 5 patients, then partly falls (14.4, 17.3, 15.3, 15.3). Aleatoric does not stay flat for the evidential model. [quick_metrics.csv](28_9/results/quick_metrics.csv) | More patients and seeds, or a larger fine-tune learning rate. |
| H14 | Fine-tuning on a new patient's first days improves RMSE. | 28_9 | Not supported | RMSE flat across 0, 1, 3, and 7 days. Four of five patients improved by about 0.3 mg/dL; Wilcoxon p = 0.125. | Same as H13. |
| H15 | Evidential regression is better calibrated than a deep ensemble for glucose forecasting. | 28_9 | Supported | Mean calibration error 0.064 vs 0.094 at 0 days, and 0.065 vs 0.091 at 7 days. Five patients, one seed. | More patients and seeds. |
| H16 | Patient 27's GARCH bands are under-confident. | 28_9 | Confirmed | Coverage 69%, 86%, and 92% at nominal 55%, 80%, and 90%. One patient. | Check on other patients. |
| H17 | Personalized uncertainty training improves over time for each patient (Direction 2). | 28_9 | Testing | Covered by H13 and H14, which are preliminary and negative so far. | Decide the direction with Prof. Watson. |
| H18 | Teacher forcing separates epistemic from aleatoric error in recurrent forecasting. | 5_10 | Not yet tested | Planned for week 5_10 in the Notion sprint plan. | Set up the smallest run in week 5_10. |
| H19 | Sizing the two band edges separately (wide bottom, narrow top) cuts false alarms without losing danger caught. | 21_9 | Not yet tested | Suggested as a next step in 21_9, not run. | Compare separate edge sizes at matched false-trigger rates. |

## Testing now

- **H17 (Direction 2, personalization)**: preliminary and negative so far (H13, H14). The most reproducible signal is the evidential epistemic spike at 1 day, seen in all 5 patients.
- **H12 (Direction 1, width vs volatility)**: supported, but the journal notes it may not be valuable. Decide with Prof. Watson whether to pursue it.
- **H18 (teacher forcing)**: planned for week 5_10.
