# Week 28_9: Sep 28 to Oct 4, 2026

## Hypotheses this week
- **H11, confirmed**: the plain model has better interval quality than the weighted variants. AUC 0.797 vs 0.731 and 0.730; p < 0.0001; best for 22 of 25 patients.
- **H12, supported**: volatility bands (EWMA 1.67x, GARCH 1.39x) widen with error size; conformal does not (1.10x). Still under-tracks error.
- **H13, not supported (quick run)**: after fine-tuning, epistemic uncertainty shrinks and aleatoric stays flat. The evidential epistemic spikes at 1 day in all 5 patients, and the ensemble's epistemic stays flat.
- **H14, not supported (quick run)**: fine-tuning improves RMSE. Flat across 0, 1, 3, and 7 days.
- **H15, supported (preliminary)**: evidential regression is better calibrated than a deep ensemble. Five patients, one seed.
- **H16, confirmed (one patient)**: patient 27's GARCH bands are under-confident.
- **H17, testing**: Direction 2 (personalization over time). Preliminary and negative so far, through H13 and H14.

## Findings
- Interval quality ranks the plain model first. Weighting the loss does not help the bands.
- Volatility methods adapt their width. Conformal does not. Even so, bands are far less responsive than the error they are meant to track.
- Fine-tuning on up to 7 days of a new patient's data showed no clear accuracy gain in this run.
- The two uncertainty methods split total uncertainty very differently: about 1% epistemic for the ensemble, about 57% for the evidential model at 7 days.

## Outcomes
- Patient 27 (573 days): plain RMSE 23.2, fixed-weight RMSE 30.4. Training took about 8 minutes.
- Interval AUC: plain 0.797, fixed 0.731, personalized 0.730.
- Six methods, alarm curves nearly identical: AUC 0.823 to 0.833.
- Fine-tuning (25.1 minutes): ensemble RMSE 19.4 to 19.1; evidential 19.1 to 18.8. Evidential epistemic 14.4, 17.3, 15.3, 15.3 mg/dL at 0, 1, 3, and 7 days.

## Work done
- Sep 28: HUPA-UCM loader and patient 27 notebook, [01](notebooks/01_hupa_ucm_one_patient.ipynb) (3c8ffd4, 684a60d). Added the HUPA loader and a training script.
- Sep 28: literature check against the HUPA-UCM uncertainty paper (Tan and McBeth, 2026) and a patient 27 comparison. No commit.
- Sep 29: interval-utility ranking, [03](notebooks/03_interval_utility_ranking.ipynb) (c3d0aa1). Also fixed a bug: HUPA files under `data/raw` were being loaded as AZT1D subjects.
- Sep 29: EWMA, locally weighted conformal, and time-of-day quantiles added. Six-method comparison, [04](notebooks/04_more_uncertainty_methods.ipynb) (5086990, 7c9edff).
- Sep 29: fine-tuning experiment (b0c9e53).
- Oct 4: week folders and the restructure (25763df).

## Files
| File | What it is | Status |
|---|---|---|
| [notebooks/01_hupa_ucm_one_patient.ipynb](notebooks/01_hupa_ucm_one_patient.ipynb) | Patient 27 over its whole record: forecasts, GARCH bands, Clarke trends | current |
| [notebooks/03_interval_utility_ranking.ipynb](notebooks/03_interval_utility_ranking.ipynb) | Ranks the three loss variants by interval AUC, not RMSE | current (supports H11) |
| [notebooks/04_more_uncertainty_methods.ipynb](notebooks/04_more_uncertainty_methods.ipynb) | Six uncertainty methods: alarm curves, then band width vs error size | current (supports H12) |
| [scripts/train_hupa_patient.py](scripts/train_hupa_patient.py) | Trains the two patient-27 models | current |
| [scripts/build_hupa_notebook.py](scripts/build_hupa_notebook.py) | Builds notebook 01 | current |
| [scripts/build_interval_utility_notebook.py](scripts/build_interval_utility_notebook.py) | Builds notebook 03 | current |
| [scripts/build_method_comparison_notebook.py](scripts/build_method_comparison_notebook.py) | Builds notebook 04 | current |
| [scripts/run_finetune_uncertainty_experiment.py](scripts/run_finetune_uncertainty_experiment.py) | Fine-tuning experiment, H13 and H14 (25 minutes) | current |
| [results/quick_metrics.csv](results/quick_metrics.csv) | Per-patient, per-budget, per-method metrics from the experiment | current |
| [figures/quick_results.png](figures/quick_results.png) | Three panels: RMSE, uncertainty, and calibration vs budget | current |

## Remember
- The fine-tuning learning rate (1e-5) may be too small to move the model. Untested.
- Every number here is one seed. The fine-tuning experiment uses 5 patients.
- About 8% of pooled HUPA-UCM readings are straight-line filler. Unresolved.
- Patient 27 is 53% of the pooled HUPA-UCM rows. Check per-patient numbers before pooling.
- The Notion journal has not been updated with the fine-tuning results.
- The claim that no one has studied personalized uncertainty over time is unverified.

## Open
- Decide with Prof. Watson: Direction 1 (width vs volatility), Direction 2 (personalization), or both.
- Test H13 and H14 with more patients or seeds, or a larger fine-tune learning rate.
- Resolve the HUPA filler question.
