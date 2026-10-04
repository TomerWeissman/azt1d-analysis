# Week 14_9: Sep 14 to Sep 20, 2026

## Hypotheses this week
- **H3, confirmed**: weighting the loss toward danger reduces dangerous errors. Event recall 0.507 to 0.722; Clarke zone D 2.37% to 1.33%.
- **H4, invalidated**: the weighted loss improves overall accuracy. RMSE and MAE are worse than the plain loss.
- **H5, invalidated**: per-patient GA weights beat fixed weights. The earlier gain came from every patient sharing one seed. After the fix it is 41.12 vs 41.18 (CNN-LSTM), and 44.13 for the Transformer.
- **H6, confirmed (qualitative)**: OhioT1DM shows the same pattern as AZT1D.
- **H7, invalidated**: a recursive forecaster that predicts every input can trace the glucose curve. It produces a flat trajectory.
- **H8, supported (qualitative)**: real meal inputs make the recursive forecast usable.

## Findings
- The weighted loss trades overall accuracy for catching danger. The trade is real on both datasets.
- The personalization gain was an artifact of the seed bug, not a real effect.
- The recursive forecast needs real meal inputs to work at all.

## Outcomes
- v1 vs v0: RMSE and MAE worse; danger handling better (numbers in notebook 03).
- v2: event recall 0.507 to 0.722; Clarke zone D 2.37% to 1.33%.
- Personalized CNN-LSTM: 37.44 before the fix, 41.12 after. Fixed weights: 41.18.
- Transformer personalized: 39.38 before the fix, 44.13 after.

## Work done
- Sep 14: GLIMMER v1 weighted loss, [03](notebooks/03_glimmer_v1_weighted_loss.ipynb) (9738f8e).
- Sep 14: GLIMMER v2 clinical metrics, [04](notebooks/04_glimmer_v2_clinical_metrics.ipynb) (84423b6).
- Sep 14: GLIMMER v3, per-patient GA, CNN-LSTM and Transformer, [05](notebooks/05_glimmer_v3_full_replication.ipynb) (69ca208, 05eb0b9).
- Sep 14: MetaboNet loader checked row by row against AZT1D (810f911). OhioT1DM replication, [06](notebooks/06_glimmer_ohiot1dm.ipynb) (ad769f3, 3475f9f).
- Sep 15: presentation summary and ranked theories, [07](notebooks/07_summary.ipynb) (4648073 to 2f5b9da).
- Sep 15: GA seed bug found and fixed. Notebooks 05, 06, and 07 re-run with the fix. Committed in this week's backup.
- Sep 17: recursive forecasting, [08](notebooks/08_recursive_forecasting.ipynb) (d5ff3ef, 897758b, 9a3f435).

## Files
| File | What it is | Status |
|---|---|---|
| [notebooks/03_glimmer_v1_weighted_loss.ipynb](notebooks/03_glimmer_v1_weighted_loss.ipynb) | Weighted loss vs plain loss on RMSE/MAE and region errors | current |
| [notebooks/04_glimmer_v2_clinical_metrics.ipynb](notebooks/04_glimmer_v2_clinical_metrics.ipynb) | Event precision, recall, F1, and Clarke zones | current |
| [notebooks/05_glimmer_v3_full_replication.ipynb](notebooks/05_glimmer_v3_full_replication.ipynb) | Per-patient GA search, CNN-LSTM and Transformer | current; seed-fix version |
| [notebooks/06_glimmer_ohiot1dm.ipynb](notebooks/06_glimmer_ohiot1dm.ipynb) | Same pipeline on OhioT1DM via MetaboNet | current; seed-fix version |
| [notebooks/07_summary.ipynb](notebooks/07_summary.ipynb) | Presentation-style summary of all results | **stale narrative**: text still describes the pre-fix personalization story |
| [notebooks/08_recursive_forecasting.ipynb](notebooks/08_recursive_forecasting.ipynb) | Recursive multi-step forecasting | current |
| [scripts/build_summary_notebook.py](scripts/build_summary_notebook.py) | Builds notebook 07 from saved outputs | current |
| [scripts/build_recursive_notebook.py](scripts/build_recursive_notebook.py) | Builds notebook 08 | current |

## Remember
- Notebook 07's narrative and the repo README still show the pre-fix numbers. Fix them before anything is cited.
- The seed fix is the reason the personalization result changed. Check any other per-patient search for the same bug.

## Open
- Rewrite the 07 narrative for the corrected results.
- Update the repo README numbers.
