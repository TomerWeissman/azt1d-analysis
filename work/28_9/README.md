# Week 28_9: Sep 28 to Oct 4

## Completed
- Sep 28: HUPA-UCM loader and patient 27 whole-record notebook (`notebooks/01_hupa_ucm_one_patient.ipynb`) (3c8ffd4, 684a60d)
- Sep 28: literature check against the HUPA-UCM uncertainty paper (Tan and McBeth, 2026), with a patient 27 comparison. No commit.
- Sep 29: interval-utility ranking (`notebooks/03_interval_utility_ranking.ipynb`) (c3d0aa1)
- Sep 29: fixed a bug where HUPA files under `data/raw` were loaded as AZT1D subjects. Moved them to `data/hupa_ucm/` (c3d0aa1)
- Sep 29: added EWMA, locally weighted conformal, and time-of-day quantiles. Six-method comparison (`notebooks/04_more_uncertainty_methods.ipynb`) (5086990, 7c9edff)
- Sep 29: fine-tuning uncertainty experiment (`scripts/run_finetune_uncertainty_experiment.py`; output in `results/` and `figures/`) (b0c9e53)

## Outcomes
- Patient 27 (573 days): plain RMSE 23.2, fixed-weight 30.4. About 8% of pooled readings are straight-line filler (unresolved).
- Patient 27 scores below the paper on point metrics (hypo sensitivity 0.18 vs 0.63 pooled). Its GARCH bands are under-confident: coverage is 69%, 86%, and 92% at nominal 55%, 80%, and 90%.
- The plain model has the best interval AUC: 0.797 vs 0.731 and 0.730 (p < 0.0001), and is best for 22 of 25 patients.
- The six methods' alarm curves nearly coincide (AUC 0.823 to 0.833).
- Band width vs error size: EWMA widens 1.67x from easiest to hardest, GARCH 1.39x, conformal 1.10x. Conformal is flat by design.
- Fine-tuning: RMSE is flat across budgets. The evidential epistemic spikes at 1 day in all 5 patients. At 7 days, epistemic is 0.8% of variance for the ensemble and 57% for evidential. Runtime was 25.1 minutes.

## Open
- The Notion journal does not have the fine-tuning results yet.
- The novelty claim for personalized uncertainty training is unverified.
- The HUPA straight-line filler is unresolved.
- Fine-tuning: 5 patients and 1 seed. More patients or seeds are needed before citing it.
