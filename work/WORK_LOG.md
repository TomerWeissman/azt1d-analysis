# Claude work log: AZT1D Analysis

Machine-readable record of work done with Claude in this repo. For exact commits, files, and numbers, read this file. The weekly READMEs in `work/<week>/` are the short human summaries.

## Format

- Each entry starts with a heading: `## <YYYY-MM-DD> | <week folder> | <title>`
- Then `key: value` lines in this order: `type`, `status`, `files`, `commits`, `runtime` (optional), `outcome`, `notes` (optional)
- `type`: analysis, notebook, script, library, experiment, fix, docs, research, correction
- `status`: done, uncommitted, open
- Entries are chronological. New entries go at the bottom.
- Old entries are never edited. A correction is a new entry with `type: correction` that names the entry it fixes.
- Entries before this log existed were backfilled from git history and session notes. The Sep 15 seed-fix date and the Sep 28 literature-check date are approximate.

---

## 2026-09-13 | 7_9 | Initial AZT1D exploration and GLIMMER v0 baseline
type: notebook
status: done
files: notebooks/01_data_exploration.ipynb, notebooks/02_glimmer_v0_baseline.ipynb, src/azt1d/
commits: 49680f3
outcome: plain CNN-LSTM v0 baseline, one model per patient, 60-minute horizon, no region weighting yet.

## 2026-09-14 | 14_9 | GLIMMER v1: region-aware weighted loss
type: notebook
status: done
files: notebooks/03_glimmer_v1_weighted_loss.ipynb, src/azt1d/glimmer/losses.py
commits: 9738f8e
outcome: weighted loss helps hypo and hyper regions, makes overall RMSE and MAE worse than v0.

## 2026-09-14 | 14_9 | GLIMMER v2: clinical metrics
type: notebook
status: done
files: notebooks/04_glimmer_v2_clinical_metrics.ipynb, src/azt1d/glimmer/clinical.py
commits: 84423b6
outcome: v1 vs v0: event recall 0.507 to 0.722; Clarke zone D 2.37% to 1.33%.

## 2026-09-14 | 14_9 | GLIMMER v3: per-patient genetic algorithm, CNN-LSTM and Transformer
type: notebook
status: done
files: notebooks/05_glimmer_v3_full_replication.ipynb, src/azt1d/glimmer/ga.py, src/azt1d/glimmer/model.py
commits: 69ca208, 05eb0b9
outcome: pre-fix result: personalized CNN-LSTM RMSE 37.44 vs fixed weights 41.18. Invalidated by the seed fix (see the next-but-one entry).

## 2026-09-14 | 14_9 | MetaboNet loader and OhioT1DM replication
type: library
status: done
files: src/azt1d/metabonet.py, notebooks/06_glimmer_ohiot1dm.ipynb, README.md
commits: 810f911, ad769f3, 3475f9f, 3d15a3a
outcome: MetaboNet loader checked row by row against AZT1D (CGM exact on 98.2-100% of readings). OhioT1DM shows the same qualitative pattern as AZT1D.

## 2026-09-15 | 14_9 | Consolidated summary notebook and theories for the paper gap
type: notebook
status: done
files: notebooks/07_summary.ipynb, scripts/build_summary_notebook.py
commits: 4648073, d64ce08, ffcc6b3, 7e47f67, e66421f, 933f425, 02dc834, 2f5b9da
outcome: presentation-style walkthrough built from saved results. Ranked theories for the gap to the paper.

## 2026-09-15 | 14_9 | GA seed bug: every patient started from seed=0; fixed to seed=sid
type: fix
status: uncommitted
files: src/azt1d/glimmer/ga.py, notebooks/05_glimmer_v3_full_replication.ipynb, notebooks/06_glimmer_ohiot1dm.ipynb, notebooks/07_summary.ipynb, notebooks/02_glimmer_v0_baseline.ipynb, data/processed/checkpoints_pre_seedfix_backup_2026-09-15/
commits: none
outcome: after the fix, personalized CNN-LSTM RMSE 41.12 vs fixed 41.18 (no gain). Transformer personalized 39.38 to 44.13. Old numbers are in the backup folder.
notes: README and 07_summary narrative still describe the pre-fix result. The four notebooks are not committed yet.

## 2026-09-17 | 14_9 | Recursive multi-step forecasting
type: notebook
status: done
files: notebooks/08_recursive_forecasting.ipynb, src/azt1d/glimmer/recursive.py, src/azt1d/glimmer/sequences.py, src/azt1d/glimmer/model.py, scripts/build_recursive_notebook.py
commits: d5ff3ef, 897758b, 9a3f435
outcome: predicting every input flattened the curve. A time-of-day feature did not help. Using real meal inputs improved the recursive curve.

## 2026-09-22 | 21_9 | Capstone research paper draft
type: docs
status: done
files: capstone_paper_draft.md, capstone_paper_draft.html
commits: 3c02cea
outcome: draft with section headings, methods, and results. Table 5.2 still has pre-fix personalized numbers.
notes: the untracked capstone_paper_draft.html is not committed.

## 2026-09-26 | 21_9 | Stage 2: uncertainty bands on the fixed-weight model
type: notebook
status: done
files: notebooks_2/01_uncertainty_bands.ipynb, scripts/build_uncertainty_notebook.py, src/azt1d/glimmer/uncertainty.py, requirements.txt
commits: 47546d8, 0e4340a, a7a636e
outcome: 80% and 95% bands falsely trigger on 68-100% of safe readings. Raw forecast alone: 32% false triggers, catches 72.2% of danger. At matched 35% false triggers, conformal, GARCH, and analog catch 74-76.5% of danger. Most false triggers come from the top edge.

## 2026-09-26 | 21_9 | One-patient view: alarm plots and Clarke grid
type: notebook
status: done
files: notebooks_2/02_one_patient_view.ipynb, scripts/build_one_patient_notebook.py
commits: 495c937, 29bb669
outcome: subject 1 Clarke grids and zone trend lines. Growing the GARCH band moves predictions from zone A into B and C.

## 2026-09-28 | 28_9 | HUPA-UCM loader and patient 27 whole-record notebook
type: notebook
status: done
files: src/azt1d/hupa.py, scripts/train_hupa_patient.py, scripts/build_hupa_notebook.py, notebooks_3/01_hupa_ucm_one_patient.ipynb, src/azt1d/glimmer/uncertainty.py
commits: 3c8ffd4, 684a60d
runtime: about 8 min training for patient 27 (two models)
outcome: patient 27 (573 days): plain RMSE 23.2, fixed-weight 30.4. About 8% of pooled readings are straight-line filler (unresolved).

## 2026-09-28 | 28_9 | Literature check and patient 27 vs the HUPA-UCM uncertainty paper
type: research
status: done
files: none
commits: none
outcome: Tan and McBeth (2026) compared. Patient 27 scores below the paper on point metrics (hypo sensitivity 0.18 vs 0.63 pooled). Its GARCH bands are under-confident (coverage 69/86/92% at nominal 55/80/90%).
notes: date is approximate.

## 2026-09-29 | 28_9 | Interval-utility ranking and loader collision fix
type: notebook
status: done
files: notebooks_2/03_interval_utility_ranking.ipynb, scripts/build_interval_utility_notebook.py, src/azt1d/glimmer/uncertainty.py, src/azt1d/hupa.py, .gitignore
commits: c3d0aa1
outcome: plain model has the best interval AUC: 0.797 vs 0.731 and 0.730 (p < 0.0001), best for 22 of 25 patients. Also fixed a bug where HUPA files under data/raw were loaded as AZT1D subjects. The files now live in data/hupa_ucm/.

## 2026-09-29 | 28_9 | Six uncertainty methods and band-width analysis
type: notebook
status: done
files: notebooks_2/04_more_uncertainty_methods.ipynb, scripts/build_method_comparison_notebook.py, src/azt1d/glimmer/uncertainty.py
commits: 5086990, 7c9edff
outcome: alarm curves nearly identical (AUC 0.823-0.833). Widest-to-narrowest band ratio across error bins: EWMA 1.67x, GARCH 1.39x, time-of-day 1.29x, analog 1.20x, normalized conformal 1.21x, conformal 1.10x.

## 2026-09-29 | 28_9 | Fine-tuning uncertainty experiment: deep ensemble vs deep evidential
type: experiment
status: done
files: scripts/run_finetune_uncertainty_experiment.py, results/quick_metrics.csv, figures/quick_results.png
commits: b0c9e53
runtime: 25.1 min
outcome: RMSE flat across 0, 1, 3, and 7 days of adaptation. Evidential epistemic spikes at 1 day in all 5 patients. At 7 days, epistemic is 0.8% of total variance for the ensemble and 57% for evidential.
notes: 5 target patients, 1 seed. Notion journal not yet updated with these results.

## 2026-10-04 | 28_9 | Restructure: code moved into week folders
type: docs
status: done
files: work/7_9/notebooks/, work/14_9/notebooks/, work/14_9/scripts/, work/21_9/notebooks/, work/21_9/scripts/, work/21_9/docs/, work/28_9/notebooks/, work/28_9/scripts/, work/28_9/results/, work/28_9/figures/, README.md
commits: 25763df (the seed-fix notebooks 02, 05, 06, 07 are not in it)
outcome: every notebook and script moved out of notebooks/, notebooks_2/, notebooks_3/, scripts/, results/, figures/. src/azt1d stays shared.
notes: path map. notebooks/01, 02 -> 7_9/notebooks/. notebooks/03, 04, 05, 06, 07, 08 -> 14_9/notebooks/. notebooks_2/01, 02 -> 21_9/notebooks/. notebooks_2/03, 04 and notebooks_3/01 -> 28_9/notebooks/. scripts/build_summary_notebook.py and build_recursive_notebook.py -> 14_9/scripts/. scripts/build_uncertainty_notebook.py and build_one_patient_notebook.py -> 21_9/scripts/. build_interval_utility, build_method_comparison, build_hupa, train_hupa_patient, run_finetune -> 28_9/scripts/. capstone_paper_draft.md/.html -> 21_9/docs/. results/quick_metrics.csv -> 28_9/results/. figures/quick_results.png -> 28_9/figures/. Earlier entries keep their original paths; use this map to find them.
notes: verified. All 13 notebooks resolve the repo root from their new folders. notebooks 01 (7_9) and 03 (28_9) were executed end to end from their new folders with no cell errors. Outputs went to the scratch folder, not the committed notebooks. Uncommitted: 7_9/notebooks/02, 14_9/notebooks/05, 06, 07 (seed-fix changes). The old copies of those four are still in git history at their old paths.

## 2026-10-04 | 5_10 | Week start: goals and carry-forward
type: docs
status: done
files: work/5_10/README.md, work/5_10/results/quick_metrics.csv, work/5_10/figures/quick_results.png, work/HYPOTHESES.md, work/OVERVIEW.md
commits: pending (this week's start commit)
outcome: started week 5_10 a day early at the user's request. Chose the learning-curve and unseen-patient test to find which uncertainty is epistemic. Carried forward the fine-tuning results (copies; originals stay in 28_9). Overview sentence corrected: the teacher-forcing claim was reframed.
notes: teacher forcing does not separate epistemic from aleatoric on its own. It is kept as an optional diagnostic (H21). Test needs approval for run time before the full run.

## 2026-10-04 | 5_10 | Literature review: uncertainty in glucose forecasting, and gaps
type: research
status: done
files: work/5_10/docs/literature_review_uncertainty.md, work/5_10/README.md
commits: pending (this week's commit)
outcome: seven gaps. Not found in the literature: alarms built from bands per patient (G1), confidence over a new patient's data (G2), a glucose method comparison of epistemic labels (G3), a noise ceiling for 60-minute forecasts (G4), conformal 60-minute CGM forecasts (G6). Partly found: width tracking error (G5), sensor error in a ceiling (G7). Position paper (arXiv 2505.23506) supports the method disagreement we found.
notes: most sources read from abstracts and search summaries. Several full texts did not open (ScienceDirect 403, some arXiv PDFs unparsed). Verify before citing. Not yet searched: medical-device alarm standards (ISO 15197) that define acceptable false-alarm rates.
