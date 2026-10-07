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

## 2026-10-04 | 5_10 | Cheap tests H23, H24, H25 (look-alike noise bound, band alarms, conformal coverage)
type: experiment
status: done
files: work/5_10/scripts/cheap_tests_h23_h25.py, work/5_10/results/h23_lookalike.txt, work/5_10/results/h24_alarms.csv, work/5_10/results/h25_conformal_coverage.csv, work/5_10/results/verdicts_h23_h25.txt, work/HYPOTHESES.md, work/5_10/README.md
commits: pending (this week's commit)
runtime: 44 s
outcome: H23 inconclusive (look-alike upper bound 106% of model error). H24 not supported (band alarm 10.2 onsets/day vs point 7.7; test sensitivity 0.73 vs 0.67). H25 not supported (16 of 25 patients inside 75% to 85%; median 79.1%).
notes: the first H24 run had a bug (tuned on test-period bands). Fixed before the reported run: thresholds now tuned on validation bands only. Caveat for H24: validation bands are in-sample for their own tuning, so tuned sensitivity is approximate.

## 2026-10-04 | 5_10 | Literature check and predictability map (H26)
type: experiment
status: done
files: work/5_10/scripts/predictability_map.py, work/5_10/results/predictability_map.csv, work/5_10/results/verdict_h26.txt, work/5_10/figures/predictability_map.png, work/HYPOTHESES.md, work/5_10/README.md
commits: pending (this week's commit)
runtime: 41 s
outcome: H26 not supported. Look-alike noise bounds are 78% to 118% of model error across situations, too loose to separate. Error varies strongly by situation (RMSE 26 mg/dL steady, 61 mg/dL fast rise above 250). 80% band coverage varies from 64% to 95% by situation.
notes: literature check found region-based error grids (CG-EGA) and state-switching models, but no per-region noise decomposition. Two key papers did not open (state-switching preprint: title only; CG-EGA: 403). Verify before claiming novelty.

## 2026-10-04 | 5_10 | Situation-specific bands (H27)
type: experiment
status: done
files: work/5_10/scripts/situation_bands.py, work/5_10/results/situation_bands.csv, work/5_10/results/verdict_h27.txt, work/5_10/figures/situation_bands.png, work/HYPOTHESES.md, work/5_10/README.md
commits: pending (this week's commit)
runtime: 40 s
outcome: H27 not supported (narrowly). Situation-specific widths: 18 of 25 situations inside 75% to 85%, rule needed 20. Current GARCH bands: 13 of 25. Coverage spread narrowed from 64% to 95% to 67% to 83%. Two situations used the pooled width.
notes: a first run failed on a column-assignment bug, fixed before the reported run. The remaining misses are high-glucose slow falls and flat readings, consistent with validation-to-test drift.

## 2026-10-04 | 5_10 | H28 (OhioT1DM situation bands) and H29 (drift)
type: experiment
status: done
files: work/5_10/scripts/ohio_situation_bands.py, work/5_10/scripts/drift.py, work/5_10/results/h28_ohio_situations.csv, work/5_10/results/h28_ohio_patients.csv, work/5_10/results/verdict_h28.txt, work/5_10/results/h29_drift.csv, work/5_10/results/verdict_h29.txt, work/5_10/figures/h28_ohio_situation_bands.png, work/5_10/figures/h29_drift.png, work/HYPOTHESES.md, work/5_10/README.md
commits: pending (this week's commit); pre-registration pushed before the runs
runtime: H28 4 s, H29 about 40 s
outcome: H28 supported: OhioT1DM coverage error 6.7 to 3.9 points (41% cut), 9 of 12 patients closer, bands 12% narrower. H29 not supported: median test/validation RMSE 1.03; per-patient changes track glucose variability (r = 0.77).
notes: H28 width gain is partly because GARCH over-covers on Ohio (83.2% overall). H29 corrects earlier wording: validation-to-test changes are patient variability, not systematic decay.

## 2026-10-04 | 5_10 | Learning curve for the noise ceiling (H30)
type: experiment
status: done
files: work/5_10/scripts/learning_curve_ceiling.py, work/5_10/results/learning_curve_runs.csv, work/5_10/results/verdict_h30.txt, work/5_10/figures/learning_curve_ceiling.png, work/HYPOTHESES.md, work/5_10/README.md
commits: pending (this week's commit); pre-registration pushed before the run
runtime: 1978 s (250 trainings: 25 patients x 5 fractions x 2 seeds)
outcome: H30 not supported. Asymptote 1054 vs full-data MSE 1074 (98%); bootstrap 90% interval 93% to 99%. More data of this kind cuts error by about 2%.
notes: the remaining error is noise plus model/input limits; a capacity or input test is needed to split them. Scaling used full-train statistics at every fraction (train-only information).

## 2026-10-04 | 5_10 | GARCH x situation hybrid bands (H31)
type: experiment
status: done
files: work/5_10/scripts/hybrid_bands.py, work/5_10/results/h31_hybrid_ohiot1dm.csv, work/5_10/results/h31_hybrid_azt1d.csv, work/5_10/results/verdict_h31.txt, work/5_10/figures/h31_hybrid_summary.png, work/5_10/figures/example_patient_588_three_bands.png, work/HYPOTHESES.md, work/5_10/README.md
commits: pending (this week's commit); pre-registration pushed before the run
runtime: about 1 min
outcome: H31 supported narrowly. OhioT1DM: hybrid 3.1 vs situation 4.0 vs GARCH 6.7 points from target (21% better than situation), 7 of 12 patients better (the rule's minimum). AZT1D: 3.1 vs 3.8 vs 6.0, 14 of 25 better.
notes: first run had no sensor-range clipping, so bands went below 0. Rerun with clipping to 40-400 mg/dL, as every earlier band; verdict unchanged. OhioT1DM is the same data as H28, so this is not a second fresh dataset.

## 2026-10-04 | 5_10 | Literature check: Mondrian conformal prediction
type: research
status: done
files: work/5_10/docs/literature_review_uncertainty.md
commits: pending (this week's commit)
outcome: situation-only bands are Mondrian conformal prediction; the hybrid is normalized conformal inside Mondrian groups. Neither method is new. No glucose/CGM paper found using group-conditional conformal calibration or measuring band coverage by level and trend. Paper contribution reframed as the application and finding, not the method.
notes: abstracts and summaries only. Paper must cite Mondrian and normalized conformal, and compare against plain conformal and CQR.

## 2026-10-04 | 5_10 | Verification checks H32-H35 for situation-grouped conformal bands
type: experiment
status: done
files: work/5_10/scripts/verification_checks.py, work/5_10/results/verify_h32_alternatives.csv, verify_h33_shuffle.csv, verify_h34_models.csv, verify_h35_bootstrap.csv, verdicts_h32_h35.txt, work/5_10/figures/verify_h32_alternatives_ohiot1dm.png, verify_h32_alternatives_azt1d.png, verify_h33_shuffle.png, verify_h34_models.png, verify_h35_bootstrap.png, work/HYPOTHESES.md, work/5_10/README.md
commits: pending (this week's commit); pre-registration pushed before the run
runtime: 157 s
outcome: H32 not supported (Mondrian vs CQR 18% on OhioT1DM, rule 20%; 52% on AZT1D; hybrid beats CQR on both). H33 supported (real 4.0 vs shuffled 6.6-6.9). H34 supported (23-47% vs GARCH, 4 combos). H35 supported (Mondrian-vs-GARCH 90% CI 0.3 to 3.6; hybrid-vs-Mondrian CI -0.5 to 2.1, not robust).
notes: CQR here uses summary features (forecast, level, trend, GARCH sigma) and half of validation for training. A stronger CQR (full input window) could close the gap further. No seed variation of the forecasting models; bootstrap is over patients only.

## 2026-10-04 | 5_10 | Tuned full-window CQR vs situation bands (H36)
type: experiment
status: done
files: work/5_10/scripts/strong_cqr.py, work/5_10/results/h36_strong_cqr.csv, work/5_10/results/verdict_h36.txt, work/5_10/figures/h36_strong_cqr.png, work/HYPOTHESES.md, work/5_10/README.md
commits: pending (this week's commit); pre-registration pushed before the run
runtime: 144 s
outcome: H36 supported by its rule. OhioT1DM: Mondrian 4.0 vs CQR 4.9 points (19% better, rule 10%), widths 87 vs 94; patient-bootstrap interval for the gap -0.7 to 2.4 (crosses zero). AZT1D: Mondrian 3.8 vs CQR 8.9, interval 2.4 to 6.1.
notes: tuning picked the smallest setting in the grid for both quantiles on both datasets (learning rate 0.05, 15 leaves), so a grid extending smaller could do better. Bootstrap holds the fitted models fixed (not refit per resample). CQR conformalizes on 30% of validation, Mondrian on all of it.

## 2026-10-04 | 5_10 | Level-free coverage-vs-width curves (H37)
type: experiment
status: done
files: work/5_10/scripts/coverage_width_curves.py, work/5_10/results/h37_curves.csv, work/5_10/results/verdict_h37.txt, work/5_10/figures/h37_coverage_width.png, work/HYPOTHESES.md, work/5_10/README.md
commits: pending (this week's commit); pre-registration pushed before the run
runtime: 77 s
outcome: H37 supported by its rule, small effect. At GARCH's widths, situation bands catch +0.1 to +1.8 points (OhioT1DM, 7 of 7) and +0.1 to +1.0 (AZT1D, 6 of 6). Curves nearly overlap. Interval score at 80%: OhioT1DM CQR 130, Mondrian 131, conformal 133, hybrid 135, GARCH 138; AZT1D CQR 111, Mondrian 114. At 95% CQR also best.
notes: changes the paper claim. Situation bands are not more efficient overall; their gain is per-situation calibration. CQR is the overall-efficiency leader on interval score.

## 2026-10-04 | 5_10 | Band width and catch rate by Clarke zone (descriptive)
type: analysis
status: done
files: work/5_10/scripts/bands_by_clarke_zone.py, work/5_10/results/bands_by_clarke_zone.csv, work/5_10/figures/bands_by_clarke_zone.png, work/5_10/README.md
commits: pending (this week's commit)
outcome: situation band width is nearly flat across Clarke zones (OhioT1DM A 87, B 86, D 86 mg/dL) while GARCH widens in B, C, D. Caught in zone D: situation 41% vs GARCH 62% (OhioT1DM), 22% vs 39% (AZT1D). Zone A: 98% vs 96%.
notes: important caveat for the paper. Clarke zone depends on the realised error, which no band can know in advance; GARCH partly tracks it through recent errors.

## 2026-10-04 | 5_10 | Combined band and Clarke zone D (H38)
type: experiment
status: done
files: work/5_10/scripts/hybrid_by_clarke_zone.py, work/5_10/results/h38_zone_catch.csv, work/5_10/results/verdict_h38.txt, work/5_10/figures/h38_zone_catch.png, work/HYPOTHESES.md, work/5_10/README.md
commits: pending (this week's commit); pre-registration pushed before the run
runtime: about 1.5 min
outcome: H38 not supported. Zone D caught: OhioT1DM GARCH 62%, situation 41%, combined 54% (-8 vs GARCH); AZT1D 39%, 22%, 34% (-5, just outside the rule). Combined recovers about 60-70% of the gap. Per-situation error stays 3.1 vs GARCH 6.0-6.7.
notes: clear trade-off. GARCH is best on dangerous misses, situation bands best on per-situation honesty, the combined band sits between on both. Zones C and E omitted (too few points).

## 2026-10-06 | 5_10 | Teacher-forcing encoder-decoder bottleneck PCA (H39)
type: experiment
status: done
files: work/5_10/scripts/teacher_forcing_autoencoder.py, work/5_10/results/autoencoder_pca.csv, work/5_10/results/autoencoder_summary.txt, work/5_10/figures/autoencoder_pca.png, work/HYPOTHESES.md, work/5_10/README.md
commits: pending (this week's commit); pre-registration pushed before the full run (a 1-epoch timing run was seen first, noted in the register)
runtime: 69 s (8 epochs, CPU)
outcome: H39 not supported. Bottleneck is used (decoder error 5.5 vs 11.1 mg/dL with it zeroed) but PC1 is current glucose (Spearman 0.99) and holds 94% of variance; PC2's arc is likely a horseshoe artifact of a curved 1D structure. LOO error predicting patient glucose SD: bottleneck 10.0, mean glucose 7.4 mg/dL.
notes: a separate cluster at PC2 about -1.35 comes mostly from patient 10's windows; not investigated (possible data artifact). Global scaling used so patient differences are kept. One seed.

## 2026-10-06 | 5_10 | HUPA-UCM plain CNN-LSTM for all patients, and teacher-forcing bottleneck (H40)
type: experiment
status: done
files: work/5_10/scripts/train_hupa_all_v0.py, work/5_10/scripts/hupa_autoencoder.py, data/processed/checkpoints/hupa_ucm_cnn_lstm_v0/, work/5_10/results/verdict_h40.txt, work/5_10/results/hupa_autoencoder_patients.csv, work/5_10/figures/hupa_autoencoder_pca.png, work/HYPOTHESES.md, work/5_10/README.md
commits: pending (this week's commit); pre-registration pushed before the run
runtime: training 156 s (24 new patients; 27 reused); encoder 10 s
outcome: HUPA plain CNN-LSTM test RMSE 19.4 to 82.3 mg/dL (patient 6 outlier, 8 days of data). H40 not supported: bottleneck PC1 = mean glucose (0.99), 95% of variance; LOO error predicting patient RMSE: bottleneck 12.8, mean glucose 12.5, glucose SD 9.8 mg/dL.
notes: these HUPA checkpoints also enable the third dataset for the situation-band study. Training windows capped at 3,000 per patient for the encoder. HUPA filler left in.

## 2026-10-06 | 5_10 | Compressibility vs CNN-LSTM error (H41, H42)
type: experiment
status: done
files: work/5_10/scripts/compressibility.py, work/5_10/results/compressibility_patients.csv, work/5_10/results/verdict_h41_h42.txt, work/5_10/figures/compressibility.png, work/HYPOTHESES.md, work/5_10/README.md
commits: pending (this week's commit); pre-registration pushed before the run
runtime: 72 s
outcome: both not supported. LOO error predicting patient test RMSE: glucose SD alone 5.61; SD + sample entropy 5.09 (+9%, rule 15%); SD + bits per step 6.03 (-7%). Glucose SD vs RMSE Spearman +0.64.
notes: measures from the first 80% of each patient, target the last 20%. Bits per step overlaps with SD (Spearman +0.33). Sample entropy is scale-free and partly independent of SD (Spearman -0.57); weak hint only. 25 patients, one seed.

## 2026-10-06 | 5_10 | Decision: close the patient-predictability line
type: docs
status: done
files: work/5_10/README.md
commits: pending (this week's commit)
outcome: user decided to stop the autoencoder and compressibility line (H39-H42), which produced no insight beyond glucose variability. Next steps to be discussed with Volkan (meeting Thursday Oct 8).
notes: situation-aware bands (H27-H38) remain the main result.

## 2026-10-07 | 5_10 | Quick look: t-SNE of the teacher-forcing bottleneck
type: analysis
status: done
files: work/5_10/scripts/tsne_bottleneck.py, work/5_10/figures/tsne_bottleneck.png, work/5_10/README.md
commits: pending (this week's commit)
runtime: 158 s (both encoders retrained with the same settings and seed)
outcome: same picture as PCA. The map is ordered by current glucose (smooth gradient on both datasets). Forecast-error colours are mixed with no error clusters; high-error windows sit at the high-glucose end. AZT1D patients are fully mixed; HUPA-UCM shows strands from consecutive overlapping windows and some low-glucose patients (18, 22) apart.
notes: exploratory, after the predictability line was closed. Patient labels at average t-SNE positions are not meaningful (t-SNE distorts between-group distances).

## 2026-10-07 | 5_10 | Quick look: t-SNE of each patient's average bottleneck
type: analysis
status: done
files: work/5_10/scripts/tsne_patient_average.py, work/5_10/figures/tsne_patient_average.png, work/5_10/README.md
commits: pending (this week's commit)
runtime: 77 s
outcome: patients form one chain, same order in both seeds. Chain position vs mean glucose: AZT1D 0.99, HUPA-UCM 0.97 (absolute Spearman). Vs forecast error about 0.46, no more than mean glucose itself (0.49 AZT1D, 0.39 HUPA-UCM).
notes: exploratory; 25 points per dataset, perplexity 5.

## 2026-10-07 | 5_10 | Quick look: bottleneck t-SNE coloured by situation
type: analysis
status: done
files: work/5_10/scripts/tsne_situation.py, work/5_10/scripts/tsne_bottleneck.py, work/5_10/figures/tsne_situation.png, work/5_10/README.md
commits: pending (this week's commit)
runtime: 141 s
outcome: on both datasets the map splits into regions by current level, and within each region windows are ordered by 30-minute trend (fast rises and fast falls form separate bands). The encoder's structure matches the situation grid.
notes: visual, not quantified. tsne_bottleneck.embed gained an optional return_prev flag (default unchanged).

## 2026-10-07 | 5_10 | Learned-situation bands from the encoder summary (H43)
type: experiment
status: done
files: work/5_10/scripts/learned_situation_bands.py, work/5_10/results/h43_learned_situation.csv, work/5_10/results/verdict_h43.txt, work/5_10/figures/h43_learned_situation.png, work/HYPOTHESES.md, work/5_10/README.md
commits: pending (this week's commit); pre-registration pushed before the run
runtime: 97 s
outcome: H43 not supported. Interval score: AZT1D grid 114, k-means 114, kNN 113; OhioT1DM 131, 132, 132. Zone D caught: AZT1D 22%, 22%, 25%; OhioT1DM 41%, 43%, 36%. Per-situation error: grid 3.9 on both; learned 4.9-6.3.
notes: per-situation error is measured on the hand-drawn situations, which favours the grid; but the learned versions also fail to gain on the two neutral measures. Encoder trained on first 64% per patient; windows checked aligned with the forecaster. One seed.

## 2026-10-07 | 5_10 | Quick look: HUPA t-SNE coloured by each recorded variable
type: analysis
status: done
files: work/5_10/scripts/tsne_hupa_variables.py, work/5_10/scripts/tsne_bottleneck.py, work/5_10/figures/tsne_hupa_variables.png, work/5_10/README.md
commits: pending (this week's commit)
runtime: 41 s
outcome: glucose shows the smooth gradient seen before. Basal insulin shows clear regions (near-zero strand at top, a mid-basal mass, a high-basal strand at bottom); basal is largely set per patient (pump settings, or none recorded), so it acts as a second organizing factor tied to therapy setup. Bolus and carbs (mostly zero) show no clear structure. Heart rate, steps, calories and time of day (not encoder inputs) show no clear structure.
notes: visual only. Raw rows checked aligned with encoded windows (glucose matches). tsne_bottleneck.embed gained an optional return_picks flag (defaults unchanged).
