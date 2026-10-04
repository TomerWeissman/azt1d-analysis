# Week 5_10: Oct 5 to Oct 11, 2026

Started Oct 4 at the user's request, a day early. Goals set at week start.

## Hypotheses this week
- **H18, testing**: the ensemble's disagreement variance is epistemic. It shrinks as training data grows, while the predicted noise variance does not. Rule set before running: epistemic is supported if it falls by at least 20% from the smallest to the largest training set, and aleatoric moves by less than 10%.
- **H20, testing**: the ensemble's epistemic variance is higher on windows from unseen patients than on familiar ones. Aleatoric is about the same.
- **H21, not yet tested (optional)**: free-running error grows faster than teacher-forced error over the horizon. A diagnostic for compounding, not a measure of epistemic uncertainty.
- **H23, inconclusive**: noise is at most 75% of the error. The look-alike bound came out at 106%, so it says nothing.
- **H24, not supported**: band alarms do not beat a tuned threshold.
- **H25, not supported**: conformal 80% bands are roughly right on median (79%) but not for every patient.
- **H26, not supported**: the noise share of error does not differ twofold between glucose situations (the look-alike bounds are too loose to separate them).
- **H27, not supported (narrowly)**: situation-specific widths put 18 of 25 situations at 75% to 85% coverage. The rule needed 20. Current GARCH bands: 13 of 25.
- **H28, supported**: on fresh data (OhioT1DM), situation-specific bands cut the distance from the 80% target by 41% vs GARCH (6.7 to 3.9 points), closer for 9 of 12 patients, and 12% narrower (87 vs 99 mg/dL).
- **H29, not supported**: the model does not get worse over time on average (median test/validation error 1.03). Per-patient shifts (0.81 to 1.45) track changes in glucose variability (correlation 0.77).
- **H30, not supported**: unlimited data of this kind would cut the plain model's error by only about 2% (asymptote 98% of today's error, 90% interval 93% to 99%). The model is not data-limited.
- **H31, supported (narrowly)**: GARCH x situation bands beat situation-only by 21% on OhioT1DM (3.1 vs 4.0 points from the 80% target), better for 7 of 12 patients (the minimum). AZT1D agrees: 20% better, 14 of 25 patients.
- **H32, not supported (narrowly)**: Mondrian beats plain conformal by 41-44% but CQR by only 18% on OhioT1DM (rule: 20%). On AZT1D it beats CQR by 52%. The hybrid beats CQR on both (3.1 vs 4.9 and 7.9).
- **H33, supported**: real situations 4.0 points vs 6.6 to 6.9 for 20 shuffled groupings. The situations matter, not just the number of groups.
- **H34, supported**: the gain over GARCH holds for GLIMMER-weighted and Transformer models on both datasets (23% to 47%).
- **H35, supported**: Mondrian-vs-GARCH gain 90% interval 0.3 to 3.6 points. Hybrid-vs-Mondrian interval -0.5 to 2.1 includes zero: the hybrid's extra gain is not robust.
- **H36, supported (by the rule; not robust on OhioT1DM)**: situation bands 19% better calibrated than tuned full-window CQR on OhioT1DM (4.0 vs 4.9 points), and narrower (87 vs 94). Patient-bootstrap interval -0.7 to 2.4 crosses zero. On AZT1D clearly better (3.8 vs 8.9; interval 2.4 to 6.1).
- **H37, supported by the rule, but the effect is small**: at the same width, situation bands catch 0.1 to 1.8 more points than GARCH at every level, on both datasets. All methods' curves nearly overlap. On interval score, CQR is best on both datasets, situation bands close second.
- **H17, pending**: personalization direction. To be decided with Prof. Watson.

## Goals for the week
1. Run the learning-curve and unseen-patient test (H18, H20).
2. Decide the direction with Prof. Watson (H17, H12).
3. Prepare slides for the Thursday Oct 8 meeting with Volkan.

## Findings
- Level-free check (H37): overall, all methods trade width for coverage almost identically. The situation bands' advantage is not overall efficiency; it is being honest in each situation (the per-situation results). CQR edges ahead on interval score. The paper should claim per-situation calibration, not overall efficiency.
- Verification (H32-H35): situation grouping is real (beats shuffled groups), works across models, and survives patient resampling. The strongest rival is CQR, which comes close on OhioT1DM. The hybrid's edge over plain Mondrian is not statistically robust.
- More data won't help this model (H30). The curve is flat by about 35% of today's data. The remaining error is either noise or a limit of the model and its inputs; this test can't tell which.
- Situation-specific bands hold up on a dataset they never touched (H28). This is the week's main confirmed result.
- Earlier 'drift' was a misreading: errors move with each patient's glucose variability in the later period, not with time itself (H29).
- Situation-specific widths narrow the coverage spread from 31 points to 16 points. More situations land near 80%, but the rule missed by two.
- The remaining misses are high-glucose slow falls and flat readings (67% to 69%). Their validation errors were smaller than their test errors, which is drift.
- Error depends strongly on the situation. Steady readings between 70 and 180 have RMSE about 26 mg/dL. Rapid rises above 250 reach 61 mg/dL.
- Band coverage varies by situation: from 64% (70 to 120, fast rise) to 95% (under 70, fast fall). The 80% bands are roughly right on average, but not in every situation.
- Look-alike noise bounds are 78% to 118% of the model's error in every scored cell. They are too loose to say which situations are mostly noise.
- Literature check: region-based error grids and state-switching models exist. I found no paper that splits each region's error into noise and reducible parts. Two papers did not open.
- Look-alike windows still differ too much to bound the noise. With many inputs, true look-alikes are rare.
- A tuned point threshold beats the band alarm. The band alarm fires about a third more often (10.2 vs 7.7 onsets per day).
- Thresholds tuned to 90% sensitivity on validation reach only 67% to 73% on test. The validation and test periods differ.
- Conformal is right on median (79%), but coverage ranges from 65% to 87% across patients. The promise holds on average, not per patient.

## Outcomes
- H31: OhioT1DM coverage error GARCH 6.7, situation 4.0, hybrid 3.1 points; widths 99, 87, 89 mg/dL. AZT1D: 6.0, 3.8, 3.1 points; widths 83, 71, 76 mg/dL. Patient 588 is a case where the hybrid is worse (77% vs 82% overall coverage).
- H30: pooled test RMSE 34.2 (10%), 33.0 (20%), 32.6 (35%), 32.2 (60%), 32.8 (100%) mg/dL. Asymptote 32.5 mg/dL. Seed-to-seed spread at full data (about 1 mg/dL) is larger than the gain from more data. Runtime 33 min.
- H28 (OhioT1DM): coverage error GARCH 6.7, situation-specific 3.9 points (41% cut). 9 of 12 patients closer. Overall coverage GARCH 83.2%, new 79.4%. Mean width 99.3 vs 87.0 mg/dL. Worse for patients 570, 540, 552 and situations 250+ flat and under 70 fast fall.
- H29 (AZT1D): median ratio 1.03; 8 of 25 at least 10% worse; 11 better; range 0.81 to 1.45; median RMSE validation 29.4, test halves 30.8 and 30.0.
- H27: situation-specific 18 of 25 inside 75% to 85%, coverage range 67% to 83%. GARCH on the same windows: 13 of 25, range 64% to 95%. Two situations fell back to the pooled width.
- H26: lowest noise bound 78% (70 to 120, flat, n=9,126). Highest 118% (180 to 250, flat, n=2,839). Rule: lowest at most half of highest. Not met.
- H23: upper bound on noise 1093 mg2/dL2, versus model error 1030 mg2/dL2 (106%). Rule: at most 75%.
- H24: band 10.2 onsets/day, point 7.7 onsets/day. Sensitivity on test: band 0.73, point 0.67.
- H25: 16 of 25 patients inside 75% to 85%. Median 79.1%. Range 64.6% to 86.8%.

## Work done
- Oct 4: week start. Carried the 28_9 fine-tuning results forward (copies; originals stay in 28_9). Corrected the overview sentence about teacher forcing. Reframed H18 as a learning-curve test.
- Oct 4: literature review on uncertainty in glucose forecasting and seven gaps (see the docs file above).

## Files
| File | What it is | Status |
|---|---|---|
| [results/quick_metrics.csv](results/quick_metrics.csv) | Carried from 28_9. Per-patient metrics from the fine-tuning experiment, used as the baseline | carried; original in 28_9 |
| [figures/quick_results.png](figures/quick_results.png) | Carried from 28_9. Three panels: RMSE, uncertainty, and calibration vs budget | carried; original in 28_9 |
| [docs/literature_review_uncertainty.md](docs/literature_review_uncertainty.md) | Literature review on uncertainty in glucose forecasting, with seven gaps | current |
| [scripts/learning_curve_unseen_patients.py](scripts/learning_curve_unseen_patients.py) | H18 and H20 test: learning curve and unseen patients. Written, not yet run (needs approval for run time) | current, not run |
| [scripts/cheap_tests_h23_h25.py](scripts/cheap_tests_h23_h25.py) | Runs H23, H24, H25 on the saved plain-model checkpoints. No training | current |
| [results/verdicts_h23_h25.txt](results/verdicts_h23_h25.txt) | Verdicts and numbers for H23, H24, H25 | current |
| [figures/h23_lookalike.png](figures/h23_lookalike.png) | H23: outcome gaps between look-alike windows, and the noise bound against model error | current |
| [figures/h24_alarms.png](figures/h24_alarms.png) | H24: alarms per day per patient, band vs point; pooled sensitivity | current |
| [figures/h25_conformal_coverage.png](figures/h25_conformal_coverage.png) | H25: per-patient coverage of conformal 80% bands | current |
| [scripts/predictability_map.py](scripts/predictability_map.py) | H26: predictability map of 25 glucose situations (level by 30-minute trend). No training | current |
| [results/predictability_map.csv](results/predictability_map.csv) | Per-situation RMSE, band coverage, width, and noise bounds | current |
| [figures/predictability_map.png](figures/predictability_map.png) | H26 heatmaps: model error and noise bound by situation | current |
| [scripts/situation_bands.py](scripts/situation_bands.py) | H27: situation-specific band widths vs GARCH. No training | current |
| [results/situation_bands.csv](results/situation_bands.csv) | Per-situation widths, coverage, and width ratio | current |
| [results/verdict_h27.txt](results/verdict_h27.txt) | H27 verdict and counts | current |
| [figures/situation_bands.png](figures/situation_bands.png) | H27: coverage by situation, both methods | current |
| [scripts/ohio_situation_bands.py](scripts/ohio_situation_bands.py) | H28: situation bands vs GARCH on OhioT1DM. No training | current |
| [figures/h28_ohio_situation_bands.png](figures/h28_ohio_situation_bands.png) | H28: distance from 80% target, by situation and by patient | current |
| [scripts/drift.py](scripts/drift.py) | H29: error by period, and its link to glucose variability | current |
| [figures/h29_drift.png](figures/h29_drift.png) | H29: validation vs test error per patient; error change vs variability change | current |
| [scripts/learning_curve_ceiling.py](scripts/learning_curve_ceiling.py) | H30: retrains the plain model on 10% to 100% of its data, fits where error levels off | current |
| [results/learning_curve_runs.csv](results/learning_curve_runs.csv) | Every training run: patient, fraction, seed, test error | current |
| [figures/learning_curve_ceiling.png](figures/learning_curve_ceiling.png) | H30: forecast error vs amount of training data | current |
| [scripts/example_patient_bands.py](scripts/example_patient_bands.py) | One-patient example: GARCH vs situation-specific bands over 24 hours | current |
| [figures/example_patient_588_bands.png](figures/example_patient_588_bands.png) | OhioT1DM patient 588, the 24 hours with the widest glucose range, both bands | current |
| [scripts/hybrid_bands.py](scripts/hybrid_bands.py) | H31: GARCH, situation-only, and GARCH x situation bands on OhioT1DM and AZT1D | current |
| [figures/h31_hybrid_summary.png](figures/h31_hybrid_summary.png) | H31: distance from 80% target and width, three methods, two datasets | current |
| [figures/example_patient_588_three_bands.png](figures/example_patient_588_three_bands.png) | Patient 588 over 24 hours, all three bands | current |
| [scripts/verification_checks.py](scripts/verification_checks.py) | H32-H35: alternatives, shuffled groups, other models, bootstrap | current |
| [results/verdicts_h32_h35.txt](results/verdicts_h32_h35.txt) | Verdicts and numbers for H32-H35 | current |
| [figures/verify_h32_alternatives_ohiot1dm.png](figures/verify_h32_alternatives_ohiot1dm.png) | H32: all methods compared, OhioT1DM | current |
| [figures/verify_h32_alternatives_azt1d.png](figures/verify_h32_alternatives_azt1d.png) | H32: all methods compared, AZT1D | current |
| [figures/verify_h33_shuffle.png](figures/verify_h33_shuffle.png) | H33: real situations vs 20 shuffled groupings | current |
| [figures/verify_h34_models.png](figures/verify_h34_models.png) | H34: the gain on four model/dataset combinations | current |
| [figures/verify_h35_bootstrap.png](figures/verify_h35_bootstrap.png) | H35: bootstrap distributions of the gains | current |
| [scripts/strong_cqr.py](scripts/strong_cqr.py) | H36: tuned full-window CQR vs situation bands | current |
| [results/verdict_h36.txt](results/verdict_h36.txt) | H36 verdict, numbers, bootstrap intervals, chosen settings | current |
| [figures/h36_strong_cqr.png](figures/h36_strong_cqr.png) | H36: GARCH, tuned CQR, situation bands on both datasets | current |
| [scripts/coverage_width_curves.py](scripts/coverage_width_curves.py) | H37: caught vs width for every method at levels 50% to 95% | current |
| [results/verdict_h37.txt](results/verdict_h37.txt) | H37 verdict, matched-width comparisons, interval scores | current |
| [figures/h37_coverage_width.png](figures/h37_coverage_width.png) | H37: points caught vs band width, all methods, both datasets | current |
| README.md | This page | current |

## Remember
- The situation-specific band is Mondrian conformal prediction, and the hybrid is normalized conformal inside Mondrian groups. Name them that way. The novelty is applying them to glucose and the miscalibration finding, not the method.
- The fine-tuning baseline used five target patients and one seed. Any new result should be read against that.
- Ensemble disagreement was about 1.6 mg/dL last week. With three members, variance estimates are noisy. Expect large error bars.
- Teacher forcing does not separate epistemic from aleatoric uncertainty by itself. It is kept only as the optional H21 diagnostic.
- Thursday Oct 8: meeting with Volkan. Slides due before it.
- Notion Gantt: Second Reader Meeting is Oct 16. Draft Workshop is Oct 26. Midterm Deliverables are Oct 30.

## Open
- Build and run the learning-curve and unseen-patient test (H18, H20). The full run needs approval for run time.
- Decide the personalization direction with Prof. Watson (H17).
- Prepare slides for Volkan.
- Start the paper structure, documenting the work so far.
- Build a workback plan to the Dec 11 full draft.
- Fix stale numbers in the paper draft and the Google Doc.
