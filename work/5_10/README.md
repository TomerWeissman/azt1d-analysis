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
- **H17, pending**: personalization direction. To be decided with Prof. Watson.

## Goals for the week
1. Run the learning-curve and unseen-patient test (H18, H20).
2. Decide the direction with Prof. Watson (H17, H12).
3. Prepare slides for the Thursday Oct 8 meeting with Volkan.

## Findings
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
| README.md | This page | current |

## Remember
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
