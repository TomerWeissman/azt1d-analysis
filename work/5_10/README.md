# Week 5_10: Oct 5 to Oct 11, 2026

Started Oct 4 at the user's request, a day early. Goals set at week start.

## Hypotheses this week
- **H18, testing**: the ensemble's disagreement variance is epistemic. It shrinks as training data grows, while the predicted noise variance does not. Rule set before running: epistemic is supported if it falls by at least 20% from the smallest to the largest training set, and aleatoric moves by less than 10%.
- **H20, testing**: the ensemble's epistemic variance is higher on windows from unseen patients than on familiar ones. Aleatoric is about the same.
- **H21, not yet tested (optional)**: free-running error grows faster than teacher-forced error over the horizon. A diagnostic for compounding, not a measure of epistemic uncertainty.
- **H23, inconclusive**: noise is at most 75% of the error. The look-alike bound came out at 106%, so it says nothing.
- **H24, not supported**: band alarms do not beat a tuned threshold.
- **H25, not supported**: conformal 80% bands are roughly right on median (79%) but not for every patient.
- **H17, pending**: personalization direction. To be decided with Prof. Watson.

## Goals for the week
1. Run the learning-curve and unseen-patient test (H18, H20).
2. Decide the direction with Prof. Watson (H17, H12).
3. Prepare slides for the Thursday Oct 8 meeting with Volkan.

## Findings
- Look-alike windows still differ too much to bound the noise. With many inputs, true look-alikes are rare.
- A tuned point threshold beats the band alarm. The band alarm fires about a third more often (10.2 vs 7.7 onsets per day).
- Thresholds tuned to 90% sensitivity on validation reach only 67% to 73% on test. The validation and test periods differ.
- Conformal is right on median (79%), but coverage ranges from 65% to 87% across patients. The promise holds on average, not per patient.

## Outcomes
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
