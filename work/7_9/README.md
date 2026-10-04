# Week 7_9: Sep 7 to Sep 13, 2026

## Hypotheses this week
- **H1, confirmed**: a plain per-patient CNN-LSTM forecasts 60-minute glucose on AZT1D. This is the GLIMMER baseline.
- **H2, dropped**: ML-Glucose has an unexplored gap. Not checked; switched to AZT1D on Sep 13.

## Findings
- AZT1D is usable as the dataset for the replication.
- The baseline is built. It has no region weighting yet.

## Outcomes
- Baseline numbers are in [02_glimmer_v0_baseline](notebooks/02_glimmer_v0_baseline.ipynb). Not copied here.

## Work done
- Sep 13: exploratory analysis, [01](notebooks/01_data_exploration.ipynb) (commit 49680f3).
- Sep 13: plain CNN-LSTM baseline, [02](notebooks/02_glimmer_v0_baseline.ipynb) (commit 49680f3). The seed-fix edits made later in week 14_9 are in this file too.

## Files
| File | What it is | Status |
|---|---|---|
| [notebooks/01_data_exploration.ipynb](notebooks/01_data_exploration.ipynb) | AZT1D exploratory analysis: demographics, glycemic metrics, daily profiles | current |
| [notebooks/02_glimmer_v0_baseline.ipynb](notebooks/02_glimmer_v0_baseline.ipynb) | Plain CNN-LSTM baseline, one model per patient | current; includes the seed-fix edits |

## Remember
- Sep 13 decision: use AZT1D first. OhioT1DM access was still pending.
- Sep 10 idea (ML-Glucose gap) was never verified.

## Open
- None recorded.
