# Week 21_9: Sep 21 to Sep 27, 2026

## Hypotheses this week
- **H9, invalidated**: ordinary 80% and 95% bands can raise danger alarms without many false alarms. They falsely trigger on 68 to 100% of safe readings.
- **H10, invalidated**: one of GARCH, analog ensemble, or conformal gives clearly better alarms. At matched false-trigger rates, all three catch 74 to 76.5% of danger.
- **H19, not yet tested**: sizing the two band edges separately could cut false alarms.

## Findings
- A band only adds room on either side of the forecast, so it can only add alarms. Every method has a floor set by its center line.
- Most false alarms come from the top edge (above 180): the model forecasts high.
- Per-patient false-alarm rates at one band size range from 0% to 87%.

## Outcomes
- Raw forecast alone: 32% false triggers, catches 72.2% of danger.
- At matched 35% false triggers: conformal, GARCH, and analog catch 74 to 76.5% of danger.
- Subject 1, Clarke zone trends: growing the GARCH band moves predictions from zone A into B and C.

## Work done
- Sep 22: capstone research paper draft (3c02cea).
- Sep 26: stage 2 uncertainty bands on the fixed-weight model, [01](notebooks/01_uncertainty_bands.ipynb) (47546d8, 0e4340a, a7a636e). Added the `arch` library to requirements.
- Sep 26: one-patient view with alarm plots and a Clarke grid, [02](notebooks/02_one_patient_view.ipynb) (495c937, 29bb669).

## Files
| File | What it is | Status |
|---|---|---|
| [notebooks/01_uncertainty_bands.ipynb](notebooks/01_uncertainty_bands.ipynb) | Stage 2: GARCH, analog, and conformal bands on the fixed-weight model, pooled across patients | current |
| [notebooks/02_one_patient_view.ipynb](notebooks/02_one_patient_view.ipynb) | Subject 1: forecasts, bands, alarm plots, Clarke grids | current |
| [scripts/build_uncertainty_notebook.py](scripts/build_uncertainty_notebook.py) | Builds notebook 01 | current |
| [scripts/build_one_patient_notebook.py](scripts/build_one_patient_notebook.py) | Builds notebook 02 | current |
| [docs/capstone_paper_draft.md](docs/capstone_paper_draft.md) | Capstone research paper draft, markdown | **stale**: Table 5.2 still has pre-fix personalized numbers |
| [docs/capstone_paper_draft.html](docs/capstone_paper_draft.html) | Same draft, HTML | **stale**: same as the markdown |

## Remember
- The Sep 26 "edges sized separately" idea (H19) is still untested.
- The paper draft is public, since the repo is public.

## Open
- Fix the stale numbers in the paper draft and the Google Doc.
- Test H19.
