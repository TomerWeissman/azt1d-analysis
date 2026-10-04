# Week 21_9: Sep 21 to Sep 27

## Completed
- Sep 22: capstone research paper draft (`capstone_paper_draft.md` and `.html`) (3c02cea)
- Sep 26: stage 2 uncertainty bands on the fixed-weight model (`notebooks_2/01_uncertainty_bands.ipynb`) (47546d8, 0e4340a, a7a636e)
- Sep 26: one-patient view with alarm plots and Clarke grid (`notebooks_2/02_one_patient_view.ipynb`) (495c937, 29bb669)
- Sep 26: added the `arch` library for GARCH (`requirements.txt`)

## Outcomes
- Ordinary 80% and 95% bands falsely trigger on 68-100% of safe readings.
- The raw forecast alone has 32% false triggers and catches 72.2% of danger.
- At matched 35% false triggers, conformal, GARCH, and analog catch 74-76.5% of danger. No method clearly wins.
- Most false triggers come from the top edge (above 180).
- At one band size, per-patient false-trigger rates range from 0% to 87%.

## Open
- Paper draft Table 5.2 still has the pre-fix personalized numbers.
- Suggested next step, not started: size the two band edges separately.
