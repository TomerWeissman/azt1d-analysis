# Week 14_9: Sep 14 to Sep 20

## Completed
- Sep 14: GLIMMER v1 weighted loss (`notebooks/03`) (9738f8e)
- Sep 14: GLIMMER v2 clinical metrics (`notebooks/04`) (84423b6)
- Sep 14: GLIMMER v3, per-patient genetic algorithm, CNN-LSTM and Transformer (`notebooks/05`) (69ca208, 05eb0b9)
- Sep 14: MetaboNet loader checked row by row against AZT1D (810f911)
- Sep 14: OhioT1DM replication through MetaboNet (`notebooks/06`) (ad769f3, 3475f9f)
- Sep 14: README write-up (3d15a3a)
- Sep 15: presentation summary notebook and ranked theories for the gap to the paper (`notebooks/07`) (4648073 to 2f5b9da)
- Sep 15: GA seed bug found and fixed. Notebooks 02, 05, 06, and 07 re-run. **Uncommitted.**
- Sep 17: recursive multi-step forecasting, three iterations (`notebooks/08`) (d5ff3ef, 897758b, 9a3f435)

## Outcomes
- v1 helps the hypo and hyper regions but makes overall RMSE and MAE worse than v0.
- v2: event recall rises from 0.507 to 0.722, and Clarke zone D falls from 2.37% to 1.33%.
- Before the seed fix, personalized CNN-LSTM RMSE was 37.44 vs fixed weights 41.18. After the fix it is 41.12 vs 41.18, so no real gain.
- OhioT1DM shows the same pattern as AZT1D.
- Recursive forecasting: predicting every input flattens the curve. A time-of-day feature did not help. Real meal inputs improved it.

## Open
- Commit the seed-fix changes (notebooks 02, 05, 06, 07).
- Update the README and the 07_summary narrative. Both still show the pre-fix result.
