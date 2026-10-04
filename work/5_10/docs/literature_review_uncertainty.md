# Literature review: uncertainty in glucose forecasting, and the gaps

Written Oct 4, 2026 for week 5_10. Most claims come from abstracts and search summaries. Several full texts did not open (ScienceDirect returned 403, and some arXiv PDFs did not parse), so treat each claim as "as reported in the abstract or summary," not as checked against the full paper.

## The short version

Uncertainty for glucose forecasting is an active field, but most of it asks "is the band calibrated?" or "is accuracy good?" Few papers ask the questions our 2026 work asks: does the band help an alarm, does the model get more confident about a new person, and do the two uncertainty types mean what their labels say.

## What exists

- **Uncertainty from one model.** Transformers with uncertainty output ([Gluformer, 2022](https://arxiv.org/pdf/2209.04526)), a Temporal Fusion Transformer with quantile outputs ([PMC12349322](https://pmc.ncbi.nlm.nih.gov/articles/PMC12349322/)), and evidential output layers ([Tan and McBeth, 2026](https://arxiv.org/abs/2603.04955)). The last compares Monte Carlo dropout with evidential heads on HUPA-UCM and reports the evidential heads as best calibrated.
- **Personalization.** Fine-tuning a population model, meta-learning, federated learning, and patient ID as a covariate. These report accuracy, and in some cases calibration, at one data size ([PMC12349322](https://pmc.ncbi.nlm.nih.gov/articles/PMC12349322/); [EVIDENT, 2026](https://arxiv.org/abs/2606.05373), abstract only).
- **Alarms.** Decades of hypoglycemia alarm work, mostly point-prediction thresholds. One example reports 86% sensitivity at 0.42 false alarms per day, from 69 events in 17 subjects ([PMC3692235](https://pmc.ncbi.nlm.nih.gov/articles/PMC3692235/)). I checked that paper: it does not use bands at all.
- **Evaluation beyond accuracy.** A 2026 task-aware framework evaluates hypoglycemia warning by event recall and false alarms per patient-day, and argues that accuracy does not predict task usefulness ([arXiv:2605.00645](https://arxiv.org/abs/2605.00645)). It does not evaluate bands.
- **Critique of uncertainty decomposition.** A 2025 position paper argues that second-order methods (evidential and similar) overestimate aleatoric uncertainty and underestimate epistemic uncertainty, and that each method captures only part of the epistemic variance ([arXiv:2505.23506](https://arxiv.org/abs/2505.23506)). This is a general ML argument, not a glucose study.
- **Conformal prediction.** Used for imputation with CGM data ([arXiv:2403.18069](https://arxiv.org/pdf/2403.18069)). A federated personalization summary mentions client-side conformalized quantile regression (FedPhysio-Adapter). I did not open that paper.

## Gaps, in plain language

| # | Gap | Plain version | Found in literature? | Our evidence |
|---|---|---|---|---|
| G1 | Bands as alarms, per patient | Alarms are built from the single guess, not from the range of possible values. | Not found. The alarm paper uses point thresholds; the task-aware paper uses events but no bands. | Ordinary bands falsely trigger on 68 to 100% of safe readings (21_9). |
| G2 | Confidence as a person's data grows | Nobody has shown whether the model gets more confident about a new person with the first few days. | Not found. Personalized papers report one data size. | Epistemic did not shrink with data; evidential spiked at 1 day in all 5 patients (28_9). |
| G3 | Two methods, two answers | The labels "epistemic" and "aleatoric" depend on the method, and the critique says so in general ML. | The critique exists (2025). Method comparisons on the same glucose data for the decomposition: not found. | Ensemble about 1% epistemic, evidential about 57%, same data (28_9). |
| G4 | The noise ceiling | Nobody has said how low the error can go for glucose. | Not found. The concept is standard: for squared-error regression the Bayes error equals the noise variance ([Wikipedia](https://en.wikipedia.org/wiki/Bayes_error_rate)). | Not measured yet (H23). |
| G5 | Does width track error? | Do bands widen when the forecast is about to be wrong? | Not found in the papers I read. Not checked exhaustively. | Volatility bands partly yes (EWMA 1.67x, GARCH 1.39x); conformal flat (28_9). |
| G6 | Conformal for 60-minute forecasts | A standard tool with guarantees, but I didn't find it used for 60-minute CGM forecasts with per-patient calibration. | Imputation and one personalization summary only. | Not tested. |
| G7 | Sensor error in the ceiling | Part of the target's error is the sensor's own error, and that error is partly predictable from its own past. | Sensor noise is characterized ([ScienceDirect, 2020](https://www.sciencedirect.com/science/article/abs/pii/S1746809420300902), not opened). No paper I found adds it to a model ceiling. | Not used yet. |

## Why our results fail where they do (simple versions)

- **Bands cry wolf.** A band can only add room, so it can only add alarms. The point forecast already sets a floor on false alarms. The literature's alarm methods avoid this by using thresholds directly, which is the same floor.
- **All methods look the same at a matched false-alarm rate.** The limit seems to be the point forecast, not the band shape.
- **Personalization shows no gain.** Either the test is too small (5 patients, 1 seed), the fine-tune step is too gentle (learning rate 1e-5), or there's little per-person signal to learn. The literature doesn't separate these.
- **Epistemic disagrees by method.** Matches the 2025 critique. Our result is a concrete example on the same data.
- **Bands under-track error.** Error grows about 36 times from easiest to hardest, but the band grows under twice. Nothing in the literature I read checks this.

## Candidate next steps, mapped to gaps

1. **G1, alarms from bands, per patient.** Uses existing bands and alarm code. No training. Report false alarms per patient-day, as the 2026 task-aware paper does. Cheapest, and it closes the most visible gap.
2. **G3, method comparison.** Run ensemble, evidential, and quantile heads on the same windows, then test each decomposition with the same learning-curve check (H18). Medium cost.
3. **G2, budget curve at scale.** The H13 test with more patients and seeds, and a larger fine-tune learning rate as a sensitivity check. Medium to high cost.
4. **G4 and G7, noise ceiling.** H23 as planned, with a sensor-error source added for the lower end. Medium to high cost, and needs the sensor source.
5. **G6, conformal forecast intervals.** Per-patient conformal bands, compared with GARCH on coverage and alarms. Cheap, and it gives a guarantee the other methods don't.

## Open questions for this review

- The FedPhysio-Adapter claim and the EVIDENT details come from summaries, not full texts. Check them before citing.
- I didn't search for medical-device or regulatory work (for example ISO 15197 alarm standards), which may define acceptable false-alarm rates. That's the next search.

## Addendum (Oct 4): Mondrian conformal prediction, and what it means for novelty

**The method is not new.**
- Our "situation-only" band is **Mondrian conformal prediction**: partition the data into groups with a fixed rule, then run split conformal inside each group. It guarantees coverage within each group, not just on average ([MAPIE docs on Mondrian](https://mapie.readthedocs.io/en/v0.9.0/theoretical_description_mondrian.html); [Vovk, conditional validity of inductive conformal predictors](https://www.researchgate.net/publication/230996365_Conditional_validity_of_inductive_conformal_predictors); [Conformal Prediction With Conditional Guarantees, arXiv:2305.12616](https://arxiv.org/pdf/2305.12616)).
- Our "GARCH x situation" band is **normalized conformal prediction inside Mondrian groups**: errors divided by a noise estimate (here, GARCH's error size) before calibrating. Normalized conformal for heteroscedastic data is standard ([Conditional validity of heteroskedastic conformal regression](https://www.researchgate.net/publication/391703798_Conditional_validity_of_heteroskedastic_conformal_regression); [CQR, Romano et al. 2019](https://arxiv.org/pdf/1905.03222)). Using GARCH as the noise estimate is a choice, not a new method.
- Known weakness, which we saw: per-group quantiles become unstable when a group has few calibration samples (two of our 25 situations fell back to the pooled width).

**Not found: application to glucose forecasting.**
- No glucose or CGM paper found that uses Mondrian or group-conditional conformal calibration, or that measures band coverage by glucose level and trend. Glucose papers stratify *point accuracy* by glycemic range (for example GlyRAG's range-stratified error grid, [arXiv:2601.05353](https://arxiv.org/html/2601.05353)), not band coverage.
- Checked and ruled out: SSM-CGM ([arXiv:2510.04386](https://arxiv.org/html/2510.04386), abstract only; no group-conditional calibration mentioned) and CASCADE ([arXiv:2605.20468](https://arxiv.org/pdf/2605.20468), Parkinson's dosing, not glucose).
- Limits: abstracts and search summaries only; not exhaustive.

**What this means for a paper.** The contribution is not the method. It is:
1. The finding that standard glucose uncertainty bands (GARCH, plain conformal) are miscalibrated in specific, clinically meaningful situations (for example 64% coverage during fast rises from 70 to 120).
2. Showing that Mondrian conformal with clinically meaningful groups (level by trend) fixes it, on two datasets, with narrower bands than GARCH.
3. The GARCH-normalized Mondrian variant as the best-calibrated option.

Required in the paper: cite Mondrian and normalized conformal as the methods, describe them as such, and compare against plain conformal and conformalized quantile regression.
