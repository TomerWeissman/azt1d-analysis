# Replicating GLIMMER: Region-Aware Loss Weighting for Blood Glucose Forecasting in Type 1 Diabetes

Tomer Weissman

## Academic Abstract (draft, ~200 words)

Blood glucose forecasting models for type 1 diabetes are typically trained to minimize
average prediction error, treating a mistake during a dangerous hypo- or hyperglycemic
moment the same as a mistake in the safe, everyday range. GLIMMER (Khamesian et al.,
2026) proposes a region-aware weighted loss that penalizes errors more heavily in
dangerous glucose zones, with per-patient weights found by a genetic algorithm search,
and reports that this improves both overall accuracy and danger-zone accuracy together.
This project replicates GLIMMER from scratch on two real-world datasets, AZT1D and
OhioT1DM (the latter accessed via MetaboNet, a public consolidation of 14 T1D
datasets), across two model architectures (CNN-LSTM and CNN-Transformer). The
replication reproduces GLIMMER's clinical claim -- weighted loss trades some average
accuracy for meaningfully better detection of dangerous glucose events and roughly
halves the rate of the most dangerous prediction errors on the Clarke Error Grid -- but
does not reproduce its accuracy claim: in this implementation, weighted training makes
average error worse, not better, on all four architecture/dataset combinations tested.
A scoped-down genetic algorithm search budget is identified as the most likely,
directly evidenced cause of this gap. A follow-on experiment tests whether a trained
model can be run recursively to generate multi-hour glucose trajectories, finding that
it can simulate a believable response to a known meal but cannot anticipate one.

## 1. Introduction

*(expand: motivation -- why glucose forecasting, why region-aware loss, what gap in
the literature this addresses per the Journal's original framing)*

Type 1 diabetes management increasingly relies on continuous glucose monitors (CGM)
paired with automated insulin delivery systems. A model that can forecast glucose 60
minutes ahead gives a person, or their pump, time to act before a dangerous high or low
actually happens. Most forecasting models are trained and scored on plain average
error (RMSE/MAE), which does not distinguish a mistake during a dangerous moment from
a mistake during an ordinary one. GLIMMER (Khamesian et al., 2026) proposes training
with a loss that explicitly weights errors by glucose region (hypoglycemic, normal,
hyperglycemic), with weights searched per patient via a genetic algorithm.

This project has two goals: (1) replicate GLIMMER independently, from its written
description rather than its (unavailable) source code, on real data across two
datasets and two architectures, to test whether its central claim generalizes; and (2)
extend past GLIMMER's own evaluation by testing what happens when a trained model is
run recursively -- feeding its own predictions back in as input to generate an
extended trajectory -- since GLIMMER, like most forecasting work in this space, is
only evaluated on single-shot, fixed-horizon predictions.

## 2. Related Work

*(partial -- expand with the "Cites ML-Glucose" paper and additional entropy-methods
literature per Sprint Planning next steps)*

- Khamesian, S. et al. (2026). "Glycemic-Aware and Architecture-Agnostic Training
  Framework for Blood Glucose Forecasting in Type 1 Diabetes." *Human-Centric
  Intelligent Systems* (GLIMMER paper), Springer.
  DOI: 10.1186/s44398-026-00032-x
- AZT1D dataset. Mendeley Data.
  https://data.mendeley.com/datasets/gk9m674wcx/1
- OhioT1DM dataset (Marling & Bunescu). University of North Carolina at Charlotte /
  Ohio University. https://webpages.charlotte.edu/rbunescu/data/ohiot1dm/OhioT1DM-dataset.html
- MetaboNet dataset (14-dataset T1D consolidation, includes OhioT1DM).
  *Journal of Diabetes Science and Technology*.
  https://journals.sagepub.com/doi/abs/10.1177/19322968261441637
- ML-Glucose (arXiv:2507.14077) -- initial candidate dataset/framework investigated
  before pivoting to AZT1D; kept as related work since it surveys the same modeling
  space.
- Paper citing ML-Glucose (arXiv:2606.18640) -- *not yet reviewed in depth; flagged in
  Resources, needs a pass to confirm relevance and whether it changes the "no
  continued research on this dataset" assessment from the Sep 10 journal entry.*
- Pincus, S. -- Approximate Entropy and Sample Entropy tutorial (MDPI *Entropy*,
  21(6), 541) -- flagged in Sprint Planning as a candidate method for quantifying
  glucose signal complexity/predictability; not yet incorporated.
- Clarke, W. L. et al. -- Clarke Error Grid Analysis (clinical accuracy zones A-E for
  glucose predictions); implementation cross-checked against the public reference
  implementation at github.com/suetAndTie/ClarkeErrorGrid.

*Bibliography is partial by design at this checkpoint -- next pass should add: 1-2
more papers situating region-weighted/cost-sensitive loss functions in forecasting
generally (outside the glucose domain), and a closer read of the two ML-Glucose-adjacent
papers to decide whether they belong in Related Work or should be cited as a
considered-and-rejected alternative dataset path.*

## 3. Data

Two real-world datasets, both from people with type 1 diabetes using continuous
glucose monitors:

- **AZT1D**: 25 patients, all on a closed-loop automated insulin delivery system
  (Tandem t:slim X2 with Control-IQ), roughly one month of data each, 5-minute CGM
  sampling. Accessed directly from the Mendeley archive.
- **OhioT1DM**: 12 patients, sensor-augmented pump therapy (not closed-loop),
  5-minute CGM sampling. Accessed through MetaboNet rather than requesting the
  dataset directly, after the direct request (sent ~Sep 8) had not been fulfilled in
  time to be actionable -- MetaboNet's OhioT1DM subset was validated row-by-row
  against what a direct load would produce (exact match on glucose/bolus/carbs,
  basal matching after a unit-scale correction) before being trusted for this work.

Both datasets log, every 5 minutes: CGM glucose, background ("basal") insulin, meal
("bolus") insulin, and carbohydrate intake.

*(expand: demographics table, glycemic-control summary stats, the specific data
idiosyncrasies worth flagging for a reader -- right-skewed glucose distribution,
per-patient variability in control, sparse/spiky meal events)*

## 4. Methods

### 4.1 Forecasting task

Given the last 2 hours of a patient's data (glucose, basal, bolus, carbs, a glucose
moving average, and a hypo/normal/hyper region label -- 6 features), predict glucose
60 minutes ahead. One model is trained per patient (not one shared model), matching
GLIMMER's own per-patient personalization design. Two architectures are tested,
following GLIMMER's own descriptions: a small CNN + LSTM, and a CNN + single
Transformer encoder block.

### 4.2 Three training approaches compared

- **Standard**: ordinary MSE loss, no region weighting.
- **Fixed region-weighted**: GLIMMER's published population-average weights (hypo
  errors weighted 3.29x, hyper errors 2.38x, relative to normal-zone errors),
  applied identically to every patient.
- **Personalized region-weighted**: per-patient weights found via a genetic algorithm
  (Algorithm 1 in GLIMMER), scoped down from the paper's own search budget (population
  6, 6 generations vs. the paper's 20 and 25) for compute-time reasons -- the literal
  budget was estimated at 18+ hours of compute across all patients for one
  architecture/dataset combination alone.

### 4.3 Evaluation

Beyond plain RMSE/MAE: event-level precision/recall/F1 for detecting dysglycemic
moments, and Clarke Error Grid zone classification (the standard clinical-accuracy
tool, grading predictions by what a person would actually do differently because of
the error, not just how far off it was in mg/dL).

### 4.4 Recursive forecasting extension

Following a suggestion after presenting initial results, a separate experiment tests
whether a model can be run recursively -- predicting one 5-minute step ahead, then
feeding that prediction back in as input to predict the next step, and so on -- to
generate an extended trajectory, rather than the single 60-minute-ahead prediction
used everywhere else in this project. This required a differently-trained model
(predicting one step ahead across all four loggable quantities at once, since a
recursive rollout needs a value for every input at every step, not just glucose).
*(see Results 5.4 for what this found; methods section for this part should be
expanded with the specific model/rollout design once the paper's scope is finalized)*

## 5. Results

### 5.1 Baseline replication

The standard (unweighted) model's accuracy is a reasonable match to GLIMMER's own
reported baseline on both datasets, which is a basic sanity check that this
independent implementation is doing roughly the right thing:

| Dataset | This replication (standard) | GLIMMER's reported baseline |
|---|---|---|
| AZT1D | 31.54 +/- 6.10 mg/dL RMSE | 29.55 +/- 6.49 mg/dL RMSE |
| OhioT1DM | 36.79 +/- 4.37 mg/dL RMSE | 31.98 +/- 4.15 mg/dL RMSE |

### 5.2 Does region-weighted training improve accuracy? (No.)

Across all four combinations tested (2 architectures x 2 datasets), fixed
region-weighting makes plain RMSE *worse*, not better, and personalized weighting
recovers only part of that gap -- the opposite of GLIMMER's own reported result, where
their best (GLIMMER) configuration beats their own baseline on both datasets:

| | Paper's baseline | Paper's GLIMMER (best) | This replication, standard | This replication, personalized |
|---|---|---|---|---|
| AZT1D | 29.55 +/- 6.49 | 22.48 +/- 3.57 | 31.54 +/- 6.10 | 37.44 +/- 6.35 |
| OhioT1DM | 31.98 +/- 4.15 | 23.97 +/- 3.77 | 36.79 +/- 4.37 | 39.38 +/- 6.46 |

(RMSE, mg/dL. Personalized-weighting numbers shown here are from the properly
independent per-patient search; see 5.3 for why an earlier version of this result
looked more favorable and was wrong.)

### 5.3 A bug that mattered: the GA search wasn't actually independent per patient

An implementation bug was found and fixed during this work: the genetic algorithm's
random seed was accidentally fixed at the same value for every patient, so every
patient's search started from an identical set of candidate weights rather than its
own independent random draw. Because milder weighting nearly always scores better on
plain validation RMSE (any deviation from "no weighting" costs some accuracy by
construction), patients whose shared starting pool happened to include a mild
candidate converged toward it regardless of what that specific patient's own data
actually called for -- inflating the apparent benefit of "personalization." Fixing the
seed and rerunning showed personalized weighting provides substantially less benefit
than first measured (Table in 5.2 reflects the corrected numbers).

### 5.4 Clinical accuracy tells a different, more favorable story

Even though weighted training makes plain RMSE worse, it improves the metrics that
map more directly to patient safety, on both datasets:

- Event-level recall for dysglycemic moments improves substantially (e.g. AZT1D:
  0.507 -> 0.722), at the cost of precision (more false alarms) -- a trade generally
  favorable in a safety context, since missing a real danger event is worse than an
  extra false alarm.
- Clarke Error Grid zone D (a true dangerous moment predicted as safe -- the worst
  kind of miss for a warning system) drops by roughly half on AZT1D and further on
  OhioT1DM.

### 5.5 Recursive forecasting: the model can react to a meal, but not anticipate one

Running a next-step model recursively across roughly 8 hours found: when the model was
asked to predict every input (glucose, basal, bolus, carbs) recursively, it produced a
near-flat, uninformative glucose trajectory, because meal insulin and carbs -- sparse,
behavior-driven events -- were no better predicted than simply guessing their average
value. Substituting real logged meal data for the recursive prediction (keeping only
glucose and basal generated by the model) fixed this specifically: in the hour
surrounding a real logged meal, error dropped from 44.1 to 18.5 mg/dL, and the
predicted trajectory visibly reproduced the post-meal glucose rise for the first time.
Error over the full multi-hour window did not improve as cleanly, for reasons distinct
from meal prediction (a longer horizon than any model in this project was ever trained
or evaluated on).

*(Results are checkpoint-complete for the AZT1D + OhioT1DM replication across both
architectures; the recursive forecasting extension currently covers one patient, one
architecture, as a proof of concept -- flagged as a scope decision to revisit)*

## 6. Discussion / Next Steps

*(to expand)* The central open question this replication raises: is the gap to
GLIMMER's reported numbers a property of this implementation (most likely, given the
directly-evidenced search-budget effect in 5.3), or does it point to something
underspecified in the paper itself? Concrete next steps already identified in Sprint
Planning: try quantitative modeling strategies not covered in GLIMMER, and evaluate
glucose signal predictability directly via approximate/sample entropy (Pincus) as a
way to characterize per-patient forecast difficulty independent of any one model.

## 7. Process Notes (for Process Documentation, not the academic paper itself)

- Sep 10: began literature search; considered ML-Glucose as a candidate
  dataset/framework, ruled out short-term.
- Sep 13: settled on AZT1D + GLIMMER replication; built initial exploratory notebook
  and a first (non-region-weighted) model.
- Sep 14: continued GLIMMER replication; identified OhioT1DM access risk (email
  request pending ~1 week) and began evaluating MetaboNet as a lower-risk path to the
  same data.
- Sep 16: presented in AI Lab; advisor suggested the recursive-forecasting extension.
- Sep 17-19: built and iterated on the recursive forecasting extension across three
  attempts (predict everything -> add time-of-day -> use real meal data for the
  unpredictable inputs), each tested and reported honestly rather than only keeping
  favorable-looking numbers.
