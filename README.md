# AZT1D Analysis

Exploratory analysis of the [AZT1D dataset](https://doi.org/10.17632/gk9m674wcx.1)
(Khamesian et al., 2025, [arXiv:2506.14789](https://arxiv.org/abs/2506.14789)), plus a
from-scratch replication of [GLIMMER](https://arxiv.org/abs/2502.14183), a blood-glucose
forecasting method the same research group built on top of AZT1D and OhioT1DM.

## Status

Real data is loaded and in use for both datasets the paper evaluates: AZT1D directly
(`data/raw/CGM Records/`) and OhioT1DM through [MetaboNet](https://arxiv.org/abs/2601.11505),
a separate consolidation of 14 public T1D datasets, pulled via `azt1d.metabonet` (see
"Known limitations" for how that path was validated). All six notebooks below have been
run end to end. If `data/raw/` is ever empty (a fresh checkout without the data),
AZT1D-specific notebooks fall back to a synthetic placeholder dataset with the same
schema instead of failing.

**Headline finding, in short:** across both datasets and both architectures the paper
evaluates (four independent combinations), the paper's region-weighted loss trained with
one fixed population-average weight pair makes plain RMSE/MAE worse than an ordinary
baseline, and a real per-patient genetic-algorithm weight search recovers some but not
all of that gap. The clinical picture is more favorable: the weighted loss consistently
trades precision for recall and cuts the most dangerous prediction errors (Clarke Error
Grid zone D) roughly in half or better, on both datasets. See notebook 06's final section
for the full account, including the specific, checkable reason this project's numbers
likely differ from the paper's own.

## Notebooks

- **`01_data_exploration.ipynb`**: general AZT1D exploration. Demographics, glycemic
  control metrics (time-in-range, GMI, CV), daily glucose profiles, insulin/carb patterns,
  device mode and time-of-day dysglycemia breakdowns.
- **`02_glimmer_v0_baseline.ipynb`**: a plain CNN-LSTM forecaster, one personalized model
  per patient, 60-minute horizon, ordinary loss. The baseline GLIMMER's own paper compares
  against, not GLIMMER itself yet. Includes a full "getting to know the data" section
  (distributions, correlations, outliers).
- **`03_glimmer_v1_weighted_loss.ipynb`**: adds the paper's actual idea, a loss that
  weights hypo/hyperglycemic errors more heavily than in-range errors, using the paper's
  own published fixed weights. Finds it helps in the hypo/hyper regions but hurts overall
  RMSE/MAE relative to v0.
- **`04_glimmer_v2_clinical_metrics.ipynb`**: does that RMSE-based verdict hold up
  clinically? Event-level precision/recall/F1 and a Clarke Error Grid analysis. Finds a
  more nuanced picture: v1 catches more real danger events (recall 0.507 -> 0.722) and
  meaningfully cuts the most dangerous error type (Clarke zone D, missed danger,
  2.37% -> 1.33%), even though its plain RMSE is worse.
- **`05_glimmer_v3_full_replication.ipynb`**: the actual method, not an approximation of
  it. Runs the paper's per-patient genetic algorithm search (scoped down, see
  `azt1d/glimmer/ga.py`) for both architectures the paper evaluates (CNN-LSTM and
  CNN-Transformer). Per-patient tuning is a real improvement over the fixed weights on
  both (CNN-LSTM: RMSE 41.18 -> 37.44; Transformer: 40.66 -> 37.43, landing on almost the
  same number), but neither recovers the plain baseline.
- **`06_glimmer_ohiot1dm.ipynb`**: the same full v0-v3 treatment, both architectures, on
  OhioT1DM, the paper's other dataset, pulled entirely through `azt1d.metabonet` with no
  dataset-specific code. Reproduces the exact same qualitative pattern notebooks 03-05
  found on AZT1D, plus a direct side-by-side comparison of the two datasets.

## Setup

```bash
source venv/bin/activate
pip install -e .          # installs the azt1d package + dependencies from requirements.txt
python -m ipykernel install --user --name azt1d --display-name "AZT1D (venv)"
jupyter notebook notebooks/01_data_exploration.ipynb
```

(Already done once in this checkout, only needed again if you rebuild the venv.)

## Layout

```
src/azt1d/
  reference.py    schema constants, Table I demographics, GLIMMER's published weights
  loading.py      finds/extracts AZT1D's own raw files, cleans real-data quirks
  metabonet.py    loads any of MetaboNet's 14 datasets onto the same canonical schema,
                   validated row-by-row against loading.py's own AZT1D output
  cleaning.py     the basal forward-fill / zero-fill / missing-CGM handling shared by
                   both loaders above, so downstream code doesn't care which one ran
  synthetic.py    generates placeholder data with the same schema for pre-download work
  metrics.py      time-in-range, GMI, CV, daily insulin/carb aggregates
  plotting.py     shared matplotlib style (validated categorical + status colors)
  glimmer/
    features.py     the paper's 6 model input features (moving average, region label, etc.)
    sequences.py     windowing and the paper's chronological train/val/test split
    model.py         CNNLSTM and CNNTransformer architectures
    losses.py        the region-aware weighted loss (paper Eq. 4)
    train.py         per-patient training loop, dataset-agnostic, shared across every
                      version above and both loaders
    ga.py            the scoped genetic algorithm weight search
    clinical.py      event-level P/R/F1 and Clarke Error Grid
    checkpoint.py    per-subject checkpointing: trained models and predictions persist to
                      disk so a long run can resume after a crash, and so results can be
                      reused without retraining
notebooks/
  01-06 as described above
data/
  raw/            put the real AZT1D dataset here (gitignored, not ours to redistribute)
  processed/      extracted data, synthetic data, and checkpoints (all gitignored)
```

## Known limitations

- **The genetic algorithm search is deliberately scoped down** from the paper's own
  budget (population 6 and 6 generations here, vs. their 20 and 25), because the literal
  recipe is an estimated 18+ hours of compute per dataset per architecture. See
  `azt1d/glimmer/ga.py` for the exact tradeoff. Notebook 06 found a specific, checkable
  consequence: run on OhioT1DM (the dataset the paper's own published weights actually
  came from), this search never found anything close to their reported (3.29, 2.38)
  average, consistently landing on much gentler corrections instead.
- **Several architecture and training details aren't specified in the paper** (input
  lookback window, Conv1D filter sizes, the Transformer's exact embedding width, optimizer,
  learning rate, epoch count) and were chosen by us to land near the paper's own reported
  parameter counts and results. Documented inline in each module's docstring.
- **MetaboNet's AZT1D subset is missing 2 of AZT1D's 25 subjects** (10 and 23), a gap in
  their harmonization, not this codebase. AZT1D's own notebooks (01-05) use `loading.py`
  directly and aren't affected; only `metabonet.py` (used for OhioT1DM in notebook 06, and
  available for AZT1D too) has this gap.
- The demographics join (`subject_id` -> paper's Table I "No.") is provisional, not
  confirmed against any ground truth beyond matching subject counts.
