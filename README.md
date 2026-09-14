# AZT1D Analysis

Exploratory analysis of the [AZT1D dataset](https://doi.org/10.17632/gk9m674wcx.1)
(Khamesian et al., 2025, [arXiv:2506.14789](https://arxiv.org/abs/2506.14789)), plus a
from-scratch replication of [GLIMMER](https://arxiv.org/abs/2502.14183), a blood-glucose
forecasting method the same research group built on top of AZT1D.

## Status

Real AZT1D data is loaded and in use (`data/raw/CGM Records/`). All five notebooks below
have been run end to end against it. If `data/raw/` is ever empty (a fresh checkout
without the data), everything falls back to a synthetic placeholder dataset with the same
schema instead of failing.

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
  `azt1d/glimmer/ga.py`) instead of using one fixed weight pair for everyone, and checks
  the second architecture (CNN-Transformer) the paper also evaluates. Per-patient tuning
  is a real improvement over the fixed weights (RMSE 41.18 -> 37.44) but still doesn't
  recover the plain baseline (31.54), unlike what the paper reports.

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
  loading.py      finds/extracts the data, discovers per-subject CSVs, cleans real-data quirks
  synthetic.py    generates placeholder data with the same schema for pre-download work
  metrics.py      time-in-range, GMI, CV, daily insulin/carb aggregates
  plotting.py     shared matplotlib style (validated categorical + status colors)
  glimmer/
    features.py     the paper's 6 model input features (moving average, region label, etc.)
    sequences.py     windowing and the paper's chronological train/val/test split
    model.py         CNNLSTM and CNNTransformer architectures
    losses.py        the region-aware weighted loss (paper Eq. 4)
    train.py         per-patient training loop, shared across every version above
    ga.py            the scoped genetic algorithm weight search
    clinical.py      event-level P/R/F1 and Clarke Error Grid
    checkpoint.py    per-subject checkpointing: trained models and predictions persist to
                      disk so a long run can resume after a crash, and so results can be
                      reused without retraining
notebooks/
  01-05 as described above
data/
  raw/            put the real dataset here (gitignored, not ours to redistribute)
  processed/      extracted data, synthetic data, and checkpoints (all gitignored)
```

## Known limitations

- **The genetic algorithm search is deliberately scoped down** from the paper's own
  budget (population 6 and 6 generations here, vs. their 20 and 25), because the literal
  recipe is an estimated 18+ hours of compute for CNN-LSTM alone across 25 patients. See
  `azt1d/glimmer/ga.py` for the exact tradeoff.
- **Several architecture and training details aren't specified in the paper** (input
  lookback window, Conv1D filter sizes, the Transformer's exact embedding width, optimizer,
  learning rate, epoch count) and were chosen by us to land near the paper's own reported
  parameter counts and results. Documented inline in each module's docstring.
- **OhioT1DM (the paper's second dataset) isn't used here.** [MetaboNet](https://arxiv.org/abs/2601.11505),
  a separate consolidated dataset spanning 14 public T1D sources including both AZT1D and
  OhioT1DM, has been investigated as a path to closing this gap but isn't integrated yet.
- The demographics join (`subject_id` -> paper's Table I "No.") is provisional, not
  confirmed against any ground truth beyond matching subject counts.
