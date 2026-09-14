# AZT1D Analysis

Exploratory data analysis of the [AZT1D dataset](https://doi.org/10.17632/gk9m674wcx.1)
(Khamesian et al., 2025, [arXiv:2506.14789](https://arxiv.org/abs/2506.14789)) — CGM,
insulin delivery, carbohydrate intake, and device-mode data from 25 individuals with
type 1 diabetes on automated insulin delivery systems.

## Status

The real dataset (`AZT1D 2025.zip`, ~776 MB) hasn't been downloaded yet. Everything here
already runs end-to-end against a **synthetic placeholder dataset** with the same schema,
so the loading/EDA pipeline is pre-built and verified. Once the real zip finishes
downloading, drop it into `data/raw/` and re-run the notebook — no code changes needed.

## Setup

```bash
source venv/bin/activate
pip install -e .          # installs the azt1d package + dependencies from requirements.txt
python -m ipykernel install --user --name azt1d --display-name "AZT1D (venv)"
jupyter notebook notebooks/01_data_exploration.ipynb
```

(Already done once in this checkout — re-run only if you rebuild the venv.)

## Getting the real data

1. Download `AZT1D 2025.zip` from https://doi.org/10.17632/gk9m674wcx.1
2. Place it in `data/raw/`
3. Re-run `notebooks/01_data_exploration.ipynb` top to bottom — the loader in Section 1
   auto-detects the zip, extracts it into `data/processed/extracted/`, and switches off
   the synthetic fallback automatically.

If the real per-subject CSVs use different column header spellings than the paper
describes, `azt1d/loading.py` will raise a clear error naming the missing columns — add
the real spelling to `NAME_VARIANTS` in that file.

## Layout

```
src/azt1d/
  reference.py    schema constants + Table I demographics, transcribed from the paper
  loading.py      finds/extracts the zip, discovers per-subject CSVs, normalizes columns
  synthetic.py    generates placeholder data with the same schema for pre-download work
  metrics.py      time-in-range, GMI, CV, daily insulin/carb aggregates
  plotting.py     shared matplotlib style (validated categorical + status colors)
notebooks/
  01_data_exploration.ipynb   the write-up
data/
  raw/            put the downloaded zip here (gitignored, not ours to redistribute)
  processed/      extracted real data / generated synthetic data (gitignored)
```

## Known caveat

The exact internal file/folder layout of the real archive wasn't visible ahead of
download (Mendeley only exposes it as one zip). `loading.py` discovers subject files by
pattern rather than a hardcoded path, but if the real layout is unusual, adjust
`discover_subject_files()` accordingly. The demographics join (`subject_id` -> paper's
Table I "No.") is also provisional until confirmed against the real filenames.
