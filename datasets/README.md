# Datasets

Every dataset this project has imported, with where it came from and where it lives.

- `datasets/<name>` links to the data on disk. These are symlinks, so the files stay where the loaders expect them.
- `sources.csv` has the same catalog in machine-readable form.
- Raw data is not in git. It is gitignored because the dataset files are large and not ours to redistribute.

| Name | What it is | Local location | Source | Imported |
|---|---|---|---|---|
| [azt1d](azt1d) | AZT1D: 25 patients, real-world CGM and insulin records | `data/raw/CGM Records/` | [doi:10.17632/gk9m674wcx.1](https://doi.org/10.17632/gk9m674wcx.1) (Mendeley Data, CC BY 4.0) | 2026-09-13 |
| [hupa_ucm](hupa_ucm) | HUPA-UCM Diabetes Dataset: 25 patients, CGM with insulin, carbs, and heart rate | `data/hupa_ucm/HUPA-UCM Diabetes Dataset/` | [data.mendeley.com/datasets/3hbcscwz44/1](https://data.mendeley.com/datasets/3hbcscwz44/1) | 2026-09-28 |
| [metabonet](metabonet) | MetaboNet: 14 public T1D datasets in one parquet file, including OhioT1DM | `~/Downloads/metabonet_public.parquet` (1.3 GB, outside the repo) | Paper: [arXiv:2601.11505](https://arxiv.org/abs/2601.11505). Download link not recorded. | 2026-09-14 |

Not on disk:
- **OhioT1DM**: accessed only through MetaboNet. A direct access request was sent about Sep 8 and had no reply as of Oct 4 (Notion journal).
- **ML-Glucose** ([arXiv:2507.14077](https://arxiv.org/html/2507.14077v1)): considered on Sep 10 and never imported.

## How to add a dataset

1. Put the files under `data/<name>/` (raw files are gitignored).
2. Add a row to this table and to `sources.csv`. Record the source URL, the license, the date imported, and the loader in `src/azt1d/`.
3. Add a symlink in `datasets/`: `ln -s "../data/<name>" datasets/<name>`.
4. Note the import in the week's README and in `work/WORK_LOG.md`.

Where things are: `datasets/` is the catalog and the entry points. `data/` holds the files the loaders read, plus processed outputs and checkpoints.
