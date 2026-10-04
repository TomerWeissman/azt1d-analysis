# Capstone: how work goes

This repo is the capstone: glucose forecasting and uncertainty for type 1 diabetes. The Python package is still named `azt1d`, so imports look like `from azt1d ...`.

## Where things are

- `work/OVERVIEW.md`: where the capstone stands and why. Rewritten at the end of every week.
- `work/README.md`: index of all weeks.
- `work/HYPOTHESES.md`: register of every hypothesis, with status (testing, confirmed, supported, not supported, invalidated, not yet tested, dropped) and evidence. Update it whenever a result changes a status.
- `work/WORK_LOG.md`: machine-readable log of every unit of work.
- `work/<week>/`: one folder per week, named by the Monday that starts it, written as day_month with no zero padding (`28_9` is Sep 28, 2026). Each has a `README.md` (hypotheses, findings, outcomes, work done, a file list, what to remember, open items) plus `notebooks/`, `scripts/`, `results/`, `figures/`, and `docs/`.
- `datasets/`: catalog of every dataset imported, with source links and licenses (`README.md`, `sources.csv`), and symlinks to where each one lives.
- `src/azt1d/`: shared library. Stays put, not week-specific.
- `data/`: files the loaders read, processed outputs, and checkpoints.

## Weeks

- The next week always builds on the last. Files the user keeps working on are **copied** into the new week, never moved. The originals stay where they were.
- When a new week starts (the `start_week` skill), first update `work/OVERVIEW.md` for the week that just ended. Then ask the user what they found, what their goals are, and which files carry forward.
- Check for a new week at the start of every session. If `work/<this week>/` does not exist, run `start_week` before other work.

## Every unit of work

- Record it with the `log-work` skill: the week README and `work/WORK_LOG.md`.
- Hypothesis work goes through the `hypothesis_testing` skill, and the outcome goes into `work/HYPOTHESES.md`.
- New notebooks and scripts go in the current week's folder. Never in the top-level folders.
- Notebooks find the repo root with `PROJECT_ROOT = next(p for p in [Path.cwd(), *Path.cwd().parents] if (p / "pyproject.toml").exists())`. Scripts use `ROOT = next(p for p in Path(__file__).resolve().parents if (p / "pyproject.toml").exists())` and `WEEK = Path(__file__).resolve().parent.parent`. Never hardcode a path that depends on folder depth.
- Don't re-run long notebooks for text-only edits.
- Before running anything over 10 minutes, say so and get a yes.

## Datasets

- Record every imported dataset in `datasets/README.md` and `datasets/sources.csv`: the source link, license, import date, loader, and where it lives.
- Add a symlink in `datasets/` pointing at the data.
- Raw dataset files stay out of git (gitignored). Their catalog entries are in git.

## Backups

- Everything is backed up to GitHub: code, notebooks, results, figures, docs, checkpoints, skills, and the datasets catalog.
- Exceptions: raw dataset files (`data/raw/`, `data/hupa_ucm/`, and the MetaboNet parquet), because they are large and not ours to redistribute. The Notion manifest in `.claude/notion_manifest.md`, because it holds private workspace identifiers and the repo is public.
- Stage files by name, never with `git add -A` or `git add .` without paths. Commit and push finished work.
- The repo is public. Don't commit anything private.

## Numbers and claims

- Report numbers only from outputs shown in the session or from saved result files.
- Results from one seed or five patients are preliminary. Say so.
- Use plain language. No em dashes in written output.
- A verdict is supported, refuted, or inconclusive against a threshold stated before the run.

## Traps already hit

- **Seeds.** Each ensemble member and each patient needs its own seed. One shared seed made every per-patient search identical.
- **Test data stays out** of fitting, calibration, and early stopping.
- **Wider bands only add alarms.** Compare methods at a matched false-trigger rate.
- **Pooled results can be one patient.** In HUPA-UCM, patient 27 is 53% of the rows.
- **HUPA-UCM has straight-line filler** (about 8% of pooled readings).
- **The AZT1D loader scans `data/raw/` recursively.** Never put another dataset's CSVs there.
