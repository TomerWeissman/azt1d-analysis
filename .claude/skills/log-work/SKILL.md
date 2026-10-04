---
name: log-work
description: Record work just finished in the capstone project. Files new notebooks and scripts in the current week's folder, updates that week's README (hypotheses, findings, outcomes, work done, file list, what to remember, open items), updates work/HYPOTHESES.md when a result changes a status, records any imported dataset in datasets/, and appends an entry to work/WORK_LOG.md. Use after any analysis, notebook, script, experiment, fix, docs change, or dataset import, or when the user says "log this" or runs /log-work.
---

# Log work

## Layout

```
work/<week>/
  README.md        hypotheses, findings, outcomes, work done, file list, remember, open
  notebooks/       notebooks made that week
  scripts/         builders and experiment scripts made that week
  results/         result tables (if any)
  figures/         figures (if any)
  docs/            written drafts (if any)
work/OVERVIEW.md   where the capstone stands (rewritten at the end of each week)
work/HYPOTHESES.md register of every hypothesis and its status
work/WORK_LOG.md   one entry per unit of work, machine-readable
datasets/          catalog of imported datasets (README.md, sources.csv, symlinks)
src/azt1d/         shared library, stays put
data/              files the loaders read, processed outputs, checkpoints
```

## Week folder

Name each week by the Monday that starts it, written as day_month with no zero padding. Oct 5 2026 gives `5_10`. Sep 28 gives `28_9`.

## Steps

1. Get today's date with `date +%Y-%m-%d`, unless the user gives one. Find the Monday on or before it. That is the folder, `work/<day>_<month>/`. Create the subfolders you need.
2. Check the real state with `git log -n 5 --format='%h %ad %s' --date=short` and `git status --short`. Record only work actually done in this task. Use only numbers that appear in this session's outputs or in saved result files. Do not guess outcomes.
3. Put new notebooks in `work/<week>/notebooks/` and new scripts in `work/<week>/scripts/`. Notebooks use the `PROJECT_ROOT` walk-up line and scripts use the `ROOT`/`WEEK` lines (see CLAUDE.md). Never hardcode a path that depends on folder depth.
4. Update `work/<week>/README.md` from the template below. Add bullets under the existing headings. Do not rewrite earlier bullets except to correct them.
5. If the result changes a hypothesis's status, update its row in `work/HYPOTHESES.md`. Record the evidence with a link to the file that shows it.
6. If you imported a dataset, add it to `datasets/README.md` and `datasets/sources.csv`. Record the source link, license, import date, and loader, and add a symlink in `datasets/`.
7. Append one entry to `work/WORK_LOG.md` in the format given at the top of that file. Keep the keys in order. Use the commit hash for `commits:`, or `none`. Entries are never edited. To correct one, append a `type: correction` entry that names the entry it fixes.
8. Commit and push only the files this task made or changed. Stage each path by name. Never stage the Notion manifest (`.claude/notion_manifest.md`). Follow the project's commit convention.

## README template

```
# Week <folder>: <Monday date> to <Sunday date>

## Hypotheses this week
- **<ID>, <status>**: <hypothesis, in one line>. <evidence in one line>.

## Findings
- <what we learned, in plain language>

## Outcomes
- <result, with its number>

## Work done
- <date>: <item> (<commit hash, or "uncommitted">)

## Files
| File | What it is | Status |
|---|---|---|
| [notebooks/<name>](notebooks/<name>) | <one line: what it does and what it shows> | current / stale / superseded |

## Remember
- <caveats, decisions, things to check later>

## Open
- <item>
```
