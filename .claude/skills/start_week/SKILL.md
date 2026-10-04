---
name: start_week
description: Start a new work week in the capstone project. Rewrites the overview for the week that just ended, checks the Notion capstone pages (using .claude/notion_manifest.md to decide what to read in depth), asks what the user found and what their goals are, copies the files they keep working on into the new week, and creates the week's folder and README. Run it whenever work/<week>/ does not exist for the current Monday-to-Sunday week, or when the user says "start the week" or runs /start_week.
---

# Start a week

Weeks run Monday to Sunday. The folder is named by the Monday, as day_month with no zero padding: Oct 5 2026 is `work/5_10/`.

Never skip the questions in step 5. The user decides what carries forward and what the goals are.

## 1. Check that it is a new week

Get today's date with `date +%Y-%m-%d` and find the Monday on or before it. If `work/<folder>/` already exists, say so and stop, unless the user asks to restart it.

## 2. Read the week that just ended

- `work/<previous>/README.md`: hypotheses, findings, outcomes, open items, and what to remember.
- `work/HYPOTHESES.md`: any rows whose status changed that week.
- The last entries of `work/WORK_LOG.md` for that week.
- `git log -n 10 --format='%h %ad %s' --date=short` to confirm what landed.

## 3. Rewrite `work/OVERVIEW.md`

It is one half paragraph: the current stage and why the project is there. Rewrite it from step 2, keep it short, and update the date at the top. Show the user the new version and ask them to correct it in step 5.

## 4. Read Notion, at the depth the manifest sets

Read `.claude/notion_manifest.md` first. If it is missing, rebuild it: fetch the capstone root and record each child with its role and depth, using the rules in the file. Ask the user to classify anything new.

1. Fetch the capstone root and compare its children with the manifest.
   - **New child:** ask how deep to read it, and add a row.
   - **Missing child:** check whether it moved to Trash. Update the row.
2. Read **full** pages that changed since `last_read`.
   - **Journal:** the entries dated in the previous week and in the first days of the new one, plus any goal statements.
   - **My Gantt** (the planning database): query the rows. Pull the course week that covers the new week, and the deadlines within three weeks.
3. Skim **skim** pages only if they changed.
4. Don't open **skip** pages.

Record today's date as `last_read` for each page you read.

## 5. Show where the user is, then ask

Show a summary of at most 15 lines:
- what the previous week produced (outcomes from its README)
- hypotheses and their current status
- open items, plus anything the journal says is next
- the next deadlines from the Gantt

Then ask, using AskUserQuestion for the choices and free text for the rest:
1. **Findings.** "Here is what I think you found last week: [list]. What is right, what is wrong, and what did I miss?"
2. **Overview.** "Does this paragraph describe where you are?"
3. **Goals.** Up to three goals for this week.
4. **Carry-forward.** Multi-select the files from the previous week's `notebooks/`, `scripts/`, `results/`, `figures/`, and `docs/` that you'll keep working on. Include an option for none.
5. **Datasets.** Did you import any new data this week? If so, record it in `datasets/` (see step 6).

## 6. Create the week

1. Create `work/<folder>/` with `README.md` and the subfolders you need: `notebooks/`, `scripts/`, `results/`, `figures/`, `docs/`. Skip any you won't use.
2. **Copy** each file the user chose into the matching subfolder. Use `cp`, never `mv`. The original stays in the previous week.
3. Write `work/<folder>/README.md` from the template in `log-work`:
   - **Hypotheses this week:** the rows from `HYPOTHESES.md` the user is testing, with their current status.
   - **Findings, Outcomes, Work done:** empty.
   - **Files:** the copied files, marked as carried from the previous week, with their original paths.
   - **Remember:** caveats the user raised.
   - **Open:** the goals from step 5, plus any open items the user kept.
4. Add one entry to `work/WORK_LOG.md`: type `docs`, title "Week start: goals and carry-forward", with the goals and the carried files in `notes:`.
5. Update the `last_read` dates in `.claude/notion_manifest.md`.
6. If the user imported data, add it to `datasets/README.md` and `datasets/sources.csv`, and add the symlink in `datasets/`.

## 7. Commit and push

Commit the new week's files, the copied files, the updated `OVERVIEW.md`, the WORK_LOG entry, and any dataset changes. Stage each path by name. Push.

## Notes

- Course weeks (the Gantt's "Week N") don't line up exactly with the Monday folders. Use the folders for work and the Gantt for deadlines.
- If Notion is unreachable, say so, and ask the user for next steps instead of guessing them from the old journal.
