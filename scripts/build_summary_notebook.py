"""Assembles notebooks/07_summary.ipynb from already-computed cells in notebooks 01-06.

Every code cell copied here keeps its original `outputs` untouched -- nothing in this
script re-executes anything. See plans/lazy-dreaming-boole.md for the full rationale
and manifest this script implements.
"""
import copy
import re
from pathlib import Path

import nbformat
from nbformat.v4 import new_markdown_cell

ROOT = Path(__file__).resolve().parent.parent
NB_DIR = ROOT / "notebooks"

SRC_PATHS = {
    "nb01": NB_DIR / "01_data_exploration.ipynb",
    "nb02": NB_DIR / "02_glimmer_v0_baseline.ipynb",
    "nb03": NB_DIR / "03_glimmer_v1_weighted_loss.ipynb",
    "nb04": NB_DIR / "04_glimmer_v2_clinical_metrics.ipynb",
    "nb05": NB_DIR / "05_glimmer_v3_full_replication.ipynb",
    "nb06": NB_DIR / "06_glimmer_ohiot1dm.ipynb",
}

_notebooks = {k: nbformat.read(p, as_version=4) for k, p in SRC_PATHS.items()}
_index = {k: {c["id"]: c for c in nb.cells if "id" in c} for k, nb in _notebooks.items()}


def _source_text(cell):
    src = cell["source"]
    return src if isinstance(src, str) else "".join(src)


def code(nbkey, cid):
    """Copy a code cell verbatim (source + outputs), execution_count cleared."""
    c = copy.deepcopy(_index[nbkey][cid])
    assert c["cell_type"] == "code", f"{nbkey}/{cid} is not a code cell"
    c["execution_count"] = None
    for out in c.get("outputs", []):
        if "execution_count" in out:
            out["execution_count"] = None
    return c


def md_verbatim(nbkey, cid, level=3):
    """Copy a markdown cell, stripping any leading numbered header and
    demoting it to `level` so it nests under this notebook's own section
    headers instead of carrying the source notebook's own numbering."""
    c = copy.deepcopy(_index[nbkey][cid])
    assert c["cell_type"] == "markdown", f"{nbkey}/{cid} is not a markdown cell"
    text = _source_text(c)
    lines = text.split("\n")
    if lines and lines[0].lstrip().startswith("#"):
        heading = lines[0].lstrip("#").strip()
        heading = re.sub(r"^\d+\.\s*", "", heading)
        lines[0] = "#" * level + " " + heading
    c["source"] = "\n".join(lines)
    return c


def md(text):
    return new_markdown_cell(text)


def md_edit(nbkey, cid, replacements, level=3):
    """Like md_verbatim, but also applies a handful of exact string swaps --
    for cells whose body text references things ("notebook 03", "the intro
    above") that don't resolve correctly once merged into one document."""
    c = md_verbatim(nbkey, cid, level=level)
    text = _source_text(c)
    for old, new in replacements:
        assert old in text, f"replacement text not found in {nbkey}/{cid}: {old!r}"
        text = text.replace(old, new)
    c["source"] = text
    return c


def slug(title):
    s = re.sub(r"[^\w\s-]", "", title).strip().lower()
    return re.sub(r"[\s]+", "-", s)


# ---------------------------------------------------------------------------
# Sections: (number, title, intro paragraph, [cells])
# ---------------------------------------------------------------------------

sections = []

sections.append((
    1, "The dataset",
    "Who this data is about, and whether it's trustworthy enough to build on.",
    [
        md(
            "AZT1D is real-world data from 25 people with type 1 diabetes, each already "
            "using an automated insulin delivery system: a continuous glucose monitor "
            "(Dexcom G6 Pro) paired with an insulin pump (Tandem t:slim X2) that "
            "automatically adjusts insulin based on glucose trends. Researchers at Mayo "
            "Clinic Arizona pulled about a month of data per person straight off each "
            "patient's own pump and sensor during routine care between December 2023 and "
            "April 2024, with informed consent and ethics review board approval. It comes "
            "from a named, credentialed team, went through ethics review, and is published "
            "openly on Mendeley Data. Worth keeping in mind: it's brand new (2025), just 25 "
            "people from one clinic, and skews older (ages 27-80, averaging around 60) -- "
            "findings here describe this group well, not necessarily children or other "
            "populations without checking."
        ),
        md(
            "### Demographics\n\n"
            "Table I from the paper (age, sex, A1c) is transcribed in "
            "`azt1d.reference.DEMOGRAPHICS_TABLE1`. Caveat: the paper's patient index is "
            "used here as `subject_id` provisionally -- this join hasn't been "
            "independently confirmed."
        ),
        code("nb01", "f3037d49"),
    ],
))

sections.append((
    2, "What the data actually shows",
    "The patterns in the raw data worth knowing before any modeling starts.",
    [
        md(
            "**What's in each reading:** glucose (CGM, mg/dL, every 5 minutes; roughly "
            "70-180 is the everyday target range), background (\"basal\") insulin (a "
            "steady trickle, U/hr), meal-time (\"bolus\") insulin (a one-time dose for food "
            "or to correct a high, in units), and carbs eaten at logged meals (grams)."
        ),
        md_verbatim("nb01", "65418108"),
        code("nb01", "791b640d"),
        code("nb01", "f9f55a84"),
        code("nb02", "78bd8e35"),
        md_verbatim("nb02", "e9848a9d"),
        code("nb02", "7540dc83"),
        md_verbatim("nb02", "75206ffa"),
        code("nb02", "545c4c50"),
        md_verbatim("nb02", "ba2fa6dc"),
        code("nb02", "6f83c334"),
        md_verbatim("nb02", "305e47fe"),
        code("nb02", "52f2c99f"),
        md_verbatim("nb02", "bac0ad40"),
        md(
            "### Insulin & carbohydrate patterns\n\n"
            "Per-subject-per-day totals for basal, bolus, and carbohydrate intake."
        ),
        code("nb01", "73beeef8"),
        md(
            "### Time-of-day patterns\n\n"
            "Echoing the paper's elderly-patient analysis (Fig. 3): hypo/hyperglycemia "
            "burden isn't evenly spread across the day."
        ),
        code("nb01", "0f790482"),
    ],
))

sections.append((
    3, "A simple baseline",
    "Before touching the paper's actual idea: how well does an ordinary model do at "
    "predicting glucose an hour ahead?",
    [
        code("nb02", "7f786165"),
        md_verbatim("nb02", "ae2fc3c0"),
        md_verbatim("nb02", "643c8180"),
        code("nb02", "4b34ca0f"),
        code("nb02", "4d1ed3ca"),
        md(
            "**Observation:** Does someone with a more erratic glucose pattern actually "
            "get worse predictions? A positive correlation says yes: the model struggles "
            "most exactly where the person's own glucose is hardest to predict in the "
            "first place, not from random noise."
        ),
    ],
))

sections.append((
    4, "Weighting the loss toward danger zones",
    "The paper's core idea: don't treat every prediction error the same. Does it "
    "actually help?",
    [
        md(
            "Split every prediction error into three groups based on what the true "
            "glucose actually was: below 70 (hypo), between 70 and 180 (normal), above "
            "180 (hyper). Average the error within each group separately, then combine "
            "the three averages with different weights instead of one plain overall "
            "average. A hypo error counts 3.29 times as much as a normal error, a hyper "
            "error counts 2.38 times as much -- the paper's own published weights, used "
            "here as a fixed population-wide setting (per-patient tuning comes later)."
        ),
        md_verbatim("nb03", "14eaa804"),
        code("nb03", "bd595f58"),
        code("nb03", "65d321a9"),
        md_verbatim("nb03", "49958436"),
        code("nb03", "82bff859"),
        code("nb03", "e392b253"),
        md_verbatim("nb03", "30ace8b2"),
    ],
))

sections.append((
    5, "Does that trade-off matter clinically",
    "The weighted loss made plain RMSE worse. Does it still make the model more "
    "clinically useful?",
    [
        md(
            "### Event detection: does v1 catch more real danger moments?\n\n"
            "This is the metric that should move if the weighted loss is doing its job, "
            "even though the plain RMSE comparison above was worse overall."
        ),
        code("nb04", "5a8b2770"),
        code("nb04", "e5d7734b"),
        code("nb04", "f8b75474"),
        code("nb04", "c23e7820"),
        md_edit("nb04", "1bee05e1", [
            ("Notebook 03 found v1's weighted loss", "Section 4 above found v1's weighted loss"),
        ]),
    ],
))

sections.append((
    6, "Tuning the weights per patient",
    "The real GLIMMER method, not an approximation of it: a genetic algorithm searches "
    "each patient's own best weights, on AZT1D.",
    [
        md(
            "### The genetic algorithm: search for each patient's own weights\n\n"
            "This is Algorithm 1 from the paper, scoped down (see `azt1d/glimmer/ga.py` "
            "for the exact tradeoff), run once per patient. Each patient's search is "
            "independent of everyone else's, exactly like the paper's own "
            "per-patient personalization."
        ),
        code("nb05", "cccc074f"),
        code("nb05", "f4c60d07"),
        md(
            "### v0 vs. v1 (fixed weights) vs. v3 (per-patient tuned weights)\n\n"
            "The comparison that actually tests whether per-patient tuning fixes what "
            "went wrong with the fixed weights above."
        ),
        code("nb05", "745e76d2"),
        code("nb05", "c2b99943"),
        md(
            "### The complete picture: both architectures, all three versions\n\n"
            "Everything built so far, side by side."
        ),
        code("nb05", "77df66ce"),
        code("nb05", "8c22dc5b"),
        md(
            "The Transformer confirms this isn't a CNN-LSTM quirk: its v3 RMSE (37.43) "
            "lands almost exactly on CNN-LSTM's (37.44), and the same pattern holds "
            "throughout, v1 (fixed weights) worse than v0, v3 (per-patient tuned) better "
            "than v1 but still short of v0, every comparison statistically significant "
            "(Wilcoxon, all p < 0.0001) in that direction. Two different architectures "
            "landing on essentially the same number is a strong signal the gap to the "
            "paper's reported results is a property of this implementation (the "
            "scoped-down GA search, most likely) rather than either specific model.\n\n"
            "The per-patient weights found for the Transformer spread out similarly to "
            "CNN-LSTM's: mostly gentle corrections in the 1-2 range, with a handful of "
            "outliers needing much stronger weights. Subject 9 is the extreme case on "
            "both architectures (CNN-LSTM: 4.36, 6.26; Transformer: 5.89, 9.42, the "
            "strongest correction of anyone on either run), though which *other* patients "
            "stand out differs by architecture, subject 13 needed strong correction on "
            "CNN-LSTM (4.90, 1.00) but was mild on the Transformer (1.45, 1.18); subjects "
            "18 and 24 were the Transformer's other outliers instead."
        ),
    ],
))

sections.append((
    7, "Testing it on a second dataset",
    "Everything so far is AZT1D only. OhioT1DM, the paper's other dataset, gets pulled "
    "in through MetaboNet (a separate consolidation of 14 public T1D datasets) and run "
    "through the exact same pipeline, with no dataset-specific code.",
    [
        md_edit("nb06", "57e9103c", [
            ("Notebook 04 found v1 trades", "Section 5 above found v1 trades"),
        ]),
        code("nb06", "d7278fce"),
        md_edit("nb06", "94b1a1b0", [
            ("Same scoped-down search as notebook 05", "Same scoped-down search as the AZT1D one above"),
        ]),
        code("nb06", "e8bb88a3"),
        md_verbatim("nb06", "8976bee5"),
        code("nb06", "44053f4f"),
        code("nb06", "e798271a"),
        md_verbatim("nb06", "66ceb909"),
        code("nb06", "0f3dcf08"),
        code("nb06", "eff6091e"),
    ],
))

sections.append((
    8, "Bottom line",
    "",
    [
        md(
            "OhioT1DM shows the exact same pattern AZT1D did, on both architectures: v1 "
            "(fixed weights) worse than v0, v3 (per-patient tuned) better than v1 but "
            "still short of v0, every comparison statistically significant (Wilcoxon, all "
            "p < 0.005). That's four independent tests of the same claim, two datasets "
            "times two architectures, all landing the same way. This is strong evidence "
            "the gap to the paper's reported numbers is a real, consistent property of "
            "this implementation (almost certainly the scoped-down GA search, population "
            "6 and 6 generations against the paper's 20 and 25) rather than something "
            "specific to one dataset or model.\n\n"
            "**A genuine surprise:** the (3.29, 2.38) weights used throughout this project "
            "as \"the paper's published weights\" came from a GA search on OhioT1DM "
            "specifically, not AZT1D. Running our own search on that same dataset was the "
            "most direct test available of whether this GA can reproduce the paper's own "
            "number, and it doesn't come close: every one of the 12 OhioT1DM patients "
            "found weights under 2.3, nowhere near the paper's average of (3.29, 2.38), "
            "let alone the higher end of their own reported range. A search this much "
            "smaller than the paper's own consistently converges toward much gentler "
            "corrections. That's a specific, checkable claim about what a bigger search "
            "would need to close, not just a general excuse.\n\n"
            "**A real reversal worth flagging:** on AZT1D, per-patient tuning (v3) beat "
            "the fixed weights (v1) in the hypo region specifically. On OhioT1DM it's the "
            "opposite, v1's hypo MAE (21.85) is notably better than v3's (30.45), both far "
            "better than v0's (44.08). The fixed population-average weights, exactly "
            "because they're less tailored to any one person, apparently do something "
            "right for OhioT1DM's hypo region that this smaller per-patient search doesn't "
            "reliably find. Not a contradiction of the overall story, but a reminder that "
            "\"per-patient is always better\" doesn't hold in every region on every "
            "dataset.\n\n"
            "**The clinical picture replicates too, and more strongly:** OhioT1DM's event "
            "detection shows the same precision-for-recall trade (0.768 to 0.585 "
            "precision, 0.693 to 0.833 recall) found on AZT1D, and the Clarke Error Grid's "
            "dangerous Zone D drops even further in relative terms, 3.65% down to 1.32%, "
            "nearly a 3x reduction versus AZT1D's roughly 2x. Whatever this loss weighting "
            "is actually doing to the model's behavior, it's doing it consistently across "
            "both datasets.\n\n"
            "**What \"full replication\" means at this point:** both datasets, both "
            "architectures, the real per-patient GA search (not just the paper's "
            "published averages), clinical metrics, and statistical significance testing "
            "throughout. What's still different from the paper: the GA's own compute "
            "budget (this one is deliberately much smaller, for reasons laid out in "
            "`azt1d/glimmer/ga.py`), and the ~11 other benchmark models in the paper's "
            "Table 7, which were never in scope here. Closing the GA budget gap is the "
            "one concrete, well-evidenced next step if this ever needs to go further -- "
            "everything else here now points at that same conclusion from four "
            "independent directions."
        ),
    ],
))

# ---------------------------------------------------------------------------
# Assemble
# ---------------------------------------------------------------------------

cells = []

cells.append(md(
    "# GLIMMER on AZT1D and OhioT1DM: findings\n\n"
    "*A pre-rendered summary assembled from notebooks 01-06 -- every plot and table "
    "here is copied output, not recomputed. This notebook is not meant to be re-run "
    "top to bottom (it draws on six separate kernel sessions and would error if "
    "executed fresh). For the full detailed record, including methodology and every "
    "intermediate step, see notebooks 01-06.*"
))

toc_lines = ["## Table of contents", ""]
for num, title, _, _ in sections:
    heading = f"{num}. {title}"
    toc_lines.append(f"{num}. [{title}](#{slug(heading)})")
cells.append(md("\n".join(toc_lines)))

for num, title, intro, piece_cells in sections:
    heading = f"{num}. {title}"
    header_cell = md(f"## {heading}")
    cells.append(header_cell)
    if intro:
        cells.append(md(f"*{intro}*"))
    cells.extend(piece_cells)

new_nb = nbformat.v4.new_notebook()
new_nb["cells"] = cells
new_nb["metadata"] = copy.deepcopy(_notebooks["nb02"]["metadata"])

out_path = NB_DIR / "07_summary.ipynb"
nbformat.write(new_nb, out_path)
print(f"Wrote {out_path} ({len(cells)} cells)")
