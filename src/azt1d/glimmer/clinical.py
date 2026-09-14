"""
The paper's clinical-relevance evaluation (v2): event-level precision/recall/F1
for dysglycemia detection (paper Eq. 7), and Clarke Error Grid (CEG) zone
classification (paper Section 5.3.4).

Assumption: the paper defines an "event" only in words ("event-level detection,
e.g. hypoglycemia onset"), not as an exact algorithm. We treat every individual
5-minute prediction as its own event, classified as dysglycemic if the true (or
predicted) value falls outside 70-180 mg/dL. That's a simpler, point-wise
reading of Eq. 7 rather than a runs-based "onset" detector that would need its
own definition of how long a run has to be, or how much overlap counts as a
match. Worth revisiting if point-wise recall/precision look implausibly high or
low next to the paper's own numbers.

Clarke Error Grid zone boundaries below follow the widely-used public
implementation of Clarke's original rule set (e.g.
github.com/suetAndTie/ClarkeErrorGrid), verified directly against that source
rather than reconstructed from memory.
"""

from __future__ import annotations

import numpy as np

from .. import reference as ref


def dysglycemia_event_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    hypo_threshold: float = ref.HYPO_THRESHOLD,
    hyper_threshold: float = ref.HYPER_THRESHOLD,
) -> dict[str, float]:
    """Point-wise precision/recall/F1 for "is this reading dysglycemic" (paper Eq. 7)."""
    true_event = (y_true < hypo_threshold) | (y_true > hyper_threshold)
    pred_event = (y_pred < hypo_threshold) | (y_pred > hyper_threshold)

    tp = int(np.sum(true_event & pred_event))
    fp = int(np.sum(~true_event & pred_event))
    fn = int(np.sum(true_event & ~pred_event))

    precision = tp / (tp + fp) if (tp + fp) > 0 else float("nan")
    recall = tp / (tp + fn) if (tp + fn) > 0 else float("nan")
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else float("nan")

    return {"precision": precision, "recall": recall, "f1": f1, "tp": tp, "fp": fp, "fn": fn}


def clarke_error_grid_zones(reference_values: np.ndarray, prediction_values: np.ndarray) -> np.ndarray:
    """
    Classify each (reference, prediction) pair into a Clarke EGA zone (A-E).
    Vectorized, first-match-wins in the order A, E, C, D, B -- matching the
    standard reference implementation's if/elif precedence exactly.
    """
    ref_v = np.asarray(reference_values, dtype=float)
    pred_v = np.asarray(prediction_values, dtype=float)

    zone_a = ((ref_v <= 70) & (pred_v <= 70)) | ((pred_v <= 1.2 * ref_v) & (pred_v >= 0.8 * ref_v))
    zone_e = ((ref_v >= 180) & (pred_v <= 70)) | ((ref_v <= 70) & (pred_v >= 180))
    zone_c = (((ref_v >= 70) & (ref_v <= 290)) & (pred_v >= ref_v + 110)) | (
        ((ref_v >= 130) & (ref_v <= 180)) & (pred_v <= (7 / 5) * ref_v - 182)
    )
    zone_d = (
        ((ref_v >= 240) & (pred_v >= 70) & (pred_v <= 180))
        | ((ref_v <= 175 / 3) & (pred_v <= 180) & (pred_v >= 70))
        | (((ref_v >= 175 / 3) & (ref_v <= 70)) & (pred_v >= (6 / 5) * ref_v))
    )

    return np.select([zone_a, zone_e, zone_c, zone_d], ["A", "E", "C", "D"], default="B")


def clarke_zone_percentages(reference_values: np.ndarray, prediction_values: np.ndarray) -> dict[str, float]:
    """Percent of points in each CEG zone, matching the shape of the paper's Table 6."""
    zones = clarke_error_grid_zones(reference_values, prediction_values)
    n = len(zones)
    return {z: 100.0 * np.sum(zones == z) / n for z in ("A", "B", "C", "D", "E")}


def draw_clarke_grid(ax, reference_values: np.ndarray, prediction_values: np.ndarray, color: str) -> None:
    """
    Scatter plot with the standard CEG zone boundary lines overlaid, matching
    the paper's Fig. 4 style. Boundary line coordinates follow the same public
    reference implementation the zone-classification rules above were checked
    against.
    """
    ax.scatter(reference_values, prediction_values, s=6, alpha=0.25, color=color)
    ax.plot([0, 400], [0, 400], ":", color="black", linewidth=1)

    boundary_lines = [
        ([0, 175 / 3], [70, 70]),
        ([175 / 3, 400 / 1.2], [70, 400]),
        ([70, 70], [84, 400]),
        ([0, 70], [180, 180]),
        ([70, 290], [180, 400]),
        ([70, 70], [0, 56]),
        ([70, 400], [56, 320]),
        ([180, 180], [0, 70]),
        ([180, 400], [70, 70]),
        ([240, 240], [70, 180]),
        ([240, 400], [180, 180]),
        ([130, 180], [0, 70]),
    ]
    for xs, ys in boundary_lines:
        ax.plot(xs, ys, "-", color="black", linewidth=0.8)

    ax.set_xlim(0, 400)
    ax.set_ylim(0, 400)
    ax.set_aspect("equal")
    ax.set_xlabel("Reference (actual) glucose (mg/dL)")
    ax.set_ylabel("Predicted glucose (mg/dL)")
