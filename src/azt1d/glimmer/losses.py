"""
The region-aware weighted loss from the GLIMMER paper (Eq. 4):

    Error_total = w_hypo * MAE_hypo + w_normal * MAE_normal + w_hyper * MAE_hyper

where MAE_region is the mean absolute error computed only over samples whose
*true* glucose value falls in that clinical region (hypo/normal/hyper). w_normal
is fixed at 1 in the paper; w_hypo and w_hyper are the free parameters their
genetic algorithm tunes per patient (see azt1d.reference.GLIMMER_PAPER_WEIGHTS
for their published averages, which this v1 uses as fixed values instead of
running the search ourselves -- that's v3).

Implementation note: training targets are z-scored (see train.py) for the same
convergence reason as v0. Region membership is therefore computed by comparing
against the hypo/hyper thresholds *also converted to z-score units* for that
subject, rather than against raw mg/dL -- an affine transform, so the same
samples end up in the same regions either way. The weighted loss itself is then
computed in z-scored units too; since MAE scales linearly, this only rescales
the loss by a constant factor and doesn't change what the weights are pulling
the model towards.
"""

from __future__ import annotations

import torch


def region_weighted_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    hypo_threshold: float,
    hyper_threshold: float,
    w_hypo: float,
    w_normal: float,
    w_hyper: float,
) -> torch.Tensor:
    """
    pred/target: 1D tensors in the same (e.g. z-scored) units as the thresholds.
    Region membership is decided by `target` (the true value), matching the
    paper: we're weighting errors by how clinically dangerous the *true*
    glucose was, not what the model guessed.
    """
    abs_err = torch.abs(pred - target)
    hypo_mask = target < hypo_threshold
    hyper_mask = target > hyper_threshold
    normal_mask = ~hypo_mask & ~hyper_mask

    total = torch.zeros((), device=pred.device, dtype=pred.dtype)
    for weight, mask in ((w_hypo, hypo_mask), (w_normal, normal_mask), (w_hyper, hyper_mask)):
        if mask.any():
            total = total + weight * abs_err[mask].mean()
    return total
