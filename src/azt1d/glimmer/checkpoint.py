"""
Checkpointing so a long run can be resumed after a crash without losing
already-completed subjects, and so a completed run's trained models and
predictions can be reused later (new plots, new metrics, reloading the actual
model for fresh inference) without retraining.

One file per subject per run, under a directory the caller names -- this
module doesn't impose a naming scheme; azt1d.glimmer.train.run_with_checkpoints
is what actually decides where things live (data/processed/checkpoints/<run>/).

Model-training checkpoints (save_result/load_result) are pickled via
torch.save, since a SubjectResult carries a state_dict of tensors alongside
its plain Python/numpy fields. Loading doesn't need to import SubjectResult
here -- pickle resolves the class from the module path stored in the file.

GA checkpoints (save_ga_result/load_ga_result) are plain JSON instead, since a
GA result has no tensors in it and JSON is easy to read directly if you just
want to peek at what weights a patient landed on without opening Python.

Every write goes to a temp file first, then an atomic rename -- a crash mid
write can never leave a corrupt checkpoint behind for the next run to trip on.
"""

from __future__ import annotations

import json
from pathlib import Path

import torch
from torch import nn

from .model import MODEL_CLASSES


def _result_path(checkpoint_dir: Path, subject_id: int) -> Path:
    return Path(checkpoint_dir) / f"subject_{subject_id}.pt"


def save_result(checkpoint_dir: Path, result) -> Path:
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    path = _result_path(checkpoint_dir, result.subject_id)
    tmp_path = path.with_suffix(".pt.tmp")
    torch.save(result, tmp_path)
    tmp_path.replace(path)
    return path


def load_result(checkpoint_dir: Path, subject_id: int):
    path = _result_path(Path(checkpoint_dir), subject_id)
    if not path.exists():
        return None
    return torch.load(path, weights_only=False)


def has_result(checkpoint_dir: Path, subject_id: int) -> bool:
    return _result_path(Path(checkpoint_dir), subject_id).exists()


def load_model(result, device: torch.device | None = None) -> nn.Module:
    """Reconstruct the trained model from a checkpointed SubjectResult, ready for inference."""
    model_class = MODEL_CLASSES[result.architecture]
    model = model_class(n_features=result.n_features)
    model.load_state_dict(result.model_state_dict)
    model.eval()
    if device is not None:
        model = model.to(device)
    return model


def _ga_path(checkpoint_dir: Path, subject_id: int) -> Path:
    return Path(checkpoint_dir) / f"subject_{subject_id}.json"


def save_ga_result(
    checkpoint_dir: Path,
    subject_id: int,
    best_weights: dict[str, float],
    best_fitness: float,
    history: list[float],
    n_evaluations: int,
) -> Path:
    # Cast defensively: callers often hand us a numpy.int64 straight out of
    # df["subject_id"].unique() (e.g. from `for sid in sorted(df[...].unique())`),
    # which json.dumps cannot serialize even though it behaves like a plain int.
    subject_id = int(subject_id)
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    path = _ga_path(checkpoint_dir, subject_id)
    tmp_path = path.with_suffix(".json.tmp")
    payload = {
        "subject_id": subject_id,
        "best_weights": best_weights,
        "best_fitness": best_fitness,
        "history": history,
        "n_evaluations": n_evaluations,
    }
    tmp_path.write_text(json.dumps(payload, indent=2))
    tmp_path.replace(path)
    return path


def load_ga_result(checkpoint_dir: Path, subject_id: int) -> dict | None:
    path = _ga_path(Path(checkpoint_dir), subject_id)
    if not path.exists():
        return None
    return json.loads(path.read_text())


def has_ga_result(checkpoint_dir: Path, subject_id: int) -> bool:
    return _ga_path(Path(checkpoint_dir), subject_id).exists()
