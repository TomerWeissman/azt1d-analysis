"""
Scoped replication of the paper's Algorithm 1: a genetic algorithm that
searches per-patient (w_hypo, w_hyper) weight pairs for the region-weighted
loss, using validation RMSE as fitness -- matching the paper's own stated
fitness definition ("the fitness of each candidate weight pair is defined as
the validation RMSE of the forecasting model").

Scoped down from the paper's own budget (population 20, 25 generations) on
purpose: that literal recipe evaluates every individual in the population every
generation (20 x 25 = 500 full model trainings per patient by the pseudocode as
written), which by our own measured per-training time is somewhere around 18
hours of compute for CNN-LSTM alone across all 25 patients. This version:

  1. Only evaluates *new* individuals each generation and reuses each
     survivor's already-known fitness, rather than re-training every survivor
     again -- a standard, low-risk GA optimization that doesn't change the
     search dynamics (same selection/crossover/mutation), just avoids redoing
     work whose answer hasn't changed.
  2. Uses a smaller population and generation count (defaults: 6 and 6,
     instead of 20 and 25).
  3. Caps epochs per candidate evaluation lower than a full final training run
     (default 8 epochs, patience 3) -- meant to rank candidates relative to
     each other, not fully converge each one. The paper doesn't specify a
     lighter regime for GA fitness checks vs. final training; this is our own
     budget cut.

The selection/crossover/mutation mechanics themselves (keep the fitter half,
average two random survivors, add Gaussian noise, clip to the weight range)
follow Algorithm 1 exactly.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .train import PreparedSubjectData, train_prepared_model

DEFAULT_POPULATION_SIZE = 6
DEFAULT_GENERATIONS = 6
DEFAULT_WEIGHT_RANGE = (1.0, 10.0)  # matches the paper's w_hypo, w_hyper ~ U(1, 10)
DEFAULT_MUTATION_STD = 0.5  # matches the paper's m ~ N(0, 0.5)
DEFAULT_CANDIDATE_EPOCHS = 8
DEFAULT_CANDIDATE_PATIENCE = 3


@dataclass
class GAResult:
    subject_id: int
    best_weights: dict[str, float]  # w_hypo, w_normal, w_hyper
    best_fitness: float  # validation RMSE, mg/dL -- lower is better
    history: list[float] = field(default_factory=list)  # best-so-far fitness per generation
    n_evaluations: int = 0


def _evaluate(
    data: PreparedSubjectData,
    w_hypo: float,
    w_hyper: float,
    architecture: str,
    candidate_epochs: int,
    candidate_patience: int,
    seed: int,
) -> float:
    weights = {"w_hypo": float(w_hypo), "w_normal": 1.0, "w_hyper": float(w_hyper)}
    result = train_prepared_model(
        data,
        architecture=architecture,
        epochs=candidate_epochs,
        patience=candidate_patience,
        region_weights=weights,
        seed=seed,
    )
    return result.val_rmse


def search_patient_weights(
    data: PreparedSubjectData,
    architecture: str = "cnn_lstm",
    population_size: int = DEFAULT_POPULATION_SIZE,
    generations: int = DEFAULT_GENERATIONS,
    weight_range: tuple[float, float] = DEFAULT_WEIGHT_RANGE,
    mutation_std: float = DEFAULT_MUTATION_STD,
    candidate_epochs: int = DEFAULT_CANDIDATE_EPOCHS,
    candidate_patience: int = DEFAULT_CANDIDATE_PATIENCE,
    seed: int = 0,
) -> GAResult:
    """
    Algorithm 1 for one patient. Expected number of _evaluate calls:
    population_size + (generations - 1) * (population_size - population_size // 2).
    With the defaults (6, 6): 6 + 5*3 = 21 evaluations.
    """
    rng = np.random.default_rng(seed)
    lo, hi = weight_range
    n_survivors = population_size // 2
    n_offspring = population_size - n_survivors

    population = rng.uniform(lo, hi, size=(population_size, 2))  # columns: w_hypo, w_hyper
    fitness = None
    history: list[float] = []
    best_weights: np.ndarray | None = None
    best_fitness = float("inf")
    n_evaluations = 0

    for gen in range(generations):
        # Only score individuals we don't already have a fitness for (all of
        # them in generation 0, just the new offspring afterwards).
        to_score = population if fitness is None else population[n_survivors:]
        new_fitness = np.array([
            _evaluate(data, w_hypo, w_hyper, architecture, candidate_epochs, candidate_patience,
                      seed=seed + gen * 1000 + i)
            for i, (w_hypo, w_hyper) in enumerate(to_score)
        ])
        n_evaluations += len(to_score)

        fitness = new_fitness if fitness is None else np.concatenate([fitness[:n_survivors], new_fitness])

        order = np.argsort(fitness)
        population = population[order]
        fitness = fitness[order]

        if fitness[0] < best_fitness:
            best_fitness = float(fitness[0])
            best_weights = population[0].copy()
        history.append(best_fitness)

        survivors = population[:n_survivors]
        offspring = []
        while len(offspring) < n_offspring:
            p1, p2 = survivors[rng.integers(0, n_survivors)], survivors[rng.integers(0, n_survivors)]
            child = 0.5 * (p1 + p2) + rng.normal(0, mutation_std, size=2)
            offspring.append(np.clip(child, lo, hi))

        population = np.concatenate([survivors, np.array(offspring)])
        # fitness currently holds n_survivors known values; next loop iteration
        # only scores the n_offspring new ones appended above.
        fitness = fitness[:n_survivors]

    return GAResult(
        subject_id=data.subject_id,
        best_weights={"w_hypo": float(best_weights[0]), "w_normal": 1.0, "w_hyper": float(best_weights[1])},
        best_fitness=best_fitness,
        history=history,
        n_evaluations=n_evaluations,
    )
