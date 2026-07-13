"""Find A, E_ref, and k for the bond-breaking survival model."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from parameter_finder import (
    ObjectiveSettings,
    ParameterBounds,
    ParameterFinder,
    Parameters,
)
from parameter_utils import (
    count_matrix,
    load_samples,
    write_diagnostics_csv,
    write_log,
    write_parameter_json,
    write_scored_csv,
)


HERE = Path(__file__).resolve().parent
DATA_DIR = HERE.parent / "org_dataset"
OUTPUT_DIR = HERE / "output"

POSITIVE_FILE = DATA_DIR / "filtered_master.json"
PSEUDO_NEGATIVE_FILE = DATA_DIR / "filtered_pseudo_negative.json"

BOND_TYPES = ("C#C", "C-C", "C-O", "C=C", "C=O")
BOND_ENERGIES = np.asarray((835.0, 346.0, 358.0, 602.0, 732.0))


# =============================================================================
# TUNABLE PARAMETER-FINDING SETTINGS
# Change values in this block to control the objective and search procedure.
# =============================================================================

# Physical search bounds.
PARAMETER_BOUNDS = ParameterBounds(
    A=(1.0, 100.0),
    E_ref=(50.0, 1000.0),
    k=(10.0, 1_000_000.0),
)

# Weak prior used only to keep the search away from extreme solutions.
REFERENCE_PARAMETERS = Parameters(A=10.0, E_ref=300.0, k=1000.0)
REGULARIZATION_WEIGHTS = (1e-3, 1e-3, 1e-3)

# Objective weights. Pseudo-negatives receive less influence, but beta must be
# large enough to overcome the different numerical scales of the two losses.
POSITIVE_WEIGHT = 1.00
PSEUDO_NEGATIVE_WEIGHT = 0.75
SEPARATION_MARGIN = 0.10

# Use None for mean pseudo-negative score. Use 0.90 for the optional tail check.
PSEUDO_SCORE_QUANTILE = None

# Global random search followed by bounded local refinement.
RANDOM_SEED = 20260713
RANDOM_TRIALS = 20_000
KEEP_BEST_RANDOM = 20
LOCAL_STARTS = 10
LOCAL_MAX_ITERATIONS = 1000
LOCAL_TOLERANCE = 1e-9

# Numerical protection for log(1 - P_break).
PROBABILITY_EPSILON = 1e-12

# =============================================================================


def main() -> None:
    """Load data, run the search, and write reproducible outputs."""
    positive_samples = load_samples(POSITIVE_FILE, BOND_TYPES)
    pseudo_samples = load_samples(PSEUDO_NEGATIVE_FILE, BOND_TYPES)

    objective = ObjectiveSettings(
        alpha=POSITIVE_WEIGHT,
        beta=PSEUDO_NEGATIVE_WEIGHT,
        margin=SEPARATION_MARGIN,
        reference=REFERENCE_PARAMETERS,
        regularization=REGULARIZATION_WEIGHTS,
        probability_epsilon=PROBABILITY_EPSILON,
        pseudo_quantile=PSEUDO_SCORE_QUANTILE,
    )
    finder = ParameterFinder(
        positive_counts=count_matrix(positive_samples),
        pseudo_counts=count_matrix(pseudo_samples),
        bond_energies=BOND_ENERGIES,
        bounds=PARAMETER_BOUNDS,
        objective=objective,
    )

    random_results = finder.random_search(RANDOM_TRIALS, RANDOM_SEED)
    starts = random_results[: min(LOCAL_STARTS, KEEP_BEST_RANDOM)]
    refined_results = finder.refine(
        starts,
        maximum_iterations=LOCAL_MAX_ITERATIONS,
        tolerance=LOCAL_TOLERANCE,
    )
    best = min(random_results[0], refined_results[0], key=lambda item: item.total_loss)

    settings = {
        "parameter_bounds": {
            "A": PARAMETER_BOUNDS.A,
            "E_ref": PARAMETER_BOUNDS.E_ref,
            "k": PARAMETER_BOUNDS.k,
        },
        "reference_parameters": {
            "A": REFERENCE_PARAMETERS.A,
            "E_ref": REFERENCE_PARAMETERS.E_ref,
            "k": REFERENCE_PARAMETERS.k,
        },
        "regularization_weights": REGULARIZATION_WEIGHTS,
        "positive_weight": POSITIVE_WEIGHT,
        "pseudo_negative_weight": PSEUDO_NEGATIVE_WEIGHT,
        "separation_margin": SEPARATION_MARGIN,
        "pseudo_score_quantile": PSEUDO_SCORE_QUANTILE,
        "random_seed": RANDOM_SEED,
        "random_trials": RANDOM_TRIALS,
        "keep_best_random": KEEP_BEST_RANDOM,
        "local_starts": LOCAL_STARTS,
        "local_max_iterations": LOCAL_MAX_ITERATIONS,
        "local_tolerance": LOCAL_TOLERANCE,
    }

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    write_parameter_json(
        OUTPUT_DIR / "tuned_bond_breaking_parameters.json",
        best,
        finder,
        BOND_TYPES,
        BOND_ENERGIES,
        len(positive_samples),
        len(pseudo_samples),
        settings,
    )
    write_diagnostics_csv(
        OUTPUT_DIR / "tuning_diagnostics.csv",
        random_results,
        refined_results,
    )
    write_scored_csv(
        OUTPUT_DIR / "scored_formula_degeneracies.csv",
        positive_samples,
        pseudo_samples,
        finder,
        best.parameters,
        BOND_TYPES,
    )
    write_log(
        OUTPUT_DIR / "log_tuned_bond_breaking_parameters.txt",
        best,
        finder,
        BOND_TYPES,
        len(positive_samples),
        len(pseudo_samples),
    )

    print("Best bond-breaking parameters")
    print(f"  A     = {best.parameters.A:.12g}")
    print(f"  E_ref = {best.parameters.E_ref:.12g}")
    print(f"  k     = {best.parameters.k:.12g}")
    print(f"  loss  = {best.total_loss:.12g}")
    print(f"Output: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
