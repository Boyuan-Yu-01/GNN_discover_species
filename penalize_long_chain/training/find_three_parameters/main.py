"""Find A, E_ref, and k with one broad pointwise numerical search."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from parameter_finder import (
    Evaluation,
    ObjectiveSettings,
    ParameterBounds,
    ParameterFinder,
    Parameters,
    RefinementTrace,
)
from parameter_animation import AnimationSettings, write_search_animation
from parameter_utils import (
    count_matrix,
    load_samples,
    write_log,
    write_parameter_json,
    write_scored_csv,
)


HERE = Path(__file__).resolve().parent
DATA_DIR = HERE.parent / "org_dataset"
OUTPUT_DIR = HERE / "output"

# The search reads but never modifies these two source datasets.
POSITIVE_FILE = DATA_DIR / "filtered_master.json"
PSEUDO_NEGATIVE_FILE = DATA_DIR / "filtered_pseudo_negative.json"

# Count-matrix columns and BDE values must remain in exactly the same order.
BOND_TYPES = ("C#C", "C-C", "C-O", "C=C", "C=O")
BOND_ENERGIES = np.asarray((835.0, 346.0, 358.0, 602.0, 732.0))


# =============================================================================
# TUNABLE PARAMETER-FINDING SETTINGS
# Edit this block to control the objective and numerical search.
# =============================================================================

# Broad physical ranges restored after rejecting the much smaller RT values.
PARAMETER_BOUNDS = ParameterBounds(
    A=(1.0, 100.0),
    E_ref=(50.0, 1000.0),  # kJ/mol, matching BOND_ENERGIES
    k=(10.0, 1_000_000.0),
)

# The exact reference candidate is included in the random-search population.
# E_ref=300 kJ/mol is near the earlier exploratory result around 310 kJ/mol.
REFERENCE_PARAMETERS = Parameters(A=10.0, E_ref=300.0, k=1000.0)

# We accepted that E_ref should not be pulled toward its initial guess. The
# middle zero preserves that choice while weakly regularizing A and k.
REGULARIZATION_WEIGHTS = (1e-3, 1e-3, 1e-3)  # log(A), log(E_ref), log(k)

# Observed positives dominate; fabricated pseudo-negatives are weak evidence.
POSITIVE_WEIGHT = 1.00
PSEUDO_NEGATIVE_WEIGHT = 0.50

# Full-range log-uniform search followed by bounded Powell refinement.
RANDOM_SEED = 20260713                      # Controls reproducible random-number generation
RANDOM_TRIALS = 20_000                      # Total candidates evaluated globally
KEEP_BEST_RANDOM = 50                       # Maximum random candidates eligible for refinement
LOCAL_STARTS = 50                           # Number of candidates actually refined
LOCAL_MAX_ITERATIONS = 1000                 # Maximum Powell iterations per start
LOCAL_TOLERANCE = 1e-12                      # Powell convergence precision

# Numerical and identifiability checks.
PROBABILITY_EPSILON = 1e-12
IDENTIFIABILITY_TOLERANCE = 1e-6

# Staged MP4: bounds -> random candidates -> retained starts -> Powell probes.
GENERATE_SEARCH_ANIMATION = True  # Set False to skip animation rendering.
ANIMATION_FILENAME = "parameter_search_animation.mp4"
ANIMATION_SETTINGS = AnimationSettings(
    fps=10,
    dpi=1000,
    space_frames=12,
    population_frames=18,
    selection_frames=18,
    # One frame per requested objective evaluation: show every Powell probe.
    evaluations_per_frame=1,
    hold_frames=12,
)

# =============================================================================


def make_finder(
    positive_counts: np.ndarray,
    pseudo_counts: np.ndarray,
) -> ParameterFinder:
    """Construct the shared physical model and pointwise objective."""
    objective = ObjectiveSettings(
        alpha=POSITIVE_WEIGHT,
        beta=PSEUDO_NEGATIVE_WEIGHT,
        reference=REFERENCE_PARAMETERS,
        regularization=REGULARIZATION_WEIGHTS,
        probability_epsilon=PROBABILITY_EPSILON,
    )
    return ParameterFinder(
        positive_counts=positive_counts,
        pseudo_counts=pseudo_counts,
        bond_energies=BOND_ENERGIES,
        bounds=PARAMETER_BOUNDS,
        objective=objective,
    )


def run_search(
    finder: ParameterFinder,
) -> tuple[list[Evaluation], list[RefinementTrace], Evaluation]:
    """Run the global candidate stage and local refinement stage."""
    # Stage 1 explores the complete allowed parameter volume. The exact
    # (10, 300, 1000) reference is inserted by random_search(); the remaining
    # RANDOM_TRIALS - 1 triples cover all three physical bounds log-uniformly.
    # The returned list is already sorted from lowest to highest total loss.
    random_results = finder.random_search(RANDOM_TRIALS, RANDOM_SEED)

    # Stage 2 begins from several strong global candidates. LOCAL_STARTS controls
    # expensive Powell calls; KEEP_BEST_RANDOM limits how much of the ranked
    # global population is eligible. min() keeps inconsistent settings safe.
    starts = random_results[: min(LOCAL_STARTS, KEEP_BEST_RANDOM)]
    refinement_traces = finder.refine(
        starts,
        maximum_iterations=LOCAL_MAX_ITERATIONS,
        tolerance=LOCAL_TOLERANCE,
    )

    # Each trace contains the start, final Powell point, retained best result,
    # every objective probe, and outer-iteration endpoints for diagnostics.
    refined_results = [trace.best for trace in refinement_traces]

    # Both lists are sorted, so index zero is the strongest result from its
    # stage. Compare them explicitly: local refinement is useful, not trusted
    # blindly, and cannot erase a better global-search candidate.
    best = min(random_results[0], refined_results[0], key=lambda item: item.total_loss)
    return random_results, refinement_traces, best


def search_settings() -> dict:
    """Return every control needed to reproduce the search."""
    return {
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
        "regularization_weights": {
            "log_A": REGULARIZATION_WEIGHTS[0],
            "log_E_ref": REGULARIZATION_WEIGHTS[1],
            "log_k": REGULARIZATION_WEIGHTS[2],
        },
        "positive_weight": POSITIVE_WEIGHT,
        "pseudo_negative_weight": PSEUDO_NEGATIVE_WEIGHT,
        "individual_loss": "squared_probability_error",
        "random_seed": RANDOM_SEED,
        "random_trials": RANDOM_TRIALS,
        "keep_best_random": KEEP_BEST_RANDOM,
        "local_starts": LOCAL_STARTS,
        "local_max_iterations": LOCAL_MAX_ITERATIONS,
        "local_tolerance": LOCAL_TOLERANCE,
        "probability_epsilon": PROBABILITY_EPSILON,
        "identifiability_tolerance": IDENTIFIABILITY_TOLERANCE,
        "animation": {
            "enabled": GENERATE_SEARCH_ANIMATION,
            "filename": ANIMATION_FILENAME,
            "fps": ANIMATION_SETTINGS.fps,
            "dpi": ANIMATION_SETTINGS.dpi,
            "space_frames": ANIMATION_SETTINGS.space_frames,
            "population_frames": ANIMATION_SETTINGS.population_frames,
            "selection_frames": ANIMATION_SETTINGS.selection_frames,
            "evaluations_per_frame": ANIMATION_SETTINGS.evaluations_per_frame,
            "hold_frames": ANIMATION_SETTINGS.hold_frames,
        },
    }


def main() -> None:
    """Load data, search the three parameters, and write reproducible output."""
    positive_samples = load_samples(POSITIVE_FILE, BOND_TYPES)
    pseudo_samples = load_samples(PSEUDO_NEGATIVE_FILE, BOND_TYPES)
    finder = make_finder(
        count_matrix(positive_samples),
        count_matrix(pseudo_samples),
    )

    random_results, refinement_traces, best = run_search(finder)
    probability_spread = finder.probability_spread(best.parameters)
    if probability_spread < IDENTIFIABILITY_TOLERANCE:
        raise RuntimeError(
            "best parameters give indistinguishable bond probabilities; "
            "inspect the model"
        )

    # Files are created only after the complete search and identifiability check.
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    write_parameter_json(
        OUTPUT_DIR / "tuned_bond_breaking_parameters.json",
        best,
        finder,
        BOND_TYPES,
        BOND_ENERGIES,
        len(positive_samples),
        len(pseudo_samples),
        IDENTIFIABILITY_TOLERANCE,
        search_settings(),
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
        IDENTIFIABILITY_TOLERANCE,
    )

    # Render only after the numerical result passes identifiability checks. The
    # animation uses the same random candidates and Powell paths as the outputs.
    if GENERATE_SEARCH_ANIMATION:
        write_search_animation(
            OUTPUT_DIR / ANIMATION_FILENAME,
            random_results,
            refinement_traces,
            best,
            PARAMETER_BOUNDS,
            ANIMATION_SETTINGS,
        )

    print("Best bond-breaking parameters")
    print(f"  A     = {best.parameters.A:.12g}")
    print(f"  E_ref = {best.parameters.E_ref:.12g} kJ/mol")
    print(f"  k     = {best.parameters.k:.12g}")
    print(f"  loss  = {best.total_loss:.12g}")
    print(f"Output: {OUTPUT_DIR}")
    if GENERATE_SEARCH_ANIMATION:
        print(f"Animation: {OUTPUT_DIR / ANIMATION_FILENAME}")


if __name__ == "__main__":
    main()
