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
    write_score_plot,
)


HERE = Path(__file__).resolve().parent
DATA_DIR = HERE.parent / "org_dataset"
# OUTPUT_DIR = HERE / "output"
OUTPUT_DIR = HERE / "output_all_bonds"

# The search reads but never modifies these two source datasets.
POSITIVE_FILE = DATA_DIR / "filtered_master.json"
PSEUDO_NEGATIVE_FILE = DATA_DIR / "filtered_pseudo_negative.json"
# PSEUDO_NEGATIVE_FILE = DATA_DIR / "filtered_pseudo_negative_remove_OOO.json"

# Count-matrix columns and BDE values must remain in exactly the same order.
# BOND_TYPES = ("C#C", "C-C", "C-O", "C=C", "C=O", "O-O", "O=O", "C#O")
# BOND_ENERGIES = np.asarray((835.0, 346.0, 358.0, 602.0, 732.0, 146.0, 498.0, 1072.0))  # kJ/mol

BOND_TYPES = ("C#C", "C-C", "C-O", "C=C", "C=O", "O-O", "O=O", "C#O", "C-H", "O-H", "H-H")
BOND_ENERGIES = np.asarray((835.0, 346.0, 358.0, 602.0, 732.0, 146.0, 498.0, 1072.0, 413.0, 463.0, 436.0))  # kJ/mol


# =============================================================================
# TUNABLE PARAMETER-FINDING SETTINGS
# Edit this block to control the objective and numerical search.
# =============================================================================

# Broad physical ranges restored after rejecting the much smaller RT values.
PARAMETER_BOUNDS = ParameterBounds(
    A=(1.0, 100.0),
    E_ref=(50.0, 10000.0),  # kJ/mol, matching BOND_ENERGIES
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
PSEUDO_NEGATIVE_WEIGHT = 0.3

# Full-range log-uniform search followed by bounded Powell refinement.
RANDOM_SEED = 20260713                      # Controls reproducible random-number generation
RANDOM_TRIALS = 20_000                      # Total candidates evaluated globally
KEEP_BEST_RANDOM = 30                       # Maximum random candidates eligible for refinement
LOCAL_STARTS = 30                           # Number of candidates actually refined
LOCAL_MAX_ITERATIONS = 10000                 # Maximum Powell iterations per start
LOCAL_TOLERANCE = 1e-15                      # Powell convergence precision
# Choose "log" for multiplicative parameter steps or "physical" for direct
# A, E_ref, k steps. This selects refine_log() or refine(), respectively.
REFINEMENT_SPACE = "log"

# Number of distinct low-loss parameter triples written after one search.
NUMBER_OF_PARAMETER_SETS = 1

# Numerical and identifiability checks.
PROBABILITY_EPSILON = 1e-12
IDENTIFIABILITY_TOLERANCE = 1e-6

# Staged MP4: bounds -> random candidates -> retained starts -> Powell paths.
GENERATE_SEARCH_ANIMATION = True  # Set False to skip animation rendering.
ANIMATION_FILENAME = "parameter_search_animation.mp4"
ANIMATION_SETTINGS = AnimationSettings(
    fps=10,
    dpi=100,
    space_frames=12,
    population_frames=18,
    selection_frames=18,
    trajectory_frames=60,
    hold_frames=12,
)

# =============================================================================


def make_finder(
    positive_counts: np.ndarray,
    pseudo_counts: np.ndarray,
    positive_sample_ids: tuple[str, ...] | None = None,
    pseudo_sample_ids: tuple[str, ...] | None = None,
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
        positive_sample_ids=positive_sample_ids,
        pseudo_sample_ids=pseudo_sample_ids,
        positive_source=str(POSITIVE_FILE),
        pseudo_source=str(PSEUDO_NEGATIVE_FILE),
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
    if REFINEMENT_SPACE == "log":
        refinement = finder.refine_log
    elif REFINEMENT_SPACE == "physical":
        refinement = finder.refine
    else:
        raise ValueError('REFINEMENT_SPACE must be "log" or "physical"')

    refinement_traces = refinement(
        starts,
        maximum_iterations=LOCAL_MAX_ITERATIONS,
        tolerance=LOCAL_TOLERANCE,
    )

    # Each trace contains the start, final Powell point, retained best result,
    # and completed Powell iteration endpoints for the 3D animation.
    refined_results = [trace.best for trace in refinement_traces]

    # Both lists are sorted, so index zero is the strongest result from its
    # stage. Compare them explicitly: local refinement is useful, not trusted
    # blindly, and cannot erase a better global-search candidate.
    best = min(random_results[0], refined_results[0], key=lambda item: item.total_loss)
    return random_results, refinement_traces, best


def select_parameter_sets(
    finder: ParameterFinder,
    random_results: list[Evaluation],
    refinement_traces: list[RefinementTrace],
) -> list[Evaluation]:
    """Return the requested number of distinct, identifiable low-loss triples."""
    if NUMBER_OF_PARAMETER_SETS <= 0:
        raise ValueError("NUMBER_OF_PARAMETER_SETS must be positive")

    # Local results are included alongside every global candidate. Sorting the
    # combined list selects by objective value, rather than by start-point order.
    candidates = [trace.best for trace in refinement_traces] + random_results
    candidates.sort(key=lambda item: item.total_loss)

    selected: list[Evaluation] = []
    seen: set[tuple[float, float, float]] = set()
    for candidate in candidates:
        # Log coordinates make a scale-independent identity key. Rounding
        # prevents numerically indistinguishable Powell endpoints from filling
        # several output folders.
        key = tuple(round(value, 12) for value in candidate.log_parameters)
        if key in seen:
            continue
        seen.add(key)
        if finder.probability_spread(candidate.parameters) < IDENTIFIABILITY_TOLERANCE:
            continue
        selected.append(candidate)
        if len(selected) == NUMBER_OF_PARAMETER_SETS:
            return selected

    raise RuntimeError(
        "fewer identifiable distinct candidates than NUMBER_OF_PARAMETER_SETS"
    )


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
        "refinement_space": REFINEMENT_SPACE,
        "number_of_parameter_sets": NUMBER_OF_PARAMETER_SETS,
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
            "trajectory_frames": ANIMATION_SETTINGS.trajectory_frames,
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
        tuple(sample.degeneracy_id for sample in positive_samples),
        tuple(sample.degeneracy_id for sample in pseudo_samples),
    )

    random_results, refinement_traces, _ = run_search(finder)
    selected_sets = select_parameter_sets(finder, random_results, refinement_traces)
    best = selected_sets[0]

    # Each selected candidate receives an independent, self-contained report.
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for rank, candidate in enumerate(selected_sets, start=1):
        set_directory = OUTPUT_DIR / f"parameter_set_{rank:02d}"
        settings = search_settings()
        settings["selected_set_rank"] = rank
        write_parameter_json(
            set_directory / "tuned_bond_breaking_parameters.json",
            candidate,
            finder,
            BOND_TYPES,
            BOND_ENERGIES,
            len(positive_samples),
            len(pseudo_samples),
            IDENTIFIABILITY_TOLERANCE,
            settings,
        )
        write_scored_csv(
            set_directory / "scored_formula_degeneracies.csv",
            positive_samples,
            pseudo_samples,
            finder,
            candidate.parameters,
            BOND_TYPES,
        )
        write_log(
            set_directory / "log_tuned_bond_breaking_parameters.txt",
            candidate,
            finder,
            BOND_TYPES,
            len(positive_samples),
            len(pseudo_samples),
            IDENTIFIABILITY_TOLERANCE,
        )
        write_score_plot(
            set_directory / "positive_pseudo_negative_scores.png",
            positive_samples,
            pseudo_samples,
            finder,
            candidate.parameters,
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

    print("Best selected bond-breaking parameters")
    print(f"  A     = {best.parameters.A:.12g}")
    print(f"  E_ref = {best.parameters.E_ref:.12g} kJ/mol")
    print(f"  k     = {best.parameters.k:.12g}")
    print(f"  loss  = {best.total_loss:.12g}")
    print(f"Parameter sets: {len(selected_sets)}")
    print(f"Output: {OUTPUT_DIR}")
    if GENERATE_SEARCH_ANIMATION:
        print(f"Animation: {OUTPUT_DIR / ANIMATION_FILENAME}")


if __name__ == "__main__":
    main()
