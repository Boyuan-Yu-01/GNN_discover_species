"""Data loading and reporting for the three-parameter search."""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from parameter_finder import Evaluation, ParameterFinder, Parameters


@dataclass(frozen=True)
class Sample:
    """One formula-degeneracy record reduced to fields used by the search."""

    degeneracy_id: str
    formula: str
    is_ring: bool
    # Counts follow the BOND_TYPES order established in main.py.
    bond_counts: tuple[int, ...]


def formula_from_id(degeneracy_id: str) -> str:
    """Convert an ID such as C10H20_002 into formula C10H20."""
    # Split only at the final underscore because the numeric suffix identifies
    # the degeneracy while the preceding text identifies the molecular formula.
    formula, separator, suffix = degeneracy_id.rpartition("_")

    # Refuse malformed IDs instead of quietly emitting an incorrect formula.
    if not separator or not suffix.isdigit():
        raise ValueError(f"invalid degeneracy ID: {degeneracy_id!r}")
    return formula


def load_samples(path: Path, bond_types: tuple[str, ...]) -> list[Sample]:
    """Load and validate formula-degeneracy records from filtered JSON."""
    # Read-only access protects the source dataset from accidental modification.
    with path.open(encoding="utf-8") as file_handle:
        data = json.load(file_handle)

    # Both filtered datasets use this top-level mapping. Failing here gives a
    # clear error before any numerical search begins.
    records = data.get("formula_degeneracies")
    if not isinstance(records, dict) or not records:
        raise ValueError(f"{path} has no formula_degeneracies mapping")

    samples: list[Sample] = []
    for degeneracy_id, entry in records.items():
        # Only bond_counts, is_ring, and the ID-derived formula are required by
        # this parameter search; unrelated source fields remain untouched.
        bond_counts = entry.get("bond_counts")
        if not isinstance(bond_counts, dict):
            raise ValueError(f"{degeneracy_id} has no bond_counts mapping")

        counts: list[int] = []
        for bond in bond_types:
            # Missing configured bonds mean zero occurrences in the sparse JSON.
            value = bond_counts.get(bond, 0)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{degeneracy_id} has invalid {bond} count: {value!r}")
            counts.append(value)

        is_ring = entry.get("is_ring", False)
        # is_ring is preserved for reporting but is not part of this objective.
        if not isinstance(is_ring, bool):
            raise ValueError(f"{degeneracy_id} has non-boolean is_ring")

        # Convert the sparse source mapping into one immutable, ordered record.
        samples.append(
            Sample(
                degeneracy_id=degeneracy_id,
                formula=formula_from_id(degeneracy_id),
                is_ring=is_ring,
                bond_counts=tuple(counts),
            )
        )

    return samples


def count_matrix(samples: list[Sample]) -> np.ndarray:
    """Build a dense count matrix in configured bond order."""
    if not samples:
        raise ValueError("cannot build a count matrix from no samples")

    # Rows remain individual degeneracies and columns follow BOND_TYPES. No
    # source-level mean is calculated anywhere in this conversion.
    return np.asarray([sample.bond_counts for sample in samples], dtype=float)


def write_parameter_json(
    path: Path,
    best: Evaluation,
    finder: ParameterFinder,
    bond_types: tuple[str, ...],
    bond_energies: np.ndarray,
    positive_count: int,
    pseudo_count: int,
    identifiability_tolerance: float,
    search_settings: dict,
) -> None:
    """Write the winning triple and enough context to reproduce it."""
    # Recalculate bond-level quantities from the final physical parameters so
    # the JSON can be audited without rerunning the optimizer.
    parameters = best.parameters
    arrhenius = finder.arrhenius_terms(parameters)
    probabilities = finder.bond_break_probabilities(parameters)
    spread = finder.probability_spread(parameters)

    # Distribution summaries use only rows that actually constrained the fit.
    # Zero-backbone records remain counted separately below.
    positive_survival = finder.survival_probabilities(
        finder.positive_objective_counts,
        parameters,
    )
    pseudo_survival = finder.survival_probabilities(
        finder.pseudo_objective_counts,
        parameters,
    )

    # Keep important fields at the top level for simple downstream consumption.
    payload = {
        # Selected physical parameter triple.
        "A": parameters.A,
        "E_ref": parameters.E_ref,
        "k": parameters.k,
        # The exact objective decomposition used to rank this candidate.
        "objective_value": best.total_loss,
        "positive_loss": best.positive_loss,
        "pseudo_negative_loss": best.pseudo_loss,
        "regularization_loss": best.regularization_loss,
        # Source-level means are diagnostics, not the fitting objective.
        "mean_positive_log_score": best.mean_positive_score,
        "mean_pseudo_negative_log_score": best.mean_pseudo_score,
        "mean_positive_survival": best.mean_positive_survival,
        "mean_pseudo_negative_survival": best.mean_pseudo_survival,
        "score_gap": best.score_gap,
        "positive_survival_quantiles": _quantiles(positive_survival),
        "pseudo_negative_survival_quantiles": _quantiles(pseudo_survival),
        # Total counts and objective counts make exclusions explicit.
        "positive_sample_count": positive_count,
        "pseudo_negative_sample_count": pseudo_count,
        "positive_objective_count": len(finder.positive_objective_counts),
        "pseudo_negative_objective_count": len(finder.pseudo_objective_counts),
        "ignored_positive_count": finder.ignored_positive_count,
        "ignored_pseudo_negative_count": finder.ignored_pseudo_count,
        # zip relies on the common BOND_TYPES order established in main.py.
        "bond_types": list(bond_types),
        "bde_by_bond": dict(zip(bond_types, map(float, bond_energies))),
        "arrhenius_by_bond": dict(zip(bond_types, map(float, arrhenius))),
        "p_break_by_bond": dict(zip(bond_types, map(float, probabilities))),
        # These checks flag flat bond responses and boundary-driven solutions.
        "probability_spread": spread,
        "weakly_identifiable": spread < identifiability_tolerance,
        "parameter_at_boundary": finder.is_at_boundary(parameters),
        # Store seeds, bounds, weights, and tolerances for reproducibility.
        "search_settings": search_settings,
    }

    # Centralizing JSON formatting keeps output stable between runs.
    _write_json(path, payload)


def write_scored_csv(
    path: Path,
    positive_samples: list[Sample],
    pseudo_samples: list[Sample],
    finder: ParameterFinder,
    parameters: Parameters,
    bond_types: tuple[str, ...],
) -> None:
    """Write every original record's score and individual loss."""
    # The schema contains both model output and the exact features that produced
    # it, allowing any row's score to be checked independently.
    fields = [
        "source",
        "degeneracy_id",
        "formula",
        "is_ring",
        "used_in_objective",
        "log_P_exist",
        "P_exist",
        "individual_loss",
        *bond_types,
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file_handle:
        writer = csv.DictWriter(file_handle, fieldnames=fields)
        writer.writeheader()
        for source, samples in (
            ("positive", positive_samples),
            ("pseudo_negative", pseudo_samples),
        ):
            # Score in one matrix operation, then preserve one output row per sample.
            matrix = count_matrix(samples)
            scores = finder.score_counts(matrix, parameters)

            # Convert non-positive log scores to probabilities. -745 is near the
            # smallest exponent representable by a double-precision float.
            probabilities = np.exp(np.clip(scores, -745.0, 0.0))
            for sample, score, probability in zip(samples, scores, probabilities):
                # Match evaluate(): positives target P=1, while weak fabricated
                # negatives target P=0. The beta weight is applied only when the
                # complete source objective is assembled, not in this raw column.
                loss = (
                    (1.0 - probability) ** 2
                    if source == "positive"
                    else probability**2
                )
                row = {
                    "source": source,
                    "degeneracy_id": sample.degeneracy_id,
                    "formula": sample.formula,
                    "is_ring": sample.is_ring,
                    # All-zero rows are reported but did not influence fitting.
                    "used_in_objective": any(sample.bond_counts),
                    "log_P_exist": float(score),
                    "P_exist": float(probability),
                    "individual_loss": float(loss),
                }
                # Append the ordered bond counts as named CSV columns.
                row.update(dict(zip(bond_types, sample.bond_counts)))
                writer.writerow(row)


def write_log(
    path: Path,
    best: Evaluation,
    finder: ParameterFinder,
    bond_types: tuple[str, ...],
    positive_count: int,
    pseudo_count: int,
    identifiability_tolerance: float,
) -> None:
    """Write a compact human-readable search summary."""
    # Recompute physical diagnostics from the winner instead of depending on
    # intermediate optimizer state.
    parameters = best.parameters
    arrhenius = finder.arrhenius_terms(parameters)
    probabilities = finder.bond_break_probabilities(parameters)
    spread = finder.probability_spread(parameters)
    # Build the complete report in memory, then write it atomically in one call.
    lines = [
        "BOND-BREAKING PARAMETER SEARCH",
        "=" * 38,
        f"Positive groups: {positive_count}",
        f"Pseudo-negative groups: {pseudo_count}",
        f"Positive groups used: {len(finder.positive_objective_counts)}",
        f"Pseudo-negative groups used: {len(finder.pseudo_objective_counts)}",
        f"Ignored zero-backbone positives: {finder.ignored_positive_count}",
        f"Ignored zero-backbone pseudo-negatives: {finder.ignored_pseudo_count}",
        "",
        "Best parameters",
        f"  A: {parameters.A:.12g}",
        f"  E_ref: {parameters.E_ref:.12g} kJ/mol",
        f"  k: {parameters.k:.12g}",
        "",
        "Objective",
        f"  total: {best.total_loss:.12g}",
        f"  positive: {best.positive_loss:.12g}",
        f"  pseudo-negative: {best.pseudo_loss:.12g}",
        f"  regularization: {best.regularization_loss:.12g}",
        f"  mean positive log score: {best.mean_positive_score:.12g}",
        f"  mean pseudo-negative log score: {best.mean_pseudo_score:.12g}",
        f"  mean positive survival: {best.mean_positive_survival:.12g}",
        f"  mean pseudo-negative survival: {best.mean_pseudo_survival:.12g}",
        f"  score gap: {best.score_gap:.12g}",
        "",
        "Checks",
        f"  P_break spread: {spread:.12g}",
        f"  weakly identifiable: {spread < identifiability_tolerance}",
        f"  parameter at boundary: {finder.is_at_boundary(parameters)}",
        "",
        "Bond diagnostics",
    ]

    # One line per configured bond makes monotonicity and saturation visible.
    lines.extend(
        f"  {bond}: Arr={arr:.12g}, P_break={probability:.12g}"
        for bond, arr, probability in zip(bond_types, arrhenius, probabilities)
    )

    path.parent.mkdir(parents=True, exist_ok=True)
    # A final newline keeps the text file friendly to terminals and version control.
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _quantiles(values: np.ndarray) -> dict[str, float]:
    """Return source-distribution summaries that expose overlap and tails."""
    # q05/q95 describe tails; q50 is the median; q25/q75 describe the middle half.
    levels = (0.05, 0.25, 0.50, 0.75, 0.95)
    return {
        f"q{int(level * 100):02d}": float(np.quantile(values, level))
        for level in levels
    }


def _write_json(path: Path, payload: dict) -> None:
    """Write stable, human-readable JSON formatting."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file_handle:
        # indent=2 favors manual review; Python floats remain full precision.
        json.dump(payload, file_handle, indent=2)
        # POSIX text files conventionally end with one newline.
        file_handle.write("\n")
