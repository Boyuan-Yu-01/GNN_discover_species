"""File handling and reporting for the three-parameter search."""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from math import exp
from pathlib import Path

import numpy as np

from parameter_finder import Evaluation, ParameterFinder, Parameters


@dataclass(frozen=True)
class Sample:
    """One formula-degeneracy record reduced to the fields used here."""

    degeneracy_id: str
    formula: str
    is_ring: bool
    bond_counts: tuple[int, ...]


def formula_from_id(degeneracy_id: str) -> str:
    """Convert C10H20_002 to C10H20."""
    formula, separator, suffix = degeneracy_id.rpartition("_")
    if not separator or not suffix.isdigit():
        raise ValueError(f"invalid degeneracy ID: {degeneracy_id!r}")
    return formula


def load_samples(path: Path, bond_types: tuple[str, ...]) -> list[Sample]:
    """Load and validate formula-degeneracy records from a filtered JSON file."""
    with path.open(encoding="utf-8") as file_handle:
        data = json.load(file_handle)

    records = data.get("formula_degeneracies")
    if not isinstance(records, dict) or not records:
        raise ValueError(f"{path} has no formula_degeneracies mapping")

    samples: list[Sample] = []
    for degeneracy_id, entry in records.items():
        bond_counts = entry.get("bond_counts")
        if not isinstance(bond_counts, dict):
            raise ValueError(f"{degeneracy_id} has no bond_counts mapping")

        counts: list[int] = []
        for bond in bond_types:
            value = bond_counts.get(bond, 0)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{degeneracy_id} has invalid {bond} count: {value!r}")
            counts.append(value)

        is_ring = entry.get("is_ring", False)
        if not isinstance(is_ring, bool):
            raise ValueError(f"{degeneracy_id} has non-boolean is_ring")

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
    """Build a dense count matrix in the configured bond order."""
    if not samples:
        raise ValueError("cannot build a count matrix from no samples")
    return np.asarray([sample.bond_counts for sample in samples], dtype=float)


def write_parameter_json(
    path: Path,
    best: Evaluation,
    finder: ParameterFinder,
    bond_types: tuple[str, ...],
    bond_energies: np.ndarray,
    positive_count: int,
    pseudo_count: int,
    search_settings: dict,
) -> None:
    """Write the winning parameters and enough context to reproduce them."""
    probabilities = finder.bond_break_probabilities(best.parameters)
    payload = {
        "A": best.parameters.A,
        "E_ref": best.parameters.E_ref,
        "k": best.parameters.k,
        "objective_value": best.total_loss,
        "positive_loss": best.positive_loss,
        "separation_loss": best.separation_loss,
        "regularization_loss": best.regularization_loss,
        "mean_positive_log_score": best.mean_positive_score,
        "pseudo_log_score_statistic": best.pseudo_score_statistic,
        "score_gap": best.score_gap,
        "positive_sample_count": positive_count,
        "pseudo_negative_sample_count": pseudo_count,
        "bond_types": list(bond_types),
        "bde_by_bond": dict(zip(bond_types, map(float, bond_energies))),
        "p_break_by_bond": dict(zip(bond_types, map(float, probabilities))),
        "search_settings": search_settings,
    }
    _write_json(path, payload)


def write_diagnostics_csv(
    path: Path,
    random_results: list[Evaluation],
    refined_results: list[Evaluation],
) -> None:
    """Write every random trial and each final refined candidate."""
    fields = [
        "stage",
        "rank",
        "A",
        "E_ref",
        "k",
        "total_loss",
        "positive_loss",
        "separation_loss",
        "regularization_loss",
        "mean_positive_log_score",
        "pseudo_log_score_statistic",
        "score_gap",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file_handle:
        writer = csv.DictWriter(file_handle, fieldnames=fields)
        writer.writeheader()
        for stage, results in (("random", random_results), ("refined", refined_results)):
            for rank, result in enumerate(results, start=1):
                writer.writerow(_evaluation_row(stage, rank, result))


def write_scored_csv(
    path: Path,
    positive_samples: list[Sample],
    pseudo_samples: list[Sample],
    finder: ParameterFinder,
    parameters: Parameters,
    bond_types: tuple[str, ...],
) -> None:
    """Score every source record with the selected parameter triple."""
    fields = [
        "source",
        "degeneracy_id",
        "formula",
        "is_ring",
        "log_P_exist",
        "P_exist",
        *bond_types,
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file_handle:
        writer = csv.DictWriter(file_handle, fieldnames=fields)
        writer.writeheader()
        for source, samples in (("positive", positive_samples), ("pseudo_negative", pseudo_samples)):
            scores = finder.score_counts(count_matrix(samples), parameters)
            for sample, score in zip(samples, scores):
                probability = exp(max(float(score), -745.0))
                row = {
                    "source": source,
                    "degeneracy_id": sample.degeneracy_id,
                    "formula": sample.formula,
                    "is_ring": sample.is_ring,
                    "log_P_exist": float(score),
                    "P_exist": probability,
                }
                row.update(dict(zip(bond_types, sample.bond_counts)))
                writer.writerow(row)


def write_log(
    path: Path,
    best: Evaluation,
    finder: ParameterFinder,
    bond_types: tuple[str, ...],
    positive_count: int,
    pseudo_count: int,
) -> None:
    """Write a compact human-readable search summary."""
    probabilities = finder.bond_break_probabilities(best.parameters)
    lines = [
        "BOND-BREAKING PARAMETER SEARCH",
        "=" * 38,
        f"Positive groups: {positive_count}",
        f"Pseudo-negative groups: {pseudo_count}",
        "",
        "Best parameters",
        f"  A: {best.parameters.A:.12g}",
        f"  E_ref: {best.parameters.E_ref:.12g}",
        f"  k: {best.parameters.k:.12g}",
        "",
        "Objective",
        f"  total: {best.total_loss:.12g}",
        f"  positive: {best.positive_loss:.12g}",
        f"  separation: {best.separation_loss:.12g}",
        f"  regularization: {best.regularization_loss:.12g}",
        f"  mean positive log score: {best.mean_positive_score:.12g}",
        f"  pseudo log score statistic: {best.pseudo_score_statistic:.12g}",
        f"  score gap: {best.score_gap:.12g}",
        "",
        "Bond-breaking probabilities",
    ]
    lines.extend(
        f"  {bond}: {probability:.12g}"
        for bond, probability in zip(bond_types, probabilities)
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _evaluation_row(stage: str, rank: int, result: Evaluation) -> dict:
    return {
        "stage": stage,
        "rank": rank,
        "A": result.parameters.A,
        "E_ref": result.parameters.E_ref,
        "k": result.parameters.k,
        "total_loss": result.total_loss,
        "positive_loss": result.positive_loss,
        "separation_loss": result.separation_loss,
        "regularization_loss": result.regularization_loss,
        "mean_positive_log_score": result.mean_positive_score,
        "pseudo_log_score_statistic": result.pseudo_score_statistic,
        "score_gap": result.score_gap,
    }


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file_handle:
        json.dump(payload, file_handle, indent=2)
        file_handle.write("\n")
