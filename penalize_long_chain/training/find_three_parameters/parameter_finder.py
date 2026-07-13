"""Numerical search for the three bond-breaking parameters."""

from __future__ import annotations

from dataclasses import dataclass
from math import exp, log

import numpy as np
from scipy.optimize import minimize


@dataclass(frozen=True)
class Parameters:
    """Physical parameters used by the bond-breaking model."""

    A: float
    E_ref: float
    k: float


@dataclass(frozen=True)
class ParameterBounds:
    """Closed search bounds in physical parameter space.
    
    Defines min/max ranges for each of the three bond-breaking parameters.
    Used to constrain the optimization search space.
    """

    # Lower and upper bounds for the pre-exponential factor A
    A: tuple[float, float]
    
    # Lower and upper bounds for the reference energy parameter E_ref
    E_ref: tuple[float, float]
    
    # Lower and upper bounds for the dimensionless parameter k
    k: tuple[float, float]

    def as_log_bounds(self) -> tuple[tuple[float, float], ...]:
        """Convert bounds from linear space to log space.
        
        Returns the natural logarithm of each bound, which is useful for
        log-uniform sampling and optimization in log-parameter space.
        This ensures parameters are searched uniformly on a logarithmic scale.
        
        Returns:
            Tuple of three (log_low, log_high) tuples in the same order as as_tuple()
        """
        return tuple((log(low), log(high)) for low, high in self.as_tuple())

    def as_tuple(self) -> tuple[tuple[float, float], ...]:
        """Return all three bounds as a single tuple.
        
        Provides a convenient way to iterate over bounds in order: A, E_ref, k.
        Used by as_log_bounds() and validation routines.
        
        Returns:
            Tuple of (A_bounds, E_ref_bounds, k_bounds)
        """
        return self.A, self.E_ref, self.k


@dataclass(frozen=True)
class ObjectiveSettings:
    """Fixed choices that define the parameter-search objective."""

    alpha: float
    beta: float
    margin: float   # fix-score margin for positive vs pseudo-negative separation
    reference: Parameters
    regularization: tuple[float, float, float]
    probability_epsilon: float = 1e-12
    pseudo_quantile: float | None = None


@dataclass(frozen=True)
class Evaluation:
    """One parameter triple and all objective components."""

    parameters: Parameters
    log_parameters: tuple[float, float, float]
    total_loss: float   # total loss = alpha * positive_loss + beta * separation_loss + regularization_loss
    positive_loss: float
    separation_loss: float
    regularization_loss: float
    mean_positive_score: float
    pseudo_score_statistic: float
    score_gap: float


class ParameterFinder:
    """Search A, E_ref, and k without pairing individual samples.

    The workflow is:
    1) validate data and configuration,
    2) evaluate loss for candidate parameters,
    3) run random search,
    4) locally refine the best candidates.
    """

    def __init__(
        self,
        positive_counts: np.ndarray,
        pseudo_counts: np.ndarray,
        bond_energies: np.ndarray,
        bounds: ParameterBounds,
        objective: ObjectiveSettings,
    ) -> None:
        # Convert to well-formed 2D float arrays and validate basic constraints.
        self.positive_counts = self._validate_counts(positive_counts, "positive")
        self.pseudo_counts = self._validate_counts(pseudo_counts, "pseudo-negative")
        self.bond_energies = np.asarray(bond_energies, dtype=float)
        self.bounds = bounds
        self.objective = objective

        if self.bond_energies.shape != (self.positive_counts.shape[1],):
            raise ValueError("bond-energy count does not match bond-count columns")
        if np.any(self.bond_energies <= 0.0):
            raise ValueError("bond energies must be positive")

        self._validate_settings()
        # Cache dataset means once; evaluate() uses them repeatedly.
        self._positive_mean_counts = self.positive_counts.mean(axis=0)
        self._pseudo_mean_counts = self.pseudo_counts.mean(axis=0)

    @staticmethod
    def _validate_counts(counts: np.ndarray, source: str) -> np.ndarray:
        # Counts are expected as [sample, bond_type] matrices.
        array = np.asarray(counts, dtype=float)
        if array.ndim != 2 or not array.shape[0] or not array.shape[1]:
            raise ValueError(f"{source} bond counts must be a non-empty matrix")
        if not np.all(np.isfinite(array)) or np.any(array < 0.0):
            raise ValueError(f"{source} bond counts must be finite and non-negative")
        return array

    def _validate_settings(self) -> None:
        # Log-parameter optimization requires strictly positive bounds.
        for name, (low, high) in zip(("A", "E_ref", "k"), self.bounds.as_tuple()):
            if low <= 0.0 or high <= low:
                raise ValueError(f"invalid bounds for {name}: {(low, high)}")

        settings = self.objective
        if settings.alpha <= 0.0 or settings.beta < 0.0 or settings.margin < 0.0:
            raise ValueError("alpha must be positive; beta and margin must be non-negative")
        if len(settings.regularization) != 3 or any(x < 0.0 for x in settings.regularization):
            raise ValueError("regularization must contain three non-negative values")
        if not 0.0 < settings.probability_epsilon < 0.5:
            raise ValueError("probability_epsilon must lie between 0 and 0.5")
        if settings.pseudo_quantile is not None and not 0.0 < settings.pseudo_quantile < 1.0:
            raise ValueError("pseudo_quantile must lie between 0 and 1")

    @staticmethod
    def decode(log_parameters: np.ndarray | tuple[float, ...]) -> Parameters:
        """Convert unconstrained log parameters to positive physical values."""
        # Exponentiation maps real-valued search variables to positive parameters.
        values = tuple(exp(float(value)) for value in log_parameters)
        return Parameters(*values)

    def bond_break_probabilities(self, parameters: Parameters) -> np.ndarray:
        """Return one breaking probability per configured backbone bond."""
        # Arrhenius-like term: larger bond energy lowers effective break tendency.
        arrhenius = parameters.A * np.exp(-self.bond_energies / parameters.E_ref)
        probabilities = 1.0 / (1.0 + parameters.k * np.exp(-arrhenius))
        epsilon = self.objective.probability_epsilon
        # Clip away from exactly 0/1 to keep logs finite and numerically stable.
        return np.clip(probabilities, epsilon, 1.0 - epsilon)

    def bond_log_survival(self, parameters: Parameters) -> np.ndarray:
        """Return log(1 - P_break) for each backbone bond."""
        return np.log1p(-self.bond_break_probabilities(parameters))

    def score_counts(self, counts: np.ndarray, parameters: Parameters) -> np.ndarray:
        """Return log-survival scores for a matrix of bond counts."""
        return np.asarray(counts, dtype=float) @ self.bond_log_survival(parameters)

    def evaluate(self, log_parameters: np.ndarray | tuple[float, ...]) -> Evaluation:
        """Evaluate the non-paired distribution objective."""
        log_values = tuple(float(value) for value in log_parameters)
        parameters = self.decode(log_values)
        bond_scores = self.bond_log_survival(parameters)

        # Positive score summary uses the dataset mean for efficiency.
        mean_positive = float(self._positive_mean_counts @ bond_scores)
        # Pseudo-negative summary is either mean or quantile, per configuration.
        if self.objective.pseudo_quantile is None:
            pseudo_statistic = float(self._pseudo_mean_counts @ bond_scores)
        else:
            pseudo_scores = self.pseudo_counts @ bond_scores
            pseudo_statistic = float(
                np.quantile(pseudo_scores, self.objective.pseudo_quantile)
            )

        # Margin-based separation: no penalty when gap already exceeds margin.
        gap = mean_positive - pseudo_statistic
        positive_loss = -mean_positive
        separation_loss = max(0.0, self.objective.margin - gap)
        regularization_loss = self._regularization(log_values)
        # Weighted objective optimized by random search and Powell refinement.
        total_loss = (
            self.objective.alpha * positive_loss
            + self.objective.beta * separation_loss
            + regularization_loss
        )

        return Evaluation(
            parameters=parameters,
            log_parameters=log_values,
            total_loss=total_loss,
            positive_loss=positive_loss,
            separation_loss=separation_loss,
            regularization_loss=regularization_loss,
            mean_positive_score=mean_positive,
            pseudo_score_statistic=pseudo_statistic,
            score_gap=gap,
        )

    def _regularization(self, log_parameters: tuple[float, ...]) -> float:
        # L2 penalty in log space around a reference parameter triple.
        reference = self.objective.reference
        log_reference = (log(reference.A), log(reference.E_ref), log(reference.k))
        return sum(
            weight * (value - center) ** 2
            for weight, value, center in zip(
                self.objective.regularization, log_parameters, log_reference
            )
        )

    def random_search(
        self,
        trial_count: int,
        seed: int,
    ) -> list[Evaluation]:
        """Evaluate log-uniform random parameter triples."""
        if trial_count <= 0:
            raise ValueError("trial_count must be positive")

        rng = np.random.default_rng(seed)
        log_bounds = np.asarray(self.bounds.as_log_bounds(), dtype=float)
        # Uniform sampling in log space corresponds to log-uniform in linear space.
        samples = rng.uniform(log_bounds[:, 0], log_bounds[:, 1], size=(trial_count, 3))
        evaluations = [self.evaluate(sample) for sample in samples]
        # Best candidates first (lowest total_loss).
        evaluations.sort(key=lambda item: item.total_loss)
        return evaluations

    def refine(
        self,
        starts: list[Evaluation],
        maximum_iterations: int,
        tolerance: float,
    ) -> list[Evaluation]:
        """Refine the best random candidates with bounded Powell search."""
        if not starts:
            raise ValueError("at least one refinement start is required")

        refined: list[Evaluation] = []
        bounds = self.bounds.as_log_bounds()

        for start in starts:
            # Local derivative-free optimization from each random-search seed.
            result = minimize(
                lambda values: self.evaluate(values).total_loss,
                np.asarray(start.log_parameters),
                method="Powell",
                bounds=bounds,
                options={"maxiter": maximum_iterations, "xtol": tolerance, "ftol": tolerance},
            )
            candidate = self.evaluate(result.x)
            # Keep whichever is better: original start or refined candidate.
            refined.append(min(start, candidate, key=lambda item: item.total_loss))

        refined.sort(key=lambda item: item.total_loss)
        return refined

