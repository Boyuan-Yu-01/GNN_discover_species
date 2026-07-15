"""Pointwise numerical search for A, E_ref, and k."""

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
    """Closed bounds in physical parameter space."""

    A: tuple[float, float]
    E_ref: tuple[float, float]
    k: tuple[float, float]

    def as_tuple(self) -> tuple[tuple[float, float], ...]:
        """Return bounds in optimizer order: A, E_ref, k."""
        return self.A, self.E_ref, self.k

    def as_log_bounds(self) -> tuple[tuple[float, float], ...]:
        """Convert positive physical bounds to natural-log space."""
        # Searching logs guarantees that decoded physical parameters stay positive.
        return tuple((log(low), log(high)) for low, high in self.as_tuple())


@dataclass(frozen=True)
class ObjectiveSettings:
    """Weights, reference candidate, and numerical safeguards."""

    # alpha weights reliable observations; beta weights fabricated negatives.
    alpha: float
    beta: float
    reference: Parameters
    # Regularization order is A, E_ref, k. Set the E_ref weight to zero when its
    # reference value should be an initialization rather than a physical prior.
    regularization: tuple[float, float, float]
    probability_epsilon: float = 1e-12


@dataclass(frozen=True)
class Evaluation:
    """One parameter candidate and its complete objective diagnostics."""

    parameters: Parameters
    # Optimizer order is log(A), log(E_ref), log(k).
    log_parameters: tuple[float, float, float]
    total_loss: float
    positive_loss: float
    pseudo_loss: float
    regularization_loss: float
    mean_positive_score: float
    mean_pseudo_score: float
    mean_positive_survival: float
    mean_pseudo_survival: float
    score_gap: float


@dataclass(frozen=True)
class RefinementTrace:
    """One Powell run, including completed Powell iteration endpoints."""

    start: Evaluation
    result: Evaluation
    best: Evaluation
    # Every entry is a completed Powell outer-iteration endpoint, expressed in
    # log(A), log(E_ref), log(k) so either optimizer mode can share one plot.
    log_path: tuple[tuple[float, float, float], ...]
    optimizer_iterations: int
    optimizer_function_evaluations: int
    optimizer_success: bool
    optimizer_message: str


class ParameterFinder:
    """Optimize A, E_ref, and k using every usable degeneracy independently."""

    def __init__(
        self,
        positive_counts: np.ndarray,
        pseudo_counts: np.ndarray,
        bond_energies: np.ndarray,
        bounds: ParameterBounds,
        objective: ObjectiveSettings,
    ) -> None:
        # Retain complete matrices for reporting, but fit only rows containing
        # at least one bond whose response can change with the parameters.
        self.positive_counts = self._validate_counts(positive_counts, "positive")
        self.pseudo_counts = self._validate_counts(pseudo_counts, "pseudo-negative")
        self.positive_objective_counts = self._rows_with_backbone(self.positive_counts)
        self.pseudo_objective_counts = self._rows_with_backbone(self.pseudo_counts)
        self.ignored_positive_count = len(self.positive_counts) - len(
            self.positive_objective_counts
        )
        self.ignored_pseudo_count = len(self.pseudo_counts) - len(
            self.pseudo_objective_counts
        )

        # Bond-energy order must match the count-matrix column order from main.py.
        self.bond_energies = np.asarray(bond_energies, dtype=float)
        self.bounds = bounds
        self.objective = objective

        if self.bond_energies.shape != (self.positive_counts.shape[1],):
            raise ValueError("bond-energy count does not match bond-count columns")
        if np.any(~np.isfinite(self.bond_energies)) or np.any(self.bond_energies <= 0.0):
            raise ValueError("bond energies must be finite and positive")
        if self.pseudo_counts.shape[1] != self.positive_counts.shape[1]:
            raise ValueError("positive and pseudo-negative matrices need equal columns")

        self._validate_settings()

    @staticmethod
    def _validate_counts(counts: np.ndarray, source: str) -> np.ndarray:
        """Return a finite, non-negative, two-dimensional count matrix."""
        array = np.asarray(counts, dtype=float)
        if array.ndim != 2 or not array.shape[0] or not array.shape[1]:
            raise ValueError(f"{source} bond counts must be a non-empty matrix")
        if np.any(~np.isfinite(array)) or np.any(array < 0.0):
            raise ValueError(f"{source} bond counts must be finite and non-negative")
        return array

    @staticmethod
    def _rows_with_backbone(counts: np.ndarray) -> np.ndarray:
        """Return rows containing at least one configured backbone bond."""
        # All-zero rows always have P_exist=1 and provide no parameter gradient.
        usable = counts[np.any(counts > 0.0, axis=1)]
        if not len(usable):
            raise ValueError("the objective has no rows with configured backbone bonds")
        return usable

    def _validate_settings(self) -> None:
        """Reject invalid bounds and objective controls early."""
        for name, (low, high) in zip(("A", "E_ref", "k"), self.bounds.as_tuple()):
            if not np.isfinite(low) or not np.isfinite(high) or low <= 0.0 or high <= low:
                raise ValueError(f"invalid bounds for {name}: {(low, high)}")

        settings = self.objective
        if settings.alpha <= 0.0 or settings.beta < 0.0:
            raise ValueError("alpha must be positive and beta must be non-negative")
        if any(
            value <= 0.0
            for value in (
                settings.reference.A,
                settings.reference.E_ref,
                settings.reference.k,
            )
        ):
            raise ValueError("reference parameters must be positive")

        # The explicitly inserted initial candidate must be a legal candidate.
        for name, value, (low, high) in zip(
            ("A", "E_ref", "k"),
            (
                settings.reference.A,
                settings.reference.E_ref,
                settings.reference.k,
            ),
            self.bounds.as_tuple(),
        ):
            if not low <= value <= high:
                raise ValueError(f"reference {name} lies outside its bounds")
        if len(settings.regularization) != 3 or any(
            weight < 0.0 for weight in settings.regularization
        ):
            raise ValueError("regularization must contain three non-negative weights")
        if not 0.0 < settings.probability_epsilon < 0.5:
            raise ValueError("probability_epsilon must lie between 0 and 0.5")

    @staticmethod
    def decode(log_parameters: np.ndarray | tuple[float, ...]) -> Parameters:
        """Convert logged optimizer values into positive physical parameters."""
        if len(log_parameters) != 3:
            raise ValueError("expected log(A), log(E_ref), and log(k)")
        values = tuple(exp(float(value)) for value in log_parameters)
        return Parameters(*values)

    @staticmethod
    def encode(parameters: Parameters) -> tuple[float, float, float]:
        """Convert physical parameters into optimizer coordinates."""
        return log(parameters.A), log(parameters.E_ref), log(parameters.k)

    def arrhenius_terms(self, parameters: Parameters) -> np.ndarray:
        """Return A exp(-E_b/E_ref), evaluated safely in log space."""
        # Subtract exponents before exponentiating to avoid premature underflow.
        log_terms = log(parameters.A) - self.bond_energies / parameters.E_ref
        return np.exp(np.clip(log_terms, -745.0, 700.0))

    def bond_break_probabilities(self, parameters: Parameters) -> np.ndarray:
        """Return one breaking probability per configured backbone bond."""
        arrhenius = self.arrhenius_terms(parameters)
        probabilities = 1.0 / (1.0 + parameters.k * np.exp(-arrhenius))

        # Exact zero or one would make a later log(1-P_break) invalid.
        epsilon = self.objective.probability_epsilon
        return np.clip(probabilities, epsilon, 1.0 - epsilon)

    def bond_log_survival(self, parameters: Parameters) -> np.ndarray:
        """Return log(1-P_break) for each configured bond."""
        return np.log1p(-self.bond_break_probabilities(parameters))

    def score_counts(self, counts: np.ndarray, parameters: Parameters) -> np.ndarray:
        """Return one independent log-survival score per input row."""
        matrix = np.asarray(counts, dtype=float)
        if matrix.ndim != 2 or matrix.shape[1] != len(self.bond_energies):
            raise ValueError("bond-count matrix has the wrong shape")

        # Matrix multiplication implements sum_b N[g,b] log(1-P_break,b).
        return matrix @ self.bond_log_survival(parameters)

    def survival_probabilities(
        self,
        counts: np.ndarray,
        parameters: Parameters,
    ) -> np.ndarray:
        """Return stable existence probabilities for a count matrix."""
        scores = self.score_counts(counts, parameters)
        return np.exp(np.clip(scores, -745.0, 0.0))

    def evaluate(self, log_parameters: np.ndarray | tuple[float, ...]) -> Evaluation:
        """Apply nonlinear losses to individual rows before averaging."""
        log_values = tuple(float(value) for value in log_parameters)
        parameters = self.decode(log_values)

        positive_scores = self.score_counts(self.positive_objective_counts, parameters)
        pseudo_scores = self.score_counts(self.pseudo_objective_counts, parameters)
        positive_survival = np.exp(np.clip(positive_scores, -745.0, 0.0))
        pseudo_survival = np.exp(np.clip(pseudo_scores, -745.0, 0.0))

        # Each row receives its own Brier-style loss before source normalization.
        positive_loss = float(np.mean(np.square(1.0 - positive_survival)))
        pseudo_loss = float(np.mean(np.square(pseudo_survival)))
        regularization_loss = self._regularization(log_values)
        total_loss = (
            self.objective.alpha * positive_loss
            + self.objective.beta * pseudo_loss
            + regularization_loss
        )

        # These means are diagnostics only; they do not define the objective.
        mean_positive_score = float(np.mean(positive_scores))
        mean_pseudo_score = float(np.mean(pseudo_scores))
        return Evaluation(
            parameters=parameters,
            log_parameters=log_values,
            total_loss=total_loss,
            positive_loss=positive_loss,
            pseudo_loss=pseudo_loss,
            regularization_loss=regularization_loss,
            mean_positive_score=mean_positive_score,
            mean_pseudo_score=mean_pseudo_score,
            mean_positive_survival=float(np.mean(positive_survival)),
            mean_pseudo_survival=float(np.mean(pseudo_survival)),
            score_gap=mean_positive_score - mean_pseudo_score,
        )

    def _regularization(self, log_parameters: tuple[float, float, float]) -> float:
        """Apply configured relative penalties around the reference candidate."""
        centers = self.encode(self.objective.reference)
        return sum(
            weight * (value - center) ** 2
            for weight, value, center in zip(
                self.objective.regularization,
                log_parameters,
                centers,
            )
        )

    def probability_spread(self, parameters: Parameters) -> float:
        """Return the range of configured bond-breaking probabilities."""
        # Near-zero spread means bond identity has effectively disappeared.
        return float(np.ptp(self.bond_break_probabilities(parameters)))

    def is_at_boundary(self, parameters: Parameters, fraction: float = 1e-4) -> bool:
        """Return whether a parameter lies near a log-space search boundary."""
        values = self.encode(parameters)
        for value, (low, high) in zip(values, self.bounds.as_log_bounds()):
            if min(value - low, high - value) <= fraction * (high - low):
                return True
        return False

    def random_search(self, trial_count: int, seed: int) -> list[Evaluation]:
        """Evaluate the exact reference plus full-range log-uniform candidates."""
        # One trial means one complete (A, E_ref, k) triple. Rejecting zero here
        # also avoids trying to rank an empty result list in main.py.
        if trial_count <= 0:
            raise ValueError("trial_count must be positive")

        # Convert physical bounds once:
        #   A      [1, 100]       -> log(A)
        #   E_ref  [50, 1000]     -> log(E_ref)
        #   k      [10, 1e6]      -> log(k)
        # Random sampling is performed in these logged units.
        bounds = np.asarray(self.bounds.as_log_bounds(), dtype=float)

        # A fixed seed makes the exact same candidate population reproducible.
        # Changing the seed changes the sampled triples, not the objective.
        rng = np.random.default_rng(seed)

        # Uniform sampling between logged bounds produces log-uniform physical
        # values after decode(). This gives each order of magnitude meaningful
        # coverage instead of concentrating candidates near the largest values.
        # Reserve one trial because the exact reference is inserted below.
        samples = rng.uniform(
            bounds[:, 0],
            bounds[:, 1],
            size=(max(0, trial_count - 1), 3),
        )

        # Put the explicit (10, 300, 1000) reference first. It is evaluated by
        # exactly the same code as every random candidate, so its loss can be
        # compared directly. vstack also handles trial_count == 1 correctly.
        samples = np.vstack((self.encode(self.objective.reference), samples))

        # evaluate() decodes one row, scores every usable degeneracy, applies
        # all individual losses, and returns the total plus diagnostics.
        evaluations = [self.evaluate(sample) for sample in samples]

        # Ascending order makes evaluations[0] the best global-search result and
        # lets callers take evaluations[:N] as local-optimization starting points.
        evaluations.sort(key=lambda item: item.total_loss)
        return evaluations

    def refine_log(
        self,
        starts: list[Evaluation],
        maximum_iterations: int,
        tolerance: float,
    ) -> list[RefinementTrace]:
        """Run Powell in log(A), log(E_ref), and log(k) coordinates."""
        return self._refine(starts, maximum_iterations, tolerance, use_log_space=True)

    def refine(
        self,
        starts: list[Evaluation],
        maximum_iterations: int,
        tolerance: float,
    ) -> list[RefinementTrace]:
        """Run Powell directly in physical A, E_ref, and k coordinates."""
        return self._refine(starts, maximum_iterations, tolerance, use_log_space=False)

    def _refine(
        self,
        starts: list[Evaluation],
        maximum_iterations: int,
        tolerance: float,
        use_log_space: bool,
    ) -> list[RefinementTrace]:
        """Implement the shared bounded Powell loop for both coordinate systems."""
        # Each entry in starts is already a fully evaluated random-search result.
        if not starts:
            raise ValueError("at least one refinement start is required")

        traces: list[RefinementTrace] = []

        # Bounds and start points must be in the same coordinate system as the
        # optimizer. evaluate() always receives log coordinates internally.
        bounds = self.bounds.as_log_bounds() if use_log_space else self.bounds.as_tuple()
        for start in starts:
            start_values = (
                start.log_parameters
                if use_log_space
                else (start.parameters.A, start.parameters.E_ref, start.parameters.k)
            )
            # The callback fires after a completed Powell outer iteration. Keep
            # this lighter path for the animation rather than every line-search
            # probe, which made the earlier video visually noisy.
            path = [start.log_parameters]

            def to_log_parameters(values: np.ndarray) -> tuple[float, float, float]:
                """Convert an optimizer point into evaluate()'s log coordinates."""
                if use_log_space:
                    return tuple(float(value) for value in values)
                return self.encode(Parameters(*(float(value) for value in values)))

            def record_iteration(values: np.ndarray) -> None:
                """Record one outer-iteration endpoint in shared log coordinates."""
                point = to_log_parameters(values)
                if point != path[-1]:
                    path.append(point)

            def objective_function(values: np.ndarray) -> float:
                """Evaluate a candidate in either optimizer coordinate system."""
                return self.evaluate(to_log_parameters(values)).total_loss

            # Refine each promising basin independently. Multiple starts reduce
            # the chance that one poor local basin determines the final answer.
            #
            # Powell is derivative-free: it probes directions using objective
            # values and therefore needs no analytic gradient through clipping,
            # exponentials, or the pointwise probability calculation.
            result = minimize(
                # The wrapper converts physical coordinates when refine() was
                # selected; evaluate() always returns the scalar total loss.
                objective_function,
                np.asarray(start_values),
                method="Powell",
                bounds=bounds,
                options={
                    # maxiter caps work; xtol controls parameter convergence;
                    # ftol controls objective-value convergence.
                    "maxiter": maximum_iterations,
                    "xtol": tolerance,
                    "ftol": tolerance,
                },
                # One callback point per completed Powell outer iteration.
                callback=record_iteration,
            )

            # Re-evaluate result.x through our own path so the returned object
            # contains physical parameters and every diagnostic loss component.
            candidate = self.evaluate(to_log_parameters(result.x))

            # Append the final optimizer point if the final callback omitted it.
            final_point = candidate.log_parameters
            if final_point != path[-1]:
                path.append(final_point)

            # Powell may stop without improving, especially near a bound or flat
            # region. Never replace a valid start with a worse local result.
            best = min(start, candidate, key=lambda item: item.total_loss)
            traces.append(
                RefinementTrace(
                    start=start,
                    result=candidate,
                    best=best,
                    log_path=tuple(path),
                    optimizer_iterations=int(result.nit),
                    optimizer_function_evaluations=int(result.nfev),
                    optimizer_success=bool(result.success),
                    optimizer_message=str(result.message),
                )
            )

        # As in random_search(), index zero contains the best retained result.
        # The trace stays attached to its start even after sorting.
        traces.sort(key=lambda item: item.best.total_loss)
        return traces
