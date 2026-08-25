"""Train a policy-value neural network from terminal-reward MCTS data.

This file is intentionally standalone.  It does not import or modify
``toy_example.py`` because that file runs its demonstration at import time.

Training proceeds in two phases:

1. Warm start: use uniform PUCT priors, ignore the untrained value head, and
   evaluate every newly expanded leaf with a rollout to an exact terminal
   reward.
2. Neural-guided search: use the learned policy as the PUCT prior and the
   learned value at most new leaves.  A configurable fraction of leaves still
   use exact terminal rollouts to keep the training data grounded.

Each completed episode produces examples (state, MCTS visit policy, terminal
return).  The policy head learns the visit policy and the value head learns the
terminal return.
"""

from __future__ import annotations

import argparse
import math
import random
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import torch
from torch import Tensor, nn
from torch.nn import functional as F


STOP = "STOP"
ALL_ACTIONS = ("form_C-O", "form_O-H", "form_C-H", STOP)
ACTION_TO_INDEX = {action: index for index, action in enumerate(ALL_ACTIONS)}


@dataclass(frozen=True)
class ToyState:
    """Presence of the three candidate bonds plus an explicit STOP flag."""

    co: int = 0
    oh: int = 0
    ch: int = 0
    stopped: bool = False


def format_state(state: ToyState) -> str:
    """Return a compact chemical label for diagnostic output."""
    if state.co and state.oh:
        label = "C-O-H"
    elif state.co:
        label = "C-O,H"
    elif state.oh:
        label = "C,O-H"
    elif state.ch:
        label = "C-H,O"
    else:
        label = "C,O,H"
    if state.stopped:
        label += " [STOP]"
    return label


class ToyChemistry:
    """Small deterministic graph-edit environment with terminal rewards."""

    def is_terminal(self, state: ToyState) -> bool:
        methanol = state.co == 1 and state.oh == 1
        methane_route = state.ch == 1
        return state.stopped or methanol or methane_route

    def terminal_value(self, state: ToyState) -> float:
        if not self.is_terminal(state):
            raise ValueError("terminal_value() requires a terminal state")
        if state.co == 1 and state.oh == 1:
            return 1.0
        if state.oh == 1:
            return 0.6
        if state.ch == 1:
            return 0.6
        if state.co == 1:
            return 0.7
        return 0.0

    def valid_actions(self, state: ToyState) -> list[str]:
        if self.is_terminal(state):
            return []

        c_free = 1 - state.co - state.ch
        o_free = 2 - state.co - state.oh
        h_free = 1 - state.oh - state.ch

        actions: list[str] = []
        if not state.co and c_free > 0 and o_free > 0:
            actions.append("form_C-O")
        if not state.oh and o_free > 0 and h_free > 0:
            actions.append("form_O-H")
        if not state.ch and c_free > 0 and h_free > 0:
            actions.append("form_C-H")
        actions.append(STOP)
        return actions

    def next_state(self, state: ToyState, action: str) -> ToyState:
        if action not in self.valid_actions(state):
            raise ValueError(f"Illegal action {action!r} at state {state}")
        if action == "form_C-O":
            return ToyState(co=1, oh=state.oh, ch=state.ch)
        if action == "form_O-H":
            return ToyState(co=state.co, oh=1, ch=state.ch)
        if action == "form_C-H":
            return ToyState(co=state.co, oh=state.oh, ch=1)
        return ToyState(
            co=state.co,
            oh=state.oh,
            ch=state.ch,
            stopped=True,
        )


class StateEncoder:
    """Convert a ToyState into fixed-size features for the toy MLP."""

    feature_names = (
        "C-O present",
        "O-H present",
        "C-H present",
        "stopped",
        "C free capacity",
        "O free capacity / 2",
        "H free capacity",
    )

    @classmethod
    def encode(cls, state: ToyState) -> np.ndarray:
        """Encode a ToyState into a fixed-size numpy array."""
        c_free = 1 - state.co - state.ch
        o_free = 2 - state.co - state.oh
        h_free = 1 - state.oh - state.ch
        return np.asarray(
            [
                state.co,
                state.oh,
                state.ch,
                float(state.stopped),
                c_free,
                o_free / 2.0,
                h_free,
            ],
            dtype=np.float32,
        )

    @classmethod
    def valid_action_mask(
        cls,
        environment: ToyChemistry,
        state: ToyState,
    ) -> np.ndarray:
        """Return a boolean mask of valid actions for a given state."""
        mask = np.zeros(len(ALL_ACTIONS), dtype=np.bool_)
        for action in environment.valid_actions(state):
            mask[ACTION_TO_INDEX[action]] = True
        return mask


class PolicyValueNetwork(nn.Module):
    """Shared MLP trunk with categorical-policy and scalar-value heads."""

    def __init__(
        self,
        input_size: int,
        hidden_size: int = 16,
        hidden_layers: int = 3,
    ):
        super().__init__()
        trunk_layers: list[nn.Module] = []
        layer_input_size = input_size
        for _ in range(hidden_layers):
            trunk_layers.extend(
                [
                    nn.Linear(layer_input_size, hidden_size),
                    nn.ReLU(),
                ]
            )
            layer_input_size = hidden_size
        self.trunk = nn.Sequential(*trunk_layers)
        self.policy_head = nn.Linear(hidden_size, len(ALL_ACTIONS))
        self.value_head = nn.Sequential(
            nn.Linear(hidden_size, 1),
            nn.Sigmoid(),
        )

    def forward(self, features: Tensor) -> tuple[Tensor, Tensor]:
        hidden = self.trunk(features)
        policy_logits = self.policy_head(hidden)
        value = self.value_head(hidden).squeeze(-1)
        return policy_logits, value


@dataclass(frozen=True)
class TrainingExample:
    """One supervised target generated by a completed search episode."""

    features: np.ndarray
    valid_action_mask: np.ndarray
    policy_target: np.ndarray   # calculated by MCTS visit counts
    value_target: float         # value_target = (gamma**steps_to_terminal) * terminal_reward


@dataclass(frozen=True)
class EpisodeDecision:
    """One real decision made after an MCTS search inside an episode.

    ``policy`` is the MCTS visit-count policy pi at ``state``. ``action`` is
    the real environment action sampled from that policy. This is different
    from the many hypothetical actions explored during MCTS simulations.
    """

    state: ToyState
    policy: np.ndarray
    action: str


class ReplayBuffer:     # Stores training examples produced by completed MCTS episodes so they can be reused during training.
    def __init__(self, capacity: int):
        self._examples: deque[TrainingExample] = deque(maxlen=capacity)
        # Store TrainingExample instances
        # preserve insertion order
        # keep at maxlen=capacity to limit memory usage
        # automatically remove the oldest item when full

    def extend(self, examples: Iterable[TrainingExample]) -> None:
        self._examples.extend(examples)

    def sample(
        self,
        batch_size: int,
        rng: random.Random,
    ) -> list[TrainingExample]:
        population = list(self._examples)
        if batch_size >= len(population):
            return population
        return rng.sample(population, batch_size)

    def __len__(self) -> int:
        return len(self._examples)


@dataclass
class EdgeStats:
    prior: float
    visits: int = 0
    value_sum: float = 0.0

    @property
    def q(self) -> float:
        return self.value_sum / self.visits if self.visits else 0.0


@dataclass
class SearchNode:   # at each expanded state, store the edges to child states and the model value for that state
    edges: dict[str, EdgeStats] = field(default_factory=dict)
    model_value: float = 0.0


@dataclass(frozen=True)
class SimulationTrace:
    root_state: ToyState
    leaf_state: ToyState
    evaluated_state: ToyState   # state that was evaluated by either rollout or neural network
    tree_actions: tuple[str, ...] # action selected at each step of the tree search from root to leaf
    rollout_actions: tuple[str, ...] # additional actions used by the random rollout to reach a terminal state, if any
    value: float        # reward or predicted value assigned to the simulation
    evaluation_source: str
    # source of the evaluation (e.g., "rollout" or "neural_network")


def masked_policy_value(
    network: PolicyValueNetwork,
    state: ToyState,
    environment: ToyChemistry,
    device: torch.device,
) -> tuple[dict[str, float], float]:
    """Run inference and normalize policy mass over legal actions only."""

    valid_actions = environment.valid_actions(state)
    if not valid_actions:
        raise ValueError("The network should not evaluate terminal states")

    # fixed-length numeric description of the board/state
    features = torch.as_tensor(
        StateEncoder.encode(state),
        dtype=torch.float32,
        device=device,
    ).unsqueeze(0)

    # valid-action boolean mask for the current state
    mask = torch.as_tensor(
        StateEncoder.valid_action_mask(environment, state),
        dtype=torch.bool,
        device=device,
    ).unsqueeze(0)

    network.eval()
    with torch.no_grad():
        logits, value = network(features)   # network outputs raw scores for every action (logits) & a scalar estimate of the state value (value)
        masked_logits = logits.masked_fill(~mask, torch.finfo(logits.dtype).min)    # mask invalid actions by setting their logits to a very low value, so they have near-zero probability after softmax
        probabilities = torch.softmax(masked_logits, dim=-1).squeeze(0).cpu()       # convert logits to probabilities using softmax, and move to CPU for further processing

    # Code keeps only the legal actions and ignores the invalid action.
    priors = {
        action: float(probabilities[ACTION_TO_INDEX[action]])
        for action in valid_actions
    }
    return priors, float(value.item())


class NeuralMCTS:
    """PUCT search supporting terminal rollouts and neural leaf evaluation."""

    def __init__(
        self,
        environment: ToyChemistry,
        network: PolicyValueNetwork,
        device: torch.device,
        rng: np.random.Generator,
        *,
        c_puct: float = 1.5,
        use_network_value: bool = True,
        uniform_priors: bool = False,
        terminal_rollout_probability: float = 0.25,
    ):
        self.environment = environment
        self.network = network
        self.device = device
        self.rng = rng
        self.c_puct = c_puct
        self.use_network_value = use_network_value
        self.uniform_priors = uniform_priors
        self.terminal_rollout_probability = terminal_rollout_probability
        self.nodes: dict[ToyState, SearchNode] = {}

    def expand(self, state: ToyState) -> SearchNode:
        # Expand a nonterminal state by creating a SearchNode with edges for each valid action and assigning priors and model value.
        valid_actions = self.environment.valid_actions(state)
        if not valid_actions:
            raise ValueError("Cannot expand a terminal state")

        if self.uniform_priors and not self.use_network_value:
            # Assign uniform priors to all valid actions.
            probability = 1.0 / len(valid_actions)
            priors = {action: probability for action in valid_actions}
            model_value = 0.0
        else:
            priors, model_value = masked_policy_value(
                self.network,
                state,
                self.environment,
                self.device,
            )
            if self.uniform_priors:
                probability = 1.0 / len(valid_actions)
                priors = {action: probability for action in valid_actions}

        node = SearchNode(
            edges={
                action: EdgeStats(prior=priors[action])
                for action in valid_actions
            },
            model_value=model_value,
        )
        self.nodes[state] = node
        return node

    def select_action(self, node: SearchNode) -> str:
        total_visits = sum(edge.visits for edge in node.edges.values())
        parent_scale = math.sqrt(total_visits + 1)

        def puct_score(action: str) -> float:
            edge = node.edges[action]
            exploration = (
                self.c_puct
                * edge.prior
                * parent_scale
                / (1 + edge.visits)
            )
            return edge.q + exploration

        return max(node.edges, key=puct_score)

    def rollout_to_terminal(
        self,
        state: ToyState,
    ) -> tuple[float, tuple[str, ...], ToyState]:
        """Use a uniform legal-action rollout to obtain an exact reward."""

        actions: list[str] = []
        rollout_state = state
        while not self.environment.is_terminal(rollout_state):
            valid_actions = self.environment.valid_actions(rollout_state)
            action_index = int(self.rng.integers(len(valid_actions)))
            action = valid_actions[action_index]
            actions.append(action)
            rollout_state = self.environment.next_state(rollout_state, action)

        return (
            self.environment.terminal_value(rollout_state),
            tuple(actions),
            rollout_state,
        )

    def simulate(self, root_state: ToyState) -> SimulationTrace:
        state = root_state
        path: list[EdgeStats] = []
        tree_actions: list[str] = []
        rollout_actions: tuple[str, ...] = ()

        while True:
            # If the current state is terminal, we can directly get its value and break the loop.
            if self.environment.is_terminal(state):
                value = self.environment.terminal_value(state)
                evaluation_source = "terminal"
                evaluated_state = state
                break
            # If the current state is not in the search tree, we expand it and decide whether to use a rollout or the neural network for evaluation.
            if state not in self.nodes:
                node = self.expand(state)
                use_rollout = (
                    not self.use_network_value
                    or self.rng.random() < self.terminal_rollout_probability
                )
                if use_rollout:
                    value, rollout_actions, evaluated_state = (
                        self.rollout_to_terminal(state)
                    )
                    evaluation_source = "rollout"
                else:
                    value = node.model_value
                    evaluation_source = "network"
                    evaluated_state = state
                break

            node = self.nodes[state]
            action = self.select_action(node)
            edge = node.edges[action]
            path.append(edge)
            tree_actions.append(action)
            state = self.environment.next_state(state, action)

        # This is a single-agent problem, so backup does not alternate signs.
        for edge in reversed(path):
            edge.visits += 1
            edge.value_sum += value

        return SimulationTrace(
            root_state=root_state,
            leaf_state=state,
            evaluated_state=evaluated_state,
            tree_actions=tuple(tree_actions),
            rollout_actions=rollout_actions,
            value=value,
            evaluation_source=evaluation_source,
        )

    def add_root_noise(
        self,
        root_state: ToyState,
        alpha: float,
        epsilon: float,
    ) -> None:
        # Without noise, the root policy can become overly deterministic, which can lead to overfitting and poor exploration. Adding Dirichlet noise to the root node's priors encourages exploration of different actions during MCTS.
        node = self.nodes[root_state]
        noise = self.rng.dirichlet([alpha] * len(node.edges))
        for edge, noise_value in zip(node.edges.values(), noise):
            edge.prior = (1.0 - epsilon) * edge.prior + epsilon * float(
                noise_value
            )

    def run(
        self,
        root_state: ToyState,
        simulations: int,
        *,
        add_root_noise: bool = False,
        dirichlet_alpha: float = 0.3,
        dirichlet_epsilon: float = 0.25,
    ) -> list[SimulationTrace]:
        if self.environment.is_terminal(root_state):
            raise ValueError("MCTS root must be nonterminal")
        if root_state not in self.nodes:
            self.expand(root_state)
        if add_root_noise:
            self.add_root_noise(
                root_state,
                alpha=dirichlet_alpha,
                epsilon=dirichlet_epsilon,
            )
        traces = []
        for _ in range(simulations):
            traces.append(self.simulate(root_state))
        return traces

    def root_policy(
        self,
        root_state: ToyState,
        temperature: float = 1.0,
    ) -> np.ndarray:
        node = self.nodes[root_state]
        counts = np.zeros(len(ALL_ACTIONS), dtype=np.float64)
        for action, edge in node.edges.items():
            counts[ACTION_TO_INDEX[action]] = edge.visits

        if temperature <= 1.0e-8:
            policy = np.zeros_like(counts)
            policy[int(np.argmax(counts))] = 1.0
            return policy.astype(np.float32)

        adjusted_counts = counts ** (1.0 / temperature)
        normalizer = float(adjusted_counts.sum())
        if normalizer == 0.0:
            for action, edge in node.edges.items():
                adjusted_counts[ACTION_TO_INDEX[action]] = edge.prior
            normalizer = float(adjusted_counts.sum())
        return (adjusted_counts / normalizer).astype(np.float32)

    def state_statistics(
        self,
        state: ToyState,
    ) -> list[dict[str, float | int | str]]:
        """Return local action statistics for any expanded state."""
        node = self.nodes[state]
        total_visits = sum(edge.visits for edge in node.edges.values())
        return [
            {
                "action": action,
                "prior": edge.prior,
                "visits": edge.visits,
                "Q": edge.q,
                "pi": edge.visits / total_visits if total_visits else 0.0,
            }
            for action, edge in node.edges.items()
        ]

    def root_statistics(
        self,
        root_state: ToyState,
    ) -> list[dict[str, float | int | str]]:
        """Backward-compatible wrapper for root statistics."""
        return self.state_statistics(root_state)


def sample_action(
    policy: np.ndarray,
    valid_actions: Sequence[str],
    rng: np.random.Generator,
) -> str:
    valid_indices = np.asarray(
        [ACTION_TO_INDEX[action] for action in valid_actions],
        dtype=np.int64,
    )
    valid_probabilities = policy[valid_indices].astype(np.float64)
    normalizer = float(valid_probabilities.sum())
    if normalizer == 0.0:
        valid_probabilities.fill(1.0 / len(valid_probabilities))
    else:
        valid_probabilities /= normalizer
    selected = int(rng.choice(valid_indices, p=valid_probabilities))
    return ALL_ACTIONS[selected]


def generate_episode(
    environment: ToyChemistry,
    network: PolicyValueNetwork,
    device: torch.device,
    rng: np.random.Generator,
    *,
    simulations: int,
    c_puct: float,
    temperature: float,
    gamma: float,
    neural_guidance: bool,
    terminal_rollout_probability: float,
    dirichlet_alpha: float,
    dirichlet_epsilon: float,
) -> tuple[
    list[TrainingExample],
    float,
    tuple[str, ...],
    list[SimulationTrace],
    list[EpisodeDecision],
    ToyState,
]:
    state = ToyState()
    decisions: list[EpisodeDecision] = []
    executed_actions: list[str] = []
    episode_traces: list[SimulationTrace] = []

    while not environment.is_terminal(state):
        search = NeuralMCTS(
            environment,
            network,
            device,
            rng,
            c_puct=c_puct,
            use_network_value=neural_guidance,
            uniform_priors=not neural_guidance,
            terminal_rollout_probability=(
                terminal_rollout_probability if neural_guidance else 1.0
            ),
        )
        episode_traces.extend(
            search.run(
                state,
                simulations,
                add_root_noise=True,
                dirichlet_alpha=dirichlet_alpha,
                dirichlet_epsilon=dirichlet_epsilon,
            )
        )
        policy = search.root_policy(state, temperature=temperature)
        action = sample_action(
            policy,
            environment.valid_actions(state),
            rng,
        )
        decisions.append(
            EpisodeDecision(
                state=state,
                policy=policy.copy(),
                action=action,
            )
        )
        executed_actions.append(action)
        state = environment.next_state(state, action)

    terminal_reward = environment.terminal_value(state)
    horizon = len(decisions)
    examples: list[TrainingExample] = []
    for time_index, decision in enumerate(decisions):
        steps_to_terminal = horizon - 1 - time_index
        value_target = (gamma**steps_to_terminal) * terminal_reward
        examples.append(
            TrainingExample(
                features=StateEncoder.encode(decision.state),
                valid_action_mask=StateEncoder.valid_action_mask(
                    environment,
                    decision.state,
                ),
                policy_target=decision.policy,
                value_target=value_target,
            )
        )

    return (
        examples,
        terminal_reward,
        tuple(executed_actions),
        episode_traces,
        decisions,
        state,
    )


def train_network(
    network: PolicyValueNetwork,
    replay_buffer: ReplayBuffer,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    rng: random.Random,
    *,
    gradient_steps: int,
    batch_size: int,
    value_loss_weight: float,
    entropy_weight: float,
) -> dict[str, float]:
    if not replay_buffer:
        raise ValueError("Cannot train with an empty replay buffer")

    totals = {"loss": 0.0, "policy": 0.0, "value": 0.0, "entropy": 0.0}
    network.train()

    for _ in range(gradient_steps):
        batch = replay_buffer.sample(batch_size, rng)
        features = torch.as_tensor(
            np.stack([example.features for example in batch]),
            dtype=torch.float32,
            device=device,
        )
        masks = torch.as_tensor(
            np.stack([example.valid_action_mask for example in batch]),
            dtype=torch.bool,
            device=device,
        )
        target_policy = torch.as_tensor(
            np.stack([example.policy_target for example in batch]),
            dtype=torch.float32,
            device=device,
        )
        target_value = torch.as_tensor(
            [example.value_target for example in batch],
            dtype=torch.float32,
            device=device,
        )

        logits, predicted_value = network(features)
        masked_logits = logits.masked_fill(
            ~masks,
            torch.finfo(logits.dtype).min,
        )
        log_probabilities = F.log_softmax(masked_logits, dim=-1)
        probabilities = log_probabilities.exp()

        policy_loss = -(target_policy * log_probabilities).sum(dim=-1).mean()
        value_loss = F.mse_loss(predicted_value, target_value)
        entropy = -(probabilities * log_probabilities).sum(dim=-1).mean()
        loss = (
            policy_loss
            + value_loss_weight * value_loss
            - entropy_weight * entropy
        )

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(network.parameters(), max_norm=5.0)
        optimizer.step()

        totals["loss"] += float(loss.detach().cpu())
        totals["policy"] += float(policy_loss.detach().cpu())
        totals["value"] += float(value_loss.detach().cpu())
        totals["entropy"] += float(entropy.detach().cpu())

    return {
        key: total / gradient_steps
        for key, total in totals.items()
    }


def format_policy(priors: dict[str, float]) -> str:
    return ", ".join(
        f"{action}={priors.get(action, 0.0):.3f}"
        for action in ALL_ACTIONS
        if action in priors
    )


def format_mcts_policy(
    policy: np.ndarray,
    environment: ToyChemistry,
    state: ToyState,
) -> str:
    """Format the legal components of an MCTS visit-count policy pi."""

    return ", ".join(
        (
            f"{action}={policy[ACTION_TO_INDEX[action]]:.4f} "
            f"({100.0 * policy[ACTION_TO_INDEX[action]]:.2f}%)"
        )
        for action in environment.valid_actions(state)
    )


def report_completed_episode(
    iteration_number: int,
    episode_number: int,
    decisions: Sequence[EpisodeDecision],
    terminal_state: ToyState,
    terminal_reward: float,
    environment: ToyChemistry,
) -> None:
    """Print pi and the real action at every decision in a finished episode."""

    print(
        f"\nIteration {iteration_number:2d}, episode {episode_number:2d} "
        "completed — REAL trajectory:"
    )
    for decision_number, decision in enumerate(decisions, start=1):
        next_state = environment.next_state(decision.state, decision.action)
        print(
            f"  decision {decision_number}: s={format_state(decision.state)}"
        )
        print(
            "    MCTS policy pi(s): "
            f"{format_mcts_policy(decision.policy, environment, decision.state)}"
        )
        print(
            f"    REAL ACTION sampled from pi: {decision.action} "
            f"-> s'={format_state(next_state)}"
        )

    route = " -> ".join(decision.action for decision in decisions)
    print(f"  REAL ROUTE: {route}")
    print(
        f"  TERMINAL STATE: {format_state(terminal_state)}; "
        f"terminal reward z={terminal_reward:.3f}"
    )


def report_leaf_evaluations(
    label: str,
    records: Sequence[tuple[int, SimulationTrace]],
    *,
    show_details: bool,
    include_episode: bool = True,
) -> None:
    """Report the evaluation-source mix and optionally every simulation.

    Percentages are calculated only across newly expanded nonterminal leaves,
    because direct terminal hits use exact rewards and are neither rollout nor
    neural evaluations.
    """

    rollout_records = [
        record for record in records if record[1].evaluation_source == "rollout"
    ]
    neural_records = [
        record for record in records if record[1].evaluation_source == "network"
    ]
    terminal_count = sum(
        trace.evaluation_source == "terminal" for _, trace in records
    )
    nonterminal_count = len(rollout_records) + len(neural_records)
    total_count = len(records)

    if nonterminal_count:
        rollout_percentage = 100.0 * len(rollout_records) / nonterminal_count
        neural_percentage = 100.0 * len(neural_records) / nonterminal_count
    else:
        rollout_percentage = 0.0
        neural_percentage = 0.0

    print(f"  {label} leaf-evaluation mix (new nonterminal leaves only):")
    print(
        f"    rollout: {len(rollout_records):4d}/{nonterminal_count:4d} "
        f"= {rollout_percentage:6.2f}%"
    )
    print(
        f"    neural:  {len(neural_records):4d}/{nonterminal_count:4d} "
        f"= {neural_percentage:6.2f}%"
    )
    print(
        f"    exact terminal hits: {terminal_count:4d}/{total_count:4d} "
        "of all simulations (excluded from the rollout/neural denominator)"
    )

    if not show_details:
        return

    print("    All simulation results (including exact terminal hits):")
    if not records:
        print("      (none)")
        return

    # A training episode can contain several real decisions, each with a new
    # MCTS search. The search counter therefore resets for each
    # (episode, decision-root-state) pair so that an 80-simulation search is
    # visibly numbered from 1 through 80.
    search_indices: dict[tuple[int, ToyState], int] = {}
    for overall_index, (episode_index, trace) in enumerate(records, start=1):
        source = trace.evaluation_source
        search_key = (episode_index, trace.root_state)
        search_indices[search_key] = search_indices.get(search_key, 0) + 1
        search_index = search_indices[search_key]
        episode_text = (
            f"episode={episode_index:2d}  " if include_episode else ""
        )
        tree_path = " -> ".join(trace.tree_actions) or "(no tree action)"
        if source == "rollout":
            rollout_path = (
                " -> ".join(trace.rollout_actions) or "(no rollout action)"
            )
            result = (
                f"leaf={format_state(trace.leaf_state)}  "
                f"rollout={rollout_path}  "
                f"terminal={format_state(trace.evaluated_state)}  "
                f"R={trace.value:.3f}"
            )
        elif source == "network":
            result = (
                f"leaf={format_state(trace.leaf_state)}  "
                f"V_theta={trace.value:.3f}"
            )
        elif source == "terminal":
            result = (
                f"leaf={format_state(trace.leaf_state)}  "
                f"R_terminal={trace.value:.3f}"
            )
        else:
            raise ValueError(f"Unknown simulation evaluation source: {source}")
        print(
            f"      simulation {overall_index:5d}: "
            f"{episode_text}root={format_state(trace.root_state):7s}  "
            f"search_sim={search_index:3d}  source={source:8s}  "
            f"tree={tree_path}  {result}"
        )


def report_network(
    network: PolicyValueNetwork,
    environment: ToyChemistry,
    device: torch.device,
) -> None:
    named_states = (
        ("C,O,H", ToyState()),
        ("C-O,H", ToyState(co=1)),
        ("C,O-H", ToyState(oh=1)),
    )
    print("\nNeural-network predictions")
    for name, state in named_states:
        priors, value = masked_policy_value(
            network,
            state,
            environment,
            device,
        )
        print(f"  {name:7s}  V_theta={value:.3f}  P_theta: {format_policy(priors)}")


def report_search_state(
    search: NeuralMCTS,
    state: ToyState,
    heading: str,
) -> None:
    """Print a locally normalized MCTS table for one expanded state."""
    node = search.nodes[state]
    rows = search.state_statistics(state)
    total_visits = sum(int(row["visits"]) for row in rows)
    print(
        f"\n{heading} at s={format_state(state)} "
        f"({total_visits} outgoing visits, V_theta={node.model_value:.3f})"
    )
    print("  action       P_theta     N       Q      pi")
    for row in rows:
        print(
            f"  {str(row['action']):10s}  "
            f"{float(row['prior']):7.3f}  "
            f"{int(row['visits']):4d}  "
            f"{float(row['Q']):6.3f}  "
            f"{float(row['pi']):6.3f}"
        )


def evaluate_search(
    network: PolicyValueNetwork,
    environment: ToyChemistry,
    device: torch.device,
    rng: np.random.Generator,
    *,
    simulations: int,
    c_puct: float,
    show_leaf_evaluations: bool,
) -> None:
    # Run a final search from the root state and report the results.
    root = ToyState()
    search = NeuralMCTS(
        environment,
        network,
        device,
        rng,
        c_puct=c_puct,
        use_network_value=True,
        uniform_priors=False,
        terminal_rollout_probability=0.0,
    )
    traces = search.run(root, simulations, add_root_noise=False)

    report_search_state(
        search,
        root,
        f"Final root search ({simulations} simulations, no root noise)",
    )
    second_level_states = (ToyState(co=1), ToyState(oh=1))
    for state in second_level_states:
        if state in search.nodes:
            report_search_state(search, state, "Final second-level search")
    report_leaf_evaluations(
        "Final evaluation",
        [(0, trace) for trace in traces],
        show_details=show_leaf_evaluations,
        include_episode=False,
    )


def resolve_device(requested: str) -> torch.device:
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def seed_everything(seed: int) -> tuple[random.Random, np.random.Generator]:
    python_rng = random.Random(seed)
    numpy_rng = np.random.default_rng(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    return python_rng, numpy_rng


def build_parser() -> argparse.ArgumentParser:
    # able to run the script from the command line with various options for training and evaluation
    parser = argparse.ArgumentParser(
        description=(
            "Train a toy policy-value network using MCTS targets generated "
            "from exact terminal rewards."
        )
    )
    parser.add_argument("--iterations", type=int, default=8)
    parser.add_argument("--episodes-per-iteration", type=int, default=32)
    parser.add_argument("--simulations", type=int, default=80)
    parser.add_argument("--evaluation-simulations", type=int, default=200)
    parser.add_argument("--warmup-iterations", type=int, default=1)
    parser.add_argument("--hidden-size", type=int, default=16)
    parser.add_argument("--hidden-layers", type=int, default=3)
    parser.add_argument("--learning-rate", type=float, default=3.0e-3)
    parser.add_argument("--weight-decay", type=float, default=1.0e-4)
    parser.add_argument("--gradient-steps", type=int, default=150)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--replay-capacity", type=int, default=10_000)
    parser.add_argument("--value-loss-weight", type=float, default=1.0)
    parser.add_argument("--entropy-weight", type=float, default=1.0e-3)
    parser.add_argument("--c-puct", type=float, default=1.5)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--gamma", type=float, default=1.0)
    parser.add_argument(
        "--terminal-rollout-probability",
        type=float,
        default=0.25,
        help="Fraction of neural-guided leaf evaluations replaced by exact rollouts.",
    )
    parser.add_argument(
        "--show-leaf-evaluations",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Print every MCTS simulation, including rollout, neural, and "
            "exact-terminal evaluations. Use "
            "--no-show-leaf-evaluations for percentage summaries only."
        ),
    )
    parser.add_argument(
        "--show-episode-decisions",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "After each completed episode, print the MCTS policy pi at every "
            "decision and the real action sampled from it. Use "
            "--no-show-episode-decisions for compact output."
        ),
    )
    parser.add_argument("--dirichlet-alpha", type=float, default=0.3)
    parser.add_argument("--dirichlet-epsilon", type=float, default=0.25)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument(
        "--device",
        default="auto",
        help="auto, cpu, cuda, mps, or another PyTorch device string",
    )
    parser.add_argument(
        "--model-out",
        type=Path,
        default=None,
        help="Optional path for a PyTorch checkpoint.",
    )
    return parser


def validate_arguments(args: argparse.Namespace) -> None:
    positive_integer_names = (
        "iterations",
        "episodes_per_iteration",
        "simulations",
        "evaluation_simulations",
        "gradient_steps",
        "batch_size",
        "replay_capacity",
        "hidden_size",
        "hidden_layers",
    )
    # Check that all positive integer arguments are indeed positive, and validate other constraints on the arguments.

    for name in positive_integer_names:
        if getattr(args, name) <= 0:
            raise ValueError(f"--{name.replace('_', '-')} must be positive")
    if args.warmup_iterations < 1:
        raise ValueError("--warmup-iterations must be at least 1")
    if args.warmup_iterations > args.iterations:
        raise ValueError("--warmup-iterations cannot exceed --iterations")
    if not 0.0 <= args.terminal_rollout_probability <= 1.0:
        raise ValueError("--terminal-rollout-probability must be in [0, 1]")
    if not 0.0 <= args.dirichlet_epsilon <= 1.0:
        raise ValueError("--dirichlet-epsilon must be in [0, 1]")
    if args.dirichlet_alpha <= 0.0:
        raise ValueError("--dirichlet-alpha must be positive")
    if args.gamma < 0.0 or args.gamma > 1.0:
        raise ValueError("--gamma must be in [0, 1]")


def main() -> None:
    args = build_parser().parse_args()  # parse command-line arguments
    validate_arguments(args)            # validate the parsed arguments
    python_rng, numpy_rng = seed_everything(args.seed)  # seed the random number generators for reproducibility
    device = resolve_device(args.device)

    environment = ToyChemistry()
    network = PolicyValueNetwork(
        input_size=len(StateEncoder.feature_names),
        hidden_size=args.hidden_size,
        hidden_layers=args.hidden_layers,
    ).to(device)
    optimizer = torch.optim.AdamW(
        network.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    replay_buffer = ReplayBuffer(args.replay_capacity)

    print(f"Device: {device}")
    print(f"Seed: {args.seed}")
    print(
        "Warm start: uniform priors + terminal rollouts; "
        "the untrained value head is ignored."
    )

    for iteration in range(args.iterations):
        # After warmup, the model starts steering tree search; before that,
        # all searches use uniform priors and exact rollouts.
        neural_guidance = iteration >= args.warmup_iterations

        # Per-iteration reward values from all generated episodes.
        rewards: list[float] = []

        # Newly created supervised examples for this training batch.
        new_examples: list[TrainingExample] = []

        # Count how often each full action route appears across episodes.
        route_counts: dict[tuple[str, ...], int] = {}

        # Keep all simulation traces for the current iteration for reporting.
        iteration_records: list[tuple[int, SimulationTrace]] = []

        for episode_number in range(1, args.episodes_per_iteration + 1):
            (
                examples,
                reward,
                route,
                traces,
                decisions,
                terminal_state,
            ) = generate_episode(
                environment,
                network,
                device,
                numpy_rng,
                simulations=args.simulations,
                c_puct=args.c_puct,
                temperature=args.temperature,
                gamma=args.gamma,
                neural_guidance=neural_guidance,
                terminal_rollout_probability=(
                    args.terminal_rollout_probability
                ),
                dirichlet_alpha=args.dirichlet_alpha,
                dirichlet_epsilon=args.dirichlet_epsilon,
            )
            if args.show_episode_decisions:
                report_completed_episode(
                    iteration + 1,
                    episode_number,
                    decisions,
                    terminal_state,
                    reward,
                    environment,
                )
            rewards.append(reward)
            new_examples.extend(examples)
            route_counts[route] = route_counts.get(route, 0) + 1
            iteration_records.extend(
                (episode_number, trace) for trace in traces
            )

        replay_buffer.extend(new_examples)
        metrics = train_network(
            network,
            replay_buffer,
            optimizer,
            device,
            python_rng,
            gradient_steps=args.gradient_steps,
            batch_size=args.batch_size,
            value_loss_weight=args.value_loss_weight,
            entropy_weight=args.entropy_weight,
        )
        root_priors, root_value = masked_policy_value(
            network,
            ToyState(),
            environment,
            device,
        )
        mode = "neural+rollout" if neural_guidance else "rollout warmup"
        print(
            f"Iteration {iteration + 1:2d} [{mode:15s}]  "
            f"mean R={np.mean(rewards):.3f}  "
            f"buffer={len(replay_buffer):4d}  "
            f"L_policy={metrics['policy']:.4f}  "
            f"L_value={metrics['value']:.4f}  "
            f"V_theta(root)={root_value:.3f}"
        )
        print(f"  post-training root P_theta: {format_policy(root_priors)}")
        for state in (ToyState(co=1), ToyState(oh=1)):
            priors, value = masked_policy_value(
                network,
                state,
                environment,
                device,
            )
            print(
                f"  post-training s={format_state(state):7s}: "
                f"V_theta={value:.3f}  P_theta: {format_policy(priors)}"
            )
        most_common_routes = sorted(
            route_counts.items(),
            key=lambda item: (-item[1], item[0]),
        )[:3]
        print(
            "  common routes: "
            + "; ".join(
                f"{' -> '.join(route)} ({count})"
                for route, count in most_common_routes
            )
        )
        report_leaf_evaluations(
            f"Iteration {iteration + 1} search",
            iteration_records,
            show_details=args.show_leaf_evaluations,
        )

    report_network(network, environment, device)
    evaluate_search(
        network,
        environment,
        device,
        numpy_rng,
        simulations=args.evaluation_simulations,
        c_puct=args.c_puct,
        show_leaf_evaluations=args.show_leaf_evaluations,
    )

    if args.model_out is not None:
        args.model_out.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "model_state_dict": network.state_dict(),
                "actions": ALL_ACTIONS,
                "feature_names": StateEncoder.feature_names,
                "arguments": vars(args),
            },
            args.model_out,
        )
        print(f"\nSaved checkpoint: {args.model_out.resolve()}")


if __name__ == "__main__":
    main()

## RUN FROM TERMINAL:
# python neural_mcts_training.py --iterations 10 --simulations 100 --seed 42
