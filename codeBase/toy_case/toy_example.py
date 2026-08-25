"""PUCT-guided Monte Carlo Tree Search for a toy bond-formation problem.

What this example demonstrates
------------------------------
The initial state contains separate C, O, and H atoms.  An action forms a
candidate bond or stops construction.  ``ToyChemistry`` supplies the legal
actions, deterministic state transitions, terminal-state test, and exact
terminal reward ``R_terminal``.

``ToyPolicyValue`` stands in for an already trained policy-value model.  Its
numbers are deliberately hard-coded for this small example:

* ``P_theta(a | s)`` is the prior preference for each legal action.
* ``V_theta(s)`` estimates the eventual reward from a nonterminal state.

These model predictions are not trained or changed by this file.  PUCT uses
them to guide search and learns search statistics for every explored edge:

* ``N(s, a)``: number of visits;
* ``W(s, a)``: sum of values backed up through the edge;
* ``Q(s, a) = W(s, a) / N(s, a)``: mean backed-up value.

The final report also derives a search-based state value

    V_MCTS(s) = sum_a W(s, a) / sum_a N(s, a).

This value is calculated on demand after the simulations; it is not updated or
used during PUCT.  Before a state has any outgoing visits, the reporting method
falls back to the fixed ``V_theta(s)`` stored when the state was expanded.

At each expanded state, PUCT selects the action maximizing

    Q(s, a) + c_puct * P_theta(a | s) * sqrt(sum_b N(s, b) + 1)
                / (1 + N(s, a)).

Each MCTS simulation then follows four steps:

1. Select actions with the PUCT score while states are already expanded.
2. Stop at a terminal state or the first previously unexpanded state.
3. Evaluate the leaf with exact ``R_terminal`` if terminal, otherwise expand
   it and use ``V_theta`` as a bootstrap estimate.
4. Back up that value through the selected edges, updating ``N``, ``W``, and
   therefore ``Q``.  This is single-agent planning, so signs are not alternated.

After all simulations, normalized root visit counts define the improved search
policy

    pi(a | s_root) = N(s_root, a) / sum_b N(s_root, b).

Thus, the quantities updated during this program are ``N``, ``W``, ``Q``, and
``pi``.  The final ``V_MCTS`` is derived from ``N`` and ``W`` only for
reporting.  The hard-coded ``P_theta`` and ``V_theta`` remain fixed, and exact
terminal rewards never change.  In a training system, pairs such as
``(state, pi, terminal return)`` would be used between searches to update a
neural network; see ``neural_mcts_training.py`` for that separate process.
"""

from dataclasses import dataclass, field
from math import sqrt


STOP = "STOP"


@dataclass(frozen=True)
class ToyState:
    """Bond-presence state for C-O, O-H, and C-H candidate bonds.
        Immutable."""

    co: int = 0
    oh: int = 0
    ch: int = 0
    stopped: bool = False


def format_state(state):
    """Return a compact chemical label for trace output."""
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
    """Small deterministic graph-edit environment."""

    def is_terminal(self, state):
        """Terminate if STOP, methanol, or methane_route is reached."""
        methanol = state.co == 1 and state.oh == 1
        methane_route = state.ch == 1
        return state.stopped or methanol or methane_route

    def terminal_value(self, state):
        """Reward completed C-O-H, O-H, C-H, and partial C-O states."""
        if state.co == 1 and state.oh == 1:
            return 1.0
        if state.oh == 1:
            return 0.6
        if state.ch == 1:
            return 0.6
        if state.co == 1:
            return 0.7
        return 0.0

    def valid_actions(self, state):
        """Return a list of valid actions for the given state."""
        if self.is_terminal(state):
            return []

        # Initial free capacities are C:1, O:2, H:1. (only for this toy example)
        c_free = 1 - state.co - state.ch
        o_free = 2 - state.co - state.oh
        h_free = 1 - state.oh - state.ch

        actions = []
        if not state.co and c_free > 0 and o_free > 0:
            actions.append("form_C-O")
        if not state.oh and o_free > 0 and h_free > 0:
            actions.append("form_O-H")
        if not state.ch and c_free > 0 and h_free > 0:
            actions.append("form_C-H")

        actions.append(STOP)
        return actions

    def next_state(self, state, action):
        if action not in self.valid_actions(state):
            raise ValueError(f"Illegal action: {action}")

        # Update the state as the action + the previous state.
        if action == "form_C-O":
            return ToyState(1, state.oh, state.ch)
        if action == "form_O-H":
            return ToyState(state.co, 1, state.ch)
        if action == "form_C-H":
            return ToyState(state.co, state.oh, 1)
        return ToyState(state.co, state.oh, state.ch, stopped=True)


class ToyPolicyValue:
    """Stand-in for a trained policy-value GNN."""

    def __call__(self, state, valid_actions):
        # initial action preference
        if state == ToyState():
            raw_priors = {
                "form_C-O": 0.45,
                "form_O-H": 0.30,
                "form_C-H": 0.20,
                STOP: 0.05,
            }
            # estimate the value of the current state (methanol is reachable, but not methane_route)
            value = 0.45

        elif state.co == 1 and state.oh == 0:
            raw_priors = {"form_O-H": 0.85, STOP: 0.15}
            value = 0.70
        elif state.oh == 1 and state.co == 0:
            raw_priors = {"form_C-O": 0.80, STOP: 0.20}
            value = 0.60
        else:   # this is a defensive fallback for any other state. Potential actions are equally likely, and the value is neutral.
            raw_priors = {action: 1.0 for action in valid_actions}
            value = 0.0

        # Mask illegal actions and renormalize.
        normalizer = sum(raw_priors.get(a, 0.0) for a in valid_actions)
        priors = {
            action: raw_priors.get(action, 0.0) / normalizer
            for action in valid_actions
        }
        return priors, value


@dataclass
class EdgeStats:    # statistics for a single edge (action) in the search tree
    prior: float # P(s,a)
    visits: int = 0 # N(s,a)
    value_sum: float = 0.0 # W(s,a)

    @property
    def q(self):
        if self.visits == 0:
            return 0.0
        return self.value_sum / self.visits # Q(s,a) = W(s,a)/N(s,a)


@dataclass
class SearchNode:
    edges: dict = field(default_factory=dict)
    # Fixed model estimate returned when this state is first expanded.
    model_value: float = 0.0
    # example: SearchNode(
            #     edges={
            #         "form_C-O": EdgeStats(prior=0.45),
            #         "form_O-H": EdgeStats(prior=0.30),
            #         "form_C-H": EdgeStats(prior=0.20),
            #         "STOP": EdgeStats(prior=0.05),
            #     }
            # )

class MCTS:
    def __init__(self, environment, model, c_puct=1.5):
        self.environment = environment      # ToyChemistry()
        self.model = model                  # ToyPolicyValue()
        self.c_puct = c_puct

        # Hashable ToyState keys form a transposition table.
        self.nodes = {}

    def expand(self, state):
        valid_actions = self.environment.valid_actions(state)   # give out the valid actions for the current state
        priors, value = self.model(state, valid_actions)        # get the prior probabilities and value for the current state from the model
        self.nodes[state] = SearchNode(     # structure used to store the search tree, with edges for each valid action and their corresponding EdgeStats
            edges={
                action: EdgeStats(prior=priors[action])
                for action in valid_actions
            },
            model_value=value,
        )
        return value

    def state_search_value(self, state):
        """Return the current visit-weighted MCTS value of an expanded state.

        Before any outgoing edge has been visited, fall back to the fixed model
        estimate V_theta(s).  Once search outcomes exist, use their mean:

            V_MCTS(s) = sum_a W(s,a) / sum_a N(s,a).

        This method is used only for the final report; model_value remains fixed.
        """
        node = self.nodes[state]
        total_visits = sum(edge.visits for edge in node.edges.values())
        if total_visits == 0:
            return node.model_value
        total_value = sum(edge.value_sum for edge in node.edges.values())
        return total_value / total_visits

    def select_action(self, node):
        """Return the selected action and every pre-selection PUCT term."""
        total_visits = sum(edge.visits for edge in node.edges.values())
        parent_scale = sqrt(total_visits + 1)

        rows = []
        for action, edge in node.edges.items():
            edge = node.edges[action]
            exploration = (
                self.c_puct
                * edge.prior
                * parent_scale
                / (1 + edge.visits)
            )
            rows.append(
                {
                    "action": action,
                    "prior": edge.prior,
                    "visits": edge.visits,
                    "Q": edge.q,
                    "U": exploration,
                    "PUCT": edge.q + exploration,
                }
            )

        selected_row = max(rows, key=lambda row: row["PUCT"])
        return selected_row["action"], rows, total_visits

    def simulate(self, root_state, return_trace=False):
        state = root_state
        path = []
        actions = []
        selections = []

        while True: # Keep walking until the game ends or a new state is encountered
            if self.environment.is_terminal(state):
                value = self.environment.terminal_value(state)
                evaluation_source = "R_terminal"
                break

            if state not in self.nodes:
                value = self.expand(state)
                evaluation_source = "V_theta"
                break

            node = self.nodes[state]
            action, puct_rows, total_visits = self.select_action(node)
            selections.append(
                {
                    "state": state,
                    "total_visits": total_visits,
                    "rows": puct_rows,
                    "selected_action": action,
                }
            )
            edge = node.edges[action]
            path.append((state, action, edge))
            actions.append(action)
            state = self.environment.next_state(state, action)

        # Single-agent backup: do not alternate the sign.
        backup_updates = []
        for edge_state, action, edge in reversed(path):
            previous_visits = edge.visits
            previous_value_sum = edge.value_sum
            previous_q = edge.q
            edge.visits += 1
            edge.value_sum += value
            backup_updates.append(
                {
                    "state": edge_state,
                    "action": action,
                    "previous_visits": previous_visits,
                    "visits": edge.visits,
                    "previous_value_sum": previous_value_sum,
                    "value_sum": edge.value_sum,
                    "previous_Q": previous_q,
                    "Q": edge.q,
                }
            )

        if return_trace:
            return {
                "actions": actions,
                "leaf_state": state,
                "value": value,
                "evaluation_source": evaluation_source,
                "selections": selections,
                "backup_updates": backup_updates,
            }
        return value

    def run(self, root_state, simulations, detailed_trace=False):
        if root_state not in self.nodes:
            self.expand(root_state)

        for simulation_number in range(1, simulations + 1):
            trace = self.simulate(root_state, return_trace=True)
            path = " -> ".join(trace["actions"]) or "(no action)"

            if detailed_trace:
                print(
                    f"\nSimulation {simulation_number:2d}: path={path}; "
                    f"leaf={format_state(trace['leaf_state'])}; "
                    f"{trace['evaluation_source']}={trace['value']:.3f}"
                )

                for selection in trace["selections"]:
                    state_label = format_state(selection["state"])
                    print(
                        f"  PUCT at s={state_label} "
                        f"(sum_b N(s,b)={selection['total_visits']}):"
                    )
                    print(
                        "    action       P(s,a)  N(s,a)  Q(s,a)  "
                        "U(s,a)  Q(s,a)+U(s,a)"
                    )
                    for row in selection["rows"]:
                        selected = (
                            "  <-- selected"
                            if row["action"] == selection["selected_action"]
                            else ""
                        )
                        print(
                            f"    {row['action']:10s}  "
                            f"{row['prior']:6.3f}  "
                            f"{row['visits']:6d}  "
                            f"{row['Q']:6.3f}  "
                            f"{row['U']:6.3f}  "
                            f"{row['PUCT']:13.3f}{selected}"
                        )

                print("  Backup Q updates (leaf to root):")
                for update in trace["backup_updates"]:
                    state_label = format_state(update["state"])
                    print(
                        f"    Q({state_label}, {update['action']}): "
                        f"N {update['previous_visits']}->{update['visits']}, "
                        f"W {update['previous_value_sum']:.3f}"
                        f"->{update['value_sum']:.3f}, "
                        f"Q {update['previous_Q']:.3f}->{update['Q']:.3f}"
                    )
            else:
                root_update = next(
                    (
                        update
                        for update in trace["backup_updates"]
                        if update["state"] == root_state
                    ),
                    None,
                )
                if root_update is None:
                    root_change = "none"
                else:
                    root_change = (
                        f"{root_update['action']}: "
                        f"N {root_update['previous_visits']}"
                        f"->{root_update['visits']}, "
                        f"Q {root_update['previous_Q']:.3f}"
                        f"->{root_update['Q']:.3f}"
                    )
                print(
                    f"Simulation {simulation_number:2d}: "
                    f"path={path}; value={trace['value']:.3f}; "
                    f"root change={root_change}"
                )

    def state_statistics(self, state):
        """Return action statistics for any expanded nonterminal state."""
        node = self.nodes[state]
        rows = []
        for action, edge in node.edges.items():
            rows.append(
                {
                    "action": action,
                    "prior": edge.prior,
                    "visits": edge.visits,
                    "Q": edge.q,
                }
            )
        return rows

    def root_statistics(self, root_state):
        """Backward-compatible wrapper for the root summary."""
        return self.state_statistics(root_state)


environment = ToyChemistry()    # Define chemistry rules, terminal states & its values, and valid actions
model = ToyPolicyValue()        # Define the policy and value networks
search = MCTS(environment, model, c_puct=1.5) # Define the MCTS search algorithm with the environment and model

root = ToyState()               # Define the initial state of the search tree: C, O, and H, no bonds
search.run(root, simulations=80, detailed_trace=True)

rows = search.root_statistics(root)     # rows is a list of action dictionaries, one dictionary per action at the root node. 
total_visits = sum(row["visits"] for row in rows) # total visits of all actions at the root node

print(f"\nRoot action policy at s={format_state(root)}:")
for row in rows:
    probability = row["visits"] / total_visits
    print(
        f"{row['action']:>10s}  "
        f"P={row['prior']:.2f}  "
        f"N={row['visits']:2d}  "
        f"Q={row['Q']:.3f}  "
        f"pi={probability:.4f}"
    )

# Report the local MCTS action policy at each expanded second-level state.
# The denominator is the number of outgoing visits from that state, not the
# number of visits at the root.  The first arrival expands the state without
# selecting one of its actions, so outgoing visits can be one less than the
# number of times its incoming root edge was visited.
second_level_states = [ToyState(co=1), ToyState(oh=1)]
for state in second_level_states:
    if state not in search.nodes:
        continue
    rows = search.state_statistics(state)
    total_visits = sum(row["visits"] for row in rows)
    print(
        f"\nSecond-level action policy at s={format_state(state)} "
        f"({total_visits} outgoing visits):"
    )
    for row in rows:
        probability = (
            row["visits"] / total_visits
            if total_visits > 0
            else row["prior"]
        )
        print(
            f"{row['action']:>10s}  "
            f"P={row['prior']:.2f}  "
            f"N={row['visits']:2d}  "
            f"Q={row['Q']:.3f}  "
            f"pi={probability:.4f}"
        )

print("\nModel estimates and updated MCTS state values:")
reported_states = [root] + second_level_states
for state in reported_states:
    if state not in search.nodes:
        continue
    node = search.nodes[state]
    print(
        f"  s={format_state(state):7s}  "
        f"V_theta={node.model_value:.3f}  "
        f"V_MCTS={search.state_search_value(state):.3f}"
    )
