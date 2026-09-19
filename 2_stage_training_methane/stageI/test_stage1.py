"""Focused checks without executing the training call at the bottom of the script."""
# ast inspects Python syntax without running the script; unittest runs the checks.
# Temporary directories keep test files/checkpoints out of the training outputs.
import ast
import json
import itertools
import random
import tempfile
import unittest
from pathlib import Path

import networkx as nx
import torch

# The training script deliberately uses direct execution, following project style.
# Load only imports/classes for isolated tests; the actual script is smoke-tested separately.
source = Path("train_stage1.py")
# Parse into a syntax tree, then retain only imports and class definitions.
# Omitting bottom-level assignments/calls avoids constructing a trainer or creating a run.
tree = ast.parse(source.read_text())
definitions = ast.Module(body=[node for node in tree.body
                              if isinstance(node, (ast.Import, ast.ImportFrom, ast.ClassDef))],
                         type_ignores=[])
# Execute those definitions in their own namespace. This uses our local Python source,
# not data from the species JSON. No training loop is invoked.
scope = {"__file__": str(source), "__name__": "stage1_test_definitions"}
exec(compile(definitions, str(source), "exec"), scope)
# Retrieve the same implementation classes the actual training script uses.
AdvancedMoleculeEnv = scope["AdvancedMoleculeEnv"]
ActionSpace = scope["ActionSpace"]
ReferenceSpecies = scope["ReferenceSpecies"]
GrowthGNN = scope["GrowthGNN"]
Stage1Trainer = scope["Stage1Trainer"]


class Stage1Tests(unittest.TestCase):
    # unittest calls setUp before EACH test, so tests do not share mutable environment state.
    def setUp(self):
        self.symbols = {0: "H", 1: "O", 2: "C"}
        # These limits constrain bond-order sums: a double bond consumes two units.
        self.valency = {0: 1, 1: 2, 2: 4}
        self.space = ActionSpace(10, self.symbols)
        self.refs = ReferenceSpecies("species_graphs.json", ["CO"],
                                     self.symbols, self.valency, 10)
        torch.set_num_threads(1)

    def test_exhaustive_counts_and_independent_ch3cho_oracle(self):
        # Independently enumerate the 7! atom orders of this tree. Every prefix
        # must stay connected; serialize actions to remove interchangeable labels.
        graph = self.refs.graphs["CH3CHO"]
        expected = set()
        labeled_count = 0
        for order in itertools.permutations(graph.nodes):
            mapping = {order[0]: 0}
            actions = [("START", graph.nodes[order[0]]["atom"])]
            for node in order[1:]:
                parents = [u for u in graph.neighbors(node) if u in mapping]
                if not parents:
                    break
                parent = parents[0]
                actions.append(("GROW", mapping[parent], graph.nodes[node]["atom"],
                                graph.edges[parent, node]["order"]))
                mapping[node] = len(mapping)
            else:
                labeled_count += 1
                expected.add(tuple(actions + [("STOP",)]))
        self.assertEqual(labeled_count, 600)
        self.assertEqual(len(expected), 100)
        self.assertEqual(set(self.refs.enumerate_sequences("CH3CHO")), expected)
        for name, count in {"H": 1, "H2": 1, "OH": 2, "H2O": 2, "CH4": 2}.items():
            self.assertEqual(len(self.refs.enumerate_sequences(name)), count)
        # Renumbering the input graph cannot change the unique action pool.
        self.refs.graphs["CH3CHO"] = nx.relabel_nodes(graph, {u: 100-u for u in graph})
        self.assertEqual(set(self.refs.enumerate_sequences("CH3CHO")), expected)

    def test_cycle_enumeration_includes_connect(self):
        graph = nx.Graph()
        graph.add_nodes_from((u, {"atom": 2}) for u in range(3))
        graph.add_edges_from((u, v, {"order": 1}) for u, v in [(0, 1), (1, 2), (2, 0)])
        self.refs.graphs = {"ring": graph}
        self.refs.build_demonstrations(self.space)
        for items in self.refs.pool["ring"]:
            self.assertEqual(sum(item["action"][0] == "CONNECT" for item in items), 1)
            env = self.env()
            for item in items:
                env.step(item["action"])
            self.assertTrue(self.refs.isomorphic(env.to_graph(), graph))

    def test_percentage_sampling_and_equal_species_weights(self):
        self.assertEqual(self.refs.sample_count(100, 0.255), 26)
        self.assertEqual(self.refs.sample_count(1, 0.2), 1)
        for fraction in [0, -0.1, 1.01, float("nan"), float("inf"), True]:
            with self.assertRaises(ValueError):
                self.refs.sample_count(100, fraction)
        # Unequal species counts AND unequal sequence lengths must still give
        # each species half the loss weight, and equal weight to its own sequences.
        cached = {"small": [(("small", i), torch.zeros(i+1)) for i in range(2)],
                  "large": [(("large", i), torch.zeros(i+1)) for i in range(10)]}
        rng = random.Random(17)
        selected, weights = Stage1Trainer.select_trajectories(cached, 0.25, rng)
        self.assertEqual(len(selected), 4)  # ceil(2*.25) + ceil(10*.25)
        identities = [entry[0] for entry in selected]
        self.assertEqual(len(identities), len(set(identities)))
        offset = 0
        totals = {"small": 0., "large": 0.}
        for identity, target in selected:
            mass = float(weights[offset:offset+len(target)].sum())
            self.assertAlmostEqual(mass, 0.5 if identity[0] == "small" else 0.5/3)
            totals[identity[0]] += mass
            offset += len(target)
        for mass in totals.values():
            self.assertAlmostEqual(mass, 0.5)
        self.assertAlmostEqual(float(weights.sum()), 1.0, places=6)
        again, _ = Stage1Trainer.select_trajectories(cached, 0.25, random.Random(17))
        self.assertEqual(identities, [entry[0] for entry in again])
        selections = {tuple(entry[0] for entry in
                            Stage1Trainer.select_trajectories(cached, 0.25, rng)[0])
                      for _ in range(3)}
        self.assertGreater(len(selections), 1)
        all_selected, _ = Stage1Trainer.select_trajectories(cached, 1.0, rng)
        self.assertEqual(len(all_selected), 12)
        self.assertEqual(len({entry[0] for entry in all_selected}), 12)

    # Make a fresh empty molecule; actions in one test should not affect another.
    def env(self):
        return AdvancedMoleculeEnv(self.symbols, self.valency, 10)

    # Replay ALL unique construction orders for every species. Verify intermediate
    # snapshots and the completed graph, not merely the final chemical formula.
    def test_all_references_and_enumerated_orders_reconstruct(self):
        self.assertEqual(len(self.refs.graphs), 31)
        self.assertNotIn("CO", self.refs.graphs)
        self.refs.build_demonstrations(self.space)
        for name, orders in self.refs.pool.items():
            sequences = [tuple(item["action"] for item in trajectory) for trajectory in orders]
            self.assertEqual(len(sequences), len(set(sequences)))
            for trajectory in orders:
                env = self.env()
                for item in trajectory:
                    self.assertEqual(env.record(), item["env"].record())
                    # Every acceptable teaching label must also be allowed by the action mask.
                    self.assertTrue(set(item["target_indices"]).issubset(
                        set(env.get_valid_action_masks(self.space).nonzero().flatten().tolist())))
                    env.step(item["action"])
                self.assertTrue(env.terminated)
                self.assertTrue(self.refs.isomorphic(env.to_graph(), self.refs.graphs[name]))

    # At OH, STOP and another H addition must BOTH be legal. The condition/policy
    # chooses the desired outcome; the mask must not force saturation into H2O.
    def test_stop_for_radicals_and_competing_additions(self):
        env = self.env()
        env.step(("START", 1))
        # GROW tuple: attachment node 0, new element code 0 (H), bond order 1.
        env.step(("GROW", 0, 0, 1))
        self.assertIn(("STOP",), env.get_valid_actions())
        self.assertIn(("GROW", 0, 0, 1), env.get_valid_actions())
        self.assertEqual(self.refs.match(env.to_graph()), "OH")
        # Branch the same OH state into two outcomes without modifying the original.
        water = env.copy()
        water.step(("GROW", 0, 0, 1))
        self.assertEqual(self.refs.match(water.to_graph()), "H2O")
        env.step(("STOP",))
        self.assertEqual(env.get_valid_actions(), [])

    # Carbon can legally receive H/O/C; desired composition is not encoded into masks.
    # Also check that an undirected bond appears only once and upgrades count correctly.
    def test_masks_are_generic_and_bonds_are_unique(self):
        env = self.env()
        env.step(("START", 2))
        for atom in self.symbols:
            self.assertIn(("GROW", 0, atom, 1), env.get_valid_actions())
        self.assertNotIn(("GROW", 0, 0, 2), env.get_valid_actions())
        env.step(("GROW", 0, 2, 1))
        self.assertIn(("CONNECT", 0, 1, 3), env.get_valid_actions())
        self.assertNotIn(("CONNECT", 1, 0, 3), self.space.index)
        env.step(("CONNECT", 0, 1, 3))
        # The final C-C order is 3, not 1+3: CONNECT requested a final order, not an increment.
        self.assertEqual(env.current_bonds, [3, 3])
        with self.assertRaises(ValueError):
            env.step(("GROW", 0, 0, 2))

    # Even at the maximum atom count, the policy must still have a legal STOP action.
    def test_size_cap_still_allows_stop(self):
        env = AdvancedMoleculeEnv(self.symbols, self.valency, 1)
        env.step(("START", 2))
        self.assertEqual(env.get_valid_actions(), [("STOP",)])

    # Equal (C,H,O) counts do not imply equal topology. Both isomers must retain
    # their own reference names when matched by atoms AND bond orders.
    def test_isomers_have_same_condition_but_distinct_graphs(self):
        for a, b in [("CH3O", "CH2OH"), ("CH2CHO", "CH3CO")]:
            self.assertEqual(self.refs.composition(self.refs.graphs[a]),
                             self.refs.composition(self.refs.graphs[b]))
            self.assertFalse(self.refs.isomorphic(self.refs.graphs[a], self.refs.graphs[b]))
            self.assertEqual(self.refs.match(self.refs.graphs[a]), a)
            self.assertEqual(self.refs.match(self.refs.graphs[b]), b)

    # Check network mechanics on a small random model, without a training run:
    # probability normalization, masks, symmetry, gradients, and checkpoint restoration.
    def test_masked_distribution_symmetry_and_finite_gradients(self):
        torch.manual_seed(11)
        model = GrowthGNN([16, 24, 16], self.space, self.symbols, self.valency)
        env = self.env()
        env.step(("START", 2))
        env.step(("GROW", 0, 2, 1))
        # The two carbons in a bare C-C fragment are interchangeable attachment sites.
        equivalent = self.refs.equivalent_actions(env, ("GROW", 0, 0, 1))
        self.assertEqual(set(equivalent), {("GROW", 0, 0, 1), ("GROW", 1, 0, 1)})
        # Mix empty and nonempty graphs in one padded batch; conditions use C,H,O order.
        inputs = model.encode([self.env(), env], [(1, 4, 0), (2, 6, 0)])
        logits = model(*inputs)
        self.assertEqual(tuple(logits.shape), (2, len(self.space.actions)))
        probabilities = logits.softmax(-1)
        self.assertTrue(torch.allclose(probabilities.sum(1), torch.ones(2)))
        self.assertTrue(torch.all(probabilities[~inputs[-1]] == 0))
        indices = [self.space.index[a] for a in equivalent]
        self.assertTrue(torch.allclose(probabilities[1, indices[0]],
                                       probabilities[1, indices[1]], atol=1e-6))
        # Both equivalent actions are correct, so differentiate -log(P(left)+P(right)).
        # This checks the same probability-summing principle used in the training loss.
        loss = -torch.logsumexp(logits.log_softmax(-1)[1, indices], dim=0)
        loss.backward()
        self.assertTrue(all(p.grad is None or torch.isfinite(p.grad).all()
                            for p in model.parameters()))
        # Reload into a separate model and require identical predictions. Saving a
        # checkpoint successfully is not enough if its parameters cannot be restored.
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "model.pt"
            torch.save(model.state_dict(), path)
            restored = GrowthGNN([16, 24, 16], self.space, self.symbols, self.valency)
            restored.load_state_dict(torch.load(path, weights_only=True))
            self.assertTrue(torch.allclose(model(*inputs), restored(*inputs)))

    # Build C-O twice with swapped node numbers. Each corresponding action should
    # have the same score, so the network cannot rely on arbitrary atom numbering.
    def test_permuting_nodes_permutates_actions(self):
        torch.manual_seed(9)
        model = GrowthGNN([16, 16], self.space, self.symbols, self.valency)
        a = self.env()
        a.step(("START", 2))
        a.step(("GROW", 0, 1, 1))
        b = self.env()
        b.step(("START", 1))
        b.step(("GROW", 0, 2, 1))
        logits = model(*model.encode([a, b], [(1, 4, 1), (1, 4, 1)]))
        for action in a.get_valid_actions():
            if action[0] == "GROW":
                # Node 0 in one graph corresponds to node 1 in the other, and vice versa.
                mapped = ("GROW", 1 - action[1], action[2], action[3])
            else:
                mapped = action
            self.assertTrue(torch.allclose(logits[0, self.space.index[action]],
                                           logits[1, self.space.index[mapped]], atol=1e-5))

    # Deliberately supply an H=O double bond. Hydrogen permits only one bond-order
    # unit, so the reference loader must reject it before training can begin.
    def test_loader_rejects_overvalence(self):
        bad = {"bad": {"nodes": [{"id": 0, "atom": 0, "symbol": "H"},
                                  {"id": 1, "atom": 1, "symbol": "O"}],
                       "edges": [{"source": 0, "target": 1, "order": 2}]}}
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "bad.json"
            path.write_text(json.dumps(bad))
            with self.assertRaisesRegex(ValueError, "valence"):
                ReferenceSpecies(path, [], self.symbols, self.valency, 10)


# Execute tests directly, following the project style. verbosity=2 prints each
# test name and its result. These checks do not measure trained chemical accuracy.
unittest.main(verbosity=2)
