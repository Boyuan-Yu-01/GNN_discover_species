"""PUCT exploration checks; run from self_improving/stageII without starting a campaign."""
import ast
import hashlib
import json
import tempfile
import unittest
from unittest.mock import patch
from pathlib import Path

import networkx as nx

source = Path("explore_stage2.py")
tree = ast.parse(source.read_text())
scope = {}
exec(compile(ast.Module(body=[n for n in tree.body if isinstance(
    n, (ast.Import, ast.ImportFrom, ast.ClassDef))], type_ignores=[]), str(source), "exec"), scope)
Stage1Loader = scope["Stage1Loader"]
Stage2Explorer = scope["Stage2Explorer"]
PUCTSearch = scope["PUCTSearch"]


class Stage2Tests(unittest.TestCase):
    def setUp(self):
        self.api = Stage1Loader.load(PATH_SETTINGS["STAGE1_DIRECTORY"])
        self.symbols = {0: "H", 1: "O", 2: "C"}
        self.valency = {0: 1, 1: 2, 2: 4}
        self.refs = self.api["ReferenceSpecies"](
            PATH_SETTINGS["REFERENCE_JSON_PATH"], ["CO"], self.symbols, self.valency, 10)

    def hydroxyl(self, stop=True):
        env = self.api["AdvancedMoleculeEnv"](self.symbols, self.valency, 10)
        env.step(("START", 1))
        env.step(("GROW", 0, 0, 1))
        if stop:
            env.step(("STOP",))
        return env

    def test_loading_definitions_does_not_execute_training(self):
        self.assertIn("GrowthGNN", self.api)
        self.assertNotIn("trainer", self.api)
        self.assertNotIn("training_config", self.api)

    def test_categories_require_stop_and_requested_composition(self):
        env = self.hydroxyl()
        self.assertEqual(Stage2Explorer.classify(env, (0, 1, 1), "OH"), "reference_match")
        self.assertEqual(Stage2Explorer.classify(env, (0, 2, 1), "OH"), "failed_composition")
        self.assertEqual(Stage2Explorer.classify(env, (0, 1, 1), None), "new_candidate")
        self.assertEqual(Stage2Explorer.classify(
            self.hydroxyl(False), (0, 1, 1), "OH"), "failed_step_limit")

    def test_conditions_validate_counts_and_remove_duplicates(self):
        self.assertEqual(Stage2Explorer.validate_conditions([(0, 1, 1)]*2, 10), [(0, 1, 1)])
        for conditions in [[], [(0, 0, 0)], [(-1, 2, 1)], [(1, 10, 0)],
                           [(1, 2)], [(1.0, 2, 1)], [(True, 2, 1)]]:
            with self.assertRaises(ValueError):
                Stage2Explorer.validate_conditions(conditions, 10)

    def test_deduplication_preserves_isomers_and_bond_orders(self):
        graph = self.refs.graphs["CH3O"]
        record = {"structure_id": "example"}
        buckets = {(1, 3, 1): [(graph, record)]}
        relabeled = nx.relabel_nodes(graph, {u: 100-u for u in graph})
        self.assertIs(Stage2Explorer.find_structure(relabeled, (1, 3, 1), buckets, self.refs), record)
        self.assertIsNone(Stage2Explorer.find_structure(
            self.refs.graphs["CH2OH"], (1, 3, 1), buckets, self.refs))
        changed = graph.copy()
        u, v = next(iter(changed.edges))
        changed.edges[u, v]["order"] = 2
        self.assertIsNone(Stage2Explorer.find_structure(changed, (1, 3, 1), buckets, self.refs))

    def test_campaign_merges_runs_without_double_counting_structures(self):
        campaign = scope["ExplorationCampaign"]
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp) / "campaign"
            cfg = {
                "stage1_directory": PATH_SETTINGS["STAGE1_DIRECTORY"],
                "checkpoint_path": PATH_SETTINGS["CHECKPOINT_PATH"],
                "reference_path": PATH_SETTINGS["REFERENCE_JSON_PATH"], "output_folder": str(output),
                "compositions": [(0, 1, 1)], "samples_per_composition": 20,
                "temperature": 1., "seed": 45, "batch_size": 10, "threads": 1,
                "save_media": False, "exploration_runs": 2,
                "search_method": "puct",
            }
            summary = campaign(cfg).run()
            attempts = json.loads((output / "attempts.json").read_text())
            unique = json.loads((output / "unique_structures.json").read_text())
            self.assertEqual(summary["total_attempts"], 40)
            self.assertEqual(len(attempts), 40)
            self.assertEqual([r["attempt"] for r in attempts], list(range(1, 41)))
            self.assertEqual({r["run"] for r in attempts}, {1, 2})
            self.assertEqual(len({r["seed"] for r in summary["runs"]}), 2)
            self.assertEqual(summary["unique_reference_structures"], 1)
            oh = next(r for r in unique if r["species_key"] == "OH")
            self.assertEqual(oh["runs_observed"], [1, 2])
            self.assertEqual(oh["occurrences"], summary["counts"]["reference_match"])
            self.assertEqual(oh["frequency_within_requested_composition"], oh["occurrences"]/40)
            self.assertEqual(sum(r["occurrences"] for r in unique),
                             sum(r["structure_id"] is not None for r in attempts))
            self.assertEqual({r["structure_id"] for r in attempts if r["structure_id"]},
                             {r["structure_id"] for r in unique})
            self.assertEqual(sum(r["total_attempts"] for r in summary["runs"]), 40)
            stats = json.loads((output / "search_statistics.json").read_text())
            self.assertEqual([r["run"] for r in stats], [1, 2])
            self.assertEqual([r["simulations"] for r in stats], [20, 20])
            # Running again at the SAME path replaces results. A smaller campaign
            # must not leave extra run folders or disabled PUCT/media outputs.
            (output / "final_structures.png").write_bytes(b"old plot")
            (output / "exploration_growth.mp4").write_bytes(b"old animation")
            replacement = campaign({**cfg, "exploration_runs": 1,
                                    "samples_per_composition": 5,
                                    "search_method": "sampling"}).run()
            self.assertEqual(replacement["total_attempts"], 5)
            self.assertEqual(len(json.loads((output / "attempts.json").read_text())), 5)
            self.assertFalse((output / "runs" / "run_002").exists())
            for name in ("final_structures.png", "exploration_growth.mp4", "search_statistics.json"):
                self.assertFalse((output / name).exists())
            self.assertNotIn("method=puct", (output / "runs" / "run_001" / "exploration.log").read_text())
            self.assertNotIn("run=2/2", (output / "exploration.log").read_text())

    def test_base_atoms_and_hydrogens(self):
        explorer = object.__new__(Stage2Explorer)
        explorer.symbols, explorer.valency = self.symbols, self.valency
        record = explorer.structure_record(self.hydroxyl())
        self.assertEqual(record["base_atom_indices"], {0: "O"})
        self.assertEqual(record["attached_h_per_base_atom"], [1])
        self.assertEqual(record["unused_valence_per_atom"], [1, 0])

    def test_puct_explores_tiny_prior_and_backs_up_terminal_reward(self):
        # A controlled two-choice problem tests the SEARCH itself, independent of
        # chemistry: the almost-zero-prior branch has the better observed reward.
        class TinyEnv:
            def __init__(self, branch=None):
                self.branch = branch
                self.terminated = branch is not None

            def copy(self):
                return TinyEnv(self.branch)

            def get_valid_actions(self):
                return [] if self.terminated else [("START", 0), ("START", 1)]

            def step(self, action):
                self.branch, self.terminated = action[1], True

            def composition(self):
                return (0, 1, 0)

            def to_graph(self):
                return self.branch

        class TinyModel:
            def encode(self, envs, conditions):
                return ()

            def __call__(self):
                return scope["torch"].tensor([[0.0, -100.0]])

        class TinySpace:
            index = {("START", 0): 0, ("START", 1): 1}

        class TinyReferences:
            def match(self, graph):
                return "Known" if graph == 0 else None

        cfg = {**PUCTSearch.DEFAULTS, "temperature": 1., "puct_require_composition": False}
        search = PUCTSearch(TinyEnv(), (0, 1, 0), TinyModel(), TinySpace(),
                            TinyReferences(), cfg, scope["torch"].Generator().manual_seed(1), 2)
        for _ in range(100):
            env, trace, diagnostic = search.simulate()
            self.assertEqual(diagnostic["terminal_reward"], 0.1 if env.branch == 0 else 1.)
        edges = search.statistics()["root_edges"]
        self.assertEqual(search.root["visits"], 100)
        self.assertEqual(sum(e["visits"] for e in edges), 100)
        self.assertLess(edges[1]["policy_probability"], 1e-30)
        self.assertGreaterEqual(edges[1]["prior"], 0.125)
        self.assertGreater(edges[1]["visits"], edges[0]["visits"])
        self.assertEqual(edges[1]["mean_reward"], 1.)

    def test_puct_formula_constraint_dead_end_and_reward(self):
        cfg = {**PUCTSearch.DEFAULTS, "temperature": 1.}
        env = self.api["AdvancedMoleculeEnv"](self.symbols, self.valency, 10)
        search = PUCTSearch(env, (0, 2, 1), None, None, self.refs, cfg, None, 10)
        self.assertEqual(search.admissible_actions(env), [("START", 0), ("START", 1)])
        env.step(("START", 1))
        self.assertNotIn(("STOP",), search.admissible_actions(env))
        self.assertTrue(all(a[0] != "GROW" or a[2] == 0 for a in search.admissible_actions(env)))
        env.step(("GROW", 0, 0, 1))
        env.step(("GROW", 0, 0, 1))
        self.assertEqual(search.admissible_actions(env), [("STOP",)])
        env.step(("STOP",))
        self.assertEqual(search.reward(env), 0.1)
        search.condition = (0, 1, 1)
        self.assertEqual(search.reward(env), 0.)
        # H2 cannot grow to connected H3 under the existing valence rules.
        blocked = self.api["AdvancedMoleculeEnv"](self.symbols, self.valency, 10)
        blocked.step(("START", 0))
        blocked.step(("GROW", 0, 0, 1))
        dead = PUCTSearch(blocked, (0, 3, 0), None, None, self.refs, cfg, None, 10)
        _, trace, info = dead.simulate()
        self.assertTrue(info["dead_end"])
        self.assertEqual(info["terminal_reward"], 0.)
        self.assertEqual(trace, [])

    def test_puct_rejects_invalid_search_settings(self):
        for key, value in [("search_method", "unknown"), ("puct_c", 0),
                           ("puct_uniform_fraction", 1.1), ("puct_c", float("nan")),
                           ("puct_reference_reward", 1.), ("puct_require_composition", 1),
                           ("generation_mode", "invalid"), ("attempts_per_run", 0)]:
            with self.assertRaises(ValueError):
                PUCTSearch.validate({**PUCTSearch.DEFAULTS, key: value})

    def test_marginal_policy_averages_probabilities_without_target(self):
        torch = scope["torch"]

        class ConditionalModel:
            def encode(self, envs, conditions):
                self.received_conditions = conditions
                return (torch.tensor([[0.2, 0.8, 0.] if c == (0, 1, 1) else [0.6, 0.4, 0.]
                                      for c in conditions]).log(),)

            def __call__(self, logits):
                return logits

        base = ConditionalModel()
        policy = scope["MarginalGrowthPolicy"](base, [(0, 1, 1), (1, 4, 0)])
        probabilities = policy(*policy.encode([object(), object()], [None, None])).softmax(-1)
        self.assertTrue(torch.allclose(probabilities, torch.tensor([[0.4, 0.6, 0.], [0.4, 0.6, 0.]])))
        self.assertEqual(base.received_conditions, [(0, 1, 1), (1, 4, 0)] * 2)

    def test_free_growth_stop_and_reward_have_no_formula_requirement(self):
        cfg = {**PUCTSearch.DEFAULTS, "generation_mode": "free", "temperature": 1.}
        env = self.api["AdvancedMoleculeEnv"](self.symbols, self.valency, 2)
        search = PUCTSearch(env, None, None, None, self.refs, cfg, None, 24)
        self.assertNotIn(("STOP",), search.admissible_actions(env))
        env.step(("START", 2))
        self.assertIn(("STOP",), search.admissible_actions(env))
        self.assertIn(("GROW", 0, 2, 1), search.admissible_actions(env))
        env.step(("GROW", 0, 2, 1))
        self.assertFalse(any(a[0] == "GROW" for a in search.admissible_actions(env)))
        env.step(("STOP",))
        self.assertEqual(search.reward(env), 1.)
        self.assertEqual(Stage2Explorer.classify(env, None, None), "new_candidate")
        self.assertEqual(search.reward(self.hydroxyl()), 0.1)

    def test_free_growth_campaign_counts_replay_and_deduplication(self):
        with tempfile.TemporaryDirectory() as temp:
            repeated = []
            for repetition in range(2):
                output = Path(temp) / str(repetition)
                cfg = {"stage1_directory": PATH_SETTINGS["STAGE1_DIRECTORY"],
                       "checkpoint_path": PATH_SETTINGS["CHECKPOINT_PATH"],
                       "reference_path": PATH_SETTINGS["REFERENCE_JSON_PATH"],
                       "output_folder": str(output), "compositions": None,
                       "attempts_per_run": 12, "exploration_runs": 2,
                       "batch_size": 5, "threads": 1, "temperature": 1., "seed": 45,
                       "save_media": False, "search_method": "puct", "generation_mode": "free"}
                summary = scope["ExplorationCampaign"](cfg).run()
                attempts = json.loads((output / "attempts.json").read_text())
                unique = json.loads((output / "unique_structures.json").read_text())
                self.assertEqual(summary["total_attempts"], 24)
                self.assertEqual(sum(p["attempts"] for p in summary["per_generated_composition"]), 24)
                self.assertEqual(summary["counts"].get("failed_composition", 0), 0)
                self.assertGreater(len({tuple(a["generated_C_H_O"]) for a in attempts}), 1)
                self.assertTrue(all(a["requested_C_H_O"] is None for a in attempts))
                for record in attempts:
                    env = self.api["AdvancedMoleculeEnv"](self.symbols, self.valency, 10)
                    for step in record["actions"]:
                        env.step(tuple(step["action"]))
                    self.assertEqual(env.record(), record["graph"])
                    self.assertLessEqual(len(env.node_types), 10)
                    self.assertLessEqual(len(record["actions"]), 24)
                    expected = {"new_candidate": 1., "reference_match": 0.1}.get(record["category"], 0.)
                    self.assertEqual(record["search"]["terminal_reward"], expected)
                self.assertEqual(sum(r["occurrences"] for r in unique),
                                 sum(r["structure_id"] is not None for r in attempts))
                self.assertEqual(len({r["structure_id"] for r in unique}), len(unique))
                for record in unique:
                    self.assertIsNone(record["frequency_within_requested_composition"])
                    self.assertEqual(record["frequency_overall"], record["occurrences"] / 24)
                repeated.append(attempts)
            self.assertEqual(repeated[0], repeated[1])

    def test_recovery_requires_stop_exact_formula_and_isomer(self):
        env = self.api["AdvancedMoleculeEnv"](self.symbols, self.valency, 10)
        for action in [("START", 2), ("GROW", 0, 1, 1),
                       ("GROW", 0, 0, 1), ("GROW", 0, 0, 1), ("GROW", 0, 0, 1)]:
            env.step(action)
        targets = {"CH3O", "CH2OH"}
        match = Stage2Explorer.recovered_names
        self.assertEqual(match(env, (1, 3, 1), self.refs, targets), set())
        env.step(("STOP",))
        self.assertEqual(match(env, (1, 3, 1), self.refs, targets), {"CH3O"})
        self.assertEqual(match(env, (1, 4, 1), self.refs, targets), set())

    def test_recovery_targets_reject_unreachable_settings(self):
        cfg = {**PUCTSearch.DEFAULTS, "search_method": "puct",
               "stop_when_all_species_recovered": True}
        resolve = Stage2Explorer.recovery_targets
        for names, conditions, steps in [(["CO"], list(self.refs.conditions), 24),
                                          (["OH"], [(1, 3, 1)], 24),
                                          (["OH"], [(0, 1, 1)], 2),
                                          ([], list(self.refs.conditions), 24)]:
            with self.assertRaises(ValueError):
                resolve({**cfg, "recovery_species": names}, self.refs, conditions, steps)
        self.assertEqual(resolve(cfg, self.refs, list(self.refs.conditions), 24), set(self.refs.graphs))

    def test_coverage_ignores_budgets_and_stops_at_last_distinct_target(self):
        # Force a duplicate isomer before the second target. Budget=1 must not
        # end the search early, and complete coverage must not run a fourth time.
        observed = []
        api, symbols, valency = self.api, self.symbols, self.valency

        def simulation(search):
            index = len(observed)
            self.assertLess(index, 3)
            observed.append(id(search))
            env = api["AdvancedMoleculeEnv"](symbols, valency, 10)
            actions = [("START", 2), ("GROW", 0, 1, 1), ("GROW", 0, 0, 1),
                       ("GROW", 0, 0, 1), ("GROW", 0 if index < 2 else 1, 0, 1), ("STOP",)]
            trace = []
            for action in actions:
                env.step(action)
                trace.append({"step": len(trace) + 1, "action": list(action), "probability": 1.})
            search.root["visits"] += 1
            return env, trace, {"dead_end": False, "simulation": index + 1,
                                "tree_depth": 1, "terminal_reward": 0.1}

        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp) / "coverage"
            cfg = {"stage1_directory": PATH_SETTINGS["STAGE1_DIRECTORY"],
                   "checkpoint_path": PATH_SETTINGS["CHECKPOINT_PATH"],
                   "reference_path": PATH_SETTINGS["REFERENCE_JSON_PATH"],
                   "output_folder": str(output), "compositions": [(1, 3, 1)],
                   "samples_per_composition": 1, "exploration_runs": 1, "batch_size": 100,
                   "threads": 1, "temperature": 1., "seed": 45, "save_media": False,
                   "search_method": "puct", "stop_when_all_species_recovered": True,
                   "recovery_species": ["CH3O", "CH2OH"]}
            with patch.object(PUCTSearch, "simulate", simulation):
                summary = scope["ExplorationCampaign"](cfg).run()
            self.assertEqual(summary["total_attempts"], 3)
            self.assertEqual(summary["termination_reason"], "all_species_recovered")
            self.assertTrue(summary["all_species_recovered"])
            self.assertEqual(summary["missing_species"], [])
            self.assertEqual(set(summary["recovered_species"]), {"CH3O", "CH2OH"})
            self.assertEqual(len(set(observed)), 1)  # Same tree across attempts.
            unique = json.loads((output / "unique_structures.json").read_text())
            self.assertEqual(sorted(r["frequency_within_requested_composition"] for r in unique),
                             [1 / 3, 2 / 3])
            progress = json.loads((output / "recovery_progress.json").read_text())
            self.assertTrue(progress["all_species_recovered"])
            self.assertEqual(progress["attempts"], 3)
            self.assertFalse((output / "runs").exists())

    def test_end_to_end_repeatable_and_checkpoint_unchanged(self):
        checkpoint = Path(PATH_SETTINGS["CHECKPOINT_PATH"])
        original = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
        with tempfile.TemporaryDirectory() as temp:
            results = []
            for run in range(4):
                output = Path(temp) / str(run // 2)
                cfg = {
                    "stage1_directory": PATH_SETTINGS["STAGE1_DIRECTORY"], "checkpoint_path": str(checkpoint),
                    "reference_path": PATH_SETTINGS["REFERENCE_JSON_PATH"],
                    "output_folder": str(output), "compositions": [(0, 1, 1), (2, 4, 1)],
                    "samples_per_composition": 7, "temperature": 1., "seed": 45,
                    "batch_size": 3, "threads": 1, "save_media": False,
                    "search_method": "sampling" if run < 2 else "puct",
                }
                summary = Stage2Explorer(cfg).run()
                attempts = json.loads((output / "attempts.json").read_text())
                unique = json.loads((output / "unique_structures.json").read_text())
                self.assertEqual(summary["total_attempts"], 14)
                self.assertEqual(sum(summary["counts"].values()), 14)
                self.assertEqual(sum(r["occurrences"] for r in unique),
                                 sum(r["category"] in ("reference_match", "new_candidate")
                                     for r in attempts))
                for record in unique:
                    self.assertEqual(record["requested_C_H_O"], record["generated_C_H_O"])
                    self.assertEqual(record["termination"], "STOP")
                # Replay every saved trace and verify its final graph exactly.
                for record in attempts:
                    env = self.api["AdvancedMoleculeEnv"](self.symbols, self.valency, 10)
                    for step in record["actions"]:
                        env.step(tuple(step["action"]))
                    self.assertEqual(env.record(), record["graph"])
                if cfg["search_method"] == "puct":
                    stats = json.loads((output / "search_statistics.json").read_text())
                    self.assertEqual([r["simulations"] for r in stats], [7, 7])
                    for item in stats:
                        self.assertEqual(sum(e["visits"] for e in item["root_edges"]), 7)
                    for record in attempts:
                        expected = {"new_candidate": 1., "reference_match": 0.1}.get(record["category"], 0.)
                        self.assertEqual(record["search"]["terminal_reward"], expected)
                        self.assertEqual(record["requested_C_H_O"], record["generated_C_H_O"])
                results.append(attempts)
            self.assertEqual(results[0], results[1])
            self.assertEqual(results[2], results[3])
        self.assertEqual(hashlib.sha256(checkpoint.read_bytes()).hexdigest(), original)


# Read the three literal path parameters from the exploration source without
# importing it (an import would start the full exploration campaign). Both the
# tests and exploration now use the same checkpoint when its path is edited.
PATH_SETTINGS = {
    node.targets[0].id: ast.literal_eval(node.value)
    for node in tree.body
    if isinstance(node, ast.Assign) and len(node.targets) == 1
    and isinstance(node.targets[0], ast.Name)
    and node.targets[0].id in ("STAGE1_DIRECTORY", "CHECKPOINT_PATH", "REFERENCE_JSON_PATH")
}

unittest.main(verbosity=2)
