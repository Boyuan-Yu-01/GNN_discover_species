"""Stage II checks; load class definitions without running exploration."""
import ast
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import networkx as nx

source = Path("explore_stage2.py")
tree = ast.parse(source.read_text())
scope = {}
exec(compile(ast.Module(body=[n for n in tree.body if isinstance(
    n, (ast.Import, ast.ImportFrom, ast.ClassDef))], type_ignores=[]), str(source), "exec"), scope)
Stage1Loader = scope["Stage1Loader"]
Stage2Explorer = scope["Stage2Explorer"]


class Stage2Tests(unittest.TestCase):
    def setUp(self):
        self.api = Stage1Loader.load("../stageI")
        self.symbols = {0: "H", 1: "O", 2: "C"}
        self.valency = {0: 1, 1: 2, 2: 4}
        self.refs = self.api["ReferenceSpecies"](
            "../stageI/species_graphs.json", ["CO"], self.symbols, self.valency, 10)

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
                "stage1_directory": "../stageI",
                "checkpoint_path": "../stageI/output_stage1/20260916_105526_866421/trained_growth_gnn.pt",
                "reference_path": "../stageI/species_graphs.json", "output_folder": str(output),
                "compositions": [(0, 1, 1)], "samples_per_composition": 20,
                "temperature": 1., "seed": 45, "batch_size": 10, "threads": 1,
                "save_media": False, "exploration_runs": 2,
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

    def test_base_atoms_and_hydrogens(self):
        explorer = object.__new__(Stage2Explorer)
        explorer.symbols, explorer.valency = self.symbols, self.valency
        record = explorer.structure_record(self.hydroxyl())
        self.assertEqual(record["base_atom_indices"], {0: "O"})
        self.assertEqual(record["attached_h_per_base_atom"], [1])
        self.assertEqual(record["unused_valence_per_atom"], [1, 0])

    def test_end_to_end_repeatable_and_checkpoint_unchanged(self):
        checkpoint = Path("../stageI/output_stage1/20260916_105526_866421/trained_growth_gnn.pt")
        original = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
        with tempfile.TemporaryDirectory() as temp:
            results = []
            for run in range(2):
                output = Path(temp) / str(run)
                cfg = {
                    "stage1_directory": "../stageI", "checkpoint_path": str(checkpoint),
                    "reference_path": "../stageI/species_graphs.json",
                    "output_folder": str(output), "compositions": [(0, 1, 1), (2, 4, 1)],
                    "samples_per_composition": 7, "temperature": 1., "seed": 45,
                    "batch_size": 3, "threads": 1, "save_media": False,
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
                results.append(attempts)
            self.assertEqual(results[0], results[1])
        self.assertEqual(hashlib.sha256(checkpoint.read_bytes()).hexdigest(), original)


unittest.main(verbosity=2)
