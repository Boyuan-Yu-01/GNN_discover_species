"""Run Stage III comparison checks without executing the full report on import."""
import ast
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import networkx as nx
from PIL import Image


class Stage3Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.reference = WorkbookGraphs(REFERENCE_WORKBOOK, "Graph definitions")
        cls.by_name = {r["species_key"]: r for r in cls.reference.records}

    @staticmethod
    def prediction(graph, identity):
        return {"structure_id": identity, "termination": "STOP", "category": "new_candidate",
                "occurrences": 2, "graph": {
                    "nodes": [{"id": i, "symbol": d["symbol"],
                               **({"formal_charge": d["formal_charge"]} if d.get("formal_charge") is not None else {})}
                              for i, d in graph.nodes(data=True)],
                    "edges": [{"source": u, "target": v, "order": d["order"]}
                              for u, v, d in graph.edges(data=True)]}}

    def test_workbook_loads_all_entries_and_charged_co(self):
        self.assertEqual(len(self.reference.records), 96)
        co = self.by_name["CO"]["graph"]
        self.assertEqual(sorted(d["formal_charge"] for _, d in co.nodes(data=True)), [-1, 1])
        self.assertEqual(next(iter(co.edges(data=True)))[2]["order"], 3)

    def test_atom_numbering_is_ignored_but_isomers_are_distinct(self):
        ethanol = self.by_name["C2H5OH"]["graph"]
        relabeled = nx.relabel_nodes(ethanol, {i: 100-i for i in ethanol})
        self.assertEqual([r["species_key"] for r in self.reference.match(relabeled)], ["C2H5OH"])
        self.assertEqual([r["species_key"] for r in self.reference.match(self.by_name["CH3CHO"]["graph"])],
                         ["CH3CHO"])
        self.assertEqual([r["species_key"] for r in self.reference.match(self.by_name["C2H4O"]["graph"])],
                         ["C2H4O"])
        changed = self.by_name["C2H2"]["graph"].copy()
        u, v = next((u, v) for u, v, d in changed.edges(data=True) if d["order"] == 3)
        changed.edges[u, v]["order"] = 1
        self.assertNotIn("C2H2", [r["species_key"] for r in self.reference.match(changed)])

    def test_unknown_charge_does_not_invent_neutrality_and_explicit_charge_is_checked(self):
        co = self.by_name["CO"]["graph"].copy()
        for _, node in co.nodes(data=True):
            node["formal_charge"] = None
        self.assertEqual([r["species_key"] for r in self.reference.match(co)], ["CO"])
        for _, node in co.nodes(data=True):
            node["formal_charge"] = 0
        self.assertEqual(self.reference.match(co), [])

    def test_electronic_states_remain_ambiguous(self):
        methane_radical = self.by_name["CH2"]["graph"]
        self.assertEqual({r["species_key"] for r in self.reference.match(methane_radical)}, {"CH2", "CH2(S)"})
        record = self.prediction(methane_radical, "M1")
        result = StageIIIComparison.compare([record], self.reference)[0]
        self.assertFalse(result["electronic_state_resolved"])
        self.assertEqual(set(result["ffcmii_species_keys"]), {"CH2", "CH2(S)"})

    def test_green_red_blue_order_and_original_order_preserved_within_groups(self):
        unknown = WorkbookGraphs.make_graph([{"id": 0, "symbol": "C"}, {"id": 1, "symbol": "C"}],
                                           [{"source": 0, "target": 1, "order": 1}])
        training = self.prediction(self.by_name["H2O"]["graph"], "T1")
        training.update(category="reference_match", in_species_training_reference=True, species_key="H2O")
        records = [training, self.prediction(unknown, "U1"),
                   self.prediction(self.by_name["CH3OH"]["graph"], "K1"),
                   self.prediction(self.by_name["OH"]["graph"], "K2"),
                   self.prediction(unknown, "U2")]
        result = StageIIIComparison.compare(records, self.reference)
        self.assertEqual([r["structure_id"] for r in result], ["K1", "K2", "U1", "U2", "T1"])
        self.assertEqual([r["structure_id"] for r in records], ["T1", "U1", "K1", "K2", "U2"])
        self.assertEqual([r["display_group"] for r in result],
                         ["ffcmii_only", "ffcmii_only", "not_in_ffcmii", "not_in_ffcmii", "training"])
        plot = ComparisonPlot({})
        self.assertEqual(plot.panel(result[0]).getpixel((5, 5)), (213, 242, 216))
        self.assertEqual(plot.panel(result[2]).getpixel((5, 5)), (250, 218, 221))
        self.assertEqual(plot.panel(result[-1]).getpixel((5, 5)), (217, 234, 254))
        # Training membership takes precedence even if a training graph is
        # absent from FFCMII. Older outputs can use the category as fallback.
        fallback = self.prediction(unknown, "T2")
        fallback["category"] = "reference_match"
        self.assertEqual(StageIIIComparison.compare([fallback], self.reference)[0]["display_group"], "training")

    def test_incomplete_and_malformed_graphs_rejected(self):
        record = self.prediction(self.by_name["OH"]["graph"], "M1")
        record["termination"] = "step_limit"
        with self.assertRaises(ValueError):
            StageIIIComparison.compare([record], self.reference)
        with self.assertRaises(ValueError):
            WorkbookGraphs.make_graph([{"id": 0, "symbol": "C"}],
                                      [{"source": 0, "target": 0, "order": 1}])

    def test_end_to_end_outputs_match_sorted_inventory_and_inputs_are_unchanged(self):
        workbook_hash = hashlib.sha256(Path(REFERENCE_WORKBOOK).read_bytes()).hexdigest()
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = root / "predictions.json"
            records = [self.prediction(self.by_name["OH"]["graph"], "M1"),
                       self.prediction(self.by_name["CH3OH"]["graph"], "M2")]
            records[0].update(in_species_training_reference=True, category="reference_match", species_key="OH")
            source.write_text(json.dumps(records))
            original = source.read_bytes()
            output = root / "output"
            cfg = {"reference_workbook": REFERENCE_WORKBOOK, "reference_sheet": "Graph definitions",
                   "predictions_path": str(source), "compact_json_source": COMPACT_JSON_SOURCE,
                   "output_folder": str(output), "overview_columns": 5,
                   "structures_per_page": 2, "page_columns": 5}
            (output / "pages").mkdir(parents=True)
            (output / "pages" / "structures_999.png").write_bytes(b"stale page")
            summary = StageIIIComparison(cfg).run()
            compared = json.loads((output / "compared_structures.json").read_text())
            self.assertEqual(summary["total_structures"], 2)
            self.assertEqual(summary["ffcmii_matched_structures"], 2)
            self.assertEqual(summary["display_group_counts"],
                             {"ffcmii_only": 1, "not_in_ffcmii": 0, "training": 1})
            self.assertEqual(summary["plot_order"], ["M2", "M1"])
            self.assertEqual(summary["plot_order"], [r["structure_id"] for r in compared])
            self.assertFalse((output / "pages" / "structures_999.png").exists())
            with Image.open(output / "final_structures.png") as image:
                self.assertEqual(image.size, (1200, 322))
                self.assertEqual(image.getpixel((4, 112)), (213, 242, 216))
                self.assertEqual(image.getpixel((244, 112)), (217, 234, 254))
            self.assertEqual(source.read_bytes(), original)
        self.assertEqual(hashlib.sha256(Path(REFERENCE_WORKBOOK).read_bytes()).hexdigest(), workbook_hash)


# Parameters and class-only loading; no comparison or Stage II exploration runs.
SOURCE = "compare_stage3.py"
REFERENCE_WORKBOOK = "../FFCMII_species_reference.xlsx"
COMPACT_JSON_SOURCE = "../stageI/compact_json.py"
tree = ast.parse(Path(SOURCE).read_text())
scope = {}
exec(compile(ast.Module(body=[n for n in tree.body if isinstance(
    n, (ast.Import, ast.ImportFrom, ast.ClassDef))], type_ignores=[]), SOURCE, "exec"), scope)
WorkbookGraphs = scope["WorkbookGraphs"]
ComparisonPlot = scope["ComparisonPlot"]
StageIIIComparison = scope["StageIIIComparison"]
unittest.main(verbosity=2)
