"""Stage VI: classify both Stage V campaigns using Stage III's FFCM2 rules."""
import ast
import json
from pathlib import Path


class StageVIComparison:
    """Keep PUCT and direct classifications separate, with the same reference."""

    def __init__(self, config):
        self.config = dict(config)

    def run(self):
        cfg = self.config
        path = Path(cfg["stage3_directory"]) / "compare_stage3.py"
        tree = ast.parse(path.read_text(encoding="utf-8"))
        tree.body = [n for n in tree.body if isinstance(n, (ast.Import, ast.ImportFrom, ast.ClassDef))]
        scope = {"__file__": str(path.resolve()), "__name__": "stage6_definitions"}
        exec(compile(tree, str(path), "exec"), scope)
        source = Path(cfg["prediction_folder"])
        # Check both campaigns before writing results. Never compare mismatched
        # model/reference inputs as if they were the same experiment.
        inputs = {}
        for method in ("puct", "direct"):
            folder = source / method
            inputs[method] = json.loads((folder / "config.json").read_text())
            if not (folder / "unique_structures.json").is_file():
                raise FileNotFoundError(f"Run Stage V first: {folder / 'unique_structures.json'}")
        for key in ("checkpoint_sha256", "reference_sha256"):
            if inputs["puct"][key] != inputs["direct"][key]:
                raise ValueError(f"Stage V campaigns use different {key}")
        results = {}
        for method in ("puct", "direct"):
            options = {"predictions_path": str(source / method / "unique_structures.json"),
                       "reference_workbook": cfg["reference_workbook"],
                       "reference_sheet": cfg["reference_sheet"],
                       "compact_json_source": cfg["compact_json_source"],
                       "output_folder": str(Path(cfg["output_folder"]) / method),
                       "overview_columns": cfg["overview_columns"]}
            print(f"STAGE VI | method={method}", flush=True)
            results[method] = scope["StageIIIComparison"](options).run()
        return results


# ==========================================
# PARAMETERS — run from self_improving/stageVI
# ==========================================
STAGE3_DIRECTORY = "../stageIII"
PREDICTION_FOLDER = "../stageV/output"
REFERENCE_WORKBOOK = "../stageIII/FFCM2_CHO_reference.xlsx"
REFERENCE_SHEET = "Graph definitions"
COMPACT_JSON_SOURCE = "../stageI/compact_json.py"
OUTPUT_FOLDER = "output"
OVERVIEW_COLUMNS = 10

comparison_config = {
    "stage3_directory": STAGE3_DIRECTORY, "prediction_folder": PREDICTION_FOLDER,
    "reference_workbook": REFERENCE_WORKBOOK, "reference_sheet": REFERENCE_SHEET,
    "compact_json_source": COMPACT_JSON_SOURCE, "output_folder": OUTPUT_FOLDER,
    "overview_columns": OVERVIEW_COLUMNS,
}
comparison = StageVIComparison(comparison_config)
comparison.run()
