"""Stage V: PUCT and direct generation from the selected Stage IV checkpoint."""
import ast
import hashlib
from pathlib import Path


class StageVPrediction:
    """Run two frozen-policy campaigns with the same checkpoint and reference."""

    @staticmethod
    def definitions(path):
        # Existing scripts intentionally execute at the bottom. Load definitions
        # only, so reusing their classes cannot start training or another run.
        path = Path(path)
        tree = ast.parse(path.read_text(encoding="utf-8"))
        tree.body = [n for n in tree.body if isinstance(n, (ast.Import, ast.ImportFrom, ast.ClassDef))]
        scope = {"__file__": str(path.resolve()), "__name__": "stage5_definitions"}
        exec(compile(tree, str(path), "exec"), scope)
        return scope

    def __init__(self, config):
        self.config = dict(config)

    def run(self):
        cfg = self.config
        stage2 = Path(cfg["stage2_directory"])
        api = self.definitions(stage2 / "explore_stage2.py")
        direct = self.definitions(stage2 / "stageII_direct_generate.py")
        # Both methods reuse graph features across composition contexts.
        api["MarginalGrowthPolicy"] = direct["DirectMarginalPolicy"]
        protected = [Path(cfg[k]) for k in ("checkpoint_path", "reference_path")]
        hashes = {p: hashlib.sha256(p.read_bytes()).hexdigest() for p in protected}
        results = {}
        for method in ("puct", "direct"):
            if any(hashlib.sha256(p.read_bytes()).hexdigest() != h for p, h in hashes.items()):
                raise RuntimeError("Checkpoint/reference changed during Stage V; rerun with fixed inputs")
            options = {**cfg, "output_folder": str(Path(cfg["output_folder"]) / method),
                       "search_method": "puct" if method == "puct" else "sampling"}
            if method == "direct":
                options["puct_uniform_fraction"] = 0.0
                options["temperature"] = 1.0
            print(f"STAGE V | method={method} | checkpoint={cfg['checkpoint_path']}", flush=True)
            results[method] = api["ExplorationCampaign"](options).run()
        if any(hashlib.sha256(p.read_bytes()).hexdigest() != h for p, h in hashes.items()):
            raise RuntimeError("Checkpoint/reference changed during Stage V")
        return results


# ==========================================
# PARAMETERS — run from self_improving/stageV
# ==========================================
STAGE1_DIRECTORY = "../stageI"
STAGE2_DIRECTORY = "../stageII"
CHECKPOINT_PATH = "../stageIV/output/trained_growth_gnn.pt"
REFERENCE_JSON_PATH = "../stageIII/species_update/expanded_species_graphs.json"
OUTPUT_FOLDER = "output"  # Separate puct/ and direct/ subfolders; replaced on rerun.
GENERATION_MODE = "free"  # Network chooses atom counts and STOP.
COMPOSITIONS = None
EXPLORATION_RUNS = 100
ATTEMPTS_PER_RUN = 1000  # Budget PER method: runs × attempts in free mode.
SAMPLES_PER_COMPOSITION = 100  # Used only for formula mode.
TEMPERATURE = 1.0
PUCT_C = 2.0
PUCT_UNIFORM_FRACTION = 0.25  # Direct sampling always uses zero.
PUCT_REFERENCE_REWARD = 0.5
PUCT_CANDIDATE_REWARD = 1.0
SEED = 24680
BATCH_SIZE = 100
CPU_THREADS = 1  # Stage II inference runs on CPU; weights stay frozen.
SAVE_MEDIA = True
PLOT_DPI = 150
VIDEO_FPS = 2

prediction_config = {
    "stage1_directory": STAGE1_DIRECTORY, "stage2_directory": STAGE2_DIRECTORY,
    "checkpoint_path": CHECKPOINT_PATH, "reference_path": REFERENCE_JSON_PATH,
    "output_folder": OUTPUT_FOLDER, "generation_mode": GENERATION_MODE,
    "compositions": COMPOSITIONS, "exploration_runs": EXPLORATION_RUNS,
    "attempts_per_run": ATTEMPTS_PER_RUN, "samples_per_composition": SAMPLES_PER_COMPOSITION,
    "temperature": TEMPERATURE, "puct_c": PUCT_C,
    "puct_uniform_fraction": PUCT_UNIFORM_FRACTION,
    "puct_require_composition": GENERATION_MODE == "formula",
    "puct_reference_reward": PUCT_REFERENCE_REWARD, "puct_candidate_reward": PUCT_CANDIDATE_REWARD,
    "seed": SEED, "batch_size": BATCH_SIZE, "threads": CPU_THREADS,
    "save_media": SAVE_MEDIA, "plot_dpi": PLOT_DPI, "video_fps": VIDEO_FPS,
}
prediction = StageVPrediction(prediction_config)
prediction.run()
