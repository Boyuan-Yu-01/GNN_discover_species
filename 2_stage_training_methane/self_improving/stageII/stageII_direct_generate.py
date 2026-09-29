"""Direct network sampling with Stage II's inventories, PNG and MP4 outputs.

Run from self_improving/stageII: python stageII_direct_generate.py
"""
import ast
import os
from pathlib import Path


class DirectMarginalPolicy:
    """Average context probabilities while convolving each partial graph once."""

    def __init__(self, model, conditions):
        self.model = model
        self.conditions = list(conditions)
        if not self.conditions:
            raise ValueError("Free growth requires at least one reference context")

    def encode(self, environments, unused_conditions):
        return (environments,)

    def __call__(self, environments):
        import math
        import torch
        logs = self.model.log_probs_over_contexts(environments, self.conditions, 256)
        return torch.logsumexp(logs, dim=1) - math.log(len(self.conditions))


class DirectGeneration:
    """Reuse Stage II sampling without executing its bottom-level PUCT calls."""

    def __init__(self, config):
        path = Path(__file__).with_name("explore_stage2.py")
        tree = ast.parse(path.read_text(encoding="utf-8"))
        tree.body = [node for node in tree.body
                     if isinstance(node, (ast.Import, ast.ImportFrom, ast.ClassDef))]
        scope = {"__file__": str(path), "__name__": "direct_generation_stage2"}
        exec(compile(tree, str(path), "exec"), scope)
        scope["MarginalGrowthPolicy"] = DirectMarginalPolicy
        # Sample masked network probabilities: no tree, rewards or uniform mixing.
        self.config = {**config, "search_method": "sampling", "puct_uniform_fraction": 0.0}
        self.campaign = scope["ExplorationCampaign"](self.config)

    def run(self):
        return self.campaign.run()


# ==========================================
# PARAMETERS — relative to your working directory
# ==========================================
STAGE1_DIRECTORY = "../stageI"
# Same model/reference as PUCT for comparison. For Stage IV, change BOTH paths
# to its selected checkpoint and the expanded species reference.
CHECKPOINT_PATH = "../stageI/output_stage1/trained_growth_gnn.pt"
REFERENCE_JSON_PATH = "../stageI/species_graphs.json"
OUTPUT_FOLDER = os.environ.get("DIRECT_OUTPUT_FOLDER", "output_direct")
# Free growth averages action probabilities across reference formula contexts;
# the network chooses atom counts and STOP, without a prescribed formula.
GENERATION_MODE = "free"
COMPOSITIONS = None
EXPLORATION_RUNS = int(os.environ.get("DIRECT_RUNS", "100"))
ATTEMPTS_PER_RUN = int(os.environ.get("DIRECT_ATTEMPTS", "1000"))
SAMPLES_PER_COMPOSITION = 100  # Only used in formula mode.
TEMPERATURE = 1.0  # Preserve the learned action distribution.
SEED = 24680
BATCH_SIZE = 100
CPU_THREADS = 1  # Same CPU inference implementation as Stage II.
SAVE_MEDIA = True
PLOT_DPI = 150
VIDEO_FPS = 2

exploration_config = {
    "stage1_directory": STAGE1_DIRECTORY, "checkpoint_path": CHECKPOINT_PATH,
    "reference_path": REFERENCE_JSON_PATH, "output_folder": OUTPUT_FOLDER,
    "generation_mode": GENERATION_MODE, "compositions": COMPOSITIONS,
    "exploration_runs": EXPLORATION_RUNS, "attempts_per_run": ATTEMPTS_PER_RUN,
    "samples_per_composition": SAMPLES_PER_COMPOSITION,
    "temperature": TEMPERATURE, "seed": SEED, "batch_size": BATCH_SIZE,
    "threads": CPU_THREADS, "save_media": SAVE_MEDIA,
    "plot_dpi": PLOT_DPI, "video_fps": VIDEO_FPS,
}
explorer = DirectGeneration(exploration_config)
explorer.run()
