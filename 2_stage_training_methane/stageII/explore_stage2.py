"""Stage II: sample a fixed stage I network and inventory distinct structures.

Run from stageII. Classes first, editable parameters below, then direct calls.
"""
import ast
import hashlib
import json
import math
import os
import time
from collections import Counter
from datetime import datetime
from pathlib import Path

import torch


class Stage1Loader:
    """Reuse the actual stage I classes without executing its training calls."""

    @staticmethod
    def definitions(path, scope=None):
        # Stage I deliberately has direct execution. Retain imports/classes only,
        # just as its test script does; no training parameters or calls execute.
        path = Path(path)
        tree = ast.parse(path.read_text(encoding="utf-8"))
        nodes = [node for node in tree.body
                 if isinstance(node, (ast.Import, ast.ImportFrom, ast.ClassDef))]
        # compact_json is loaded explicitly from the configured stage I folder.
        nodes = [node for node in nodes
                 if not (isinstance(node, ast.ImportFrom) and node.module == "compact_json")]
        namespace = {} if scope is None else dict(scope)
        namespace.update(__name__="stageII_loaded_definitions", __file__=str(path))
        exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), "exec"), namespace)
        return namespace

    @classmethod
    def load(cls, directory):
        directory = Path(directory)
        compact = cls.definitions(directory / "compact_json.py")
        return cls.definitions(directory / "train_stage1.py",
                               {"CompactJSON": compact["CompactJSON"]})


class ExplorationMedia:
    """Draw all distinct successful structures and replay their first growth traces."""

    def __init__(self, api, symbols, valency, max_atoms, config):
        # Use a noninteractive canvas: files can be rendered without a GUI.
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import networkx as nx
        self.plt, self.nx = plt, nx
        self.api, self.symbols, self.valency = api, symbols, valency
        self.max_atoms, self.config = max_atoms, config

    def layout(self, record):
        graph = self.nx.Graph()
        graph.add_nodes_from(n["id"] for n in record["graph"]["nodes"])
        graph.add_edges_from((e["source"], e["target"]) for e in record["graph"]["edges"])
        return self.nx.spring_layout(graph, seed=42)

    def draw(self, ax, record, graph_record, positions, caption=""):
        import numpy as np
        ax.clear()
        candidate = record["category"] == "new_candidate"
        accent = "#c76b13" if candidate else "#16847a"
        ax.set_facecolor("#fff8ef" if candidate else "#f2faf8")
        for spine in ax.spines.values():
            spine.set_color(accent)
        # Draw bond order with parallel strokes; single/double/triple remain explicit.
        for edge in graph_record["edges"]:
            a, b = positions[edge["source"]], positions[edge["target"]]
            delta = b-a
            normal = np.array([-delta[1], delta[0]]) / max(float(np.linalg.norm(delta)), 1e-9)
            order = edge["order"]
            for i in range(order):
                shift = normal * (i-(order-1)/2) * 0.045
                ax.plot([a[0]+shift[0], b[0]+shift[0]],
                        [a[1]+shift[1], b[1]+shift[1]], color="#66717c", lw=1.8, zorder=1)
        colors = {"C": "#566273", "H": "#bcdcff", "O": "#e16c68"}
        for node in graph_record["nodes"]:
            x, y = positions[node["id"]]
            ax.scatter([x], [y], s=290, c=colors[node["symbol"]],
                       edgecolors="white", linewidths=1, zorder=2)
            ax.text(x, y, node["symbol"], ha="center", va="center", fontsize=9,
                    color="white" if node["symbol"] == "C" else "#172b3a", zorder=3)
        c, h, o = record["requested_C_H_O"]
        name = "Candidate" if candidate else record["species_key"]
        ax.set_title(f"{record['structure_id']}  {name} | C={c} H={h} O={o}"
                     f"\nCount: {record['occurrences']}  |  {record['frequency_within_requested_composition']:.1%}",
                     fontsize=9, color=accent, pad=8)
        ax.set_xlabel(caption, fontsize=8, labelpad=6)
        ax.set_xlim(-1.35, 1.35)
        ax.set_ylim(-1.35, 1.35)
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])

    def save(self, records, output, summary, log):
        from matplotlib.animation import FFMpegWriter, writers
        if not writers.is_available("ffmpeg"):
            raise RuntimeError("MP4 output requires ffmpeg on PATH")
        # Candidates appear first for convenient inspection; IDs still match JSON.
        records = sorted(records, key=lambda r: (r["category"] != "new_candidate", r["structure_id"]))
        positions = [self.layout(r) for r in records]
        columns = 5
        rows = max(1, math.ceil(len(records)/columns))
        fig, axes = self.plt.subplots(rows, columns, figsize=(18, rows*3.4+1.1), squeeze=False)
        try:
            for index, ax in enumerate(axes.flat):
                if index < len(records):
                    self.draw(ax, records[index], records[index]["graph"], positions[index])
                else:
                    ax.set_axis_off()
            fig.suptitle(
                f"Exploration II | {len(records)} distinct structures from {summary['total_attempts']:,} attempts"
                "\nOrange: new candidates (unvalidated) | Teal: reference matches | C: grey, H: blue, O: red",
                fontsize=15, y=0.99)
            fig.tight_layout(rect=(0, 0, 1, 0.955))
            fig.savefig(output / "final_structures.png", dpi=self.config.get("plot_dpi", 150))
        finally:
            self.plt.close(fig)
        log.info(f"Saved final structure plot: {output / 'final_structures.png'}")
        # Six structures per page keep individual growth steps readable at 1440x900.
        fig, axes = self.plt.subplots(2, 3, figsize=(14.4, 9), squeeze=False)
        writer = FFMpegWriter(fps=self.config.get("video_fps", 2), codec="libx264",
                             extra_args=["-pix_fmt", "yuv420p", "-crf", "20"])
        try:
            with writer.saving(fig, str(output / "exploration_growth.mp4"), dpi=100):
                if not records:
                    for ax in axes.flat:
                        ax.set_axis_off()
                    fig.suptitle("No successful structures in this exploration")
                    writer.grab_frame()
                for start in range(0, len(records), 6):
                    page = records[start:start+6]
                    envs = [self.api["AdvancedMoleculeEnv"](
                        self.symbols, self.valency, self.max_atoms) for _ in page]
                    steps = max(len(r["actions"]) for r in page)
                    for step in range(steps+1):
                        for index, ax in enumerate(axes.flat):
                            if index >= len(page):
                                ax.set_axis_off()
                                continue
                            ax.set_axis_on()
                            record, env = page[index], envs[index]
                            if 0 < step <= len(record["actions"]):
                                action = tuple(record["actions"][step-1]["action"])
                                env.step(action)
                            caption = ("Empty graph" if step == 0 else
                                       "STOP — complete" if env.terminated else
                                       f"Step {step}: {record['actions'][step-1]['action']}")
                            self.draw(ax, record, env.record(), positions[start+index], caption)
                        fig.suptitle(
                            f"Exploration II | growth of distinct structures | page {start//6+1}/{math.ceil(len(records)/6)}"
                            "\nFirst successful trace per structure; fixed final positions. Orange = unvalidated candidate.",
                            fontsize=14, y=0.98)
                        fig.tight_layout(rect=(0, 0.01, 1, 0.93))
                        writer.grab_frame()
                    # Hold complete structures for two seconds before changing page.
                    for _ in range(2*self.config.get("video_fps", 2)):
                        writer.grab_frame()
                    log.info(f"Rendered video page {start//6+1}/{math.ceil(len(records)/6)}")
        finally:
            self.plt.close(fig)
        log.info(f"Saved growth animation: {output / 'exploration_growth.mp4'}")


class Stage2Explorer:
    """Generate, classify, and deduplicate graphs; no optimizer or weight updates."""

    def __init__(self, config):
        self.config = dict(config)
        self.api = Stage1Loader.load(config["stage1_directory"])
        self.json = self.api["CompactJSON"]
        self.output = Path(config["output_folder"])
        self.output.mkdir(parents=True, exist_ok=False)
        self.log = self.api["TrainingLogger"](self.output / "exploration.log")

    @staticmethod
    def validate_conditions(conditions, max_atoms):
        result = []
        for condition in conditions:
            condition = tuple(condition)
            if (len(condition) != 3 or any(type(n) is not int or n < 0 for n in condition)
                    or not 1 <= sum(condition) <= max_atoms):
                raise ValueError(f"Invalid C,H,O composition: {condition}; atom limit={max_atoms}")
            if condition not in result:
                result.append(condition)
        if not result:
            raise ValueError("At least one composition is required")
        return result

    @staticmethod
    def classify(env, condition, reference_match):
        # A reference graph with the WRONG formula is still a failed request.
        if not env.terminated:
            return "failed_step_limit"
        if env.composition() != tuple(condition):
            return "failed_composition"
        return "reference_match" if reference_match is not None else "new_candidate"

    def structure_record(self, env):
        graph = env.to_graph()
        base = [u for u in graph if self.symbols[graph.nodes[u]["atom"]] != "H"]
        pure_h = not base
        if pure_h:
            base = list(graph)
        mapping = {u: i for i, u in enumerate(base)}
        return {
            "base_atom_indices": {i: self.symbols[graph.nodes[u]["atom"]]
                                  for u, i in mapping.items()},
            "base_bonds": [[mapping[u], mapping[v], d["order"]]
                           for u, v, d in graph.edges(data=True) if u in mapping and v in mapping],
            "attached_h_per_base_atom": [] if pure_h else [
                sum(self.symbols[graph.nodes[v]["atom"]] == "H" for v in graph.neighbors(u))
                for u in base],
            # Unfilled valence is diagnostic, not a confirmed radical/electronic state.
            "unused_valence_per_atom": [
                self.valency[a] - env.current_bonds[u] for u, a in enumerate(env.node_types)],
            "graph": env.record(),
        }

    @staticmethod
    def find_structure(graph, condition, buckets, references):
        # Formula narrows the search; exact atom/bond isomorphism decides identity.
        for old_graph, record in buckets.get(tuple(condition), []):
            if references.isomorphic(graph, old_graph):
                return record
        return None

    def run(self):
        try:
            return self.explore()
        except Exception:
            self.log.exception("EXPLORATION FAILED; existing output is retained.")
            raise
        finally:
            self.log.close()

    def explore(self):
        cfg = self.config
        for key in ("samples_per_composition", "batch_size", "threads"):
            if type(cfg[key]) is not int or cfg[key] < 1:
                raise ValueError(f"{key} must be a positive integer")
        if not math.isfinite(cfg["temperature"]) or cfg["temperature"] <= 0:
            raise ValueError("temperature must be positive and finite")
        torch.set_num_threads(cfg["threads"])
        torch.use_deterministic_algorithms(True)
        torch.manual_seed(cfg["seed"])
        checkpoint_path = Path(cfg["checkpoint_path"])
        # Only tensor/basic-data checkpoint loading; always perform exploration on CPU.
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        trained = checkpoint["config"]
        self.symbols = trained["atom_symbols"]
        self.valency = trained["max_valency"]
        max_atoms = trained["max_atoms"]
        max_steps = trained["max_steps"]
        space = self.api["ActionSpace"](max_atoms, self.symbols)
        if [tuple(a) for a in checkpoint["action_space"]] != space.actions:
            raise ValueError("Checkpoint action ordering differs from the stage I code")
        references = self.api["ReferenceSpecies"](
            cfg["reference_path"], trained["excluded_species"], self.symbols, self.valency, max_atoms)
        # Exact source hashes are recorded. Formatting differences can also change a hash.
        source_matches = references.source_hash == checkpoint["source_sha256"]
        if not source_matches:
            self.log.info("REFERENCE HASH DIFFERS from training; matches use the supplied reference file.")
        conditions = self.validate_conditions(
            list(references.conditions) if cfg["compositions"] is None else cfg["compositions"], max_atoms)
        model = self.api["GrowthGNN"](trained["hidden_dims"], space, self.symbols, self.valency)
        model.load_state_dict(checkpoint["model_state_dict"], strict=True)
        model.eval()
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        generator = torch.Generator().manual_seed(cfg["seed"])
        provenance = {
            **cfg, "resolved_compositions_C_H_O": conditions, "checkpoint_epoch": checkpoint["epoch"],
            "checkpoint_sha256": hashlib.sha256(checkpoint_path.read_bytes()).hexdigest(),
            "reference_sha256": references.source_hash, "reference_matches_training_hash": source_matches,
            "stage1_code_sha256": hashlib.sha256(
                (Path(cfg["stage1_directory"]) / "train_stage1.py").read_bytes()).hexdigest(),
            "max_atoms": max_atoms, "max_steps": max_steps,
        }
        self.json.write(self.output / "config.json", provenance)
        self.log.info(f"STAGE II | fixed network | checkpoint={checkpoint_path} | epoch={checkpoint['epoch']}")
        self.log.info(f"Compositions={len(conditions)} | samples/composition={cfg['samples_per_composition']}"
                      f" | temperature={cfg['temperature']} | seed={cfg['seed']}")
        self.log.info("New candidates satisfy the requested formula and construction masks;"
                      " they are not confirmed stable or chemically important species.")
        attempts, unique, buckets, summaries = [], [], {}, []
        started = time.monotonic()
        for condition in conditions:
            counts = Counter()
            for start in range(0, cfg["samples_per_composition"], cfg["batch_size"]):
                size = min(cfg["batch_size"], cfg["samples_per_composition"] - start)
                environments = [self.api["AdvancedMoleculeEnv"](
                    self.symbols, self.valency, max_atoms) for _ in range(size)]
                traces = [[] for _ in environments]
                with torch.inference_mode():
                    for step in range(max_steps):
                        active = [i for i, env in enumerate(environments) if not env.terminated]
                        if not active:
                            break
                        inputs = model.encode([environments[i] for i in active], [condition] * len(active))
                        probabilities = (model(*inputs) / cfg["temperature"]).softmax(-1)
                        if not torch.isfinite(probabilities).all():
                            raise RuntimeError("Non-finite sampling probabilities")
                        choices = torch.multinomial(probabilities, 1, generator=generator).squeeze(1)
                        for row, index in enumerate(active):
                            action = space.actions[int(choices[row])]
                            traces[index].append({
                                "step": step + 1, "action": list(action),
                                "probability": float(probabilities[row, choices[row]])})
                            environments[index].step(action)
                for env, trace in zip(environments, traces):
                    match = references.match(env.to_graph())
                    category = self.classify(env, condition, match)
                    counts[category] += 1
                    record = {
                        "attempt": len(attempts) + 1, "requested_C_H_O": list(condition),
                        "generated_C_H_O": list(env.composition()),
                        "category": category, "species_key": match or "Unknown",
                        "in_species_training_reference": match is not None,
                        "termination": "STOP" if env.terminated else "step_limit",
                        "structure_id": None, **self.structure_record(env),
                        "actions": trace,
                    }
                    if category in ("reference_match", "new_candidate"):
                        found = self.find_structure(env.to_graph(), condition, buckets, references)
                        if found is None:
                            found = {**record, "structure_id": f"M{len(unique)+1:05d}",
                                     "occurrences": 0, "first_attempt": record["attempt"]}
                            unique.append(found)
                            buckets.setdefault(tuple(condition), []).append((env.to_graph(), found))
                        found["occurrences"] += 1
                        record["structure_id"] = found["structure_id"]
                    attempts.append(record)
            summary = {"requested_C_H_O": list(condition),
                       "seen_composition_in_training": condition in references.conditions,
                       "attempts": cfg["samples_per_composition"],
                       **{key: counts[key] for key in ("reference_match", "new_candidate",
                                                      "failed_composition", "failed_step_limit")},
                       "unique_structures": len(buckets.get(condition, []))}
            summaries.append(summary)
            self.log.info(f"C,H,O={condition} | reference={counts['reference_match']}"
                          f" | candidate samples={counts['new_candidate']}"
                          f" | wrong formula={counts['failed_composition']}"
                          f" | step limit={counts['failed_step_limit']}"
                          f" | unique={summary['unique_structures']}")
        for record in unique:
            record["frequency_within_requested_composition"] = (
                record["occurrences"] / cfg["samples_per_composition"])
        matches = [r for r in unique if r["category"] == "reference_match"]
        candidates = [r for r in unique if r["category"] == "new_candidate"]
        summary = {
            "total_attempts": len(attempts),
            "counts": dict(Counter(r["category"] for r in attempts)),
            "unique_reference_structures": len(matches), "unique_new_candidates": len(candidates),
            "per_composition": summaries,
        }
        for filename, data in [
            ("attempts.json", attempts), ("unique_structures.json", unique),
            ("reference_matches.json", matches), ("new_candidates.json", candidates),
            ("summary.json", summary),
        ]:
            self.json.write(self.output / filename, data)
        if cfg.get("save_media", True):
            ExplorationMedia(self.api, self.symbols, self.valency, max_atoms, cfg).save(
                unique, self.output, summary, self.log)
        self.log.info(f"FINISHED | attempts={len(attempts)} | reference structures={len(matches)}"
                      f" | new candidate structures={len(candidates)}"
                      f" | elapsed={time.monotonic()-started:.1f}s")
        self.log.info(f"Saved exploration outputs: {self.output}")
        return summary


class ExplorationCampaign:
    """Run multiple independent seeds and merge results using graph isomorphism."""

    def __init__(self, config):
        self.config = dict(config)

    def run(self):
        cfg = self.config
        runs = cfg["exploration_runs"]
        if type(runs) is not int or runs < 1:
            raise ValueError("exploration_runs must be a positive integer")
        output = Path(cfg["output_folder"])
        output.mkdir(parents=True, exist_ok=False)
        api = Stage1Loader.load(cfg["stage1_directory"])
        log = api["TrainingLogger"](output / "exploration.log")
        try:
            return self.explore(output, api, log)
        except Exception:
            log.exception("CAMPAIGN FAILED; completed runs remain available.")
            raise
        finally:
            log.close()

    def explore(self, output, api, log):
        cfg = self.config
        attempts, unique, buckets, run_summaries = [], [], {}, []
        condition_counts = {}
        references = None
        first_config = None
        for index in range(cfg["exploration_runs"]):
            run_id = index + 1
            run_output = output / "runs" / f"run_{run_id:03d}"
            run_config = {**cfg, "output_folder": str(run_output),
                          "seed": cfg["seed"] + index * 1000, "save_media": False}
            log.info(f"CAMPAIGN run={run_id}/{cfg['exploration_runs']} | seed={run_config['seed']}")
            explorer = Stage2Explorer(run_config)
            result = explorer.run()
            saved_config = json.loads((run_output / "config.json").read_text())
            if first_config is None:
                first_config = saved_config
                checkpoint = torch.load(cfg["checkpoint_path"], map_location="cpu", weights_only=True)
                trained = checkpoint["config"]
                references = api["ReferenceSpecies"](
                    cfg["reference_path"], trained["excluded_species"],
                    trained["atom_symbols"], trained["max_valency"], trained["max_atoms"])
            else:
                # Refuse to merge runs made with changing inputs.
                for key in ("checkpoint_sha256", "reference_sha256", "stage1_code_sha256",
                            "resolved_compositions_C_H_O"):
                    if saved_config[key] != first_config[key]:
                        raise ValueError(f"Campaign input changed between runs: {key}")
            for record in json.loads((run_output / "attempts.json").read_text()):
                record["run"] = run_id
                record["run_attempt"] = record["attempt"]
                record["attempt"] = len(attempts) + 1
                record["structure_id"] = None
                condition = tuple(record["requested_C_H_O"])
                condition_counts.setdefault(condition, Counter())[record["category"]] += 1
                if record["category"] in ("reference_match", "new_candidate"):
                    graph = api["nx"].Graph()
                    graph.add_nodes_from((n["id"], {"atom": n["atom"]}) for n in record["graph"]["nodes"])
                    graph.add_edges_from((e["source"], e["target"], {"order": e["order"]})
                                         for e in record["graph"]["edges"])
                    found = Stage2Explorer.find_structure(graph, condition, buckets, references)
                    if found is None:
                        found = {**record, "structure_id": f"M{len(unique)+1:05d}",
                                 "first_attempt": record["attempt"], "occurrences": 0,
                                 "runs_observed": []}
                        unique.append(found)
                        buckets.setdefault(condition, []).append((graph, found))
                    found["occurrences"] += 1
                    if run_id not in found["runs_observed"]:
                        found["runs_observed"].append(run_id)
                    record["structure_id"] = found["structure_id"]
                attempts.append(record)
            run_summaries.append({"run": run_id, "seed": run_config["seed"], **result})
            log.info(f"MERGED run={run_id} | attempts={len(attempts)}"
                     f" | distinct reference={sum(r['category']=='reference_match' for r in unique)}"
                     f" | distinct candidates={sum(r['category']=='new_candidate' for r in unique)}")
        for record in unique:
            record["frequency_within_requested_composition"] = (
                record["occurrences"] / sum(condition_counts[tuple(record["requested_C_H_O"])].values()))
        matches = [r for r in unique if r["category"] == "reference_match"]
        candidates = [r for r in unique if r["category"] == "new_candidate"]
        summary = {
            "exploration_runs": cfg["exploration_runs"], "total_attempts": len(attempts),
            "counts": dict(Counter(r["category"] for r in attempts)),
            "unique_reference_structures": len(matches), "unique_new_candidates": len(candidates),
            "per_composition": [
                {"requested_C_H_O": list(condition), "attempts": sum(counts.values()),
                 "seen_composition_in_training": condition in references.conditions,
                 **{key: counts[key] for key in ("reference_match", "new_candidate",
                                                 "failed_composition", "failed_step_limit")},
                 "unique_structures": len(buckets.get(condition, []))}
                for condition, counts in condition_counts.items()],
            "runs": run_summaries,
        }
        provenance = {**first_config, **cfg,
                      "run_seeds": [r["seed"] for r in run_summaries],
                      "total_samples_per_composition": cfg["samples_per_composition"] * cfg["exploration_runs"]}
        for filename, data in [
            ("config.json", provenance), ("attempts.json", attempts), ("summary.json", summary),
            ("unique_structures.json", unique), ("reference_matches.json", matches),
            ("new_candidates.json", candidates),
        ]:
            api["CompactJSON"].write(output / filename, data)
        if cfg.get("save_media", True):
            ExplorationMedia(api, trained["atom_symbols"], trained["max_valency"],
                             trained["max_atoms"], cfg).save(unique, output, summary, log)
        log.info(f"CAMPAIGN FINISHED | runs={cfg['exploration_runs']} | attempts={len(attempts)}"
                 f" | reference structures={len(matches)} | candidate structures={len(candidates)}")
        return summary


# ==========================================
# PARAMETERS — relative to your working directory
# ==========================================
STAGE1_DIRECTORY = "../stageI"
CHECKPOINT_PATH = "../stageI/output_stage1/20260916_105526_866421/trained_growth_gnn.pt"
REFERENCE_JSON_PATH = "../stageI/species_graphs.json"
OUTPUT_FOLDER = os.environ.get(
    "STAGE2_OUTPUT_FOLDER", "output_stage2/" + datetime.now().strftime("%Y%m%d_%H%M%S_%f"))
# None explores every unique reference composition (29 currently).
# To request selected/nearby compositions, use e.g. [(2, 4, 1), (2, 6, 1)].
# Counts are ALWAYS ordered C,H,O. Total atoms must fit the trained size limit.
COMPOSITIONS = None
# Each run uses a different seed; results are deduplicated across all runs.
EXPLORATION_RUNS = int(os.environ.get("STAGE2_RUNS", "10"))
SAMPLES_PER_COMPOSITION = int(os.environ.get("STAGE2_SAMPLES", "100"))
# 1.0 uses the learned distribution; lower values concentrate on high-score actions,
# higher values spread probability more broadly over chemically masked actions.
TEMPERATURE = 1.0
SEED = 24680
BATCH_SIZE = 100
CPU_THREADS = 1
# PNG shows all unique successful structures; MP4 replays their first growth traces.
SAVE_MEDIA = True
PLOT_DPI = 150
VIDEO_FPS = 2

# ==========================================
# RUN EXPLORATION — direct calls, no main guard
# ==========================================
exploration_config = {
    "stage1_directory": STAGE1_DIRECTORY, "checkpoint_path": CHECKPOINT_PATH,
    "reference_path": REFERENCE_JSON_PATH, "output_folder": OUTPUT_FOLDER,
    "compositions": COMPOSITIONS, "samples_per_composition": SAMPLES_PER_COMPOSITION,
    "temperature": TEMPERATURE, "seed": SEED, "batch_size": BATCH_SIZE, "threads": CPU_THREADS,
    "save_media": SAVE_MEDIA, "plot_dpi": PLOT_DPI, "video_fps": VIDEO_FPS,
    "exploration_runs": EXPLORATION_RUNS,
}
explorer = ExplorationCampaign(exploration_config)
explorer.run()
