"""Stage II: sample or search a fixed stage I policy and inventory structures.

Run from self_improving/stageII. Classes first, editable parameters below, direct calls.
"""
import ast
import hashlib
import json
import math
import os
import shutil
import time
from collections import Counter
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


class MarginalGrowthPolicy:
    """Use a formula-conditioned checkpoint without selecting a target formula.

    Average action probabilities over reference compositions at each state.
    These are soft policy hints only; they impose no atom-count restrictions.
    This adapter does not train an unconditional model or a new STOP head.
    """

    def __init__(self, model, conditions):
        self.model = model
        self.conditions = list(conditions)
        if not self.conditions:
            raise ValueError("Free growth requires at least one reference context")

    def encode(self, environments, unused_conditions):
        expanded = [env for env in environments for _ in self.conditions]
        contexts = self.conditions * len(environments)
        return self.model.encode(expanded, contexts)

    def __call__(self, *inputs):
        log_probs = self.model(*inputs).log_softmax(-1)
        # Return log probabilities as logits so downstream temperature and
        # uniform exploration mixing continue to work with the same interface.
        grouped = log_probs.reshape(-1, len(self.conditions), log_probs.shape[-1])
        return torch.logsumexp(grouped, dim=1) - math.log(len(self.conditions))


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
        c, h, o = record["generated_C_H_O"]
        frequency = (record["frequency_overall"] if record["requested_C_H_O"] is None
                     else record["frequency_within_requested_composition"])
        name = "Candidate" if candidate else record["species_key"]
        ax.set_title(f"{record['structure_id']}  {name} | C={c} H={h} O={o}"
                     f"\nCount: {record['occurrences']}  |  {frequency:.1%}",
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
        fig, axes = self.plt.subplots(rows, columns, figsize=(18, rows*3.8+1.2), squeeze=False)
        try:
            for index, ax in enumerate(axes.flat):
                if index < len(records):
                    self.draw(ax, records[index], records[index]["graph"], positions[index])
                else:
                    ax.set_axis_off()
            fig.suptitle(
                f"{self.config.get('search_method', 'sampling').upper()} exploration | "
                f"{len(records)} distinct structures from {summary['total_attempts']:,} attempts"
                "\nOrange: new candidates (unvalidated) | Teal: reference matches | C: grey, H: blue, O: red",
                fontsize=15, y=0.99)
            fig.tight_layout(rect=(0, 0, 1, 0.945), h_pad=3.5, w_pad=1.2)
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
                            f"{self.config.get('search_method', 'sampling').upper()} exploration | "
                            f"growth of distinct structures | page {start//6+1}/{math.ceil(len(records)/6)}"
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


class PUCTSearch:
    """One persistent search tree per requested composition, with policy rollouts.

    Each simulation is a complete recorded attempt. Q is the average terminal
    reward backed up through an edge, not a neural value prediction. No optimizer
    is used. All admissible actions remain available, including tiny-prior ones.
    """

    DEFAULTS = {"search_method": "sampling", "puct_c": 2.0,
                "puct_uniform_fraction": 0.25, "puct_require_composition": True,
                "puct_reference_reward": 0.1, "puct_candidate_reward": 1.0,
                "stop_when_all_species_recovered": False, "recovery_species": None,
                "generation_mode": "formula", "attempts_per_run": 1000}

    @classmethod
    def validate(cls, config):
        if config["search_method"] not in ("sampling", "puct"):
            raise ValueError("search_method must be sampling or puct")
        if config["generation_mode"] not in ("free", "formula"):
            raise ValueError("generation_mode must be free or formula")
        if type(config["attempts_per_run"]) is not int or config["attempts_per_run"] < 1:
            raise ValueError("attempts_per_run must be a positive integer")
        for key in ("puct_c", "puct_uniform_fraction", "puct_reference_reward",
                    "puct_candidate_reward"):
            if not math.isfinite(config[key]):
                raise ValueError(f"{key} must be finite")
        if config["puct_c"] <= 0 or not 0 <= config["puct_uniform_fraction"] <= 1:
            raise ValueError("PUCT needs c > 0 and a uniform fraction in [0, 1]")
        if not 0 <= config["puct_reference_reward"] < config["puct_candidate_reward"] <= 1:
            raise ValueError("Rewards must satisfy 0 <= reference < candidate <= 1")
        if type(config["puct_require_composition"]) is not bool:
            raise ValueError("puct_require_composition must be boolean")
        if type(config["stop_when_all_species_recovered"]) is not bool:
            raise ValueError("stop_when_all_species_recovered must be boolean")
        if config["stop_when_all_species_recovered"]:
            if config["search_method"] != "puct" or config["puct_uniform_fraction"] <= 0:
                raise ValueError("Full recovery requires PUCT with a positive uniform prior fraction")

    def __init__(self, env, condition, model, space, references, config, generator, max_steps):
        self.condition = tuple(condition) if condition is not None else None
        self.model, self.space = model, space
        self.references, self.config, self.generator = references, config, generator
        self.max_steps = max_steps
        self.root = self.node(env)
        self.model_calls = 0
        self.node_count = 1

    @staticmethod
    def node(env):
        return {"env": env, "visits": 0, "edges": None}

    def admissible_actions(self, env):
        actions = env.get_valid_actions()
        if self.condition is None or not self.config["puct_require_composition"]:
            return actions
        # This is a search constraint only. Stage I's training masks are unchanged.
        current = env.composition()
        indices = {"C": 0, "H": 1, "O": 2}
        result = []
        for action in actions:
            if action[0] == "STOP" and current != self.condition:
                continue
            if action[0] in ("START", "GROW"):
                atom = action[1] if action[0] == "START" else action[2]
                index = indices[env.atom_symbols[atom]]
                if current[index] >= self.condition[index]:
                    continue
            result.append(action)
        return result

    def distribution(self, env):
        actions = self.admissible_actions(env)
        if not actions:
            return []
        logits = self.model(*self.model.encode([env], [self.condition]))[0]
        self.model_calls += 1
        # Keep the original network probability as a separate diagnostic. The
        # constrained prior is renormalized BEFORE mixing in uniform exploration.
        logits = logits / self.config["temperature"]
        policy = logits.softmax(-1)
        ids = [self.space.index[a] for a in actions]
        constrained = logits[ids].softmax(-1)
        fraction = self.config["puct_uniform_fraction"]
        priors = (1 - fraction) * constrained + fraction / len(actions)
        if not torch.isfinite(policy).all() or not torch.isfinite(priors).all():
            raise RuntimeError("Non-finite PUCT probabilities")
        return [{"action": action, "prior": float(priors[i]),
                 "policy_probability": float(policy[ids[i]]),
                 "visits": 0, "value_sum": 0.0, "child": None}
                for i, action in enumerate(actions)]

    def score(self, node, edge):
        q = edge["value_sum"] / edge["visits"] if edge["visits"] else 0.0
        u = (self.config["puct_c"] * edge["prior"]
             * math.sqrt(max(1, node["visits"])) / (1 + edge["visits"]))
        return q, u

    @staticmethod
    def trace_step(edge, step, method, q=None, u=None):
        # Tree selection is deterministic given the current tree. It is NOT a
        # sample from the network. Rollout probability is the mixed prior.
        return {"step": step, "action": list(edge["action"]),
                "selection_method": method,
                "probability": 1.0 if method == "puct" else edge["prior"],
                "policy_probability": edge["policy_probability"],
                "search_prior": edge["prior"], "visits_before": edge["visits"],
                "q_before": q, "u_before": u}

    def reward(self, env):
        if not env.terminated or (self.condition is not None and env.composition() != self.condition):
            return 0.0
        known = self.references.match(env.to_graph()) is not None
        return self.config["puct_reference_reward" if known else "puct_candidate_reward"]

    def simulate(self):
        node, path, trace = self.root, [], []
        env = node["env"].copy()
        dead_end = False
        with torch.inference_mode():
            while not env.terminated and len(trace) < self.max_steps:
                if node["edges"] is None:
                    node["edges"] = self.distribution(env)
                if not node["edges"]:
                    dead_end = True
                    break
                edge = max(node["edges"], key=lambda e: sum(self.score(node, e)))
                q, u = self.score(node, edge)
                trace.append(self.trace_step(edge, len(trace) + 1, "puct", q, u))
                path.append((node, edge))
                env.step(edge["action"])
                new_child = edge["child"] is None
                if new_child:
                    edge["child"] = self.node(env.copy())
                    self.node_count += 1
                node = edge["child"]
                if new_child:
                    # No value head is trained in stage I. Complete this leaf
                    # with a policy rollout, then back up its observed reward.
                    while not env.terminated and len(trace) < self.max_steps:
                        choices = self.distribution(env)
                        if not choices:
                            dead_end = True
                            break
                        index = int(torch.multinomial(
                            torch.tensor([e["prior"] for e in choices], dtype=torch.float64),
                            1, generator=self.generator))
                        selected = choices[index]
                        trace.append(self.trace_step(selected, len(trace) + 1, "rollout"))
                        env.step(selected["action"])
                    break
        value = self.reward(env)
        # Single-agent search: every ancestor receives the SAME reward sign.
        node["visits"] += 1
        for parent, edge in path:
            parent["visits"] += 1
            edge["visits"] += 1
            edge["value_sum"] += value
        return env, trace, {"terminal_reward": value, "tree_depth": len(path),
                            "dead_end": dead_end, "simulation": self.root["visits"]}

    def statistics(self):
        return {"requested_C_H_O": list(self.condition) if self.condition is not None else None,
                "simulations": self.root["visits"], "tree_nodes": self.node_count,
                "model_evaluations": self.model_calls,
                "root_edges": [{"action": list(e["action"]), "prior": e["prior"],
                                "policy_probability": e["policy_probability"],
                                "visits": e["visits"],
                                "mean_reward": e["value_sum"] / e["visits"] if e["visits"] else 0.0}
                               for e in (self.root["edges"] or [])]}


class ExplorationOutput:
    """Replace generated results, including stale media and previous run folders."""

    @staticmethod
    def prepare(path):
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        # Remove only exploration-owned outputs. Clearing optional files matters
        # when a new run disables media or changes from PUCT to sampling.
        for name in ("exploration.log", "config.json", "summary.json", "attempts.json",
                     "unique_structures.json", "reference_matches.json", "new_candidates.json",
                     "search_statistics.json", "recovery_progress.json",
                     "final_structures.png", "exploration_growth.mp4"):
            (path / name).unlink(missing_ok=True)
        runs = path / "runs"
        if runs.is_symlink():
            runs.unlink()
        elif runs.exists():
            shutil.rmtree(runs)
        return path


class Stage2Explorer:
    """Generate, classify, and deduplicate graphs; no optimizer or weight updates."""

    def __init__(self, config):
        self.config = {**PUCTSearch.DEFAULTS, **config}
        PUCTSearch.validate(self.config)
        self.api = Stage1Loader.load(config["stage1_directory"])
        self.json = self.api["CompactJSON"]
        self.output = ExplorationOutput.prepare(config["output_folder"])
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
        if condition is not None and env.composition() != tuple(condition):
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

    @staticmethod
    def recovery_targets(config, references, conditions, max_steps):
        if not config["stop_when_all_species_recovered"]:
            return set()
        names = config["recovery_species"]
        if names is not None and (not isinstance(names, (list, tuple))
                                  or not names or any(not isinstance(n, str) for n in names)):
            raise ValueError("recovery_species must be None or a nonempty list of species keys")
        targets = set(references.graphs if names is None else names)
        unknown = targets - references.graphs.keys()
        if unknown:
            raise ValueError(f"Recovery targets are missing or excluded: {sorted(unknown)}")
        unavailable = [name for name in sorted(targets)
                       if None not in conditions and references.composition(references.graphs[name]) not in conditions]
        if unavailable:
            raise ValueError(f"COMPOSITIONS omits recovery targets: {unavailable}; "
                             "use COMPOSITIONS=None or select RECOVERY_SPECIES explicitly")
        # START + one action per final bond (growth or connect) + STOP is the
        # shortest possible construction with this connected-graph action space.
        too_long = [name for name in sorted(targets)
                    if references.graphs[name].number_of_edges() + 2 > max_steps]
        if too_long:
            raise ValueError(f"Checkpoint step limit cannot recover: {too_long}")
        return targets

    @staticmethod
    def recovered_names(env, condition, references, targets):
        if not env.terminated or (condition is not None and env.composition() != tuple(condition)):
            return set()
        graph = env.to_graph()
        return {name for name in references.conditions.get(env.composition(), [])
                if name in targets and references.isomorphic(graph, references.graphs[name])}

    def search_batches(self, conditions, searches, references, targets, recovered):
        if not self.config["stop_when_all_species_recovered"]:
            budget = (self.config["attempts_per_run"] if self.config["generation_mode"] == "free"
                      else self.config["samples_per_composition"])
            for condition in conditions:
                for start in range(0, budget, self.config["batch_size"]):
                    yield condition, start, min(self.config["batch_size"],
                          budget - start), searches[condition]
            return
        # Round-robin incomplete formulas; retain every tree across rounds.
        # One simulation per yield lets the caller stop immediately on full coverage.
        completed = Counter()
        while targets - recovered:
            for condition in conditions:
                pending = ((targets - recovered) if condition is None else
                           set(references.conditions.get(condition, [])) & (targets - recovered))
                if pending:
                    yield condition, completed[condition], 1, searches[condition]
                    completed[condition] += 1

    def save_recovery_progress(self, targets, recovered, attempts, excluded):
        progress = {"target_species": sorted(targets), "recovered_species": sorted(recovered),
                    "missing_species": sorted(targets - recovered), "attempts": attempts,
                    "recovered_count": len(recovered), "target_count": len(targets),
                    "all_species_recovered": targets <= recovered,
                    "excluded_species": list(excluded)}
        self.json.write(self.output / "recovery_progress.json", progress)
        self.log.info(f"RECOVERY {len(recovered)}/{len(targets)} | attempts={attempts}"
                      f" | missing={progress['missing_species']}")

    def sample_batch(self, size, condition, model, space, generator, max_atoms, max_steps):
        cfg = self.config
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
        return environments, traces

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
        free_growth = cfg["generation_mode"] == "free"
        budget = cfg["attempts_per_run"] if free_growth else cfg["samples_per_composition"]
        for key in (("attempts_per_run" if free_growth else "samples_per_composition"), "batch_size", "threads"):
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
        conditions = [None] if free_growth else self.validate_conditions(
            list(references.conditions) if cfg["compositions"] is None else cfg["compositions"], max_atoms)
        targets = self.recovery_targets(cfg, references, conditions, max_steps)
        recovered = set()
        model = self.api["GrowthGNN"](trained["hidden_dims"], space, self.symbols, self.valency)
        model.load_state_dict(checkpoint["model_state_dict"], strict=True)
        model.eval()
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        if free_growth:
            model = MarginalGrowthPolicy(model, references.conditions)
        generator = torch.Generator().manual_seed(cfg["seed"])
        provenance = {
            **cfg, "resolved_compositions_C_H_O": conditions, "checkpoint_epoch": checkpoint["epoch"],
            "checkpoint_sha256": hashlib.sha256(checkpoint_path.read_bytes()).hexdigest(),
            "reference_sha256": references.source_hash, "reference_matches_training_hash": source_matches,
            "stage1_code_sha256": hashlib.sha256(
                (Path(cfg["stage1_directory"]) / "train_stage1.py").read_bytes()).hexdigest(),
            "max_atoms": max_atoms, "max_steps": max_steps,
            "policy_context_mode": "mean_probability_over_reference_formulas" if free_growth else "requested_formula",
            "policy_contexts_C_H_O": list(references.conditions) if free_growth else conditions,
        }
        self.json.write(self.output / "config.json", provenance)
        self.log.info(f"STAGE II | method={cfg['search_method']} | fixed network | checkpoint={checkpoint_path} | epoch={checkpoint['epoch']}")
        self.log.info(f"Generation={cfg['generation_mode']} | request groups={len(conditions)}"
                      f" | attempts/group={budget} | temperature={cfg['temperature']} | seed={cfg['seed']}")
        if free_growth:
            self.log.info("FREE GROWTH | no target atom counts; STOP is available after the first atom."
                          " Policy averages over reference formula contexts; weights remain frozen."
                          f" Hard limits: atoms={max_atoms}, actions={max_steps}.")
        self.log.info("New candidates satisfy construction masks and any active formula constraint;"
                      " they are not confirmed stable or chemically important species.")
        attempts, unique, buckets, summaries = [], [], {}, []
        search_statistics = []
        if cfg["search_method"] == "puct":
            self.log.info(f"PUCT | c={cfg['puct_c']} | uniform={cfg['puct_uniform_fraction']}"
                          f" | exact composition constraint={cfg['puct_require_composition'] and not free_growth}"
                          f" | rewards: failure=0, reference={cfg['puct_reference_reward']},"
                          f" candidate={cfg['puct_candidate_reward']}")
        started = time.monotonic()
        searches = {
            condition: (PUCTSearch(self.api["AdvancedMoleculeEnv"](
                self.symbols, self.valency, max_atoms), condition, model, space,
                references, cfg, generator, max_steps) if cfg["search_method"] == "puct" else None)
            for condition in conditions}
        condition_counts = {condition: Counter() for condition in conditions}
        generated_counts = {}
        if targets:
            self.log.info("STOP CONDITION: recover every target species; sample/run budgets do not stop PUCT."
                          f" Excluded by checkpoint: {trained['excluded_species']}")
            self.save_recovery_progress(targets, recovered, 0, trained["excluded_species"])
        for condition, start, size, search in self.search_batches(
                conditions, searches, references, targets, recovered):
            counts = condition_counts[condition]
            diagnostics = [None] * size
            if search is None:
                environments, traces = self.sample_batch(
                    size, condition, model, space, generator, max_atoms, max_steps)
            else:
                environments, traces, diagnostics = [], [], []
                for offset in range(size):
                    env, trace, diagnostic = search.simulate()
                    environments.append(env)
                    traces.append(trace)
                    diagnostics.append(diagnostic)
                    completed = start + offset + 1
                    if completed % 25 == 0:
                        self.log.info(f"PUCT C,H,O={condition} | simulation={completed}"
                                      f"/{'coverage' if targets else budget}"
                                      f" | nodes={search.node_count}"
                                      f" | latest reward={diagnostic['terminal_reward']:.2f}")
            for env, trace, diagnostic in zip(environments, traces, diagnostics):
                match = references.match(env.to_graph())
                category = self.classify(env, condition, match)
                if diagnostic and diagnostic["dead_end"]:
                    category = "failed_dead_end"
                counts[category] += 1
                generated_condition = env.composition()
                generated_counts.setdefault(generated_condition, Counter())[category] += 1
                record = {
                    "attempt": len(attempts) + 1,
                    "requested_C_H_O": list(condition) if condition is not None else None,
                    "generated_C_H_O": list(env.composition()),
                    "category": category, "species_key": match or "Unknown",
                    "in_species_training_reference": match is not None,
                    "termination": ("STOP" if env.terminated else
                                    "dead_end" if diagnostic and diagnostic["dead_end"] else "step_limit"),
                    "structure_id": None, **self.structure_record(env),
                    "actions": trace,
                    **({"search": diagnostic} if diagnostic is not None else {}),
                }
                if category in ("reference_match", "new_candidate"):
                    found = self.find_structure(env.to_graph(), generated_condition, buckets, references)
                    if found is None:
                        found = {**record, "structure_id": f"M{len(unique)+1:05d}",
                                 "occurrences": 0, "first_attempt": record["attempt"]}
                        unique.append(found)
                        buckets.setdefault(generated_condition, []).append((env.to_graph(), found))
                    found["occurrences"] += 1
                    record["structure_id"] = found["structure_id"]
                attempts.append(record)
                if targets:
                    newly_recovered = self.recovered_names(env, condition, references, targets) - recovered
                    recovered.update(newly_recovered)
                    if newly_recovered or len(attempts) % 25 == 0:
                        self.save_recovery_progress(
                            targets, recovered, len(attempts), trained["excluded_species"])
        search_statistics = [search.statistics() for search in searches.values() if search is not None]
        for condition, counts in (generated_counts if free_growth else condition_counts).items():
            summary = {("generated_C_H_O" if free_growth else "requested_C_H_O"): list(condition),
                       "seen_composition_in_training": condition in references.conditions,
                       "attempts": sum(counts.values()),
                       **{key: counts[key] for key in ("reference_match", "new_candidate",
                                                      "failed_composition", "failed_step_limit", "failed_dead_end")},
                       "unique_structures": len(buckets.get(condition, []))}
            summaries.append(summary)
            self.log.info(f"C,H,O={condition} | reference={counts['reference_match']}"
                          f" | candidate samples={counts['new_candidate']}"
                          f" | wrong formula={counts['failed_composition']}"
                          f" | step limit={counts['failed_step_limit']}"
                          f" | dead ends={counts['failed_dead_end']}"
                          f" | unique={summary['unique_structures']}")
        for record in unique:
            record["frequency_within_requested_composition"] = (
                None if free_growth else
                record["occurrences"] / sum(condition_counts[tuple(record["requested_C_H_O"])].values()))
            record["frequency_overall"] = record["occurrences"] / len(attempts)
        if search_statistics:
            self.json.write(self.output / "search_statistics.json", search_statistics)
        matches = [r for r in unique if r["category"] == "reference_match"]
        candidates = [r for r in unique if r["category"] == "new_candidate"]
        summary = {
            "search_method": cfg["search_method"],
            "generation_mode": cfg["generation_mode"],
            "exploration_seconds": time.monotonic() - started,
            "termination_reason": "all_species_recovered" if targets else "attempt_budget_complete",
            **({"target_species": sorted(targets), "recovered_species": sorted(recovered),
                "missing_species": sorted(targets - recovered),
                "all_species_recovered": targets <= recovered,
                "excluded_species": trained["excluded_species"]} if targets else {}),
            "total_attempts": len(attempts),
            "counts": dict(Counter(r["category"] for r in attempts)),
            "unique_reference_structures": len(matches), "unique_new_candidates": len(candidates),
            ("per_generated_composition" if free_growth else "per_composition"): summaries,
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
        self.config = {**PUCTSearch.DEFAULTS, **config}

    def run(self):
        cfg = self.config
        if cfg["stop_when_all_species_recovered"]:
            # Coverage uses one persistent set of trees, rather than restarting
            # searches after fixed-size runs. Output stays directly in output/.
            return Stage2Explorer(cfg).run()
        runs = cfg["exploration_runs"]
        if type(runs) is not int or runs < 1:
            raise ValueError("exploration_runs must be a positive integer")
        api = Stage1Loader.load(cfg["stage1_directory"])
        output = ExplorationOutput.prepare(cfg["output_folder"])
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
        free_growth = cfg["generation_mode"] == "free"
        attempts, unique, buckets, run_summaries = [], [], {}, []
        search_statistics = []
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
            stats_path = run_output / "search_statistics.json"
            if stats_path.exists():
                search_statistics.extend({"run": run_id, **item}
                                         for item in json.loads(stats_path.read_text()))
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
                condition = tuple(record["generated_C_H_O"] if free_growth else record["requested_C_H_O"])
                condition_counts.setdefault(condition, Counter())[record["category"]] += 1
                if record["category"] in ("reference_match", "new_candidate"):
                    graph = api["nx"].Graph()
                    graph.add_nodes_from((n["id"], {"atom": n["atom"]}) for n in record["graph"]["nodes"])
                    graph.add_edges_from((e["source"], e["target"], {"order": e["order"]})
                                         for e in record["graph"]["edges"])
                    graph_condition = tuple(record["generated_C_H_O"])
                    found = Stage2Explorer.find_structure(graph, graph_condition, buckets, references)
                    if found is None:
                        found = {**record, "structure_id": f"M{len(unique)+1:05d}",
                                 "first_attempt": record["attempt"], "occurrences": 0,
                                 "runs_observed": []}
                        unique.append(found)
                        buckets.setdefault(graph_condition, []).append((graph, found))
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
                None if free_growth else
                record["occurrences"] / sum(condition_counts[tuple(record["requested_C_H_O"])].values()))
            record["frequency_overall"] = record["occurrences"] / len(attempts)
        matches = [r for r in unique if r["category"] == "reference_match"]
        candidates = [r for r in unique if r["category"] == "new_candidate"]
        summary = {
            "search_method": cfg["search_method"],
            "generation_mode": cfg["generation_mode"],
            "exploration_runs": cfg["exploration_runs"], "total_attempts": len(attempts),
            "counts": dict(Counter(r["category"] for r in attempts)),
            "unique_reference_structures": len(matches), "unique_new_candidates": len(candidates),
            ("per_generated_composition" if free_growth else "per_composition"): [
                {("generated_C_H_O" if free_growth else "requested_C_H_O"): list(condition),
                 "attempts": sum(counts.values()),
                 "seen_composition_in_training": condition in references.conditions,
                 **{key: counts[key] for key in ("reference_match", "new_candidate",
                                                 "failed_composition", "failed_step_limit", "failed_dead_end")},
                 "unique_structures": len(buckets.get(condition, []))}
                for condition, counts in condition_counts.items()],
            "runs": run_summaries,
        }
        provenance = {**first_config, **cfg,
                      "run_seeds": [r["seed"] for r in run_summaries],
                      "total_samples_per_composition": (None if free_growth else
                          cfg["samples_per_composition"] * cfg["exploration_runs"]),
                      "total_attempts": len(attempts)}
        for filename, data in [
            ("config.json", provenance), ("attempts.json", attempts), ("summary.json", summary),
            ("unique_structures.json", unique), ("reference_matches.json", matches),
            ("new_candidates.json", candidates),
        ]:
            api["CompactJSON"].write(output / filename, data)
        if search_statistics:
            # Trees are independent across runs, so retain run IDs instead of
            # adding edge visit counts from different trees together.
            api["CompactJSON"].write(output / "search_statistics.json", search_statistics)
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
CHECKPOINT_PATH = "../stageI/output_stage1/20260921_221023_510656/trained_growth_gnn.pt"
REFERENCE_JSON_PATH = "../stageI/species_graphs.json"
# Each execution replaces the previous exploration in this folder.
OUTPUT_FOLDER = os.environ.get("STAGE2_OUTPUT_FOLDER", "output")
# In formula mode, None explores every unique reference composition (29 currently).
# To request selected/nearby formulas, use e.g. [(2, 4, 1), (2, 6, 1)].
# Counts are ALWAYS ordered C,H,O. Total atoms must fit the trained size limit.
GENERATION_MODE = "free"  # "free": policy chooses counts and STOP; "formula": fixed formula
COMPOSITIONS = None  # Used only in formula mode.
ATTEMPTS_PER_RUN = int(os.environ.get("STAGE2_ATTEMPTS", "1000"))
# Each run uses a different seed; results are deduplicated across all runs.
# PUCT reuses a tree within each run and composition. More simulations allow
# Q estimates to guide exploration beyond the initial network probabilities.
SEARCH_METHOD = os.environ.get("STAGE2_METHOD", "puct")  # "puct" or "sampling"
PUCT_C = 2.0
PUCT_UNIFORM_FRACTION = 0.25
PUCT_REQUIRE_COMPOSITION = GENERATION_MODE == "formula"
PUCT_REFERENCE_REWARD = 0.1
PUCT_CANDIDATE_REWARD = 1.0
# False: stop after the configured runs and attempts for the selected mode.
# Optional True (PUCT only): ignore those budgets and stop at full recovery.
STOP_WHEN_ALL_SPECIES_RECOVERED = False
RECOVERY_SPECIES = None
# Free mode: total attempts = EXPLORATION_RUNS * ATTEMPTS_PER_RUN (default 10,000).
# Formula mode: runs * samples per composition * number of compositions.
EXPLORATION_RUNS = int(os.environ.get("STAGE2_RUNS", "100"))
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
    "generation_mode": GENERATION_MODE, "attempts_per_run": ATTEMPTS_PER_RUN,
    "temperature": TEMPERATURE, "seed": SEED, "batch_size": BATCH_SIZE, "threads": CPU_THREADS,
    "save_media": SAVE_MEDIA, "plot_dpi": PLOT_DPI, "video_fps": VIDEO_FPS,
    "exploration_runs": EXPLORATION_RUNS, "search_method": SEARCH_METHOD,
    "puct_c": PUCT_C, "puct_uniform_fraction": PUCT_UNIFORM_FRACTION,
    "puct_require_composition": PUCT_REQUIRE_COMPOSITION,
    "puct_reference_reward": PUCT_REFERENCE_REWARD,
    "puct_candidate_reward": PUCT_CANDIDATE_REWARD,
    "stop_when_all_species_recovered": STOP_WHEN_ALL_SPECIES_RECOVERED,
    "recovery_species": RECOVERY_SPECIES,
}
explorer = ExplorationCampaign(exploration_config)
explorer.run()
