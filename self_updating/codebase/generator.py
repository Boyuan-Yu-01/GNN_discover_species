"""Frozen-policy PUCT and direct sampling with the same checkpoint and reference."""
import hashlib
import math
import shutil
import sys
import time
from collections import Counter
from pathlib import Path
import networkx as nx
import torch
from tqdm import tqdm
from .molecule import AdvancedMoleculeEnv, ActionSpace, ReferenceSpecies
from .model import GrowthGNN
from .storage import CompactJSON, Storage
from .config import Settings
from .reporter import TrainingLogger, ExplorationMedia

class MarginalGrowthPolicy:
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


class PUCTSearch:
    """One persistent search tree per requested composition, with policy rollouts.

    Each simulation is a complete recorded attempt. Q is the average terminal
    reward backed up through an edge, not a neural value prediction. No optimizer
    is used. All admissible actions remain available, including tiny-prior ones.
    """

    DEFAULTS = {**Settings.GENERATION, "search_method": "sampling",
                "puct_require_composition": True, "generation_mode": "formula",
                "exploration_runs": 1}

    @classmethod
    def validate(cls, config):
        if config["search_method"] not in ("sampling", "puct"):
            raise ValueError("search_method must be sampling or puct")
        if config["generation_mode"] not in ("free", "formula"):
            raise ValueError("generation_mode must be free or formula")
        if type(config["attempts_per_run"]) is not int or config["attempts_per_run"] < 1:
            raise ValueError("attempts_per_run must be a positive integer")
        if type(config["exploration_runs"]) is not int or config["exploration_runs"] < 1:
            raise ValueError("exploration_runs must be a positive integer")
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
        if type(config["log_level"]) is not int or config["log_level"] not in (0, 1, 2):
            raise ValueError("log_level must be 0, 1, or 2")

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
                     "final_structures.png", "classified_structures.png", "classified_summary.json"):
            (path / name).unlink(missing_ok=True)
        runs = path / "runs"
        if runs.is_symlink():
            runs.unlink()
        elif runs.exists():
            shutil.rmtree(runs)
        return path


class FrozenPolicy:
    """Load a campaign's model and graph references once; each run owns its RNG/tree."""

    def __init__(self, config):
        self.checkpoint_path = Path(config["checkpoint_path"])
        self.reference_path = Path(config["reference_path"])
        checkpoint = torch.load(self.checkpoint_path, map_location="cpu", weights_only=True)
        self.trained, self.epoch = checkpoint["config"], checkpoint["epoch"]
        trained = self.trained
        self.space = ActionSpace(trained["max_atoms"], trained["atom_symbols"])
        if [tuple(a) for a in checkpoint["action_space"]] != self.space.actions:
            raise ValueError("Checkpoint action ordering differs from the model")
        self.references = ReferenceSpecies(self.reference_path, trained["excluded_species"],
            trained["atom_symbols"], trained["max_valency"], trained["max_atoms"])
        self.model = GrowthGNN(trained["hidden_dims"], self.space,
                              trained["atom_symbols"], trained["max_valency"])
        self.model.load_state_dict(checkpoint["model_state_dict"], strict=True)
        self.model.eval()
        self.model.requires_grad_(False)
        self.free_model = MarginalGrowthPolicy(self.model, self.references.conditions)
        self.checkpoint_hash = Storage.digest(self.checkpoint_path)
        self.source_matches = self.references.source_hash == checkpoint["source_sha256"]

    def verify(self):
        if (Storage.digest(self.checkpoint_path) != self.checkpoint_hash or
                Storage.digest(self.reference_path) != self.references.source_hash):
            raise ValueError("Campaign checkpoint or reference changed between runs")


class Stage2Explorer:
    """Generate, classify, and deduplicate graphs; no optimizer or weight updates."""

    def __init__(self, config, context=None):
        self.config = {**PUCTSearch.DEFAULTS, **config}
        PUCTSearch.validate(self.config)
        self.context = context or FrozenPolicy(self.config)
        self.json = CompactJSON
        self.output = ExplorationOutput.prepare(config["output_folder"])
        self.log = TrainingLogger(self.output / "exploration.log", level=self.config["log_level"])

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

    def search_batches(self, conditions, searches):
        """Run exactly the configured attempts for each requested condition."""
        budget = (self.config["attempts_per_run"] if self.config["generation_mode"] == "free"
                  else self.config["samples_per_composition"])
        for condition in conditions:
            for start in range(0, budget, self.config["batch_size"]):
                yield condition, start, min(self.config["batch_size"],
                      budget - start), searches[condition]

    def sample_batch(self, size, condition, model, space, generator, max_atoms, max_steps):
        cfg = self.config
        environments = [AdvancedMoleculeEnv(
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
        shared=self.context
        shared.verify()
        trained=shared.trained
        checkpoint_path=shared.checkpoint_path
        self.symbols, self.valency=trained["atom_symbols"], trained["max_valency"]
        max_atoms, max_steps=trained["max_atoms"], trained["max_steps"]
        space, references=shared.space, shared.references
        source_matches=shared.source_matches
        if not source_matches:
            self.log.info("REFERENCE HASH DIFFERS from training; matches use the supplied reference file.")
        conditions = [None] if free_growth else self.validate_conditions(
            list(references.conditions) if cfg["compositions"] is None else cfg["compositions"], max_atoms)
        model=shared.free_model if free_growth else shared.model
        generator = torch.Generator().manual_seed(cfg["seed"])
        provenance = {
            **cfg, "resolved_compositions_C_H_O": conditions, "checkpoint_epoch": shared.epoch,
            "checkpoint_sha256": shared.checkpoint_hash,
            "reference_sha256": references.source_hash, "reference_matches_training_hash": source_matches,
            "model_code_sha256": hashlib.sha256(
                Path(__file__).with_name("model.py").read_bytes()).hexdigest(),
            "max_atoms": max_atoms, "max_steps": max_steps,
            "policy_context_mode": "mean_probability_over_reference_formulas" if free_growth else "requested_formula",
            "policy_contexts_C_H_O": list(references.conditions) if free_growth else conditions,
        }
        self.json.write(self.output / "config.json", provenance)
        # Every updating round uses the same stage names. PUCT and direct
        # sampling are separate exploration passes, so they have II-A and II-B.
        stage = "II-A" if cfg["search_method"] == "puct" else "II-B"
        method_name = "PUCT" if cfg["search_method"] == "puct" else "Direct-sampling"
        round_label = f"ROUND {cfg['round_number']}/{cfg['round_total']} | " if "round_total" in cfg else ""
        if self.config.get("show_stage_objective", True):
            self.log.objective(f"{round_label}STAGE {stage} | {method_name} exploration")
        self.log.info(f"GENERATION | method={cfg['search_method']} | fixed network | checkpoint={checkpoint_path} | epoch={shared.epoch}")
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
            condition: (PUCTSearch(AdvancedMoleculeEnv(
                self.symbols, self.valency, max_atoms), condition, model, space,
                references, cfg, generator, max_steps) if cfg["search_method"] == "puct" else None)
            for condition in conditions}
        condition_counts = {condition: Counter() for condition in conditions}
        generated_counts = {}
        self.log.info(f"STOP CONDITION: complete {budget * len(conditions)} attempts in this run.")
        progress = tqdm(total=budget * len(conditions), desc=f"{method_name} attempts", unit="attempt",
                        file=sys.stdout, dynamic_ncols=True, mininterval=0.5,
                        disable=not self.log.progress_enabled)
        for condition, start, size, search in self.search_batches(conditions, searches):
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
                                      f"/{budget}"
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
            progress.update(size)
        progress.close()
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
            "termination_reason": "attempt_budget_complete",
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
            ExplorationMedia(self.symbols, self.valency, max_atoms, cfg).save(
                unique, self.output, summary, self.log)
        self.log.info(f"FINISHED | attempts={len(attempts)} | reference structures={len(matches)}"
                      f" | new candidate structures={len(candidates)}"
                      f" | elapsed={time.monotonic()-started:.1f}s")
        self.log.info(f"Saved exploration outputs: {self.output}")
        self.result={"attempts":attempts, "unique":unique, "summary":summary,
                     "config":provenance, "search_statistics":search_statistics}
        return summary


class ExplorationCampaign:
    """Run multiple independent seeds and merge results using graph isomorphism."""

    def __init__(self, config):
        self.config = {**PUCTSearch.DEFAULTS, **config}

    def run(self):
        cfg = self.config
        runs = cfg["exploration_runs"]
        if type(runs) is not int or runs < 1:
            raise ValueError("exploration_runs must be a positive integer")
        self.context = FrozenPolicy(cfg)
        output = ExplorationOutput.prepare(cfg["output_folder"])
        log = TrainingLogger(output / "exploration.log", level=cfg["log_level"])
        try:
            return self.explore(output, log)
        except Exception:
            log.exception("CAMPAIGN FAILED; completed runs remain available.")
            raise
        finally:
            log.close()

    def explore(self, output, log):
        cfg = self.config
        stage = "II-A" if cfg["search_method"] == "puct" else "II-B"
        method_name = "PUCT" if cfg["search_method"] == "puct" else "Direct-sampling"
        round_label = f"ROUND {cfg['round_number']}/{cfg['round_total']} | " if "round_total" in cfg else ""
        log.objective(f"{round_label}STAGE {stage} | {method_name} exploration")
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
                          "seed": cfg["seed"] + index * 1000, "save_media": False,
                          "show_stage_objective": False}
            log.info(f"CAMPAIGN run={run_id}/{cfg['exploration_runs']} | seed={run_config['seed']}")
            explorer = Stage2Explorer(run_config, self.context)
            result = explorer.run()
            search_statistics.extend({"run":run_id, **item}
                                     for item in explorer.result["search_statistics"])
            saved_config = explorer.result["config"]
            if first_config is None:
                first_config = saved_config
                trained = self.context.trained
                references = self.context.references
            else:
                # Refuse to merge runs made with changing inputs.
                for key in ("checkpoint_sha256", "reference_sha256", "model_code_sha256",
                            "resolved_compositions_C_H_O"):
                    if saved_config[key] != first_config[key]:
                        raise ValueError(f"Campaign input changed between runs: {key}")
            for record in explorer.result["attempts"]:
                record["run"] = run_id
                record["run_attempt"] = record["attempt"]
                record["attempt"] = len(attempts) + 1
                record["structure_id"] = None
                condition = tuple(record["generated_C_H_O"] if free_growth else record["requested_C_H_O"])
                condition_counts.setdefault(condition, Counter())[record["category"]] += 1
                if record["category"] in ("reference_match", "new_candidate"):
                    graph = nx.Graph()
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
            "termination_reason": "attempt_budget_complete",
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
            CompactJSON.write(output / filename, data)
        if search_statistics:
            # Trees are independent across runs, so retain run IDs instead of
            # adding edge visit counts from different trees together.
            CompactJSON.write(output / "search_statistics.json", search_statistics)
        if cfg.get("save_media", True):
            ExplorationMedia(trained["atom_symbols"], trained["max_valency"],
                             trained["max_atoms"], cfg).save(unique, output, summary, log)
        log.info(f"CAMPAIGN FINISHED | runs={cfg['exploration_runs']} | attempts={len(attempts)}"
                 f" | reference structures={len(matches)} | candidate structures={len(candidates)}")
        return summary


class Generator:
    """One configured campaign; PUCT and direct sampling share a frozen model."""
    DEFAULTS = Settings.GENERATION

    def __init__(self, config=None):
        self.config=Settings.merge(self.DEFAULTS,config)
        methods=self.config['methods']
        if not isinstance(methods,list) or not methods or len(methods)!=len(set(methods)) or any(m not in ('puct','direct') for m in methods):
            raise ValueError('methods must be a nonempty, nonduplicated list of puct/direct')
        if self.config['device']!='cpu':
            raise ValueError('Generation currently supports CPU only; training device is separate')
        for key in ('runs','attempts_per_run','samples_per_composition','batch_size','threads','plot_dpi'):
            if type(self.config[key]) is not int or self.config[key]<1:
                raise ValueError(f'{key} must be a positive integer')
        if not math.isfinite(self.config["temperature"]) or self.config["temperature"] <= 0:
            raise ValueError("temperature must be positive and finite")
        PUCTSearch.validate({**PUCTSearch.DEFAULTS,**self.config,'search_method':'puct',
                             'puct_require_composition':self.config['generation_mode']=='formula'})

    def run(self, method, checkpoint_path, reference_path, output_folder, round_number=1, round_total=None):
        cfg={**self.config,'checkpoint_path':str(checkpoint_path),'reference_path':str(reference_path),
             'output_folder':str(output_folder),'exploration_runs':self.config['runs'],
             'search_method':'puct' if method=='puct' else 'sampling',
             'seed':self.config['seed']+(round_number-1)*1000000,
             'round_number':round_number,
             'puct_require_composition':self.config['generation_mode']=='formula'}
        if round_total is not None:
            cfg['round_total'] = round_total
        if method not in self.config['methods']:
            raise ValueError('Method is not enabled')
        if method=='direct':cfg.update(temperature=1.0,puct_uniform_fraction=0.0)
        return ExplorationCampaign(cfg).run()
