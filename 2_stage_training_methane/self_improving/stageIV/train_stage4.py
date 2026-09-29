"""Stage IV: fine-tune the shared GNN using positive and rejected molecular graphs.

Run from stageIV. Classes first, editable parameters below, then direct calls.
Stage I/II definitions are reused without executing training or exploration.
"""
import ast
import csv
import gzip
import hashlib
import json
import math
import random
import sys
import tempfile
import time
from collections import Counter
from pathlib import Path

import networkx as nx
import torch


class Stage4Loader:
    """Load reusable classes without running a file's bottom-level calls."""

    @staticmethod
    def definitions(path, scope=None):
        path = Path(path)
        tree = ast.parse(path.read_text())
        nodes = [n for n in tree.body if isinstance(n, (ast.Import, ast.ImportFrom, ast.ClassDef))]
        nodes = [n for n in nodes if not (isinstance(n, ast.ImportFrom) and n.module == "compact_json")]
        result = dict(scope or {}, __file__=str(path), __name__="stage4_loaded")
        exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), "exec"), result)
        return result

    @classmethod
    def load(cls, directory):
        directory = Path(directory)
        writer = cls.definitions(directory / "compact_json.py")["CompactJSON"]
        return cls.definitions(directory / "train_stage1.py", {"CompactJSON": writer})


class Stage4RoutePool:
    """Keep exhaustive routes on disk; sample indices uniformly without replacement."""

    def __init__(self, path, count):
        self.path, self.count = Path(path), count

    def __len__(self):
        return self.count

    def sample(self, rng, count):
        selected = set(rng.sample(range(self.count), count))
        with gzip.open(self.path, "rt", encoding="utf-8") as handle:
            for index, line in enumerate(handle):
                if index in selected:
                    yield tuple(tuple(action) for action in json.loads(line))


class Stage4Dataset:
    """Validate both sets; preserve one graph per topology and let positives win."""

    def __init__(self, config, trained, api):
        self.config, self.trained, self.api = config, trained, api
        self.hashes = {key: hashlib.sha256(Path(config[key]).read_bytes()).hexdigest()
                       for key in ("positive_path", "negative_path")}
        raw_positive = json.loads(Path(config["positive_path"]).read_text())
        raw_negative = json.loads(Path(config["negative_path"]).read_text())
        if not isinstance(raw_positive, dict) or not raw_positive or not isinstance(raw_negative, list):
            raise ValueError("Expected positive species-key -> graph dictionary and negative record list")
        negatives = {}
        for record in raw_negative:
            key = record["structure_id"]
            if not isinstance(key, str) or not key or key in negatives:
                raise ValueError(f"Invalid/duplicate negative structure ID: {key}")
            negatives[key] = record["graph"]
        # The current GNN has no charge feature. Do not silently train charged graphs.
        for label, records in (("positive", raw_positive), ("negative", negatives)):
            for key, record in records.items():
                if label == "positive" and key in trained["excluded_species"]:
                    continue
                for node in record["nodes"]:
                    if type(node["atom"]) is not int:
                        raise ValueError(f"{label} {key}: atom type must be an integer")
                    charge = node.get("formal_charge")
                    if charge is not None and (type(charge) is not int or charge != 0):
                        raise ValueError(f"{label} {key}: charged graphs are unsupported by the parent model")
        self.positive = self.references(raw_positive, trained["excluded_species"])
        self.negative = self.references(negatives, []) if negatives else None
        self.removed = {"positive_duplicates": {}, "negative_duplicates": {}, "positive_negative_overlap": []}
        self.deduplicate(self.positive, "positive_duplicates")
        if self.negative:
            self.deduplicate(self.negative, "negative_duplicates")
            for name, graph in list(self.negative.graphs.items()):
                if self.positive.match(graph) is not None:
                    self.removed["positive_negative_overlap"].append(name)
                    del self.negative.graphs[name]
            self.reindex(self.negative)
            if not self.negative.graphs:
                self.negative = None
        self.positive.source_hash = self.hashes["positive_path"]
        self.sequences = {"positive": {}, "negative": {}}

    def references(self, records, excluded):
        # Reuse Stage I's complete validation without creating repository input copies.
        with tempfile.TemporaryDirectory(prefix="stage4_graphs_") as folder:
            path = Path(folder) / "graphs.json"
            path.write_text(json.dumps(records))
            return self.api["ReferenceSpecies"](
                path, excluded, self.trained["atom_symbols"], self.trained["max_valency"],
                self.trained["max_atoms"])

    @staticmethod
    def reindex(reference):
        reference.conditions = {}
        for name, graph in reference.graphs.items():
            reference.conditions.setdefault(reference.composition(graph), []).append(name)

    def deduplicate(self, reference, label):
        kept = []
        for name, graph in list(reference.graphs.items()):
            duplicate = next((key for key, other in kept if reference.isomorphic(graph, other)), None)
            if duplicate is None:
                kept.append((name, graph))
            else:
                self.removed[label][name] = duplicate
                del reference.graphs[name]
        self.reindex(reference)

    def enumerate_routes(self, reference, name):
        """Stage I's finite route family: introduce each final bond directly.

        This excludes optional intermediate bond-order upgrades, as Stage I does.
        Full-reference automorphisms remove interchangeable reference-node choices
        with identical futures, while retaining every distinct action sequence.
        Limits raise an error; the route pool is never silently truncated.
        """
        graph = reference.graphs[name]
        if graph.number_of_edges() + 2 > self.trained["max_steps"]:
            raise ValueError(f"{name}: complete routes exceed the checkpoint action limit")
        matcher = nx.algorithms.isomorphism.GraphMatcher(
            graph, graph, node_match=reference.node_match, edge_match=reference.edge_match)
        automorphisms = []
        for mapping in matcher.isomorphisms_iter():
            automorphisms.append(mapping)
            if len(automorphisms) > self.config["max_enumeration_states_per_graph"]:
                raise ValueError(f"{name}: too many graph symmetries for the enumeration budget")
        sequences, visits = set(), 0

        def visit(mapping, built, actions, symmetries):
            nonlocal visits
            visits += 1
            if visits > self.config["max_enumeration_states_per_graph"]:
                raise ValueError(f"{name}: enumeration exceeded MAX_ENUMERATION_STATES_PER_GRAPH; increase it explicitly")
            choices = []
            if not mapping:
                choices = [(("START", graph.nodes[u]["atom"]), u, None) for u in graph]
            else:
                for u, v, edge in graph.edges(data=True):
                    bond = frozenset((u, v))
                    if u in mapping and v not in mapping:
                        choices.append((("GROW", mapping[u], graph.nodes[v]["atom"], edge["order"]), v, bond))
                    elif v in mapping and u not in mapping:
                        choices.append((("GROW", mapping[v], graph.nodes[u]["atom"], edge["order"]), u, bond))
                    elif u in mapping and v in mapping and bond not in built:
                        a, b = sorted((mapping[u], mapping[v]))
                        choices.append((("CONNECT", a, b, edge["order"]), None, bond))
            if not choices:
                if len(mapping) != len(graph):
                    raise ValueError(f"{name}: incomplete route")
                sequences.add(actions + (("STOP",),))
                if len(sequences) > self.config["max_routes_per_graph"]:
                    raise ValueError(f"{name}: enumeration exceeded MAX_ROUTES_PER_GRAPH; increase it explicitly")
                return
            seen = set()
            for action, new_node, bond in choices:
                orbit = min(m[new_node] for m in symmetries) if new_node is not None else None
                key = action, orbit
                if key in seen:
                    continue
                seen.add(key)
                next_mapping = dict(mapping)
                stabilizer = symmetries
                if new_node is not None:
                    next_mapping[new_node] = len(mapping)
                    stabilizer = [m for m in symmetries if m[new_node] == new_node]
                visit(next_mapping, built | {bond} if bond is not None else built,
                      actions + (action,), stabilizer)

        visit({}, set(), (), automorphisms)
        return sorted(sequences)

    def prepare(self, log, include_negative=True):
        total = 0
        cache = Path(self.config["output_folder"]) / "route_cache"
        cache.mkdir(parents=True, exist_ok=True)
        for label in ("positive", "negative"):
            for old in cache.glob(label + "_*.jsonl.gz"):
                old.unlink()
        for label, reference in (("positive", self.positive), ("negative", self.negative)):
            if reference is None or (label == "negative" and not include_negative):
                continue
            fraction = self.config[label + "_trajectory_fraction"]
            for index, name in enumerate(reference.graphs):
                routes = self.enumerate_routes(reference, name)
                total += len(routes)
                if total > self.config["max_total_routes"]:
                    raise ValueError(f"{label} {name}: exceeded MAX_TOTAL_ROUTES; increase the limit explicitly")
                path = cache / f"{label}_{index:05d}.jsonl.gz"
                with gzip.open(path, "wt", encoding="utf-8") as handle:
                    for route in routes:
                        handle.write(json.dumps(route, separators=(",", ":")) + "\n")
                self.sequences[label][name] = Stage4RoutePool(path, len(routes))
                log.info(f"ROUTES {label} {name} | total={len(routes)}"
                         f" | sampled when species selected={math.ceil(len(routes) * fraction)}")

    def replay(self, reference, name, sequence, space, equivalent):
        env = self.api["AdvancedMoleculeEnv"](
            self.trained["atom_symbols"], self.trained["max_valency"], self.trained["max_atoms"])
        states, targets = [], []
        for action in sequence:
            options = reference.equivalent_actions(env, action) if equivalent else [action]
            legal = env.get_valid_actions()
            if any(a not in legal for a in options):
                raise ValueError(f"{name}: illegal route action")
            states.append(env.copy())
            targets.append([space.index[a] for a in options])
            env.step(action)
        if not env.terminated or not reference.isomorphic(env.to_graph(), reference.graphs[name]):
            raise ValueError(f"{name}: route does not reconstruct its graph")
        return states, targets

    def summary(self):
        return {"source_sha256": self.hashes, "positive_graphs": len(self.positive.graphs),
                "negative_graphs": len(self.negative.graphs) if self.negative else 0,
                "removed": self.removed,
                "enumeration_scope": "Stage I direct-final-bond action sequences, including STOP",
                "negative_label": "Invalid for this training round; reference absence is not chemical proof",
                "negative_species_per_epoch": min(self.config["negative_species_per_epoch"],
                                                    len(self.sequences["negative"])),
                "negative_sampling": "Uniform without replacement each epoch; average over selected species",
                "routes": {label: {name: {"total": len(routes), "sampled_when_species_selected":
                    math.ceil(len(routes) * self.config[label + "_trajectory_fraction"])}
                    for name, routes in pool.items()} for label, pool in self.sequences.items()}}


class Stage4Trainer:
    """Balanced positive imitation plus complete-route negative unlikelihood."""

    OUTPUTS = ("training.log", "metrics.csv", "config.json", "dataset_summary.json", "evaluation.json",
               "trained_growth_gnn.pt", "latest_growth_gnn.pt")

    def __init__(self, config):
        self.config = dict(config)
        self.validate()
        self.api = Stage4Loader.load(config["stage1_directory"])
        self.device = self.api["TrainingDevice"].resolve(config.get("device", "cpu"))
        self.writer = self.api["CompactJSON"]
        self.output = Path(config["output_folder"])
        outputs = {(self.output / name).resolve() for name in self.OUTPUTS}
        inputs = {Path(config[key]).resolve() for key in ("checkpoint_path", "positive_path", "negative_path")}
        if outputs & inputs:
            raise ValueError("Stage IV outputs must not overwrite any input")
        self.parent_hash = hashlib.sha256(Path(config["checkpoint_path"]).read_bytes()).hexdigest()
        self.parent = torch.load(config["checkpoint_path"], map_location="cpu", weights_only=True)
        self.trained = self.parent["config"]
        self.dataset = Stage4Dataset(config, self.trained, self.api)
        self.space = self.api["ActionSpace"](self.trained["max_atoms"], self.trained["atom_symbols"])
        if self.space.actions != [tuple(a) for a in self.parent["action_space"]]:
            raise ValueError("Parent checkpoint action order differs from Stage I")
        self.model = self.api["GrowthGNN"](self.trained["hidden_dims"], self.space,
                                           self.trained["atom_symbols"], self.trained["max_valency"])
        self.model.load_state_dict(self.parent["model_state_dict"], strict=True)
        self.model.to(self.device)
        self.contexts = list(self.dataset.positive.conditions)
        self.positive_rng = random.Random(config["seed"] + 1)
        self.negative_rng = random.Random(config["seed"] + 2)
        # Separate species selection from route sampling and positive examples.
        self.negative_species_rng = random.Random(config["seed"] + 3)
        self.selected_state = self.api["TrainingDevice"].cpu_data(self.model.state_dict())
        self.selected_epoch = 0
        self.history = []

    def validate(self):
        cfg = self.config
        for key in ("epochs", "eval_every", "eval_samples", "free_eval_attempts", "routes_per_batch",
                    "forward_batch_size", "threads", "max_routes_per_graph", "max_total_routes",
                    "max_enumeration_states_per_graph", "negative_species_per_epoch"):
            if type(cfg[key]) is not int or cfg[key] < 1:
                raise ValueError(f"{key} must be a positive integer")
        for key in ("positive_trajectory_fraction", "negative_trajectory_fraction"):
            if isinstance(cfg[key], bool) or not 0 < cfg[key] <= 1:
                raise ValueError(f"{key} must be in (0, 1]")
        for key in ("lambda_negative", "learning_rate", "gradient_clip", "reconstruction_tolerance",
                    "completion_tolerance"):
            if isinstance(cfg[key], bool) or not math.isfinite(cfg[key]) or cfg[key] < 0:
                raise ValueError(f"{key} must be finite and nonnegative")
        if cfg["learning_rate"] == 0 or cfg["gradient_clip"] == 0:
            raise ValueError("learning_rate and gradient_clip must be positive")
        if cfg["reconstruction_tolerance"] > 1 or cfg["completion_tolerance"] > 1:
            raise ValueError("Rate tolerances must not exceed 1")

    @staticmethod
    def negative_route_loss(log_probability):
        # expm1 is accurate near zero; float64 delays underflow on long routes.
        # A route probability rounded to 1 needs a finite numerical upper bound.
        # Metal has no float64. Transfer only the small vector of selected log
        # probabilities to CPU; autograd still propagates back into the GPU GNN.
        log_probability = log_probability.cpu().double().clamp(max=-1e-12)
        return torch.where(log_probability < -math.log(2),
                           -torch.log1p(-torch.exp(log_probability)),
                           -torch.log(-torch.expm1(log_probability)))

    def select_species(self, label):
        """Select a fresh uniform negative subset; retain all positive species."""
        pool = self.dataset.sequences[label]
        if label == "negative" and len(pool) > self.config["negative_species_per_epoch"]:
            names = self.negative_species_rng.sample(list(pool), self.config["negative_species_per_epoch"])
            return {name: pool[name] for name in names}
        return pool

    def batches(self, label, pool=None):
        if pool is None:
            pool = self.select_species(label)
        rng = self.positive_rng if label == "positive" else self.negative_rng
        batch = []
        for name, routes in pool.items():
            count = math.ceil(len(routes) * self.config[label + "_trajectory_fraction"])
            for sequence in routes.sample(rng, count):
                # Average over selected graphs, not the full negative set. Uniform
                # graph sampling preserves the full-set mean loss in expectation.
                batch.append((name, sequence, 1 / (len(pool) * count)))
                if len(batch) == self.config["routes_per_batch"]:
                    yield batch
                    batch = []
        if batch:
            yield batch

    def route_log_probabilities(self, batch, label):
        reference = self.dataset.positive if label == "positive" else self.dataset.negative
        states, conditions, targets, lengths = [], [], [], []
        for name, sequence, weight in batch:
            envs, ids = self.dataset.replay(reference, name, sequence, self.space, label == "positive")
            lengths.append(len(envs))
            states.extend(envs)
            targets.extend(ids)
            if label == "positive":
                conditions.extend([reference.composition(reference.graphs[name])] * len(envs))
        if label == "negative":
            # Each partial state is encoded/convolved once, then scored under all
            # compositions. Keep only its demonstrated action before averaging:
            # logsumexp of all-masked actions would have undefined gradients.
            all_logs = self.model.log_probs_over_contexts(states, self.contexts, self.config["forward_batch_size"])
            indices = torch.tensor([ids[0] for ids in targets], dtype=torch.long, device=all_logs.device)
            selected = all_logs.gather(2, indices[:, None, None].expand(-1, len(self.contexts), 1)).squeeze(2)
            logs = torch.logsumexp(selected, dim=1) - math.log(len(self.contexts))
            if not torch.isfinite(logs).all():
                raise RuntimeError("Non-finite negative target probabilities")
            return list(logs.split(lengths))
        step_logs = []
        size = self.config["forward_batch_size"]
        for start in range(0, len(states), size):
            inputs = self.model.encode(states[start:start+size], conditions[start:start+size])
            logits = self.model(*inputs)
            mask = torch.zeros_like(logits, dtype=torch.bool)
            for row, ids in enumerate(targets[start:start+size]):
                mask[row, ids] = True
            logs = torch.logsumexp(logits.log_softmax(-1).masked_fill(~mask, -torch.inf), dim=-1)
            if not torch.isfinite(logs).all():
                raise RuntimeError("Non-finite target probabilities")
            step_logs.append(logs)
        logs = torch.cat(step_logs)
        return list(logs.split(lengths))

    def train_epoch(self, optimizer, epoch):
        self.model.train()
        optimizer.zero_grad(set_to_none=True)
        losses = {"positive": 0.0, "negative": 0.0}
        counts = Counter()
        positive_gradients = []
        for label in ("positive", "negative"):
            if label == "negative" and self.config["lambda_negative"] == 0:
                continue
            started = time.monotonic()
            pool = self.select_species(label)
            counts[label + "_species"] = len(pool)
            if label == "negative":
                self.log.info(f"EPOCH {epoch} | negative species={len(pool)}/"
                              f"{len(self.dataset.sequences[label])} | selected IDs={','.join(pool)}")
            total_routes = sum(math.ceil(len(routes) * self.config[label + "_trajectory_fraction"])
                               for routes in pool.values())
            # Count completed routes rather than batches, including a partial final batch.
            with self.api["tqdm"](total=total_routes, desc=f"Epoch {epoch} {label}",
                                  unit="route", file=sys.stdout, position=1, leave=False,
                                  dynamic_ncols=True, mininterval=0.5) as progress:
                for index, batch in enumerate(self.batches(label, pool), 1):
                    logs = self.route_log_probabilities(batch, label)
                    per_route = [(-values.mean() if label == "positive" else
                                  self.negative_route_loss(values.cpu().double().sum())) for values in logs]
                    loss = sum(value * item[2] for value, item in zip(per_route, batch))
                    if not torch.isfinite(loss):
                        raise RuntimeError(f"Non-finite {label} loss")
                    scale = 1.0 if label == "positive" else self.config["lambda_negative"]
                    (scale * loss).backward()
                    losses[label] += float(loss.detach())
                    counts[label] += len(batch)
                    progress.set_postfix(loss=f"{losses[label]:.5g}", refresh=False)
                    progress.update(len(batch))
                    if time.monotonic() - started > 30:
                        self.log.info(f"EPOCH {epoch} {label} | routes processed={counts[label]} | batches={index}")
                        started = time.monotonic()
            if label == "positive":
                positive_gradients = [p.grad.detach().clone() if p.grad is not None else torch.zeros_like(p)
                                      for p in self.model.parameters()]
        positive_norm = sum(g.cpu().double().square().sum() for g in positive_gradients).sqrt()
        negative_norm = sum(((p.grad if p.grad is not None else torch.zeros_like(p)) - g).detach().cpu().double().square().sum()
                            for p, g in zip(self.model.parameters(), positive_gradients)).sqrt()
        # One update per epoch, as in Stage I; batch size controls memory only.
        norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config["gradient_clip"], error_if_nonfinite=True)
        optimizer.step()
        self.api["TrainingDevice"].synchronize(self.device)
        return {"positive_loss": losses["positive"], "negative_loss": losses["negative"],
                "weighted_negative_loss": self.config["lambda_negative"] * losses["negative"],
                "total_loss": losses["positive"] + self.config["lambda_negative"] * losses["negative"],
                "positive_routes": counts["positive"], "negative_routes": counts["negative"],
                "negative_species": counts["negative_species"],
                "positive_gradient_norm": float(positive_norm),
                "weighted_negative_gradient_norm": float(negative_norm),
                "gradient_norm": float(norm)}

    def evaluate(self, epoch, seed):
        """Measure conditioned reconstruction and free-policy negative recurrence."""
        self.model.eval()
        generator = torch.Generator().manual_seed(seed)
        metrics = {"epoch": epoch, "seed": seed}
        with torch.inference_mode():
            for mode in ("conditioned", "free"):
                conditions = ([c for c in self.contexts for _ in range(self.config["eval_samples"])]
                              if mode == "conditioned" else [None] * self.config["free_eval_attempts"])
                recovered, rejected_ids = Counter(), Counter()
                completed = correct = composition_ok = rejected = small = total_atoms = 0
                # Bound evaluation memory, including the free-policy context expansion.
                size = max(1, self.config["forward_batch_size"] // len(self.contexts))
                for start in range(0, len(conditions), size):
                    requested = conditions[start:start+size]
                    envs = [self.api["AdvancedMoleculeEnv"](self.trained["atom_symbols"],
                            self.trained["max_valency"], self.trained["max_atoms"]) for _ in requested]
                    for step in range(self.trained["max_steps"]):
                        active = [i for i,e in enumerate(envs) if not e.terminated]
                        if not active:
                            break
                        if mode == "free":
                            probs = self.model.log_probs_over_contexts(
                                [envs[i] for i in active], self.contexts,
                                self.config["forward_batch_size"]).exp().mean(1)
                        else:
                            inputs = self.model.encode([envs[i] for i in active], [requested[i] for i in active])
                            probs = self.model(*inputs).softmax(-1)
                        choices = torch.multinomial(probs.cpu(),1,generator=generator).squeeze(1)
                        for i,choice in zip(active,choices):
                            envs[i].step(self.space.actions[int(choice)])
                    for env,condition in zip(envs,requested):
                        if not env.terminated:
                            continue
                        completed += 1
                        total_atoms += len(env.node_types)
                        small += len(env.node_types) <= 3
                        match = self.dataset.positive.match(env.to_graph())
                        formula_ok = condition is None or env.composition() == condition
                        composition_ok += formula_ok
                        if match is not None and formula_ok:
                            correct += 1
                            recovered[match] += 1
                        negative = self.dataset.negative.match(env.to_graph()) if self.dataset.negative else None
                        if negative is not None:
                            rejected += 1
                            rejected_ids[negative] += 1
                total = len(conditions)
                metrics[mode] = {"attempts":total, "reconstruction_rate":correct/total,
                    "composition_rate":composition_ok/total if mode == "conditioned" else None,
                    "coverage":len(recovered), "recovered_species":dict(recovered),
                    "missing_species":sorted(self.dataset.positive.graphs.keys()-recovered.keys()),
                    "completion_rate":completed/total, "negative_rate":rejected/total,
                    "negative_rate_among_completed":rejected/completed if completed else None,
                    "negative_occurrences":dict(rejected_ids), "mean_atoms_completed":total_atoms/completed if completed else None,
                    "small_molecule_fraction_completed":small/completed if completed else None}
        self.log.info(f"EVAL epoch={epoch} | positive coverage={metrics['conditioned']['coverage']}/{len(self.dataset.positive.graphs)}"
                      f" | reconstruction={metrics['conditioned']['reconstruction_rate']:.3f}"
                      f" | free negative recurrence={metrics['free']['negative_rate']:.3f}"
                      f" | free STOP={metrics['free']['completion_rate']:.3f}")
        return metrics

    def acceptable(self, candidate, baseline):
        cfg = self.config
        c, b = candidate["conditioned"], baseline["conditioned"]
        if c["coverage"] < b["coverage"] or c["reconstruction_rate"] < b["reconstruction_rate"] - cfg["reconstruction_tolerance"]:
            return False
        if any(candidate[m]["completion_rate"] < baseline[m]["completion_rate"] - cfg["completion_tolerance"]
               for m in ("conditioned", "free")):
            return False
        if cfg["lambda_negative"] == 0:
            return True
        # Require a reduction when the baseline generated negatives. If it did
        # not, retain zero recurrence instead of demanding a rate below zero.
        old, new = baseline["free"]["negative_rate"], candidate["free"]["negative_rate"]
        return new < old if old > 0 else new == 0

    def rank(self, evaluation):
        c, f = evaluation["conditioned"], evaluation["free"]
        if self.config["lambda_negative"] == 0:
            return c["coverage"], c["reconstruction_rate"], f["completion_rate"]
        return -f["negative_rate"], c["coverage"], c["reconstruction_rate"], f["completion_rate"]

    def checkpoint(self, state, epoch):
        # Preserve the Stage I checkpoint schema so Stage II loads this model.
        return {"model_state_dict":self.api["TrainingDevice"].cpu_data(state), "epoch":epoch,
                "config":{**self.trained, "reference_path":self.config["positive_path"],
                          "output_folder":self.config["output_folder"], "device":str(self.device),
                          "learning_rate":self.config["learning_rate"],
                          "epochs":self.config["epochs"], "seed":self.config["seed"],
                          "trajectory_sample_fraction":self.config["positive_trajectory_fraction"],
                          "eval_every":self.config["eval_every"], "eval_samples":self.config["eval_samples"],
                          "gradient_clip":self.config["gradient_clip"], "threads":self.config["threads"]},
                "source_sha256":self.dataset.hashes["positive_path"], "action_space":self.space.actions,
                "stage4_config":self.config, "parent_checkpoint_sha256":self.parent_hash,
                "negative_source_sha256":self.dataset.hashes["negative_path"]}

    def run(self):
        torch.set_num_threads(self.config["threads"])
        torch.manual_seed(self.config["seed"])
        torch.use_deterministic_algorithms(True, warn_only=self.device.type == "mps")
        self.output.mkdir(parents=True,exist_ok=True)
        for name in self.OUTPUTS:
            (self.output/name).unlink(missing_ok=True)
        self.log = self.api["TrainingLogger"](self.output/"training.log")
        try:
            return self.train()
        except Exception:
            self.log.exception("STAGE IV FAILED; see traceback. Parent checkpoint and input datasets are preserved.")
            raise
        finally:
            self.log.close()

    def train(self):
        cfg = self.config
        self.log.info(f"STAGE IV | parent={cfg['checkpoint_path']} | lambda_negative={cfg['lambda_negative']}"
                      f" | positive fraction={cfg['positive_trajectory_fraction']} | negative fraction={cfg['negative_trajectory_fraction']}"
                      f" | negative species limit/epoch={cfg['negative_species_per_epoch']}")
        self.writer.write(self.output/"config.json",{**cfg,"parent_checkpoint_sha256":self.parent_hash,
                          "input_sha256":self.dataset.hashes,"policy_contexts_C_H_O":self.contexts})
        self.log.info(f"DEVICE: {self.device} | neural forward/backward uses this device; graph preparation and float64 route loss use CPU.")
        self.dataset.prepare(self.log, include_negative=cfg["lambda_negative"] > 0)
        self.writer.write(self.output/"dataset_summary.json",self.dataset.summary())
        optimizer = torch.optim.Adam(self.model.parameters(),lr=cfg["learning_rate"])
        baseline = self.evaluate(0,cfg["seed"]+10000)
        self.history.append(baseline)
        selected_evaluation = baseline
        # Parent fallback uses the expanded context set too, for a fair comparison.
        torch.save(self.checkpoint(self.selected_state,0),self.output/"trained_growth_gnn.pt")
        self.writer.write(self.output/"evaluation.json",{"baseline":baseline,"history":self.history})
        with (self.output/"metrics.csv").open("w",newline="") as handle, self.api["tqdm"](
                total=cfg["epochs"], desc="Stage IV epochs", unit="epoch", file=sys.stdout,
                position=0, dynamic_ncols=True, mininterval=0.5) as epoch_progress:
            columns=["epoch","positive_loss","negative_loss","weighted_negative_loss","total_loss",
                     "positive_routes","negative_routes","negative_species","positive_gradient_norm",
                     "weighted_negative_gradient_norm","gradient_norm","seconds"]
            writer=csv.DictWriter(handle,fieldnames=columns);writer.writeheader()
            for epoch in range(1,cfg["epochs"]+1):
                started=time.monotonic()
                metrics={"epoch":epoch,**self.train_epoch(optimizer,epoch),"seconds":time.monotonic()-started}
                writer.writerow(metrics);handle.flush()
                epoch_progress.set_postfix(loss=f"{metrics['total_loss']:.5g}", refresh=False)
                self.log.info(f"EPOCH {epoch}/{cfg['epochs']} | positive={metrics['positive_loss']:.6g}"
                              f" | negative={metrics['negative_loss']:.6g} | weighted negative={metrics['weighted_negative_loss']:.6g}"
                              f" | total={metrics['total_loss']:.6g} | routes={metrics['positive_routes']}+{metrics['negative_routes']}"
                              f" | negative species={metrics['negative_species']}"
                              f" | gradient positive={metrics['positive_gradient_norm']:.6g}"
                              f" negative(weighted)={metrics['weighted_negative_gradient_norm']:.6g}"
                              f" | gradient norm={metrics['gradient_norm']:.6g} | seconds={metrics['seconds']:.1f}")
                if epoch%cfg["eval_every"]==0 or epoch==cfg["epochs"]:
                    result=self.evaluate(epoch,cfg["seed"]+10000)
                    self.history.append(result)
                    eligible=self.acceptable(result,baseline)
                    if eligible and (self.selected_epoch==0 or self.rank(result)>self.rank(selected_evaluation)):
                        self.selected_epoch=epoch
                        selected_evaluation=result
                        self.selected_state=self.api["TrainingDevice"].cpu_data(self.model.state_dict())
                        torch.save(self.checkpoint(self.selected_state,epoch),self.output/"trained_growth_gnn.pt")
                    self.log.info(f"SELECTION | eligible={eligible} | selected_epoch={self.selected_epoch} (0 means parent fallback)")
                    self.writer.write(self.output/"evaluation.json",{"baseline":baseline,"history":self.history,
                                      "selected_epoch":self.selected_epoch,"selected":selected_evaluation})
                latest=self.checkpoint(self.model.state_dict(),epoch)
                latest.update(optimizer_state_dict=self.api["TrainingDevice"].cpu_data(optimizer.state_dict()),
                              positive_rng_state=self.positive_rng.getstate(),negative_rng_state=self.negative_rng.getstate(),
                              negative_species_rng_state=self.negative_species_rng.getstate(),
                              torch_rng_state=torch.get_rng_state())
                torch.save(latest,self.output/"latest_growth_gnn.pt")
                epoch_progress.update(1)
        self.model.load_state_dict(self.selected_state)
        verification=self.evaluate(self.selected_epoch,cfg["seed"]+20000)
        self.model.load_state_dict(self.parent["model_state_dict"])
        verification_parent=self.evaluate(0,cfg["seed"]+20000)
        verified=self.selected_epoch>0 and self.acceptable(verification,verification_parent)
        if self.selected_epoch>0 and not verified:
            self.log.info("Separate-seed verification failed; selected output falls back to the parent weights.")
            self.selected_epoch=0
            self.selected_state=self.api["TrainingDevice"].cpu_data(self.parent["model_state_dict"])
        self.model.load_state_dict(self.selected_state)
        torch.save(self.checkpoint(self.selected_state,self.selected_epoch),self.output/"trained_growth_gnn.pt")
        self.writer.write(self.output/"evaluation.json",{"baseline":baseline,"history":self.history,
            "candidate_verification":verification,"parent_verification":verification_parent,
            "verification_passed":verified,"selected_epoch":self.selected_epoch,
            "selection":"fine_tuned" if self.selected_epoch else "parent_fallback",
            "scope":"Reconstruction and free-policy sampling; PUCT and independent positive-only comparisons remain separate experiments."})
        self.log.info(f"FINISHED | selected epoch={self.selected_epoch} | output={self.output}"
                      " | latest checkpoint retains the final optimizer state; automatic resume is not implemented.")
        return {"selected_epoch":self.selected_epoch,"dataset":self.dataset.summary()}


# ==========================================
# PARAMETERS — edit here; paths relative to stageIV/
# ==========================================
STAGE1_DIRECTORY = "../stageI"
CHECKPOINT_PATH = "../stageI/output_stage1/trained_growth_gnn.pt"
POSITIVE_SPECIES_PATH = "../stageIII/species_update/expanded_species_graphs.json"
NEGATIVE_SPECIES_PATH = "../stageIII/species_update/unmatched_species.json"
OUTPUT_FOLDER = "output"

# L = L_positive + LAMBDA_NEGATIVE * L_negative. Set 0 for the positive-only baseline.
LAMBDA_NEGATIVE = 1.0
POSITIVE_TRAJECTORY_FRACTION = 0.20
NEGATIVE_TRAJECTORY_FRACTION = 0.20
# Uniform fresh subset each epoch; use a value >= the negative set size for all.
NEGATIVE_SPECIES_PER_EPOCH = 96
LEARNING_RATE = 0.0003
TOTAL_EPOCHS = 100
EVALUATE_EVERY = 10
EVALUATION_SAMPLES_PER_COMPOSITION = 100
FREE_EVALUATION_ATTEMPTS = 500
SEED = 12345
TRAINING_DEVICE = "mps"  # Apple GPU; set "cpu" explicitly for a CPU run.
CPU_THREADS = 1
GRADIENT_CLIP = 5.0
ROUTES_PER_BATCH = 8
FORWARD_BATCH_SIZE = 256
# These limits stop with an error, never substitute a partial trajectory pool.
MAX_ROUTES_PER_GRAPH = 1000000
MAX_ENUMERATION_STATES_PER_GRAPH = 10000000
MAX_TOTAL_ROUTES = 20000000
# Absolute rate tolerances; 0.02 permits a two-percentage-point change.
RECONSTRUCTION_TOLERANCE = 0.02
COMPLETION_TOLERANCE = 0.02

training_config = {
    "stage1_directory":STAGE1_DIRECTORY,"checkpoint_path":CHECKPOINT_PATH,
    "positive_path":POSITIVE_SPECIES_PATH,"negative_path":NEGATIVE_SPECIES_PATH,
    "output_folder":OUTPUT_FOLDER,"lambda_negative":LAMBDA_NEGATIVE,
    "positive_trajectory_fraction":POSITIVE_TRAJECTORY_FRACTION,
    "negative_trajectory_fraction":NEGATIVE_TRAJECTORY_FRACTION,
    "negative_species_per_epoch":NEGATIVE_SPECIES_PER_EPOCH,
    "learning_rate":LEARNING_RATE,"epochs":TOTAL_EPOCHS,"eval_every":EVALUATE_EVERY,
    "eval_samples":EVALUATION_SAMPLES_PER_COMPOSITION,"free_eval_attempts":FREE_EVALUATION_ATTEMPTS,
    "seed":SEED,"threads":CPU_THREADS,"device":TRAINING_DEVICE,"gradient_clip":GRADIENT_CLIP,
    "routes_per_batch":ROUTES_PER_BATCH,"forward_batch_size":FORWARD_BATCH_SIZE,
    "max_routes_per_graph":MAX_ROUTES_PER_GRAPH,"max_enumeration_states_per_graph":MAX_ENUMERATION_STATES_PER_GRAPH,
    "max_total_routes":MAX_TOTAL_ROUTES,"reconstruction_tolerance":RECONSTRUCTION_TOLERANCE,
    "completion_tolerance":COMPLETION_TOLERANCE,
}
trainer = Stage4Trainer(training_config)
trainer.run()
