"""Initial imitation training and fine-tuning; one optimizer update per epoch."""
import csv
import hashlib
import json
import math
import random
import shutil
import sys
import time
from collections import Counter
from pathlib import Path
import torch
import torch.nn.functional as F
from tqdm import tqdm
from .molecule import ActionSpace, ReferenceSpecies
from .model import TrainingDevice, GrowthGNN
from .storage import CompactJSON
from .reporter import TrainingLogger
from .trajectories import TrajectoryDataset, RouteCache
from .molecule import SpeciesInput
from .config import Settings
from .checkpoint import Checkpoint
from .evaluator import PolicyEvaluator, InitialEvaluator

class InitialTrainer(InitialEvaluator):
    """Balanced imitation, readable saved logs, and independent free rollouts."""

    def __init__(self, config):
        config = Settings.merge(Settings.INITIAL, config,
                                ("reference_path", "output_folder", "route_cache_folder"))
        Settings.training(config, initial=True)
        self.config = dict(config)
        self.device = TrainingDevice.resolve(config["device"])
        self.output = Path(config["output_folder"])
        # Reuse the selected folder; outputs are replaced on each training run.
        self.log = None
        self.atom_symbols = config["atom_symbols"]
        self.max_valency = config["max_valency"]
        self.seed = config["seed"]

    def clear_previous_outputs(self):
        """Remove generated reports/checkpoint so shorter reruns leave no stale results."""
        names = {
            "config.json", "formation_sequences.json", "training_examples.json",
            "metrics.csv", "trained_growth_gnn.pt", "latest_growth_gnn.pt", "baseline_greedy.json",
            "baseline_greedy.csv", "generated_species_inventory.json",
            "generated_species_inventory.csv",
        }
        # Delete only known Stage I outputs; preserve inputs, unrelated files,
        # and timestamped historical folders from earlier script versions.
        paths = {self.output / name for name in names}
        for pattern in ("greedy_epoch_*.json", "greedy_epoch_*.csv"):
            paths.update(self.output.glob(pattern))
        source = Path(self.config["reference_path"]).resolve()
        if any(path.resolve() == source for path in paths):
            raise ValueError("Reference input must not use a generated output filename")
        for path in paths:
            if path.is_file() or path.is_symlink():
                path.unlink()
        checkpoints=self.output/'checkpoints'
        if checkpoints.is_dir():
            shutil.rmtree(checkpoints)




    def run(self, resume=False):
        self.resume = resume
        self.output.mkdir(parents=True, exist_ok=True)
        self.log = TrainingLogger(self.output / "training.log", append=resume,
                                  level=self.config["log_level"])
        try:
            self.train()
        except Exception:
            self.log.exception("TRAINING FAILED: see traceback; existing logs are retained.")
            raise
        # Close file handlers on success or failure so the saved log is complete.
        finally:
            self.log.close()

    @staticmethod
    def cache_demonstrations(model, references, action_space):
        """Cache static inputs and exact partial-graph IDs, never learned features."""
        cached, graph_ids = {}, {}
        device = next(model.parameters()).device
        for name, trajectories in references.pool.items():
            cached[name] = []
            for items in trajectories:
                inputs = model.encode([item["env"] for item in items],
                                      [item["condition"] for item in items])
                targets = torch.zeros(len(items), len(action_space.actions),
                                      dtype=torch.bool, device=device)
                route_ids = []
                for i, item in enumerate(items):
                    targets[i, item["target_indices"]] = True
                    env = item["env"]
                    # Preserve node order: action indices refer to these exact atoms.
                    # Composition and target actions do not enter graph convolutions.
                    key = (tuple(env.node_types), tuple(env.current_bonds),
                           tuple(sorted(env.bonds.items())))
                    route_ids.append(graph_ids.setdefault(key, len(graph_ids)))
                if not torch.all((targets & inputs[-1]).any(dim=1)):
                    raise ValueError("Demonstration target was masked out")
                cached[name].append((inputs, targets, route_ids))
        return cached

    @staticmethod
    def training_logits(model, selected):
        """Convolve each distinct partial graph once; score every example separately."""
        inputs = tuple(torch.cat([entry[0][i] for entry in selected]) for i in range(5))
        targets = torch.cat([entry[1] for entry in selected])
        unique, first_rows, inverse = {}, [], []
        for entry in selected:
            for graph_id in entry[2]:
                if graph_id not in unique:
                    unique[graph_id] = len(first_rows)
                    first_rows.append(len(inverse))
                inverse.append(unique[graph_id])
        device = inputs[0].device
        first_rows = torch.tensor(first_rows, dtype=torch.long, device=device)
        inverse = torch.tensor(inverse, dtype=torch.long, device=device)
        h, pooled = model.graph_features(*(value[first_rows] for value in inputs[:3]))
        # Indexing sums all repeated examples' gradients into the shared features.
        # Recompute features each epoch after the previous optimizer update.
        logits = model.action_logits(h[inverse], pooled[inverse], inputs[3], inputs[4])
        return logits, targets, len(unique)

    @staticmethod
    def select_trajectories(cached, fraction, rng):
        """Sample ceil(N_s * fraction) routes and give each species total weight 1/S."""
        selected, step_weights = [], []
        for entries in cached.values():
            count = ReferenceSpecies.sample_count(len(entries), fraction)
            # rng persists across epochs; random.sample selects without replacement.
            for entry in rng.sample(entries, count):
                selected.append(entry)
                steps = len(entry[1])
                # s = species, k = sampled trajectory, t = action step:
                # each step receives 1 / (S * K_s * T_s,k).
                step_weights.append(torch.full((steps,), 1 / (len(cached) * count * steps)))
        return selected, torch.cat(step_weights)

    def train(self):
        cfg = self.config
        Settings.training(cfg, initial=True)
        random.seed(self.seed)
        torch.manual_seed(self.seed)
        torch.set_num_threads(cfg["threads"])
        torch.use_deterministic_algorithms(True, warn_only=self.device.type == "mps")
        # This RNG selects training trajectories independently of demonstration generation.
        rng = random.Random(self.seed + 1)
        references = ReferenceSpecies(cfg["reference_path"], cfg["excluded_species"],
                                      self.atom_symbols, self.max_valency, cfg["max_atoms"])
        # Validate the input/settings before clearing the previous reports/model.
        # A resumed run keeps its log and trims metrics to the saved epoch.
        signature = Checkpoint.signature(cfg, {"positive": references.source_hash})
        resumed = Checkpoint.load(self.output, "initial", signature, self.resume)
        if resumed is None:
            self.clear_previous_outputs()
        else:
            Checkpoint.trim_metrics(self.output / "metrics.csv", resumed["epoch"])
            self.log.info(f"RESUME initial training | completed epoch={resumed['epoch']}")
            if resumed["epoch"] % cfg["checkpoint_every"] == 0 or resumed["epoch"] == cfg["epochs"]:
                Checkpoint.save(self.output/"checkpoints"/f"epoch_{resumed['epoch']:04d}.pt", resumed)
        action_space = ActionSpace(cfg["max_atoms"], self.atom_symbols)
        self.log.objective("STAGE I | Initial supervised imitation training")
        self.log.info(f"DEVICE: {self.device} | MPS uses the Apple GPU; graph preparation stays on CPU.")
        self.log.info(f"Reference={cfg['reference_path']} | SHA256={references.source_hash}")
        self.log.info(f"Active species={len(references.graphs)} | distinct compositions="
                      f"{len(references.conditions)} | excluded={cfg['excluded_species']}")
        self.log.info("One molecule per episode. Condition order: C,H,O. "
                      "Masks use chemistry only; STOP is allowed before valence saturation.")
        self.log.info("Training and evaluation use supplied reference structures. "
                      "These are reconstruction metrics, not held-out molecular generalization.")
        self.log.info(f"Config: {json.dumps(cfg, sort_keys=True)}")
        CompactJSON.write(self.output / "config.json", cfg)
        route_cache = RouteCache(cfg.get("route_cache_folder") or self.output/"route_cache", cfg, cfg)
        references.build_demonstrations(action_space, route_cache, cfg)
        references.save_sequences(self.output / "formation_sequences.json", cfg["trajectory_sample_fraction"])
        references.save_demonstrations(self.output / "training_examples.json", action_space)
        for name, trajectories in references.pool.items():
            example = trajectories[0]
            self.log.info(f"REFERENCE {name:<7} | {self.condition_text(example[0]['condition'])}"
                          f" | atoms={len(references.graphs[name])}"
                          f" | unique sequences={len(trajectories)}"
                          f" | sampled/epoch={references.sample_count(len(trajectories), cfg['trajectory_sample_fraction'])}"
                          f" | steps={len(example)}")
            self.log.info("  " + " -> ".join(action_space.describe(s["action"], self.atom_symbols)
                                           for s in example))
        model = GrowthGNN(cfg["hidden_dims"], action_space, self.atom_symbols, self.max_valency).to(self.device)
        optimizer = torch.optim.Adam(model.parameters(), lr=cfg["learning_rate"])
        self.log.info(f"MODEL node_features={model.feature_dim} | hidden_dims={cfg['hidden_dims']}"
                      f" | trainable_parameters={sum(p.numel() for p in model.parameters())}")
        # Save an untrained baseline to distinguish learning from success due to masks alone.
        baseline = (resumed["baseline"] if resumed else
                    self.evaluate(model, references, action_space, 0, 1, "greedy"))
        self.save_evaluation(baseline, "baseline_greedy")
        # Only teacher-forced partial graphs enter the network, never full targets.
        cached = self.cache_demonstrations(model, references, action_space)
        self.log.info("Graph convolutions are shared by identical partial graphs within each epoch; "
                      "compositions, target actions and example weights remain separate.")
        start_epoch = resumed["epoch"] if resumed else 0
        if resumed:
            model.load_state_dict(resumed["model_state_dict"])
            optimizer.load_state_dict(resumed["optimizer_state_dict"])
            Checkpoint.restore_random(resumed["random_states"], self.device, trajectories=rng)
        datasets = {"positive": SpeciesInput.read(cfg["reference_path"]), "negative": []}

        def training_checkpoint(epoch):
            return {
                "model_state_dict": TrainingDevice.cpu_data(model.state_dict()),
                "optimizer_state_dict": TrainingDevice.cpu_data(optimizer.state_dict()),
                "epoch": epoch, "config": cfg, "source_sha256": references.source_hash,
                "action_space": action_space.actions, "resume_version": Checkpoint.VERSION,
                "phase": "initial", "resume_signature": signature, "baseline": baseline,
                "datasets": datasets,
                "random_states": Checkpoint.random_states(self.device, trajectories=rng),
            }

        metrics_path = self.output / "metrics.csv"
        with metrics_path.open("a" if resumed else "w", newline="", encoding="utf-8") as metrics_file, tqdm(
                total=cfg["epochs"], initial=start_epoch, desc="Initial training epochs", unit="epoch", file=sys.stdout,
                dynamic_ncols=True, mininterval=0.5,
                disable=not self.log.progress_enabled) as progress:
            writer = csv.writer(metrics_file)
            if not resumed: writer.writerow(["epoch", "mean_species_nll", "equivalent_action_accuracy",
                             "mean_target_probability", "gradient_norm", "seconds"])
            started = time.monotonic()
            for epoch in range(start_epoch + 1, cfg["epochs"] + 1):
                epoch_start = time.monotonic()
                # A fresh subset per species, with no duplicates within this epoch.
                # Cached tensors describe teacher-forced states, not model rollouts.
                selected, weights = self.select_trajectories(
                    cached, cfg["trajectory_sample_fraction"], rng)
                weights = weights.to(self.device)
                model.train()
                # Clear previous gradients before the single optimizer update for this epoch.
                optimizer.zero_grad()
                logits, targets, unique_graphs = self.training_logits(model, selected)
                log_probs = F.log_softmax(logits, dim=-1)
                # Numerically stable log(sum of probabilities of acceptable actions).
                # With one target, this is ordinary cross-entropy; equivalent targets
                # share credit instead of penalizing an interchangeable attachment site.
                target_log_prob = torch.logsumexp(log_probs.masked_fill(~targets, -torch.inf), dim=1)
                loss = -(weights * target_log_prob).sum()
                if not torch.isfinite(loss):
                    raise RuntimeError("Non-finite training loss")
                # Backpropagate through the GNN and all action heads. Gradient clipping
                # limits the combined gradient norm when an update would be unusually large.
                loss.backward()
                gradient_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["gradient_clip"])
                optimizer.step()
                TrainingDevice.synchronize(self.device)
                # Teacher-action accuracy is measured on these demonstrated states BEFORE
                # the update; it is not the complete-molecule free-generation success rate.
                action_correct = targets.gather(1, logits.argmax(1, keepdim=True)).squeeze(1)
                accuracy = float((weights * action_correct).sum())
                probability = float((weights * target_log_prob.detach().exp()).sum())
                elapsed = time.monotonic() - epoch_start
                writer.writerow([epoch, float(loss.detach()), accuracy, probability,
                                 float(gradient_norm), elapsed])
                metrics_file.flush()
                progress.set_postfix(loss=f"{float(loss.detach()):.5f}", refresh=False)
                self.log.info(f"EPOCH {epoch:04d}/{cfg['epochs']} | loss={float(loss.detach()):.5f}"
                              f" | sampled sequences={len(selected)}"
                              f" | unique partial graphs={unique_graphs}/{len(targets)} steps"
                              f" | teacher-action accuracy={accuracy:.1%}"
                              f" | mean correct-action probability={probability:.4f}"
                              f" | grad_norm={float(gradient_norm):.3f} | {elapsed:.2f}s")
                if epoch % cfg["eval_every"] == 0 or epoch == cfg["epochs"]:
                    result = self.evaluate(model, references, action_space, epoch, 1, "greedy",
                                           save_trace=epoch == cfg["epochs"])
                    self.save_evaluation(result, f"greedy_epoch_{epoch:04d}")
                saved = training_checkpoint(epoch)
                Checkpoint.save(self.output / "latest_growth_gnn.pt", saved)
                if epoch % cfg["eval_every"] == 0 or epoch == cfg["epochs"]:
                    Checkpoint.save(self.output / "trained_growth_gnn.pt", saved)
                if epoch % cfg["checkpoint_every"] == 0 or epoch == cfg["epochs"]:
                    Checkpoint.save(self.output / "checkpoints" / f"epoch_{epoch:04d}.pt", saved)
                progress.update(1)
            # Recreate the selected output even if interruption followed the last epoch save.
            Checkpoint.save(self.output / "trained_growth_gnn.pt", training_checkpoint(cfg["epochs"]))
            # Final stochastic evaluation measures isomer diversity beyond one greedy output.
            sampled = self.evaluate(model, references, action_space, cfg["epochs"],
                                    cfg["eval_samples"], "sampled")
            self.save_evaluation(sampled, "generated_species_inventory")
            self.log.info(f"FINISHED | elapsed={time.monotonic() - started:.1f}s"
                          f" | sampled reconstruction={sampled['rate']:.1%}"
                          f" | sampled coverage={sampled['coverage']}/{len(references.graphs)}")
            self.log.info(f"Saved log: {self.output / 'training.log'}")
            self.log.info(f"Saved model: {self.output / 'trained_growth_gnn.pt'}")
            self.log.info(f"Saved generated inventory: {self.output / 'generated_species_inventory.csv'}")


class Trainer(PolicyEvaluator):
    """Balanced positive imitation plus complete-route negative unlikelihood."""

    OUTPUTS = ("training.log", "metrics.csv", "config.json", "dataset_summary.json", "evaluation.json",
               "trained_growth_gnn.pt", "latest_growth_gnn.pt")

    DEFAULTS = Settings.TRAINING

    def __init__(self, config):
        config = Settings.merge(self.DEFAULTS, config,
                                ("checkpoint_path", "positive_path", "negative_path",
                                 "output_folder", "route_cache_folder", "round_number",
                                 "round_total"))
        self.config = dict(config)
        self.validate()
        self.device = TrainingDevice.resolve(config.get("device", "cpu"))
        self.writer = CompactJSON
        self.output = Path(config["output_folder"])
        outputs = {(self.output / name).resolve() for name in self.OUTPUTS}
        inputs = {Path(config[key]).resolve() for key in ("checkpoint_path", "positive_path", "negative_path")}
        if outputs & inputs:
            raise ValueError("Training outputs must not overwrite any input")
        self.parent_hash = hashlib.sha256(Path(config["checkpoint_path"]).read_bytes()).hexdigest()
        self.parent = torch.load(config["checkpoint_path"], map_location="cpu", weights_only=True)
        self.trained = self.parent["config"]
        self.dataset = TrajectoryDataset(config, self.trained)
        self.space = ActionSpace(self.trained["max_atoms"], self.trained["atom_symbols"])
        if self.space.actions != [tuple(a) for a in self.parent["action_space"]]:
            raise ValueError("Parent checkpoint action order differs from Stage I")
        self.model = GrowthGNN(self.trained["hidden_dims"], self.space,
                                           self.trained["atom_symbols"], self.trained["max_valency"])
        self.model.load_state_dict(self.parent["model_state_dict"], strict=True)
        self.model.to(self.device)
        self.contexts = list(self.dataset.positive.conditions)
        self.positive_rng = random.Random(config["seed"] + 1)
        self.negative_rng = random.Random(config["seed"] + 2)
        self.selected_state = TrainingDevice.cpu_data(self.model.state_dict())
        self.selected_epoch = 0
        self.history = []

    def validate(self):
        Settings.training(self.config)

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
        """Use all positives and the pipeline-selected hard-negative species."""
        pool = self.dataset.sequences[label]
        if label == "negative" and self.config['hard_negative_ids'] is not None:
            names=self.config['hard_negative_ids']
            missing=set(names)-set(pool)
            if missing:
                raise ValueError(f'Hard-negative IDs are absent from the negative dataset: {sorted(missing)}')
            return {name:pool[name] for name in names}
        return pool

    def batches(self, label, pool=None):
        if pool is None:
            pool = self.select_species(label)
        rng = self.positive_rng if label == "positive" else self.negative_rng
        batch = []
        for name, routes in pool.items():
            count = math.ceil(len(routes) * self.config[label + "_trajectory_fraction"])
            for sequence in routes.sample(rng, count):
                # Average over active graphs. For negatives, the pipeline has
                # deliberately chosen a hard subset rather than estimating the
                # mean loss of every stored negative graph.
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
            with tqdm(total=total_routes, desc=f"Epoch {epoch} {label}",
                                  unit="route", file=sys.stdout, position=1, leave=False,
                                  dynamic_ncols=True, mininterval=0.5,
                                  disable=not self.log.progress_enabled) as progress:
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
        TrainingDevice.synchronize(self.device)
        return {"positive_loss": losses["positive"], "negative_loss": losses["negative"],
                "weighted_negative_loss": self.config["lambda_negative"] * losses["negative"],
                "total_loss": losses["positive"] + self.config["lambda_negative"] * losses["negative"],
                "positive_routes": counts["positive"], "negative_routes": counts["negative"],
                "negative_species": counts["negative_species"],
                "positive_gradient_norm": float(positive_norm),
                "weighted_negative_gradient_norm": float(negative_norm),
                "gradient_norm": float(norm)}




    def checkpoint(self, state, epoch):
        # Preserve the Stage I checkpoint schema so Stage II loads this model.
        return {"model_state_dict":TrainingDevice.cpu_data(state), "epoch":epoch,
                "config":{**self.trained, "reference_path":self.config["positive_path"],
                          "output_folder":self.config["output_folder"], "device":str(self.device),
                          "learning_rate":self.config["learning_rate"],
                          "epochs":self.config["epochs"], "seed":self.config["seed"],
                          "trajectory_sample_fraction":self.config["positive_trajectory_fraction"],
                          "eval_every":self.config["eval_every"], "eval_samples":self.config["eval_samples"],
                          "gradient_clip":self.config["gradient_clip"], "threads":self.config["threads"]},
                "source_sha256":self.dataset.hashes["positive_path"], "action_space":self.space.actions,
                "stage4_config":self.config, "parent_checkpoint_sha256":self.parent_hash,
                "negative_source_sha256":self.dataset.hashes["negative_path"],
                "datasets":self.dataset.records}

    def run(self, resume=False):
        self.resume = resume
        torch.set_num_threads(self.config["threads"])
        torch.manual_seed(self.config["seed"])
        torch.use_deterministic_algorithms(True, warn_only=self.device.type == "mps")
        self.output.mkdir(parents=True,exist_ok=True)
        self.resume_signature = Checkpoint.signature(self.config,
            {**self.dataset.hashes, "parent": self.parent_hash})
        self.resumed = Checkpoint.load(self.output, "fine_tuning", self.resume_signature, resume)
        if self.resumed is None:
            for name in self.OUTPUTS:
                (self.output/name).unlink(missing_ok=True)
            checkpoints = self.output/"checkpoints"
            if checkpoints.is_dir():
                shutil.rmtree(checkpoints)
        else:
            Checkpoint.trim_metrics(self.output/"metrics.csv", self.resumed["epoch"])
            epoch = self.resumed["epoch"]
            if epoch % self.config["checkpoint_every"] == 0 or epoch == self.config["epochs"]:
                Checkpoint.save(self.output/"checkpoints"/f"epoch_{epoch:04d}.pt", self.resumed)
        self.log = TrainingLogger(self.output/"training.log", append=self.resumed is not None,
                                  level=self.config["log_level"])
        try:
            return self.train()
        except Exception:
            self.log.exception("FINE-TUNING FAILED; see traceback. Parent checkpoint and input datasets are preserved.")
            raise
        finally:
            self.log.close()

    def train(self):
        cfg = self.config
        round_label = (f"ROUND {cfg['round_number']}/{cfg['round_total']} | "
                       if "round_number" in cfg and "round_total" in cfg else "")
        self.log.objective(f"{round_label}STAGE IV | Fine-tune and select checkpoint")
        self.log.info(f"FINE-TUNING | parent={cfg['checkpoint_path']} | lambda_negative={cfg['lambda_negative']}"
                      f" | positive fraction={cfg['positive_trajectory_fraction']} | negative fraction={cfg['negative_trajectory_fraction']}"
                      f" | hard negatives={len(cfg['hard_negative_ids']) if cfg['hard_negative_ids'] is not None else 'all'}")
        self.writer.write(self.output/"config.json",{**cfg,"parent_checkpoint_sha256":self.parent_hash,
                          "input_sha256":self.dataset.hashes,"policy_contexts_C_H_O":self.contexts})
        self.log.info(f"DEVICE: {self.device} | neural forward/backward uses this device; graph preparation and float64 route loss use CPU.")
        self.dataset.prepare(self.log, include_negative=cfg["lambda_negative"] > 0)
        self.writer.write(self.output/"dataset_summary.json",self.dataset.summary())
        optimizer = torch.optim.Adam(self.model.parameters(),lr=cfg["learning_rate"])
        resumed = self.resumed
        if resumed:
            baseline = resumed["baseline"]
            self.history = resumed["history"]
            self.selected_epoch = resumed["selected_epoch"]
            self.selected_state = resumed["selected_model_state_dict"]
            selected_evaluation = resumed["selected_evaluation"]
            self.model.load_state_dict(resumed["model_state_dict"])
            optimizer.load_state_dict(resumed["optimizer_state_dict"])
            Checkpoint.restore_random(resumed["random_states"], self.device,
                                      positive=self.positive_rng, negative=self.negative_rng)
            self.log.info(f"RESUME fine-tuning | completed epoch={resumed['epoch']}")
        else:
            baseline = self.evaluate(0,cfg["seed"]+10000)
            self.history.append(baseline)
            selected_evaluation = baseline
        start_epoch = resumed["epoch"] if resumed else 0
        # A Stage IV update is always adopted.  Evaluation remains a report,
        # rather than a gate that can revert the run to the parent network.
        Checkpoint.save(self.output/"trained_growth_gnn.pt", self.checkpoint(self.selected_state,self.selected_epoch))
        self.writer.write(self.output/"evaluation.json",{"baseline":baseline,"history":self.history})
        with (self.output/"metrics.csv").open("a" if resumed else "w",newline="") as handle, tqdm(
                total=cfg["epochs"], initial=start_epoch, desc="Fine-tuning epochs", unit="epoch", file=sys.stdout,
                position=0, dynamic_ncols=True, mininterval=0.5,
                disable=not self.log.progress_enabled) as epoch_progress:
            columns=["epoch","positive_loss","negative_loss","weighted_negative_loss","total_loss",
                     "positive_routes","negative_routes","negative_species","positive_gradient_norm",
                     "weighted_negative_gradient_norm","gradient_norm","seconds"]
            writer=csv.DictWriter(handle,fieldnames=columns)
            if not resumed:writer.writeheader()
            for epoch in range(start_epoch+1,cfg["epochs"]+1):
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
                    self.writer.write(self.output/"evaluation.json",{"baseline":baseline,"history":self.history,
                                      "selection":"pending final epoch adoption"})
                latest=self.checkpoint(self.model.state_dict(),epoch)
                latest.update(
                    resume_version=Checkpoint.VERSION, phase="fine_tuning",
                    resume_signature=self.resume_signature,
                    optimizer_state_dict=TrainingDevice.cpu_data(optimizer.state_dict()),
                    baseline=baseline, history=self.history, selected_epoch=self.selected_epoch,
                    selected_model_state_dict=self.selected_state, selected_evaluation=selected_evaluation,
                    random_states=Checkpoint.random_states(self.device,
                        positive=self.positive_rng, negative=self.negative_rng))
                Checkpoint.save(self.output/"latest_growth_gnn.pt",latest)
                if epoch % cfg["checkpoint_every"] == 0 or epoch == cfg["epochs"]:
                    Checkpoint.save(self.output/"checkpoints"/f"epoch_{epoch:04d}.pt",latest)
                epoch_progress.update(1)
        # Keep the final optimizer update.  This ensures every completed
        # self-updating round becomes the parent of the next round.
        self.selected_epoch=cfg["epochs"]
        self.selected_state=TrainingDevice.cpu_data(self.model.state_dict())
        selected_evaluation=self.history[-1]
        self.model.load_state_dict(self.selected_state)
        Checkpoint.save(self.output/"trained_growth_gnn.pt", self.checkpoint(self.selected_state,self.selected_epoch))
        self.writer.write(self.output/"evaluation.json",{"baseline":baseline,"history":self.history,
            "selected":selected_evaluation,"selected_epoch":self.selected_epoch,
            "selection":"final_epoch_adopted",
            "scope":"Every completed Stage IV update is adopted. Reconstruction and free-policy metrics are reports, not checkpoint rejection criteria."})
        self.log.info(f"FINISHED | selected epoch={self.selected_epoch} | output={self.output}"
                      " | latest checkpoint supports continuation from the next epoch.")
        return {"selected_epoch":self.selected_epoch,"dataset":self.dataset.summary()}
