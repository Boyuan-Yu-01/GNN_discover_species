"""Evaluate reconstruction and rejected-species recurrence."""
import csv
from .storage import CompactJSON
from collections import Counter
import torch
from .molecule import AdvancedMoleculeEnv


class PolicyEvaluator:
    """Parent-relative evaluation and checkpoint selection."""

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
                    envs = [AdvancedMoleculeEnv(self.trained["atom_symbols"],
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


class InitialEvaluator:
    """Conditioned reconstruction reports for initial positive-only training."""

    def condition_text(self, condition):
        return ", ".join(f"{symbol}={count}" for symbol, count in zip(("C", "H", "O"), condition))

    def evaluate(self, model, references, action_space, epoch, samples, mode, save_trace=False):
        # Free generation uses the model's own actions, unlike teacher-forced training.
        # No reference next-action labels are consulted while constructing these molecules.
        model.eval()
        # Evaluate each unique composition once per sample, rather than giving an
        # isomer pair twice as many requests merely because it has two reference names.
        conditions = [condition for condition in references.conditions for _ in range(samples)]
        environments = [AdvancedMoleculeEnv(self.atom_symbols, self.max_valency,
                                            self.config["max_atoms"]) for _ in conditions]
        traces = [[] for _ in conditions]
        # Hitting the step cap is not successful termination, even if the final graph matches.
        reasons = ["step_limit"] * len(conditions)
        # Use a separate fixed evaluation RNG so sampling does not alter training choices.
        generator = torch.Generator().manual_seed(self.seed + 10000)
        # Evaluation does not update weights or need the gradient history.
        with torch.no_grad():
            for step in range(self.config["max_steps"]):
                # Finished molecules leave the active batch; the others keep growing.
                active = [i for i, env in enumerate(environments) if not env.terminated]
                if not active:
                    break
                inputs = model.encode([environments[i] for i in active], [conditions[i] for i in active])
                logits = model(*inputs)
                probabilities = logits.softmax(-1)
                # Greedy chooses the largest probability; sampled draws from the same
                # distribution and can reveal multiple isomers for one condition.
                if mode == "greedy":
                    choices = logits.argmax(-1).cpu()
                else:
                    choices = torch.multinomial(probabilities.cpu(), 1, generator=generator).squeeze(1)
                for row, index in enumerate(active):
                    action = action_space.actions[int(choices[row])]
                    env = environments[index]
                    if save_trace:
                        traces[index].append({
                            "step": step + 1, "before": env.describe(), "action": list(action),
                            "description": action_space.describe(action, self.atom_symbols),
                            "probability": float(probabilities[row, choices[row]]),
                        })
                    env.step(action)
                    if env.terminated:
                        reasons[index] = "STOP"
        counts = Counter()
        summaries = {condition: {"correct": 0, "composition": 0, "stopped": 0, "outputs": Counter()}
                     for condition in references.conditions}
        records = []
        for i, (condition, env) in enumerate(zip(conditions, environments)):
            match = references.match(env.to_graph())
            composition_ok = env.composition() == condition
            # A formula match is insufficient: require an exact reference structure,
            # the requested composition, and an explicit STOP action.
            correct = reasons[i] == "STOP" and composition_ok and match in references.conditions[condition]
            summary = summaries[condition]
            summary["correct"] += int(correct)
            summary["composition"] += int(composition_ok)
            summary["stopped"] += int(reasons[i] == "STOP")
            summary["outputs"][match or "unknown"] += 1
            if correct:
                counts[match] += 1
            graph = env.to_graph()
            # Export compact base atoms/bonds and H counts alongside the complete graph.
            # For H or H2, retain H as a base atom because no heavy atom exists.
            base = [u for u in graph if self.atom_symbols[graph.nodes[u]["atom"]] != "H"]
            pure_h = not base
            if pure_h:
                base = list(graph)
            mapping = {u: j for j, u in enumerate(base)}
            record = {
                "epoch": epoch, "mode": mode, "sample": i + 1,
                "requested_C_H_O": list(condition), "generated_C_H_O": list(env.composition()),
                "species_key": match or "Unknown",
                "base_atom_indices": {j: self.atom_symbols[graph.nodes[u]["atom"]]
                                      for u, j in mapping.items()},
                "base_bonds": [(mapping[u], mapping[v], d["order"])
                               for u, v, d in graph.edges(data=True) if u in mapping and v in mapping],
                "attached_h_per_base_atom": [] if pure_h else [
                    sum(self.atom_symbols[graph.nodes[v]["atom"]] == "H" for v in graph.neighbors(u))
                    for u in base],
                "in_species_training_reference": match is not None,
                "correct_for_condition": correct, "termination": reasons[i],
                "graph": env.record(),
            }
            if save_trace:
                record["trace"] = traces[i]
            records.append(record)
        # Reconstruction rate counts successful samples. Coverage counts distinct species.
        # A run can cover every species while still producing many incorrect samples.
        rate = sum(s["correct"] for s in summaries.values()) / len(conditions)
        self.log.info(f"EVALUATION epoch={epoch} mode={mode} | correct structure+composition="
                      f"{rate:.1%} | coverage={len(counts)}/{len(references.graphs)}"
                      f" | STOP={sum(r == 'STOP' for r in reasons)}/{len(reasons)}")
        for condition, summary in summaries.items():
            self.log.info(
                f"  [{self.condition_text(condition)}] expected={','.join(references.conditions[condition])}"
                f" | correct={summary['correct']}/{samples}"
                f" | composition={summary['composition']}/{samples}"
                f" | generated={dict(summary['outputs'])}")
        for name in references.graphs:
            self.log.info(f"  SPECIES {name:<7} | correctly generated={counts[name]}")
        # Keep the console readable by showing detailed steps for OH, H2O and CH4.
        # Every final greedy trace is still included in the saved evaluation JSON.
        if save_trace:
            for record in records:
                if record["requested_C_H_O"] in ([0, 1, 1], [0, 2, 1], [1, 4, 0]):
                    self.log.info(f"ROLLOUT [{self.condition_text(record['requested_C_H_O'])}] "
                                  f"-> {record['species_key']} ({record['termination']})")
                    for item in record["trace"]:
                        self.log.info(f"  step {item['step']:02d} | {item['before']} | "
                                      f"{item['description']} | p={item['probability']:.4f}")
        return {"rate": rate, "coverage": len(counts), "counts": dict(counts), "records": records}

    def save_evaluation(self, evaluation, filename):
        records = evaluation["records"]
        CompactJSON.write(self.output / f"{filename}.json", evaluation)
        # JSON retains graphs and optional traces; CSV provides the compact Excel-readable table.
        columns = ["epoch", "mode", "sample", "requested_C_H_O", "generated_C_H_O", "species_key",
                   "base_atom_indices", "base_bonds", "attached_h_per_base_atom",
                   "in_species_training_reference", "correct_for_condition", "termination"]
        with (self.output / f"{filename}.csv").open("w", newline="", encoding="utf-8-sig") as output:
            writer = csv.DictWriter(output, fieldnames=columns)
            writer.writeheader()
            for record in records:
                writer.writerow({key: record[key] for key in columns})
