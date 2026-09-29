"""Shared defaults and validation; run_cycle.py supplies experiment overrides."""
import copy
import math

class Settings:
    ROUTES = {
        "max_routes_per_graph": 1000000,
        "max_enumeration_states_per_graph": 10000000,
        "max_total_routes": 20000000,
        "replay_cache_size": 2048,
    }
    INITIAL = {
        **ROUTES, "atom_symbols": {0: "H", 1: "O", 2: "C"},
        "max_valency": {0: 1, 1: 2, 2: 4}, "excluded_species": ["CO"],
        "max_atoms": 10, "max_steps": 24, "hidden_dims": [32, 64, 32, 32, 16],
        "learning_rate": 0.003, "epochs": 300, "trajectory_sample_fraction": 0.20,
        "eval_every": 50, "eval_samples": 100, "checkpoint_every": 50,
        "gradient_clip": 5.0, "seed": 12345, "threads": 1, "device": "mps", "gpu_ids": None, "log_level": 2,
    }
    TRAINING = {
        **ROUTES, "lambda_negative": 1.0, "positive_trajectory_fraction": 1.0,
        "negative_trajectory_fraction": 0.20, "hard_negative_ids": None,
        "learning_rate": 0.0003, "epochs": 100, "eval_every": 10, "eval_samples": 100,
        "free_eval_attempts": 500, "seed": 12345, "threads": 1, "device": "mps", "gpu_ids": None,
        "checkpoint_every": 10, "gradient_clip": 5.0, "routes_per_batch": 8,
        "forward_batch_size": 256, "reconstruction_tolerance": 0.02,
        "completion_tolerance": 0.02, "log_level": 2,
    }
    GENERATION = {
        "methods": ["puct", "direct"], "runs": 100, "attempts_per_run": 1000,
        "samples_per_composition": 100, "generation_mode": "free", "compositions": None,
        "temperature": 1.0, "seed": 24680, "batch_size": 100, "threads": 1, "device": "cpu",
        "save_media": True, "plot_dpi": 150, "puct_c": 2.0, "puct_uniform_fraction": 0.25,
        "puct_reference_reward": 0.5, "puct_candidate_reward": 1.0, "log_level": 2,
    }
    HARD_NEGATIVE = {"percent": 0.20, "max_count": 50, "puct_weight": 0.25}
    LABELING = {"unmatched_as_negative": False}

    @staticmethod
    def merge(defaults, overrides=None, runtime_keys=()):
        overrides = overrides or {}
        unknown = set(overrides) - set(defaults) - set(runtime_keys)
        if unknown:
            raise ValueError(f"Unknown settings: {sorted(unknown)}")
        return {**copy.deepcopy(defaults), **copy.deepcopy(overrides)}

    @staticmethod
    def training(config, initial=False):
        integers = ("epochs", "eval_every", "eval_samples", "threads", "checkpoint_every",
                    "max_routes_per_graph", "max_total_routes",
                    "max_enumeration_states_per_graph", "replay_cache_size")
        integers += ("max_atoms", "max_steps") if initial else (
            "free_eval_attempts", "routes_per_batch", "forward_batch_size")
        for key in integers:
            if type(config[key]) is not int or config[key] < 1:
                raise ValueError(f"{key} must be a positive integer")
        fractions = ("trajectory_sample_fraction",) if initial else (
            "positive_trajectory_fraction", "negative_trajectory_fraction")
        for key in fractions:
            value = config[key]
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not 0 < value <= 1:
                raise ValueError(f"{key} must be in (0, 1]")
        positive = ("learning_rate", "gradient_clip")
        nonnegative = () if initial else ("lambda_negative", "reconstruction_tolerance", "completion_tolerance")
        for key in positive + nonnegative:
            value = config[key]
            if (isinstance(value, bool) or not isinstance(value, (int, float))
                    or not math.isfinite(value) or value < 0 or (key in positive and value == 0)):
                raise ValueError(f"Invalid {key}: expected a finite {'positive' if key in positive else 'nonnegative'} number")
        if type(config["seed"]) is not int or config["seed"] < 0:
            raise ValueError("seed must be a nonnegative integer")
        gpu_ids = config.get("gpu_ids")
        if gpu_ids is not None and (not isinstance(gpu_ids, list) or not gpu_ids
                                    or any(type(index) is not int or index < 0 for index in gpu_ids)
                                    or len(gpu_ids) != len(set(gpu_ids))):
            raise ValueError("gpu_ids must be a nonempty list of unique nonnegative integers or None")
        if type(config["log_level"]) is not int or config["log_level"] not in (0, 1, 2):
            raise ValueError("log_level must be 0, 1, or 2")
        if initial:
            if not config["hidden_dims"] or any(type(w) is not int or w <= 0 for w in config["hidden_dims"]):
                raise ValueError("hidden_dims must contain positive integers")
        else:
            if config["reconstruction_tolerance"] > 1 or config["completion_tolerance"] > 1:
                raise ValueError("Rate tolerances must not exceed 1")
            ids = config["hard_negative_ids"]
            if ids is not None and (not isinstance(ids, list)
                    or not all(isinstance(item, str) and item for item in ids) or len(ids) != len(set(ids))):
                raise ValueError("hard_negative_ids must be unique nonempty strings or None")
