"""Atomic training checkpoints with optimizer, dataset and random-number state."""
import os
import random
import tempfile
from pathlib import Path
import torch
from .storage import Storage

class Checkpoint:
    VERSION = 1

    @staticmethod
    def save(path, payload):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        # Saving to an open file avoids embedding a random temporary filename.
        with tempfile.NamedTemporaryFile(dir=path.parent, prefix=path.name+".",
                                         suffix=".tmp", delete=False) as handle:
            temporary = Path(handle.name)
            try:
                torch.save(payload, handle)
                handle.flush()
                os.fsync(handle.fileno())
            except BaseException:
                temporary.unlink(missing_ok=True)
                raise
        try:
            temporary.replace(path)
        finally:
            temporary.unlink(missing_ok=True)

    @staticmethod
    def copy(source, destination):
        import shutil
        destination=Path(destination)
        destination.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(dir=destination.parent, delete=False) as handle:
            temporary=Path(handle.name)
            try:
                with Path(source).open("rb") as reader:
                    shutil.copyfileobj(reader, handle)
                handle.flush()
                os.fsync(handle.fileno())
            except BaseException:
                temporary.unlink(missing_ok=True)
                raise
        try:
            temporary.replace(destination)
        finally:
            temporary.unlink(missing_ok=True)

    @staticmethod
    def random_states(device, **generators):
        result = {"python": random.getstate(), "torch": torch.get_rng_state(),
                  "generators": {key: rng.getstate() for key, rng in generators.items()}}
        if device.type == "mps":
            result["mps"] = torch.mps.get_rng_state().cpu()
        return result

    @staticmethod
    def restore_random(states, device, **generators):
        random.setstate(states["python"])
        torch.set_rng_state(states["torch"])
        for key, rng in generators.items():
            rng.setstate(states["generators"][key])
        if device.type == "mps":
            if "mps" not in states:
                raise ValueError("MPS random state is missing from the resume checkpoint")
            torch.mps.set_rng_state(states["mps"])

    @staticmethod
    def signature(config, inputs):
        # Locations may change when a complete saved run is copied; contents and
        # training settings determine whether optimizer state can be continued.
        paths = {"output_folder", "reference_path", "positive_path", "negative_path",
                 "checkpoint_path", "route_cache_folder"}
        return Storage.fingerprint({"config": {k:v for k,v in config.items() if k not in paths},
                                    "inputs": inputs})

    @classmethod
    def load(cls, output, phase, signature, resume):
        path = Path(output) / "latest_growth_gnn.pt"
        if not resume:
            return None
        if not path.exists():
            archived = sorted((Path(output)/"checkpoints").glob("epoch_*.pt"),
                              key=lambda p: int(p.stem.split("_")[-1]))
            if not archived:
                return None
            path = archived[-1]
        result = torch.load(path, map_location="cpu", weights_only=True)
        if result.get("resume_version") != cls.VERSION or result.get("phase") != phase:
            raise ValueError(f"Checkpoint does not support {phase} training resume: {path}")
        if result.get("resume_signature") != signature:
            raise ValueError("Training settings or datasets changed since the saved checkpoint")
        return result

    @staticmethod
    def trim_metrics(path, epoch):
        """Discard rows written after the last fully saved epoch following a crash."""
        import csv
        path = Path(path)
        if not path.exists():
            if epoch:
                raise ValueError(f"Missing metrics for a saved training checkpoint: {path}")
            return
        with path.open(newline="") as handle:
            reader = csv.DictReader(handle)
            columns = reader.fieldnames
            rows = [row for row in reader if int(row["epoch"]) <= epoch]
        if [int(row["epoch"]) for row in rows] != list(range(1, epoch + 1)):
            raise ValueError(f"Metrics do not cover saved epochs 1 through {epoch}: {path}")
        with path.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=columns)
            writer.writeheader()
            writer.writerows(rows)
