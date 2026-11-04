"""Content fingerprints and deterministic identities for independent computation units."""

from __future__ import annotations

import hashlib
import json
import platform
from pathlib import Path
from typing import Any, Iterable

import pandas as pd

from . import __version__
from .core import CheckpointSpec, weight_files
from .results.models import data_digest


def canonical_json(value: Any) -> str:
    """Encode validated configuration with stable ordering and no special floats."""
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def file_sha256(path: str | Path) -> str:
    """Hash file contents in bounded chunks without loading a checkpoint into memory."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def checkpoint_fingerprint(path: str | Path) -> dict[str, Any]:
    """Fingerprint the selected weight format, all its shards, and model configuration."""
    directory = Path(path).expanduser().resolve()
    files = weight_files(directory)
    if not files:
        raise FileNotFoundError(f"No supported weights in {directory}")
    if files[0].suffix == ".json":
        mapping = json.loads(files[0].read_text(encoding="utf8")).get("weight_map")
        if not isinstance(mapping, dict) or not mapping:
            raise ValueError(f"Invalid shard index: {files[0]}")
        for name in sorted(set(mapping.values())):
            if not isinstance(name, str) or Path(name).name != name:
                raise ValueError(f"Unsafe shard name: {name!r}")
            files.append(directory / name)
    if (directory / "config.json").is_file():
        files.append(directory / "config.json")
    return {"path": str(directory), "files": {p.name: file_sha256(p) for p in files}}


def computation_provenance(
    *,
    configuration: dict[str, Any],
    checkpoints: Iterable[CheckpointSpec],
    originals: dict[str, Path],
    tables: dict[str, pd.DataFrame | None],
    selection: dict[str, Any] | None = None,
    intermediates: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Record immutable content identities, numerical versions, and experiment dimensions."""
    import numpy
    import scipy
    import torch

    by_path: dict[str, dict[str, Any]] = {}
    inputs = []
    for checkpoint in sorted(checkpoints, key=lambda item: item.uid):
        path = str(checkpoint.path)
        if path not in by_path:
            by_path[path] = checkpoint_fingerprint(path)
        inputs.append(
            {
                "checkpoint": checkpoint.uid,
                "backbone": checkpoint.backbone,
                "language": checkpoint.language,
                "entity": checkpoint.entity,
                "fold": checkpoint.fold,
                "seed": checkpoint.seed,
                "checkpoint_id": checkpoint.checkpoint_id,
                **by_path[path],
            }
        )
    base_inputs = {}
    for backbone, path in sorted(originals.items()):
        key = str(path)
        if key not in by_path:
            by_path[key] = checkpoint_fingerprint(path)
        base_inputs[backbone] = by_path[key]
    table_inputs = {}
    for name, table_df in tables.items():
        if table_df is not None:
            table_inputs[name] = data_digest(table_df)
    versions = {
        "weight_analysis": __version__,
        "python": platform.python_version(),
        "numpy": numpy.__version__,
        "scipy": scipy.__version__,
        "torch": torch.__version__,
        "pandas": pd.__version__,
        "cuda_runtime": torch.version.cuda,
    }
    execution_device = configuration.get("options", {}).get("device", "cpu")
    if execution_device.startswith("cuda:"):
        selected_device = torch.device(execution_device)
        versions["cuda_device"] = {
            "name": torch.cuda.get_device_name(selected_device),
            "capability": list(torch.cuda.get_device_capability(selected_device)),
        }
    payload = {
        "configuration": configuration,
        "inputs": inputs,
        "originals": base_inputs,
        "tables": table_inputs,
        "versions": versions,
        "definitions": "selective-v2-device-resident",
        "selection": selection or {},
        "intermediates": intermediates or {},
    }
    payload["unit_id"] = hashlib.sha256(canonical_json(payload).encode()).hexdigest()
    return payload
