"""Content identities and explicit evidence for comparative checkpoint studies."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Mapping, Sequence

import pandas as pd

from ..provenance import checkpoint_fingerprint
from .checkpoints import CheckpointSpec
from .parameters import selected_keys
from .reading import WeightReader, config_metadata


def stable_id(value) -> str:
    def normalize(item):
        if item is None or item is pd.NA:
            return None
        if hasattr(item, "tolist") and not isinstance(item, (str, bytes)):
            return normalize(item.tolist())
        if hasattr(item, "item"):
            item = item.item()
        if isinstance(item, float) and math.isnan(item):
            return None
        if isinstance(item, dict):
            return {str(key): normalize(val) for key, val in item.items()}
        if isinstance(item, (list, tuple)):
            return [normalize(val) for val in item]
        return item

    def convert(item):
        if hasattr(item, "item"):
            return item.item()
        if isinstance(item, Path):
            return str(item)
        raise TypeError(f"Unsupported identifier value: {type(item).__name__}")

    encoded = json.dumps(
        normalize(value), sort_keys=True, separators=(",", ":"), allow_nan=False, default=convert
    )
    return hashlib.sha256(encoded.encode()).hexdigest()


def input_digest(path: str | Path) -> str:
    """Identify weights/configuration contents independently of filesystem location."""
    return stable_id(checkpoint_fingerprint(path)["files"])


def geometry_dimensions(a: CatalogEntry, b: CatalogEntry | None = None) -> dict:
    """Dimension labels permit partitioning without embedding selection/F1 in geometry."""
    dimensions = {}
    for field in ("language", "entity", "backbone", "fold", "seed"):
        value = getattr(a.checkpoint, field)
        dimensions[field] = (
            value if b is None or value == getattr(b.checkpoint, field) else f"__cross_{field}__"
        )
    dimensions["comparison_type"] = "checkpoint" if b is None else "direct"
    return dimensions


def parameter_space(path) -> str:
    with WeightReader(path) as reader:
        keys = [(key, reader.shape_dtype(key)[0]) for key in selected_keys(reader)]
    config = config_metadata(path, strict=True)
    architecture = {key: value for key, value in config.items() if key != "architectures"}
    return stable_id({"parameters": keys, "architecture": architecture})


@dataclass(frozen=True)
class CheckpointEvidence:
    """Externally supplied training evidence; shapes alone cannot establish ancestry."""

    initialization_id: str
    provenance_id: str
    tokenizer_id: str | None = None
    training_run_id: str | None = None
    configuration_id: str | None = None

    def __post_init__(self):
        if not self.initialization_id:
            raise ValueError("Declare the content identity of the initialization")


@dataclass(frozen=True)
class CatalogEntry:
    checkpoint: CheckpointSpec
    artifact_id: str
    content_id: str
    parameter_space_id: str
    evidence: CheckpointEvidence

    def row(self):
        cp = self.checkpoint
        return {
            "artifact_id": self.artifact_id,
            "content_id": self.content_id,
            "parameter_space_id": self.parameter_space_id,
            "backbone": cp.backbone,
            "language": cp.language,
            "entity": cp.entity,
            "fold": cp.fold,
            "seed": cp.seed,
            "checkpoint_id": cp.checkpoint_id,
            **asdict(self.evidence),
        }


@dataclass(frozen=True)
class CheckpointCatalog:
    entries: tuple[CatalogEntry, ...]
    issues: tuple[dict, ...] = ()

    @property
    def table(self):
        return pd.DataFrame([entry.row() for entry in self.entries])

    @property
    def by_id(self):
        return {entry.artifact_id: entry for entry in self.entries}


def build_catalog(
    checkpoints: Sequence[CheckpointSpec],
    evidence: Mapping[str, CheckpointEvidence],
    *,
    reporter=None,
    on_error="continue",
) -> CheckpointCatalog:
    """Hash and register candidates, retaining unavailable inputs as visible incidents."""
    from .reporting import RunReporter

    reporter = reporter or RunReporter()
    entries = []
    for checkpoint in sorted(checkpoints, key=lambda cp: cp.uid):

        def register():
            ev = evidence[checkpoint.uid]
            content = input_digest(checkpoint.path)
            row = {
                "content": content,
                "backbone": checkpoint.backbone,
                "language": checkpoint.language,
                "entity": checkpoint.entity,
                "fold": checkpoint.fold,
                "seed": checkpoint.seed,
                "snapshot": checkpoint.checkpoint_id,
                "run": ev.training_run_id,
            }
            return CatalogEntry(
                checkpoint, stable_id(row), content, parameter_space(checkpoint.path), ev
            )

        entry = reporter.attempt(checkpoint.uid, register, on_error=on_error, stage="catalog")
        if entry is not None:
            entries.append(entry)
    if len({entry.artifact_id for entry in entries}) != len(entries):
        raise ValueError(
            "Duplicate logical checkpoints; declare independent run/snapshot identities"
        )
    return CheckpointCatalog(
        tuple(sorted(entries, key=lambda e: e.artifact_id)),
        tuple(e for e in reporter.events if e["status"] == "failed"),
    )


@dataclass(frozen=True)
class ReferenceSpec:
    path: str | Path
    kind: str = "pretraining"
    initialization_id: str | None = None
    tokenizer_id: str | None = None
    provenance_id: str | None = None
    fold: int | None = None
    seed: int | None = None
    scope: str = "matched"

    def __post_init__(self):
        if self.kind not in ("pretraining", "finetuned_baseline"):
            raise ValueError("Reference kind must be pretraining or finetuned_baseline")
        if self.scope not in ("matched", "shared_declared"):
            raise ValueError("Reference scope must be matched or shared_declared")
        if self.kind == "finetuned_baseline" and not self.initialization_id:
            raise ValueError("A fine-tuned baseline must declare its initialization")


def reference_for(entry: CatalogEntry, references: Mapping[str, ReferenceSpec]):
    ref = references.get(entry.artifact_id, references.get(entry.checkpoint.backbone))
    if ref is None:
        raise ValueError(f"Missing reference for {entry.artifact_id}")
    return ref


def validate_reference(
    entry: CatalogEntry, ref: ReferenceSpec, *, allow_unverified=False, content_id=None
):
    content = content_id if content_id is not None else input_digest(ref.path)
    initialization = content if ref.kind == "pretraining" else ref.initialization_id
    if initialization != entry.evidence.initialization_id:
        raise ValueError("Reference and checkpoint declare different initialization contents")
    if not entry.evidence.provenance_id and not allow_unverified:
        raise ValueError("Training provenance is required; shape equality cannot prove ancestry")
    cp = entry.checkpoint
    if ref.kind == "finetuned_baseline":
        if not ref.provenance_id and not allow_unverified:
            raise ValueError("Baseline training provenance is required")
        if ref.scope == "matched" and (ref.fold, ref.seed) != (cp.fold, cp.seed):
            raise ValueError("Baseline fold/seed differ; explicitly declare a shared reference")
    return stable_id(
        {
            "content": content,
            "kind": ref.kind,
            "initialization": initialization,
            "scope": ref.scope,
            "fold": ref.fold,
            "seed": ref.seed,
        }
    )
