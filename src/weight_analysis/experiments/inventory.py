"""Exact on-site inventory and evidence contracts for weights/language/entity/model."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import pandas as pd

from ..core import CheckpointSpec, WeightReader, selected_keys, weight_files
from ..core.blocks import parameter_location
from ..core.catalog import (
    CatalogEntry,
    CheckpointCatalog,
    CheckpointEvidence,
    ReferenceSpec,
    input_digest,
    parameter_space,
    stable_id,
)
from ..core.reporting import RunReporter
from ..provenance import file_sha256


def input_signature(directory):
    """Cheap mutation detection between units, in addition to full content hashes."""
    paths = [Path(directory)] + sorted(path for path in Path(directory).iterdir() if path.is_file())
    return {
        str(path): [path.stat().st_size, path.stat().st_mtime_ns, path.stat().st_ctime_ns]
        for path in paths
    }


def verify_signature(signature):
    for name, expected in signature.items():
        stat = Path(name).stat()
        if [stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns] != expected:
            raise ValueError(f"Pinned model input changed during run: {name}")


def tokenizer_digest(path):
    path = Path(path)
    if (path / "tokenizer.json").is_file():
        try:
            return stable_id(json.loads((path / "tokenizer.json").read_text()))
        except (OSError, ValueError):
            return None  # Missing vocabulary evidence must not discard valid encoder weights.
    names = [
        n
        for n in (
            "vocab.json",
            "vocab.txt",
            "sentencepiece.bpe.model",
            "spiece.model",
            "merges.txt",
        )
        if (path / n).is_file()
    ]
    return stable_id({name: file_sha256(path / name) for name in names}) if names else None


def discover_inventory(
    trained,
    originals,
    *,
    evidence_path=None,
    allow_unverified_origin=False,
    reference_kind="pretraining",
    default_fold=1,
    default_seed=None,
    reporter=None,
):
    reporter = reporter or RunReporter()
    trained, originals = (
        Path(trained).expanduser().resolve(),
        Path(originals).expanduser().resolve(),
    )
    if not trained.is_dir() or not originals.is_dir():
        raise FileNotFoundError(f"Weight roots unavailable on this machine: {trained}, {originals}")
    declarations = json.loads(Path(evidence_path).read_text()) if evidence_path else []
    indexed = {}
    for row in declarations:
        key = (row["language"], row["entity"], row["backbone"])
        if key in indexed:
            raise ValueError("Duplicate provenance identity")
        indexed[key] = row
    bases, records = {}, []
    for language in sorted(trained.iterdir()):
        if not language.is_dir() or language.name.startswith("."):
            continue
        for entity in sorted(language.iterdir()):
            if not entity.is_dir() or entity.name.startswith("."):
                continue
            for directory in sorted(entity.iterdir()):
                if not directory.is_dir() or directory.name.startswith("."):
                    continue

                def register():
                    if not weight_files(directory):
                        raise FileNotFoundError(f"No checkpoint: {directory}")
                    key = (language.name, entity.name, directory.name)
                    declared = indexed.get(key, {})
                    base = originals / directory.name
                    if directory.name not in bases:
                        if not weight_files(base):
                            raise FileNotFoundError(f"Missing exact reference: {base}")
                        bases[directory.name] = dict(
                            path=str(base),
                            content_id=input_digest(base),
                            tokenizer_id=tokenizer_digest(base),
                        )
                    info = bases[directory.name]
                    kind = declared.get("reference_kind", reference_kind)
                    initialization = (
                        info["content_id"]
                        if kind == "pretraining"
                        else declared.get("initialization_id")
                    )
                    if not initialization:
                        raise ValueError(
                            "Fine-tuned reference needs documented original initialization_id"
                        )
                    provenance = declared.get(
                        "provenance_id", "user-declared-pretraining-reference-layout"
                    )
                    if not provenance and not allow_unverified_origin:
                        raise ValueError(
                            "Training ancestry unavailable; supply evidence or explicit --allow-unverified-origin"
                        )
                    cp = CheckpointSpec(
                        directory.name,
                        language.name,
                        entity.name,
                        directory,
                        fold=declared.get("fold", default_fold),
                        seed=declared.get("seed", default_seed),
                    )
                    ev = CheckpointEvidence(
                        initialization,
                        provenance,
                        tokenizer_id=tokenizer_digest(directory),
                        training_run_id=declared.get("training_run_id"),
                        configuration_id=declared.get("configuration_id"),
                    )
                    content = input_digest(directory)
                    aid = stable_id(
                        dict(
                            content=content,
                            backbone=cp.backbone,
                            language=cp.language,
                            entity=cp.entity,
                            fold=cp.fold,
                            seed=cp.seed,
                            snapshot=cp.checkpoint_id,
                            run=ev.training_run_id,
                        )
                    )
                    entry = CatalogEntry(cp, aid, content, parameter_space(directory), ev)
                    with WeightReader(directory) as reader:
                        locations = [
                            parameter_location(reader, key) for key in selected_keys(reader)
                        ]
                    return dict(
                        input_signature=input_signature(directory),
                        reference_signature=input_signature(base),
                        **entry.row(),
                        path=str(directory),
                        reference=asdict(
                            ReferenceSpec(
                                info["path"],
                                kind=kind,
                                initialization_id=initialization,
                                tokenizer_id=info["tokenizer_id"],
                                provenance_id=declared.get("reference_provenance_id"),
                                scope="shared_declared",
                            )
                        ),
                        reference_content_id=info["content_id"],
                        ancestry_status="training_manifest_declared"
                        if evidence_path and provenance
                        else "user_declared_layout_not_independently_verified",
                        fold_status="declared" if cp.fold is not None else "unknown_not_inferred",
                        seed_status="declared" if cp.seed is not None else "unknown_not_inferred",
                        layers=sorted(
                            {loc["layer"] for loc in locations if loc["layer"] is not None}
                        ),
                        embedding_tensors=sorted(
                            loc["tensor"] for loc in locations if loc["component"] == "embeddings"
                        ),
                        nonlayer_tensors=sorted(
                            loc["tensor"]
                            for loc in locations
                            if loc["layer"] is None
                            and loc["component"] != "embeddings"
                            and "classifier" not in loc["tensor"]
                        ),
                        metadata_sha256={
                            p.name: file_sha256(p)
                            for p in directory.iterdir()
                            if p.is_file()
                            and p.name in ("encoding.json", "tokenizer.json", "config.json")
                        },
                    )

                row = reporter.attempt(str(directory), register, stage="inventory")
                if row is not None:
                    records.append(row)
    return records, reporter.events


def restore_catalog(records):
    entries, references = [], {}
    for row in records:
        cp = CheckpointSpec(
            row["backbone"],
            row["language"],
            row["entity"],
            row["path"],
            fold=row["fold"],
            seed=row["seed"],
            checkpoint_id=row.get("checkpoint_id"),
        )
        evidence = CheckpointEvidence(
            **{
                key: row.get(key)
                for key in (
                    "initialization_id",
                    "provenance_id",
                    "tokenizer_id",
                    "training_run_id",
                    "configuration_id",
                )
            }
        )
        entries.append(
            CatalogEntry(
                cp, row["artifact_id"], row["content_id"], row["parameter_space_id"], evidence
            )
        )
        references[row["artifact_id"]] = ReferenceSpec(**row["reference"])
    return CheckpointCatalog(tuple(entries)), references


def match_scores(scores, records):
    inventory = pd.DataFrame(
        [
            {
                key: row[key]
                for key in (
                    "language",
                    "entity",
                    "backbone",
                    "artifact_id",
                    "content_id",
                    "path",
                    "fold",
                    "seed",
                    "ancestry_status",
                )
            }
            for row in records
        ]
    )
    if inventory.empty:
        inventory = pd.DataFrame(
            columns=[
                "language",
                "entity",
                "backbone",
                "artifact_id",
                "content_id",
                "path",
                "fold",
                "seed",
                "ancestry_status",
            ]
        )
    if inventory.duplicated(["language", "entity", "backbone"]).any():
        raise ValueError(
            "Ambiguous checkpoint layout: several physical weights per score configuration"
        )
    merged = scores.drop(columns=["fold", "seed"]).merge(
        inventory,
        on=["language", "entity", "backbone"],
        how="outer",
        validate="1:1",
        indicator=True,
    )
    merged["match_status"] = (
        merged.pop("_merge")
        .astype(str)
        .map(
            {
                "both": "exact_layout_aggregate_only",
                "left_only": "score_without_checkpoint",
                "right_only": "checkpoint_without_score",
            }
        )
    )
    missing_model = merged.model_id.isna()
    merged.loc[missing_model, "model_id"] = [
        stable_id({key: row[key] for key in ("language", "entity", "backbone")})
        for row in merged[missing_model].to_dict("records")
    ]
    merged["fold_level_alignment"] = False
    return merged
