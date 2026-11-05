"""Portable CPU example: F1 selection, disjoint units, shared geometry and Parquet.

Run with the package installed: python -m weight_analysis.examples.comparative_geometry
All inputs and results live in a temporary directory; no network or scheduler is used.
"""

from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory

import pandas as pd
import torch
from safetensors.torch import save_file

from weight_analysis import (
    CheckpointEvidence,
    CheckpointSpec,
    ParameterSelection,
    ReferenceSpec,
    RunReporter,
    aggregate_geometry,
    analyze_shared_adaptation,
    build_catalog,
    compare_selected,
    comparison_view,
    input_digest,
    merge_comparative,
    read_dataset,
    select_performance_pairs,
    write_dataset,
)


def main() -> None:
    with TemporaryDirectory(prefix="comparative-geometry-") as temporary:
        root = Path(temporary)
        original = root / "original"
        original.mkdir()
        weights = {
            f"bert.encoder.layer.{i}.attention.self.query.weight": torch.eye(3) * (i + 1)
            for i in range(2)
        }
        save_file(weights, original / "model.safetensors")
        (original / "config.json").write_text(json.dumps({"model_type": "bert"}))
        initialization = input_digest(original)
        checkpoints, evidence = [], {}
        for entity, scale in (("disease", 1.0), ("symptom", 0.5)):
            for run in (0, 1):
                path = root / entity / str(run)
                path.mkdir(parents=True)
                save_file(
                    {key: value + (run + 1) * scale * 0.1 for key, value in weights.items()},
                    path / "model.safetensors",
                )
                (path / "config.json").write_text(json.dumps({"model_type": "bert"}))
                cp = CheckpointSpec(
                    "bert", "es", entity, path, fold=1, seed=7, checkpoint_id=f"run-{run}"
                )
                checkpoints.append(cp)
                evidence[cp.uid] = CheckpointEvidence(
                    initialization, "synthetic-manifest", training_run_id=f"{entity}-{run}"
                )
        catalog = build_catalog(checkpoints, evidence, reporter=RunReporter(verbose=False))
        performance = pd.DataFrame(
            [
                dict(
                    score_id=e.artifact_id + ":score",
                    artifact_id=e.artifact_id,
                    strict_f1_mean=0.6 + 0.1 * int(e.checkpoint.checkpoint_id[-1]),
                    strict_f1_std=0.01,
                    score_scope="checkpoint",
                    protocol_id="synthetic-strict",
                    evaluation_id="synthetic-held-out",
                )
                for e in catalog.entries
            ]
        )
        selection = select_performance_pairs(checkpoints=catalog, performance=performance)
        common = dict(
            selection=selection,
            references={"bert": ReferenceSpec(original)},
            parameters=ParameterSelection(components=("q",)),
            device="cpu",
            metrics=("l2_distance", "cosine"),
            drift_metrics=("delta_energy",),
            cache_dir=root / "cache",
        )
        # Each caller can execute a separate fragment. Selection stays frozen.
        pieces = [
            compare_selected(
                **common, pair_ids=[cid], reporter=RunReporter(log_path=root / "events.jsonl")
            )
            for cid in selection.pairs.comparison_id
        ]
        combined = aggregate_geometry(merge_comparative(pieces))
        write_dataset(combined, root / "result")
        restored = read_dataset(root / "result")
        view = comparison_view(restored, table="geometry_aggregated")
        print(view.query("scope == 'encoder'")[["metric", "value", "abs_f1_gap"]])
        shared = analyze_shared_adaptation(
            checkpoints=catalog,
            axis="entity",
            references=common["references"],
            device="cpu",
            parameters=common["parameters"],
            performance=performance,
            member_ids=[
                e.artifact_id for e in catalog.entries if e.checkpoint.checkpoint_id == "run-0"
            ],
            reporter=RunReporter(log_path=root / "shared.jsonl"),
        )
        print(
            aggregate_geometry(shared)
            .tables["geometry_aggregated"]
            .query("scope == 'encoder'")[["metric", "value", "status"]]
        )
        assert all(piece.complete for piece in pieces) and shared.complete
        print("Synthetic comparative analysis and Parquet round-trip completed.")


if __name__ == "__main__":
    main()
