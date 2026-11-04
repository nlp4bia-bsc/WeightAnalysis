"""Compute, persist, and merge tiny independent CPU analyses with no external data."""

from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory

import pandas as pd
import torch
from safetensors.torch import save_file

from weight_analysis import (
    CheckpointSpec,
    analyze,
    merge_results,
    read_result,
    write_result,
)


def write_checkpoint(directory: Path, tensors: dict[str, torch.Tensor]) -> None:
    """Write explicitly requested synthetic model files inside a temporary directory."""
    directory.mkdir(parents=True)
    save_file(tensors, directory / "model.safetensors")
    config = {"model_type": "bert", "hidden_size": 3, "num_hidden_layers": 2}
    (directory / "config.json").write_text(json.dumps(config), encoding="utf-8")


def main() -> None:
    """Run two drift units and a norm-only fingerprint, then verify persistence."""
    with TemporaryDirectory(prefix="weight-analysis-example-") as temporary:
        root = Path(temporary)
        base = {
            f"bert.encoder.layer.{layer}.attention.self.query.weight": torch.eye(
                3, dtype=torch.float64
            )
            * (layer + 1)
            for layer in range(2)
        }
        original = root / "original"
        write_checkpoint(original, base)
        checkpoints = []
        for language, update in (("es", 0.1), ("en", 0.2)):
            trained = root / language
            write_checkpoint(trained, {name: value + update for name, value in base.items()})
            checkpoints.append(
                CheckpointSpec(
                    backbone="tiny_bert",
                    language=language,
                    entity="disease",
                    path=trained,
                    fold=1,
                    seed=42,
                    checkpoint_id=f"synthetic-{language}",
                )
            )

        originals = {"tiny_bert": original}
        common = {
            "analysis": "pretraining_drift",
            "originals": originals,
            "components": ["q"],
            "metrics": ["relative_frobenius"],
        }
        units = [analyze(checkpoints=[checkpoint], **common) for checkpoint in checkpoints]
        combined = merge_results(reversed(units))
        joint = analyze(checkpoints=checkpoints, **common)
        columns = sorted(combined.data.columns)

        def ordered(frame: pd.DataFrame) -> pd.DataFrame:
            return (
                frame.loc[:, columns].sort_values(["checkpoint", "tensor"]).reset_index(drop=True)
            )

        pd.testing.assert_frame_equal(ordered(combined.data), ordered(joint.data))
        output = root / "drift.parquet"
        write_result(combined, output)
        restored = read_result(output)
        pd.testing.assert_frame_equal(ordered(restored.data), ordered(combined.data))

        fingerprints = analyze(
            analysis="spectral_fingerprints",
            checkpoints=checkpoints,
            originals=originals,
            sources=["delta"],
            components=["q"],
            metrics=["matrix_fro"],
        )
        print(combined.data.to_string(index=False))
        print(f"Validated {len(combined.data)} drift rows and {len(fingerprints.data)} norm rows.")
        print("Synthetic files were confined to the temporary directory and will be removed.")


if __name__ == "__main__":
    main()
