"""Streaming scalar review bundles for historical and normalized Parquet results."""

from __future__ import annotations

import argparse
import gzip
import json
import os
import shutil
from pathlib import Path
from uuid import uuid4

import pyarrow as pa
import pyarrow.parquet as pq


def export_review(input_dir, output_dir, *, legacy_encoder_pairs=False, batch_size=30_000):
    source = Path(input_dir).expanduser().resolve(strict=True)
    destination = Path(output_dir).expanduser().absolute()
    if destination.exists() or destination.is_symlink() or Path(str(destination) + ".zip").exists():
        raise FileExistsError(f"Review output already exists: {destination}")
    if source == destination or source in destination.parents:
        raise ValueError("Review output must be outside the input tree")
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name("." + destination.name + "." + uuid4().hex + ".partial")
    temporary.mkdir()
    inventory = []
    try:
        for path in sorted(source.rglob("*.parquet")):
            parquet = pq.ParquetFile(path)
            columns = [
                field.name
                for field in parquet.schema_arrow
                if (
                    pa.types.is_integer(field.type)
                    or pa.types.is_floating(field.type)
                    or pa.types.is_boolean(field.type)
                    or pa.types.is_string(field.type)
                    or pa.types.is_large_string(field.type)
                )
            ]
            relative = path.relative_to(source)
            output = temporary / relative.with_suffix(".csv.gz")
            output.parent.mkdir(parents=True, exist_ok=True)
            encoder_only = (
                legacy_encoder_pairs
                and path.name.startswith(("03_", "04_", "05_"))
                and "level" in columns
            )
            rows = 0
            with gzip.open(output, "wt", encoding="utf8") as stream:
                for batch in parquet.iter_batches(columns=columns, batch_size=batch_size):
                    frame = batch.to_pandas()
                    if encoder_only:
                        frame = frame[frame.level == "encoder"]
                    if frame.empty:
                        continue
                    frame.to_csv(stream, index=False, header=rows == 0)
                    rows += len(frame)
            if not rows:
                output.unlink()
            inventory.append(
                dict(
                    input=str(relative),
                    source_rows=parquet.metadata.num_rows,
                    exported_rows=rows,
                    columns=parquet.schema_arrow.names,
                    exported_columns=columns,
                    only_encoder=encoder_only,
                )
            )
        (temporary / "inventory.json").write_text(json.dumps(inventory, indent=2), encoding="utf8")
        (temporary / "inventario.txt").write_text(
            "\n\n".join(
                f"{row['input']}\nFilas originales: {row['source_rows']}\nFilas exportadas: {row['exported_rows']}\nSolo encoder: {row['only_encoder']}\nColumnas: {', '.join(row['columns'])}"
                for row in inventory
            ),
            encoding="utf8",
        )
        os.rename(temporary, destination)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    archive = shutil.make_archive(str(destination), "zip", root_dir=destination)
    print("Archivo generado:", archive, flush=True)
    return Path(archive)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument(
        "--legacy-encoder-pairs",
        action="store_true",
        help="Reproduce old 03/04/05 encoder-only export",
    )
    parser.add_argument("--batch-size", type=int, default=30_000)
    args = parser.parse_args(argv)
    export_review(
        args.input_dir,
        args.output_dir,
        legacy_encoder_pairs=args.legacy_encoder_pairs,
        batch_size=args.batch_size,
    )


if __name__ == "__main__":
    main()
