"""Export all persisted studies with explicit measurement and endpoint F1 metadata."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.parquet as pq

from weight_analysis.experiments.compact.query import dataset
from weight_analysis.experiments.compact.storage import SCHEMA
from weight_analysis.provenance import file_sha256

STUDIES = {
    "01_pretraining_drift": (
        "geometry",
        "checkpoint_vs_pretraining",
        "displacement_from_reference",
    ),
    "02_spectral_fingerprints": (
        "fingerprint",
        "spectral_fingerprint",
        "matrix_spectrum_and_normalized_quantiles",
    ),
    "03_cross_language": (
        "geometry",
        "same_entity_cross_language",
        "same_backbone_entity_different_languages",
    ),
    "04_cross_entity": (
        "geometry",
        "same_language_cross_entity",
        "same_backbone_language_different_entities",
    ),
    "05_cross_language_entity": (
        "geometry",
        "cross_language_cross_entity",
        "same_backbone_different_languages_and_entities",
    ),
    "07_cross_backbone": (
        "geometry",
        "cross_backbone_intrinsic",
        "intrinsic_fingerprints_never_weight_subtraction",
    ),
    "08_language_entity_decomposition": (
        "geometry",
        "factorial_language_entity",
        "balanced_geometric_sum_of_squares_not_causal",
    ),
    "12_entity_adaptation_by_language": (
        "geometry",
        "entity_mean_by_language",
        "updates_to_shared_entity_direction",
    ),
    "13_language_adaptation_by_entity": (
        "geometry",
        "language_mean_by_entity",
        "updates_to_shared_language_direction",
    ),
    "14_mean_adaptation_by_backbone": (
        "geometry",
        "mean_by_backbone",
        "updates_to_shared_complete_grid_direction",
    ),
    "15_coverage": ("coverage", None, "observed_over_expected_coordinates_original_policy"),
    "16_exclusions": ("exclusion", None, "excluded_coordinates_and_groups"),
    "17_sufficient_statistics": ("sufficient", None, "reusable_vector_and_shared_statistics"),
    "18_intrinsic_profiles": (
        "fingerprint",
        "intrinsic_profile",
        "explicit_relative_depth_fingerprints",
    ),
}
FIELDS = (
    "model_id",
    "score_id",
    "backbone",
    "language",
    "entity",
    "strict_f1_mean",
    "strict_f1_std",
    "n_folds",
    "score_scope",
    "protocol_status",
    "ancestry_status",
    "reference_id",
)

METRICS = {
    "relative_l2": "||W-Wref||2 / ||Wref||2; dimensionless reference-relative drift",
    "delta_energy": "sum((W-Wref)^2); additive squared update magnitude",
    "delta_rms": "sqrt(delta_energy/n_parameters)",
    "l2_distance": "||Wa-Wb||2; shared-origin invariant; stored once under final",
    "rmse": "l2_distance/sqrt(n_parameters); shared-origin invariant",
    "cosine": "dot(a,b)/(norm(a)*norm(b)); final and delta are distinct",
    "pearson": "centered coordinate correlation, not model performance correlation",
    "normalized_l2": "||a-b||2/sqrt(||a||2^2+||b||2^2)",
    "mean_update_energy": "squared norm of the equal-weight shared UPDATE direction",
    "dispersion_energy": "mean squared distance of updates to their shared direction",
    "projected_energy_fraction": "dot(update,mean)^2/(||update||2^2*||mean||2^2)",
    "specific_residual_energy": "update energy outside its projection onto the shared direction",
    "leave_one_out_distance": "distance from a member update to the mean of the other members",
    "trace_similarity": "component-averaged depth-profile correlation; not functional similarity",
    "spectral_curve_l1": "mean absolute difference between normalized spectral energy curves on explicit grids",
    "normalized_quantile_w1": "trapezoidal quantile-grid approximation of W1 between RMS-normalized entries",
    "effective_rank_fraction_gap": "absolute difference of entropy effective rank divided by maximum rank",
    "spectral_fingerprint": "payload contains singular values, matrix shape, scalar fingerprints and quantile/energy grids; inspect exactness",
    "coordinate_fraction": "observed coordinates/expected coordinates under ORIGINAL run selection policy",
}


def output_schema():
    fields = list(SCHEMA)
    for side in ("a", "b"):
        for name in FIELDS:
            column = "f1_" + name + "_" + side
            dtype = (
                pa.float64()
                if name in ("strict_f1_mean", "strict_f1_std", "n_folds")
                else pa.string()
            )
            fields.append(pa.field(column, dtype))
    fields += [
        pa.field(name, pa.string()) for name in ("measurement", "interpretation", "f1_alignment")
    ]
    fields += [pa.field("abs_f1_gap", pa.float64())]
    return pa.schema(fields)


def aligned(rows, lookup, measurement):
    for row in rows:
        scores = []
        for side in ("a", "b"):
            endpoint = lookup.get(row["artifact_" + side], {})
            for name in FIELDS:
                value = endpoint.get(name)
                if value is not None and not isinstance(value, (list, dict)) and pd.isna(value):
                    value = None
                row["f1_" + name + "_" + side] = value
            scores.append(endpoint.get("strict_f1_mean"))
        row["abs_f1_gap"] = abs(scores[0] - scores[1]) if all(pd.notna(s) for s in scores) else None
        row.update(
            measurement=measurement,
            interpretation="descriptive_not_causal_no_iid_pair_inference",
            f1_alignment="configuration_aggregate_not_individual_fold_checkpoint",
        )
    return rows


def centrality_export(output, links):
    from itertools import combinations

    from weight_analysis.comparisons.centrality import model_centrality

    frame = pq.read_table(
        output / "07_cross_backbone.parquet", filters=[("scope", "in", ["encoder", "component"])]
    ).to_pandas()
    keys = ["language", "entity", "metric", "scope", "source", "component", "fold_a", "seed_a"]
    results = []
    for identity, group in frame.groupby(keys, dropna=False, sort=True):
        stratum = dict(zip(keys, identity))
        candidates = links[
            (links.language == stratum["language"]) & (links.entity == stratum["entity"])
        ]
        for col in ("fold", "seed"):
            value = stratum[col + "_a"]
            candidates = candidates[
                candidates[col].isna() if pd.isna(value) else candidates[col].eq(value)
            ]
        ids = sorted(candidates.artifact_id)
        edges = {}
        for row in group.itertuples():
            pair = tuple(sorted((row.artifact_a, row.artifact_b)))
            if pair in edges:
                raise ValueError("Duplicate centrality graph edge; select a finer stratum")
            edges[pair] = 1 - row.value if row.metric == "trace_similarity" else row.value
        pairs = [
            dict(
                language=stratum["language"],
                entity=stratum["entity"],
                backbone_a=a,
                backbone_b=b,
                distance=edges.get((a, b), float("nan")),
            )
            for a, b in combinations(ids, 2)
        ]
        if not pairs:
            continue
        result = model_centrality(pd.DataFrame(pairs), metrics=["distance"]).rename(
            columns={"backbone": "artifact_id"}
        )
        for key, value in stratum.items():
            result[key] = value
        result["distance_definition"] = (
            "1-trace_similarity" if stratum["metric"] == "trace_similarity" else stratum["metric"]
        )
        result = result.merge(
            candidates[
                [
                    "artifact_id",
                    "backbone",
                    "strict_f1_mean",
                    "strict_f1_std",
                    "n_folds",
                    "score_scope",
                ]
            ],
            on="artifact_id",
            validate="1:1",
        )
        results.append(result)
    table = (
        pd.concat(results, ignore_index=True)
        if results
        else pd.DataFrame(columns=["artifact_id", "mean_distance"])
    )
    table.to_parquet(output / "11_model_centrality.parquet", compression="zstd", index=False)
    index = frame.groupby(keys, dropna=False).size().rename("n_pair_observations").reset_index()
    index["data_file"] = "07_cross_backbone.parquet"
    index.to_parquet(
        output / "09_cross_backbone_by_entity_index.parquet", compression="zstd", index=False
    )
    return len(table)


def export_studies(run_dir, output_dir, *, batch_size=8192):
    root, output = Path(run_dir).resolve(), Path(output_dir).resolve()
    if output == root or root in output.parents:
        raise ValueError("Use a separate export directory; the source run is read-only")
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("Choose a NEW output directory to avoid mixing exports")
    manifest = json.loads((root / "manifest/datasets.json").read_text())
    if manifest["f1_status"] != "ready":
        raise ValueError("F1 is blocked; resolve the source audit before exporting aligned studies")
    for item in manifest["files"]:
        if file_sha256(root / item["path"]) != item["sha256"]:
            raise ValueError("Source Parquet integrity failure: " + item["path"])
    links = pd.read_parquet(root / "manifest/checkpoint_scores.parquet")
    links = links[links.artifact_id.notna()]
    if links.artifact_id.duplicated().any():
        raise ValueError("Ambiguous checkpoint/F1 identities")
    lookup = links.set_index("artifact_id").to_dict("index")
    source = dataset(root)
    # Resolve exact fingerprint aliases once, without loading any weights.
    expr = (ds.field("kind") == "fingerprint") & (ds.field("metric") == "spectral_fingerprint")
    aliases = {}
    needed = set()
    for batch in source.scanner(
        filter=expr, columns=["unit_id", "payload"], batch_size=batch_size
    ).to_batches():
        for row in batch.to_pylist():
            value = json.loads(row["payload"])
            if "alias_unit_id" in value:
                aliases[row["unit_id"]] = value["alias_unit_id"]
                needed.add(value["alias_unit_id"])
    canonical = {}
    if needed:
        for batch in source.scanner(
            filter=expr & ds.field("unit_id").isin(sorted(needed)),
            columns=["unit_id", "payload"],
            batch_size=batch_size,
        ).to_batches():
            canonical.update({r["unit_id"]: r["payload"] for r in batch.to_pylist()})
    if needed - set(canonical):
        raise ValueError("Unresolved fingerprint aliases")
    output.mkdir(parents=True, exist_ok=True)
    report = dict(
        run_id=manifest["run_id"],
        source_execution_status=manifest["state"],
        source_retry_indices=manifest["retry_array_indices"],
        source_manifest_sha256=file_sha256(root / "manifest/datasets.json"),
        checkpoint_scores_sha256=file_sha256(root / "manifest/checkpoint_scores.parquet"),
        export_code_sha256=file_sha256(__file__),
        metric_definitions=METRICS,
        scope_definitions=dict(
            tensor="one parameter tensor",
            layer="concatenation of eligible layer coordinates",
            component="concatenation of eligible component coordinates",
            encoder="eligible encoder selection; retain coverage flags",
            relative_depth="interpolated explicit depth grid, not aligned neurons",
        ),
        source_generation=manifest["generation"],
        studies={},
        coverage_policy="Original coverage retained. cls.* heads/pooler can inflate expected encoder coordinates; no coverage correction claimed.",
        historical_mapping_notes={
            "09_cross_backbone_by_entity": "09 is a filter of 07 by entity; not a new metric",
        },
    )
    schema = output_schema().with_metadata(
        {b"study_export": b"1", b"run_id": manifest["run_id"].encode()}
    )
    for name, (kind, family, measurement) in STUDIES.items():
        expression = ds.field("kind") == kind
        if family:
            expression &= ds.field("family") == family
        path, temporary = output / (name + ".parquet"), output / (name + ".partial")
        count = 0
        with pq.ParquetWriter(temporary, schema, compression="zstd", compression_level=6) as writer:
            for batch in source.scanner(filter=expression, batch_size=batch_size).to_batches():
                rows = batch.to_pylist()
                if name == "02_spectral_fingerprints":
                    for row in rows:
                        if row["unit_id"] in aliases:
                            row["payload"] = canonical[aliases[row["unit_id"]]]
                table = pa.Table.from_pylist(aligned(rows, lookup, measurement), schema=schema)
                writer.write_table(table)
                count += len(rows)
        if pq.ParquetFile(temporary).metadata.num_rows != count:
            raise ValueError("Export row count mismatch")
        temporary.replace(path)
        report["studies"][name] = dict(
            rows=count, sha256=file_sha256(path), measurement=measurement
        )
        print("[EXPORTED]", name, count, flush=True)
    for filename in ("scores.parquet", "rankings.parquet", "pairs.parquet"):
        import shutil

        shutil.copyfile(root / "parquet/f1" / filename, output / ("f1_" + filename))
    report["centrality"] = dict(
        rows=centrality_export(output, links),
        policy="Mean distance to all available checkpoint nodes in the same language/entity/fold/seed; missing edges prevent ranking. Separate metrics, scopes and components.",
    )
    (output / "catalog.json").write_text(json.dumps(report, indent=2, ensure_ascii=False))
    (output / "README.md").write_text(
        "# Exportaciones por estudio\n\nCada fila conserva métrica, source (final/delta), scope, "
        "tensor/capa/componente y estado original. payload contiene estadísticas/espectros. "
        "Las columnas f1_*_a/b identifican los dos extremos y su F1 medio/SD agregado. "
        "Un centroide no recibe un F1 inventado. No promediar normas ni usar pares como réplicas independientes. "
        "Consultar catalog.json: 09 indexa selecciones de 07 sin duplicar los pares; 11 resume distancias del grafo y exige cobertura para ordenar. "
        "Los agregados encoder con partial_coverage necesitan revisión de cabezas/pooler.\n"
    )
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    export_studies(args.run_dir, args.output_dir)


if __name__ == "__main__":
    main()
