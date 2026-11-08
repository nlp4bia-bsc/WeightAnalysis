"""Predicate-pushed PyArrow reads and logical F1/DuckDB views without copied observations."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pyarrow.dataset as ds

from .storage import SCHEMA


def dataset(root, *, area=None):
    root = Path(root).resolve()
    manifest = json.loads((root / "manifest/datasets.json").read_text())
    files = [
        str(root / item["path"])
        for item in manifest["files"]
        if area is None or item["area"] == area
    ]
    if not files:
        import pyarrow as pa

        return ds.dataset(pa.Table.from_pylist([], schema=SCHEMA))
    return ds.dataset(files, format="parquet", schema=SCHEMA)


def read_geometry(root, *, area=None, filters=None, columns=None):
    expression = ds.field("kind") == "geometry"
    for key, value in (filters or {}).items():
        if key not in SCHEMA.names:
            raise ValueError("Unknown filter: " + key)
        expression &= (
            ds.field(key).isin(value)
            if isinstance(value, (list, tuple))
            else ds.field(key) == value
        )
    return dataset(root, area=area).to_table(filter=expression, columns=columns)


def joined_geometry(root, *, area=None, filters=None):
    root = Path(root)
    report = json.loads((root / "manifest/datasets.json").read_text())
    if report["f1_status"] != "ready":
        raise ValueError("F1 source blocked; geometry remains available")
    frame = read_geometry(root, area=area, filters=filters).to_pandas()
    links = pd.read_parquet(root / "manifest/checkpoint_scores.parquet")
    links = links[links.artifact_id.notna()]
    for side in ("a", "b"):
        columns = [
            c
            for c in (
                "artifact_id",
                "model_id",
                "score_id",
                "backbone",
                "language",
                "entity",
                "fold",
                "seed",
                "strict_f1_mean",
                "strict_f1_std",
                "n_folds",
                "score_scope",
                "protocol_status",
                "ancestry_status",
                "reference_id",
                "reference_kind",
            )
            if c in links
        ]
        endpoint = links[columns].rename(columns={key: key + "_" + side for key in columns})
        frame = frame.merge(
            endpoint,
            left_on="artifact_" + side,
            right_on="artifact_id_" + side,
            how="left",
            validate="m:1",
            suffixes=("", "_linked"),
        )
    frame["abs_f1_gap"] = (frame.strict_f1_mean_a - frame.strict_f1_mean_b).abs()
    gap = frame.strict_f1_mean_a - frame.strict_f1_mean_b
    frame["performance_order"] = gap.map(
        lambda x: (
            "unknown" if pd.isna(x) else "a_better" if x > 0 else "b_better" if x < 0 else "tie"
        )
    )
    frame["interpretation"] = "descriptive_geometry_vs_configuration_aggregate_F1_not_fold_F1"
    return frame


def write_views(root):
    root = Path(root).resolve()
    manifest = json.loads((root / "manifest/datasets.json").read_text())

    def sql_string(value):
        return "'" + str(value).replace("'", "''") + "'"

    files = ",".join(sql_string(root / item["path"]) for item in manifest["files"])
    if not files:
        return
    lines = [
        "CREATE OR REPLACE VIEW weight_records AS SELECT * FROM read_parquet(["
        + files
        + "], union_by_name=true);",
        "CREATE OR REPLACE VIEW weight_geometry AS SELECT * FROM weight_records WHERE kind='geometry';",
        "CREATE OR REPLACE VIEW weight_sufficient_statistics AS SELECT * FROM weight_records WHERE kind='sufficient';",
        "CREATE OR REPLACE VIEW weight_fingerprints AS SELECT f.* EXCLUDE(payload), "
        "coalesce(c.payload,f.payload) AS payload FROM weight_records f LEFT JOIN weight_records c "
        "ON c.unit_id=json_extract_string(f.payload,'$.alias_unit_id') AND c.kind='fingerprint' "
        "WHERE f.kind='fingerprint';",
    ]
    if manifest["f1_status"] == "ready":
        lines += [
            "CREATE OR REPLACE VIEW checkpoint_scores AS SELECT * FROM read_parquet("
            + sql_string(root / "manifest/checkpoint_scores.parquet")
            + ");",
            "CREATE OR REPLACE VIEW weight_geometry_f1_joined AS SELECT g.*, "
            "a.model_id AS model_id_a, b.model_id AS model_id_b, "
            "a.backbone AS backbone_a, b.backbone AS backbone_b, "
            "a.strict_f1_mean AS strict_f1_mean_a, a.strict_f1_std AS strict_f1_std_a, "
            "b.strict_f1_mean AS strict_f1_mean_b, b.strict_f1_std AS strict_f1_std_b, "
            "a.score_scope AS score_scope_a, b.score_scope AS score_scope_b, "
            "a.reference_id AS reference_id_a, b.reference_id AS reference_id_b, "
            "a.ancestry_status AS ancestry_status_a, b.ancestry_status AS ancestry_status_b, "
            "abs(a.strict_f1_mean-b.strict_f1_mean) AS abs_f1_gap "
            "FROM weight_geometry g LEFT JOIN checkpoint_scores a ON g.artifact_a=a.artifact_id "
            "LEFT JOIN checkpoint_scores b ON g.artifact_b=b.artifact_id;",
            "CREATE OR REPLACE VIEW f1_all_valid_pairs AS SELECT * FROM read_parquet("
            + sql_string(root / "parquet/f1/pairs.parquet")
            + ");",
            "CREATE OR REPLACE VIEW f1_best_worst_pairs AS SELECT * FROM f1_all_valid_pairs WHERE is_best_worst;",
        ]
        for name in (
            "model_ranking",
            "ranking_by_language",
            "ranking_by_entity",
            "ranking_by_backbone",
        ):
            lines.append(
                "CREATE OR REPLACE VIEW f1_"
                + name
                + " AS SELECT * FROM read_parquet("
                + sql_string(root / "parquet/f1/rankings.parquet")
                + ") WHERE ranking_table="
                + sql_string("f1_" + name)
                + ";"
            )
        for scope in ("layer", "component", "embeddings", "encoder", "relative_depth"):
            lines.append(
                "CREATE OR REPLACE VIEW weight_geometry_f1_by_"
                + scope
                + " AS SELECT * FROM weight_geometry_f1_joined WHERE scope="
                + sql_string(scope)
                + ";"
            )
        for name in ("language", "entity", "backbone"):
            lines.append(
                "CREATE OR REPLACE VIEW weight_geometry_f1_by_"
                + name
                + " AS SELECT * FROM weight_geometry_f1_joined;"
            )
    (root / "manifest/views.sql").write_text("\n".join(lines) + "\n")


def duckdb_connection(root):
    import duckdb

    connection = duckdb.connect()
    connection.execute((Path(root) / "manifest/views.sql").read_text())
    return connection
