"""Stream scientific coverage/undefined counts without materializing all observations."""

from __future__ import annotations

import json
from collections import Counter

import pyarrow.compute as pc
import pyarrow.dataset as ds

from weight_analysis_scheduler.resources import usage

from ..plan import atomic_json
from .storage import SCHEMA


def write_summary(root, report):
    paths = [str(root / item["path"]) for item in report["files"]]
    counts = Counter()
    undefined = Counter()
    if paths:
        data = ds.dataset(paths, format="parquet", schema=SCHEMA)
        for batch in data.scanner(columns=["kind", "status", "family"]).to_batches():
            table = batch.to_pandas()
            for key, number in (
                table.groupby(["kind", "status", "family"], dropna=False).size().items()
            ):
                counts["/".join(str(k) for k in key)] += int(number)
        fingerprints = data.to_table(
            filter=(ds.field("kind") == "fingerprint")
            & (ds.field("metric") == "spectral_fingerprint"),
            columns=["unit_id", "payload"],
        )
        payloads = {row["unit_id"]: json.loads(row["payload"]) for row in fingerprints.to_pylist()}
        for value in payloads.values():
            resolved = payloads[value["alias_unit_id"]] if "alias_unit_id" in value else value
            for key, number in resolved.items():
                if number is None:
                    undefined[key] += 1
        exclusions = data.to_table(filter=ds.field("kind") == "exclusion", columns=["payload"])
        reasons = Counter(
            json.loads(v).get("reason", "unspecified")
            for v in exclusions.column("payload").to_pylist()
        )
        coverage = data.to_table(filter=ds.field("kind") == "coverage", columns=["value"])
        minimum = pc.min(coverage["value"]).as_py() if coverage.num_rows else None
    else:
        reasons, minimum = Counter(), None
    summary = dict(
        run_id=report["run_id"],
        execution_status=report["state"],
        f1_status=report["f1_status"],
        observations=dict(sorted(counts.items())),
        undefined_spectral_fields=dict(sorted(undefined.items())),
        exclusion_reasons=dict(sorted(reasons.items())),
        minimum_coordinate_coverage=minimum,
        parquet_files=len(paths),
        disk=usage(root),
        retry_indices=report["retry_array_indices"],
        policy="Completion refers to eligible planned units; exclusions remain scientific coverage limits",
    )
    atomic_json(root / "reports/summary.json", summary)
    return summary
