"""Independent fault-isolated parameter units with explicit reusable cache artifacts."""

from __future__ import annotations

import json
import math
from dataclasses import asdict
from pathlib import Path

import pandas as pd
import torch

from ..core import WeightReader, audit_alignment, config_metadata
from ..core.blocks import iter_blocks, parameter_location, select_parameters
from ..core.catalog import (
    geometry_dimensions,
    input_digest,
    reference_for,
    stable_id,
    validate_reference,
)
from ..core.geometry_schema import DEFINITION, observation
from ..core.reporting import RunReporter
from ..devices import resolve_device
from ..metrics import spectral_metrics
from ..results import AnalysisResult, read_result, write_result
from ..results.models import _cell_value
from ..statistics.aggregation import VectorMoments, merge_moments


class ComparativeContext:
    """Run-local reuse of verified inputs and results; no global state or scheduling."""

    def __init__(
        self,
        *,
        options,
        device,
        parameters,
        references,
        reporter=None,
        on_error="continue",
        cache_dir=None,
        resume=True,
        block_elements=100_000,
        allow_unverified=False,
    ):
        if on_error not in ("continue", "raise") or block_elements < 1:
            raise ValueError("Invalid error policy or block size")
        self.options, self.device = options, str(resolve_device(device or options.device))
        self.parameters, self.references = parameters, references
        self.reporter = reporter or RunReporter()
        self.on_error, self.resume, self.block_elements = on_error, resume, block_elements
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self.allow_unverified = allow_unverified
        self.verified, self.reference_cache, self.unit_cache, self.fingerprints = set(), {}, {}, {}
        self.reference_rows = {}
        self.reference_contents = {}
        self.fingerprint_plan = {}
        self.definition = {
            "version": DEFINITION,
            "options": {**asdict(options), "device": self.device},
            "device": self.device,
            "torch": torch.__version__,
            "cuda_runtime": torch.version.cuda,
            "block_elements": block_elements,
            "hardware": torch.cuda.get_device_name(self.device)
            if self.device.startswith("cuda")
            else "cpu",
            "capability": list(torch.cuda.get_device_capability(self.device))
            if self.device.startswith("cuda")
            else None,
        }

    def verify(self, entry):
        if entry.artifact_id not in self.verified:
            if input_digest(entry.checkpoint.path) != entry.content_id:
                raise ValueError("Checkpoint changed after catalog registration")
            self.verified.add(entry.artifact_id)

    def reference(self, entry):
        ref = reference_for(entry, self.references)
        key = (entry.artifact_id, ref)
        if key not in self.reference_cache:
            if ref.path not in self.reference_contents:
                self.reference_contents[ref.path] = input_digest(ref.path)
            content = self.reference_contents[ref.path]
            rid = validate_reference(
                entry, ref, allow_unverified=self.allow_unverified, content_id=content
            )
            if self.parameters.embeddings and (
                not ref.tokenizer_id or ref.tokenizer_id != entry.evidence.tokenizer_id
            ):
                raise ValueError(
                    "Reference embeddings require common tokenizer/vocabulary evidence"
                )
            self.reference_cache[key] = (ref, rid)
            self.reference_rows[rid] = {
                "reference_id": rid,
                "kind": ref.kind,
                "content_id": content,
                "initialization_id": entry.evidence.initialization_id,
                "fold": ref.fold,
                "seed": ref.seed,
                "scope": ref.scope,
                "provenance_id": ref.provenance_id,
                "tokenizer_id": ref.tokenizer_id,
            }
        return self.reference_cache[key]

    def aligned(self, reader, reference):
        a, b = (
            config_metadata(reader.path, strict=True),
            config_metadata(reference.path, strict=True),
        )
        for field in (
            "model_type",
            "hidden_size",
            "num_hidden_layers",
            "num_attention_heads",
            "vocab_size",
        ):
            if a.get(field) is not None and b.get(field) is not None and a[field] != b[field]:
                raise ValueError(f"Reference architecture mismatch: {field}")
        return audit_alignment(reader, reference, 1.0)[0]

    def keys(self, entry):
        self.verify(entry)
        with WeightReader(entry.checkpoint.path) as reader:
            keys = select_parameters(reader, self.parameters)
        if not keys:
            raise ValueError("No encoder parameters match the explicit selection")
        return keys

    def unit(self, identity, operation, **context):
        uid = stable_id({"identity": identity, "definition": self.definition})
        if uid in self.unit_cache:
            return uid, self.unit_cache[uid]
        path = self.cache_dir / f"{uid}.parquet" if self.cache_dir else None
        if self.resume and path is not None and path.is_file():
            try:
                restored = read_result(path)
                if restored.provenance != {"unit_id": uid, "definition": self.definition}:
                    raise ValueError("Cache metadata differ from the requested computation")
                tables = {}
                for row in restored.data.to_dict("records"):
                    tables.setdefault(row["table"], []).append(json.loads(row["record"]))
                self.unit_cache[uid] = tables
                self.reporter.emit("cached", uid, "Reused validated unit", **context)
                return uid, tables
            except Exception as exc:
                self.reporter.emit(
                    "warning", uid, "Invalid cache; recomputing", exception=exc, **context
                )
        result = self.reporter.attempt(
            uid, lambda: operation(uid), on_error=self.on_error, device=self.device, **context
        )
        if result is None:
            return uid, {}
        if path is not None:

            def persist():
                rows = [
                    {"table": name, "record": json.dumps(_cell_value(row), sort_keys=True)}
                    for name, records in result.items()
                    for row in records
                ]
                write_result(
                    AnalysisResult(
                        "comparative_unit",
                        pd.DataFrame(rows, columns=["table", "record"]),
                        {"unit_id": uid, "definition": self.definition},
                    ),
                    path,
                )
                return True

            self.reporter.attempt(
                uid + ":persist", persist, on_error=self.on_error, stage="persistence"
            )
        self.unit_cache[uid] = result
        return uid, result

    def fingerprint(self, entry, key, source, names):
        self.verify(entry)
        requested = set(names)
        names = tuple(sorted(requested | self.fingerprint_plan.get(source, set())))
        ref, rid = self.reference(entry) if source in ("base", "delta") else (None, None)
        identity = {
            "kind": "fingerprint",
            "artifact": entry.artifact_id,
            "tensor": key,
            "source": source,
            "reference": rid,
            "metrics": sorted(names),
        }

        def compute(uid):
            with WeightReader(entry.checkpoint.path) as reader:
                location = parameter_location(reader, key)
                if source == "final":
                    matrix = reader.get(key).to(device=self.device, dtype=torch.float64)
                else:
                    with WeightReader(ref.path) as base:
                        self.aligned(reader, base)
                        initial = base.get(key).to(device=self.device, dtype=torch.float64)
                    matrix = (
                        initial
                        if source == "base"
                        else reader.get(key).to(device=self.device, dtype=torch.float64) - initial
                    )
                if matrix.ndim != 2:
                    raise ValueError("Spectral fingerprints require matrix parameters")
                values = spectral_metrics(
                    matrix,
                    metrics=list(names),
                    device=self.device,
                    mode=self.options.spectral_mode,
                    max_exact_elements=self.options.max_svd_elements,
                    topk=self.options.topk,
                )
                layers = [parameter_location(reader, k)["layer"] for k in reader.keys]
                maximum = max((l for l in layers if l is not None), default=0)
            meta = {
                **geometry_dimensions(entry),
                "execution_id": stable_id(self.definition),
                **location,
                "subject_kind": "checkpoint",
                "subject_id": entry.artifact_id,
                "source": source,
                "reference_id": rid,
                "n_parameters": matrix.numel(),
            }
            wide = {
                "checkpoint": entry.artifact_id,
                "level": "tensor",
                **location,
                "source": source,
                "relative_depth": location["layer"] / maximum
                if location["layer"] is not None and maximum
                else 0.0,
                **values,
            }
            detailed, points = [], []
            for name in names:
                value = values[name]
                method = values["status"]
                if (
                    name in ("matrix_fro", "normalized_weight_quantiles")
                    and method != "zero_matrix"
                ):
                    method = "not_required"
                if isinstance(value, list):
                    profile_id = stable_id([meta, name])
                    for index, item in enumerate(value):
                        if math.isinf(float(item)):
                            raise ValueError(f"Fingerprint {name} overflowed")
                        points.append(
                            {
                                **meta,
                                "profile_id": profile_id,
                                "metric": name,
                                "position": index / (len(value) - 1),
                                "value": item,
                                "point_id": stable_id([profile_id, index]),
                                "status": method,
                            }
                        )
                else:
                    detailed.append(observation(uid, meta, name, value, method_status=method))
            return {"geometry_detailed": detailed, "profile_points": points, "fingerprints": [wide]}

        unit = self.unit(
            identity,
            compute,
            stage="fingerprint",
            checkpoint=entry.artifact_id,
            tensor=key,
            source=source,
        )[1]
        return {
            name: (
                [row for row in rows if row["metric"] in requested]
                if name in ("geometry_detailed", "profile_points")
                else rows
            )
            for name, rows in unit.items()
        }

    def vector(
        self, entry_a, entry_b, key, source, names, *, subject_id, kind="pair", histogram_edges=None
    ):
        self.verify(entry_a)
        if entry_b is not None:
            self.verify(entry_b)
        if source == "delta" or kind == "drift":
            ref, rid = self.reference(entry_a)
            if entry_b is not None:
                _, other = self.reference(entry_b)
                if other != rid:
                    raise ValueError("Compatible update comparisons require one shared reference")
        else:
            ref, rid = None, None
        identity = {
            "kind": kind,
            "artifacts": [entry_a.artifact_id, entry_b.artifact_id if entry_b else None],
            "reference": rid,
            "source": source,
            "tensor": key,
            "bins": histogram_edges,
        }

        def compute(uid):
            from contextlib import ExitStack

            with ExitStack() as stack:
                ra = stack.enter_context(WeightReader(entry_a.checkpoint.path))
                rb = stack.enter_context(WeightReader(entry_b.checkpoint.path)) if entry_b else None
                base = stack.enter_context(WeightReader(ref.path)) if ref else None
                if base:
                    self.aligned(ra, base)
                    if rb:
                        self.aligned(rb, base)
                if rb and ra.shape_dtype(key) != rb.shape_dtype(key):
                    raise ValueError("Pair parameter shape/dtype incompatibility")
                location = parameter_location(ra, key)
                meta = {
                    **geometry_dimensions(entry_a, entry_b),
                    "execution_id": stable_id(self.definition),
                    **location,
                    "subject_id": subject_id,
                    "subject_kind": kind,
                    "source": source,
                    "reference_id": rid,
                    "definition_id": DEFINITION,
                }
                a_blocks = iter_blocks(ra, key, self.block_elements)
                b_blocks = iter_blocks(rb if rb else base, key, self.block_elements)
                base_blocks = iter_blocks(base, key, self.block_elements) if base and rb else None
                rows, histograms = [], []
                for (start, a), (other, b) in zip(a_blocks, b_blocks, strict=True):
                    if start != other:
                        raise ValueError("Coordinate blocks are misaligned")
                    a, b = (
                        a.to(self.device, dtype=torch.float64),
                        b.to(self.device, dtype=torch.float64),
                    )
                    if base_blocks:
                        original_start, initial = next(base_blocks)
                        if original_start != start:
                            raise ValueError("Reference block mismatch")
                        initial = initial.to(self.device, dtype=torch.float64)
                        a, b = a - initial, b - initial
                    moments = VectorMoments.from_tensors(a, b, sign_floor=self.options.sign_floor)
                    rows.append(
                        {
                            **meta,
                            **asdict(moments),
                            "block_start": start,
                            "block_end": start + a.numel(),
                            "statistic_id": stable_id([uid, start]),
                        }
                    )
                    if histogram_edges is not None:
                        edges = torch.tensor(
                            histogram_edges, device=self.device, dtype=torch.float64
                        )
                        values = (a - b).reshape(-1)
                        bins = torch.bucketize(values, edges, right=True) - 1
                        bins = torch.where(values == edges[-1], len(edges) - 2, bins)
                        valid = (values >= edges[0]) & (values <= edges[-1])
                        counts = (
                            torch.bincount(bins[valid], minlength=len(edges) - 1).cpu().tolist()
                        )
                        histograms.append(
                            {
                                **meta,
                                "edges": list(histogram_edges),
                                "counts": counts,
                                "below": int((values < edges[0]).sum()),
                                "above": int((values > edges[-1]).sum()),
                                "histogram_id": stable_id([uid, start, "histogram"]),
                                "block_start": start,
                            }
                        )
            return {
                "sufficient_statistics": rows,
                "histograms": histograms,
            }

        uid, unit = self.unit(
            identity, compute, stage=kind, subject=subject_id, tensor=key, source=source
        )
        rows = unit.get("sufficient_statistics", [])
        if not rows:
            return unit
        moments = merge_moments(rows)
        meta = {
            key: val
            for key, val in rows[0].items()
            if key not in VectorMoments.__dataclass_fields__
            and key not in ("block_start", "block_end", "statistic_id")
        }
        return {
            **unit,
            "sufficient_statistics": [
                {**row, "metric_names": ",".join(sorted(names))} for row in rows
            ],
            "geometry_detailed": [
                observation(uid, {**meta, "n_parameters": moments.n}, name, moments.value(name))
                for name in names
            ],
        }
