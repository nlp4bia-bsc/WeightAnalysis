"""Stable mergeable vector statistics and explicit geometric aggregation."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch

from ..metrics import EPS


@dataclass(frozen=True)
class VectorMoments:
    n: int
    qa: float
    qb: float
    dot: float
    qdiff: float
    l1: float
    maximum: float
    mean_a: float
    mean_b: float
    m2_a: float
    m2_b: float
    covariance: float
    sign_n: int
    sign_agree: int

    @classmethod
    def from_tensors(cls, a, b, *, sign_floor=1e-8):
        a, b = a.detach().reshape(-1).double(), b.detach().reshape(-1).double()
        if a.shape != b.shape or not a.numel() or a.device != b.device:
            raise ValueError("Statistics require nonempty aligned tensors on one device")
        if not torch.isfinite(a).all() or not torch.isfinite(b).all():
            raise ValueError("Nonfinite geometric input")
        da, db = a - a.mean(), b - b.mean()
        mask = (a.abs() > sign_floor) & (b.abs() > sign_floor)
        values = (
            torch.stack(
                [
                    a.square().sum(),
                    b.square().sum(),
                    torch.dot(a, b),
                    (a - b).square().sum(),
                    (a - b).abs().sum(),
                    (a - b).abs().max(),
                    a.mean(),
                    b.mean(),
                    da.square().sum(),
                    db.square().sum(),
                    torch.dot(da, db),
                    mask.sum().double(),
                    (mask & (a.sign() == b.sign())).sum().double(),
                ]
            )
            .cpu()
            .tolist()
        )
        if not all(math.isfinite(value) for value in values):
            raise ValueError("Geometric reductions overflowed")
        return cls(a.numel(), *values[:11], int(values[11]), int(values[12]))

    def merge(self, other):
        n = self.n + other.n
        da, db = other.mean_a - self.mean_a, other.mean_b - self.mean_b
        weight = self.n * other.n / n
        return VectorMoments(
            n,
            self.qa + other.qa,
            self.qb + other.qb,
            self.dot + other.dot,
            self.qdiff + other.qdiff,
            self.l1 + other.l1,
            max(self.maximum, other.maximum),
            self.mean_a + da * other.n / n,
            self.mean_b + db * other.n / n,
            self.m2_a + other.m2_a + da * da * weight,
            self.m2_b + other.m2_b + db * db * weight,
            self.covariance + other.covariance + da * db * weight,
            self.sign_n + other.sign_n,
            self.sign_agree + other.sign_agree,
        )

    def value(self, name):
        na, nb, d = math.sqrt(self.qa), math.sqrt(self.qb), math.sqrt(self.qdiff)

        def ratio(a, b):
            return a / b if b > EPS else math.nan

        cosine = float(np.clip(ratio(self.dot, na * nb), -1, 1)) if min(na, nb) > EPS else math.nan
        values = {
            "n_parameters": self.n,
            "delta_l2": d,
            "l2_distance": d,
            "delta_energy": self.qdiff,
            "relative_l2": ratio(d, nb),
            "relative_frobenius": ratio(d, nb),
            "delta_rms": d / math.sqrt(self.n),
            "rmse": d / math.sqrt(self.n),
            "delta_l1": self.l1,
            "delta_mae": self.l1 / self.n,
            "l1_mean_distance": self.l1 / self.n,
            "max_abs_delta": self.maximum,
            "delta_mean": self.mean_a - self.mean_b,
            "base_l2": nb,
            "final_l2": na,
            "norm_a": na,
            "norm_b": nb,
            "cosine": cosine,
            "cosine_final_base": cosine,
            "angle_over_pi": math.acos(cosine) / math.pi if math.isfinite(cosine) else math.nan,
            "normalized_l2": ratio(d, math.hypot(na, nb)),
            "pearson": float(
                np.clip(ratio(self.covariance, math.sqrt(max(0, self.m2_a * self.m2_b))), -1, 1)
            ),
            "sign_agreement": self.sign_agree / self.sign_n if self.sign_n else math.nan,
            "sign_conflict": 1 - self.sign_agree / self.sign_n if self.sign_n else math.nan,
        }
        if name not in values:
            raise ValueError(f"Metric {name!r} is not aggregable from vector statistics")
        return values[name]


def merge_moments(records):
    """Use a canonical reduction order and reject overlapping parameter coordinates."""
    if not records:
        raise ValueError("Cannot aggregate no statistics")
    ordered = sorted(records, key=lambda r: (r["tensor"], r["block_start"], r["block_end"]))
    ends = {}
    result = None
    for record in ordered:
        tensor = record["tensor"]
        if (
            record["block_start"] < ends.get(tensor, 0)
            or record["block_end"] <= record["block_start"]
        ):
            raise ValueError("Overlapping or invalid coordinate fragments")
        ends[tensor] = record["block_end"]
        moments = VectorMoments(**{key: record[key] for key in VectorMoments.__dataclass_fields__})
        if moments.n != record["block_end"] - record["block_start"]:
            raise ValueError("Fragment coordinate count mismatch")
        result = moments if result is None else result.merge(moments)
    return result


def aggregate_geometry(result, *, scopes=("layer_component", "layer", "component", "encoder")):
    """Finalize existing metrics from disjoint statistics, never average tensor norms."""
    from ..core.catalog import stable_id
    from ..results.comparative import ComparativeResult

    stats = result.tables.get("sufficient_statistics", pd.DataFrame())
    rows = []
    if not stats.empty:
        identity = [
            "subject_id",
            "subject_kind",
            "source",
            "reference_id",
            "definition_id",
            "execution_id",
            "metric_names",
        ]
        for scope in scopes:
            if scope not in ("layer_component", "layer", "component", "encoder"):
                raise ValueError(f"Unknown aggregation scope: {scope}")
            extra = {
                "layer_component": ["layer", "component", "embedding_type"],
                "layer": ["layer"],
                "component": ["component", "embedding_type"],
                "encoder": [],
            }[scope]
            for _, group in stats.groupby(identity + extra, dropna=False, sort=True):
                records = group.to_dict("records")
                moments = merge_moments(records)
                exemplar = records[0]
                for metric in exemplar["metric_names"].split(","):
                    row = {key: exemplar[key] for key in identity if key != "metric_names"}
                    row.update(
                        {
                            key: exemplar[key]
                            for key in (
                                "language",
                                "entity",
                                "backbone",
                                "fold",
                                "seed",
                                "comparison_type",
                            )
                            if key in exemplar
                        }
                    )
                    row.update(
                        scope=scope,
                        layer=exemplar.get("layer") if "layer" in extra else None,
                        component=exemplar.get("component") if "component" in extra else None,
                        embedding_type=exemplar.get("embedding_type")
                        if "embedding_type" in extra
                        else None,
                        tensor=None,
                        metric=metric,
                        value=moments.value(metric),
                        n_parameters=moments.n,
                        aggregation="energy_and_stable_sufficient_statistics",
                    )
                    row["observation_id"] = stable_id(
                        {
                            **{k: v for k, v in row.items() if k != "value"},
                            "fragments": sorted(group.statistic_id),
                        }
                    )
                    row["status"] = "ok" if math.isfinite(row["value"]) else "undefined"
                    rows.append(row)
    from .shared import aggregate_shared_geometry

    rows.extend(aggregate_shared_geometry(result, scopes=scopes))
    return ComparativeResult(
        {**result.tables, "geometry_aggregated": pd.DataFrame(rows)},
        {**result.provenance, "aggregation_scopes": list(scopes)},
    )
