"""Parameter selection and canonical normalized geometric observations."""

from __future__ import annotations

import math
from dataclasses import dataclass

from .catalog import stable_id

DEFINITION = "fine-grained-1"
SHIFT_INVARIANT = frozenset({"l2_distance", "l1_mean_distance", "rmse"})


@dataclass(frozen=True)
class ParameterSelection:
    layers: tuple[int, ...] | None = None
    components: tuple[str, ...] | None = None
    embeddings: tuple[str, ...] = ()
    tensors: tuple[str, ...] | None = None

    def matches(self, location):
        if self.tensors is not None and location["tensor"] not in self.tensors:
            return False
        if location["component"] == "embeddings":
            return "*" in self.embeddings or location["embedding_type"] in self.embeddings
        return (self.layers is None or location["layer"] in self.layers) and (
            self.components is None or location["component"] in self.components
        )


def observation(unit_id, metadata, metric, value, *, scope="tensor", **extra):
    metadata = {key: val for key, val in metadata.items() if key != "metric_names"}
    if value is not None and math.isinf(float(value)):
        raise ValueError(f"Metric {metric} overflowed; refusing an infinite result")
    status = "ok" if value is not None and math.isfinite(float(value)) else "undefined"
    return {
        **metadata,
        "definition_id": DEFINITION,
        "metric": metric,
        "value": float(value) if value is not None else None,
        "status": status,
        "scope": scope,
        "observation_id": stable_id([metric, scope, metadata, extra, DEFINITION]),
        **extra,
    }
