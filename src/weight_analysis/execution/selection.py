"""Selective selection for independent scientific computations."""

from __future__ import annotations

from itertools import combinations
from typing import Any, Sequence

from ..core import CheckpointSpec


def _names(values: Sequence[Any] | None) -> list[Any] | None:
    if values is None:
        return None
    if isinstance(values, (str, bytes)):
        raise TypeError("Selectors must be sequences, not a single string")
    return sorted(set(values), key=lambda value: (value is not None, str(value)))


def _eligible(a: CheckpointSpec, b: CheckpointSpec, analysis: str) -> bool:
    if (a.fold, a.seed) != (b.fold, b.seed):
        return False
    if analysis == "cross_backbone":
        return a.backbone != b.backbone and (a.language, a.entity) == (b.language, b.entity)
    if a.backbone != b.backbone:
        return False
    if analysis == "cross_language":
        return a.language != b.language and a.entity == b.entity
    if analysis == "cross_entity":
        return a.language == b.language and a.entity != b.entity
    return a.language != b.language and a.entity != b.entity


def _pairs(
    checkpoints: Sequence[CheckpointSpec], analysis: str, pairs: Sequence[tuple[str, str]] | None
) -> list[tuple[CheckpointSpec, CheckpointSpec]]:
    lookup = {checkpoint.uid: checkpoint for checkpoint in checkpoints}
    if pairs is None:
        return [(a, b) for a, b in combinations(checkpoints, 2) if _eligible(a, b, analysis)]
    selected = set()
    for pair in pairs:
        if len(pair) != 2 or pair[0] == pair[1] or not set(pair) <= lookup.keys():
            raise ValueError(f"Pair must contain two distinct selected checkpoint uids: {pair}")
        left, right = sorted(pair)
        if not _eligible(lookup[left], lookup[right], analysis):
            raise ValueError(f"Pair violates the {analysis} experimental design: {pair}")
        selected.add((left, right))
    return [(lookup[left], lookup[right]) for left, right in sorted(selected)]
