"""Descriptive balanced language-by-entity sums of squares."""

from __future__ import annotations

import logging
import math
from collections import defaultdict
from contextlib import ExitStack
from itertools import product
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch

from ..core import Checkpoint, WeightReader, audit_alignment, identify
from ..metrics import EPS

LOG = logging.getLogger(__name__)


def validate_factorial_design(checkpoints: Sequence[Checkpoint]) -> None:
    """Require one checkpoint in each cell of a single backbone/fold/seed design.

    The descriptive decomposition has no replication term. Different folds or
    seeds must be analyzed separately rather than silently pooled into cells.
    This validator is opt-in for the historical ``factorial`` entry point.
    """
    if not checkpoints:
        raise ValueError("checkpoints must contain a complete language-by-entity grid")

    identities = {
        (checkpoint.backbone, checkpoint.fold, getattr(checkpoint, "seed", None))
        for checkpoint in checkpoints
    }
    if len(identities) != 1:
        raise ValueError("factorial checkpoints must share one backbone, fold and seed")

    cells = [(checkpoint.language, checkpoint.entity) for checkpoint in checkpoints]
    if len(set(cells)) != len(cells):
        raise ValueError("factorial checkpoints contain duplicate language/entity cells")

    languages = {language for language, _ in cells}
    entities = {entity for _, entity in cells}
    if len(languages) < 2 or len(entities) < 2:
        raise ValueError("factorial checkpoints need at least two languages and two entities")
    missing = set(product(languages, entities)) - set(cells)
    if missing:
        raise ValueError(
            f"factorial checkpoints have missing language/entity cells: {sorted(missing)[:10]}"
        )


def factorial(
    checkpoints: list[Checkpoint],
    base_path: Path,
    settings: Any,
    *,
    layers: Sequence[int] | None = None,
    components: Sequence[str] | None = None,
    tensors: Sequence[str] | None = None,
) -> list[dict[str, Any]]:
    """Compute historical descriptive sums of squares on selected aligned tensors.

    Filters apply after full-design alignment validation and before reading
    tensor values. Use ``validate_factorial_design`` for strict fold/seed and
    replication validation; the historical skip policy remains unchanged here.
    """
    ls = sorted({x.language for x in checkpoints})
    es = sorted({x.entity for x in checkpoints})
    if len(ls) < 2 or len(es) < 2:
        return []
    lookup = {(c.language, c.entity): c for c in checkpoints}
    if len(lookup) != len(checkpoints) or any((l, e) not in lookup for l, e in product(ls, es)):
        LOG.warning(
            "Factorial skipped for %s: full balanced language × entity grid required",
            checkpoints[0].backbone,
        )
        return []
    rows = []
    accumulated = defaultdict(lambda: np.zeros(5, float))
    with ExitStack() as stack:
        b = stack.enter_context(WeightReader(base_path))
        readers = {
            (l, e): stack.enter_context(WeightReader(lookup[l, e].path)) for l, e in product(ls, es)
        }
        keys = []
        for r in readers.values():
            current, _ = audit_alignment(r, b, settings.min_aligned_fraction)
            keys.append(set(current))
        commonkeys = sorted(set.intersection(*keys))
        common_elements = sum(math.prod(b.shape_dtype(k)[0]) for k in commonkeys)
        matched_cells = [
            sum(math.prod(r.shape_dtype(k)[0]) for k in keyset)
            for r, keyset in zip(readers.values(), keys)
        ]
        coverage = common_elements / max(matched_cells)
        if coverage < settings.min_aligned_fraction:
            raise ValueError(f"Factorial lacks common coordinates across all cells: {coverage:.4%}")
        if layers is not None or components is not None or tensors is not None:
            selected = []
            for key in commonkeys:
                layer, component, _ = identify(key, b.shape_dtype(key)[0])
                if layers is not None and layer not in layers:
                    continue
                if components is not None and component not in components:
                    continue
                if tensors is not None and key not in tensors:
                    continue
                selected.append(key)
            commonkeys = selected
        for key in commonkeys:
            orig = b.get(key)  # hold only one parameter tensor from original
            layer, comp, _ = identify(key, tuple(orig.shape))
            # Cache ONE tensor from each checkpoint for this key. Even huge
            # embeddings are read once, not once per 30k-element slice.
            cached = {(l, e): readers[l, e].get(key).reshape(-1) for l, e in product(ls, es)}
            ss = np.zeros(5, float)
            for start in range(0, orig.numel(), settings.factorial_chunk_elements):
                end = min(orig.numel(), start + settings.factorial_chunk_elements)
                w0 = orig.reshape(-1)[start:end].to(torch.float64)
                z = torch.stack(
                    [
                        torch.stack([(cached[l, e][start:end].to(torch.float64) - w0) for e in es])
                        for l in ls
                    ]
                )
                if not torch.isfinite(z).all():
                    raise ValueError(f"Nonfinite factorial input {key}")
                mu = z.mean((0, 1))
                lf = z.mean(1) - mu
                ef = z.mean(0) - mu
                interaction = z - mu - lf[:, None, :] - ef[None, :, :]
                ss += np.array(
                    [
                        len(es) * float(lf.square().sum()),
                        len(ls) * float(ef.square().sum()),
                        float(interaction.square().sum()),
                        float((z - mu).square().sum()),
                        z.shape[-1],
                    ],
                    float,
                )
            cached.clear()
            if not np.isclose(ss[:3].sum(), ss[3], rtol=2e-9, atol=1e-10):
                raise ValueError(f"Factorial identity broken for {key}: {ss}")
            for level, lay, section, tensor in [
                ("tensor", layer, comp, key),
                ("layer", layer, None, None),
                ("component", None, comp, None),
                ("encoder", None, None, None),
            ]:
                if level == "tensor":
                    rows.append(
                        _factorial_row(
                            checkpoints[0].backbone, ls, es, level, lay, section, tensor, ss
                        )
                    )
                else:
                    accumulated[level, lay, section] += ss
        for (level, layer, comp), ss in accumulated.items():
            rows.append(
                _factorial_row(checkpoints[0].backbone, ls, es, level, layer, comp, None, ss)
            )
    return rows


def _factorial_row(
    backbone: str,
    ls: list[str],
    es: list[str],
    level: str,
    layer: int | None,
    comp: str | None,
    key: str | None,
    ss: np.ndarray,
) -> dict[str, Any]:
    a, b, c, total, n = ss
    return dict(
        backbone=backbone,
        fold=1,
        n_languages=len(ls),
        n_entities=len(es),
        level=level,
        layer=layer,
        component=comp,
        tensor=key,
        n_parameters=int(n),
        ss_language=a,
        ss_entity=b,
        ss_interaction=c,
        ss_total=total,
        fraction_language=a / total if total > EPS else math.nan,
        fraction_entity=b / total if total > EPS else math.nan,
        fraction_interaction=c / total if total > EPS else math.nan,
        interpretation="descriptive_one_checkpoint_per_cell_no_inference",
    )
