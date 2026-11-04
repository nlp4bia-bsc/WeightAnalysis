"""Drift and spectral fingerprints for one aligned checkpoint."""

from __future__ import annotations

import logging
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch

from ..core import (
    SPECTRAL_COMPONENTS,
    Checkpoint,
    WeightReader,
    audit_alignment,
    config_metadata,
    get_delta,
    head_views,
    identify,
)
from ..metrics import EPS, drift, spectral

LOG = logging.getLogger(__name__)


def common(cp: Checkpoint) -> dict[str, Any]:
    """Return the historical checkpoint identity columns."""
    return dict(
        checkpoint=cp.uid,
        trained_path=str(cp.path),
        backbone=cp.backbone,
        language=cp.language,
        entity=cp.entity,
        fold=cp.fold,
    )


def scope_rows(
    info: dict[str, Any], metric: dict[str, Any], level: str = "tensor", **extra: Any
) -> dict[str, Any]:
    """Combine an observation's identity, metrics and scope columns."""
    return dict(level=level, **info, **metric, **extra)


def describe_checkpoint(
    cp: Checkpoint,
    base_path: Path,
    settings: Any,
    base_spectral_cache: dict | None = None,
    *,
    compute_drift: bool = True,
    compute_spectra: bool = True,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """A1: compute delta drift; build separate spectral fingerprints for base/final/delta."""
    from ..devices import resolve_device

    compute_device = resolve_device(getattr(settings, "device", "cpu"))
    drift_rows = []
    spectra = []
    aggregate = defaultdict(lambda: dict(n=0, ds=0.0, bs=0.0, fs=0.0, dot=0.0, l1=0.0, maximum=0.0))
    head_config = config_metadata(base_path)
    with WeightReader(cp.path) as r, WeightReader(base_path) as b:
        keys, status = audit_alignment(r, b, settings.min_aligned_fraction)
        LOG.info(
            "A1 %s matched %.2f%% of encoder elements, %d tensors",
            cp.uid,
            100 * status["element_coverage"],
            len(keys),
        )
        layers = [identify(k, r.shape_dtype(k)[0])[0] for k in keys]
        recognized_elements = sum(
            math.prod(r.shape_dtype(k)[0])
            for k in keys
            if identify(k, r.shape_dtype(k)[0])[1] not in ("other", "norm", "embeddings")
        )
        recognized_fraction = (
            recognized_elements / status["matched_elements"] if status["matched_elements"] else 0.0
        )
        if recognized_fraction < 0.35:
            LOG.warning(
                "Only %.2f%% of encoder elements are recognized attention/FFN projections for %s; spectral pair comparisons may be unavailable",
                100 * recognized_fraction,
                cp.uid,
            )
        maxlayer = max((z for z in layers if z is not None), default=0)
        for key in keys:
            fin, orig, d = get_delta(r, b, key, device=compute_device)
            layer, comp, _ = identify(key, tuple(fin.shape))
            loc = dict(
                **common(cp),
                original_path=str(base_path),
                layer=layer,
                component=comp,
                tensor=key,
                n_aligned_elements=status["matched_elements"],
                element_coverage=status["element_coverage"],
                recognized_projection_fraction=recognized_fraction,
            )
            if compute_drift:
                stats = drift(fin, orig)
                drift_rows.append(scope_rows(loc, stats))
                # Per-head comparisons for standard BERT-family projection matrices.
                for (head_i, wh), (head_j, bh) in zip(
                    head_views(fin, comp, head_config), head_views(orig, comp, head_config)
                ):
                    if head_i != head_j:
                        raise ValueError("Attention head index mismatch")
                    drift_rows.append(
                        scope_rows(
                            {**loc, "tensor": key + f"::head{head_i}", "head_index": head_i},
                            drift(wh, bh),
                            level="head",
                        )
                    )
            if compute_spectra and d.ndim == 2 and comp in SPECTRAL_COMPONENTS:
                for source, tensor in [("base", orig), ("final", fin), ("delta", d)]:
                    cache_key = (cp.backbone, key)
                    if (
                        source == "base"
                        and base_spectral_cache is not None
                        and cache_key in base_spectral_cache
                    ):
                        metrics = base_spectral_cache[cache_key]
                    else:
                        metrics = spectral(
                            tensor,
                            settings.spectral_mode,
                            settings.max_svd_elements,
                            settings.topk,
                            settings.device,
                        )
                        if source == "base" and base_spectral_cache is not None:
                            base_spectral_cache[cache_key] = metrics
                    spectra.append(
                        scope_rows(
                            loc,
                            metrics,
                            source=source,
                            relative_depth=(
                                layer / maxlayer if layer is not None and maxlayer > 0 else None
                            ),
                        )
                    )
            if compute_drift:
                for scope, lay, section in [
                    ("encoder", None, None),
                    ("layer", layer, None),
                    ("component", None, comp),
                    ("layer_component", layer, comp),
                ]:
                    a = aggregate[scope, lay, section]
                    a["n"] += d.numel()
                    a["ds"] += stats["delta_l2"] ** 2
                    a["bs"] += stats["base_l2"] ** 2
                    a["fs"] += stats["final_l2"] ** 2
                    a["dot"] += float(torch.dot(fin.flatten(), orig.flatten()))
                    a["l1"] += stats["delta_l1"]
                    a["maximum"] = max(a["maximum"], stats["max_abs_delta"])
        if compute_drift:
            for (scope, layer, comp), a in aggregate.items():
                ds = math.sqrt(a["ds"])
                bs = math.sqrt(a["bs"])
                fs = math.sqrt(a["fs"])
                drift_rows.append(
                    dict(
                        **common(cp),
                        original_path=str(base_path),
                        level=scope,
                        layer=layer,
                        component=comp,
                        tensor=None,
                        n_parameters=a["n"],
                        n_aligned_elements=status["matched_elements"],
                        element_coverage=status["element_coverage"],
                        recognized_projection_fraction=recognized_fraction,
                        delta_l2=ds,
                        base_l2=bs,
                        final_l2=fs,
                        delta_energy=a["ds"],
                        relative_l2=(ds / bs if bs > EPS else math.nan),
                        delta_rms=ds / math.sqrt(a["n"]),
                        delta_mae=a["l1"] / a["n"],
                        delta_l1=a["l1"],
                        max_abs_delta=a["maximum"],
                        cosine_final_base=a["dot"] / (fs * bs)
                        if fs > EPS and bs > EPS
                        else math.nan,
                    )
                )
    return drift_rows, spectra
