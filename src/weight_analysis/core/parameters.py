"""Encoder parameter names, metadata selection, alignment, and tensor deltas."""

from __future__ import annotations

import math
import re
from typing import TYPE_CHECKING, Any, Iterable, Iterator

import torch

if TYPE_CHECKING:
    from .reading import WeightReader

BACKBONES = (
    "bert",
    "roberta",
    "xlm_roberta",
    "deberta",
    "deberta_v2",
    "distilbert",
    "electra",
    "camembert",
    "albert",
)
COMPONENTS = ("q", "k", "v", "o", "ffn_up", "ffn_down", "ffn_gate", "norm", "embeddings", "other")
SPECTRAL_COMPONENTS = ("q", "k", "v", "o", "ffn_up", "ffn_down", "ffn_gate")
HF_ENCODER_PREFIX = {
    "bert": "bert",
    "roberta": "roberta",
    "xlm-roberta": "roberta",
    "camembert": "roberta",
    "electra": "electra",
    "distilbert": "distilbert",
    "deberta": "deberta",
    "deberta-v2": "deberta",
    "albert": "albert",
}
BARE_ENCODER_COMPONENTS = frozenset({"embeddings", "encoder", "pooler", "transformer"})


def normalize(raw: str, model_type: str | None = None) -> str:
    """Canonical HF encoder key, preserving architecture family identity.

    Known transport wrappers are stripped first. When config.json identifies a
    supported family, an HF base-model key such as 'encoder.layer.0...' is
    mapped to 'bert.encoder.layer.0...' for model_type='bert'. A key that
    already begins with 'bert.' is unchanged. Unsupported architectures are
    deliberately not guessed. Key collisions are rejected by WeightReader.
    """
    key = raw
    while True:
        prior = key
        for prefix in ("module.", "base_model.model.", "base_model.", "model."):
            if key.startswith(prefix):
                key = key[len(prefix) :]
                break
        if key == prior:
            break
    root = HF_ENCODER_PREFIX.get(model_type or "")
    if root and key.split(".", 1)[0] in BARE_ENCODER_COMPONENTS:
        return f"{root}.{key}"
    return key


def identify(key: str, shape: tuple[int, ...]) -> tuple[int | None, str, bool]:
    """Identify conservative common encoder components; unknowns kept for auditing."""
    low = key.lower()
    if re.search(r"(^|\.)(classifier|score|qa_outputs|crf|lm_head)(\.|$)", low):
        return None, "ner_head", True
    match = re.search(
        r"(?:encoder\.layer|transformer\.layer|transformer\.h|layers|block|layer)\.(\d+)\.", low
    )
    layer = int(match.group(1)) if match else None
    if any(t in low for t in ("embeddings.", "embed_tokens.")):
        return layer, "embeddings", False
    if "layernorm" in low or "layer_norm" in low or re.search(r"(^|\.)norm\.", low):
        return layer, "norm", False
    rules = [
        (r"(?:^|\.)(?:query|q_proj|q_lin)\.", "q"),
        (r"(?:^|\.)(?:key|k_proj|k_lin)\.", "k"),
        (r"(?:^|\.)(?:value|v_proj|v_lin)\.", "v"),
        (r"(?:attention\.output\.dense|self_attn\.out_proj|o_proj|attention\.out_lin)\.", "o"),
        (r"(?:intermediate\.dense|fc1|up_proj|ffn\.lin1)\.", "ffn_up"),
        (r"(?:output\.dense|fc2|down_proj|ffn\.lin2)\.", "ffn_down"),
        (r"(?:gate_proj)\.", "ffn_gate"),
    ]
    for rx, component in rules:
        if re.search(rx, low):
            # Only assign 'output.dense' to FFN when it is not attention.output.dense.
            if component == "ffn_down" and "attention.output.dense." in low:
                continue
            return layer, component, False
    return layer, "other", False


def audit_alignment(
    r: WeightReader, b: WeightReader, min_fraction: float = 0.99
) -> tuple[list[str], dict[str, Any]]:
    """Reject missing encoder tensors by ELEMENT coverage; no head subtraction.

    Config/name matching cannot prove an identical original checkpoint; folder
    provenance must still be confirmed by the training setup.
    """
    if not math.isfinite(min_fraction) or not 0 < min_fraction <= 1:
        raise ValueError("min_fraction must be finite and in (0, 1]")
    base_keys = b.keys
    keys = []
    missing = []
    mismatches = []
    matched_n = 0
    candidate_n = 0
    for key in sorted(r.keys):
        shape, floating = r.shape_dtype(key)
        if not floating:
            continue  # HF buffers such as position_ids are NOT weights.
        layer, component, head = identify(key, shape)
        if head:
            continue
        # Keep unknown floating encoder components for explicit validation.
        candidate_n += math.prod(shape)
        if key not in base_keys:
            missing.append(key)
            continue
        w0_shape, w0_floating = b.shape_dtype(key)
        if w0_shape != shape or not w0_floating:
            mismatches.append(key)
            continue
        keys.append(key)
        matched_n += math.prod(shape)
    coverage = matched_n / candidate_n if candidate_n else 0
    status = dict(
        matched_elements=matched_n,
        candidate_elements=candidate_n,
        element_coverage=coverage,
        missing_keys=missing,
        mismatched_keys=mismatches,
        n_matched_tensors=len(keys),
    )
    if not keys or coverage < min_fraction:
        raise ValueError(
            f"Invalid base/train alignment ({coverage:.2%} elements); missing={missing[:8]}, mismatch={mismatches[:8]}"
        )
    return keys, status


def get_delta(
    r: WeightReader, b: WeightReader, key: str
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Read one aligned pair and return final, original, and delta in float64."""
    fin, ini = r.get(key), b.get(key)
    if fin.shape != ini.shape or not fin.is_floating_point() or not ini.is_floating_point():
        raise ValueError(f"Incompatible parameter {key}")
    fin = fin.to(torch.float64)
    ini = ini.to(torch.float64)
    if not torch.isfinite(fin).all() or not torch.isfinite(ini).all():
        raise ValueError(f"Nonfinite weights for {key}")
    return fin, ini, fin - ini


def head_views(
    w: torch.Tensor, component: str, config: dict[str, Any]
) -> Iterator[tuple[int, torch.Tensor]]:
    """Yield (head_index, matrix) only where HF encoder head layout is known.

    Conservative: skip fused qkv, grouped-query, adapters, biases and models
    with unknown attention-head weight conventions.
    """
    if w.ndim != 2 or component not in ("q", "k", "v", "o"):
        return
    if config.get("model_type") not in (
        "bert",
        "roberta",
        "xlm-roberta",
        "distilbert",
        "electra",
        "camembert",
    ):
        return
    heads = config.get("num_attention_heads")
    if not isinstance(heads, int) or heads < 1:
        return
    rows, cols = w.shape
    size = rows if component in ("q", "k", "v") else cols
    if size % heads:
        return
    for h in range(heads):
        start, end = h * (size // heads), (h + 1) * (size // heads)
        view = w[start:end, :] if component in ("q", "k", "v") else w[:, start:end]
        yield h, view


def selected_keys(
    reader: WeightReader,
    *,
    layers: Iterable[int] | None = None,
    components: Iterable[str] | None = None,
    tensors: Iterable[str] | None = None,
) -> list[str]:
    """Select floating encoder parameters using metadata, before reading their values.

    Tensor names are exact canonical names as exposed by reader.keys. None selects all
    values of a dimension; an empty collection selects nothing. Selection never weakens
    a separate full-checkpoint alignment audit. Unsupported components remain available
    as 'other'; task heads and integer buffers are excluded.
    """
    selected_layers = None if layers is None else frozenset(layers)
    selected_components = None if components is None else frozenset(components)
    selected_tensors = None if tensors is None else frozenset(tensors)
    if selected_layers is not None and any(
        isinstance(layer, bool) or not isinstance(layer, int) or layer < 0
        for layer in selected_layers
    ):
        raise ValueError("layers must contain nonnegative integer layer indices")
    if selected_components is not None and not selected_components <= set(COMPONENTS):
        raise ValueError(f"Unknown components: {sorted(selected_components - set(COMPONENTS))}")
    keys = []
    for key in sorted(reader.keys):
        if selected_tensors is not None and key not in selected_tensors:
            continue
        shape, floating = reader.shape_dtype(key)
        layer, component, is_head = identify(key, shape)
        if not floating or is_head:
            continue
        if selected_layers is not None and layer not in selected_layers:
            continue
        if selected_components is not None and component not in selected_components:
            continue
        keys.append(key)

    return keys
