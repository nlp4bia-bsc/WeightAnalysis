"""Selected parameter coordinates in bounded, contiguous blocks."""

from __future__ import annotations

import math
from itertools import product

from .parameters import identify, selected_keys


def embedding_type(key: str) -> str:
    if "position" in key:
        return "position"
    if "token_type" in key or "segment" in key:
        return "token_type"
    if "word_embeddings" in key or "embed_tokens" in key:
        return "token"
    return "other"


def parameter_location(reader, key):
    shape, _ = reader.shape_dtype(key)
    layer, component, _ = identify(key, shape)
    return {
        "tensor": key,
        "layer": layer,
        "component": component,
        "embedding_type": embedding_type(key) if component == "embeddings" else None,
    }


def iter_blocks(reader, key: str, max_elements: int = 100_000):
    """Slice safetensors lazily; PyTorch formats retain their reader's mmap/fallback policy."""
    if max_elements < 1:
        raise ValueError("max_elements must be positive")
    shape, _ = reader.shape_dtype(key)
    file, raw = reader.mapping[key]
    source = (
        reader.handles[file].get_slice(raw) if file.suffix == ".safetensors" else reader.get(key)
    )
    if not shape:
        yield 0, reader.get(key).reshape(-1)
        return
    row_size = math.prod(shape[1:])
    if row_size <= max_elements:
        rows = max(1, max_elements // row_size)
        for first in range(0, shape[0], rows):
            yield first * row_size, source[first : first + rows].reshape(-1)
    else:
        width = shape[-1]
        for row, prefix in enumerate(product(*(range(n) for n in shape[:-1]))):
            for first in range(0, width, max_elements):
                yield (
                    row * width + first,
                    source[(*prefix, slice(first, first + max_elements))].reshape(-1),
                )


def select_parameters(reader, selection):
    return [
        key for key in selected_keys(reader) if selection.matches(parameter_location(reader, key))
    ]
