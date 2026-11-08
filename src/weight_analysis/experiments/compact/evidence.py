"""Coordinate evidence, without assuming that equal tensor shapes imply ancestry."""

from __future__ import annotations

import json
from pathlib import Path

from ...core.catalog import stable_id
from ...provenance import file_sha256


def vocabulary_digest(directory):
    directory = Path(directory)
    vocabulary = None
    added = []
    if (directory / "tokenizer.json").is_file():
        data = json.loads((directory / "tokenizer.json").read_text())
        vocabulary = data.get("model", {}).get("vocab")
        if isinstance(vocabulary, list):
            vocabulary = {row[0]: index for index, row in enumerate(vocabulary)}
        added = data.get("added_tokens", [])
    elif (directory / "vocab.json").is_file():
        vocabulary = json.loads((directory / "vocab.json").read_text())
    elif (directory / "vocab.txt").is_file():
        vocabulary = {
            token: index
            for index, token in enumerate((directory / "vocab.txt").read_text().splitlines())
        }
    if not isinstance(vocabulary, dict) or not vocabulary:
        return None
    vocabulary = dict(vocabulary)
    extra_path = directory / "added_tokens.json"
    if extra_path.is_file():
        added += [
            dict(content=token, id=index)
            for token, index in json.loads(extra_path.read_text()).items()
        ]
    for item in added:
        token, index = item["content"], item["id"]
        if token in vocabulary and vocabulary[token] != index:
            return None
        vocabulary[token] = index
    if not all(
        isinstance(token, str) and isinstance(index, int) and index >= 0
        for token, index in vocabulary.items()
    ) or len(set(vocabulary.values())) != len(vocabulary):
        return None
    return stable_id(vocabulary)


def embedding_evidence(checkpoint, reference, *, cache=None):
    try:
        return _embedding_evidence(checkpoint, reference, cache=cache)
    except (OSError, ValueError, TypeError, KeyError) as exc:
        return dict(
            compatible=False,
            kind="unreadable_or_invalid_vocabulary_evidence",
            evidence_id=None,
            error=str(exc),
        )


def _embedding_evidence(checkpoint, reference, *, cache=None):
    cache = cache if cache is not None else {}

    def digest(path):
        key = str(path)
        if key not in cache:
            cache[key] = vocabulary_digest(path)
        return cache[key]

    left, right = digest(checkpoint), digest(reference)
    if left and right and left == right:
        return dict(compatible=True, kind="verified_token_to_index_mapping", evidence_id=left)

    def binary_evidence(path):
        path = Path(path)
        model = next(
            (
                path / name
                for name in ("spiece.model", "sentencepiece.bpe.model")
                if (path / name).is_file()
            ),
            None,
        )
        if model is None:
            return None
        return stable_id(
            dict(
                model_sha256=file_sha256(model),
                metadata={
                    name: file_sha256(path / name)
                    for name in ("added_tokens.json", "tokenizer_config.json")
                    if (path / name).exists()
                },
            )
        )

    token_left, token_right = binary_evidence(checkpoint), binary_evidence(reference)
    if token_left is not None and token_left == token_right:
        return dict(
            compatible=True, kind="identical_sentencepiece_and_added_tokens", evidence_id=token_left
        )
    return dict(compatible=False, kind="missing_or_different_vocabulary_evidence", evidence_id=None)
