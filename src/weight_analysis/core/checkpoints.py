"""Explicit checkpoint identities and conservative legacy experiment discovery."""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass
from pathlib import Path

LOG = logging.getLogger(__name__)
EXTENSIONS = (
    "model.safetensors.index.json",
    "model.safetensors",
    "pytorch_model.bin.index.json",
    "pytorch_model.bin",
)
VALID_LANGUAGES = frozenset({"cz", "en", "es", "it", "nl", "ro", "sv"})
VALID_ENTITIES = frozenset({"disease", "procedure", "symptom"})


@dataclass(frozen=True, order=True)
class Checkpoint:
    backbone: str
    language: str
    entity: str
    path: Path
    fold: int = 1

    @property
    def uid(self) -> str:
        return f"{self.backbone}::{self.language}::{self.entity}::{self.fold}"


@dataclass(frozen=True)
class CheckpointSpec:
    """Describe one physical checkpoint without inferring experimental metadata.

    Identifiers include the resolved path and every declared dimension. A checkpoint_id
    is an optional experiment label, not an override for fold or seed. Identifiers are
    stable for identical declarations; moving a checkpoint changes its identifier.
    No experiment-specific language or entity vocabulary is imposed.
    """

    backbone: str
    language: str
    entity: str
    path: Path | str
    fold: int | None = 1
    seed: int | None = None
    checkpoint_id: str | None = None

    def __post_init__(self) -> None:
        for name in ("backbone", "language", "entity"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{name} must be a nonempty string; got {value!r}")
        if self.fold is not None and (
            isinstance(self.fold, bool) or not isinstance(self.fold, int) or self.fold < 1
        ):
            raise ValueError(f"fold must be a positive integer or None; got {self.fold!r}")
        if self.seed is not None and (
            isinstance(self.seed, bool) or not isinstance(self.seed, int)
        ):
            raise ValueError(f"seed must be an integer or None; got {self.seed!r}")
        if self.checkpoint_id is not None and (
            not isinstance(self.checkpoint_id, str) or not self.checkpoint_id.strip()
        ):
            raise ValueError("checkpoint_id must be a nonempty string or None")
        object.__setattr__(self, "path", Path(self.path).expanduser().resolve())

    @property
    def uid(self) -> str:
        """Return the deterministic, unambiguous identifier of this declaration."""
        payload = {
            "backbone": self.backbone,
            "language": self.language,
            "entity": self.entity,
            "path": str(self.path),
            "fold": self.fold,
            "seed": self.seed,
            "checkpoint_id": self.checkpoint_id,
        }
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
        return "checkpoint:" + hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def weight_files(directory: Path | str) -> list[Path]:
    """Return the preferred recognized checkpoint format without materializing it."""
    directory = Path(directory).expanduser()
    safe_idx = directory / EXTENSIONS[0]
    safe_one = directory / EXTENSIONS[1]
    pt_idx = directory / EXTENSIONS[2]
    pt_one = directory / EXTENSIONS[3]
    if safe_idx.is_file():
        return [safe_idx]
    if safe_one.is_file():
        return [safe_one]
    if pt_idx.is_file():
        return [pt_idx]
    if pt_one.is_file():
        return [pt_one]
    return []


def discover(
    originals: Path | str, trained: Path | str
) -> tuple[dict[str, Path], list[Checkpoint]]:
    """Discover only weights/<language>/<entity>/<backbone> checkpoint directories.

    Restrict the top-level languages and entities to the known MultiClinAI
    labels; never recurse into hidden backups such as `.overwrite_v2`.
    Within a valid language/entity, a visible model directory without weights
    remains a hard error so failed or partial training is not ignored.
    """
    originals = Path(originals).expanduser()
    trained = Path(trained).expanduser()
    if not originals.is_dir() or not trained.is_dir():
        raise FileNotFoundError(f"Originals={originals} and trained={trained} must be directories")

    bases = {
        d.name: d
        for d in sorted(originals.iterdir())
        if d.is_dir() and not d.name.startswith(".") and weight_files(d)
    }
    records: list[Checkpoint] = []
    invalid: list[str] = []
    for language in sorted(trained.iterdir()):
        if not language.is_dir() or language.name.startswith("."):
            continue
        lang = language.name.lower()
        if lang not in VALID_LANGUAGES:
            LOG.warning("Skipping non-MultiClinAI directory under weights/: %s", language)
            continue
        for entity in sorted(language.iterdir()):
            if not entity.is_dir() or entity.name.startswith("."):
                continue
            label = entity.name.lower()
            if label not in VALID_ENTITIES:
                raise ValueError(f"Unexpected entity directory for {lang}: {entity}")
            for model in sorted(entity.iterdir()):
                if not model.is_dir() or model.name.startswith("."):
                    continue
                if weight_files(model):
                    records.append(Checkpoint(model.name, lang, label, model))
                else:
                    invalid.append(str(model))
    if not records:
        raise ValueError(f"No trained checkpoints in {trained}/<language>/<entity>/<backbone>")
    if invalid:
        raise ValueError(f"Model dirs with no recognized weights: {invalid[:20]}")
    unknown = sorted({x.backbone for x in records if x.backbone not in bases})
    if unknown:
        raise ValueError(f"Trained backbones without ORIGINAL weights: {unknown}")
    if len({x.uid for x in records}) != len(records):
        raise ValueError("Duplicate checkpoint IDs")
    return bases, sorted(records)
