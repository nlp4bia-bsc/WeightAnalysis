"""Safe local checkpoint reading and model configuration metadata."""

from __future__ import annotations

import json
from contextlib import ExitStack
from pathlib import Path
from typing import Any

import torch
from safetensors import safe_open

from .checkpoints import weight_files
from .parameters import normalize

CONFIG_FIELDS = (
    "model_type",
    "hidden_size",
    "num_hidden_layers",
    "num_attention_heads",
    "vocab_size",
    "architectures",
)
FLOAT_DTYPES = frozenset({"F16", "F32", "F64", "BF16"})


class WeightReader:
    """Read local safetensors or weights-only PyTorch state dictionaries on CPU.

    Safetensors are lazy: shape inspection does not materialize their values. PyTorch
    files use weights_only=True, with memory mapping when their format supports it.
    The reader owns file handles only while its context is open.
    """

    def __init__(self, path: Path | str) -> None:
        self.path = Path(path).expanduser()
        self.model_type = config_metadata(self.path).get("model_type")
        self.stack = ExitStack()
        self.handles: dict[Path, Any] = {}
        self.pt: dict[Path, dict[str, torch.Tensor]] = {}
        self.mapping: dict[str, tuple[Path, str]] = {}
        self.raw_names: dict[str, str] = {}
        self._metadata: dict[str, tuple[tuple[int, ...], bool]] = {}

    def __enter__(self) -> WeightReader:
        try:
            self._open()
        except BaseException:
            self.__exit__()
            raise

        return self

    def _open(self) -> None:
        selected = weight_files(self.path)
        if not selected:
            raise FileNotFoundError(f"No weights in {self.path}")
        selection = selected[0]
        if selection.suffix != ".json":
            for key in self._keys(selection):
                self._add(key, selection)
        else:
            data = json.loads(selection.read_text(encoding="utf-8"))
            mapping = data.get("weight_map") if isinstance(data, dict) else None
            if not isinstance(mapping, dict) or not mapping:
                raise ValueError(f"Bad HF index: {selection}")
            for key, name in mapping.items():
                if not isinstance(name, str) or not name or Path(name).name != name:
                    raise ValueError(f"Unsafe shard name: {name}")
                shard_path = self.path / name
                if not shard_path.is_file():
                    raise FileNotFoundError(shard_path)
                self._add(key, shard_path)
            for shard_path in sorted({file for file, _ in self.mapping.values()}):
                actual = set(self._keys(shard_path))
                required = {raw for file, raw in self.mapping.values() if file == shard_path}
                if not required <= actual:
                    raise ValueError(f"Shard index mismatch: {shard_path}: {required - actual}")
        if not self.mapping:
            raise ValueError(f"No tensors found in {self.path}")

    def _add(self, raw: str, file: Path) -> None:
        if not isinstance(raw, str):
            raise ValueError(f"Checkpoint parameter names must be strings: {raw!r}")
        normalized = normalize(raw, self.model_type)
        if normalized in self.mapping:
            raise ValueError(f"Normalized-key collision: {normalized}, {self.path}")
        self.mapping[normalized] = (file, raw)
        self.raw_names[normalized] = raw

    def _keys(self, file: Path) -> Any:
        if file.suffix == ".safetensors":
            if file not in self.handles:
                self.handles[file] = self.stack.enter_context(
                    safe_open(str(file), framework="pt", device="cpu")
                )
            return self.handles[file].keys()
        if file not in self.pt:
            try:
                loaded = torch.load(file, map_location="cpu", weights_only=True, mmap=True)
            except (TypeError, RuntimeError, ValueError):
                loaded = torch.load(file, map_location="cpu", weights_only=True)
            for wrapper in ("state_dict", "model_state_dict", "model"):
                if isinstance(loaded, dict) and isinstance(loaded.get(wrapper), dict):
                    loaded = loaded[wrapper]
                    break
            if not isinstance(loaded, dict):
                raise ValueError(f"Expected a state_dict: {file}")
            self.pt[file] = {
                str(key): value for key, value in loaded.items() if isinstance(value, torch.Tensor)
            }

        return self.pt[file].keys()

    def shape_dtype(self, key: str) -> tuple[tuple[int, ...], bool]:
        """Return shape and floating status without materializing safetensors values."""
        if key not in self._metadata:
            file, raw = self.mapping[key]
            self._keys(file)
            if file.suffix == ".safetensors":
                tensor_slice = self.handles[file].get_slice(raw)
                metadata = (
                    tuple(tensor_slice.get_shape()),
                    tensor_slice.get_dtype() in FLOAT_DTYPES,
                )
            else:
                tensor = self.pt[file][raw]
                metadata = (tuple(tensor.shape), tensor.is_floating_point())
            self._metadata[key] = metadata

        return self._metadata[key]

    @property
    def keys(self) -> set[str]:
        """Return a copy of the canonical parameter names in this checkpoint."""
        return set(self.mapping)

    def get(self, key: str) -> torch.Tensor:
        """Materialize one canonical tensor on CPU without changing its dtype."""
        file, raw = self.mapping[key]
        self._keys(file)
        tensor = (
            self.handles[file].get_tensor(raw)
            if file.suffix == ".safetensors"
            else self.pt[file][raw]
        )

        return tensor.detach().cpu()

    def __exit__(self, *_: Any) -> None:
        self.stack.close()
        self.handles.clear()
        self.pt.clear()
        self.mapping.clear()
        self.raw_names.clear()
        self._metadata.clear()


def config_metadata(path: Path | str, *, strict: bool = False) -> dict[str, Any]:
    """Read supported HF configuration fields without requiring Transformers.

    The legacy default tolerates missing or malformed configuration. strict=True rejects
    malformed files so callers can distinguish unsupported metadata from damaged JSON.
    A missing config is allowed in either mode because raw state dictionaries need none.
    """
    config_path = Path(path).expanduser() / "config.json"
    if not config_path.is_file():
        return {}
    try:
        data = json.loads(config_path.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            raise ValueError(f"Configuration must be a JSON object: {config_path}")
    except (ValueError, OSError):
        if strict:
            raise
        return {}

    return {key: data.get(key) for key in CONFIG_FIELDS}
