#!/usr/bin/env python3
"""Freeze a *selection* of completed NER folders using symlinks, not copies.

This makes a trained/ tree whose list of checkpoints stays constant while other
training jobs add new model folders to the original trained/ directory.
Finished checkpoint files themselves MUST remain immutable during analysis.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import uuid
from pathlib import Path

from ..core import VALID_ENTITIES, VALID_LANGUAGES, weight_files


def create_snapshot(trained: Path, snapshot: Path, excluded: set[str], expected: int | None) -> int:
    trained = trained.expanduser().resolve(strict=True)
    snapshot = snapshot.expanduser().absolute()
    if snapshot.exists() or snapshot.is_symlink():
        raise FileExistsError(f"Snapshot already exists: {snapshot}; choose a new path")
    if snapshot == trained or trained in snapshot.parents:
        raise ValueError("Snapshot must NOT be inside the live weights tree")
    snapshot.parent.mkdir(parents=True, exist_ok=True)
    tmp = snapshot.parent / ("." + snapshot.name + "." + uuid.uuid4().hex + ".tmp")
    tmp.mkdir()
    included: list[str] = []
    missing: list[str] = []
    try:
        for lang_dir in sorted(trained.iterdir()):
            if not lang_dir.is_dir() or lang_dir.name not in VALID_LANGUAGES:
                continue
            for ent_dir in sorted(lang_dir.iterdir()):
                if not ent_dir.is_dir() or ent_dir.name not in VALID_ENTITIES:
                    continue
                for model_dir in sorted(ent_dir.iterdir()):
                    if not model_dir.is_dir() or model_dir.name.startswith("."):
                        continue
                    key = f"{lang_dir.name}/{ent_dir.name}/{model_dir.name}"
                    if key in excluded:
                        continue
                    if not weight_files(model_dir) or not (model_dir / "config.json").is_file():
                        missing.append(key)
                        continue
                    dst = tmp / lang_dir.name / ent_dir.name / model_dir.name
                    dst.parent.mkdir(parents=True, exist_ok=True)
                    dst.symlink_to(model_dir.resolve(strict=True), target_is_directory=True)
                    included.append(key)
        unknown = excluded - {
            f"{l.name}/{e.name}/{m.name}"
            for l in trained.iterdir()
            if l.is_dir() and l.name in VALID_LANGUAGES
            for e in l.iterdir()
            if e.is_dir() and e.name in VALID_ENTITIES
            for m in e.iterdir()
            if m.is_dir()
        }
        if unknown:
            # It is fine if excluded training directories are not created yet.
            print(
                "Excluded paths not yet present (expected for pending training):",
                sorted(unknown),
                file=sys.stderr,
            )
        if missing:
            raise RuntimeError(f"Unexpected incomplete checkpoint folders: {missing[:10]}")
        if expected is not None and len(included) != expected:
            raise RuntimeError(
                f"Found {len(included)} completed checkpoints; expected {expected}. Snapshot NOT published."
            )
        if not included:
            raise RuntimeError("No completed checkpoints found")
        (tmp / ".snapshot_info.json").write_text(
            json.dumps(
                {"source": str(trained), "included": included, "excluded": sorted(excluded)},
                indent=2,
            ),
            encoding="utf8",
        )
        os.rename(tmp, snapshot)
    except BaseException:
        shutil.rmtree(tmp, ignore_errors=True)
        raise
    print(f"SNAPSHOT={snapshot}")
    print(f"CHECKPOINTS={len(included)}")
    print("Snapshot checkpoint selection will not grow when new NER folders appear.")
    print("WARNING: checkpoint files accessed via links must not be modified during this run.")
    return len(included)


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--trained", type=Path, required=True)
    p.add_argument("--snapshot", type=Path, required=True)
    p.add_argument("--expected", type=int, default=None)
    p.add_argument(
        "--exclude",
        action="append",
        default=[],
        help="Relative path language/entity/backbone to omit",
    )
    args = p.parse_args(argv)
    create_snapshot(args.trained, args.snapshot, set(args.exclude), args.expected)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"SNAPSHOT ERROR: {exc}", file=sys.stderr)
        raise SystemExit(2)
