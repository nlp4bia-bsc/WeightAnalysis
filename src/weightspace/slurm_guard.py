"""Immutable manifest for one Picasso SLURM weight-space analysis run.

Every task revalidates the HF file inventory before doing scientific work.
No tensor files are loaded, and no external network or model inference is used.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import uuid
from pathlib import Path

from .checkpoints import discover, weight_files


def _files_in_model(directory: Path) -> list[Path]:
    selected = weight_files(directory)
    if not selected:
        raise FileNotFoundError(f'No checkpoint weights: {directory}')
    files = list(selected)
    if selected[0].name.endswith('.index.json'):
        index = json.loads(selected[0].read_text(encoding='utf-8'))
        names = index.get('weight_map', {}).values()
        for name in set(names):
            if Path(name).name != name:
                raise ValueError(f'Unexpected index filename in {selected[0]}: {name}')
            files.append(directory / name)
    config = directory / 'config.json'
    if not config.is_file():
        raise FileNotFoundError(f'Missing model config: {config}')
    return [config, *files]


def inventory(originals: Path, trained: Path) -> dict:
    originals = originals.expanduser().resolve()
    trained = trained.expanduser().resolve()
    bases, checkpoints = discover(originals, trained)
    paths = sorted({*bases.values(), *(c.path for c in checkpoints)})
    measurements = []
    for folder in paths:
        for p in _files_in_model(folder):
            stat = p.stat()
            measurements.append([str(p.resolve()), stat.st_size, stat.st_mtime_ns])
    checkpoint_ids = [c.uid for c in checkpoints]
    return dict(
        originals=str(originals),
        trained=str(trained),
        backbones=sorted(bases),
        checkpoint_ids=checkpoint_ids,
        strata=sorted([list(x) for x in {(c.language, c.entity) for c in checkpoints}]),
        files=sorted(measurements),
    )


def create(run_dir: Path, originals: Path, trained: Path, output: Path, splits: int, performance: Path | None) -> dict:
    if splits < 1:
        raise ValueError('splits-per-backbone must be positive')
    run_dir = run_dir.expanduser().resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    dest = run_dir / 'manifest.json'
    if dest.exists():
        raise FileExistsError(f'Run manifest already exists: {dest}')
    inv = inventory(originals, trained)
    scores = None
    if performance is not None:
        performance = performance.expanduser().resolve()
        st = performance.stat()
        scores = [str(performance), st.st_size, st.st_mtime_ns]
    content = dict(
        inventory=inv,
        output=str(output.expanduser().resolve()),
        splits_per_backbone=splits,
        performance=scores,
        num_backbones=len(inv['backbones']),
        num_strata=len(inv['strata']),
        num_checkpoints=len(inv['checkpoint_ids']),
    )
    tmp = dest.with_name(dest.name + '.' + uuid.uuid4().hex + '.tmp')
    tmp.write_text(json.dumps(content, sort_keys=True, indent=2), encoding='utf8')
    os.replace(tmp, dest)
    return content


def check(run_dir: Path) -> dict:
    dest = run_dir.expanduser().resolve() / 'manifest.json'
    manifest = json.loads(dest.read_text(encoding='utf8'))
    inv = manifest['inventory']
    actual = inventory(Path(inv['originals']), Path(inv['trained']))
    if actual != inv:
        old_ids, new_ids = set(inv['checkpoint_ids']), set(actual['checkpoint_ids'])
        raise ValueError('Checkpoint inventory changed since submission. '
                         f'added={sorted(new_ids-old_ids)[:8]}, '
                         f'removed={sorted(old_ids-new_ids)[:8]}. '
                         'Even an unchanged checkpoint count can hide modified weights. '
                         'Stop and start a fresh run after training is complete.')
    scores = manifest['performance']
    if scores is not None:
        p = Path(scores[0]); st = p.stat()
        if [str(p), st.st_size, st.st_mtime_ns] != scores:
            raise ValueError('Performance input changed since submission')
    return manifest


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['create', 'check', 'counts', 'info'])
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--originals', type=Path)
    parser.add_argument('--trained', type=Path)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--splits-per-backbone', type=int, default=3)
    parser.add_argument('--performance', type=Path)
    a = parser.parse_args(argv)
    if a.action == 'create':
        if not all((a.originals, a.trained, a.output)):
            parser.error('create requires --originals --trained --output')
        m = create(a.run_dir, a.originals, a.trained, a.output,
                   a.splits_per_backbone, a.performance)
        print(f"Frozen {m['num_backbones']} originals, {m['num_checkpoints']} trained, "
              f"{m['num_strata']} language/entity strata; manifest={a.run_dir/'manifest.json'}")
    elif a.action == 'check':
        check(a.run_dir)
    else:
        # Counts/paths already frozen in the manifest; no re-scan needed here.
        m = json.loads((a.run_dir.expanduser().resolve() / 'manifest.json').read_text(encoding='utf8'))
        if a.action == 'counts':
            print(m['num_backbones'], m['num_strata'], m['splits_per_backbone'])
        else:
            for val in (m['num_backbones'], m['num_strata'],m['splits_per_backbone'],
                        m['output'], m['inventory']['originals'],m['inventory']['trained'],
                        m['performance'][0] if m['performance'] else ''):
                print(val)


if __name__ == '__main__':
    try:
        main()
    except Exception as exc:
        print(f'WEIGHTSPACE RUN GUARD FAILED: {exc}', file=sys.stderr)
        raise SystemExit(2)
