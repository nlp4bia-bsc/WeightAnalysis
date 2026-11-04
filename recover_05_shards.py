#!/usr/bin/env python3
"""Recover only incomplete analysis-05 shards without touching scientific core.

Work for the original 05/0 and 05/2 is subdivided into smaller GPU tasks.
Once all pieces for a shard exist, the last task streams them into the
original part-XXXX.parquet with the original config signature.

IMPORTANT: never change weightspace/*.py while original shards exist; its
code hash is part of the config signature.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import fcntl
from itertools import combinations
import json
import logging
import os
from pathlib import Path
import uuid

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from weightspace import parallel as wp
from weightspace.checkpoints import discover
from weightspace.slurm_guard import check as check_manifest
from weightspace.storage import METADATA, load_performance, with_performance

log = logging.getLogger('recover_05_shards')


def arguments(run_dir: Path):
    manifest = check_manifest(run_dir)
    inventory = manifest['inventory']
    output = Path(manifest['output'])
    stage = output / ('.weightspace_stage_v4_' + run_dir.name)
    argv = [
        '--analysis', '05', '--originals', inventory['originals'],
        '--trained', inventory['trained'], '--output', str(output),
        '--stage-dir', str(stage), '--splits-per-backbone',
        str(manifest['splits_per_backbone']), '--spectral-mode', 'auto',
        '--max-svd-elements', '3000000', '--topk', '32',
        '--subspace-rank', '8', '--device', 'cuda',
    ]
    if manifest['performance']:
        argv += ['--performance', manifest['performance'][0]]
    parsed = wp.parser().parse_args(argv)
    return parsed, stage


def verify_signature(path: Path, signature: str):
    pf = pq.ParquetFile(path)
    value = (pf.schema_arrow.metadata or {}).get(b'config_signature', b'').decode()
    if value != signature:
        raise ValueError(f'Incompatible cached file {path}: signature={value}, expected={signature}')
    return pf


def pieces_path(stage: Path, shard: int) -> Path:
    return stage / '05' / f'.recovery_shard_{shard:04d}'


def pairs_for_shard(args, bases, checkpoints, shard):
    n = wp.shard_count('05', bases, checkpoints, args.splits_per_backbone)
    if not 0 <= shard < n:
        raise ValueError(f'Invalid shard index {shard} for {n} shards')
    back_idx, part = divmod(shard, args.splits_per_backbone)
    backbone = sorted(bases)[back_idx]
    group = [c for c in checkpoints if c.backbone == backbone]
    wp._validate_cp_configs(group, bases)
    selected = [(a, b) for a, b in combinations(group, 2)
                if a.entity != b.entity and a.language != b.language]
    selected = selected[part::args.splits_per_backbone]
    return backbone, selected


def merge_if_ready(stage: Path, shard: int, total: int, signature: str):
    directory = pieces_path(stage, shard)
    directory.mkdir(parents=True, exist_ok=True)
    target = wp._part_path(stage, '05', shard)
    lock = directory / '.merge.lock'
    with lock.open('a+') as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        if target.exists():
            verify_signature(target, signature)
            log.info('Original shard already recovered: %s', target)
            return True
        paths = [directory / f'piece-{i:04d}.parquet' for i in range(total)]
        missing = [p.name for p in paths if not p.is_file()]
        if missing:
            log.info('Shard %s: waiting for %d/%d pieces (%s)', shard, len(missing), total,
                     ', '.join(missing[:6]))
            return False
        files = [verify_signature(p, signature) for p in paths]
        present = [pf for pf in files if pf.metadata.num_rows > 0]
        if not present:
            wp._empty_shard(target, signature)
        else:
            schema = pa.unify_schemas([pf.schema_arrow.remove_metadata() for pf in present],
                                      promote_options='permissive')
            md = {**METADATA, 'analysis': target.stem,
                  'execution': 'parallel_shard', 'final_result': 'false',
                  'config_signature': signature}
            schema = schema.with_metadata({k.encode(): str(v).encode() for k, v in md.items()})
            temp = target.with_name('.' + target.name + '.' + uuid.uuid4().hex + '.partial')
            total_rows = sum(pf.metadata.num_rows for pf in present)
            written = 0
            try:
                with pq.ParquetWriter(temp, schema, compression='zstd', compression_level=5,
                                      use_dictionary=True, write_statistics=True) as writer:
                    for pf in present:
                        for batch in pf.iter_batches(batch_size=16384):
                            incoming = pa.Table.from_batches([batch])
                            cols = [incoming[field.name].cast(field.type, safe=True)
                                    if field.name in incoming.column_names
                                    else pa.nulls(incoming.num_rows, field.type)
                                    for field in schema]
                            table = pa.Table.from_arrays(cols, schema=schema)
                            writer.write_table(table, row_group_size=16384)
                            written += incoming.num_rows
                check = pq.ParquetFile(temp)
                if check.metadata.num_rows != total_rows or written != total_rows:
                    raise IOError(f'Row mismatch {written}/{total_rows}')
                verify_signature(temp, signature)
                os.replace(temp, target)
            finally:
                temp.unlink(missing_ok=True)
            log.info('RECOVERED %s (%d rows from %d pieces)', target, total_rows, total)
        for p in paths:
            p.unlink(missing_ok=True)
        return True


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run-dir', type=Path, required=True)
    p.add_argument('--shard', type=int, choices=[0, 2], required=True)
    p.add_argument('--piece', type=int, required=True)
    p.add_argument('--pieces', type=int, default=8)
    a = p.parse_args()
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
    if a.pieces < 2 or not 0 <= a.piece < a.pieces:
        p.error('Invalid --piece/--pieces')
    run_dir = a.run_dir.expanduser().resolve()
    args, stage = arguments(run_dir)
    signature = wp._configuration_signature(args)
    final_part = wp._part_path(stage, '05', a.shard)
    if final_part.exists():
        verify_signature(final_part, signature)
        log.info('Shard already complete; nothing to do: %s', final_part)
        return
    directory = pieces_path(stage, a.shard)
    path = directory / f'piece-{a.piece:04d}.parquet'
    if path.exists():
        verify_signature(path, signature)
        log.info('Piece already complete: %s', path)
    else:
        import torch
        if not torch.cuda.is_available():
            raise RuntimeError('This recovery piece requires a CUDA GPU on Picasso DGX')
        bases, cps = discover(args.originals, args.trained)
        backbone, selected = pairs_for_shard(args, bases, cps, a.shard)
        work = selected[a.piece::a.pieces]
        log.info('RECOVERY 05 original shard=%d piece=%d/%d backbone=%s pairs=%d of %d',
                 a.shard, a.piece+1, a.pieces, backbone, len(work), len(selected))
        rows, cache = [], {}
        for i, (cp_a, cp_b) in enumerate(work, 1):
            log.info('RECOVERY shard=%d piece=%d pair=%d/%d: %s <-> %s',
                     a.shard, a.piece, i, len(work), cp_a.uid, cp_b.uid)
            rows.extend(wp.compare_related(cp_a, cp_b, bases[backbone], args, cache))
        scores = load_performance(args.performance) if args.performance else pd.DataFrame()
        frame = with_performance(pd.DataFrame(rows), scores)
        wp._write_shard(frame, path, signature)
    merge_if_ready(stage, a.shard, a.pieces, signature)


if __name__ == '__main__':
    main()
