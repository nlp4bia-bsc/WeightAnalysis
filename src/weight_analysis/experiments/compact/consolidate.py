"""Seal complete atomic outputs into a small collection of per-study Parquet parts."""

from __future__ import annotations

import pyarrow.parquet as pq

from ...core.catalog import stable_id
from .storage import PackedStore


def consolidate_studies(root, plan, atomic, generation):
    config = plan["config"]
    with PackedStore(
        root,
        "studies",
        area="atomic",
        partition_family=True,
        policy=config["storage_policy"],
        target_bytes=config["target_part_bytes"],
    ) as store:
        store.generation(generation)
        for item in atomic:
            uid = stable_id(["consolidate", item["path"], item["sha256"]])
            if not store.done(uid):
                store.commit(uid, pq.read_table(root / item["path"]))
        store.compact(maximum_parts=8)
        return [dict(item, area="atomic", partition="study") for item in store.files(verify=True)]


def reclaim_task_parts(root, atomic):
    """Only reclaim superseded NEW-run parts, after the sealed manifest is published."""
    for item in atomic:
        path = root / item["path"]
        if path.parent.parent == root / "parquet/atomic" and path.parent.name != "task-studies":
            if path.is_file():
                path.unlink()
