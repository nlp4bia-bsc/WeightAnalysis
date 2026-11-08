"""One transactional journal per task, bounded Parquet parts and atomic compaction."""

from __future__ import annotations

import hashlib
import json
import os
import sqlite3
from pathlib import Path

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.dataset as ds
import pyarrow.parquet as pq

from weight_analysis_scheduler.resources import check_storage

from ...core.catalog import stable_id
from ...provenance import file_sha256

SCHEMA = pa.schema(
    [
        (name, pa.string())
        for name in (
            "record_id",
            "unit_id",
            "kind",
            "family",
            "backbone",
            "artifact_a",
            "artifact_b",
            "group_id",
            "language",
            "entity",
            "language_a",
            "language_b",
            "entity_a",
            "entity_b",
            "reference_id",
            "reference_kind",
            "source",
            "scope",
            "component",
            "embedding_type",
            "tensor",
            "metric",
            "status",
            "exactness",
            "execution_id",
        )
    ]
    + [(name, pa.int64()) for name in ("layer", "fold_a", "fold_b", "seed_a", "seed_b", "n")]
    + [("value", pa.float64()), ("relative_depth", pa.float64()), ("payload", pa.large_string())]
)


def record(metadata, *, kind, metric, value=None, payload=None, **extra):
    import math

    row = dict(metadata, kind=kind, metric=metric, **extra)
    row["record_id"] = stable_id(row)
    row["value"] = float(value) if value is not None and math.isfinite(float(value)) else None
    row.setdefault(
        "status", "ok" if row["value"] is not None or payload is not None else "undefined"
    )
    row["payload"] = (
        json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)
        if payload is not None
        else None
    )
    return {name: row.get(name) for name in SCHEMA.names}


def _encode(rows):
    table = rows if isinstance(rows, pa.Table) else pa.Table.from_pylist(rows, schema=SCHEMA)
    sink = pa.BufferOutputStream()
    with pa.ipc.new_stream(
        sink, SCHEMA, options=pa.ipc.IpcWriteOptions(compression="zstd")
    ) as writer:
        writer.write_table(table)
    return sink.getvalue().to_pybytes()


def _decode(body):
    return pa.ipc.open_stream(body).read_all()


class PackedStore:
    """Single-writer store. SQLite commits are the authority for visible fragments.

    Unit payloads are temporary Arrow IPC in SQLite; after a Parquet commit they
    are released. No tensor gets a directory, sidecar or individual Parquet file.
    """

    def __init__(
        self,
        root,
        task_id,
        *,
        area="atomic",
        policy=None,
        target_bytes=64 * 1024**2,
        partition_family=False,
    ):
        self.root, self.task_id = Path(root), str(task_id)
        self.policy, self.target_bytes = policy or {}, target_bytes
        self.partition_family = partition_family
        state = self.root / "state"
        state.mkdir(parents=True, exist_ok=True)
        self.directory = self.root / "parquet" / area / ("task-" + self.task_id)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(state / (self.task_id + ".sqlite"), timeout=30)
        self.db.execute("PRAGMA journal_mode=DELETE")
        self.db.execute("PRAGMA synchronous=FULL")
        self.db.executescript("""
            CREATE TABLE IF NOT EXISTS units (
                id TEXT PRIMARY KEY, body BLOB, digest TEXT NOT NULL, rows INTEGER NOT NULL);
            CREATE TABLE IF NOT EXISTS fragments (
                path TEXT PRIMARY KEY, digest TEXT NOT NULL, rows INTEGER NOT NULL);
            CREATE TABLE IF NOT EXISTS progress (id TEXT PRIMARY KEY, body TEXT NOT NULL);
        """)

    def done(self, unit_id):
        return self.db.execute("SELECT 1 FROM units WHERE id=?", (unit_id,)).fetchone() is not None

    def progress(self, key, value=None):
        if value is None:
            row = self.db.execute("SELECT body FROM progress WHERE id=?", (key,)).fetchone()
            return json.loads(row[0]) if row else None
        body = json.dumps(value, sort_keys=True, allow_nan=False)
        with self.db:
            self.db.execute("INSERT OR REPLACE INTO progress VALUES (?,?)", (key, body))

    def clear_progress(self, key):
        with self.db:
            self.db.execute("DELETE FROM progress WHERE id=?", (key,))

    def generation(self, identity):
        current = self.progress("_generation")
        if current == {"identity": identity}:
            return
        obsolete = self.progress("_obsolete") or {"paths": []}
        obsolete["paths"] += [item["path"] for item in self.files()]
        obsolete["paths"] += (self.progress("_compaction_gc") or {}).get("paths", [])
        with self.db:
            self.db.execute("DELETE FROM units")
            self.db.execute("DELETE FROM fragments")
            self.db.execute("DELETE FROM progress")
            self.db.executemany(
                "INSERT INTO progress VALUES (?,?)",
                [
                    ("_generation", json.dumps(dict(identity=identity))),
                    ("_obsolete", json.dumps(obsolete)),
                ],
            )

    def reclaim_obsolete(self):
        """Call only AFTER publishing the replacement run dataset manifest."""
        active = {item["path"] for item in self.files()}
        for name in (self.progress("_obsolete") or {}).get("paths", []):
            path = self.root / name
            if name not in active and path.parent == self.directory and path.is_file():
                path.unlink()
        self.clear_progress("_obsolete")

    def commit(self, unit_id, rows):
        body = _encode(rows)
        digest = hashlib.sha256(body).hexdigest()
        old = self.db.execute("SELECT digest FROM units WHERE id=?", (unit_id,)).fetchone()
        if old:
            if old[0] != digest:
                raise ValueError("Conflicting results for an already committed unit")
            return
        check_storage(self.root, self.policy, required_bytes=len(body) * 2, required_inodes=2)
        with self.db:
            self.db.execute(
                "INSERT INTO units VALUES (?,?,?,?)", (unit_id, body, digest, len(rows))
            )
            self.db.execute(
                "DELETE FROM progress WHERE id=? OR id LIKE ?", (unit_id, unit_id + ":%")
            )
        pending = self.db.execute("SELECT COALESCE(SUM(length(body)),0) FROM units").fetchone()[0]
        if pending >= self.target_bytes // 4:
            self.flush()

    def read_unit(self, unit_id):
        stored = self.db.execute("SELECT body FROM units WHERE id=?", (unit_id,)).fetchone()
        if stored is None:
            raise KeyError(unit_id)
        if stored[0] is not None:
            return _decode(stored[0]).to_pylist()
        files = [str(self.root / item["path"]) for item in self.files()]
        return (
            ds.dataset(files, schema=SCHEMA)
            .to_table(filter=ds.field("unit_id") == unit_id)
            .to_pylist()
        )

    def _write(self, table, identity):
        family = (
            table.column("family")[0].as_py() if self.partition_family and table.num_rows else None
        )

        prefix = (family or "metadata") + "--" if self.partition_family else ""
        if prefix and ("/" in prefix or "\\" in prefix):
            raise ValueError("Unsafe study name")
        path = self.directory / (prefix + "part-" + identity[:24] + ".parquet")
        check_storage(self.root, self.policy, required_bytes=table.nbytes * 2, required_inodes=2)
        temporary = path.with_suffix(".partial")
        pq.write_table(
            table,
            temporary,
            compression="zstd",
            compression_level=6,
            row_group_size=32768,
            write_statistics=True,
        )
        with temporary.open("rb") as stream:
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        descriptor = os.open(self.directory, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        return str(path.relative_to(self.root)), file_sha256(path), table.num_rows

    def flush(self):
        pending = self.db.execute(
            "SELECT id,body,digest FROM units WHERE body IS NOT NULL ORDER BY id"
        ).fetchall()
        if not pending:
            return
        if any(hashlib.sha256(row[1]).hexdigest() != row[2] for row in pending):
            raise ValueError("Journal IPC checksum mismatch")
        table = pa.concat_tables([_decode(row[1]) for row in pending])
        if len(set(table.column("record_id").to_pylist())) != table.num_rows:
            raise ValueError("Duplicate observation identities within packed output")
        count = max(1, int(self.target_bytes * table.num_rows / max(1, table.nbytes)))
        parts = []
        identity = stable_id([(row[0], row[2]) for row in pending])
        families = (
            sorted(set(table.column("family").to_pylist()), key=str)
            if self.partition_family
            else [None]
        )
        for family in families:
            part = (
                table.filter(
                    pc.is_null(table["family"])
                    if family is None
                    else pc.equal(table["family"], family)
                )
                if self.partition_family
                else table
            )
            for start in range(0, part.num_rows, count):
                parts.append(
                    self._write(part.slice(start, count), stable_id([identity, family, start]))
                )
        with self.db:
            self.db.executemany("INSERT OR REPLACE INTO fragments VALUES (?,?,?)", parts)
            self.db.executemany(
                "UPDATE units SET body=NULL WHERE id=?", [(row[0],) for row in pending]
            )

    def files(self, *, verify=False):
        rows = self.db.execute("SELECT path,digest,rows FROM fragments ORDER BY path").fetchall()
        if verify:
            for name, digest, count in rows:
                path = self.root / name
                if file_sha256(path) != digest or pq.ParquetFile(path).metadata.num_rows != count:
                    raise ValueError("Packed Parquet integrity failure: " + str(path))
        return [dict(path=name, sha256=digest, rows=count) for name, digest, count in rows]

    def compact(self, *, maximum_parts=4):
        self.flush()
        old = self.files(verify=True)
        self._reclaim_compaction()
        if len(old) <= maximum_parts:
            return old
        identity = stable_id(old)
        parts, batches, size, number = [], [], 0, 0

        def emit():
            nonlocal batches, size, number
            table = pa.Table.from_batches(batches, schema=SCHEMA)
            parts.append(self._write(table, stable_id(["compact", identity, number])))
            batches, size, number = [], 0, number + 1

        current_family = None
        for item in sorted(old, key=lambda item: item["path"]):
            family = Path(item["path"]).name.split("--part-")[0] if self.partition_family else None
            if (
                self.partition_family
                and current_family is not None
                and family != current_family
                and batches
            ):
                emit()
            current_family = family
            for batch in pq.ParquetFile(self.root / item["path"]).iter_batches(batch_size=8192):
                batches.append(batch)
                size += batch.nbytes
                if size >= self.target_bytes:
                    emit()
        if batches:
            emit()
        with self.db:
            self.db.execute("DELETE FROM fragments")
            self.db.executemany("INSERT INTO fragments VALUES (?,?,?)", parts)
            self.db.execute(
                "INSERT OR REPLACE INTO progress VALUES (?,?)",
                ("_compaction_gc", json.dumps(dict(paths=[item["path"] for item in old]))),
            )
        self._reclaim_compaction()
        return self.files(verify=True)

    def _reclaim_compaction(self):
        active = {item["path"] for item in self.files()}
        for name in (self.progress("_compaction_gc") or {}).get("paths", []):
            path = self.root / name
            if name not in active and path.parent == self.directory and path.exists():
                path.unlink()
        self.clear_progress("_compaction_gc")

    def close(self):
        self.db.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
