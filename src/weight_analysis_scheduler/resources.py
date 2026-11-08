"""Read-only filesystem/quota checks shared by host submission and compute jobs."""

import errno
import json
import os
import shutil
import subprocess
import time
from pathlib import Path

_QUOTA_CACHE = {}
_USAGE_CACHE = {}


class StoragePressure(OSError):
    """A resumable resource stop, never a failed mathematical observation."""


def is_storage_error(exc):
    return (
        (isinstance(exc, OSError) and exc.errno in (errno.EDQUOT, errno.ENOSPC))
        or (getattr(exc, "sqlite_errorcode", 0) & 255 in (10, 13))
        or "database or disk is full" in str(exc).lower()
    )


def parse_quota(text):
    """Parse plain numeric Linux quota/Lustre output; refuse unknown formats."""
    lines = text.splitlines()
    if not any(
        ("blocks" in x.lower() or "kbytes" in x.lower()) and "files" in x.lower() for x in lines
    ):
        raise ValueError("Unrecognized quota header")
    candidates = []
    for line in lines:
        tokens = line.split()
        if tokens and not tokens[0].rstrip("*").isdigit():
            tokens = tokens[1:]
        if len(tokens) not in (6, 8):
            continue
        positions = (0, 1, 2, 3, 4, 5) if len(tokens) == 6 else (0, 1, 2, 4, 5, 6)
        try:
            used, soft, hard, files, fsoft, fhard = [int(tokens[i].rstrip("*")) for i in positions]
        except ValueError:
            continue
        limits = [n for n in (soft, hard) if n > 0]
        flimits = [n for n in (fsoft, fhard) if n > 0]
        if limits and flimits:
            candidates.append(
                dict(
                    remaining_bytes=max(0, min(limits) - used) * 1024,
                    remaining_inodes=max(0, min(flimits) - files),
                )
            )
    if len(candidates) != 1:
        raise ValueError("Quota is ambiguous or has no finite byte/inode limits")
    return candidates[0]


def quota_snapshot(root):
    root = Path(root).resolve()
    for name, command in (
        ("lfs", ["lfs", "quota", "-u", str(os.getuid()), str(root)]),
        ("quota", ["quota", "-w", "-f", str(root)]),
    ):
        if shutil.which(name):
            try:
                result = subprocess.run(
                    command,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    universal_newlines=True,
                    timeout=20,
                    check=True,
                )
                return dict(
                    parse_quota(result.stdout), observed_at=time.time(), path=str(root), source=name
                )
            except (OSError, ValueError, subprocess.SubprocessError):
                continue
    return None


def usage(root):
    files, size = 0, 0
    if Path(root).exists():
        for directory, folders, names in os.walk(str(root)):
            files += len(folders) + len(names)
            for name in names:
                try:
                    size += os.stat(os.path.join(directory, name)).st_size
                except FileNotFoundError:
                    pass
    return dict(files=files, bytes=size)


def check_storage(root, policy, *, required_bytes=0, required_inodes=0, launch=False):
    root = Path(root).expanduser().resolve()
    existing = root
    while not existing.exists():
        existing = existing.parent
    stat = os.statvfs(str(existing))
    available = dict(remaining_bytes=stat.f_bavail * stat.f_frsize, remaining_inodes=stat.f_favail)
    refresh = policy.get("quota_refresh_seconds", 30)
    previous = _QUOTA_CACHE.get(str(existing))
    if previous is None or time.monotonic() - previous[0] >= refresh or launch:
        previous = (time.monotonic(), quota_snapshot(existing))
        _QUOTA_CACHE[str(existing)] = previous
    quota = previous[1]

    def current_usage():
        cached = _USAGE_CACHE.get(str(root))
        if cached is None or time.monotonic() - cached[0] >= refresh or launch:
            cached = (time.monotonic(), usage(root))
            _USAGE_CACHE[str(root)] = cached
        return cached[1]

    snapshot = policy.get("quota_file")
    if quota is None and snapshot:
        quota = json.loads(Path(snapshot).read_text())
        stamp = quota.get("observed_at", 0)
        if not 0 <= time.time() - stamp <= policy.get("quota_max_age_seconds", 86400):
            raise StoragePressure(
                errno.EDQUOT, "Quota snapshot expired; refresh real quota figures"
            )
        scope = Path(quota["path"]).resolve()
        if scope not in (root, *root.parents):
            raise StoragePressure(errno.EDQUOT, "Quota snapshot belongs to another filesystem path")
        # A declared snapshot is conservative: deduct all current run usage.
        consumed = current_usage()
        quota = dict(
            quota,
            remaining_bytes=quota["remaining_bytes"] - consumed["bytes"],
            remaining_inodes=quota["remaining_inodes"] - consumed["files"],
        )
    if quota is None and policy.get("require_quota", False):
        raise StoragePressure(
            errno.EDQUOT,
            "User quota unavailable; supply a current quota JSON, "
            "or explicitly allow unknown quota after checking it on Picasso",
        )
    if quota:
        for key in available:
            available[key] = min(available[key], quota[key])
    floor_bytes = policy.get("min_free_bytes", 2 * 1024**3)
    floor_inodes = policy.get("min_free_inodes", 1024)
    if launch:
        required_bytes = max(required_bytes, policy.get("reserve_bytes", 0))
        required_inodes = max(required_inodes, policy.get("reserve_inodes", 0))
    if (
        available["remaining_bytes"] < floor_bytes + required_bytes
        or available["remaining_inodes"] < floor_inodes + required_inodes
    ):
        raise StoragePressure(errno.EDQUOT, "Insufficient space/inodes: {}".format(available))
    if (
        policy.get("max_run_files")
        and current_usage()["files"] + required_inodes > policy["max_run_files"]
    ):
        raise StoragePressure(errno.EDQUOT, "Run file budget exhausted; compact before continuing")
    return dict(available, quota_status="verified" if quota else "explicitly_unknown")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True)
    parser.add_argument("--quota-file")
    parser.add_argument("--allow-unknown-quota", action="store_true")
    parser.add_argument("--reserve-gib", type=float, default=8)
    parser.add_argument("--reserve-inodes", type=int, default=4000)
    parser.add_argument("--max-run-files", type=int, default=8000)
    parser.add_argument("--snapshot", action="store_true")
    parser.add_argument("--output")
    parser.add_argument("--remaining-gib", type=float)
    parser.add_argument("--remaining-inodes", type=int)
    args = parser.parse_args()
    root = Path(args.root).expanduser().resolve()
    existing = root
    while not existing.exists():
        existing = existing.parent
    try:
        if args.snapshot:
            if not args.output:
                parser.error("--snapshot requires --output")
            if args.remaining_gib is not None and args.remaining_inodes is not None:
                snapshot = dict(
                    remaining_bytes=int(args.remaining_gib * 1024**3),
                    remaining_inodes=args.remaining_inodes,
                    path=str(existing),
                    observed_at=time.time(),
                    source="operator_real_quota_reading",
                )
            else:
                snapshot = quota_snapshot(existing)
            if snapshot is None:
                parser.error(
                    "No supported quota tool. Supply real remaining GiB AND inodes; do not guess."
                )
            path = Path(args.output).resolve()
            path.parent.mkdir(parents=True, exist_ok=True)
            temporary = path.with_suffix(".partial")
            temporary.write_text(json.dumps(snapshot, indent=2))
            temporary.replace(path)
            print("[QUOTA]", snapshot, flush=True)
        else:
            policy = dict(
                require_quota=not args.allow_unknown_quota,
                quota_file=args.quota_file,
                reserve_bytes=int(args.reserve_gib * 1024**3),
                reserve_inodes=args.reserve_inodes,
                max_run_files=args.max_run_files,
            )
            print("[PREFLIGHT]", check_storage(root, policy, launch=True), flush=True)
    except (OSError, ValueError) as exc:
        parser.exit(1, "[STORAGE STOP] {}\n".format(exc))
