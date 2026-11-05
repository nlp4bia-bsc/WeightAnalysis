"""Read the frozen picasso-1 contract without loading PyTorch or pandas.

This adapter accepts JSON-native values only. Its canonical JSON encoding matches
weight_analysis.core.catalog.stable_id for a persisted, finite-valued manifest.
Keep it usable with the login host's Python 3.6+, independently of compute Python.
"""

import hashlib
import json
import os
from pathlib import Path
from uuid import uuid4


def stable_id(value):
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(encoded.encode()).hexdigest()


def code_digest():
    root = Path(__file__).resolve().parents[1] / "weight_analysis"
    return stable_id(
        {
            str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted(root.rglob("*.py"))
        }
    )


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(".{}.{}.partial".format(path.name, uuid4().hex))
    try:
        temporary.write_text(
            json.dumps(value, sort_keys=True, indent=2, ensure_ascii=False), encoding="utf8"
        )
        os.replace(str(temporary), str(path))
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def load_plan(run_dir):
    path = Path(run_dir) / "plan.json"
    if not path.is_file():
        raise ValueError("Frozen plan unavailable at {}; finish prepare first".format(path))
    plan = json.loads(path.read_text(encoding="utf8"))
    identity = {key: value for key, value in plan.items() if key != "run_id"}
    if plan.get("schema_version") != "picasso-1" or stable_id(identity) != plan.get("run_id"):
        raise ValueError("Run manifest changed or unsupported version")
    if plan["code_digest"] != code_digest():
        raise ValueError(
            "Scientific code differs from frozen plan; restore code or prepare a new run"
        )
    return plan
