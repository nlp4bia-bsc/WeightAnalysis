"""Installation contracts and one-way imports of the independent library."""

from __future__ import annotations

import ast
import json
import os
import re
import subprocess
import sys
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / "src" / "weight_analysis"
FORBIDDEN_IMPORT_ROOTS = {"weightspace", "ner_lab", "lab"}
SCHEDULING_MODULES = {"slurm", "slurm_guard", "scheduler", "scheduling"}
SCHEDULING_COMMANDS = {"sbatch", "salloc", "srun", "scancel", "squeue"}


def test_canonical_package_has_no_legacy_or_reference_imports():
    violations = []
    for path in PACKAGE.rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            modules = []
            if isinstance(node, ast.Import):
                modules = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom) and node.module:
                modules = [node.module]
            for module in modules:
                components = module.split(".")
                if components[0] in FORBIDDEN_IMPORT_ROOTS or set(components) & SCHEDULING_MODULES:
                    violations.append(f"{path.relative_to(ROOT)}:{node.lineno}: {module}")

    assert not violations, "Imports must point into the independent package: " + "; ".join(
        violations
    )


def test_library_does_not_launch_cluster_jobs_or_bundle_submission_scripts():
    violations = []
    for path in PACKAGE.rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call) or not node.args:
                continue
            function = node.func
            if not isinstance(function, ast.Attribute) or function.attr not in {
                "run",
                "Popen",
                "call",
                "check_call",
                "check_output",
                "system",
                "popen",
            }:
                continue
            first_argument = node.args[0]
            if isinstance(first_argument, ast.Constant) and isinstance(first_argument.value, str):
                command = first_argument.value.split(maxsplit=1)[0]
            elif isinstance(first_argument, (ast.List, ast.Tuple)) and first_argument.elts:
                first_item = first_argument.elts[0]
                command = first_item.value if isinstance(first_item, ast.Constant) else None
            else:
                command = None
            if isinstance(command, str) and Path(command).name in SCHEDULING_COMMANDS:
                violations.append(f"{path.relative_to(ROOT)}:{node.lineno}: {command}")
    scripts = [
        path.relative_to(ROOT)
        for directory in (ROOT / "src", ROOT / "examples")
        if directory.exists()
        for path in directory.rglob("*")
        if path.is_file() and path.suffix.lower() in {".sh", ".sbatch"}
    ]

    assert not violations, "Library must not submit jobs: " + "; ".join(violations)
    assert not scripts, f"Source/examples must contain no shell or SLURM scripts: {scripts}"


def test_distribution_installs_the_independent_typed_package_without_reference_dependencies():
    with (ROOT / "pyproject.toml").open("rb") as stream:
        configuration = tomllib.load(stream)
    project = configuration["project"]
    requirements = list(project.get("dependencies", []))
    for extra in project.get("optional-dependencies", {}).values():
        requirements.extend(extra)
    for requirement in requirements:
        name = re.split(r"[\s\[<>=!~;@]", requirement, maxsplit=1)[0].lower().replace("_", "-")
        assert name not in {"lab", "ner-lab"}, f"Mandatory reference dependency: {requirement}"
        assert "github.com/nlp4bia-bsc/ner_lab" not in requirement.lower()
    assert configuration["build-system"]["build-backend"] == "hatchling.build"
    assert (
        "src/weight_analysis"
        in configuration["tool"]["hatch"]["build"]["targets"]["wheel"]["packages"]
    )
    assert (PACKAGE / "py.typed").is_file()


def test_importing_package_does_not_load_compute_or_plotting_dependencies(tmp_path):
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(ROOT / "src")
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    code = """
import json
import sys
import weight_analysis
assert callable(weight_analysis.__getattr__)
assert 'analyze' in weight_analysis.__all__
assert 'CheckpointSpec' in weight_analysis.__all__
for module in ('torch', 'numpy', 'pandas', 'scipy', 'pyarrow', 'matplotlib', 'lab', 'ner_lab', 'weightspace'):
    assert module not in sys.modules, module
try:
    getattr(weight_analysis, 'nonexistent_public_operation')
except AttributeError:
    pass
else:
    raise AssertionError('Unknown exports must raise AttributeError')
print(json.dumps({'version': weight_analysis.__version__, 'exports': weight_analysis.__all__}))
"""
    completed = subprocess.run(
        [sys.executable, "-c", code],
        cwd=tmp_path,
        env=environment,
        capture_output=True,
        text=True,
        check=True,
    )
    imported = json.loads(completed.stdout)

    assert imported["version"]
    assert len(imported["exports"]) == len(set(imported["exports"]))


def test_advertised_public_exports_resolve_without_the_reference_repository():
    import weight_analysis

    for name in weight_analysis.__all__:
        assert getattr(weight_analysis, name) is not None, name
