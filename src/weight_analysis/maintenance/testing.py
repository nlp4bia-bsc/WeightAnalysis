"""Build on login and test an installed wheel inside the actual module/container Python."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import site
import subprocess
import sys
import tempfile
import venv
from pathlib import Path


def digest(path):
    checksum = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            checksum.update(block)
    return checksum.hexdigest()


def source_digest(project_dir):
    project = Path(project_dir).resolve()
    files = [project / "pyproject.toml"] + sorted(
        p
        for p in (project / "src").rglob("*")
        if p.is_file() and (p.suffix in (".py", ".pyi") or p.name == "py.typed")
    )
    return hashlib.sha256(
        json.dumps(
            {str(path.relative_to(project)): digest(path) for path in files}, sort_keys=True
        ).encode()
    ).hexdigest()


def build_bundle(project_dir, output_dir, *, build_isolation=True):
    project = Path(project_dir).resolve(strict=True)
    output = Path(output_dir).absolute()
    output.mkdir(parents=True, exist_ok=True)
    if list(output.glob("*.whl")) or (output / "bundle.json").exists():
        raise FileExistsError("Choose a fresh test bundle directory")
    signature = source_digest(project)
    command = [sys.executable, "-m", "pip", "wheel", "--no-deps", "--wheel-dir", str(output)]
    if not build_isolation:
        command.append("--no-build-isolation")
    subprocess.run([*command, str(project)], check=True)
    wheels = list(output.glob("*.whl"))
    if len(wheels) != 1 or source_digest(project) != signature:
        raise ValueError("Build produced no unique wheel, or source changed during build")
    manifest = dict(
        project_dir=str(project),
        source_digest=signature,
        wheel=wheels[0].name,
        wheel_sha256=digest(wheels[0]),
    )
    (output / "bundle.json").write_text(json.dumps(manifest, indent=2), encoding="utf8")
    print("[TEST BUNDLE]", output, flush=True)
    return manifest


def installed_environment(bundle_dir, project_dir, environment_dir):
    """Install without dependencies/network, retaining module-provided CUDA libraries."""
    bundle, project = Path(bundle_dir).resolve(), Path(project_dir).resolve()
    manifest = json.loads((bundle / "bundle.json").read_text())
    wheel = bundle / manifest["wheel"]
    if wheel.parent != bundle or digest(wheel) != manifest["wheel_sha256"]:
        raise ValueError("Test wheel changed or escapes its bundle")
    if source_digest(project) != manifest["source_digest"]:
        raise ValueError("Checkout changed after building the test wheel; build a fresh bundle")
    directory = Path(environment_dir).absolute()
    if directory.exists():
        raise FileExistsError(directory)
    venv.EnvBuilder(system_site_packages=True, with_pip=False).create(directory)
    executable = directory / "bin" / "python"
    # A module may itself use a venv/conda environment: inherit its dependency
    # sites explicitly, but do not copy PYTHONPATH or editable source mappings.
    local_sites = list(directory.glob("lib/python*/site-packages"))
    if len(local_sites) != 1:
        raise RuntimeError("Cannot identify isolated environment site-packages")
    dependency_sites = list(
        dict.fromkeys(
            [
                *site.getsitepackages(),
                *(
                    path
                    for path in sys.path
                    if path and Path(path).name in {"site-packages", "dist-packages"}
                ),
            ]
        )
    )
    (local_sites[0] / "module_dependencies.pth").write_text(
        "\n".join(dependency_sites) + "\n", encoding="utf8"
    )
    environment = os.environ.copy()
    environment.pop("PYTHONPATH", None)
    # pip --python installs into this interpreter even though its venv has no pip.
    subprocess.run(
        [
            sys.executable,
            "-m",
            "pip",
            "--python",
            str(executable),
            "install",
            "--no-index",
            "--ignore-installed",
            "--no-deps",
            str(wheel),
        ],
        env=environment,
        check=True,
    )
    with tempfile.TemporaryDirectory(prefix="wa-installed-probe-") as working_dir:
        code = """import importlib.metadata, pathlib, sys
import weight_analysis, weightspace
root = pathlib.Path(sys.prefix).resolve()
for module in (weight_analysis, weightspace):
    path = pathlib.Path(module.__file__).resolve()
    assert path.is_relative_to(root), (path, root)
print("[INSTALLED]", sys.executable, importlib.metadata.version("weight-analysis"), flush=True)
print("[PACKAGE]", weight_analysis.__file__, flush=True)
"""
        subprocess.run([str(executable), "-c", code], cwd=working_dir, env=environment, check=True)
    return executable, environment


def run_tests(bundle_dir, project_dir, *, pytest_args=(), require_cuda=False):
    with tempfile.TemporaryDirectory(prefix="wa-tests-") as temporary:
        executable, environment = installed_environment(
            bundle_dir, project_dir, Path(temporary) / "venv"
        )
        environment["WEIGHT_ANALYSIS_TEST_BUNDLE"] = str(Path(bundle_dir).resolve())
        if require_cuda:
            subprocess.run(
                [
                    str(executable),
                    "-c",
                    'import torch; assert torch.cuda.is_available(), "CUDA unavailable in installed test environment"; print("[CUDA]", torch.__version__, torch.cuda.get_device_name(0), flush=True)',
                ],
                env=environment,
                check=True,
            )
        return subprocess.run(
            [str(executable), "-m", "pytest", *pytest_args],
            cwd=Path(project_dir).resolve(),
            env=environment,
        ).returncode


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    build = sub.add_parser("build")
    build.add_argument("--project-dir", required=True)
    build.add_argument("--output-dir", required=True)
    build.add_argument("--no-build-isolation", action="store_true")
    run = sub.add_parser("run")
    run.add_argument("--project-dir", required=True)
    run.add_argument("--bundle-dir", required=True)
    run.add_argument("--require-cuda", action="store_true")
    run.add_argument("pytest_args", nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    if args.command == "build":
        build_bundle(args.project_dir, args.output_dir, build_isolation=not args.no_build_isolation)
        return 0
    remaining = args.pytest_args[1:] if args.pytest_args[:1] == ["--"] else args.pytest_args
    return run_tests(
        args.bundle_dir,
        args.project_dir,
        pytest_args=remaining or ["-q", "-rs"],
        require_cuda=args.require_cuda,
    )


if __name__ == "__main__":
    raise SystemExit(main())
