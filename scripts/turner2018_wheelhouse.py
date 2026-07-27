#!/usr/bin/env python3
"""Prepare and verify the exact Turner ED offline scientific wheelhouse."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import importlib.metadata
import json
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import tomllib
import urllib.request

from packaging.tags import compatible_tags, cpython_tags
from packaging.utils import InvalidWheelFilename, parse_wheel_filename


MODULE_NAME = re.compile(r"[A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)*\Z")
DISTRIBUTION_NAME = re.compile(r"[A-Za-z0-9]+(?:[._-][A-Za-z0-9]+)*\Z")
TARGET_PYTHON = (3, 12)
MAX_GLIBC = (2, 17)
LEGACY_MANYLINUX_FLOORS = {
    "manylinux1_x86_64": (2, 5),
    "manylinux2010_x86_64": (2, 12),
    "manylinux2014_x86_64": (2, 17),
}
VERSIONED_MANYLINUX = re.compile(r"manylinux_(\d+)_(\d+)_x86_64\Z")
ISOLATED_SMOKE_SOURCE = """\
import importlib
import importlib.metadata
import json
import platform
import sys

spec = json.loads(sys.argv[1])
for module in spec["modules"]:
    importlib.import_module(module)
versions = {
    name: importlib.metadata.version(name)
    for name in spec["expected_versions"]
}
print(json.dumps({
    "imported_modules": spec["modules"],
    "machine": platform.machine(),
    "python": platform.python_version(),
    "versions": versions,
}, sort_keys=True))
"""


def _validate_smoke_import(manifest: dict) -> dict:
    smoke = manifest.get("smoke_import")
    if not isinstance(smoke, dict):
        raise RuntimeError("smoke_import must be an object")
    forbidden = sorted({"command", "source"}.intersection(smoke))
    if forbidden:
        raise RuntimeError(f"forbidden smoke import field: {forbidden[0]}")
    modules = smoke.get("modules")
    if not isinstance(modules, list) or not modules:
        raise RuntimeError("smoke import modules must be a non-empty list")
    for module in modules:
        if not isinstance(module, str) or MODULE_NAME.fullmatch(module) is None:
            raise RuntimeError(f"invalid smoke import module: {module!r}")
    if len(set(modules)) != len(modules):
        raise RuntimeError("duplicate smoke import module")
    module_distributions = smoke.get("module_distributions", {})
    if not isinstance(module_distributions, dict):
        raise RuntimeError("smoke import module_distributions must be an object")
    for module, distribution in module_distributions.items():
        if (
            module not in modules
            or not isinstance(distribution, str)
            or DISTRIBUTION_NAME.fullmatch(distribution) is None
        ):
            raise RuntimeError("smoke import modules must match expected_versions")
    distributions = {
        module_distributions.get(module, module) for module in modules
    }
    expected = smoke.get("expected_versions")
    if not isinstance(expected, dict) or distributions != set(expected):
        raise RuntimeError("smoke import modules must match expected_versions")
    return smoke


def _manylinux_floor(platform: str) -> tuple[int, int] | None:
    if platform in LEGACY_MANYLINUX_FLOORS:
        return LEGACY_MANYLINUX_FLOORS[platform]
    match = VERSIONED_MANYLINUX.fullmatch(platform)
    if match is None or match.group(1) != "2":
        return None
    return int(match.group(1)), int(match.group(2))


def _validate_wheel_platform(filename: str) -> None:
    if (
        not isinstance(filename, str)
        or not filename.endswith(".whl")
        or "/" in filename
        or "\\" in filename
    ):
        raise RuntimeError(f"invalid wheel filename: {filename!r}")
    try:
        _name, _version, _build, wheel_tags = parse_wheel_filename(filename)
    except InvalidWheelFilename as error:
        raise RuntimeError(f"invalid wheel filename: {filename!r}") from error

    pure_target_tags = set(
        compatible_tags(TARGET_PYTHON, interpreter="cp312", platforms=["any"])
    )
    if wheel_tags.intersection(pure_target_tags):
        return

    manylinux_platforms = {
        tag.platform
        for tag in wheel_tags
        if _manylinux_floor(tag.platform) is not None
    }
    compatible_platforms = {
        platform
        for platform in manylinux_platforms
        if _manylinux_floor(platform) <= MAX_GLIBC
    }
    target_binary_tags = (
        set(cpython_tags(TARGET_PYTHON, platforms=sorted(compatible_platforms)))
        if compatible_platforms
        else set()
    )
    if wheel_tags.intersection(target_binary_tags):
        return

    all_manylinux_target_tags = (
        set(cpython_tags(TARGET_PYTHON, platforms=sorted(manylinux_platforms)))
        if manylinux_platforms
        else set()
    )
    if wheel_tags.intersection(all_manylinux_target_tags):
        raise RuntimeError(
            f"binary wheel requires a glibc floor newer than glibc 2.17: {filename}"
        )
    raise RuntimeError(f"wheel is incompatible with CPython 3.12: {filename}")


def load_manifest(path: Path) -> dict:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if payload.get("schema_version") != "turner2018-wheelhouse-v1":
        raise RuntimeError("unsupported wheelhouse manifest schema")
    _validate_smoke_import(payload)
    packages = payload.get("packages")
    if not isinstance(packages, list):
        raise RuntimeError("packages must be a list")
    for package in packages:
        if not isinstance(package, dict):
            raise RuntimeError("package entries must be objects")
        _validate_wheel_platform(package.get("filename"))
    return payload


def _sha256(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def verify_lock(lock_path: Path, manifest: dict) -> list[str]:
    errors: list[str] = []
    if _sha256(lock_path) != manifest["lock_sha256"]:
        errors.append("uv.lock sha256 mismatch")
    lock = tomllib.loads(lock_path.read_text(encoding="utf-8"))
    packages = {entry["name"]: entry for entry in lock["package"]}
    for expected in manifest["packages"]:
        actual = packages.get(expected["name"])
        if actual is None or actual["version"] != expected["version"]:
            errors.append(f"locked version mismatch: {expected['name']}")
            continue
        wheel = next(
            (
                candidate
                for candidate in actual.get("wheels", [])
                if candidate["url"] == expected["url"]
                and candidate["url"].endswith(expected["filename"])
            ),
            None,
        )
        if wheel is None or wheel["hash"] != f"sha256:{expected['sha256']}":
            errors.append(f"locked wheel mismatch: {expected['name']}")
    return errors


def verify_wheelhouse(wheelhouse: Path, manifest: dict) -> list[str]:
    errors: list[str] = []
    expected_names = {package["filename"] for package in manifest["packages"]}
    for path in sorted(wheelhouse.glob("*.whl")):
        if path.name not in expected_names:
            errors.append(f"unexpected wheel: {path.name}")
    for package in manifest["packages"]:
        path = wheelhouse / package["filename"]
        if not path.is_file():
            errors.append(f"missing wheel: {package['filename']}")
        elif _sha256(path) != package["sha256"]:
            errors.append(f"wheel sha256 mismatch: {package['filename']}")
    return errors


def check_runtime(
    manifest: dict,
    *,
    python_version: tuple[int, int],
    package_versions: dict[str, str],
) -> list[str]:
    errors: list[str] = []
    expected_python = manifest["python"]
    actual_python = ".".join(str(value) for value in python_version)
    if actual_python != expected_python:
        errors.append(
            f"Python version mismatch: expected {expected_python}, found {actual_python}"
        )
    expected_versions = manifest["smoke_import"]["expected_versions"]
    for name, expected in expected_versions.items():
        actual = package_versions.get(name, "missing")
        if actual != expected:
            errors.append(
                f"package version mismatch: {name} expected {expected}, found {actual}"
            )
    return errors


def import_modules(modules: list[str]) -> None:
    for module in modules:
        importlib.import_module(module)


def run_isolated_smoke(python: str, manifest: dict) -> dict:
    smoke = _validate_smoke_import(manifest)
    payload = json.dumps(
        {
            "modules": smoke["modules"],
            "expected_versions": smoke["expected_versions"],
        },
        sort_keys=True,
    )
    result = subprocess.run(
        [str(python), "-c", ISOLATED_SMOKE_SOURCE, payload],
        check=True,
        text=True,
        capture_output=True,
    )
    proof = json.loads(result.stdout)
    if proof["machine"] != "x86_64":
        raise RuntimeError("wheel smoke test did not run on x86_64")
    for name, expected in smoke["expected_versions"].items():
        actual = proof["versions"].get(name, "missing")
        if actual != expected:
            raise RuntimeError(
                f"package version mismatch: {name} expected {expected}, found {actual}"
            )
    return proof


def prepare_wheelhouse(wheelhouse: Path, manifest: dict) -> None:
    wheelhouse.mkdir(parents=True, exist_ok=True)
    expected_names = {package["filename"] for package in manifest["packages"]}
    for path in wheelhouse.glob("*.whl"):
        if path.name not in expected_names:
            path.unlink()
    for package in manifest["packages"]:
        target = wheelhouse / package["filename"]
        if target.is_file() and _sha256(target) == package["sha256"]:
            continue
        partial = target.with_name(target.name + ".partial")
        urllib.request.urlretrieve(package["url"], partial)
        if _sha256(partial) != package["sha256"]:
            partial.unlink(missing_ok=True)
            raise RuntimeError(f"downloaded wheel hash mismatch: {target.name}")
        partial.replace(target)
    sums = "".join(
        f"{package['sha256']}  {package['filename']}\n"
        for package in manifest["packages"]
    )
    (wheelhouse / "SHA256SUMS").write_text(sums, encoding="utf-8")


def smoke_install(wheelhouse: Path, manifest: dict, python: str) -> dict:
    wheel_errors = verify_wheelhouse(wheelhouse, manifest)
    if wheel_errors:
        raise RuntimeError("; ".join(wheel_errors))
    with tempfile.TemporaryDirectory(prefix="turner-wheel-smoke-") as directory:
        environment = Path(directory)
        subprocess.run([python, "-m", "venv", str(environment)], check=True)
        executable = environment / "bin" / "python"
        exact_wheels = [
            str(wheelhouse / package["filename"])
            for package in manifest["packages"]
        ]
        subprocess.run(
            [
                str(executable),
                "-m",
                "pip",
                "install",
                "--no-index",
                *exact_wheels,
            ],
            check=True,
        )
        proof = run_isolated_smoke(str(executable), manifest)
    return proof


def main(argv: list[str] | None = None) -> int:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=root / "scripts" / "turner2018_wheelhouse_manifest.json",
    )
    parser.add_argument(
        "--wheelhouse",
        type=Path,
        default=root / ".external" / "task6-wheelhouse-cp312-manylinux-x86_64",
    )
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--check-runtime", action="store_true")
    parser.add_argument("--python", default=sys.executable)
    args = parser.parse_args(argv)

    manifest = load_manifest(args.manifest)
    if args.check_runtime:
        import_modules(manifest["smoke_import"]["modules"])
        versions = {}
        for name in manifest["smoke_import"]["expected_versions"]:
            try:
                versions[name] = importlib.metadata.version(name)
            except importlib.metadata.PackageNotFoundError:
                versions[name] = "missing"
        errors = check_runtime(
            manifest,
            python_version=sys.version_info[:2],
            package_versions=versions,
        )
        if errors:
            raise RuntimeError("; ".join(errors))
        print(
            json.dumps(
                {
                    "python": manifest["python"],
                    "versions": versions,
                    "verified": True,
                },
                sort_keys=True,
            )
        )
        return 0

    lock_errors = verify_lock(root / "uv.lock", manifest)
    if lock_errors:
        raise RuntimeError("; ".join(lock_errors))
    if args.prepare:
        prepare_wheelhouse(args.wheelhouse, manifest)
    wheel_errors = verify_wheelhouse(args.wheelhouse, manifest)
    if wheel_errors:
        raise RuntimeError("; ".join(wheel_errors))
    proof = {"verified": True}
    if args.smoke:
        proof["smoke_import"] = smoke_install(
            args.wheelhouse,
            manifest,
            args.python,
        )
    print(json.dumps(proof, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
