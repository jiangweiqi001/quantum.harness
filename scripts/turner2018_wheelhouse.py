#!/usr/bin/env python3
"""Prepare and verify the exact Turner ED offline scientific wheelhouse."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import tomllib
import urllib.request


def load_manifest(path: Path) -> dict:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if payload.get("schema_version") != "turner2018-wheelhouse-v1":
        raise RuntimeError("unsupported wheelhouse manifest schema")
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
                if candidate["url"].endswith(expected["filename"])
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


def prepare_wheelhouse(wheelhouse: Path, manifest: dict) -> None:
    wheelhouse.mkdir(parents=True, exist_ok=True)
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
        command = (
            "import json,platform,numpy,scipy,h5py;"
            "print(json.dumps({'python':platform.python_version(),"
            "'machine':platform.machine(),'versions':{'numpy':numpy.__version__,"
            "'scipy':scipy.__version__,'h5py':h5py.__version__}},sort_keys=True))"
        )
        result = subprocess.run(
            [str(executable), "-c", command],
            check=True,
            text=True,
            capture_output=True,
        )
    proof = json.loads(result.stdout)
    if proof["machine"] != "x86_64":
        raise RuntimeError("wheel smoke test did not run on x86_64")
    if proof["versions"] != manifest["smoke_import"]["expected_versions"]:
        raise RuntimeError("wheel smoke import versions mismatch")
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
