import json
from pathlib import Path
import shutil
import tomllib

import pytest

import turner2018_wheelhouse as wheelhouse
from turner2018_wheelhouse import (
    load_manifest,
    smoke_install,
    verify_lock,
    verify_wheelhouse,
)


REPO = Path(__file__).resolve().parents[2]
MANIFEST = REPO / "scripts" / "turner2018_wheelhouse_manifest.json"
EXPECTED_PACKAGES = {
    "contourpy",
    "cycler",
    "fonttools",
    "h5py",
    "kiwisolver",
    "matplotlib",
    "numpy",
    "packaging",
    "pillow",
    "pyparsing",
    "python-dateutil",
    "scipy",
    "six",
}


def test_wheel_manifest_matches_uv_lock_versions_and_hashes():
    manifest = load_manifest(MANIFEST)
    lock = tomllib.loads((REPO / "uv.lock").read_text())
    locked = {
        package["name"]: package
        for package in lock["package"]
        if package["name"] in EXPECTED_PACKAGES
    }
    assert {package["name"] for package in manifest["packages"]} == EXPECTED_PACKAGES
    assert verify_lock(REPO / "uv.lock", manifest) == []
    for package in manifest["packages"]:
        entry = locked[package["name"]]
        assert package["version"] == entry["version"]
        wheel = next(
            wheel
            for wheel in entry["wheels"]
            if wheel["url"].endswith(package["filename"])
        )
        assert wheel["hash"] == f"sha256:{package['sha256']}"


def test_wheelhouse_verifier_rejects_missing_and_mutated_wheels(tmp_path):
    manifest = load_manifest(MANIFEST)
    errors = verify_wheelhouse(tmp_path, manifest)
    assert errors and all("missing" in error for error in errors)

    source = REPO / ".external" / "task6-wheelhouse-cp312-manylinux-x86_64"
    if not source.is_dir():
        pytest.skip("local wheelhouse is unavailable")
    for package in manifest["packages"]:
        shutil.copy2(source / package["filename"], tmp_path / package["filename"])
    assert verify_wheelhouse(tmp_path, manifest) == []

    target = tmp_path / manifest["packages"][0]["filename"]
    target.write_bytes(target.read_bytes() + b"mutation")
    errors = verify_wheelhouse(tmp_path, manifest)
    assert any("sha256 mismatch" in error for error in errors)


def test_wheelhouse_verifier_rejects_unexpected_compatible_wheel(tmp_path):
    manifest = load_manifest(MANIFEST)
    for package in manifest["packages"]:
        (tmp_path / package["filename"]).write_bytes(b"placeholder")
    extra = tmp_path / "numpy-2.4.5-cp312-cp312-manylinux_2_28_x86_64.whl"
    extra.write_bytes(b"unexpected")
    errors = verify_wheelhouse(tmp_path, manifest)
    assert any("unexpected wheel" in error and extra.name in error for error in errors)


def test_smoke_install_passes_exact_manifested_wheel_paths(tmp_path, monkeypatch):
    manifest = load_manifest(MANIFEST)
    source = REPO / ".external" / "task6-wheelhouse-cp312-manylinux-x86_64"
    if not source.is_dir():
        pytest.skip("local wheelhouse is unavailable")
    for package in manifest["packages"]:
        shutil.copy2(source / package["filename"], tmp_path / package["filename"])
    calls = []

    def fake_run(command, **kwargs):
        calls.append(command)
        if "-c" in command:
            return __import__("subprocess").CompletedProcess(
                command,
                0,
                stdout=json.dumps(
                    {
                        "machine": "x86_64",
                        "python": "3.12.13",
                        "versions": manifest["smoke_import"]["expected_versions"],
                    }
                ),
            )
        return __import__("subprocess").CompletedProcess(command, 0)

    monkeypatch.setattr("turner2018_wheelhouse.subprocess.run", fake_run)
    smoke_install(tmp_path, manifest, "python3.12")
    install = next(command for command in calls if "install" in command)
    expected = [str(tmp_path / package["filename"]) for package in manifest["packages"]]
    assert install[-len(expected) :] == expected
    assert not any("==" in argument for argument in install)
    smoke = next(command for command in calls if "-c" in command)
    assert "matplotlib" in smoke[-1]


def test_runtime_check_uses_manifest_python_and_exact_package_versions():
    manifest = load_manifest(MANIFEST)
    assert manifest["smoke_import"]["expected_versions"]["matplotlib"] == "3.11.1"
    assert "matplotlib" in manifest["smoke_import"]["command"]
    assert wheelhouse.check_runtime(
        manifest,
        python_version=(3, 12),
        package_versions=manifest["smoke_import"]["expected_versions"],
    ) == []

    assert wheelhouse.check_runtime(
        manifest,
        python_version=(3, 11),
        package_versions=manifest["smoke_import"]["expected_versions"],
    ) == ["Python version mismatch: expected 3.12, found 3.11"]

    wrong = dict(manifest["smoke_import"]["expected_versions"])
    wrong["numpy"] = "0.0.0"
    assert wheelhouse.check_runtime(
        manifest,
        python_version=(3, 12),
        package_versions=wrong,
    ) == ["package version mismatch: numpy expected 2.4.6, found 0.0.0"]


def test_check_runtime_executes_manifest_smoke_import(monkeypatch):
    manifest = load_manifest(MANIFEST)
    manifest["smoke_import"]["command"] = (
        "raise RuntimeError('plotting smoke import attempted')"
    )
    monkeypatch.setattr(wheelhouse, "load_manifest", lambda _path: manifest)

    with pytest.raises(RuntimeError, match="plotting smoke import attempted"):
        wheelhouse.main(["--check-runtime"])
