import importlib.metadata
import json
from pathlib import Path
import shutil
import sys
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


def test_manifest_smoke_import_is_declarative_and_version_complete():
    smoke = load_manifest(MANIFEST)["smoke_import"]
    assert smoke["modules"] == ["numpy", "scipy", "h5py", "matplotlib"]
    assert set(smoke["modules"]) == set(smoke["expected_versions"])
    assert "command" not in smoke
    assert "source" not in smoke


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("command", "import os", "forbidden smoke import field"),
        ("source", "import os", "forbidden smoke import field"),
        ("modules", ["matplotlib; import os"], "invalid smoke import module"),
        ("modules", ["matplotlib.__dict__['x']"], "invalid smoke import module"),
        ("modules", ["matplotlib..pyplot"], "invalid smoke import module"),
        ("modules", ["matplotlib", "matplotlib"], "duplicate smoke import module"),
        ("modules", ["matplotlib.pyplot"], "must match expected_versions"),
    ],
)
def test_manifest_rejects_source_fields_and_invalid_modules(
    tmp_path, field, value, message
):
    manifest = json.loads(MANIFEST.read_text())
    manifest["smoke_import"].pop("command", None)
    manifest["smoke_import"]["modules"] = [
        "numpy",
        "scipy",
        "h5py",
        "matplotlib",
    ]
    manifest["smoke_import"][field] = value
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest))

    with pytest.raises(RuntimeError, match=message):
        load_manifest(path)


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


def test_smoke_install_uses_declarative_command_without_downloaded_wheels(
    tmp_path, monkeypatch
):
    manifest = load_manifest(MANIFEST)
    manifest["packages"] = []
    calls = []

    def fake_run(command, **kwargs):
        calls.append(command)
        if "-c" in command:
            return __import__("subprocess").CompletedProcess(
                command,
                0,
                stdout=json.dumps(
                    {
                        "imported_modules": manifest["smoke_import"]["modules"],
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
    assert install[1:] == ["-m", "pip", "install", "--no-index"]
    assert not any("==" in argument for argument in install)
    smoke = next(command for command in calls if "-c" in command)
    assert smoke[2] == wheelhouse.ISOLATED_SMOKE_SOURCE
    assert json.loads(smoke[3])["modules"][-1] == "matplotlib"


def test_generated_isolated_smoke_imports_matplotlib_and_checks_exact_versions():
    modules = ["numpy", "scipy", "h5py", "matplotlib"]
    expected = {name: importlib.metadata.version(name) for name in modules}
    manifest = {
        "smoke_import": {
            "modules": modules,
            "expected_versions": expected,
        }
    }

    proof = wheelhouse.run_isolated_smoke(sys.executable, manifest)
    assert proof["imported_modules"] == modules
    assert proof["versions"] == expected

    manifest["smoke_import"]["expected_versions"]["matplotlib"] = "0.0.0"
    with pytest.raises(RuntimeError, match="matplotlib expected 0.0.0"):
        wheelhouse.run_isolated_smoke(sys.executable, manifest)


def test_runtime_check_uses_manifest_python_and_exact_package_versions():
    manifest = load_manifest(MANIFEST)
    assert manifest["smoke_import"]["expected_versions"]["matplotlib"] == "3.11.1"
    assert "matplotlib" in manifest["smoke_import"]["modules"]
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


def test_check_runtime_imports_declarative_modules(monkeypatch):
    manifest = load_manifest(MANIFEST)
    imported = []
    monkeypatch.setattr(wheelhouse, "load_manifest", lambda _path: manifest)
    monkeypatch.setattr(
        wheelhouse,
        "import_modules",
        lambda modules: imported.extend(modules),
    )
    monkeypatch.setattr(
        wheelhouse.importlib.metadata,
        "version",
        lambda name: manifest["smoke_import"]["expected_versions"][name],
    )

    assert wheelhouse.main(["--check-runtime"]) == 0
    assert imported == manifest["smoke_import"]["modules"]
