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
    prepare_wheelhouse,
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
EXPECTED_CORE_VERSIONS = {
    "numpy": "2.2.6",
    "scipy": "1.15.3",
    "h5py": "3.14.0",
    "matplotlib": "3.10.9",
}
EXPECTED_COMPATIBILITY_VERSIONS = {
    **EXPECTED_CORE_VERSIONS,
    "contourpy": "1.3.2",
    "pillow": "12.2.0",
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
        assert package["url"] == wheel["url"]
        assert wheel["hash"] == f"sha256:{package['sha256']}"


def test_lock_verifier_rejects_manifest_url_not_in_lock():
    manifest = load_manifest(MANIFEST)
    manifest["packages"][0]["url"] = (
        f"https://example.invalid/{manifest['packages'][0]['filename']}"
    )

    assert verify_lock(REPO / "uv.lock", manifest) == [
        f"locked wheel mismatch: {manifest['packages'][0]['name']}"
    ]


def test_manifest_pins_dzeshell_core_versions_and_compatible_wheels():
    manifest = load_manifest(MANIFEST)
    assert manifest["platform"] == "manylinux2014_x86_64"
    assert manifest["smoke_import"]["expected_versions"] == EXPECTED_CORE_VERSIONS
    versions = {
        package["name"]: package["version"] for package in manifest["packages"]
    }
    assert {
        name: versions[name] for name in EXPECTED_COMPATIBILITY_VERSIONS
    } == EXPECTED_COMPATIBILITY_VERSIONS
    for package in manifest["packages"]:
        filename = package["filename"]
        if filename.endswith("-none-any.whl"):
            continue
        assert (
            "manylinux2014_x86_64" in filename
            or "manylinux_2_17_x86_64" in filename
        ), filename


def test_manifest_rejects_binary_wheel_above_glibc_2_17(tmp_path):
    manifest = json.loads(MANIFEST.read_text())
    package = manifest["packages"][0]
    package["filename"] = "synthetic-1.0-cp312-cp312-manylinux_2_28_x86_64.whl"
    package["url"] = f"https://example.invalid/{package['filename']}"
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest))

    with pytest.raises(RuntimeError, match="glibc 2.17"):
        load_manifest(path)


@pytest.mark.parametrize(
    "filename",
    [
        "synthetic-1.0-py2-none-any.whl",
        "synthetic-1.0-cp313-none-any.whl",
        "synthetic-1.0-cp313-cp313-manylinux_2_17_x86_64.whl",
        "synthetic-1.0-cp312-cp312-manylinux_2_17_aarch64.whl",
        "synthetic-1.0-cp312-cp312-linux_x86_64.whl",
        "synthetic-1.0-cp312-cp312-manylinux_1_999_x86_64.whl",
    ],
)
def test_manifest_rejects_wheels_incompatible_with_cpython_3_12(
    tmp_path, filename
):
    manifest = json.loads(MANIFEST.read_text())
    package = manifest["packages"][0]
    package["filename"] = filename
    package["url"] = f"https://example.invalid/{filename}"
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest))

    with pytest.raises(RuntimeError, match="CPython 3.12"):
        load_manifest(path)


@pytest.mark.parametrize(
    "filename",
    [
        "synthetic-1.0-py3-none.whl",
        "synthetic-1.0-py3-none-any.whl/../payload.whl",
        "synthetic--1.0-py3-none-any.whl",
    ],
)
def test_manifest_rejects_malformed_or_malicious_wheel_filenames(
    tmp_path, filename
):
    manifest = json.loads(MANIFEST.read_text())
    package = manifest["packages"][0]
    package["filename"] = filename
    package["url"] = f"https://example.invalid/{filename}"
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest))

    with pytest.raises(RuntimeError, match="invalid wheel filename"):
        load_manifest(path)


@pytest.mark.parametrize(
    "filename",
    [
        "synthetic-1.0-py3-none-any.whl",
        "synthetic-1.0-py2.py3-none-any.whl",
        "synthetic-1.0-cp38-abi3-manylinux_2_17_x86_64.whl",
        (
            "synthetic-1.0-cp313.cp312-abi3-"
            "manylinux_2_17_x86_64.manylinux_2_28_x86_64.whl"
        ),
    ],
)
def test_manifest_accepts_any_compatible_pep427_tag(tmp_path, filename):
    manifest = json.loads(MANIFEST.read_text())
    package = manifest["packages"][0]
    package["filename"] = filename
    package["url"] = f"https://example.invalid/{filename}"
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest))

    load_manifest(path)


def test_manifest_accepts_dual_tag_with_compatible_manylinux_floor(tmp_path):
    manifest = json.loads(MANIFEST.read_text())
    package = manifest["packages"][0]
    package["filename"] = (
        "synthetic-1.0-cp312-cp312-"
        "manylinux2014_x86_64.manylinux_2_17_x86_64.whl"
    )
    package["url"] = f"https://example.invalid/{package['filename']}"
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest))

    load_manifest(path)


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


def test_prepare_prunes_wheels_outside_exact_manifest_set(tmp_path):
    extra = tmp_path / "stale-1.0-py3-none-any.whl"
    extra.write_bytes(b"stale")

    prepare_wheelhouse(tmp_path, {"packages": []})

    assert not extra.exists()


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
    assert manifest["smoke_import"]["expected_versions"] == EXPECTED_CORE_VERSIONS
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
    ) == ["package version mismatch: numpy expected 2.2.6, found 0.0.0"]


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
