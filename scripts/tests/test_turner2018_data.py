import hashlib
import io
import json

import pytest

from turner2018_data import download_file, load_manifest, verify_file


def test_manifest_records_all_official_files():
    manifest = load_manifest("tracks/ed/turner-2018-data.json")
    expected = {
        "correlation_function_with_perturbations.zip": (42531, "b998e9d4b841673d0ee67f287b1db814a9926ba4864951930c6e1f473075ff39"),
        "corzz_Neel.dat": (9310, "718a3c42e1c185ae89d6db7d1220aa8acf71a8d59451fdd1ca183ec35981e9e1"),
        "eigendecomposition.zip": (44267000, "87418eb0381de777f41ea823a89d48d55c64362c15f35cbe6cdd29e9257a66a7"),
        "entanglement_entropy_with_perturbation.zip": (40675, "093a2eb042653e0f285323f9b219b5c22720c897487baed183a171ea2607e375"),
        "entanglement_entropy_growth.zip": (20675, "cddfbb2f6841cc51661462ed6150ee8c547b23085a51cd7e1ac6feabbe238b2b"),
        "energy_eigenvalues.zip": (1829474, "21fc64a37a3d41770e81b999e23682568301fdf410ec92c1162ebda91179b626"),
        "forward-scattering.zip": (10977946, "6f34540125c61ed59e978c2cfaf8e80ceaa89f5dae1870997fe6850123fcba2a"),
        "meldiag_periodic_N30_k0_p0.h5": (511424, "baf5eb9f51d72ecaa119d667ae2416665441b7b5cc543c173c0397d54951148d"),
        "overlaps_with_Neel_state.zip": (2253274, "4a179c917fe3441dda280fd2414b7c593af85baefd32e68e96cdeef265a813b2"),
        "participation_ratios.zip": (4771510, "c766038e1d5df1e173d5cff18aa77546e3ea96ee9255147d09427b04cea7301a"),
        "level_statistics.zip": (23107, "42854f72f7dfadecd9346350130eccf2f0cdf5a2df2462271709c6618b140619"),
    }

    assert manifest["doi"] == "10.5518/335"
    assert len(manifest["files"]) == 11
    assert {entry["figure"] for entry in manifest["files"]} == {
        "Fig. 2",
        "Fig. 3",
        "Fig. 4",
    }
    assert {
        entry["name"]: (entry["size"], entry["sha256"])
        for entry in manifest["files"]
    } == expected
    for index, entry in enumerate(manifest["files"], start=1):
        assert entry["url"] == (
            f"https://archive.researchdata.leeds.ac.uk/452/{index}/{entry['name']}"
        )


def test_verify_file_accepts_matching_size_and_sha256(tmp_path):
    payload = b"turner-2018"
    path = tmp_path / "sample.dat"
    path.write_bytes(payload)
    entry = {
        "size": len(payload),
        "sha256": hashlib.sha256(payload).hexdigest(),
    }

    assert verify_file(path, entry)


@pytest.mark.parametrize("change", ["size", "sha256"])
def test_verify_file_rejects_corrupt_content(tmp_path, change):
    path = tmp_path / "sample.dat"
    path.write_bytes(b"turner-2018")
    entry = {
        "size": path.stat().st_size,
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }
    if change == "size":
        entry["size"] += 1
    else:
        entry["sha256"] = "0" * 64

    assert not verify_file(path, entry)


def test_manifest_is_plain_json():
    with open("tracks/ed/turner-2018-data.json", encoding="utf-8") as handle:
        json.load(handle)


def test_download_file_is_atomic_and_removes_bad_partials(tmp_path, monkeypatch):
    payload = b"official-source-data"
    entry = {
        "name": "sample.dat",
        "url": "https://example.invalid/sample.dat",
        "size": len(payload),
        "sha256": hashlib.sha256(payload).hexdigest(),
    }
    monkeypatch.setattr(
        "turner2018_data.urllib.request.urlopen",
        lambda request, timeout: io.BytesIO(payload),
    )

    destination = download_file(entry, tmp_path)

    assert destination.read_bytes() == payload
    assert not (tmp_path / "sample.dat.part").exists()

    destination.unlink()
    entry["sha256"] = "0" * 64
    with pytest.raises(RuntimeError, match="checksum mismatch"):
        download_file(entry, tmp_path)
    assert not destination.exists()
    assert not (tmp_path / "sample.dat.part").exists()
