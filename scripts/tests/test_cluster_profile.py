"""Tests for scripts/cluster_profile.py (TOML profile parsing/validation)."""

import pytest

import cluster_profile as cp

FULL_PROFILE = """\
[identity]
name = "testhpc"

[connection]
repo_path_remote = "/home/u/harness"

[connection.ssh]
alias = "testhpc"

[scheduler]
type = "slurm"

[limits.hard]
max_walltime = "24:00:00"
max_nodes = 4

[limits.soft]
warn_cpus = 64
unusual_partitions = ["gpu-large"]

[limits.paths]
allowed_roots = ["~/scratch", "~/results"]
"""


def _write(tmp_path, text, name="active.toml"):
    p = tmp_path / name
    p.write_text(text, encoding="utf-8")
    return p


# --------------------------------------------------------------------------- #
# path resolution
# --------------------------------------------------------------------------- #
def test_resolve_explicit_wins(monkeypatch):
    monkeypatch.setenv("HARNESS_PROFILE_FILE", "/env/file.toml")
    assert cp.resolve_profile_path("/explicit.toml") == cp.Path("/explicit.toml")


def test_resolve_env_file(monkeypatch):
    monkeypatch.setenv("HARNESS_PROFILE_FILE", "/env/file.toml")
    assert cp.resolve_profile_path() == cp.Path("/env/file.toml")


def test_resolve_env_named(monkeypatch):
    monkeypatch.delenv("HARNESS_PROFILE_FILE", raising=False)
    monkeypatch.setenv("HARNESS_CLUSTER_PROFILE", "hpc2")
    assert cp.resolve_profile_path().name == "hpc2.toml"


def test_resolve_default(monkeypatch):
    monkeypatch.delenv("HARNESS_PROFILE_FILE", raising=False)
    monkeypatch.delenv("HARNESS_CLUSTER_PROFILE", raising=False)
    assert str(cp.resolve_profile_path()).endswith("active.toml")


# --------------------------------------------------------------------------- #
# load / validate
# --------------------------------------------------------------------------- #
def test_load_valid(tmp_path):
    p = _write(tmp_path, FULL_PROFILE)
    prof = cp.load_profile(p)
    assert prof["identity"]["name"] == "testhpc"


def test_load_missing_raises(tmp_path):
    with pytest.raises(cp.ProfileError, match="not found"):
        cp.load_profile(tmp_path / "nope.toml")


def test_load_malformed_raises(tmp_path):
    p = _write(tmp_path, "this is = = not toml [[[")
    with pytest.raises(cp.ProfileError, match="malformed"):
        cp.load_profile(p)


def test_validate_complete(tmp_path):
    prof = cp.load_profile(_write(tmp_path, FULL_PROFILE))
    assert cp.validate(prof) == []


def test_validate_missing_sections():
    assert any("connection" in w for w in cp.validate({"identity": {}}))


# --------------------------------------------------------------------------- #
# accessors
# --------------------------------------------------------------------------- #
def test_get_field_nested(tmp_path):
    prof = cp.load_profile(_write(tmp_path, FULL_PROFILE))
    assert cp.get_field(prof, "scheduler.type") == "slurm"
    # the fields harness_slurm.sh shells out for
    assert cp.get_field(prof, "connection.ssh.alias") == "testhpc"
    assert cp.get_field(prof, "connection.repo_path_remote") == "/home/u/harness"


def test_get_field_missing(tmp_path):
    prof = cp.load_profile(_write(tmp_path, FULL_PROFILE))
    assert cp.get_field(prof, "scheduler.nope") is None
    assert cp.get_field(prof, "absent.deep.path") is None


def test_get_field_through_nontable(tmp_path):
    prof = cp.load_profile(_write(tmp_path, FULL_PROFILE))
    # scheduler.type is a string; descending further must yield None
    assert cp.get_field(prof, "scheduler.type.more") is None


def test_get_field_returns_list(tmp_path):
    prof = cp.load_profile(_write(tmp_path, FULL_PROFILE))
    assert cp.get_field(prof, "limits.soft.unusual_partitions") == ["gpu-large"]


def test_get_limits_full(tmp_path):
    prof = cp.load_profile(_write(tmp_path, FULL_PROFILE))
    lim = cp.get_limits(prof)
    assert lim.hard["max_nodes"] == 4
    assert lim.soft["warn_cpus"] == 64
    assert lim.allowed_roots == ["~/scratch", "~/results"]
    assert lim.configured is True


def test_get_limits_absent():
    lim = cp.get_limits({"identity": {}})
    assert lim.hard == {} and lim.soft == {} and lim.allowed_roots == []
    assert lim.configured is False


def test_get_limits_malformed_types():
    prof = {"limits": "not-a-table"}
    assert cp.get_limits(prof).configured is False
    prof2 = {"limits": {"hard": "x", "soft": 1, "paths": {"allowed_roots": "y"}}}
    lim = cp.get_limits(prof2)
    assert lim.hard == {} and lim.soft == {} and lim.allowed_roots == []


def test_get_partition_by_name():
    profile = {
        "partitions": [
            {"name": "cpu"},
            {"name": "gpu", "required_gres": "gpu:a100:1"},
        ]
    }

    assert cp.get_partition(profile, "gpu") == {
        "name": "gpu",
        "required_gres": "gpu:a100:1",
    }
    assert cp.get_partition(profile, "missing") is None


def test_public_qdeshell_profile_is_safe_and_complete():
    path = cp.Path(__file__).resolve().parents[2] / (
        "skills/using-slurm/profiles/qdeshell.toml"
    )
    mirror_path = cp.Path(__file__).resolve().parents[2] / (
        ".agents/skills/using-slurm/profiles/qdeshell.toml"
    )
    profile = cp.load_profile(path)

    assert cp.validate(profile) == []
    assert mirror_path.read_bytes() == path.read_bytes()
    assert profile["connection"]["repo_path_remote"] == (
        "/work/share/giggleliu/jiangweiqi/quantum.harness"
    )
    assert profile["connection"]["ssh"] == {"alias": "qdeshell"}
    assert profile["scheduler"] == {
        "type": "slurm",
        "default_partition": "dzagnormal",
    }

    partition = profile["partitions"][0]
    assert partition["name"] == "dzagnormal"
    assert partition["required_gres"] == "gpu:NVIDIAA80080GBPCIeLC:1"
    assert partition["cores"] == 64
    assert partition["memory"] == "515704M"
    assert partition["gpu"] == "NVIDIAA80080GBPCIeLC:8"
    assert partition["cpus_per_gpu"] == 8

    limits = cp.get_limits(profile)
    assert limits.hard == {
        "max_walltime": "24:00:00",
        "max_nodes": 1,
        "max_cpus": 64,
        "max_array_size": 200,
    }
    assert limits.soft["warn_walltime"] == "08:00:00"
    assert limits.soft["warn_cpus"] == 16
    assert limits.soft["unusual_partitions"] == ["dzagnormal"]
    assert limits.allowed_roots == [
        "/work/share/giggleliu/jiangweiqi/results",
        "/work/share/giggleliu/jiangweiqi/quantum.harness/results",
    ]
    assert profile["filesystem"] == {
        "home": "~",
        "scratch": "/work/share/giggleliu/jiangweiqi/results",
        "project": "/work/share/giggleliu/jiangweiqi/quantum.harness",
        "quota": "",
    }
    assert profile["network"] == {
        "internet_from_login": False,
        "internet_from_compute": False,
    }

    forbidden_keys = {
        "host",
        "hostname",
        "user",
        "username",
        "port",
        "key",
        "key_path",
        "identity_file",
        "password",
        "token",
    }

    def all_keys(value):
        if isinstance(value, dict):
            for key, child in value.items():
                yield key.lower()
                yield from all_keys(child)
        elif isinstance(value, list):
            for child in value:
                yield from all_keys(child)

    assert forbidden_keys.isdisjoint(all_keys(profile))
    raw = path.read_text(encoding="utf-8")
    for forbidden_fragment in ("~/.ssh", "qdeshell_rsa", "windows", "private key"):
        assert forbidden_fragment not in raw.lower()


def test_public_lasg02_profile_is_safe_and_complete():
    root = cp.Path(__file__).resolve().parents[2]
    path = root / "skills/using-slurm/profiles/lasg02-student090.toml"
    mirror_path = root / ".agents/skills/using-slurm/profiles/lasg02-student090.toml"
    agents_skills = root / ".agents/skills"
    profile = cp.load_profile(path)

    assert cp.validate(profile) == []
    assert agents_skills.is_symlink()
    assert agents_skills.readlink() == cp.Path("../skills")
    assert mirror_path.read_bytes() == path.read_bytes()
    assert profile["connection"]["repo_path_remote"] == (
        "/public/home/student090/quantum.harness"
    )
    assert profile["connection"]["ssh"] == {"alias": "lasg02-student090"}
    assert profile["scheduler"] == {
        "type": "slurm",
        "default_partition": "ihicnormal",
        "account": "chenkun2025",
        "qos": "user_student090",
    }

    partition = profile["partitions"][0]
    assert partition == {
        "name": "ihicnormal",
        "class": "cpu",
        "cores": 28,
        "memory": "94179M",
        "def_mem_per_cpu": "3363M",
        "max_wall": "24:00:00",
    }

    limits = cp.get_limits(profile)
    assert limits.hard == {
        "max_walltime": "24:00:00",
        "max_nodes": 1,
        "max_cpus": 28,
        "max_array_size": 200,
    }
    assert limits.allowed_roots == [
        "/public/home/student090/results",
        "/public/home/student090/quantum.harness/results",
    ]
    assert profile["filesystem"] == {
        "home": "/public/home/student090",
        "scratch": "/public/home/student090/results",
        "project": "/public/home/student090/quantum.harness",
        "quota": "",
    }
    assert profile["network"] == {
        "internet_from_login": False,
        "internet_from_compute": False,
    }

    forbidden_keys = {
        "host",
        "hostname",
        "user",
        "username",
        "port",
        "key",
        "key_path",
        "identity_file",
        "password",
        "token",
        "secret",
    }

    def all_keys(value):
        if isinstance(value, dict):
            for key, child in value.items():
                yield key.lower()
                yield from all_keys(child)
        elif isinstance(value, list):
            for child in value:
                yield from all_keys(child)

    assert forbidden_keys.isdisjoint(all_keys(profile))
    raw = path.read_text(encoding="utf-8").lower()
    for forbidden_fragment in ("~/.ssh", "private key", "identityfile"):
        assert forbidden_fragment not in raw


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def test_cli_field_ok(tmp_path, capsys):
    p = _write(tmp_path, FULL_PROFILE)
    rc = cp.main(["--field", "scheduler.type", "--profile", str(p)])
    assert rc == 0
    assert capsys.readouterr().out.strip() == "slurm"


def test_cli_field_list(tmp_path, capsys):
    p = _write(tmp_path, FULL_PROFILE)
    cp.main(["--field", "limits.soft.unusual_partitions", "--profile", str(p)])
    assert capsys.readouterr().out.strip() == "gpu-large"


def test_cli_partition_field(tmp_path, capsys):
    profile = FULL_PROFILE + """

[[partitions]]
name = "cpu"

[[partitions]]
name = "gpu"
required_gres = "gpu:a100:1"
"""
    p = _write(tmp_path, profile)
    rc = cp.main(
        [
            "--partition",
            "gpu",
            "--field",
            "required_gres",
            "--profile",
            str(p),
        ]
    )
    assert rc == 0
    assert capsys.readouterr().out.strip() == "gpu:a100:1"


def test_cli_field_bool(tmp_path, capsys):
    p = _write(tmp_path, "[connection]\ninternet = true\n[identity]\n[scheduler]\n")
    cp.main(["--field", "connection.internet", "--profile", str(p)])
    assert capsys.readouterr().out.strip() == "true"


def test_cli_field_missing(tmp_path, capsys):
    p = _write(tmp_path, FULL_PROFILE)
    rc = cp.main(["--field", "nope.field", "--profile", str(p)])
    assert rc == 1
    assert "not set" in capsys.readouterr().err


def test_cli_bad_profile(tmp_path, capsys):
    rc = cp.main(["--field", "x.y", "--profile", str(tmp_path / "absent.toml")])
    assert rc == 1
    assert "not found" in capsys.readouterr().err
