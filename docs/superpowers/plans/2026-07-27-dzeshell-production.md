# Dzeshell Production Configuration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Configure and verify reproducible medium/large PXP ED execution on Dzeshell without submitting a real job.

**Architecture:** Public tracked files retain only scheduler facts and the local SSH alias. Size-specific immutable Slurm wrappers call one shared offline runner; a separate bootstrap synchronizes the reviewed branch, portable Python 3.12, and exact wheelhouse to shared storage.

**Tech Stack:** Bash, TOML, Python 3.12, Slurm, SSH/rsync, uv/pip wheelhouse, pytest.

## Global Constraints

- Never write SSH host, port, username, key path, or private material into Git.
- Partition is `dzagnormal`; GRES type is `NVIDIAA80080GBPCIeLC`.
- Resources are fixed as 8/1/60000M for L22–28, 16/2/120000M for L30, and 32/4/240000M for L32.
- Remote checkout/results/runtime live under `/work/share/giggleliu/jiangweiqi`.
- HOME is not used for production artifacts.
- No real `sbatch` submission occurs in this plan; only `--test-only`.

---

### Task 1: Correct tracked Dzeshell profile and Slurm wrappers

**Files:**
- Modify: `skills/using-slurm/profiles/qdeshell.toml`
- Modify: `.agents/skills/using-slurm/profiles/qdeshell.toml`
- Create: `scripts/turner2018_dzeshell_run.sh`
- Create: `scripts/turner2018_dzeshell_l22_28.sbatch`
- Create: `scripts/turner2018_dzeshell_l30.sbatch`
- Create: `scripts/turner2018_dzeshell_l32.sbatch`
- Modify: `scripts/tests/test_cluster_profile.py`
- Modify: `scripts/tests/test_turner2018_l32_server.py`
- Modify: `tracks/ed/README.md`

**Interfaces:**
- Wrappers export one allowed `TURNER_LENGTH` class and source the common run script.
- The common runner consumes `TURNER_LENGTH`, `SLURM_*`, shared repo/runtime/results paths.

- [ ] Write RED tests for exact live profile, resource classes, secret absence,
  unsupported lengths, shared paths, and common-runner environment checks.
- [ ] Run focused tests and record expected old-profile failures.
- [ ] Implement profile and wrappers without duplicating solver logic.
- [ ] Run focused/full regressions, shell syntax checks, lints, and diff check.
- [ ] Commit locally; do not push.

---

### Task 2: Bootstrap and validate the remote offline runtime

**Files:**
- Generated only under `/work/share/giggleliu/jiangweiqi`.
- No tracked source edits unless Task 1 review requires them.

**Interfaces:**
- Produces remote reviewed checkout, CPython 3.12 runtime, exact wheelhouse,
  virtual environment, and test-only scheduler evidence.

- [ ] Verify the local branch is clean and identify the reviewed HEAD.
- [ ] Synchronize source without `.gitignored` results, credentials, `.ssh`,
  local `.venv`, or private files. Preserve a Git checkout/fingerprint by using
  a pushed reviewed branch or a Git bundle; request push authorization if
  needed.
- [ ] Synchronize portable CPython 3.12 and exact wheelhouse to shared storage.
- [ ] Create the remote venv offline and verify Python, NumPy, SciPy, h5py, and
  Matplotlib versions against the tracked fingerprint.
- [ ] Run remote unit smoke tests and local-validation dry run without Slurm.
- [ ] Execute `sbatch --test-only` for L28, L30, and L32 wrappers. Record exact
  resources and estimated starts.
- [ ] Do not submit real jobs. Report queue blocker and next approval gate.
