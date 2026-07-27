# LASG02 Production Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Run restartable Turner ED at L=22–30 and four iTEBD continuations on the LASG02 CPU account, while routing L=32 to Dzeshell.

**Architecture:** A public LASG02 profile and immutable Slurm wrappers call canonical shared runners under the reviewed remote checkout. Portable CPython 3.12 and exact offline wheelhouses provide reproducible environments; independent output roots and atomic manifests make every job restartable and downloadable.

**Tech Stack:** Bash, TOML, Python 3.12, Slurm, SSH/rsync, NumPy, SciPy, h5py, Matplotlib, TeNPy, pytest.

## Global Constraints

- Never track SSH host, port, username, key path, or private key.
- LASG02 paths remain below `/public/home/student090`.
- ED L=22–30 uses `ihicnormal`, account `chenkun2025`, QOS
  `user_student090`, 24 CPUs, `80000M`, one node, and 24 hours.
- LASG02 rejects L=32; Dzeshell remains the only L=32 target.
- Python is exactly 3.12 with a glibc-2.17-compatible exact wheelhouse.
- Each job validates its runtime and paths before numerical work.
- Real submissions are authorized, but test-only validation runs first.

---

### Task 1: Add LASG02 profile and ED wrappers

**Files:**
- Create: `skills/using-slurm/profiles/lasg02-student090.toml`
- Create: `.agents/skills/using-slurm/profiles/lasg02-student090.toml`
- Create: `scripts/turner2018_lasg02_run.sh`
- Create: `scripts/turner2018_lasg02_l22_30.sbatch`
- Modify: `scripts/tests/test_cluster_profile.py`
- Modify: `scripts/tests/test_turner2018_l32_server.py`
- Modify: `tracks/ed/README.md`

**Interfaces:**
- The wrapper consumes `TURNER_LENGTH` in `{22,24,26,28,30}`.
- The runner uses the reviewed checkout, `.venv`, and
  `/public/home/student090/results/turner-l<L>`.

- [ ] Add failing tests for profile facts, exact resources, secret absence,
  canonical runner lookup, approved paths, runtime-before-compute ordering,
  length rejection, and explicit remote `TURNER_LENGTH` export.
- [ ] Run focused tests and record the expected missing-profile failures.
- [ ] Implement the profile, wrapper, runner, and documented test-only/real
  commands without duplicating ED solver logic.
- [ ] Run focused/full tests, `bash -n`, pycompile, lints, and diff checks.
- [ ] Review and commit locally; push only after approval by the task reviewer.

---

### Task 2: Deploy and submit LASG02 ED plus Dzeshell L=32

**Files:**
- Generated remotely below the approved account roots.
- No tracked changes unless Task 1 review identifies a defect.

**Interfaces:**
- Produces Slurm job IDs and restartable result directories for
  L=22,24,26,28,30 on LASG02 and L=32 on Dzeshell.

- [ ] Synchronize the reviewed commit, portable Python, exact wheelhouse, and
  official comparison data to LASG02.
- [ ] Create the remote venv and verify exact runtime imports.
- [ ] Run remote smoke tests and one test-only request per resource class.
- [ ] Submit real L=22–30 jobs to LASG02 and L=32 to Dzeshell.
- [ ] Record job IDs, queue reasons, logs, and result paths; do not report
  queued jobs as running.

---

### Task 3: Add and deploy iTEBD continuation jobs

**Files:**
- Create: `scripts/turner2018_itebd_lasg02_run.sh`
- Create: `scripts/turner2018_itebd_lasg02.sbatch`
- Modify: `scripts/turner2018_wheelhouse_manifest.json` or create a dedicated
  exact iTEBD manifest if ED exact-set isolation requires it.
- Modify: `scripts/turner2018_wheelhouse.py`
- Modify: `scripts/tests/test_turner2018_wheelhouse.py`
- Modify: `scripts/tests/test_turner2018_fig2_itebd.py`
- Modify: `tracks/ed/README.md`

**Interfaces:**
- One array or four immutable jobs consume states
  `vacuum,Z2,Z3,Z4`, their checkpoint paths, χ=400, dt=0.05, and t=30.
- Outputs remain state-specific and restartable.

- [ ] Add failing tests for exact TeNPy runtime closure, checkpoint/state
  matching, state-isolated outputs, resources, and resume arguments.
- [ ] Implement the smallest exact offline runtime and Slurm orchestration.
- [ ] Run local and remote import/checkpoint smoke tests plus test-only.
- [ ] Submit four real continuations across available accounts without sharing
  output files.
- [ ] Monitor completion, fetch artifacts, validate metrics, and render final
  Fig. 2/3/4 locally.
