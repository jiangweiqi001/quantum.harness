# LASG02 Task 1 implementation report

## Scope

Implemented Task 1 from the approved LASG02 production plan on
`group-frustration-free` from base commit `4ff307f`.

- Added identical public cluster profiles in both profile trees.
- Added a CPU-only Slurm wrapper with the exact `ihicnormal`, account, QOS,
  24-CPU, `80000M`, and 24-hour request.
- Added a shared runner that accepts only L=22,24,26,28,30, validates canonical
  paths and exact Slurm resources, verifies the offline CPython/package runtime
  before creating output, and invokes `turner2018_l32_server.py --stage all`.
- Documented explicit remote test-only and authorized real submissions for
  every accepted length.
- Preserved Dzeshell wrappers and behavior.

## TDD evidence

Before implementation, the focused profile test failed because
`lasg02-student090.toml` was missing. The focused Slurm test class failed
because the LASG02 wrapper, runner, and documentation were missing. After the
minimal implementation, both focused targets passed.

## Verification

- Focused Task 1 tests: `90 passed, 77 subtests passed`.
- Full `scripts/tests/` suite: `702 passed, 77 subtests passed`.
- `bash -n`: passed for both new shell files.
- `py_compile`: passed for the modified tests and existing ED server.
- IDE lint diagnostics: no errors.
- `git diff --check`: passed.
- Mirrored profile byte comparison: passed.

## Concerns

No remote commands were executed. Login and compute nodes remain treated as
offline. The tracked files contain only the SSH alias and approved public
cluster paths; no host, port, username, key path, or private material is stored.

## Reviewer fixes

Implemented the three validated findings after commit `00d39ba`:

- The wrapper now canonicalizes `SLURM_SUBMIT_DIR` and requires exact equality
  with `/public/home/student090/quantum.harness` before resolving or sourcing
  the runner. A descendant-checkout attacker test verifies no runner side
  effect occurs.
- The runner accepts Slurm's bare-MiB `SLURM_MEM_PER_NODE=80000` and the
  defensive `80000M` form by validating `${SLURM_MEM_PER_NODE%M}`. Other
  suffixes and all different values fail closed.
- Before runtime checks, output creation, or scientific work, the runner
  requires and validates `SLURM_JOB_PARTITION=ihicnormal`,
  `SLURM_JOB_ACCOUNT=chenkun2025`, `SLURM_JOB_QOS=user_student090`,
  `SLURM_JOB_NUM_NODES=1`, `SLURM_NTASKS=1`,
  `SLURM_CPUS_PER_TASK=24`, 80000 MiB per node, and
  `SLURM_TIMELIMIT=1440` minutes. Tests remove and mismatch every scheduler
  variable independently.

### Reviewer-fix TDD evidence

- RED: the new LASG02 class produced `20 failed, 10 passed, 9 subtests passed`.
  The descendant checkout sourced its sentinel runner, bare `80000` was
  rejected, and missing/mismatched scheduler facts reached later validation.
- GREEN focused: `94 passed, 94 subtests passed` across
  `test_cluster_profile.py` and `test_turner2018_l32_server.py`.
- GREEN full: `706 passed, 94 subtests passed` across `scripts/tests/`.
- `bash -n`, `py_compile`, `git diff --check`, byte-identical profile lookup,
  and IDE lint diagnostics all passed.

### Profile mirror architecture

The mirror finding was not applied. `git ls-tree HEAD .agents/skills` reports
mode `120000`, and the tracked link target is `../skills`; therefore
`.agents/skills/using-slurm/profiles/lasg02-student090.toml` resolves to the
canonical tracked profile in a clean checkout. The profile test now asserts the
symlink target and byte-identical resolution. Tracking another file beneath
the symlink would be invalid and would replace the repository's established
single-source architecture.
