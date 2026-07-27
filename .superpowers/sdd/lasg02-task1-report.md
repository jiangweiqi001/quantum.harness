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
