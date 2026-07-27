# Dzeshell glibc 2.17 compatibility report

## Scope

- Branch: `group-frustration-free`
- Base: `bfb8a510ac20145323279a4170e6b2d3c42cb173`
- Preserved and included the approved production design and implementation plan.
- No push, remote synchronization, Slurm submission, or remote runtime mutation was performed.

## Implementation

- Pinned NumPy 2.2.6, SciPy 1.15.3, h5py 3.14.0, and Matplotlib 3.10.9.
- Pinned the required compatible binary transitives ContourPy 1.3.2 and
  Pillow 12.2.0, then regenerated `uv.lock` with `uv lock`.
- Rebuilt the exact 13-wheel manifest exclusively from `uv.lock` records,
  including each filename, URL, version, and SHA-256. The resulting lock
  SHA-256 is
  `2c2a7b176a09612cb518698db98ee89608598141bfd3932377d74f6a4fb8e537`.
- Added manifest-load validation that accepts pure-Python wheels and CPython
  3.12 x86-64 wheels carrying either `manylinux2014_x86_64` or
  `manylinux_2_17_x86_64`. A compatible member of a dual platform tag is
  sufficient; binary wheels with no glibc 2.17-compatible tag are rejected.
- Strengthened lock verification to require the manifest URL itself, not only
  a matching filename suffix and hash.
- Made preparation remove stale wheel files before producing the exact
  manifest set. Existing exact-set verification remains mandatory.
- Preserved declarative smoke imports and the prohibition on manifest-provided
  command/source execution.
- Updated runtime assertions and ED documentation for the exact Dzeshell
  package versions and glibc floor.

## TDD evidence

The initial focused RED run produced four expected failures: the old
`manylinux_2_28` platform, missing glibc-floor rejection, old core runtime
versions, and the old Dzeshell runner fingerprint. Separate RED runs also
proved that a forged manifest URL was not rejected and that stale wheels were
not pruned. Each behavior was implemented only after its failing test.

## Verification

- Local exact prepare/install/import smoke:
  `.venv/bin/python scripts/turner2018_wheelhouse.py --prepare --smoke`
  - Passed.
  - Imported NumPy 2.2.6, SciPy 1.15.3, h5py 3.14.0, and Matplotlib 3.10.9
    on CPython 3.12.13 x86-64.
- Focused wheelhouse and Dzeshell tests:
  `33 passed, 16 subtests passed`.
- Full script suite:
  `681 passed, 66 subtests passed`.
- Modified Python files passed `py_compile`.
- Dzeshell runner and all three Slurm wrappers passed `bash -n`.
- Modified Python files passed Ruff correctness rules
  `E4,E7,E9,F`; IDE diagnostics reported no errors.
- `git diff --check` passed.

## Remaining concerns

- The compatible wheelhouse has not yet been installed or smoke-tested on the
  actual CentOS 7 login/compute nodes.
- The non-computing Slurm `--check-runtime` and allocation `--test-only` gates
  remain for the authorized remote follow-up.
- No real ED job should be submitted until those remote gates pass and the
  user explicitly approves submission.
