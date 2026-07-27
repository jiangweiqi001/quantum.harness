# Dzeshell Matplotlib Offline Runtime Report

## Status

Implemented on `group-frustration-free`; local commit only, no push.

## RED evidence

Focused tests failed before production changes because the manifest contained
only `numpy`, `scipy`, and `h5py`, had no Matplotlib expected version, and
`--check-runtime` did not execute the manifest smoke import. The initial result
was 3 expected manifest/runtime failures plus a dedicated smoke-execution
failure.

## GREEN implementation

- Expanded the exact CPython 3.12 Linux wheel set to 13 locked packages:
  NumPy, SciPy, h5py, Matplotlib 3.11.1, contourpy, cycler, fonttools,
  kiwisolver, packaging, pillow, pyparsing, python-dateutil, and six.
- Copied every filename, URL, version, and SHA-256 from `uv.lock`; no TeNPy
  package was added.
- Preserved exact-file-set wheelhouse rejection.
- Made both the isolated install smoke test and `--check-runtime` execute
  `import numpy, scipy, h5py, matplotlib`, then require exact declared versions.
- The Dzeshell common runner already invokes `--check-runtime` before the
  `--stage all` server execution; regression coverage now binds that gate to
  Matplotlib 3.11.1 and verifies its ordering before computation.

## Verification

- Exact offline prepare/install/import smoke:
  passed on CPython 3.12.13 x86_64 with NumPy 2.4.6, SciPy 1.18.0,
  h5py 3.16.0, and Matplotlib 3.11.1.
- Relevant suites:
  `64 passed, 66 subtests passed`.
- Full script regressions:
  `667 passed, 66 subtests passed`.
- `py_compile`, IDE lints, and `git diff --check`: passed.

The repository `make test` target could not collect coverage because the local
environment lacks `pytest-cov`; the same complete `scripts/tests` suite passed
directly under the project CPython 3.12 environment.

## Concerns

None in scope. The wheelhouse remains platform-specific and lock-bound by
design; changing `uv.lock` requires regenerating the manifest.

## Review remediation: declarative smoke imports

RED evidence:

- The focused suite produced 12 failures: the manifest had no declarative
  `modules` list, source-bearing fields and malicious names were accepted, no
  clean-CI isolated-smoke helper existed, and `--check-runtime` still used
  `exec`.

GREEN implementation:

- Replaced `smoke_import.command` with
  `modules: [numpy, scipy, h5py, matplotlib]`.
- Manifest loading rejects `command` and `source`, empty or duplicate module
  lists, non-ASCII identifier/dotted-name strings, and module/version-key
  mismatches.
- Both runtime paths import only through tool-controlled
  `importlib.import_module` calls. The isolated subprocess receives validated
  JSON as an argument to fixed tool source; manifest text is never evaluated.
- Added a wheel-independent subprocess unit test that imports Matplotlib and
  proves exact-version mismatch rejection. The prepare-smoke test now exercises
  generated command construction without any downloaded wheelhouse.
- Preserved all 13 exact wheels, exact-file-set rejection, and the Dzeshell
  pre-compute runtime-check ordering.

Review verification:

- Focused wheelhouse and runner suites:
  `73 passed, 66 subtests passed`.
- Full script regressions:
  `676 passed, 66 subtests passed`.
- Optional real isolated wheelhouse smoke passed on CPython 3.12.13 x86_64,
  importing NumPy, SciPy, h5py, and Matplotlib and matching all exact versions.
- `py_compile`, IDE lints, source-field/`exec` scan, and `git diff --check`
  passed.
