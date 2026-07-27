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

## Review-fix follow-up

The review identified that the first validator treated every `none-any` wheel
as compatible without checking its Python tag, and that regenerating the lock
against the default index rewrote unrelated registry URLs.

The wheel validator now uses `packaging.utils.parse_wheel_filename` and
`packaging.tags` to evaluate the complete PEP 427 compressed tag set against
CPython 3.12. It accepts `py3` and `py2.py3` universal wheels, compatible
CPython `abi3` wheels, and any compatible member of a dual tag. Binary
candidates must also provide an x86-64 manylinux tag whose glibc floor is no
newer than 2.17. Python 2-only, CPython 3.13-only, wrong-architecture,
non-manylinux, malformed, and path-bearing filenames are rejected.

`uv.lock` was regenerated from the base `bfb8a51` lock while retaining its
Tsinghua registry. The resulting base-to-final lock diff is limited to the six
changed package records (the four core pins plus ContourPy and Pillow) and the
project dependency metadata: 73 insertions and 73 deletions. Unrelated package
records and registry URLs are unchanged. The cleaned lock SHA-256 is
`2eb458c846b00744136673321f93a2afc190203d8c8e3e73d92764b5a1b1a558`;
the manifest URLs and lock fingerprint were updated from that lock.

### Review-fix TDD evidence

RED command:

```text
.venv/bin/python -m pytest scripts/tests/test_turner2018_wheelhouse.py -q
10 failed, 22 passed in 0.70s
```

The failures covered Python 2-only and CPython 3.13-only pure wheels,
incompatible binary interpreter/architecture/platform tags, malformed and
path-bearing filenames, and intended compatible `abi3`/dual tags rejected by
the old validator.

GREEN command:

```text
.venv/bin/python -m pytest scripts/tests/test_turner2018_wheelhouse.py -q
32 passed in 0.66s
```

An additional malformed PEP 600 major-version case was then added:

```text
.venv/bin/python -m pytest scripts/tests/test_turner2018_wheelhouse.py::test_manifest_rejects_wheels_incompatible_with_cpython_3_12 -q
1 failed, 5 passed in 0.05s
```

After restricting versioned manylinux tags to the glibc major version, the
complete focused file passed:

```text
.venv/bin/python -m pytest scripts/tests/test_turner2018_wheelhouse.py -q
33 passed in 0.65s
```

### Review-fix verification evidence

```text
uv lock --check
Resolved 39 packages in 0.68ms
```

```text
.venv/bin/python -m pytest scripts/tests/test_turner2018_wheelhouse.py scripts/tests/test_turner2018_l32_server.py::TurnerL32SlurmTests -q
45 passed, 16 subtests passed in 0.99s
```

```text
.venv/bin/python scripts/turner2018_wheelhouse.py --smoke
Successfully installed contourpy-1.3.2 cycler-0.12.1 fonttools-4.63.0 h5py-3.14.0 kiwisolver-1.5.0 matplotlib-3.10.9 numpy-2.2.6 packaging-26.2 pillow-12.2.0 pyparsing-3.3.2 python-dateutil-2.9.0.post0 scipy-1.15.3 six-1.17.0
{"smoke_import": {"imported_modules": ["numpy", "scipy", "h5py", "matplotlib"], "machine": "x86_64", "python": "3.12.13", "versions": {"h5py": "3.14.0", "matplotlib": "3.10.9", "numpy": "2.2.6", "scipy": "1.15.3"}}, "verified": true}
```

```text
.venv/bin/python -m pytest scripts/tests -q
693 passed, 66 subtests passed in 88.69s
```

```text
.venv/bin/python -m py_compile scripts/turner2018_wheelhouse.py scripts/tests/test_turner2018_wheelhouse.py
uvx ruff check --select E4,E7,E9,F scripts/turner2018_wheelhouse.py scripts/tests/test_turner2018_wheelhouse.py
All checks passed!
git diff --check
exit 0
```

Final post-hardening gate:

```text
uv lock --check
Resolved 39 packages in 0.88ms
.venv/bin/python -m pytest scripts/tests -q
694 passed, 66 subtests passed in 93.04s
.venv/bin/python scripts/turner2018_wheelhouse.py --smoke
{"smoke_import": {"imported_modules": ["numpy", "scipy", "h5py", "matplotlib"], "machine": "x86_64", "python": "3.12.13", "versions": {"h5py": "3.14.0", "matplotlib": "3.10.9", "numpy": "2.2.6", "scipy": "1.15.3"}}, "verified": true}
.venv/bin/python -m py_compile scripts/turner2018_wheelhouse.py scripts/tests/test_turner2018_wheelhouse.py
uvx ruff check --select E4,E7,E9,F scripts/turner2018_wheelhouse.py scripts/tests/test_turner2018_wheelhouse.py
All checks passed!
git diff --check
exit 0
```
