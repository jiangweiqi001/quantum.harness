# Task 7 Report: Fig. 3 from independent eigensystems

## Status

Implemented and locally committed on `independent-pxp-ed`.

- Implementation commits:
  - `705111c Render Fig. 3 from validated independent ED artifacts`
  - `aab8340 Harden independent Fig. 3 publication`
  - `fcdbc94 Close Fig. 3 partial cleanup gaps`
- Push: not performed
- Combined Task 6 `figures` stage: intentionally remains incomplete until Task 8 supplies Fig. 4

## Implementation evidence

- Added `--independent-results-root` while preserving the legacy `--length`/`analyze_spectrum` route.
- Recursively discovers Task 6 length directories and validates current plan fingerprints, dependency-manifest hashes, artifact SHA-256 values, scientific model attributes, validation metrics, dimensions, HDF5 dataset metadata, and observable shapes.
- Reads eigenvector shape/dtype/chunks only; generated panel data comes from persisted energies and observables.
- Uses the established `turner2018_official.select_fig3_pr2_states` and FSA matching tolerances.
- Writes atomic JSON/NPZ sidecars with per-series source labels, per-length hashes/dimensions, overlap sum, FSA matches, shell-profile evidence, PR2 selector details, DOI mismatches, missing-size policy, and acceptance.
- Added a Task 6 Fig. 3 renderer hook. It runs before the combined renderer gate, but no `figures` completion manifest is published before Task 8.

## TDD evidence

Synthetic provenance test initially failed with:

`AttributeError: module 'turner2018_fig3' has no attribute 'render_independent_fig3'`

The figures-stage hook test initially failed because `FIG3_RENDERER_ADAPTER` did not exist.

The real L20 run exposed absent DOI L20 overlap members. A regression test reproduced this and failed because the absent member was requested; exact ZIP-member discovery now skips unavailable overlays.

Final focused/regression command:

```text
.venv/bin/python -m pytest \
  scripts/tests/test_turner2018_fig3.py \
  scripts/tests/test_turner2018_official.py \
  scripts/tests/test_turner2018_ed_artifacts.py \
  scripts/tests/test_turner2018_ed_observables.py \
  scripts/tests/test_turner2018_l32_server.py -q
```

Result: `133 passed, 4 skipped, 23 subtests passed`.

`py_compile` passed for both modified production modules, IDE lints reported no errors, and `git diff --check` passed.

## Real L20 workflow and render

Generated from a fresh Task 6 `--stage all --length 20` workflow:

- PNG: `tracks/ed/results/turner-2018/task7-independent/figure/fig3_independent_L20.png`
- JSON: `tracks/ed/results/turner-2018/task7-independent/figure/fig3_independent_L20.json`
- NPZ: `tracks/ed/results/turner-2018/task7-independent/figure/fig3_independent_L20.npz`
- Eigensystem: 1.7 MB; observables: 96 KB
- Full/sector dimensions: `15127 / 455`
- Eigenvector metadata: shape `[455,455]`, chunks `[455,1]`, dtype `float64`, access `metadata-only`
- Z2 overlap sum: `0.49999999999999994`
- PR2 other/special means: `0.010756759895965749 / 0.03385521791058739`
- Selected panel energies: `-12.07121376222543` and `-2.672092951313271`
- Acceptance: passed

The PNG and JSON were inspected. Panels a-c show independent L20 ED/FSA values; panel d shows the independent L20 PR2 markers plus available DOI L26/L32 PR2 overlays.

## Concerns

- The DOI overlap archive has entries only for L=22 and above, so no L20 overlap mismatch can be computed; the JSON records it as `null` rather than substituting another size.
- DOI FSA projections remain available only for L=26 and L=32. Missing sizes are never bridged or filled.
- Generated results are gitignored local artifacts and are not part of the commit.

## Review fixes and logarithmic panel (a)

The review follow-up now additionally:

- checks the stored execution fingerprint against the current source/lock/runtime fingerprint;
- holds validated NPZ/HDF5 file handles through consumption and rechecks hashes without reading the eigenvector dataset;
- transactionally publishes PNG/NPZ/JSON as one durable generation with rollback/cleanup tests;
- uses scientifically validated independent fixtures and checks panels b-d against independently computed values;
- makes legacy `--length` and `--independent-results-root` mutually exclusive;
- records shell dimensions/conventions, match strengths, selected indices/energies, and source/scientific hashes;
- defines missing lengths only as even gaps within the available independent range;
- labels panel (c) as the matched interior special state; and
- renders panel (a) with the paper's logarithmic overlap axis in both independent and legacy paths.

The log-axis RED tests first observed `get_yscale() == "linear"` for both render
paths. They now pass with `log` and `nonpositive="mask"`. A synthetic test keeps
exact zeros as zero and a `1e-12` overlap unchanged in both the Matplotlib
collection and NPZ sidecar; no positive floor is fabricated.

Fresh verification:

- Fig. 3 focused suite: `40 passed`.
- Fig. 3 plus official/artifact/observable regressions: `122 passed, 4 skipped`.
- Server regressions excluding the two unrelated Task 8 figures-stage RED tests:
  `42 passed, 2 deselected, 23 subtests passed`.
- `py_compile`, IDE lint, and targeted `git diff --check`: passed.

Fresh L20 output:

- PNG: `tracks/ed/results/turner-2018/task7-independent/figure/fig3_independent_L20.png`
- JSON/NPZ generation ID match: `true`
- Axis metadata: `y_scale=log`, `nonpositive=masked`, stored overlap values unmodified
- Independent ED minimum positive overlap: `2.7269433915700143e-11`
- Independent ED maximum overlap: `0.08822158774232566`
- Independent FSA minimum positive overlap: `6.247879877181726e-06`
- L20 overlap sum: `0.49999999999999994`

Visual inspection confirms the low-overlap ED cloud remains visible down to
approximately `10^-11`, with no synthetic floor or replacement points.

## Review remediation

All Critical, Important, and Minor findings were addressed in `aab8340`.

- Rendering now compares the stored execution fingerprint and digest to a freshly computed source/lock/Python/package fingerprint.
- Basis NPZ, Hamiltonian NPZ, eigensystem HDF5, observables HDF5, plan, validation metrics, and all recursive stage manifests are opened before validation. The exact open handles are hashed against validated manifests and rehashed after consumption. Eigenvectors remain metadata-only.
- PNG, NPZ, and JSON now share a UUID generation. PNG/NPZ hashes are recorded in JSON, and a rollback-safe hard-link transaction durably fsyncs files/directories, preserves a prior complete generation on injected failure, and removes partial/backup files. First-generation failure leaves no generation.
- Tests use an unmodified, scientifically validated Task 6 workflow as the independent fixture. Deliberately different synthetic DOI overlays prove separation, while panels b/c weights and panel d means are checked directly against independent HDF5 values.
- `--length` and `--independent-results-root` are mutually exclusive.
- Per-length evidence now includes NPZ/HDF5/stage/fingerprint hashes, shell dimensions and conventions, all exact/FSA match indices and energies, match strengths, selector hashes, and selected-state details.
- Missing-size output is now `missing_lengths_within_independent_range` with an explicit range object; a lone L20 result reports no internal range and no missing lengths.
- Panel c is titled `matched interior special state`, matching the established selector rather than implying a zero-energy state.

Review RED tests covered stale current fingerprints, path replacement after validation, prior/first-generation publication failures, backup failure, CLI ambiguity, selector evidence, and missing-length semantics.

Final regression result: `139 passed, 4 skipped, 23 subtests passed`. `py_compile` and `git diff --check` also passed.

The reviewed L20 generation is `47bfdf63-a603-4dad-8d63-ffb4986cb4a1`:

- PNG SHA-256: `af26779c6e3cf6880f51e927bea6edd119d3255cceb0863392470927768cac70`
- NPZ SHA-256: `4e9b8f5f3b0249d9e2b9d99528ebc391fb7e11fe3bc3062c93ab888eb924fc53`
- Full/sector dimensions: `15127 / 455`
- Overlap sum: `0.49999999999999994`
- PR2 other/special means: `0.010756759895965749 / 0.03385521791058739`
- Selected tower-ground match: exact/FSA indices `0 / 0`, energies `-12.07121376222543 / -12.017010493123097`, strength `0.9818818789488392`
- Selected interior-special match: exact/FSA indices `101 / 4`, energies `-2.672092951313271 / -2.633527459988426`, strength `0.7878643145131313`

## Final partial-cleanup remediation

Commit `fcdbc94` closes the remaining pre-publication cleanup gap:

- All PNG/NPZ/JSON partial paths are predetermined and registered before any writer can create a file.
- PNG creation and fsync now run inside the same cleanup owner as NPZ/JSON creation, writes, flushes, and fsyncs.
- Every failure path durably unlinks all registered partials and closes the Matplotlib figure.
- The prior complete generation remains byte-for-byte unchanged after injected failures; first-generation failures leave no PNG, NPZ, or JSON.
- Parameterized tests cover PNG, NPZ, and JSON creation/write/fsync failures with and without a prior generation (18 cases), plus six failures injected into the real file-fsync calls.

Final focused/regression result: `163 passed, 4 skipped, 23 subtests passed`. `py_compile`, IDE lints, and `git diff --check` passed.

The refreshed L20 generation is `0f28209d-0cc5-4f2a-b10f-dd3dd73e55c9`:

- PNG SHA-256: `af26779c6e3cf6880f51e927bea6edd119d3255cceb0863392470927768cac70`
- NPZ SHA-256: `134abd98a26ed4f9483634bc5f271d5ae1a05d40f10a84524fb62cff1c2dc23a`
