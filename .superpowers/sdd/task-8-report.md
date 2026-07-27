# Task 8 Report: Fig. 4 from independent spectra

## Status

Implemented and locally committed on `independent-pxp-ed`.

- Approved baseline: `0cb2198 Use logarithmic overlap scale in Fig. 3`
- Implementation commit: `e15dd8c Render Fig. 4 from validated independent spectra`
- Push: not performed

## RED evidence

The first focused Fig. 4 run produced the expected missing-feature failures:

- `load_independent_fig4_results` and `render_independent_fig4` did not exist.
- The Task 6 server did not expose `FIG4_RENDERER_ADAPTER`.
- Result: 8 Fig. 4 failures and 2 figures-stage failures.

The strict-provenance test deliberately supplies independent energies that differ
from DOI energies. It requires generated histogram counts, densities, spacing
metrics, and source labels to come from the independent energies while retaining
DOI xy data only as a separate `official-doi` series.

Additional failure-injection tests cover creation, write, flush, fsync, backup,
and rename failures for PNG/NPZ/JSON generations, with and without a prior
complete generation.

## GREEN implementation

- Added `--independent-results-root` without changing the legacy `--length` or
  `--paper-exact` behaviors.
- Recursively validates Task 6 plan/stage chains, current execution fingerprint,
  artifact hashes, scientific model and symmetry sector, dimensions, validation
  checks, HDF5 schema, and energy/eigenvector metadata.
- Opens all validated artifacts before validation, consumes energies from the
  exact open HDF5 snapshot, and rehashes every open artifact and stage manifest
  after consumption. Eigenvectors are metadata-only and never read.
- Uses `paper_exact_level_statistics` unchanged on independent sorted energies:
  `D//5:D//2-500`, cubic unfolding, 50-level edge trim, no spacing
  renormalization, and no extra zero-energy filter.
- Produces one all-size overview and one transactional PNG/JSON/NPZ generation
  per available independent length. Missing sizes are reported and never
  bridged or substituted.
- DOI histograms are visually separate overlays and mismatch inputs only.
- Sidecars include source hashes, dimensions, window bounds/count, polynomial
  coefficients, conventions, histogram bins/counts/densities, DOI mismatches,
  missing-size semantics, acceptance, generation identity, and asset hashes.
- Task 6 now runs corrected Fig. 3 and independent Fig. 4, validates both
  generations' provenance/acceptance/hashes, then publishes a hash-validated
  combined `figures` stage.

## Verification

Focused/regression command:

```text
.venv/bin/python -m pytest \
  scripts/tests/test_turner2018_fig3.py \
  scripts/tests/test_turner2018_fig4.py \
  scripts/tests/test_turner2018_official.py \
  scripts/tests/test_turner2018_ed_artifacts.py \
  scripts/tests/test_turner2018_ed_observables.py \
  scripts/tests/test_turner2018_l32_server.py -q
```

Result: `213 passed, 10 skipped, 23 subtests passed in 23.90s`.

`py_compile` passed for all modified Python modules and tests. IDE lints reported
no errors. `git diff --check` passed.

## Real L20 render and combined stage

The current fingerprint forced a fresh real L20 Task 6 workflow through
validation. The combined figures stage then completed and its immediate restart
reported `skipped stage=figures`.

- Fig. 4 overview:
  `tracks/ed/results/turner-2018/task7-independent/L20/figures/fig4_independent_all.png`
- Fig. 4 L20:
  `tracks/ed/results/turner-2018/task7-independent/L20/figures/fig4_independent_L20.png`
- Fig. 4 L20 JSON/NPZ use generation
  `f479aa27-f883-4146-8df1-ba4d3fbcca97`
- Fig. 4 L20 PNG SHA-256:
  `bd0210d61fed9e661958a8547c61b62bea2c0ba51ed2c226f15bcc59df16bbf5`
- Fig. 4 L20 NPZ SHA-256:
  `fb7e788224d62ee5400a14a3bcfe9a168af30fd21cb5c8a9e86b8aa7959a5b1c`
- Full/sector dimensions: `15127 / 455`
- Exact raw window bounds/count: `[91, -273] / 91`
- Eigenvectors: shape `[455,455]`, chunks `[455,1]`, dtype `float64`,
  access `metadata-only`
- Acceptance: passed

The L20 image and sidecars were inspected. The exact paper slice contains only
91 levels, fewer than the 102 required to trim 50 levels from each edge and
retain a spacing. The image therefore states that the exact paper window is too
short and plots only the three theory references; it does not substitute a
finite-size window or DOI spectrum.

The corrected Fig. 3 output was also inspected at:

`tracks/ed/results/turner-2018/task7-independent/L20/figures/fig3_independent_L20.png`

Panel (a) uses the approved logarithmic overlap axis from `0cb2198`. The combined
manifest records passing, hash-checked Fig. 3 and Fig. 4 generations.

## Concerns

- Real local L20 is a provenance/orchestration render, not a paper-statistics
  histogram: its exact paper window is mathematically too short after the
  mandated trim. This is recorded explicitly rather than silently changing the
  convention.
- Real independent L28/L30/L32 eigensystems were not recomputed locally; those
  production sizes require the documented high-memory workflow.
- Generated render artifacts are gitignored and are not included in the commit.
