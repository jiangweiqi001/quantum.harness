# Task 8 Report: Fig. 4 from independent spectra

## Status

Implemented and locally committed on `independent-pxp-ed`.

- Approved baseline: `0cb2198 Use logarithmic overlap scale in Fig. 3`
- Implementation commit: `e15dd8c Render Fig. 4 from validated independent spectra`
- Review remediation commit:
  `32d319c Harden independent figure publication and restart validation`
- Second re-review code commit:
  `82798b8 Close remaining figure fail-closed gaps`
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
- Produces one all-size overview and one PNG/JSON/NPZ set per available
  independent length, and publishes the entire collection as one transactional
  generation. Missing sizes are reported and never bridged or substituted.
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

Result after second re-review remediation:
`231 passed, 10 skipped, 34 subtests passed in 26.04s`.

`py_compile` passed for all modified Python modules and tests. IDE lints reported
no errors. `git diff --check` passed.

## Real L20 render and combined stage

The current fingerprint forced a fresh real L20 Task 6 workflow through
validation. The combined figures stage was rebuilt twice successfully, then an
immediate restart deeply revalidated its references and reported
`skipped stage=figures`.

- Fig. 4 overview:
  `tracks/ed/results/turner-2018/task7-independent/L20/figures/fig4_independent_all.png`
- Fig. 4 L20:
  `tracks/ed/results/turner-2018/task7-independent/L20/figures/fig4_independent_L20.png`
- Fig. 4 L20 JSON/NPZ use generation
  `95e2f33e-6993-46d3-b5de-523aa4e62dfe`; the overview and L20 triplets share
  this whole-set generation identity.
- Fig. 4 L20 PNG SHA-256:
  `bd0210d61fed9e661958a8547c61b62bea2c0ba51ed2c226f15bcc59df16bbf5`
- Fig. 4 L20 NPZ SHA-256:
  `b96a333b3a6bc0e921db1a0b0e3bfa58377e3fd77aa92a04ce13730aadef7358`
- Full/sector dimensions: `15127 / 455`
- Exact raw window bounds/count: `[91, -273] / 91`
- Exact resolved nonnegative bounds: `[91, 182]`
- Eigenvectors: shape `[455,455]`, chunks `[455,1]`, dtype `float64`,
  access `metadata-only`
- Provenance/render acceptance: passed
- Statistics/histogram acceptance: unavailable/false
- Overall local mode: `provenance-only` (passed)

The L20 image and sidecars were inspected. The exact paper slice contains only
91 levels, fewer than the 102 required to trim 50 levels from each edge and
retain a spacing. The image therefore states that the exact paper window is too
short and plots only the three theory references; it does not substitute a
finite-size window or DOI spectrum.

The corrected Fig. 3 output was also inspected at:

`tracks/ed/results/turner-2018/task7-independent/L20/figures/fig3_independent_L20.png`

Panel (a) uses the approved logarithmic overlap axis from `0cb2198`. The combined
manifest records passing, hash-checked Fig. 3 and Fig. 4 generations.

## Review remediation evidence

The review RED suites reproduced every reported gap:

- Both Fig. 3 and Fig. 4 discovery attempted to ingest
  `figures/manifest.json`.
- A completed figures stage skipped after a referenced figure asset was
  corrupted.
- Mixed JSON/NPZ generation identities were accepted.
- Later-size Fig. 4 write/rename failures left earlier new outputs published.
- Per-size JSON leaked all-size lengths and series.
- DOI overlays lacked archive/member hashes.
- Shared rollback deleted backups after injected unlink/link restoration
  failures.
- L20 sidecars did not distinguish provenance acceptance from histogram
  availability.

The GREEN tests now verify:

- Discovery uses only `stages/plan.json` roots, so rendering and rerendering do
  not discover output manifests.
- Restart skip revalidates the Fig. 3 triplet plus every overview and
  size-specific Fig. 4 PNG/NPZ/JSON triplet, their hashes, JSON/NPZ generation
  identity, scoped lengths/series, provenance, render status, and statistics
  acceptance.
- Production L=28/30/32 requires accepted exact statistics; L20 is explicitly
  provenance-only.
- Overview and every size-specific triplet roll back together on later-size
  write or rename failure, both with and without a prior generation.
- Shared publication preserves recoverable backups under injected rollback
  unlink, restore-link, restore-fsync, and cleanup failures.
- Per-size JSON and NPZ contain exactly one matching length/series set.
- DOI provenance records resolved archive path, archive/member byte sizes, and
  archive/member SHA-256 values.
- Raw Python bounds and resolved nonnegative slice bounds are both persisted.

## Second re-review fail-closed evidence

The new RED runs reproduced the four residual gaps:

- The combined manifest represented Fig. 4 as one overview record, so the L20
  size triplet was not enumerated or revalidated.
- Production acceptance allowed false `provenance_passed` or `render_passed`
  when statistics happened to pass.
- Replacing `level_statistics.zip` between parsing and hashing could bind
  plotted arrays to different archive bytes than the recorded hash.
- Injecting backup-unlink failure at cleanup positions 0, 1, and 2 left no
  explicit durable state describing the valid current generation and remaining
  backups.

The GREEN implementation and tests establish:

- `figures/manifest.json` now records `fig4.overview` and a `fig4.sizes` entry
  for every available length. Restart reconstructs and compares the complete
  set; it validates every triplet's hashes and shared generation ID, and checks
  JSON/NPZ length scope plus per-length provenance/histogram acceptance.
- L=28/30/32 requires top-level provenance and render acceptance, required and
  passing statistics, and passing per-length provenance and histogram checks.
  Nonproduction L20 remains explicitly accepted in `provenance-only` mode.
- DOI members are read through one open archive handle. The archive is hashed
  before and after member consumption, and each member hash is calculated from
  the exact bytes parsed into the plotted array. A path-replacement race test
  proves the loaded array and recorded hash remain bound to the original open
  snapshot.
- Before sequential backup cleanup, publication durably records a recovery
  manifest containing hashes of the complete current generation and the
  backup inventory. It updates that inventory after each deletion; any failure
  leaves `backup-cleanup-failed` bookkeeping. Parameterized failures at every
  backup position preserve the complete new generation plus accurate remaining
  backup state. Successful cleanup removes the recovery manifest.

Real L20 evidence after the second re-review:

- A fresh fingerprint-driven `--stage all` completed all scientific stages.
- `--stage figures --rebuild` completed, then an immediate restart reported
  `skipped stage=figures`.
- Appending corrupt bytes to the real
  `fig4_independent_L20.npz` caused restart exit code 2 with
  `Fig. 4 generation asset hash check failed`. Restoring the exact bytes made
  the next restart report `skipped stage=figures`.
- The real combined manifest contains the overview and L20 records, each with
  all three asset hashes and shared generation ID. No recovery manifest remains
  after successful publication.

## Concerns

- Real local L20 is a provenance/orchestration render, not a paper-statistics
  histogram: its exact paper window is mathematically too short after the
  mandated trim. This is recorded explicitly rather than silently changing the
  convention.
- Real independent L28/L30/L32 eigensystems were not recomputed locally; those
  production sizes require the documented high-memory workflow.
- Generated render artifacts are gitignored and are not included in the commit.
