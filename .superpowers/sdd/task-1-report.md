# Task 1 Report: Compact constrained states and normalized orbit basis

## Scope and constraints followed
- Implemented only Task 1 in `/tmp/quantum-harness-independent-ed` on branch `independent-pxp-ed`.
- Added exactly the requested new files:
  - `scripts/turner2018_ed_engine.py`
  - `scripts/tests/test_turner2018_ed_engine.py`
- Kept production logic free of Hamiltonian assembly (Task 2 untouched).
- Did not scan `2**L` or materialize production basis transforms.
- Used `/home/footman/code/quantum.harness/.venv/bin/python` with `PYTHONPATH=scripts` for tests.

## TDD evidence (RED -> GREEN)

### RED (expected import failure before production module existed)
Command:
```bash
PYTHONPATH=scripts /home/footman/code/quantum.harness/.venv/bin/python -m pytest scripts/tests/test_turner2018_ed_engine.py -q
```

Output:
```text
==================================== ERRORS ====================================
_________ ERROR collecting scripts/tests/test_turner2018_ed_engine.py __________
ImportError while importing test module '/tmp/quantum-harness-independent-ed/scripts/tests/test_turner2018_ed_engine.py'.
...
E   ModuleNotFoundError: No module named 'turner2018_ed_engine'
=========================== short test summary info ============================
ERROR scripts/tests/test_turner2018_ed_engine.py
!!!!!!!!!!!!!!!!!!!! Interrupted: 1 error during collection !!!!!!!!!!!!!!!!!!!!
1 error in 0.28s
```

### GREEN (focused Task 1 suite)
Command:
```bash
PYTHONPATH=scripts /home/footman/code/quantum.harness/.venv/bin/python -m pytest scripts/tests/test_turner2018_ed_engine.py -q
```

Output:
```text
.......                                                                  [100%]
7 passed in 0.17s
```

### GREEN regressions (requested baseline pairing)
Command:
```bash
PYTHONPATH=scripts /home/footman/code/quantum.harness/.venv/bin/python -m pytest scripts/tests/test_turner2018_ed_engine.py scripts/tests/test_pxp_ed.py -q
```

Output:
```text
.......................                                                  [100%]
23 passed in 0.32s
```

## Implementation summary
- Implemented `enumerate_constrained_states(length)` via a two-pass recurrence:
  - pass 1 counts legal periodic blockade strings,
  - pass 2 writes directly into a preallocated `np.uint64` array,
  - one final sort and strict shape/dtype/ordering assertions.
- Implemented dihedral utilities:
  - `dihedral_members(state, length)` (sorted unique D_L orbit),
  - `canonical_dihedral(states, length)` using vectorized rotate/reflect + elementwise minima.
- Implemented `OrbitBasis` dataclass and exact `index_of` validation behavior from the brief.
- Implemented `build_orbit_basis(length, chunk_size)`:
  - canonicalizes constrained states in bounded chunks,
  - retains canonical representatives only,
  - computes orbit sizes as unique dihedral-member counts (`np.uint8`).
- Added tests that verify:
  - constrained-state equivalence to oracle at `L=10`,
  - rotation/reflection convention match,
  - canonicalization behavior,
  - orbit representative order/support/amplitudes/orthonormality against reference transform for `L=10,12,14`,
  - `OrbitBasis.index_of` correctness and error path.

## Diff hygiene checkpoint
Command:
```bash
git diff --check
```
Result: clean (no whitespace or patch-format issues).

## Commit
- SHA: `3a37a79d037f94cf187da02176ce449e1ace22cf`
- Subject: `Add compact constrained-state and orbit basis engine`
- Files:
  - `scripts/turner2018_ed_engine.py`
  - `scripts/tests/test_turner2018_ed_engine.py`

## Self-review
- Verified strict RED before production code (collection failed due missing module).
- Verified GREEN on new focused suite and requested paired regression suite.
- Verified no Task 2 work and no production transform materialization.
- Verified lints for touched files: no diagnostics.

## Concerns / follow-up notes
- Git author identity was not configured in this worktree; commit used explicit per-command `GIT_AUTHOR_*` / `GIT_COMMITTER_*` environment variables matching repository history author, without changing git config.

---

## Task 1 follow-up: L=32 and chunk-boundary coverage expansion

### Added focused tests only
- `canonical_dihedral` correctness at `length=32` using states with set bits at positions `30/31`, compared against an independent scalar enumeration of all rotations and reflections.
- `build_orbit_basis` output invariance across chunk boundaries, including `chunk_size=1` and additional boundary-shifting sizes.
- Reflection-specific standalone state test where reflected-rotation members are not a subset of pure rotations from the original state.

### Test evidence
Command:
```bash
PYTHONPATH=scripts /home/footman/code/quantum.harness/.venv/bin/python -m pytest scripts/tests/test_turner2018_ed_engine.py -q
```

Output:
```text
..........                                                               [100%]
10 passed in 0.36s
```

Command:
```bash
PYTHONPATH=scripts /home/footman/code/quantum.harness/.venv/bin/python -m pytest scripts/tests/test_turner2018_ed_engine.py scripts/tests/test_pxp_ed.py -q
```

Output:
```text
..........................                                               [100%]
26 passed in 0.52s
```

### Self-review
- Scope remained test-only; no production code changed.
- Added an independent scalar dihedral oracle in tests (no use of production canonicalization helpers for expected values).
- Confirmed chunk-boundary robustness includes the strongest edge case `chunk_size=1`.
- Kept Task 5-scale validation out of scope as requested.

### Follow-up commit
- SHA: `98da7254528d67e282b5e90f7cb3b4e8cecb8b9a`
- Subject: `Add L32 and chunk-boundary orbit basis tests`

---

## Task 1 follow-up: site-centered reflection oracle correction

### Issue corrected
- Fixed the independent scalar reflection oracle to match the binding convention used by `scripts/pxp_ed.py`: site-centered mapping `i -> (-i) % L` (not bond-centered `i -> L-1-i`).
- Preserved and completed the in-progress test edits in `scripts/tests/test_turner2018_ed_engine.py`.
- Updated the `L=32` high-bit fixture set so every sampled state has reflected-rotation members beyond pure rotations under the site-centered convention.

### TDD evidence for this fix
RED command:
```bash
PYTHONPATH=scripts /home/footman/code/quantum.harness/.venv/bin/python -m pytest scripts/tests/test_turner2018_ed_engine.py -q
```

RED output:
```text
F..........                                                              [100%]
=================================== FAILURES ===================================
________ test_scalar_reflect_matches_site_centered_reference_convention ________
...
E           assert 13 == 26
E            +  where 13 = _scalar_reflect(176, 8)
E            +  and   26 = _reflect(176, 8)
...
FAILED scripts/tests/test_turner2018_ed_engine.py::test_scalar_reflect_matches_site_centered_reference_convention
1 failed, 10 passed in 0.37s
```

GREEN command:
```bash
PYTHONPATH=scripts /home/footman/code/quantum.harness/.venv/bin/python -m pytest scripts/tests/test_turner2018_ed_engine.py -q
```

GREEN output:
```text
...........                                                              [100%]
11 passed in 0.45s
```

Paired regression command:
```bash
PYTHONPATH=scripts /home/footman/code/quantum.harness/.venv/bin/python -m pytest scripts/tests/test_turner2018_ed_engine.py scripts/tests/test_pxp_ed.py -q
```

Paired regression output:
```text
...........................                                              [100%]
27 passed in 0.50s
```

### Self-review
- Confirmed scalar oracle reflection equals `_reflect` on targeted `L=8` and `L=32` cases.
- Confirmed the `L=32` canonical-dihedral test now uses high-bit fixtures where reflection contributes additional orbit members beyond pure rotations.
- Confirmed standalone nontrivial reflection test still enforces reflected-members contribution and checks full dihedral-member equality.

---

# Task 1 Report: Correct Turner Fig. 3(b)(c)

## Scope
- Implemented the approved paper-faithful Fig. 3(b)(c) correction on
  `independent-pxp-ed`, based on `91290a7`.
- Modified only the requested Fig. 3/server source and focused tests.
- Did not change `turner2018_ed_observables.py`; persisted numerical observable
  arrays remain unchanged.
- Preserved the existing stable-snapshot consumption and transactional
  PNG/NPZ/JSON publication paths.

## TDD evidence

### RED
Tests were written before production edits.

Command:
```bash
.venv/bin/python -m pytest \
  scripts/tests/test_turner2018_fig3.py \
  scripts/tests/test_turner2018_ed_observables.py \
  scripts/tests/test_turner2018_l32_server.py -q
```

Observed expected failures:
```text
13 failed, 153 passed, 34 subtests passed in 23.00s
```

The seven selector tests failed because
`turner2018_fig3.select_fig3_shell_panel_states` did not exist. The independent
renderer test failed because it still used the former panel selection and
styles. Five server subtests failed because stale length, wrong shell count,
wrong role, nonnegative panel-(c) energy, and inconsistent normalization were
still accepted.

### GREEN
Focused command:
```bash
.venv/bin/python -m pytest \
  scripts/tests/test_turner2018_fig3.py \
  scripts/tests/test_turner2018_ed_observables.py \
  scripts/tests/test_turner2018_l32_server.py -q
```

Result:
```text
161 passed, 39 subtests passed in 24.92s
```

Full regression command:
```bash
.venv/bin/python -m pytest scripts/tests -q
```

Result:
```text
640 passed, 11 skipped, 39 subtests passed in 87.98s
```

## Implemented behavior
- Added shared `select_fig3_shell_panel_states(...) -> tuple[dict, dict]`.
- Revalidates the existing one-to-one maximum-projection match and rejects
  empty, duplicate, stale, nonfinite, wrong-shape, and ambiguous inputs.
- Panel (b) is the lowest-energy matched tower state.
- Panel (c) excludes `abs(E) <= 1e-10`, keeps only negative energies, and
  chooses minimum absolute energy with exact-index tie-breaking.
- Both independent and legacy renderers use folded coordinates `0..L/2`,
  black circle/solid exact curves, and red cross/dashed FSA curves.
- Sidecars distinguish `full_fsa_shell_count = L+1` from
  `plotted_folded_shell_count = L/2+1`, and record selection, normalization,
  source hashes, folding, and plot conventions.
- Figures-stage restart validation cross-checks the semantic selection metadata
  against NPZ shell coordinates and weight sums.

## Real L=20 workflow and inspection
Commands:
```bash
.venv/bin/python scripts/turner2018_l32_server.py \
  --stage all --length 20 \
  --output-dir tracks/ed/results/turner-2018/task7-independent/L20 \
  --declared-memory 1G --chunk-columns 32 --rebuild

.venv/bin/python scripts/turner2018_l32_server.py \
  --stage figures --length 20 \
  --output-dir tracks/ed/results/turner-2018/task7-independent/L20 \
  --declared-memory 1G --chunk-columns 32 --rebuild
```

Both commands exited 0; all computational stages and the figures stage
completed.

Selected L=20 states:
- Panel (b), `lowest-matched-scar`: exact index `0`, exact energy
  `-12.07121376222543`; FSA index `0`, FSA energy
  `-12.017010493123097`; match strength `0.9818818789488396`.
- Panel (c), `negative-adjacent-to-zero`: exact index `101`, exact energy
  `-2.672092951313271`; FSA index `4`, FSA energy
  `-2.633527459988426`; match strength `0.7878643145131315`.
- Zero tolerance: `1e-10`.
- Full recurrence shell count: `21`.
- Plotted folded shell count: `11` in each panel.
- Panel (b) exact/FSA sums:
  `0.9819602717783971` / `1.0000000000000004`.
- Panel (c) exact/FSA sums:
  `0.788540937730373` / `1.0`.

Inspected artifacts:
- `tracks/ed/results/turner-2018/task7-independent/L20/figures/fig3_independent_L20.png`
- `tracks/ed/results/turner-2018/task7-independent/L20/figures/fig3_independent_L20.json`
- `tracks/ed/results/turner-2018/task7-independent/L20/figures/fig3_independent_L20.npz`

Visual inspection confirmed 11 black/red points in each shell panel, linear
squared weights, the required titles, and explicit `L=20` labeling. Sidecar
inspection confirmed shell coordinates `[0, ..., 10]`, roles, energies,
indices, sums, and folding metadata.

The L=32 fixture test confirms 17 folded points and 33 full recurrence states;
it does not fabricate a 33-point plotting array.

## Additional verification
```bash
.venv/bin/python -m py_compile \
  scripts/turner2018_fig3.py \
  scripts/turner2018_ed_observables.py \
  scripts/turner2018_l32_server.py \
  scripts/tests/test_turner2018_fig3.py \
  scripts/tests/test_turner2018_ed_observables.py \
  scripts/tests/test_turner2018_l32_server.py
git diff --check
```

Both exited 0. IDE lint inspection reported no errors in touched files.

## Commit
- Implementation commit:
  `29d9aa904f063992c38d2998371ccdc7b7652913`
- Subject: `Correct paper-faithful Fig. 3 shell panels`
- The appended evidence report is committed separately so it can record the
  exact implementation commit without a self-referential hash.

## Concerns
- Official DOI archives are not present in this worktree, so the official L=32
  shell-count/style path is covered by the deterministic fixture test rather
  than a local DOI archive render.
- Final independent L=32 reproduction still requires the independent L=32
  eigensystem, as stated in the approved design.
- Git author configuration is absent; commits use per-command author/committer
  environment matching repository history, without changing git config.

---

## Reviewer-finding remediation

### TDD RED evidence
Tests for all reviewer findings were added before production changes.

Command:
```bash
.venv/bin/python -m pytest \
  scripts/tests/test_turner2018_fig3.py \
  scripts/tests/test_turner2018_l32_server.py \
  scripts/tests/test_turner2018_official.py -q
```

Expected RED result:
```text
6 failed, 109 passed, 4 skipped, 39 subtests passed in 21.31s
```

The failures demonstrated:
- no selected-FSA eigenvalue gap/tolerance evidence;
- arbitrary acceptance of rotated bases in a degenerate selected FSA block;
- mutable selector dictionaries/arrays;
- absent NPZ source arrays for semantic recomputation;
- overlapping panel `(c)` label;
- figures-stage inability to authenticate all selected semantics.

### Implemented remediation
- `Fig3ShellPanelState` is now a frozen, slotted dataclass implementing the
  read-only mapping interface for compatibility. Its three plot arrays are
  defensive copies with `writeable=False`; `to_metadata_dict()` is the explicit
  JSON serialization boundary.
- Selected FSA states must have nearest eigenvalue gap strictly greater than
  `FSA_EIGENVALUE_GAP_TOLERANCE = 1e-10`. Both tolerance and measured nearest
  gap are persisted per panel. Tests inject two different rotations in the
  same exactly degenerate block and verify both fail before an individual
  curve can be selected.
- The independent NPZ now persists the exact selection source arrays:
  `exact_shell_amplitudes`, `fsa_hamiltonian_sector`, and matched exact indices.
- Figures-stage validation checks those arrays against JSON hashes and matched
  source records, recomputes the selector, and requires exact equality for
  exact/FSA indices, energies, match strength, roles, gap evidence, shell
  weights/counts, and normalization.
- A restart test independently corrupts each authenticated field/array,
  refreshes the superficial asset/manifest hashes, and proves the figures
  stage still refuses to skip or accept it.
- Independent renderer expectations now derive selections directly from the
  fixture tower/energies rather than calling the production selector.
- Legacy coverage independently checks roles, energies, exact/FSA weights,
  17 folded points, and black/red marker/line styling.
- Panel `(c)` moved to the upper-right corner, away from its first high-weight
  point, without changing any numerical data.

### GREEN evidence
Focused command:
```bash
.venv/bin/python -m pytest \
  scripts/tests/test_turner2018_fig3.py \
  scripts/tests/test_turner2018_l32_server.py \
  scripts/tests/test_turner2018_official.py -q
```

Result:
```text
114 passed, 4 skipped, 50 subtests passed in 23.63s
```

Full regression:
```bash
.venv/bin/python -m pytest scripts/tests -q
```

Result:
```text
642 passed, 11 skipped, 50 subtests passed in 92.08s
```

`py_compile`, `git diff --check`, and IDE lints all passed with no output or
diagnostics.

### Real L=20 rerender and inspection
The current-fingerprint L=20 computational and figures stages were rebuilt;
both commands exited 0.

Image:
`tracks/ed/results/turner-2018/task7-independent/L20/figures/fig3_independent_L20.png`

Visual inspection confirmed:
- 11 folded black/red points in each panel;
- unchanged roles, energies, and numerical curves;
- panel `(c)` label at upper right with no overlap on the first FSA point.

Exact sidecar evidence:
- Panel (b): exact index `0`, energy `-12.07121376222543`; FSA index
  `0`, energy `-12.017010493123097`; match `0.9818818789488396`;
  nearest FSA gap `1.993766660882546`.
- Panel (c): exact index `101`, energy `-2.672092951313271`; FSA index
  `4`, energy `-2.633527459988426`; match `0.7878643145131315`;
  nearest FSA gap `2.5886606403076566`.
- FSA gap tolerance: `1e-10` for both.
- Full/plotted shell counts: `21` / `11`.
- Panel (b) exact/FSA sums:
  `0.9819602717783971` / `1.0000000000000004`.
- Panel (c) exact/FSA sums:
  `0.788540937730373` / `1.0`.
- Persisted selection source shapes:
  exact amplitudes `[11, 455]`, FSA Hamiltonian `[11, 11]`, match indices `[11]`.

### Commit
- Reviewer-fix implementation:
  `f11f1337e3a8565b8cec62c1ff962e1389a4d850`
- Subject: `Harden Fig. 3 selection evidence`
- This appended evidence is committed separately to record the exact
  implementation hash.

### Remaining concerns
- Official DOI archives remain unavailable in this worktree; official tests
  requiring them are skipped, while deterministic official-path fixtures pass.
- Independent L=32 rendering still requires the independent L=32 eigensystem.

---

## Final re-review remediation

### Correct RED evidence
The two final review behaviors were tested before production changes.

Immutable result RED:
```bash
.venv/bin/python -m pytest \
  scripts/tests/test_turner2018_fig3.py::test_shell_panel_selector_result_is_structurally_immutable -q
```
Failed because `panel_b.shell` was an owned NumPy array rather than a tuple,
so `isinstance(stored, tuple)` was false.

Coordinated rewrite RED:
```bash
.venv/bin/python -m pytest \
  scripts/tests/test_turner2018_l32_server.py::TurnerRestartWorkflowTests::test_figures_stage_rejects_coordinated_fig3_rewrite_against_upstream -q
```
Result:
```text
1 failed in 1.60s
AssertionError: RuntimeError not raised
```

The adversarial test shifted the copied eigensystem energies, rescaled copied
exact shell amplitudes, shifted the copied FSA Hamiltonian, recomputed every
selected state/weight/match/normalization field, and refreshed the Fig. 3
NPZ/JSON asset hashes, combined figures manifest hashes, and figures-stage
artifact hash. The unchanged validated upstream artifacts remained intact.
Before the fix this coordinated rewrite was accepted and the stage skipped.

### Implementation
- `Fig3ShellPanelState.shell`, `.exact_weights`, and `.fsa_weights` are now
  immutable tuples of Python scalars. The frozen dataclass normalizes all
  constructor inputs to tuples.
- Tests prove tuple item assignment is impossible, no stored object exposes
  `setflags`, and a mutable NumPy conversion can be changed without affecting
  the stored result.
- Both independent and official render paths create renderer-local NumPy
  arrays from those tuples.
- Fig. 3 acceptance now reopens the Task 6 result through the existing stable
  validated snapshot chain. It validates current stage manifests/artifact
  hashes and stable open handles, reads eigensystem energies plus persisted
  observables, and accesses eigenvector metadata only.
- The complete sidecar `source_hashes` map must exactly equal the current
  upstream snapshot hashes.
- Sidecar energies, exact shell amplitudes, FSA Hamiltonian, and matched tower
  must exactly equal the current upstream arrays.
- Selection roles, indices, energies, gaps, weights, matches, shell counts,
  and normalization are recomputed from the upstream arrays and compared to
  every NPZ/JSON value.
- The coordinated rewrite test patches HDF5 eigenvector indexing to fail,
  proving semantic restart validation never materializes eigenvectors.

### GREEN and regression evidence
Targeted GREEN:
```text
2 passed in 1.48s
```

Standalone coordinated-attack regression:
```text
1 passed in 1.62s
```

Focused suite:
```bash
.venv/bin/python -m pytest \
  scripts/tests/test_turner2018_fig3.py \
  scripts/tests/test_turner2018_l32_server.py \
  scripts/tests/test_turner2018_official.py -q
```
```text
115 passed, 4 skipped, 50 subtests passed in 25.43s
```

Full regression:
```bash
.venv/bin/python -m pytest scripts/tests -q
```
```text
643 passed, 11 skipped, 50 subtests passed in 97.03s
```

`py_compile`, `git diff --check`, and IDE lint inspection passed without
errors.

### Real L=20 rerender/restart evidence
Current-fingerprint L=20 computational stages and figures were rebuilt.
The immediate restart then reported:
```text
skipped stage=plan
skipped stage=figures
```

Inspected image:
`tracks/ed/results/turner-2018/task7-independent/L20/figures/fig3_independent_L20.png`

Visual inspection confirmed unchanged curves/data and the non-overlapping
panel `(c)` label. Programmatic inspection reported
`source_hashes_match True`, 11 plotted folded shells in each panel, and:
- panel (b) exact/FSA sums:
  `0.9819602717783971` / `1.0000000000000004`;
- panel (c) exact/FSA sums:
  `0.788540937730373` / `1.0`.

Selected indices, energies, matches, and gaps remained:
- panel (b): exact `0`, `-12.07121376222543`; FSA `0`,
  `-12.017010493123097`; match `0.9818818789488396`; gap
  `1.993766660882546`;
- panel (c): exact `101`, `-2.672092951313271`; FSA `4`,
  `-2.633527459988426`; match `0.7878643145131315`; gap
  `2.5886606403076566`.

### Commit
- Final implementation:
  `c26c71f89b40ad1679bb89c394e0877c913623ff`
- Subject: `Bind Fig. 3 evidence upstream`
- This report append is committed separately so the implementation hash can be
  recorded exactly.

### Remaining concerns
- Official DOI archives remain unavailable locally; their integration tests
  are skipped while deterministic official fixtures pass.
- Independent L=32 rendering still requires the independent L=32 eigensystem.

---

# Dzeshell Task 1: Production profile and Slurm wrappers

## Status
DONE

## Scope
- Replaced the stale tracked qdeshell scheduler facts with the verified
  Dzeshell `dzagnormal` node shape, exact GRES type, 8 CPU/GPU ratio, and
  shared project/results roots.
- Added immutable wrappers for `L=22/24/26/28`, `L=30`, and `L=32`, plus one
  common runner for the solver invocation and offline-environment checks.
- Removed the obsolete `turner2018_l32_qdagnormal.sbatch`.
- Preserved the generic provider-neutral SCNet path and its regression tests.
- Kept all SSH connection details and credentials outside tracked files.

## TDD evidence
Tests were edited before production files.

Initial RED:
```bash
.venv/bin/python -m pytest \
  scripts/tests/test_cluster_profile.py \
  scripts/tests/test_turner2018_l32_server.py -q
```
```text
8 failed, 70 passed, 50 subtests passed in 4.49s
```
The failures showed the stale shared path/partition/profile values, absent
Dzeshell wrappers/common runner, and stale submission documentation.

Stale-wrapper RED:
```bash
.venv/bin/python -m pytest \
  scripts/tests/test_turner2018_l32_server.py::TurnerL32SlurmTests::test_dzeshell_wrappers_have_immutable_length_resource_classes -q
```
```text
1 failed in 0.31s
AssertionError: True is not false
```
This proved the old qdagnormal wrapper still existed before its removal.

Focused GREEN:
```bash
.venv/bin/python -m pytest \
  scripts/tests/test_cluster_profile.py \
  scripts/tests/test_turner2018_l32_server.py -q
```
```text
75 passed, 56 subtests passed in 4.21s
```

## Verification
Full regression:
```bash
.venv/bin/python -m pytest scripts/tests -q
```
```text
646 passed, 11 skipped, 56 subtests passed in 101.69s
```

Shell syntax, Python compile, and diff hygiene:
```bash
bash -n \
  scripts/turner2018_dzeshell_run.sh \
  scripts/turner2018_dzeshell_l22_28.sbatch \
  scripts/turner2018_dzeshell_l30.sbatch \
  scripts/turner2018_dzeshell_l32.sbatch \
  scripts/turner2018_l32_scnet.sbatch
.venv/bin/python -m py_compile \
  scripts/tests/test_cluster_profile.py \
  scripts/tests/test_turner2018_l32_server.py
git diff --check
```
All exited 0. IDE lint inspection found no diagnostics. Focused secret scans
found no tracked hostname, port, username, key path/private-key material,
password, API key, or access token in the Dzeshell profile and wrappers.

## Implementation commit
- `296c7c2` — `Configure Dzeshell production resource classes`

## Concerns
- No real job was submitted, consistent with the explicit-approval boundary.
- Git author configuration is absent; the commit used per-command
  author/committer values matching repository history without changing Git
  configuration.

---

# Dzeshell Task 1 reviewer-finding remediation

## Status
DONE

## Findings resolved
- Spool-copied wrappers now source the common runner from
  `SLURM_SUBMIT_DIR/scripts/` instead of resolving relative to `BASH_SOURCE`.
  The regression copies a wrapper outside the repository, supplies a reviewed
  submit checkout, executes the copy, and observes that checkout's runner.
- The common runner fixes the production root at
  `/work/share/giggleliu/jiangweiqi`, canonicalizes repo/runtime/results/Python,
  validates an optional offline-image path, rejects traversal and external
  paths, and requires the submit checkout to equal the execution checkout.
- README test-only commands scope `TURNER_LENGTH` per invocation. A fake
  `harness_slurm.sh` executes the documented block from ambient
  `TURNER_LENGTH=99` and records `28`, `30`, and `32`.
- Runtime validation reads Python and exact NumPy/SciPy/h5py versions from
  `turner2018_wheelhouse_manifest.json`; any mismatch makes the runner exit
  under `set -e`.
- Secret checks cover the profile, all wrappers, and the common runner. They
  recursively reject secret-bearing profile keys, detect literal assignments
  to secret/connection-like shell variables, and reject private-key markers,
  local key paths, Windows paths, and credential-bearing URLs.
- Live partition, CPU, memory, GRES, wall-time, and shared-root constants remain
  unchanged. The generic SCNet wrapper remains unchanged.

## Strict TDD evidence
Reviewer tests were written before production edits.

RED command:
```bash
.venv/bin/python -m pytest \
  scripts/tests/test_turner2018_wheelhouse.py \
  scripts/tests/test_turner2018_l32_server.py::TurnerL32SlurmTests -q
```
RED result:
```text
13 failed, 11 passed, 8 subtests passed in 0.68s
```
The failures demonstrated missing manifest runtime checks, spool-relative
runner lookup, leaked README command state, absent canonical path containment,
and missing traversal/export rejection.

Focused GREEN:
```bash
.venv/bin/python -m pytest \
  scripts/tests/test_turner2018_wheelhouse.py \
  scripts/tests/test_turner2018_l32_server.py::TurnerL32SlurmTests -q
```
```text
16 passed, 16 subtests passed in 0.61s
```

Final focused security/profile/runtime regression:
```bash
.venv/bin/python -m pytest \
  scripts/tests/test_turner2018_wheelhouse.py \
  scripts/tests/test_cluster_profile.py \
  scripts/tests/test_turner2018_l32_server.py::TurnerL32SlurmTests -q
```
```text
40 passed, 16 subtests passed in 0.53s
```

## Full verification
```bash
.venv/bin/python -m pytest scripts/tests -q
```
```text
652 passed, 11 skipped, 66 subtests passed in 94.29s
```

`bash -n` covered all Dzeshell wrappers, the common runner, and the unchanged
SCNet wrapper. `py_compile`, `git diff --check`, and IDE lint inspection all
completed without errors.

## Implementation commit
- `6c54bf9` — `Harden Dzeshell production launch path`

## Concerns
- No real Slurm job was submitted. The spool behavior is simulated locally,
  while remote scheduler submission remains behind the explicit-approval gate.
- Canonicalization covers existing symlink components and lexical traversal;
  no adversarial concurrent filesystem mutation was attempted.
- Git author configuration remains absent, so commits use per-command identity
  values matching repository history without changing Git configuration.

---

# Dzeshell final review: pre-source submit validation

## Status
DONE

## Fix
- Every Dzeshell resource wrapper now performs its own validation before
  sourcing any code.
- The wrapper rejects an unset or nonexistent `SLURM_SUBMIT_DIR`, canonicalizes
  the fixed shared root and submit directory with trusted system `realpath -e`,
  rejects canonical paths outside `/work/share/giggleliu/jiangweiqi`, and
  canonicalizes the exact
  `scripts/turner2018_dzeshell_run.sh` target.
- A symlinked runner or `scripts/` escape is rejected because the canonical
  runner must exactly equal the canonical checkout-relative path and be a
  regular file.
- No validation helper is sourced or executed. Only shell builtins and
  `realpath` run before the canonical common runner is sourced.

## Strict TDD evidence
Tests were changed before wrapper production edits.

RED:
```bash
.venv/bin/python -m pytest \
  scripts/tests/test_turner2018_l32_server.py::TurnerL32SlurmTests -q
```
```text
5 failed, 11 passed, 13 subtests passed in 0.47s
```
The attacker fixture executed the external runner and returned zero before the
fix. Static checks also showed that all three wrappers sourced the unvalidated
environment path, and unset/nonexistent paths did not use the required
fail-closed exit contract.

GREEN:
```bash
.venv/bin/python -m pytest \
  scripts/tests/test_turner2018_l32_server.py::TurnerL32SlurmTests -q
```
```text
13 passed, 16 subtests passed in 0.36s
```

The focused tests prove:
- a copied spool wrapper executes the exact canonical runner for a valid
  synthetic shared-root checkout;
- an attacker runner outside the production root cannot create its sentinel;
- unset and nonexistent submit directories are rejected;
- a submit-directory symlink from inside the allowed root to an attacker
  checkout is rejected without creating its sentinel;
- each wrapper validates and canonicalizes before its only `source`.

## Full verification
```bash
.venv/bin/python -m pytest scripts/tests -q
```
```text
654 passed, 11 skipped, 66 subtests passed in 94.23s
```

`bash -n` covered all three Dzeshell wrappers, the common runner, and the
unchanged SCNet wrapper. `py_compile`, `git diff --check`, and IDE lints all
completed without errors.

## Implementation commit
- `102c4b4` — `Validate Dzeshell submit path before sourcing`

## Concerns
- `/work/share/giggleliu/jiangweiqi` is not mounted on this local host. The
  valid and symlink-escape execution cases therefore substitute a temporary
  fixed root into a copied wrapper while separately asserting the production
  literal in every tracked wrapper.
- No real Slurm submission was made; the explicit-approval gate remains intact.
