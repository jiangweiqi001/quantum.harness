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
