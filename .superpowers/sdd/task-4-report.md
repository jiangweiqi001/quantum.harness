# Task 4 Report: Independent observables and invariant validation

## Scope
- Added `scripts/turner2018_ed_observables.py` with:
  - `project_product_state`
  - `stream_pr2`
  - `stream_fsa_shells`
  - `compute_observables`
  - `compare_degenerate_invariants`
- Added `scripts/tests/test_turner2018_ed_observables.py`.

## TDD Evidence

### RED (before implementation)
Command:
```bash
PYTHONPATH=scripts /home/footman/code/quantum.harness/.venv/bin/python -m pytest scripts/tests/test_turner2018_ed_observables.py -q
```
Output:
```text
ImportError while importing test module 'scripts/tests/test_turner2018_ed_observables.py'
ModuleNotFoundError: No module named 'turner2018_ed_observables'
```

### GREEN (after minimal implementation)
Command:
```bash
PYTHONPATH=scripts /home/footman/code/quantum.harness/.venv/bin/python -m pytest scripts/tests/test_turner2018_ed_observables.py scripts/tests/test_turner2018_fig3.py -q
```
Output:
```text
...........                                                              [100%]
11 passed in 0.81s
```

## Requirements Coverage
- Z2 single-pattern norm check added:
  - `test_z2_sector_projection_has_half_weight`
- Streamed PR2 definition check added:
  - `test_streamed_pr2_matches_definition`
- FSA oracle-equivalence checks at `L=10,12,14,16` added:
  - compares full `beta` chain to `turner2018_fig3.fsa_basis`
  - compares each projected shell (allowing explicit global shell sign)
  - compares projected reduced FSA Hamiltonian
- Degeneracy-safe invariant comparison added:
  - `test_degenerate_invariant_comparison_uses_projectors_not_vector_pr2`
  - validates projector-diagonal and total Z2 invariants under basis rotation within a degenerate subspace
  - avoids claiming vector-level PR2 invariance inside degenerate groups

## Streaming/Memory-Shape Evidence
Captured with `L=16`:
```text
z2_norm 0.4999999999999999
full_dimension 2207
full_buffer_shape (2207,)
projected_shells_shape (9, 99)
beta_shape (16,)
```
Interpretation:
- Full-basis propagation uses two `float64` vectors of shape `(full_dimension,)`.
- No `(L+1) x full_dimension` allocation is used.
- Stored projected shells are only `L/2 + 1` rows in reduced dimension.

## Verification Commands
Focused Task 4 + required regression:
```bash
PYTHONPATH=scripts /home/footman/code/quantum.harness/.venv/bin/python -m pytest scripts/tests/test_turner2018_ed_observables.py scripts/tests/test_turner2018_fig3.py -q
```
```text
11 passed in 0.81s
```

Review checkpoint:
```bash
git diff --check
```
Output: clean.

Lints:
```text
No linter errors found for edited files.
```

## Files Changed
- `scripts/turner2018_ed_observables.py`
- `scripts/tests/test_turner2018_ed_observables.py`
- `.superpowers/sdd/task-4-report.md`

## Commit
- SHA: `70e731a`
- Subject: `Implement streamed ED observables and invariant validation`

## Self-Review
- `stream_fsa_shells` performs forward propagation on the full constrained basis only, with distance-increasing edges defined relative to one chosen Z2 pattern.
- Projection follows the required orbit-amplitude rule:
  - sum over unique orbit members divided by `sqrt(orbit_size)`.
- FSA comparison tests explicitly enforce `L=10..16` oracle agreement for `beta`, projected shells, and projected FSA Hamiltonian (with shell-wise global-sign freedom).
- Degenerate-group checks compare only basis-invariant diagnostics (projector diagonals and total Z2 weight), while PR2 is compared only on isolated levels.

## Concerns
- The current streamed implementation prioritizes correctness and invariant behavior over precomputed-edge performance; for much larger `L`, precomputing constrained forward destinations could reduce wall time while keeping the same memory shape guarantees.

## Review Fixes (2026-07-27)

### Finding 1: bounded vectorized FSA propagation and projection

RED:
```bash
PYTHONPATH=scripts /home/footman/code/quantum.harness/.venv/bin/python -m pytest scripts/tests/test_turner2018_ed_observables.py::test_streamed_fsa_is_invariant_to_small_chunks scripts/tests/test_turner2018_ed_observables.py::test_streamed_fsa_has_no_scalar_per_state_propagation_loop -q
```
```text
FF                                                                       [100%]
FAILED ...::test_streamed_fsa_is_invariant_to_small_chunks - TypeError: stream_fsa_shells() got an unexpected keyword argument 'chunk_size'
FAILED ...::test_streamed_fsa_has_no_scalar_per_state_propagation_loop - AssertionError: FSA propagation must not loop over individual states
2 failed in 0.49s
```

Additional projection RED:
```bash
PYTHONPATH=scripts /home/footman/code/quantum.harness/.venv/bin/python -m pytest scripts/tests/test_turner2018_ed_observables.py::test_projection_canonicalizes_full_basis_in_bounded_chunks -q
```
```text
F                                                                        [100%]
E       assert 1 == 123
FAILED ...::test_projection_canonicalizes_full_basis_in_bounded_chunks
1 failed in 0.40s
```

GREEN:
```bash
PYTHONPATH=scripts /home/footman/code/quantum.harness/.venv/bin/python -m pytest scripts/tests/test_turner2018_ed_observables.py::test_projection_canonicalizes_full_basis_in_bounded_chunks scripts/tests/test_turner2018_ed_observables.py::test_streamed_fsa_has_no_scalar_per_state_propagation_loop -q
```
```text
..                                                                       [100%]
2 passed in 0.38s
```
```bash
PYTHONPATH=scripts /home/footman/code/quantum.harness/.venv/bin/python -m pytest scripts/tests/test_turner2018_ed_observables.py::test_streamed_fsa_is_invariant_to_small_chunks scripts/tests/test_turner2018_ed_observables.py::test_streamed_fsa_matches_full_basis_oracle -q
```
```text
.....                                                                    [100%]
5 passed in 1.28s
```

Implementation now uses a compact chunk-built `uint8[D]` Hamming-distance array, two `float64[D]` shell vectors, and bounded vector temporaries. Every destination lookup is a vector `np.searchsorted`; accumulation uses `np.add.at`. Full-state orbit canonicalization/projection is chunked. No `(L+1)×D` allocation or scalar state propagation loop remains.

### Finding 2: consume Task 3 eigensystems without internal ED

RED:
```bash
PYTHONPATH=scripts /home/footman/code/quantum.harness/.venv/bin/python -m pytest scripts/tests/test_turner2018_ed_observables.py::test_compute_observables_consumes_eigensystem_without_dense_solve scripts/tests/test_turner2018_ed_observables.py::test_compute_observables_validates_dimensions_and_finiteness -q
```
```text
FFFFF                                                                    [100%]
FAILED ...::test_compute_observables_consumes_eigensystem_without_dense_solve - TypeError: compute_observables() got an unexpected keyword argument 'chunk_size'
FAILED ...::test_compute_observables_validates_dimensions_and_finiteness[...] - TypeError: compute_observables() got an unexpected keyword argument 'chunk_size'
5 failed in 0.46s
```

GREEN:
```bash
PYTHONPATH=scripts /home/footman/code/quantum.harness/.venv/bin/python -m pytest scripts/tests/test_turner2018_ed_observables.py::test_compute_observables_consumes_eigensystem_without_dense_solve scripts/tests/test_turner2018_ed_observables.py::test_compute_observables_validates_dimensions_and_finiteness -q
```
```text
.....                                                                    [100%]
5 passed in 0.39s
```

`compute_observables` now requires `energies` and `eigenvectors`, validates basis/matrix/eigensystem dimensions, finiteness, sorting, and orthonormality, and returns overlap, PR2, projected shell, FSA Hamiltonian, beta, and exact-shell-amplitude outputs. It contains no dense conversion, eigensolve, or vacuous self-comparison.

### Finding 3: complete degenerate-subspace invariants

RED:
```bash
PYTHONPATH=scripts /home/footman/code/quantum.harness/.venv/bin/python -m pytest scripts/tests/test_turner2018_ed_observables.py::test_degenerate_invariants_detect_same_diagonal_different_subspace scripts/tests/test_turner2018_ed_observables.py::test_degenerate_invariants_reject_candidate_partition_mismatch scripts/tests/test_turner2018_ed_observables.py::test_degenerate_invariants_reject_invalid_inputs -q
```
```text
FFFFFFFFFF                                                               [100%]
TypeError: compare_degenerate_invariants() got an unexpected keyword argument 'reference_z2_sector_state'
10 failed in 0.55s
```

Additional positive-tolerance RED:
```text
........F                                                                [100%]
FAILED ...::test_degenerate_invariants_reject_invalid_inputs[tolerance-True-tolerance]
1 failed, 8 passed in 0.40s
```

GREEN:
```bash
PYTHONPATH=scripts /home/footman/code/quantum.harness/.venv/bin/python -m pytest scripts/tests/test_turner2018_ed_observables.py::test_degenerate_invariant_comparison_uses_projectors_not_vector_pr2 scripts/tests/test_turner2018_ed_observables.py::test_degenerate_invariants_detect_same_diagonal_different_subspace scripts/tests/test_turner2018_ed_observables.py::test_degenerate_invariants_reject_candidate_partition_mismatch scripts/tests/test_turner2018_ed_observables.py::test_degenerate_invariants_reject_invalid_inputs -q
```
```text
...........                                                              [100%]
11 passed in 0.54s
```
```text
.........                                                                [100%]
9 passed in 0.36s
```

Validation now rejects malformed/nonfinite/unsorted energies, nonpositive or boolean tolerances, malformed/nonfinite/nonorthonormal vectors, mismatched Z2 states, and changed degeneracy boundaries. Complete subspaces are compared through singular values of `V_refᴴ V_candidate`; projector diagonals remain diagnostic only.

### Finding 4: complex PR2

RED:
```text
F                                                                        [100%]
E       Mismatched elements: 2 / 2 (100%)
E        ACTUAL: array([0.25 , 0.125])
E        DESIRED: array([0.5, 0.5])
ComplexWarning: Casting complex values to real discards the imaginary part
1 failed, 1 warning in 0.85s
```

GREEN:
```text
.                                                                        [100%]
1 passed in 0.36s
```

### Finding 5: strict `max_shell`

RED after the chunk API was present:
```text
.FFF.                                                                    [100%]
FAILED ...::test_streamed_fsa_rejects_invalid_max_shell[13] - Failed: DID NOT RAISE
FAILED ...::test_streamed_fsa_rejects_invalid_max_shell[1.5] - Failed: DID NOT RAISE
FAILED ...::test_streamed_fsa_rejects_invalid_max_shell[True] - Failed: DID NOT RAISE
3 failed, 2 passed in 0.54s
```

GREEN:
```text
.....                                                                    [100%]
5 passed in 0.37s
```

### Finding 6: shell-sign Hamiltonian similarity

RED:
```text
F                                                                        [100%]
NameError: name '_apply_shell_sign_similarity' is not defined
FAILED ...::test_shell_sign_similarity_transform_changes_signed_offdiagonals
1 failed in 0.40s
```

GREEN:
```text
.....                                                                    [100%]
5 passed in 0.81s
```

The oracle Hamiltonian is now transformed as `S H S`, using the same shell signs applied when aligning shell vectors.

### Finding 7: expanded coverage

The focused suite now covers compute-observables behavior and invalid inputs, no internal dense solve, same-diagonal/different-subspace detection, candidate partition mismatch, complex PR2, chunk invariance, full-basis bounded projection, and a structural scalar-search guard.

### Final verification

Final combined Task 4, Fig. 3, engine, solver, and artifact regressions:
```bash
PYTHONPATH=scripts /home/footman/code/quantum.harness/.venv/bin/python -m pytest scripts/tests/test_turner2018_ed_observables.py scripts/tests/test_turner2018_fig3.py scripts/tests/test_turner2018_ed_engine.py scripts/tests/test_turner2018_ed_solver.py scripts/tests/test_turner2018_ed_artifacts.py -q
```
```text
........................................................................ [ 80%]
..................                                                       [100%]
90 passed in 8.36s
```

L=24 benchmark and L=32 operation/memory shape:
```text
L24 full_dimension=103682 reduced_dimension=2359
L24 build_seconds=0.308262 fsa_seconds=6.283034
chunk_size=65536 max_source_chunk=8145 max_projection_chunk=65536 distance_dtype=uint8
L32 full_dimension=4870847 core_state_vector_distance_bytes=121771175 core_mib=116.130
L32_forbidden_(L+1)xD_float64_bytes=1285903608
```

The 116.130 MiB L=32 core is the compact states array, current/next vectors, and `uint8` distances; all additional propagation/projection arrays are bounded by `chunk_size`. This replaces the forbidden 1.286 GB shell matrix and supports L=32 feasibility.

### Review-fix commit
- SHA: `1571647`
- Subject: `Fix streamed ED observables review findings`
- Local only; not pushed.

## Production-blocker fixes (2026-07-27)

### RED evidence

Focused blocker command:
```bash
PYTHONPATH=scripts /home/footman/code/quantum.harness/.venv/bin/python -m pytest scripts/tests/test_turner2018_ed_observables.py::test_validation_paths_have_no_full_square_gram_or_identity scripts/tests/test_turner2018_ed_observables.py::test_degenerate_rotated_subspace_residual_is_stable scripts/tests/test_turner2018_ed_observables.py::test_compute_observables_rejects_nonhermitian_matrix scripts/tests/test_turner2018_ed_observables.py::test_compute_observables_rejects_orthonormal_wrong_eigenpairs scripts/tests/test_turner2018_ed_observables.py::test_state_inputs_are_validated_before_uint64_conversion scripts/tests/test_turner2018_ed_observables.py::test_streamed_pr2_rejects_invalid_chunk_columns scripts/tests/test_turner2018_ed_solver.py::test_dense_solver_validation_has_no_full_square_gram_or_identity scripts/tests/test_turner2018_ed_solver.py::test_dense_solver_rejects_invalid_validation_chunk_columns -q
```
```text
FFFFFFFF..FF.F.....                                                      [100%]
11 failed, 8 passed in 0.63s
```

The failures directly demonstrated both full-Gram structural paths, the missing bounded-validation API, unstable subspace handling, accepted non-Hermitian/wrong eigenpairs, uncontrolled uint64 conversion, and permissive scalar validation.

### GREEN evidence and implementation

The focused blocker set after implementation:
```text
......................                                                   [100%]
22 passed in 0.58s
```

`turner2018_ed_validation.py` is now shared by Tasks 3 and 4. It:
- validates eigenvector finiteness and every column norm in `D×chunk_columns` blocks;
- checks all eigenpair residuals in the same bounded shape;
- checks sparse/dense Hermiticity in bounded row/column strips;
- checks deterministic bounded off-diagonal orthogonality samples without `VᴴV`;
- rejects boolean, non-integral, zero, and negative chunk/sample counts;
- validates states as non-boolean integers in `[0, 2^64−1]` before conversion.

`compare_degenerate_invariants` preserves candidate/reference energy and degeneracy-partition checks, total Z2 weight, projector-diagonal diagnostics, isolated-level complex PR2, and same-diagonal/different-subspace rejection. Its principal-angle calculation was replaced by stable two-way residuals, `C−R(RᴴC)` and `R−C(CᴴR)`, with candidate/reference columns chunked. It does not form a full projector or unbounded `k×k` overlap. Validation chunk and sample counts are exposed in diagnostics; `compute_observables` exposes all-column counts and maximum norm, sampled-overlap, Hermiticity, and residual errors in `validation_metadata`.

Task 3 `solve_full_eigensystem` now uses the same bounded all-column residual/norm and sampled-orthogonality validation. Structural regressions inspect the solver, observables, and shared helper and reject full identity/self-Gram paths.

### Final requested regressions

Command:
```bash
PYTHONPATH=scripts /home/footman/code/quantum.harness/.venv/bin/python -m pytest scripts/tests/test_turner2018_ed_observables.py scripts/tests/test_turner2018_fig3.py scripts/tests/test_turner2018_ed_engine.py scripts/tests/test_turner2018_ed_solver.py scripts/tests/test_turner2018_ed_artifacts.py -q
```
```text
........................................................................ [ 61%]
..............................................                           [100%]
118 passed in 8.46s
```

This includes exact `L=10,12,14,16` FSA oracle agreement, shell-sign Hamiltonian similarity, strict `max_shell`, complex PR2, degeneracy partition checks, Task 3 solver, engine, and artifact regressions.

### Benchmark and production memory shape

Measured `L=16`, with `chunk_columns=7`, `orthogonality_samples=257`:
```text
L16 reduced_dimension=99 compute_seconds=0.057348
L16 validation_metadata={'chunk_columns': 7, 'finite_columns_checked': 99, 'norm_columns_checked': 99, 'residual_columns_checked': 99, 'orthogonality_sample_count': 257, 'max_norm_error': 2.9976021664879227e-15, 'max_sample_overlap': 1.447518320485397e-15, 'max_hermiticity_error': 0.0, 'max_eigenpair_residual': 3.9968028886505635e-15}
L16 identity_subspace_residual=3.009e-15
```

For production `L=32`, reduced dimension `D=77,436`:
```text
forbidden Gram/identity shape=(77436,77436), float64 bytes=47970672768
bounded validation block shape=(77436,256), float64 bytes=158588928
default orthogonality samples=4096, sample batch columns=256
```

Thus neither Task 3 nor Task 4 adds the hidden 47.971 GB Gram/identity/difference array. Residual, orthogonality-sample, and subspace calculations can have several live `D×chunk_columns` temporaries, but their dimensions are explicitly bounded and exposed. The inherent dense `L=32` eigenvector output remains approximately 47.971 GB and is already covered by Task 3 resource gating.

### Remaining concern

Orthogonality validation is intentionally not exhaustive over all `D(D−1)/2` column pairs: it validates every norm and a deterministic bounded sample, as required. The sample count is configurable and reported, so production runs can increase it without changing the bounded-column memory shape.

## Final pair-sampling validation fixes (2026-07-27)

### RED evidence

Command:
```bash
PYTHONPATH=scripts /home/footman/code/quantum.harness/.venv/bin/python -m pytest scripts/tests/test_turner2018_ed_observables.py::test_default_l32_pair_sampling_spans_early_middle_and_late_spectrum scripts/tests/test_turner2018_ed_observables.py::test_default_sampling_detects_nonorthogonal_late_spectrum_pair scripts/tests/test_turner2018_ed_observables.py::test_pair_sampling_stays_lazy_and_batch_bounded_for_huge_sample_count scripts/tests/test_turner2018_ed_observables.py::test_compute_observables_consumes_eigensystem_without_dense_solve -q
```
```text
FFFF                                                                     [100%]
3 failures: AttributeError: module 'turner2018_ed_validation' has no attribute 'sample_pair_batches'
1 failure: KeyError: 'orthogonality_sampling_policy'
4 failed in 0.86s
```

The failures established that pair generation had no lazy batch interface and that validation metadata did not state its sampling and pair-memory policy.

### GREEN implementation and focused verification

`sample_pair_batches` now yields at most `batch_size` pairs at a time. Validation sets that batch size from `chunk_columns`, so its only pair metadata is two `int64[pair_batch_size]` arrays. It never stores a sample-sized set or sample-sized/full-pair arrays. This remains true for every accepted `orthogonality_samples <= D(D-1)/2`; exhaustive requests cost time but not pair-metadata growth.

The exact deterministic policy is:
- when the requested/accepted sample count is at most `D-1`, first indices are midpoint-stratified over canonical first-index positions `[0,D-1)`, and partners are selected from the valid upper-index interval by deterministic SplitMix64 slots;
- for larger sample counts, midpoint-stratified ranks from the complete lexicographic canonical-pair space are unranked one at a time;
- both regimes produce distinct canonical pairs with `first < partner`, and the all-pairs request visits every pair once.

Focused command:
```bash
PYTHONPATH=scripts /home/footman/code/quantum.harness/.venv/bin/python -m pytest scripts/tests/test_turner2018_ed_observables.py scripts/tests/test_turner2018_ed_solver.py -q
```
```text
.......................................................................  [100%]
71 passed in 1.66s
```

### Sampling and memory evidence

Behavior probe:
```text
L32 first-index thirds=[1365, 1366, 1365] min=9 max=77425 samples=4096
huge-request first_batch=23 peak_bytes=1260 unequal=True
exhaustive-small-dimensions=2..64 unique-complete
```

The L32-shaped default sample therefore reaches early, middle, and late first-index regions rather than concentrating near zero. A regression duplicates a sampled late-spectrum pair while preserving both column norms and confirms orthogonality validation rejects it. The huge request is `499,999,500,000` pairs for `D=1,000,000`; consuming its first batch creates only 23 pairs and measured 1,260 bytes of traced Python allocation.

`validation_metadata` and degenerate-invariant diagnostics now report:
```text
orthogonality_sampling_policy =
  midpoint-stratified first indices with SplitMix64 partner slots when
  samples <= D-1; otherwise midpoint-stratified canonical pair ranks
orthogonality_pair_batch_size = min(chunk_columns, accepted samples)
orthogonality_pair_metadata_memory =
  bounded by two int64 arrays of pair_batch_size
```

This supersedes the earlier incomplete wording “deterministic bounded off-diagonal orthogonality samples”: the previous implementation was deterministic but its low-index prefix was biased and its set/array metadata was not bounded by validation chunk size.

### Complete relevant regression

Command:
```bash
PYTHONPATH=scripts /home/footman/code/quantum.harness/.venv/bin/python -m pytest scripts/tests/test_turner2018_ed_observables.py scripts/tests/test_turner2018_fig3.py scripts/tests/test_turner2018_ed_engine.py scripts/tests/test_turner2018_ed_solver.py scripts/tests/test_turner2018_ed_artifacts.py -q
```
```text
........................................................................ [ 59%]
.................................................                        [100%]
121 passed in 8.28s
```

This is the prior complete 118-test set plus the three new pair-sampling regressions. It preserves all-column norm/residual validation, stable degenerate-subspace checks, FSA behavior, and Task 3 solver integration.
