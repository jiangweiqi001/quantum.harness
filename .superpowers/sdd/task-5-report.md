# Task 5 Report: Real local validation gate for L=10–20

## Status
- PASS.
- Branch: `independent-pxp-ed`.
- Implementation commit: `dec81d6` (`Implement independent local ED validation gate`).
- No push was performed.
- Task 6 cluster orchestration and offline-environment work were not changed.

## Scope and Files
- Replaced the `analyze_spectrum` self-comparison in `scripts/turner2018_l32_server.py`.
- Added a real independent-engine versus existing `pxp_ed`/`turner2018_fig3` oracle comparison.
- Added the all-length `--validate-local` fail-closed CLI and atomic `validation/local-equivalence.json` summary.
- Added perturbation and CLI gate regressions in `scripts/tests/test_turner2018_l32_server.py`.
- Added this report at `.superpowers/sdd/task-5-report.md`.

## Design
Each length constructs exactly one candidate and one oracle:
- Candidate: `build_orbit_basis` and `assemble_reduced_hamiltonian`, followed by `solve_full_eigensystem` and Task 4 `compute_observables` using the supplied eigensystem.
- Oracle: existing `pxp_ed` constrained basis, full sparse PXP Hamiltonian, and `symmetry_basis_k0_inversion_even`, plus `turner2018_fig3.fsa_basis`.
- Degenerate eigenspaces are compared with Task 4 `compare_degenerate_invariants`.
- PR2 differences are evaluated only for isolated one-dimensional eigenspaces. No arbitrary eigenvector comparison is made inside degenerate groups.

Every metric has an explicit `passed` field. The summary records invocation, Git revision, Python/platform/package provenance, candidate/reference engine descriptions, and SHA-256 hashes of all source modules used by the gate. Publication uses the existing sibling `.partial` plus atomic rename mechanism.

## Strict TDD Evidence

### RED
Tests were changed before production code. Command:
```bash
PYTHONPATH=scripts /home/footman/code/quantum.harness/.venv/bin/python -m pytest scripts/tests/test_turner2018_l32_server.py -q
```

Observed result:
```text
FAILED ...::test_small_l_gate_detects_perturbed_independent_matrix
FAILED ...::test_validate_local_cli_runs_every_requested_even_length
FAILED ...::test_validate_local_cli_writes_failed_atomic_gate_and_returns_nonzero
3 failed, 7 passed in 0.49s
```

The failures were expected: the server module exposed no independent matrix assembly hook, `--validate-local` did not exist, and `--stage` was mandatory.

### GREEN
After the minimal implementation:
```bash
PYTHONPATH=scripts /home/footman/code/quantum.harness/.venv/bin/python -m pytest scripts/tests/test_turner2018_l32_server.py -q
```

```text
..........                                                               [100%]
10 passed in 1.05s
```

The perturbation regression monkeypatches the independent assembler, changes one symmetric matrix pair by `1e-6`, and verifies that `validate_small_l(10)` returns `passed=false` with the reduced-matrix metric failing its `1e-11` threshold.

## All-Length Local Gate
Final command, run after the implementation commit so the artifact provenance names `dec81d6`:
```bash
/usr/bin/time -v env PYTHONPATH=scripts \
  /home/footman/code/quantum.harness/.venv/bin/python \
  scripts/turner2018_l32_server.py \
  --validate-local 10 12 14 16 18 20 \
  --output-dir tracks/ed/results/turner-2018/independent-ed-validation
```

Output:
```text
tracks/ed/results/turner-2018/independent-ed-validation/validation/local-equivalence.json
Elapsed (wall clock) time: 0:02.89
Maximum resident set size (kbytes): 139984
Exit status: 0
```

The JSON was read directly after execution. It contains `status="passed"`, `passed=true`, all six requested lengths, exact dimensions, explicit per-metric pass/fail, source hashes, and provenance. No `.partial` file remained.

## Per-Length Metrics
Metric order below is matrix, spectrum, total Z2, projector diagonal, subspace sine, FSA beta, projected FSA shell, isolated PR2.

- L=10; dimensions 123/14; `4.441e-16`, `3.553e-15`, `4.805e-16`, `1.790e-15`, `7.301e-15`, `8.882e-16`, `2.220e-16`, `8.882e-16`.
- L=12; dimensions 322/26; `1.332e-15`, `5.329e-15`, `1.943e-16`, `3.081e-15`, `1.081e-14`, `2.220e-15`, `2.220e-16`, `1.277e-15`.
- L=14; dimensions 843/49; `1.332e-15`, `1.421e-14`, `8.153e-16`, `3.574e-15`, `2.944e-14`, `1.776e-15`, `2.220e-16`, `1.138e-15`.
- L=16; dimensions 2207/99; `6.661e-16`, `7.105e-15`, `8.465e-16`, `7.827e-15`, `6.750e-14`, `1.243e-14`, `1.110e-16`, `1.700e-15`.
- L=18; dimensions 5778/209; `8.882e-16`, `5.329e-15`, `4.944e-16`, `4.398e-14`, `5.627e-13`, `2.220e-14`, `1.665e-16`, `6.391e-15`.
- L=20; dimensions 15127/455; `1.776e-15`, `8.882e-15`, `3.886e-16`, `2.939e-14`, `6.089e-13`, `2.665e-15`, `3.886e-16`, `1.263e-14`.

## Validation Maxima
- Reduced matrix: `1.7763568394002505e-15` (`<=1e-11`).
- Complete spectrum: `1.4210854715202004e-14` (`<=1e-10`).
- Degenerate total Z2 weight: `8.465450562766819e-16` (`<=1e-10`).
- Degenerate projector diagonal: `4.3978709562964013e-14` (`<=1e-10`).
- Degenerate subspace sine: `6.089074432530575e-13` (`<=1e-10`).
- FSA beta: `2.220446049250313e-14` (`<=1e-11`).
- Projected FSA shell: `3.885780586188048e-16` (`<=1e-11`).
- PR2 on isolated eigenspaces: `1.2625317458159202e-14` (`<=1e-10`).

## Required ED Regression Suite
Command:
```bash
PYTHONPATH=scripts /home/footman/code/quantum.harness/.venv/bin/python -m pytest \
  scripts/tests/test_pxp_ed.py \
  scripts/tests/test_turner2018_ed_engine.py \
  scripts/tests/test_turner2018_ed_solver.py \
  scripts/tests/test_turner2018_ed_artifacts.py \
  scripts/tests/test_turner2018_ed_observables.py \
  scripts/tests/test_turner2018_fig3.py \
  scripts/tests/test_turner2018_fig4.py \
  scripts/tests/test_turner2018_l32_server.py -q
```

Output:
```text
163 passed, 6 skipped in 8.83s
```

The six skips are pre-existing optional official-data tests.

## Review Evidence
- `git diff --check`: clean.
- IDE diagnostics for both edited Python files: no linter errors.
- The final JSON was inspected rather than trusting only process status.
- Exact full/sector dimensions were 123/14, 322/26, 843/49, 2207/99, 5778/209, and 15127/455.
- CLI failure behavior is covered: one failed length produces an atomic failed summary and return code 1.
- Per-length unexpected exceptions are converted into an explicit failed execution metric so the complete multi-length gate is still published.

## Self-Review
- The old false-positive path is gone; `analyze_spectrum` is not called by local validation.
- Candidate and oracle matrices are assembled by separate code paths.
- Candidate observables consume the Task 3 eigensystem and do not re-solve internally.
- Matrix, complete spectrum, degeneracy-safe invariants, FSA, isolated PR2, and dimensions all use their binding thresholds.
- Shell vectors permit only a physically irrelevant whole-shell sign alignment before comparison.
- The command returns nonzero if any result fails and writes one summary after all requested lengths.
- No Task 6 orchestration, environment packaging, or cluster behavior was added.

## Concerns
- The reference oracle intentionally densifies only the reduced matrices. This is appropriate for the bounded local L=10–20 gate but is not intended for L=32.
- BLAS used approximately 17 CPU cores in the resource run (`1706%` CPU), so elapsed time will vary with thread configuration; peak RSS remained about 136.7 MiB.
- The generated validation artifact lives under the ignored results tree and is therefore not part of the implementation commit; its source hashes and committed Git revision make the inspected local artifact reproducible.
