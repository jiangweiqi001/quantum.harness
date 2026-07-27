# Task 5 Report: Real local validation gate for L=10–20

## Initial Task 5 Status (superseded below)
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

## Initial All-Length Local Gate
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

## Initial Required ED Regression Suite
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

## Initial Concerns
- The reference oracle intentionally densifies only the reduced matrices. This is appropriate for the bounded local L=10–20 gate but is not intended for L=32.
- BLAS used approximately 17 CPU cores in the resource run (`1706%` CPU), so elapsed time will vary with thread configuration; peak RSS remained about 136.7 MiB.
- The generated validation artifact lives under the ignored results tree and is therefore not part of the implementation commit; its source hashes and committed Git revision make the inspected local artifact reproducible.

## Important-Finding Fixes (2026-07-27)

### Scope
- `--validate-local` now accepts only the exact canonical sequence `10 12 14 16 18 20`.
- Subsets, duplicates, extras, and reordered values are rejected before `run_local_validation`.
- Validation mode rejects `--stage`, `--dry-run`, `--length`, `--chunk-columns`, and `--validate-small-l`; output and validation-specific official-data paths remain available.
- A `status=running, passed=false` summary atomically replaces any prior result before hashing, provenance, or scientific work.
- Hashing, provenance, all per-length work, aggregate construction, and serialization are guarded. Pre-publication failures atomically replace the running marker with a minimal JSON-safe failed summary.
- Final publication failure propagates and leaves the already-published running marker authoritative.
- Initial marker publication failure propagates immediately and is not reported as recovered.
- FSA beta, projected shells, and reduced FSA Hamiltonian now record exact candidate/reference/expected shape metrics. Numerical beta and shell subtraction occurs only after every FSA shape gate passes, preventing NumPy broadcasting from masking corruption.
- Git revision uses `git -C <script repository root> rev-parse HEAD`, independent of caller cwd.

### Strict TDD RED
Tests were written before review-fix production changes.

Command:
```bash
PYTHONPATH=scripts /home/footman/code/quantum.harness/.venv/bin/python -m pytest \
  scripts/tests/test_turner2018_l32_server.py -q
```

Observed:
```text
17 failed, 12 passed in 3.75s
```

The failures covered:
- final-publication marker preservation;
- source-hash and provenance fallback summaries;
- stale prior-pass replacement and initial-marker failure;
- canonical exact-length and execution-mode rejection;
- all three FSA shape metrics and broadcast rejection;
- cwd-independent Git revision.

### GREEN
Same focused command after implementation:
```text
....................                                            [100%]
20 passed, 9 subtests passed in 0.90s
```

The nine subtests are four malformed length sequences and five conflicting execution-mode combinations.

### Fixed all-length gate and resources
Command:
```bash
/usr/bin/time -v env PYTHONPATH=scripts \
  /home/footman/code/quantum.harness/.venv/bin/python \
  scripts/turner2018_l32_server.py \
  --validate-local 10 12 14 16 18 20 \
  --output-dir tracks/ed/results/turner-2018/independent-ed-validation
```

Observed:
```text
Elapsed (wall clock) time: 0:02.95
Maximum resident set size (kbytes): 139708
Percent of CPU this job got: 1714%
Exit status: 0
```

The complete JSON was read directly. It had canonical requested/result order, `status=passed`, `passed=true`, seven source hashes, and every numerical, dimensional, and FSA shape metric passed.

Measured FSA shapes:
- L=10: beta `[10]`, projected shells `[6,14]`, reduced FSA Hamiltonian `[6,6]`.
- L=12: beta `[12]`, projected shells `[7,26]`, reduced FSA Hamiltonian `[7,7]`.
- L=14: beta `[14]`, projected shells `[8,49]`, reduced FSA Hamiltonian `[8,8]`.
- L=16: beta `[16]`, projected shells `[9,99]`, reduced FSA Hamiltonian `[9,9]`.
- L=18: beta `[18]`, projected shells `[10,209]`, reduced FSA Hamiltonian `[10,10]`.
- L=20: beta `[20]`, projected shells `[11,455]`, reduced FSA Hamiltonian `[11,11]`.

Numerical maxima were unchanged:
- reduced matrix `1.7763568394002505e-15`;
- complete spectrum `1.4210854715202004e-14`;
- total Z2 `8.465450562766819e-16`;
- projector diagonal `4.3978709562964013e-14`;
- subspace sine `6.089074432530575e-13`;
- FSA beta `2.220446049250313e-14`;
- projected FSA shell `3.885780586188048e-16`;
- isolated-level PR2 `1.2625317458159202e-14`.

### Full ED regression
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

```text
173 passed, 6 skipped, 9 subtests passed in 9.18s
```

The skips remain the optional official-data tests.

### Failure-injection evidence
- A stale passing summary is observed as `running/false` from the first mocked scientific call.
- Source-hash and Git-provenance exceptions publish minimal `failed/false` summaries containing a JSON-safe error string and failure phase.
- Injecting failure into the second atomic write (final publication) raises and leaves the first `running/false` marker on disk.
- Injecting failure into the first atomic write raises before source hashing; no recovery is claimed.
- Truncating candidate beta to `(L-1,)` yields a failed shape metric and failed numerical metric without subtraction or broadcasting.
- Calling `_git_revision` from an unrelated temporary cwd returns the same repository HEAD as `git -C <repo> rev-parse HEAD`.

### Current concerns
- The local oracle still intentionally densifies only reduced matrices and remains bounded to L=10–20.
- Resource timing depends on multithreaded BLAS availability; this run used about 17 CPU cores and 136.4 MiB peak RSS.
- The results artifact is ignored by Git. It must be regenerated after the review-fix commit so its provenance records the actual committed HEAD; the final handoff verifies that condition.

## Final Important-Gap Fixes (2026-07-27)

### Commit
- SHA: `5b63a43`
- Subject: `Harden local validation invocation provenance`

### Scope
- `ArgumentParser` now sets `allow_abbrev=False`.
- `--stage` and `--validate-local` are members of one structural mutually exclusive group.
- Remaining stage-only raw options are normalized at the first `=` before conflict checking. Validation rejects separated and equals forms, including explicit defaults.
- `--official-data-dir` is stage-only and rejected in local validation mode.
- `_git_revision` requires both `returncode == 0` and nonempty stdout. A real subprocess failure or blank revision raises a controlled `RuntimeError`, and the already-published running marker is replaced by a minimal failed provenance summary.

### RED
Command:
```bash
PYTHONPATH=scripts /home/footman/code/quantum.harness/.venv/bin/python -m pytest \
  scripts/tests/test_turner2018_l32_server.py -q
```

Observed before production changes:
```text
8 failed, 21 passed, 9 subtests passed in 1.11s
```

The eight failures covered:
- accepted `--validate-loc` abbreviation reaching scientific work;
- equals forms for `--stage`, `--length`, `--chunk-columns`, `--validate-small-l`, and `--official-data-dir`;
- a real `CompletedProcess(returncode=128, stdout="")`;
- a nominally successful `CompletedProcess(returncode=0, stdout="")`.

### GREEN
Focused command after implementation and commit:
```text
......................                                   [100%]
22 passed, 16 subtests passed in 0.89s
```

The 16 subtests include prior canonical/conflict cases, five equals-form conflicts, and both non-passing Git result forms.

### HEAD-consistent all-length evidence
Command:
```bash
/usr/bin/time -v env PYTHONPATH=scripts \
  /home/footman/code/quantum.harness/.venv/bin/python \
  scripts/turner2018_l32_server.py \
  --validate-local 10 12 14 16 18 20 \
  --output-dir tracks/ed/results/turner-2018/independent-ed-validation
```

Observed:
```text
Elapsed (wall clock) time: 0:02.87
Maximum resident set size (kbytes): 140060
Percent of CPU this job got: 1709%
Exit status: 0
```

Direct JSON inspection:
```text
revision 5b63a43 status passed hashes 7
```

The canonical six lengths, all dimensional and FSA shape gates, all numerical gates, and every per-metric `passed` field were true. Numerical maxima remained:
- reduced matrix `1.7763568394002505e-15`;
- complete spectrum `1.4210854715202004e-14`;
- total Z2 `8.465450562766819e-16`;
- projector diagonal `4.3978709562964013e-14`;
- subspace sine `6.089074432530575e-13`;
- FSA beta `2.220446049250313e-14`;
- projected FSA shell `3.885780586188048e-16`;
- isolated-level PR2 `1.2625317458159202e-14`.

### Final full ED regression
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

```text
175 passed, 6 skipped, 16 subtests passed in 8.85s
```

### Final concerns
- The oracle remains intentionally limited to local L=10–20 reduced matrices.
- BLAS thread availability affects elapsed time; the final implementation run used about 17 CPU cores and 136.8 MiB peak RSS.
- The report is committed separately from the implementation so it can record the actual implementation SHA. The ignored validation artifact is regenerated once more after the report commit, and the final handoff checks that its Git provenance equals final HEAD.
