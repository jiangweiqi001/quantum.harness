# Paper-faithful Fig. 3(b)(c) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make independent and legacy Fig. 3(b)(c) use the paper's folded shell convention, deterministic scar-state selection, styling, and auditable sidecars.

**Architecture:** Keep the validated full FSA recurrence and folded sector arrays unchanged. Add one shared selector/plot-data adapter that both render paths consume, then strengthen artifact metadata and server validation around the selected states and shell convention.

**Tech Stack:** Python 3.12, NumPy, Matplotlib, h5py, pytest.

## Global Constraints

- The recurrence has `L+1` internal shells; the plotted `k=0`, inversion-even sector has exactly `L/2+1` shells.
- Never mirror folded arrays to fabricate `L+1` plotted values.
- Panel (b) selects the lowest-energy member of the one-to-one matched scar tower.
- Panel (c) excludes exact zero modes and selects the negative-energy matched member closest to zero.
- Exact weights are black circles/solid; FSA weights are red crosses/dashed.
- Independent arrays remain `source=independent-ed`; DOI arrays are overlay/mismatch only.
- Preserve stable-snapshot provenance and transactional PNG/JSON/NPZ publication.

---

### Task 1: Select and render paper-faithful shell panels

**Files:**
- Modify: `scripts/turner2018_fig3.py`
- Modify: `scripts/turner2018_ed_observables.py` only if metadata cannot be emitted without changing numerical arrays.
- Modify: `scripts/turner2018_l32_server.py`
- Test: `scripts/tests/test_turner2018_fig3.py`
- Test: `scripts/tests/test_turner2018_ed_observables.py`
- Test: `scripts/tests/test_turner2018_l32_server.py`

**Interfaces:**
- Produces `select_fig3_shell_panel_states(...) -> tuple[dict, dict]`.
- Produces per-panel exact/FSA indices, energies, match strength, role, zero tolerance, shell counts, folding convention, and normalization evidence.
- Consumes existing `exact_shell_amplitudes`, `fsa_hamiltonian_sector`, and matched-tower selector output.

- [ ] **Step 1: Write failing shell-count and selection tests**

Create synthetic spectra with exact zeros, particle-hole pairs, deliberately
permuted FSA matches, and values distinct from DOI fixtures. Assert:

```python
assert metadata["full_fsa_shell_count"] == length + 1
assert metadata["plotted_folded_shell_count"] == length // 2 + 1
assert panel_b["role"] == "lowest-matched-scar"
assert panel_c["role"] == "negative-adjacent-to-zero"
assert energies[panel_c["exact_index"]] < -zero_tolerance
```

Also assert L=20 has 11 plotted points and the official L=32 fixture has 17.

- [ ] **Step 2: Run tests and confirm RED**

Run:

```bash
.venv/bin/python -m pytest \
  scripts/tests/test_turner2018_fig3.py \
  scripts/tests/test_turner2018_ed_observables.py \
  scripts/tests/test_turner2018_l32_server.py -q
```

Expected: failures because the shared selector and explicit folding metadata do
not yet exist.

- [ ] **Step 3: Implement the shared selector**

Reuse the existing one-to-one exact/FSA tower match. Reject empty, duplicated,
nonfinite, or ambiguous matches. Select the lowest exact tower energy for
panel (b). For panel (c), remove `abs(E) <= zero_tolerance`, retain `E < 0`,
and choose minimum `abs(E)` with deterministic index tie-breaking. Return
immutable metadata dictionaries without changing persisted numerical arrays.

- [ ] **Step 4: Route both renderers through shared plot data**

Use folded shell coordinates:

```python
shell = np.arange(length // 2 + 1)
panel.plot(shell, exact_weights, "o-", color="black", label="exact")
panel.plot(shell, fsa_weights, "x--", color="red", label="FSA")
```

Use titles `lowest scar-tower state` and
`scar-tower state adjacent to E=0`. Keep squared weights linear and preserve
all raw arrays in NPZ.

- [ ] **Step 5: Persist and validate evidence**

Record the full/plotted shell counts, folding text, selected indices and
energies, match strength, zero tolerance, normalization sums, source hashes,
and plot styles in JSON. Extend Task 6 figures-stage validation so stale or
inconsistent selection metadata prevents restart skipping.

- [ ] **Step 6: Add failure and regression tests**

Test zero-only candidates, no negative adjacent state, duplicated matches,
nonfinite weights, wrong shell count, stale sidecars, and corrupted selection
metadata. Assert Fig. 3(a)/(d), Fig. 4, legacy CLI, stable snapshots, and
transactional publication are unchanged.

- [ ] **Step 7: Run real L=20 workflow and inspect**

Regenerate current-fingerprint L=20 artifacts and Fig. 3. Inspect that panels
(b)(c) each contain 11 black/red points, selected roles and energies agree
with JSON, and the figure clearly says L=20 rather than implying final L=32.

- [ ] **Step 8: Verify and commit**

Run focused/regression pytest, `py_compile`, IDE lints, and `git diff --check`.
Commit only Task 1 source/tests/report changes locally; do not push.
