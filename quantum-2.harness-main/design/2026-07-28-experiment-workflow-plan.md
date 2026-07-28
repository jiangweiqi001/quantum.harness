# Numerical Experiment Workflow Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a two-layer numerical experiment workflow — an orchestrator (`/experiment`) that owns the six-phase lifecycle (MVP → Verify → Smoke → Refactor → Organize → Scan) with user-gated transitions, and six independently-invocable phase skills.

**Architecture:** Seven new local skills sharing an `experiment.json` state file at `tracks/<track>/experiment.json`. The orchestrator reads phase status, dispatches the active phase skill, presents results at user gates, and advances on `continue`. Each phase skill reads its input block from the state file and writes its output back.

**Tech Stack:** Markdown (SKILL.md files), TOML (Ion.toml), JSON (experiment.json state)

**Spec:** `design/2026-07-28-experiment-workflow-design.md`

## Global Constraints

- Branch: `version-1.0002`
- Zero modifications to `.knowledge/`, existing skills, or `Makefile`
- All changes are additive — create new files + append to Ion.toml + append to AGENTS.md
- Follow existing skill conventions: YAML frontmatter with `name` and `description`, markdown body
- All skills are `{ type = "local" }` in Ion.toml
- Phase skills MUST be independently invocable as `/experiment-<name>`

---

### Task 1: Create `/experiment-mvp` skill

**Files:**
- Create: `skills/experiment-mvp/SKILL.md`

**Interfaces:**
- Produces: `/experiment-mvp` command. Reads `tracks/<track>/experiment.json`, checks `phases.mvp.status == "active"`, builds MVP script, writes output.

- [ ] **Step 1: Create skill directory**

```bash
mkdir -p skills/experiment-mvp
```

- [ ] **Step 2: Write SKILL.md**

Create `skills/experiment-mvp/SKILL.md`:

```markdown
---
name: experiment-mvp
description: Build a minimal working prototype script for a numerical physics idea. Phase 1 of the experiment workflow. Triggered by /experiment-mvp or by the /experiment orchestrator.
---

# experiment-mvp

Build a minimal, self-contained, runnable script that computes the
target quantity. This is the prototype phase — correctness matters more
than elegance. A single file, one command to run, one number out.

## When

- `/experiment` orchestrator dispatches this phase.
- User invokes `/experiment-mvp` directly to prototype a new idea.

## Input

Read `tracks/<track>/experiment.json`. The `phases.mvp` block provides:
- `model`: the physical model name (maps to `.knowledge/models/<name>/MODEL.md`)
- `method`: the computational method
- `target_quantity`: what to compute (gap, energy, order parameter, etc.)
- `parameters`: parameter values for the first run

If `experiment.json` does not exist, ask the user for:
- Model, method, target quantity, and parameter values.
- The track path. Create a minimal state file: `{"track": "...", "phase": "mvp",
  "phases": {"mvp": {"status": "active"}}}`.

## What it does

1. **Read the model card** at `.knowledge/models/<name>/MODEL.md` for the
   Hamiltonian form, conventions, and recommended defaults.
2. **Read the method card** at `skills/method-<name>/SKILL.md` for code
   shape and tool selection.
3. **Read project knowledge** if `tracks/<track>/knowledge/README.md` exists
   — use benchmarks and prior runs to anchor parameter choices.
4. **Build a single-file script.** Use the model card's Hamiltonian and the
   method card's code shape. Keep it self-contained: one `.py` or `.jl` file
   that runs with one command and prints the target quantity.
5. **Run once** at the smallest sensible system size. Verify it produces a
   number (not NaN, not inf).
6. **Write output** to `experiment.json`:
   ```json
   "mvp": {
     "status": "done",
     "script": "run_model.py",
     "result": { "value": 0.34, "unit": "energy gap" },
     "run_command": "python run_model.py",
     "notes": ""
   }
   ```

## Output

Report to the orchestrator (or user): script path, result value, run command,
and any concerns. The script is saved at `tracks/<track>/run_<model>.py` (or `.jl`).

## Not this

- Don't optimize or refactor — that's `/experiment-refactor`.
- Don't run at multiple sizes — that's `/experiment-scan`.
- Don't verify against benchmarks — that's `/experiment-verify`.
- Don't create directory structure — that's `/experiment-organize`.
```

- [ ] **Step 3: Validate the skill**

```bash
ion skill validate skills/experiment-mvp
```

- [ ] **Step 4: Commit**

```bash
git add skills/experiment-mvp/SKILL.md
git commit -m "feat: add experiment-mvp phase skill

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>"
```

---

### Task 2: Create `/experiment-verify` skill

**Files:**
- Create: `skills/experiment-verify/SKILL.md`

**Interfaces:**
- Consumes: `experiment.json` with `phases.mvp.status == "done"` and `phases.verify.status == "active"`
- Produces: `phases.verify` block with benchmark comparison result

- [ ] **Step 1: Create skill directory**

```bash
mkdir -p skills/experiment-verify
```

- [ ] **Step 2: Write SKILL.md**

Create `skills/experiment-verify/SKILL.md`:

```markdown
---
name: experiment-verify
description: Verify a numerical script against known benchmarks — analytic limits, published values, or prior harness runs. Phase 2 of the experiment workflow. Triggered by /experiment-verify or by the /experiment orchestrator.
---

# experiment-verify

Run the MVP script against a known benchmark value and confirm agreement
within tolerance. If no benchmark exists, flag this and suggest the user
provide one or explicitly skip.

## When

- `/experiment` orchestrator dispatches this phase.
- User invokes `/experiment-verify` directly to check a script.

## Input

Read `tracks/<track>/experiment.json`. The `phases.verify` block provides:
- `benchmark_source`: where the reference value comes from (analytic limit,
  published paper, prior harness run)
- `benchmark_value`: the expected value
- `tolerance`: acceptable relative error (default 0.01 = 1%)

The `phases.mvp` block provides the script path and run command.

If `experiment.json` does not exist or benchmark info is missing, ask the user.
Check `tracks/<track>/knowledge/benchmarks.md` for known reference values.

## What it does

1. **Identify the benchmark.** In priority order:
   - User-specified value in `experiment.json`.
   - `tracks/<track>/knowledge/benchmarks.md` entries matching the model.
   - Analytic limits from `.knowledge/limits.md`.
   - Ask the user if none found.
2. **Run the script** at the benchmark parameter point.
3. **Compare.** Compute `|computed - benchmark| / |benchmark|`.
4. **Verdict:**
   - PASS: relative error ≤ tolerance. Write `"verdict": "pass"`.
   - FAIL: relative error > tolerance. Write `"verdict": "fail"` with
     diagnostic: possible sign error, missing factor, wrong sector.
   - NO_BENCHMARK: no reference available. Write `"verdict": "skipped"` with
     `"reason": "no benchmark available"`.
5. **Write output** to `experiment.json`:
   ```json
   "verify": {
     "status": "done",
     "benchmark_source": "analytic: 2|t1-t2|",
     "benchmark_value": 1.0,
     "computed_value": 1.0003,
     "relative_error": 0.0003,
     "verdict": "pass",
     "notes": ""
   }
   ```

## Output

Report: benchmark value, computed value, relative error, verdict (pass/fail/skipped).
On fail, include diagnostic suggestions. On no-benchmark, note that the user may
proceed but verification is incomplete.

## Not this

- Don't run at multiple sizes — that's `/experiment-smoke` and `/experiment-scan`.
- Don't fix the script if it fails — report and let the user decide.
```

- [ ] **Step 3: Validate and commit**

```bash
ion skill validate skills/experiment-verify
git add skills/experiment-verify/SKILL.md
git commit -m "feat: add experiment-verify phase skill

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>"
```

---

### Task 3: Create `/experiment-smoke` skill

**Files:**
- Create: `skills/experiment-smoke/SKILL.md`

- [ ] **Step 1: Create skill directory**

```bash
mkdir -p skills/experiment-smoke
```

- [ ] **Step 2: Write SKILL.md**

Create `skills/experiment-smoke/SKILL.md`:

```markdown
---
name: experiment-smoke
description: Run a verified script at the smallest non-trivial size to confirm no crashes and plausible output magnitudes. Phase 3 of the experiment workflow. Triggered by /experiment-smoke or by the /experiment orchestrator.
---

# experiment-smoke

Run the verified script at the smallest non-trivial system size. Confirm:
no exceptions, memory usage within budget, runtime plausible, output
magnitudes physically reasonable.

## When

- `/experiment` orchestrator dispatches this phase.
- User invokes `/experiment-smoke` directly.

## Input

Read `tracks/<track>/experiment.json`. Uses:
- `phases.mvp.script`: path to the script
- `phases.mvp.run_command`: how to run it
- `phases.verify.verdict`: must be `"pass"` or `"skipped"` — smoke test
  requires prior verification (or explicit skip)

## What it does

1. **Determine the smallest non-trivial size.** For a 1D chain: L=4 or L=6.
   For a 2D lattice: 2×2 or 3×3. Ask the user if ambiguous.
2. **Run the script** at that size.
3. **Crash check:** Script exits with code 0. No uncaught exceptions.
4. **Memory check:** Resident memory < 1 GB (or warn if larger).
5. **Runtime check:** Wall time < 5 minutes (or warn if longer).
6. **Sanity checks on output:**
   - Ground state energy is negative (for AFM models with J>0).
   - Gap is non-negative.
   - Order parameters are in physically valid ranges (e.g., magnetization
     in [0, 1/2] for spin-1/2).
   - No NaN or inf values.
7. **If the model has a solvable limit**, run a limit check (e.g., U=0 for
   Hubbard should match tight-binding).
8. **Write output** to `experiment.json`:
   ```json
   "smoke": {
     "status": "done",
     "system_size": "L=4",
     "wall_time_s": 2.3,
     "memory_mb": 150,
     "crashes": false,
     "output_sane": true,
     "notes": ""
   }
   ```

## Output

Report: system size, wall time, memory, crash status, sanity verdict.
If sanity checks fail, list which ones and suggest possible causes
(sign convention, sector mismatch, normalization).

## Not this

- Don't verify against benchmarks — that's `/experiment-verify`.
- Don't run at multiple sizes — that's `/experiment-scan`.
```

- [ ] **Step 3: Validate and commit**

```bash
ion skill validate skills/experiment-smoke
git add skills/experiment-smoke/SKILL.md
git commit -m "feat: add experiment-smoke phase skill

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>"
```

---

### Task 4: Create `/experiment-refactor` skill

**Files:**
- Create: `skills/experiment-refactor/SKILL.md`

- [ ] **Step 1: Create skill directory**

```bash
mkdir -p skills/experiment-refactor
```

- [ ] **Step 2: Write SKILL.md**

Create `skills/experiment-refactor/SKILL.md`:

```markdown
---
name: experiment-refactor
description: Transform a working monolithic script into clean, modular code while preserving numerical output bit-for-bit. Phase 4 of the experiment workflow. Triggered by /experiment-refactor or by the /experiment orchestrator.
---

# experiment-refactor

Decompose a working monolithic script into modules: model definition,
solver algorithms, and observables. The refactored code MUST reproduce
the verified numerical result exactly.

## When

- `/experiment` orchestrator dispatches this phase.
- User invokes `/experiment-refactor` directly.

## Input

Read `tracks/<track>/experiment.json`. Uses:
- `phases.mvp.script`: path to original monolithic script
- `phases.verify`: benchmark values to preserve
- `phases.smoke`: known-good system size and output

## Target structure

```
<project>/
├── src/
│   ├── __init__.py
│   ├── model.py          # Model class (Hamiltonian, lattice, parameters)
│   ├── algorithms.py     # Solver methods (diagonalize, DMRG, MC steps)
│   └── observables.py    # Energy, gap, order parameters, correlations
├── scripts/
│   ├── run.py            # Main entry point — imports from src/
│   └── plot.py           # Plotting utilities
├── tests/
│   └── test_model.py     # Basic correctness tests
├── configs/
│   └── default.yaml      # All parameters explicit and documented
└── README.md             # Model, method, usage, dependencies
```

## What it does

1. **Read the original script.** Identify the logical sections: model setup,
   solver configuration, computation, and output.
2. **Extract model** into `src/model.py` — a class or function that builds the
   Hamiltonian/lattice from parameters.
3. **Extract solver** into `src/algorithms.py` — the computational kernel.
4. **Extract observables** into `src/observables.py` — post-processing of
   raw results into physical quantities.
5. **Write entry point** `scripts/run.py` — imports from `src/`, reads config,
   runs computation, writes results.
6. **Write `scripts/plot.py`** — basic visualization of the computed quantity.
7. **Write `configs/default.yaml`** — all parameters with comments.
8. **Write `tests/test_model.py`** — verify: model builds without error,
   known symmetry is respected, refactored output matches original script
   output to machine precision.
9. **Run the regression test.** `tests/test_model.py` MUST pass — refactored
   code output equals original script output.
10. **Write `README.md`** — model description, method, usage example,
    dependencies.
11. **Write output** to `experiment.json`:
    ```json
    "refactor": {
      "status": "done",
      "structure": "src/ scripts/ tests/ configs/",
      "regression_test": "pass",
      "notes": ""
    }
    ```

## Critical rule

The refactored code must produce **bit-for-bit identical** numerical output
to the original script at the smoke-test parameter point. Run both and
assert equality. If refactoring changes the output, fix the refactored
code — never accept a discrepancy.

## Not this

- Don't add features — this is restructuring only.
- Don't change the algorithm or parameters.
- Don't create the directory structure for the track — that's
  `/experiment-organize`.
```

- [ ] **Step 3: Validate and commit**

```bash
ion skill validate skills/experiment-refactor
git add skills/experiment-refactor/SKILL.md
git commit -m "feat: add experiment-refactor phase skill

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>"
```

---

### Task 5: Create `/experiment-organize` skill

**Files:**
- Create: `skills/experiment-organize/SKILL.md`

- [ ] **Step 1: Create skill directory**

```bash
mkdir -p skills/experiment-organize
```

- [ ] **Step 2: Write SKILL.md**

Create `skills/experiment-organize/SKILL.md`:

```markdown
---
name: experiment-organize
description: Verify the refactored project structure, ensure tests pass, update project knowledge records, and prepare for production scanning. Phase 5 of the experiment workflow. Triggered by /experiment-organize or by the /experiment orchestrator.
---

# experiment-organize

Final quality gate before production scanning. Verify the refactored
structure matches the target layout, tests pass, configs are explicit,
and the track knowledge base is updated.

## When

- `/experiment` orchestrator dispatches this phase.
- User invokes `/experiment-organize` directly.

## Input

Read `tracks/<track>/experiment.json`. Uses:
- `phases.refactor.structure`: confirms refactoring is done
- `phases.refactor.regression_test`: must be `"pass"`

## What it does

1. **Verify directory structure** matches the target layout:
   - `src/__init__.py`, `src/model.py`, `src/algorithms.py`, `src/observables.py`
   - `scripts/run.py`, `scripts/plot.py`
   - `tests/test_model.py`
   - `configs/default.yaml`
   - `README.md`
   Flag any missing files.

2. **Run the full test suite:**
   ```bash
   python -m pytest tests/ -v
   ```
   (or Julia equivalent). All tests must pass.

3. **Verify `configs/default.yaml`** contains every parameter. No hardcoded
   magic numbers in source files.

4. **Verify `__init__.py` exports** are clean — `from src.model import
   ModelName` works.

5. **Update project knowledge:**
   - Append the verified benchmark result to `tracks/<track>/knowledge/prior-runs.md`.
   - If new benchmarks were discovered during verification, update
     `tracks/<track>/knowledge/benchmarks.md`.

6. **Write output** to `experiment.json`:
   ```json
   "organize": {
     "status": "done",
     "structure_ok": true,
     "tests_pass": true,
     "config_complete": true,
     "knowledge_updated": true,
     "notes": ""
   }
   ```

## Output

Report: structure check result, test result, config completeness, knowledge
update status.

## Not this

- Don't run production scans — that's `/experiment-scan`.
- Don't refactor code — that's `/experiment-refactor`.
```

- [ ] **Step 3: Validate and commit**

```bash
ion skill validate skills/experiment-organize
git add skills/experiment-organize/SKILL.md
git commit -m "feat: add experiment-organize phase skill

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>"
```

---

### Task 6: Create `/experiment-scan` skill

**Files:**
- Create: `skills/experiment-scan/SKILL.md`

- [ ] **Step 1: Create skill directory**

```bash
mkdir -p skills/experiment-scan
```

- [ ] **Step 2: Write SKILL.md**

Create `skills/experiment-scan/SKILL.md`:

```markdown
---
name: experiment-scan
description: Run a production-scale parameter sweep using the organized, tested codebase. Composes with /parameter-scan and /using-slurm. Phase 6 of the experiment workflow. Triggered by /experiment-scan or by the /experiment orchestrator.
---

# experiment-scan

Production parameter sweep. Runs the organized, tested codebase across
a parameter grid, collects results, produces publication-quality plots.
Composes with `/parameter-scan` for grid enumeration and `/using-slurm`
for cluster execution.

## When

- `/experiment` orchestrator dispatches this phase.
- User invokes `/experiment-scan` directly.

## Input

Read `tracks/<track>/experiment.json`. Uses:
- `phases.organize`: must be `"done"` with `tests_pass: true`
- `phases.smoke`: known wall time and memory per run point
- Model and method from top-level fields

Ask the user for:
- Parameter axes to sweep (e.g., "L in [4,6,8,10], delta in [0, 0.5, 1.0]")
- Whether to run locally or on cluster

## What it does

1. **Define the parameter grid.** From user input, build axis definitions.
   One row per (L, coupling, ...) combination.
2. **Estimate total cost.** Per-cell: smoke-test wall time × safety factor
   (2× for same size, scaling law for larger sizes). Total: sum of cells.
3. **Decide local vs cluster.** If total wall > 30 min or total memory >
   8 GB → cluster (compose with `/using-slurm`). Otherwise local.
4. **Plan the scan.** Compose with `/parameter-scan`:
   - `scripts/parameter_scan.py plan --axes axes.json --run-id <run>`
5. **Execute.** Submit to cluster or run locally. For cluster: ship code,
   submit array job, monitor.
6. **Monitor progress.** Print intermediate estimates every few cells.
   Surface any failed cells immediately.
7. **Collect results.** `scripts/parameter_scan.py collect` to assemble CSV.
8. **Plot.** `scripts/parameter_scan.py plot` for publication-ready figures.
9. **Shape labels.** `scripts/parameter_scan.py shape` for generic curve
   classification.
10. **Write output** to `experiment.json`:
    ```json
    "scan": {
      "status": "done",
      "axes": {"L": [4,6,8,10], "delta": [0, 0.5, 1.0]},
      "total_cells": 12,
      "cells_completed": 12,
      "cells_failed": 0,
      "where": "local",
      "wall_time_total_s": 180,
      "artifacts": {
        "csv": "results/<run>/parameter-scan.csv",
        "plot": "results/<run>/parameter-scan.png"
      },
      "notes": ""
    }
    ```

## Output

Report: scan grid, completion status (N/N cells), plot path, CSV path,
key observed trends, and shape labels for each curve.

Compose with `/report` if the user wants an HTML writeup.

## Not this

- Don't define scan ranges without user input.
- Don't skip the cost estimate — always estimate before running.
- Don't write the per-cell scripts — `/parameter-scan` does that.
```

- [ ] **Step 3: Validate and commit**

```bash
ion skill validate skills/experiment-scan
git add skills/experiment-scan/SKILL.md
git commit -m "feat: add experiment-scan phase skill

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>"
```

---

### Task 7: Create `/experiment` orchestrator skill

**Files:**
- Create: `skills/experiment/SKILL.md`

**Interfaces:**
- Consumes: All six phase skills (experiment-mvp, experiment-verify, experiment-smoke, experiment-refactor, experiment-organize, experiment-scan) — invokes them by name
- Produces: `/experiment` orchestrator that manages the full lifecycle with user gates

- [ ] **Step 1: Create skill directory**

```bash
mkdir -p skills/experiment
```

- [ ] **Step 2: Write SKILL.md**

Create `skills/experiment/SKILL.md`:

```markdown
---
name: experiment
description: Orchestrate the full numerical experiment lifecycle — from prototype script to production parameter sweep. Use when starting a new computational physics project or resuming an in-progress experiment. The six-phase pipeline (MVP → Verify → Smoke → Refactor → Organize → Scan) keeps the agent phase-aware so it neither rushes to production nor over-engineers a prototype.
---

# experiment

Entry point for new numerical physics projects. Owns the six-phase
lifecycle. At each phase boundary, the user controls the gate:
`continue` (advance), `redo` (repeat current phase), or `abort`
(save state and exit).

## When

- User says "start a new experiment," "new computational project," or
  describes a physics idea they want to compute from scratch.
- User says "resume experiment" for an in-progress `experiment.json`.

## Phase flow

```
MVP ──[gate]──> Verify ──[gate]──> Smoke ──[gate]──> Refactor ──[gate]──> Organize ──[gate]──> Scan
```

All gates are user-controlled. The orchestrator never advances without
explicit user confirmation.

## State file

All state lives in `tracks/<track>/experiment.json`. The orchestrator
creates this file at the start of phase 1 and each phase skill writes
its output into it. Schema:

```json
{
  "track": "tracks/<track-name>",
  "model": "<model-name>",
  "method": "<method-family>",
  "phase": "<current-phase>",
  "phases": {
    "mvp":      { "status": "pending|active|done" },
    "verify":   { "status": "pending|active|done" },
    "smoke":    { "status": "pending|active|done" },
    "refactor": { "status": "pending|active|done" },
    "organize": { "status": "pending|active|done" },
    "scan":     { "status": "pending|active|done" }
  }
}
```

## Orchestrator behavior

1. **Detect or initialize state.**
   - If `tracks/<track>/experiment.json` exists: read it, find the phase
     marked `"active"`, resume from there.
   - If absent: ask the user for model, method, and track. Create a fresh
     state file with all phases `"pending"` and `phase: "mvp"`. Set
     `phases.mvp.status = "active"`.

2. **Dispatch the active phase.** Invoke the matching phase skill:
   `/experiment-mvp`, `/experiment-verify`, `/experiment-smoke`,
   `/experiment-refactor`, `/experiment-organize`, or `/experiment-scan`.
   Pass the state file path.

3. **Present results at gate.** After the phase skill completes, read its
   output from `experiment.json`. Show a compact summary:
   - Phase name and what was done.
   - Key result (number, plot, or status).
   - Any warnings or concerns the phase skill flagged.

4. **Ask gate question.** Present three options:
   - `continue` — mark current phase `"done"`, set next phase `"active"`,
     advance `phase` field, go to step 2.
   - `redo` — reset current phase output fields, set status back to
     `"active"`, go to step 2 (re-dispatch same phase).
   - `abort` — save state file as-is, exit. User can resume later.

5. **Completion.** After Scan phase gate passes, mark scan `"done"`, set
   `phase: "complete"`. Offer writeup handoff via `/report`.

## Phase dispatch contract

Each phase skill:
- Reads `tracks/<track>/experiment.json` on start.
- Checks `phases.<name>.status == "active"`.
- Executes. Writes output fields into `phases.<name>`.
- Sets `phases.<name>.status = "done"`.
- Reports a compact summary.

The orchestrator reads the updated state file after each phase.

## Starting mid-pipeline

If the user already has a working script, they can jump to any phase:
- `/experiment-verify` — "I have a script, verify it."
- `/experiment-refactor` — "My code works, clean it up."
- `/experiment-scan` — "Everything is ready, run the sweep."

The orchestrator detects the current state and resumes. Phase skills
invoked directly create a minimal state file if absent.

## Gate example

```
Phase MVP complete.
  Script: run_rice_mele_ed.py
  Result: gap = 0.342 at L=8, delta=0.5
  Concerns: none

Continue to Verify / Redo MVP / Abort?
```

## Not this

- Don't skip phases without user consent.
- Don't auto-advance past gates.
- Don't run computations the phase skill doesn't own.
- Don't execute phase tasks directly — always dispatch to the phase skill.
```

- [ ] **Step 3: Validate and commit**

```bash
ion skill validate skills/experiment
git add skills/experiment/SKILL.md
git commit -m "feat: add experiment orchestrator skill

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>"
```

---

### Task 8: Register all 7 skills in Ion.toml

**Files:**
- Modify: `Ion.toml` (add 7 lines)

**Interfaces:**
- Consumes: All 7 skill files from Tasks 1-7
- Produces: Registered skills available as `/experiment`, `/experiment-mvp`, etc.

- [ ] **Step 1: Add skill registrations**

Open `Ion.toml`. Add the following block after the existing local skill registrations, in the workflow skills area (near `solve`, `reproduce-paper`):

```toml
# Numerical experiment workflow
experiment = { type = "local" }
experiment-mvp = { type = "local" }
experiment-verify = { type = "local" }
experiment-smoke = { type = "local" }
experiment-refactor = { type = "local" }
experiment-organize = { type = "local" }
experiment-scan = { type = "local" }
```

- [ ] **Step 2: Verify TOML validity**

```bash
python3 -c "import tomllib; tomllib.load(open('Ion.toml','rb')); print('valid')"
```

- [ ] **Step 3: Sync skills**

```bash
ion add
```
Expected: completes without errors. Verify with `ion list | grep experiment`.

- [ ] **Step 4: Commit**

```bash
git add Ion.toml
git commit -m "feat: register all 7 experiment workflow skills

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>"
```

---

### Task 9: Update AGENTS.md

**Files:**
- Modify: `AGENTS.md` (append a paragraph to Repository Layout section, after the existing skill list)

- [ ] **Step 1: Add experiment workflow mention**

Open `AGENTS.md`. In the "Repository Layout" section, after the line about cluster profiles, insert:

```markdown
- Experiment workflow: `skills/experiment/SKILL.md` — orchestrates the six-phase
  numerical experiment lifecycle (MVP → Verify → Smoke → Refactor → Organize →
  Scan). Each phase is independently invocable: `/experiment-mvp`,
  `/experiment-verify`, `/experiment-smoke`, `/experiment-refactor`,
  `/experiment-organize`, `/experiment-scan`. State tracked in
  `tracks/<track>/experiment.json`.
```

- [ ] **Step 2: Verify**

```bash
grep -n "experiment" AGENTS.md
```
Expected: at least one match for the newly inserted lines.

- [ ] **Step 3: Commit**

```bash
git add AGENTS.md
git commit -m "docs: add experiment workflow to repository layout

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>"
```

---

## Task Dependencies

```
Tasks 1-6 (phase skills)  ── independent, can run in parallel
Task 7 (orchestrator)      ── depends on Tasks 1-6 (references phase skill names)
Task 8 (Ion.toml)          ── depends on Tasks 1-7 (all skill files must exist)
Task 9 (AGENTS.md)          ── independent, can run anytime
```

Recommended dispatch order: Tasks 1-6 in parallel → Task 7 → Task 8 → Task 9.

## Self-Review

1. **Spec coverage:**
   - Orchestrator skill → Task 7
   - Six phase skills → Tasks 1-6
   - Ion.toml registration → Task 8
   - AGENTS.md update → Task 9
   - `experiment.json` state file convention → embedded in Tasks 1 and 7
   - User gate behavior (continue/redo/abort) → Task 7
   - Phase skills independently invocable → each phase skill's "When" section
   - Refactor target layout → Task 4
   - All items from spec are covered.

2. **Placeholder scan:** No TODOs or vague instructions. All skill content is
   concrete. All verification steps have exact commands.

3. **Type consistency:** Skill names (`experiment-mvp`, `experiment-verify`,
   etc.) are consistent across all 7 SKILL.md files, Ion.toml registrations,
   and AGENTS.md mention. `experiment.json` field names (`phases.<name>.status`,
   `track`, `model`, `method`, `phase`) are consistent across all files.
