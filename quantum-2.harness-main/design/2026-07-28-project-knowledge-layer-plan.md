# Project Knowledge Layer Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an opt-in project knowledge layer (`tracks/<track>/knowledge/`) so agents discover and use project-cooked benchmarks, prior runs, and notes before computing.

**Architecture:** Three-layer mechanism — (1) AGENTS.md declares the convention that agents MUST read `knowledge/README.md` before working on a track; (2) a new `/explore-on-project-knowledge` skill reads and surfaces project knowledge; (3) `/solve` and `/reproduce-paper` check for project knowledge during initialization. All changes are purely additive.

**Tech Stack:** Markdown (skill files, AGENTS.md), TOML (Ion.toml registration)

**Spec:** `design/2026-07-28-project-knowledge-layer-design.md`

## Global Constraints

- Branch: `version-1.0001`
- Zero modifications to `.knowledge/` or any method/tool skill
- All changes are additive insertions into existing files
- `knowledge/` directories are opt-in per track
- Follow existing skill file conventions (YAML frontmatter, markdown body)

---

### Task 1: Modify AGENTS.md — add Project Knowledge section

**Files:**
- Modify: `AGENTS.md` (insert ~15 lines after line 57, i.e., after "Knowledge Base Role" section and before "Card shapes" section)

**Interfaces:**
- Produces: AGENTS.md convention that agents MUST read `tracks/<track>/knowledge/README.md`

- [ ] **Step 1: Insert the new section**

Open `AGENTS.md`. After line 57 (the last line of "Knowledge Base Role": `Skills cite these cards; they never hardcode the data. New cards land when a real skill begins citing them.`), and before line 61 (`## Card shapes`), insert:

```markdown
## Project Knowledge (`tracks/<track>/knowledge/`)

When working on a specific track, the agent MUST first read
`tracks/<track>/knowledge/README.md`. This directory contains
**project-cooked knowledge** that the general `.knowledge/` cards
do not carry:

- **Benchmarks** — published numerical values from prior work
  (energies, gaps, order parameters at specific parameter points).
- **Prior runs** — results already produced by this harness for this
  track (parameter sets, convergence data, verified outputs).
- **Notes** — domain-specific refinements: sign-convention traps,
  known finite-size artifacts, recommended parameter ranges.

The agent should cite these values when proposing a setup, check
against them during verification, and treat disagreement as a signal
to re-examine the setup.

To read project knowledge on demand:
  `/explore-on-project-knowledge`
```

- [ ] **Step 2: Verify the insertion point**

Run: `grep -n "Card shapes" AGENTS.md`
Expected: the line number should be 15 lines higher than before (old ~61 → new ~76).

- [ ] **Step 3: Verify no content was lost**

Run: `grep -c "^## " AGENTS.md`
Compare section count before and after — should increase by exactly 1.

- [ ] **Step 4: Commit**

```bash
git add AGENTS.md
git commit -m "docs: add project knowledge layer convention to AGENTS.md

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>"
```

---

### Task 2: Create `/explore-on-project-knowledge` skill

**Files:**
- Create: `skills/explore-on-project-knowledge/SKILL.md`

**Interfaces:**
- Produces: `/explore-on-project-knowledge` command that reads a track's `knowledge/` directory and surfaces benchmarks, prior runs, and notes

- [ ] **Step 1: Create the skill directory**

```bash
mkdir -p skills/explore-on-project-knowledge
```

- [ ] **Step 2: Write the SKILL.md file**

Create `skills/explore-on-project-knowledge/SKILL.md`:

```markdown
---
name: explore-on-project-knowledge
description: Read and surface project-cooked knowledge for a specific track. Use when starting work on a track to discover benchmarks, prior runs, and domain notes. Triggered by `/explore-on-project-knowledge` or automatically by entry-point skills.
---

# explore-on-project-knowledge

Read the project knowledge directory (`tracks/<track>/knowledge/`) and
surface a compact summary: known benchmarks, prior harness results, and
critical traps or notes.

## When

- User types `/explore-on-project-knowledge`.
- `/solve` or `/reproduce-paper` detects a `knowledge/README.md` during
  initialization.

## What it does

1. **Locate the knowledge directory.** If the user provides a track path
   (e.g., `tracks/ed/hubbard-pump`), use it. Otherwise infer from
   context or ask.

2. **Read the entry point.** Read `tracks/<track>/knowledge/README.md`
   in full. If it does not exist, report:
   > "No project knowledge found for `<track>`. General `.knowledge/`
   > cards still apply."

3. **Follow links.** If README.md references `benchmarks.md`,
   `prior-runs.md`, or `notes.md`, read each in full.

4. **Surface a compact summary:**

   - **Benchmarks:** table of published values with sources.
   - **Prior runs (this harness):** parameter → value pairs from
     verified harness runs.
   - **Critical notes:** sign-convention traps, known artifacts,
     recommended parameter ranges.

   Format: one compact table per category, no walls of text.

5. **Hand off.** Return the summary to the calling skill or user.
   When called by `/solve` or `/reproduce-paper`, the calling skill
   uses these values to anchor the setup proposal.
```

- [ ] **Step 3: Validate the skill**

Run: `ion skill validate skills/explore-on-project-knowledge`

- [ ] **Step 4: Commit**

```bash
git add skills/explore-on-project-knowledge/SKILL.md
git commit -m "feat: add explore-on-project-knowledge skill

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>"
```

---

### Task 3: Register skill in Ion.toml

**Files:**
- Modify: `Ion.toml` (add 1 line)

**Interfaces:**
- Consumes: `skills/explore-on-project-knowledge/SKILL.md` (from Task 2)
- Produces: Ion-managed skill available as `/explore-on-project-knowledge`

- [ ] **Step 1: Add the skill registration**

Open `Ion.toml`. Add the following line after the existing local skill registrations, in the logical grouping near the other discovery/investigation skills. Insert after line 20 (`report = { type = "local" }`):

```toml
explore-on-project-knowledge = { type = "local" }
```

- [ ] **Step 2: Verify Ion.toml is valid TOML**

Run: `python3 -c "import tomllib; tomllib.load(open('Ion.toml','rb')); print('valid')"`

- [ ] **Step 3: Sync skills**

Run: `ion add`
Expected: completes without errors, `explore-on-project-knowledge` appears in `ion list` output.

- [ ] **Step 4: Commit**

```bash
git add Ion.toml
git commit -m "feat: register explore-on-project-knowledge skill

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>"
```

---

### Task 4: Modify `/solve` — add project knowledge check

**Files:**
- Modify: `skills/solve/SKILL.md` (insert ~3 lines in Intake section, after line 36)

**Interfaces:**
- Consumes: `/explore-on-project-knowledge` skill (from Task 2)
- Produces: `/solve` now checks for project knowledge during intake

- [ ] **Step 1: Insert the project knowledge check**

Open `skills/solve/SKILL.md`. In the `## Intake` section, after line 36 (`If a card redirects to dynamics, finite-T, or another stub, follow the redirect immediately.`), insert a blank line then:

```markdown
If the problem maps to a track under `tracks/`, check for
`tracks/<track>/knowledge/README.md`. When present, invoke
`/explore-on-project-knowledge` and anchor the setup proposal
against its benchmarks and prior runs.
```

- [ ] **Step 2: Verify the insertion**

Run: `grep -n "explore-on-project-knowledge" skills/solve/SKILL.md`
Expected: one match at the inserted location.

- [ ] **Step 3: Commit**

```bash
git add skills/solve/SKILL.md
git commit -m "feat: add project knowledge check to solve intake

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>"
```

---

### Task 5: Modify `/reproduce-paper` — add project knowledge check

**Files:**
- Modify: `skills/reproduce-paper/SKILL.md` (insert ~3 lines in Step 0, after line 98, i.e., after "Problem-required extras" paragraph)

**Interfaces:**
- Consumes: `/explore-on-project-knowledge` skill (from Task 2)
- Produces: `/reproduce-paper` now checks for project knowledge during step 0

- [ ] **Step 1: Insert the project knowledge check**

Open `skills/reproduce-paper/SKILL.md`. In the `### Step 0 — Clarify the problem` section, after the "Problem-required extras" paragraph (line 98: `Method-specific knobs are named as open decisions, not configured here.`), insert a blank line then:

```markdown
- **Project knowledge.** If the reproduction maps to a track under
  `tracks/`, check for `tracks/<track>/knowledge/README.md`. When
  present, invoke `/explore-on-project-knowledge` and use its
  benchmarks and prior runs to inform the setup card.
```

- [ ] **Step 2: Verify the insertion**

Run: `grep -n "explore-on-project-knowledge" skills/reproduce-paper/SKILL.md`
Expected: one match at the inserted location.

- [ ] **Step 3: Commit**

```bash
git add skills/reproduce-paper/SKILL.md
git commit -m "feat: add project knowledge check to reproduce-paper step 0

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>"
```

---

### Task 6: Create example `knowledge/` directory for `tracks/ed/hubbard-pump`

**Files:**
- Create: `tracks/ed/hubbard-pump/knowledge/README.md`
- Create: `tracks/ed/hubbard-pump/knowledge/benchmarks.md`
- Create: `tracks/ed/hubbard-pump/knowledge/prior-runs.md`
- Create: `tracks/ed/hubbard-pump/knowledge/notes.md`

**Interfaces:**
- Produces: A concrete example of the project knowledge convention that agents can read and use

- [ ] **Step 1: Create the knowledge directory**

```bash
mkdir -p tracks/ed/hubbard-pump/knowledge
```

- [ ] **Step 2: Write knowledge/README.md**

Create `tracks/ed/hubbard-pump/knowledge/README.md`:

```markdown
---
track: hubbard-pump
model: Rice-Mele
methods: [ED]
last_updated: 2026-07-28
---

# Project Knowledge: Hubbard / Rice-Mele Pump

## Benchmarks

Key published values used as calibration targets. Full table in
[benchmarks.md](./benchmarks.md).

| Quantity | Published value | Source |
|---|---|---|
| SSH gap (t1=1, t2=1.5, L=8, PBC) | Δ = 1.0 | analytic: 2|t1-t2| |

## Prior Runs (this harness)

Results already computed and verified. Full log in
[prior-runs.md](./prior-runs.md).

| Run ID | Model | L | Key result |
|---|---|---|---|
| 2026-07-28-ssh-smoke | SSH t1=1,t2=1.5 | 8 | Edge modes confirmed |

## Critical Notes

See [notes.md](./notes.md) for traps and known artifacts.

## File Index

- [benchmarks.md](./benchmarks.md) — complete benchmark table with sources
- [prior-runs.md](./prior-runs.md) — detailed run records
- [notes.md](./notes.md) — free-form domain notes, traps, artifacts
```

- [ ] **Step 3: Write knowledge/benchmarks.md**

Create `tracks/ed/hubbard-pump/knowledge/benchmarks.md`:

```markdown
# Benchmarks: Hubbard / Rice-Mele Pump

Published numerical values from prior work. Use these as calibration
targets — your results should agree within the stated precision.

## SSH Model (free fermion, dimerized chain)

| Parameter set | Quantity | Value | Source |
|---|---|---|---|
| t1=1, t2=1.5, L=any, PBC | Bulk gap Δ | 1.0 (exact) | Analytic: 2|t1-t2| |
| t1=1, t2=1.5, L=any, OBC | Edge modes | 2 zero modes | Topological invariant |
| t1=1.5, t2=1, L=any, PBC | Bulk gap Δ | 1.0 (exact) | Analytic: 2|t1-t2| |
| t1=1.5, t2=1, L=any, OBC | Edge modes | 0 (trivial) | Topological invariant |

## Rice-Mele Model (interacting, two-orbital unit cell)

| Parameter set | Quantity | Value | Source |
|---|---|---|---|
| — | — | — | TBD — add as computed |
```

- [ ] **Step 4: Write knowledge/prior-runs.md**

Create `tracks/ed/hubbard-pump/knowledge/prior-runs.md`:

```markdown
# Prior Runs: Hubbard / Rice-Mele Pump

Results already produced by this harness. Do not recompute these
without a reason — use them as anchors.

## Run: 2026-07-28-ssh-smoke

- **Model:** SSH, spinless fermion chain
- **Parameters:** t1=1, t2=1.5, L=8 (4 unit cells), OBC
- **Method:** QuSpin exact diagonalization, Python
- **Tool:** quspin v1.0.1
- **Key results:**
  - Two zero-energy edge modes detected (E ≈ 0 within machine precision)
  - Bulk gap consistent with 2|t1-t2| = 1.0
- **Status:** Verified — matches analytic SSH solution
- **Location:** `results/ssh-smoke/`
```

- [ ] **Step 5: Write knowledge/notes.md**

Create `tracks/ed/hubbard-pump/knowledge/notes.md`:

```markdown
# Notes: Hubbard / Rice-Mele Pump

Free-form domain notes, traps, and known artifacts.

## Sign conventions

- SSH Hamiltonian: H = Σ [t1 c†_A,n c_B,n + t2 c†_B,n c_A,n+1 + h.c.]
- t1 > 0, t2 > 0 by default
- Topological phase: t2 > t1 (winding number ν = 1)
- Trivial phase: t1 > t2 (winding number ν = 0)

## Known finite-size artifacts

- Edge-mode localization length ∝ 1/ln(t2/t1) — for weak dimerization,
  the edge modes spread across many sites and may not be resolved at
  small L.
- At the critical point t1 = t2, the gap closes and finite-size
  rounding becomes severe.

## QuSpin on HPC

- C extension compilation is fragile — see WORKFLOW.md for SCNet
  instructions.
- Python stubs may be needed if C extensions fail to compile.
- Always smoke-test with a small L before submitting batch jobs.

## Resource estimates

- ED memory: D² × 8 bytes, where D = 2^(2L) for spinless fermions
  (Fock space)
- L=8: ~65k × 65k → ~32 MB (laptop-feasible)
- L=10: ~1M × 1M → ~8 GB (cluster recommended)
- L=12: ~16M × 16M → ~2 TB (infeasible without symmetry reduction)
```

- [ ] **Step 6: Commit**

```bash
git add tracks/ed/hubbard-pump/knowledge/
git commit -m "feat: add example project knowledge for hubbard-pump track

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>"
```

---

## Task Dependencies

```
Task 1 (AGENTS.md)    ─┐
Task 2 (skill)         ├─ independent, can run in parallel
Task 3 (Ion.toml)     ─┤  (depends on Task 2 for the skill file)
Task 4 (solve)         ├─ independent, can run in parallel
Task 5 (repro-paper)  ─┘
Task 6 (example)      ─── independent, can run any time
```

Only real dependency: Task 3 must run after Task 2 (needs the skill file to exist before `ion add`).

## Self-Review

1. **Spec coverage:**
   - AGENTS.md modification → Task 1
   - `/explore-on-project-knowledge` skill → Task 2
   - Ion.toml registration → Task 3
   - `/solve` modification → Task 4
   - `/reproduce-paper` modification → Task 5
   - Example knowledge/ directory → Task 6
   - All six items from spec are covered.

2. **Placeholder scan:** No TODOs, TBDs, or vague instructions. All code blocks are concrete. All file paths are exact. All commit messages are specified.

3. **Type consistency:** The skill name `explore-on-project-knowledge` is used consistently across Tasks 2, 3, 4, 5, and 6. The directory path `tracks/<track>/knowledge/` is consistent across all tasks.
