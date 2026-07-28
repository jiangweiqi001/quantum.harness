# Design Spec: Project Knowledge Layer (`tracks/<track>/knowledge/`)

**Date:** 2026-07-28
**Status:** draft
**Branch:** `version-1.0001`

## Problem

`.knowledge/` (~22 MB) mixes two kinds of content:

- **General reference** — Hamiltonian definitions, symmetries, phase diagrams, method
  recommendations. Belongs in `.knowledge/models/`, `.knowledge/physics/`,
  `.knowledge/methods/`.
- **Project-cooked knowledge** — numerical benchmarks from prior work, harness-computed
  results at specific parameters, sign-convention traps, known finite-size artifacts.
  Currently inlined in model cards, hard to discover and maintain.

An agent working on a concrete track (e.g., `tracks/ed/hubbard-pump`) has no
standard place to find "what benchmarks should I check against?" or "what did we
already compute?" — it must scan the entire model card or rerun from scratch.

## Design

Add an opt-in **project knowledge layer** under `tracks/<track>/knowledge/`. Three
mechanisms ensure agents discover and use it:

1. **AGENTS.md convention** — declares that agents MUST read the project knowledge
   README before working on a track.
2. **Skill initialization step** — `/solve` and `/reproduce-paper` check for
   `knowledge/README.md` during setup.
3. **Dedicated command** — `/explore-on-project-knowledge` reads and surfaces
   project knowledge on demand.

### Directory layout

```
tracks/<track>/
├── knowledge/                   ← project-cooked knowledge (NEW)
│   ├── README.md                ← required entry point; agent reads this first
│   ├── benchmarks.md            ← published numerical values from prior work
│   ├── prior-runs.md            ← results already produced by this harness
│   └── notes.md                 ← free-form domain notes, traps, artifacts
├── runs/                        ← existing
├── results/                     ← existing
├── solutions/                   ← existing
└── README.md                    ← existing track description
```

`knowledge/` is opt-in — tracks without it work exactly as before. No existing
directory is modified.

### `knowledge/README.md` format

YAML frontmatter for machine readability, then structured sections:

```markdown
---
track: <track-name>
model: <model-name>
methods: [<method-list>]
last_updated: <YYYY-MM-DD>
---

# Project Knowledge: <track-name>

## Benchmarks
Key published values with source references. Full table in [benchmarks.md](./benchmarks.md).

## Prior Runs (this harness)
Key results already computed. Full log in [prior-runs.md](./prior-runs.md).

## Critical Notes
Sign-convention traps, known finite-size artifacts, recommended parameter ranges.

## File Index
- [benchmarks.md](./benchmarks.md) — complete benchmark table
- [prior-runs.md](./prior-runs.md) — detailed run records
- [notes.md](./notes.md) — free-form domain notes
```

Agent behavior: read this file first, then follow links to detailed files as
needed. Cite benchmark values when proposing a setup. Check against them during
verification. Treat disagreement as a signal to re-examine the setup.

### AGENTS.md modification

Insert a new section after "Knowledge Base Role" and before "Verification practice":

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

~15 lines, purely additive. No existing section is touched.

### `/explore-on-project-knowledge` skill

New local skill at `skills/explore-on-project-knowledge/SKILL.md`.

**Behavior:**

- Receives a track path (e.g., `tracks/ed/hubbard-pump`) from the user or
  calling skill.
- Reads `knowledge/README.md` as the entry point.
- Recursively reads linked files (`benchmarks.md`, `prior-runs.md`, `notes.md`).
- Outputs a compact summary:
  - Known benchmark values (table).
  - Prior harness results (parameter → value).
  - Critical traps and notes.
- If no `knowledge/` directory exists for the track, reports that and falls back
  to general `.knowledge/` cards.

**Trigger:** user types `/explore-on-project-knowledge` or an agent invokes it
programmatically when entering a track.

**Registration in Ion.toml:** one new line:
```toml
explore-on-project-knowledge = { type = "local" }
```

### Modifications to `/solve` and `/reproduce-paper`

Each skill's initialization phase gains a check step (~3 lines):

```markdown
## Project Knowledge Check

Before setting up the computation, check whether
`tracks/<track>/knowledge/README.md` exists. If it does, invoke
`/explore-on-project-knowledge` and surface the benchmarks and prior
runs to the user during setup confirmation.
```

This is added to the "Intake" section of `/solve` and step 0 ("Clarify the
problem") of `/reproduce-paper`. Both additions are ~3-4 lines, purely additive.

### File inventory

| File | Action | Impact |
|---|---|---|
| `AGENTS.md` | Insert ~15 lines after "Knowledge Base Role" | Low — additive |
| `skills/explore-on-project-knowledge/SKILL.md` | Create new file | None — new |
| `Ion.toml` | Add 1 line for the new skill | Minimal |
| `skills/solve/SKILL.md` | Add ~3 lines to Intake section | Low — additive |
| `skills/reproduce-paper/SKILL.md` | Add ~3 lines to step 0 | Low — additive |
| `tracks/<track>/knowledge/` | Per-track, opt-in | None — new |

**What is NOT changed:**

- `.knowledge/` — zero modifications.
- Any method skill (`method-ed`, `method-mps`, etc.) — untouched.
- Any tool skill (`using-quspin`, `using-itensors`, etc.) — untouched.
- `Makefile` — no new targets.
- Existing track files (`runs/`, `results/`, `solutions/`) — untouched.

## Usage Example

Agent session working on `tracks/ed/hubbard-pump`:

```
1. User: "/solve Rice-Mele pump at L=8, delta=0.5"
2. Agent (solve intake): checks tracks/ed/hubbard-pump/knowledge/README.md → exists
3. Agent: invokes /explore-on-project-knowledge
4. Agent surfaces: "Known benchmark: gap = 0.342 at delta=0.5, L=8 (Ref X).
   Prior harness run: gap = 0.341 ± 0.002 (2026-07-27)."
5. Agent proposes setup anchored to these values.
6. User confirms. Computation runs. Result checked against benchmark.
```

## Migration

No migration needed for existing tracks. `knowledge/` directories are created
on-demand when a track accumulates cooked knowledge worth preserving.

## Self-Review

- **Placeholder scan:** No TODOs or incomplete sections.
- **Internal consistency:** AGENTS.md, skill behaviors, and README format all
  describe the same three knowledge categories (benchmarks, prior runs, notes).
- **Scope:** Focused — one new directory convention, one new skill, two minor
  skill edits, one AGENTS.md insertion. No refactoring of existing systems.
- **Ambiguity check:** All terms are concrete. "MUST read" is the behavioral
  contract. README format is specified by example.
