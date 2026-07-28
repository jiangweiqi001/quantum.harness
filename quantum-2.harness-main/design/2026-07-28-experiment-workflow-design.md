# Design Spec: Numerical Experiment Workflow (`/experiment`)

**Date:** 2026-07-28
**Status:** draft
**Branch:** `version-1.0002`

## Problem

Current harness workflows are either one-shot (`/solve`) or paper-reproduction-centric
(`/reproduce-paper`). Neither owns the **code maturity lifecycle** — taking a
physics idea from prototype to production-scale parameter sweep. Without explicit
phase awareness, agents default to one-shot behavior: build a script, run it, report
a number, and stop. This produces results the user didn't want, at quality they
didn't ask for.

The user needs a workflow where the agent **knows which development stage it is in**
and calibrates its behavior accordingly: exploratory and fast during MVP, rigorous
during verification, disciplined during refactoring, and production-minded during
scanning.

## Design

Two-layer architecture:

- **Orchestrator** (`/experiment`) — owns the lifecycle, enforces gates, passes
  context between phases. Entry point for new projects.
- **Phase skills** (`/experiment-mvp`, `/experiment-verify`, etc.) — each owns one
  stage. Independently invocable via direct command. Receives context from
  orchestrator via state file, returns results via the same file.

### Phase flow

```
MVP ──[gate]──> Verify ──[gate]──> Smoke ──[gate]──> Refactor ──[gate]──> Organize ──[gate]──> Scan
```

All gates are **user-gated** — the orchestrator pauses at each gate, presents results,
and waits for user confirmation. User chooses: `continue`, `redo` (re-run current phase),
or `abort`.

### Phase responsibilities

| Phase | Skill | Input | Output | Gate question |
|---|---|---|---|---|
| 1. MVP | `/experiment-mvp` | Physical model description | A running script | "Does the physics look right?" |
| 2. Verify | `/experiment-verify` | MVP script + benchmark source | Pass/fail against benchmark | "Matches known values?" |
| 3. Smoke | `/experiment-smoke` | Verified script | Small-size run: no crashes, plausible output | "No crashes, magnitudes correct?" |
| 4. Refactor | `/experiment-refactor` | Working script | Clean, modular code | "Readable and maintainable?" |
| 5. Organize | `/experiment-organize` | Refactored code | src/test/scripts/configs layout | "Structure reasonable?" |
| 6. Scan | `/experiment-scan` | Organized codebase | Production parameter sweep results | "Data sufficient for publication?" |

### Context passing: `experiment.json`

The orchestrator maintains a state file at `tracks/<track>/experiment.json`. Each
phase skill reads its input from this file and writes its output back to it. The
orchestrator updates the `phase` field after each gate.

```json
{
  "track": "tracks/ed/hubbard-pump",
  "model": "Rice-Mele",
  "method": "ED",
  "phase": "verify",
  "phases": {
    "mvp": {
      "status": "done",
      "script": "run_model.py",
      "notes": "User notes from gate review"
    },
    "verify": {
      "status": "active",
      "benchmark_source": "analytic: 2|t1-t2|",
      "benchmark_value": 1.0,
      "result": null
    }
  }
}
```

**Rules:**
- Each phase can only read its own input block and write its own output block.
- The `phase` field is only updated by the orchestrator after gate confirmation.
- Phase skills check `phases.<name>.status == "active"` on start; if not `"active"`,
  they report an error.
- On completion, phase skills set `phases.<name>.status = "done"` and populate their
  output fields.
- Orchestrator on `redo`: resets the phase's `status` to `"active"` and clears its
  output fields.

### Orchestrator (`/experiment`) behavior

1. **Detect state.** Check `tracks/<track>/experiment.json`. If absent, initialize
   from phase 1 (MVP). If present, resume from the phase marked `"active"`.
2. **Dispatch phase.** Invoke the appropriate phase skill. Pass the state file path.
3. **Present results at gate.** After phase completes, show a compact summary:
   what was done, key output, any concerns.
4. **Ask gate question.** User chooses `continue / redo / abort`.
5. **Advance or repeat.** On `continue`, mark phase done, advance `phase` field,
   dispatch next phase. On `redo`, reset phase and re-dispatch. On `abort`, save
   state and exit.
6. **Completion.** After Scan phase gate passes, mark experiment complete. Offer
   writeup handoff.

### Phase skills

Each phase skill (`/experiment-mvp`, `/experiment-verify`, etc.) is independently
invocable. When invoked directly (not via orchestrator), it looks for
`experiment.json` in the current track; if absent, it asks the user to provide the
necessary inputs and creates a minimal state file.

**`/experiment-mvp`:**
- Build a minimal working script from the user's physics description.
- Use the model card + method card for defaults (Hamiltonian form, solver choice).
- Script must be self-contained and runnable with one command.
- Output: path to script, one-line description, one successful run result.

**`/experiment-verify`:**
- Run the MVP script at a parameter point with a known benchmark value.
- Check against: analytic limit, published value, or prior harness run (from
  `tracks/<track>/knowledge/benchmarks.md`).
- Compute relative error. Pass if within stated tolerance; fail with diagnostic
  otherwise.
- If no benchmark exists, flag this and suggest the user provide one or skip.

**`/experiment-smoke`:**
- Run at the smallest non-trivial system size.
- Verify: no exceptions, memory within budget, runtime plausible.
- Check output sanity: energy negative for AFM, gap non-negative, order parameter
  in [0,1] where bounded.
- If the model is solvable in some limit, run a limit check.

**`/experiment-refactor`:**
- Transform a working monolithic script into modular code.
- Target structure (the user's specified layout):
  ```
  <project>/
  ├── src/
  │   ├── __init__.py
  │   ├── model.py          # Model class
  │   ├── algorithms.py     # Solver methods
  │   └── observables.py    # Energy, order parameters, etc.
  ├── scripts/
  │   ├── run.py            # Main entry point
  │   └── plot.py           # Plotting
  ├── tests/
  │   └── test_model.py
  ├── configs/
  │   └── default.yaml
  └── README.md
  ```
- Extract model definition, solver, and observables into separate modules.
- Preserve exact numerical output — refactored code must reproduce the verified
  result bit-for-bit.
- Add README.md with: model, method, usage, dependencies.

**`/experiment-organize`:**
- Verify the refactored structure matches the target layout.
- Ensure `__init__.py` exports are clean.
- Write `configs/default.yaml` with all parameters explicit.
- Ensure `tests/` exist and pass.
- Update track `knowledge/prior-runs.md` with the verified result.

**`/experiment-scan`:**
- Define the parameter grid from user input + model card defaults.
- Compose with `/parameter-scan` for grid enumeration and collection.
- Compose with `/using-slurm` for cluster execution if needed.
- Monitor progress, surface intermediate results.
- Final output: scan plot, table, convergence diagnostics.

### Orchestrator ↔ phase interface

Phase skills are invoked by the orchestrator or directly by the user. The contract:

1. Phase skill reads `tracks/<track>/experiment.json` on start.
2. Phase skill checks `phases.<phase>.status == "active"`.
3. Phase skill executes, writes output fields into `phases.<phase>`.
4. Phase skill sets `phases.<phase>.status = "done"`.
5. Phase skill reports a compact summary to stdout (for the orchestrator to
   present at gate).

When invoked directly (not via orchestrator), the phase skill additionally:
- Creates `experiment.json` if absent (prompting for minimal inputs).
- Reports completion but does NOT advance to the next phase — that is the
  orchestrator's job.

### File inventory

| File | Action | Purpose |
|---|---|---|
| `skills/experiment/SKILL.md` | Create | Orchestrator skill |
| `skills/experiment-mvp/SKILL.md` | Create | Phase 1 |
| `skills/experiment-verify/SKILL.md` | Create | Phase 2 |
| `skills/experiment-smoke/SKILL.md` | Create | Phase 3 |
| `skills/experiment-refactor/SKILL.md` | Create | Phase 4 |
| `skills/experiment-organize/SKILL.md` | Create | Phase 5 |
| `skills/experiment-scan/SKILL.md` | Create | Phase 6 |
| `Ion.toml` | Modify | Register 7 new skills |
| `AGENTS.md` | Modify | Add experiment workflow to Repository Layout section |

### What is NOT changed

- `.knowledge/` — zero modifications.
- Any existing skill (`/solve`, `/reproduce-paper`, `/parameter-scan`, etc.).
- `Makefile`.
- Existing track files.

### Self-Review

1. **Placeholder scan:** No TODOs or incomplete sections.
2. **Internal consistency:** `experiment.json` schema matches orchestrator behavior
   and phase skill contracts. Phase flow order is consistent throughout.
3. **Scope:** Seven new skills, one state file, one AGENTS.md edit. No refactoring
   of existing code. Scope is a single coherent workflow — not multiple subsystems.
4. **Ambiguity check:** Gate behavior (continue/redo/abort), state file schema, and
   phase skill contracts are all specified concretely. Refactor target layout is
   explicit.
