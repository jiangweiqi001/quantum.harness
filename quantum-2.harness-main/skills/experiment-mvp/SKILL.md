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
