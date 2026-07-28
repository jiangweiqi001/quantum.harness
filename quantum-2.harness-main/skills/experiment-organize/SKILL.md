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
