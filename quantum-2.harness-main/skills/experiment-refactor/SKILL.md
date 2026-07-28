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
