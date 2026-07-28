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
