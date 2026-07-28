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
   For a 2D lattice: 2x2 or 3x3. Ask the user if ambiguous.
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
