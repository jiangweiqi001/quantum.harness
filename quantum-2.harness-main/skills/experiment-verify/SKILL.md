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
