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
