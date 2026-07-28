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
