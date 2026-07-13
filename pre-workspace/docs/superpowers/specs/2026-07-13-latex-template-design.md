# LaTeX Academic Report Template — Design Spec

**Date:** 2026-07-13
**Status:** Approved

## Overview

A custom LaTeX class (`academic-report.cls`) for physics/math academic reports in English. Supports two modes via class options: `paper` (preprint/research) and `assignment` (coursework). Visual identity: deep navy blue & gold, modern bold styling with hybrid typography.

## File Structure

```
pre-workspace/
├── academic-report.cls          # Custom class file (all styling)
├── template-paper.tex           # Paper/preprint mode example
├── template-assignment.tex      # Assignment mode example
```

## Color Palette

| Role                    | Color        | Hex       |
|-------------------------|--------------|-----------|
| Primary (headings, rules, boxes) | Deep Navy    | `1B2A4A`  |
| Accent (highlights, links)       | Muted Gold   | `C8A951`  |
| Secondary accent (subtle blocks) | Pale Gold    | `F5EDD6`  |
| Surface (boxed environments)     | Ice Blue     | `EEF2F7`  |
| Body text                         | Near-black   | `1A1A1A`  |

## Typography

- **Body + math:** Latin Modern Roman (`lmodern`, `fontspec` not required)
- **Headings:** Latin Modern Sans
- **Monospace:** Latin Modern Mono
- Title in bold sans-serif, deep navy, separated from body by a thin gold rule
- Section headings: navy, sans-serif, with gold rule below
- Subsections: navy, sans-serif, no rule

## Class Options

| Option         | Default | Effect                                      |
|----------------|---------|---------------------------------------------|
| `paper`        | yes     | Full preprint mode (abstract, bibliography) |
| `assignment`   | no      | Coursework mode (problem/solution, metadata)|

Switch via:

```latex
\documentclass[paper]{academic-report}     % or [assignment]
```

## Features — Both Modes

- **Title block:** navy rules top and bottom, gold accent line, title in navy sans-serif bold, author/subtitle in muted text
- **Sections:** numbered, navy sans-serif, thin gold rule below; subsections navy sans-serif no rule
- **Theorem environments:** ice-blue shaded box with navy left border (`theorem`, `lemma`, `definition`, `proof`, `remark`, `corollary`)
- **Math:** `amsmath` loaded, equations numbered per section
- **Figures/tables:** captions in small navy text, centered
- **Hyperlinks:** muted gold via `hyperref`, no colored boxes around links
- **Page header/footer:** navy thin rule at top, gold page numbers at bottom
- **Code blocks:** pale gold background, navy text, courtesy of `listings`
- **Bibliography:** `biblatex` with a simple author-year style

## Features — Paper Mode

- Abstract environment: pale gold background block with navy left border
- Appendix support via `\appendix` command
- Author + affiliation + date fields

## Features — Assignment Mode

- `problem` environment: numbered, navy heading
- `solution` environment: ice-blue shaded block
- Course metadata fields: `\course`, `\instructor`, `\duedate`
- Submission info box in the header area

## Dependencies (all standard TeX Live packages)

`amsmath`, `amsthm`, `amssymb`, `geometry`, `fancyhdr`, `hyperref`, `xcolor`, `titlesec`, `listings`, `biblatex`, `lmodern`, `caption`, `environ`, `tcolorbox`, `tikz`

## Self-Review

- No TBDs or incomplete sections
- All features consistent with the physics/math focus
- Scope: one class file + two template examples, no overreach
- All color hex values defined; no ambiguous direction
