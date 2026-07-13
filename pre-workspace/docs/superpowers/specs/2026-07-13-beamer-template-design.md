# Beamer Academic Presentation Template — Design Spec

**Date:** 2026-07-13
**Status:** Approved

## Overview

A custom Beamer theme (`beamerthemeAcademic.sty`) for physics/math academic presentations. Shares the same deep navy & gold visual identity as the existing academic report template. Copenhagen-style top navigation bar, bottom info bar with progress, and styled block environments.

## File Structure

```
template-beamer/
├── beamerthemeAcademic.sty      # Main theme file (orchestrates sub-themes)
├── beamercolorthemeAcademic.sty # Color definitions and palette mapping
├── beamerfontthemeAcademic.sty  # Font settings
├── beamerinnerthemeAcademic.sty # Inner elements (title page, items, blocks)
├── beamerouterthemeAcademic.sty # Outer elements (headline, footline, frametitle)
└── template-beamer.tex          # Example presentation
```

`beamerthemeAcademic.sty` is the entry point: `\usetheme{Academic}` loads all sub-themes from the same directory.

## Color Palette (shared with academic-report.cls)

| Role                    | Color        | Hex       |
|-------------------------|--------------|-----------|
| Primary (headings, bars) | Deep Navy    | `1B2A4A`  |
| Accent (highlights, links)| Muted Gold   | `C8A951`  |
| Secondary (subtle blocks)| Pale Gold    | `F5EDD6`  |
| Surface (block bodies)   | Ice Blue     | `EEF2F7`  |
| Body text                | Near-black   | `1A1A1A`  |

## Beamer Palette Mapping

| Beamer Palette | Color | Used For |
|----------------|-------|----------|
| `palette primary` (bg=navy, fg=white) | Deep Navy | Top navigation bar background |
| `palette secondary` (bg=gold, fg=navy) | Muted Gold | Current section highlight in nav |
| `palette tertiary` (bg=iceBlue, fg=navy) | Ice Blue | Block title background |
| `palette quaternary` (bg=paleGold, fg=body) | Pale Gold | Example block background |
| `structure` | Muted Gold | Bullets, enumerate markers, citations, links |
| `alerted text` | Muted Gold | Alert text (not red, to match scheme) |
| Background | White | Slide background |
| Block body | Ice Blue | Theorem/definition/example block body |
| Block body alerted | Pale Gold | Alert block body |
| Block body example | Ice Blue (lighter) | Example block body |

## Typography

- **Body + math:** Latin Modern Roman (`lmodern`)
- **Headings:** Latin Modern Sans
- **Monospace:** Latin Modern Mono
- **Frametitle:** Sans-serif, bold, navy
- **Block title:** Sans-serif, bold, navy on iceBlue
- **Navigation bar:** Sans-serif, small, white on navy, current in gold

## Outer Theme (Slide Layout)

### Headline (Top)
- Copenhagen-style navigation bar
- Navy background full-width
- All section names listed horizontally, white text, small font
- Current section: **bold gold** text with thin gold underline
- Height: ~3.5ex
- Right corner: page number "N / Total" in white

### Footline (Bottom)
- Thin light gray rule across top
- Left: short title in navy (small, sans-serif)
- Center: author name in gray
- Right: date in gray
- Height: ~2.5ex

### Frametitle
- Sans-serif bold, navy color, large
- Subtitles in navy, slightly smaller
- Thin gold rule separating frametitle from content
- Left-aligned

## Inner Theme

### Title Page
- Centered layout
- Title: sans-serif, very large, bold, navy
- Subtitle: navy, smaller
- Author: body text, centered below title
- Date/institute: gray, at bottom
- Gold decorative line between title and author area

### Items (Bullets and Enumerations)
- Bullet markers: gold filled circles (level 1), navy dashes (level 2), gray dots (level 3)
- Enumerate markers: navy numbers
- Item text: body color

### Blocks
Three block types, all with rounded corners (2pt radius):

| Block Type | Title Style | Body Style |
|------------|-------------|------------|
| Standard `block` | Navy bg, white text | Ice blue bg, navy border |
| `alertblock` | Gold bg, navy text | Pale gold bg, gold border |
| `exampleblock` | Navy bg, white text | Ice blue bg (lighter), navy border |

### Theorem-like Environments
- Block-based, reusing Beamer's block infrastructure
- `theorem`, `lemma`, `corollary`: styled as standard block
- `definition`: styled as example block
- `proof`: styled as standard block with gold leftborder accent (via tcolorbox if needed)

### Table of Contents
- Section entries: navy, sans-serif
- Current section highlighted in gold
- Bullet or numbered markers optional

## Features

- `\maketitle` — custom title page
- `\tableofcontents` — styled TOC (optionally at each section start)
- `\section`, `\subsection` — reflected in navigation bar
- `{block}`, `{alertblock}`, `{exampleblock}` — custom styled
- Math support via `amsmath`
- Code listings via `listings` (matching paper template: paleGold bg, navy text)
- Figure/table support with navy captions
- Thank-you end page: centered "Thank You" in navy, gold line below

## Dependencies

Standard TeX Live: `beamer`, `lmodern`, `amsmath`, `amssymb`, `listings`, `xcolor`, `tikz`

## Self-Review

- No TBDs or incomplete sections
- Colors, fonts consistent with academic-report.cls
- File structure follows Beamer theme conventions (color/inner/outer/font sub-themes)
- Scope: one theme bundle + one example `.tex`, no overreach
- Copenhagen-style navigation chosen and specified
