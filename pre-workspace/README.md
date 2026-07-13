# Academic Report Template

A LaTeX template for physics/mathematics academic reports with a modern deep navy & gold visual design.

## Quick Start

**Paper / Preprint mode (default):**

```latex
\documentclass[papermode]{academic-report}

\title{Your Paper Title}
\author{Author Name}
\date{\today}

\begin{document}
\maketitle
\begin{abstract}
Your abstract here.
\end{abstract}
\section{Introduction}
...
\end{document}
```

**Assignment / Coursework mode:**

```latex
\documentclass[assignment]{academic-report}

\title{Problem Set Title}
\author{Student Name}
\date{\today}
\course{PHYS 641: Advanced Quantum Mechanics}
\instructor{Prof. Name}
\duedate{October 15, 2026}

\begin{document}
\maketitle
\begin{problem}[Problem description]
...
\end{problem}
\begin{solution}
...
\end{solution}
\end{document}
```

## Features

- **Two modes:** `papermode` (default, for preprints/research) and `assignment` (for coursework)
- **Deep navy & gold color scheme** with ice-blue shaded theorem boxes
- **Theorem environments:** theorem, lemma, corollary, definition, remark, proof
- **Assignment environments:** problem (numbered) and solution (shaded)
- **Code listings** with pale gold background via `listings`
- **Latin Modern fonts:** Roman for body/math, Sans for headings, Mono for code
- **Custom title block** with navy rules and gold accent
- **Bibliography** via `biblatex` (author-year style)

## Files

- `academic-report.cls` — the class file (all styling logic)
- `template-paper.tex` — example paper/preprint
- `template-assignment.tex` — example assignment

## Requirements

A standard TeX Live distribution with the following packages (all included in TeX Live full):
`amsmath`, `amsthm`, `amssymb`, `geometry`, `fancyhdr`, `hyperref`, `xcolor`, `titlesec`, `listings`, `biblatex`, `lmodern`, `caption`, `environ`, `tcolorbox`, `tikz`

## Color Palette

| Role | Color | Hex |
|------|-------|-----|
| Primary (headings, rules) | Deep Navy | `#1B2A4A` |
| Accent (links, highlights) | Muted Gold | `#C8A951` |
| Secondary (subtle blocks) | Pale Gold | `#F5EDD6` |
| Surface (boxed environments) | Ice Blue | `#EEF2F7` |
| Body text | Near-black | `#1A1A1A` |
