# Academic Beamer Theme

A clean, modern Beamer theme designed for academic presentations. Features a warm gold/blue color palette and professional typography.

## Files

| File | Purpose |
|------|---------|
| `beamerthemeAcademic.sty` | Main theme — loads all sub-themes and configures listings |
| `beamercolorthemeAcademic.sty` | Color definitions (primary, accent, paleGold, iceBlue, etc.) |
| `beamerfontthemeAcademic.sty` | Font families and sizes |
| `beamerinnerthemeAcademic.sty` | Title page, itemize, blocks, theorems, TOC |
| `beamerouterthemeAcademic.sty` | Headline, footline, frametitle, sidebar, logo |
| `template-beamer.tex` | Example presentation demonstrating all features |

## Usage

```latex
\documentclass[aspectratio=169]{beamer}
\usetheme{Academic}

\title{Your Title}
\author{Your Name}
\institute{Your Institution}
\date{\today}

\begin{document}
\begin{frame}
  \titlepage
\end{frame}
\end{document}
```

Compile with `pdflatex`, `xelatex`, or `lualatex`:

```bash
pdflatex template-beamer.tex
```

## Features

- Custom title page with academic styling
- Section-based table of contents with progress highlighting
- Block, alertblock, exampleblock, proofblock, and theorem environments
- Code listings with `listings` package (matching paper template style)
- Thank-you slide via `\thankyoupage`
- 16:9 aspect ratio by default
