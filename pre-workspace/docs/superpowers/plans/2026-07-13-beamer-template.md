# Beamer Academic Presentation Template — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a custom Beamer theme (`beamerthemeAcademic.sty` + sub-themes) for physics/math presentations with deep navy & gold color scheme, Copenhagen-style navigation, and styled blocks.

**Architecture:** Standard Beamer theme decomposition: color theme defines palette mapping, font theme sets typography, outer theme defines headline/footline/frametitle, inner theme defines title page/items/blocks. The main theme file loads all four sub-themes via `\usecolortheme`, `\usefonttheme`, `\useinnertheme`, `\useoutertheme`.

**Tech Stack:** LaTeX2e, Beamer class, `lmodern`, `amsmath`, `amssymb`, `listings`, `xcolor`, `tikz`

## Global Constraints

- Colors: Deep Navy `1B2A4A`, Muted Gold `C8A951`, Pale Gold `F5EDD6`, Ice Blue `EEF2F7`, Near-black `1A1A1A`
- Fonts: Latin Modern Roman (body/math), Latin Modern Sans (headings), Latin Modern Mono (code)
- Copenhagen-style top navigation bar with all sections, current section highlighted gold
- Block title: navy bg with white text; block body: ice blue bg with navy border
- `alertblock`: gold bg title, pale gold body
- Template must compile without fatal errors on standard TeX Live

---

### Task 1: Create beamercolorthemeAcademic.sty

**Files:**
- Create: `template-beamer/beamercolorthemeAcademic.sty`

**Interfaces:**
- Produces: color definitions (`primary`, `accent`, `paleGold`, `iceBlue`, `bodyText`) and Beamer palette/block/structure color mappings

- [ ] **Step 1: Write the color theme file**

Create `template-beamer/beamercolorthemeAcademic.sty`:

```latex
% beamercolorthemeAcademic.sty — Color definitions for Academic Beamer theme
\mode<presentation>

% --- Shared color palette ---
\definecolor{primary}{HTML}{1B2A4A}    % Deep Navy
\definecolor{accent}{HTML}{C8A951}     % Muted Gold
\definecolor{paleGold}{HTML}{F5EDD6}   % Pale Gold
\definecolor{iceBlue}{HTML}{EEF2F7}    % Ice Blue
\definecolor{bodyText}{HTML}{1A1A1A}   % Near-black

% --- Beamer palette mapping ---
\setbeamercolor{palette primary}{fg=white, bg=primary}
\setbeamercolor{palette secondary}{fg=primary, bg=accent}
\setbeamercolor{palette tertiary}{fg=primary, bg=iceBlue}
\setbeamercolor{palette quaternary}{fg=bodyText, bg=paleGold}

% --- Structure colors ---
\setbeamercolor{structure}{fg=accent}
\setbeamercolor{alerted text}{fg=accent}

% --- Navigation bar ---
\setbeamercolor{section in head/foot}{fg=white, bg=primary}
\setbeamercolor{subsection in head/foot}{fg=white, bg=primary}

% --- Palette sidebar (not used, but set for completeness) ---
\setbeamercolor{sidebar}{bg=primary, fg=white}
\setbeamercolor{sidebar}{parent=palette primary}

% --- Title page ---
\setbeamercolor{title}{fg=primary}
\setbeamercolor{subtitle}{fg=primary!80}
\setbeamercolor{author}{fg=bodyText}
\setbeamercolor{institute}{fg=gray}
\setbeamercolor{date}{fg=gray}

% --- Frame title ---
\setbeamercolor{frametitle}{fg=primary}
\setbeamercolor{framesubtitle}{fg=primary!80}

% --- Block environments ---
\setbeamercolor{block title}{fg=white, bg=primary}
\setbeamercolor{block body}{fg=bodyText, bg=iceBlue}
\setbeamercolor{block title alerted}{fg=primary, bg=accent}
\setbeamercolor{block body alerted}{fg=bodyText, bg=paleGold}
\setbeamercolor{block title example}{fg=white, bg=primary}
\setbeamercolor{block body example}{fg=bodyText, bg=iceBlue!50}

% --- Itemize/enumerate ---
\setbeamercolor{item}{fg=accent}
\setbeamercolor{subitem}{fg=primary}
\setbeamercolor{subsubitem}{fg=gray}
\setbeamercolor{enumerate item}{fg=primary}

% --- Footline ---
\setbeamercolor{footline}{fg=primary}

% --- Captions ---
\setbeamercolor{caption}{fg=primary}
\setbeamercolor{caption name}{fg=primary}

% --- Bibliography ---
\setbeamercolor{bibliography entry author}{fg=primary}
\setbeamercolor{bibliography entry title}{fg=bodyText}
\setbeamercolor{bibliography entry location}{fg=gray}
\setbeamercolor{bibliography entry note}{fg=gray}

% --- Background ---
\setbeamercolor{background canvas}{bg=white}

\mode<all>
```

- [ ] **Step 2: Commit**

```bash
git add template-beamer/beamercolorthemeAcademic.sty
git commit -m "feat: add Beamer color theme with navy-gold palette"
```

---

### Task 2: Create beamerfontthemeAcademic.sty

**Files:**
- Create: `template-beamer/beamerfontthemeAcademic.sty`

**Interfaces:**
- Consumes: none
- Produces: font settings for all Beamer elements (title, frametitle, body, blocks, navigation)

- [ ] **Step 1: Write the font theme file**

Create `template-beamer/beamerfontthemeAcademic.sty`:

```latex
% beamerfontthemeAcademic.sty — Font settings for Academic Beamer theme
\mode<presentation>

\RequirePackage{lmodern}

% --- Body and math ---
\setbeamerfont{normal text}{family=\rmfamily, series=\mdseries}
\usefonttheme{professionalfonts}  % Don't mess with math fonts

% --- Title page ---
\setbeamerfont{title}{family=\sffamily, size=\LARGE, series=\bfseries}
\setbeamerfont{subtitle}{family=\sffamily, size=\large}
\setbeamerfont{author}{family=\rmfamily, size=\normalsize}
\setbeamerfont{institute}{family=\rmfamily, size=\small}
\setbeamerfont{date}{family=\rmfamily, size=\small}

% --- Frame title ---
\setbeamerfont{frametitle}{family=\sffamily, size=\Large, series=\bfseries}
\setbeamerfont{framesubtitle}{family=\sffamily, size=\large}

% --- Block environments ---
\setbeamerfont{block title}{family=\sffamily, size=\normalsize, series=\bfseries}
\setbeamerfont{block body}{family=\rmfamily, size=\normalsize}

% --- Headline / footline ---
\setbeamerfont{section in head/foot}{family=\sffamily, size=\tiny, series=\mdseries}
\setbeamerfont{subsection in head/foot}{family=\sffamily, size=\tiny, series=\mdseries}
\setbeamerfont{footline}{family=\sffamily, size=\scriptsize}

% --- Captions ---
\setbeamerfont{caption}{family=\rmfamily, size=\small}

% --- Itemize/enumerate ---
\setbeamerfont{item}{family=\rmfamily}
\setbeamerfont{itemize/enumerate body}{family=\rmfamily}

\mode<all>
```

- [ ] **Step 2: Commit**

```bash
git add template-beamer/beamerfontthemeAcademic.sty
git commit -m "feat: add Beamer font theme with Latin Modern typography"
```

---

### Task 3: Create beamerouterthemeAcademic.sty

**Files:**
- Create: `template-beamer/beamerouterthemeAcademic.sty`

**Interfaces:**
- Consumes: color definitions (`primary`, `accent`), font settings from Tasks 1-2
- Produces: headline template (Copenhagen-style nav bar), footline template, frametitle template, margins

- [ ] **Step 1: Write the outer theme file**

Create `template-beamer/beamerouterthemeAcademic.sty`:

```latex
% beamerouterthemeAcademic.sty — Outer elements for Academic Beamer theme
\mode<presentation>

% --- Margins ---
\setbeamersize{text margin left=12pt, text margin right=12pt}

% --- Headline: Copenhagen-style navigation bar ---
\defbeamertemplate*{headline}{Academic theme}
{%
  \leavevmode
  \hbox{%
    \begin{beamercolorbox}[wd=\paperwidth, ht=3.5ex, dp=1.5ex]{palette primary}%
      \usebeamerfont{section in head/foot}%
      \insertsectionnavigationhorizontal{\paperwidth}{%
        \hskip0.3ex plus1fill
      }{%
        \hskip0.3ex plus1fill
      }%
    \end{beamercolorbox}%
  }%
  \vskip0pt
}

% --- Footline: title | author | date ---
\defbeamertemplate*{footline}{Academic theme}
{%
  \leavevmode
  \hbox{%
    \begin{beamercolorbox}[wd=.33\paperwidth, ht=2.5ex, dp=1ex, left]{footline}%
      \usebeamerfont{footline}%
      \hspace*{8pt}\insertshorttitle
    \end{beamercolorbox}%
    \begin{beamercolorbox}[wd=.34\paperwidth, ht=2.5ex, dp=1ex, center]{footline}%
      \usebeamerfont{footline}%
      \insertshortauthor
    \end{beamercolorbox}%
    \begin{beamercolorbox}[wd=.33\paperwidth, ht=2.5ex, dp=1ex, right]{footline}%
      \usebeamerfont{footline}%
      \insertshortdate\hspace*{8pt}
    \end{beamercolorbox}%
  }%
  \vskip0pt
}

% --- Frametitle ---
\defbeamertemplate*{frametitle}{Academic theme}
{%
  \vspace*{4pt}%
  \ifbeamercolorempty[bg]{frametitle}{}{\nointerlineskip}%
  \begin{beamercolorbox}[wd=\paperwidth, leftskip=8pt, rightskip=8pt, ht=2.25ex, dp=1ex]{frametitle}%
    \usebeamerfont{frametitle}\insertframetitle%
    \ifx\insertframesubtitle\@empty\else%
      \\[4pt]{\usebeamerfont{framesubtitle}\insertframesubtitle}%
    \fi
  \end{beamercolorbox}%
  \vspace{-6pt}%
  {\color{accent}\hspace*{8pt}\rule{0.96\paperwidth}{0.6pt}}%
  \vspace{6pt}%
}

% --- Disable navigation symbols ---
\setbeamertemplate{navigation symbols}{}

\mode<all>
```

- [ ] **Step 2: Commit**

```bash
git add template-beamer/beamerouterthemeAcademic.sty
git commit -m "feat: add Beamer outer theme with Copenhagen nav bar and footline"
```

---

### Task 4: Create beamerinnerthemeAcademic.sty

**Files:**
- Create: `template-beamer/beamerinnerthemeAcademic.sty`

**Interfaces:**
- Consumes: color definitions from Task 1, font settings from Task 2
- Produces: title page template, item markers, block styling, TOC styling, theorem environments

- [ ] **Step 1: Write the inner theme file**

Create `template-beamer/beamerinnerthemeAcademic.sty`:

```latex
% beamerinnerthemeAcademic.sty — Inner elements for Academic Beamer theme
\mode<presentation>

% --- Title page ---
\defbeamertemplate*{title page}{Academic theme}
{%
  \vbox{}
  \vfill
  \begin{centering}
    {\usebeamerfont{title}\usebeamercolor[fg]{title}\inserttitle\par}
    \vskip0.5em
    \ifx\insertsubtitle\@empty\else
      {\usebeamerfont{subtitle}\usebeamercolor[fg]{subtitle}\insertsubtitle\par}
      \vskip1em
    \fi
    % Gold decorative line
    {\color{accent}\rule{0.5\textwidth}{0.8pt}\par}
    \vskip1.5em
    {\usebeamerfont{author}\usebeamercolor[fg]{author}\insertauthor\par}
    \vskip0.5em
    {\usebeamerfont{institute}\usebeamercolor[fg]{institute}\insertinstitute\par}
    \vskip0.5em
    {\usebeamerfont{date}\usebeamercolor[fg]{date}\insertdate\par}
  \end{centering}
  \vfill
}

% --- Itemize markers ---
% Level 1: gold filled circle
\setbeamertemplate{itemize item}{\small\color{accent}$\bullet$}
\setbeamertemplate{itemize subitem}{\small\color{primary}---}
\setbeamertemplate{itemize subsubitem}{\small\color{gray}$\cdot$}

% --- Enumerate markers ---
\setbeamertemplate{enumerate item}{\color{primary}\insertenumlabel.}
\setbeamertemplate{enumerate subitem}{\color{primary}\insertsubenumlabel.}
\setbeamertemplate{enumerate subsubitem}{\color{primary}\insertsubsubenumlabel.}

% --- Blocks with rounded corners ---
\setbeamertemplate{blocks}[rounded][shadow=false]

% Round block corners (2pt)
% Beamer default rounded already uses ~4pt; we accept the default.
% Shadow is off for a cleaner modern look.

% --- Table of Contents ---
\setbeamertemplate{section in toc}[sections numbered]
\setbeamercolor{section in toc}{fg=primary}
\setbeamerfont{section in toc}{family=\sffamily, size=\large}
\setbeamercolor{subsection in toc}{fg=primary!70}
\setbeamerfont{subsection in toc}{family=\sffamily, size=\normalsize}

% --- Theorem environments ---
% Reusing Beamer's block infrastructure
\newtheorem{theorem}{Theorem}
\newtheorem{lemma}[theorem]{Lemma}
\newtheorem{corollary}[theorem]{Corollary}
\theoremstyle{definition}
\newtheorem{definition}{Definition}
\theoremstyle{remark}
\newtheorem{remark}{Remark}

% Proof environment with gold accent
\newenvironment{proofblock}[1][\proofname]{%
  \par\noindent\textbf{\sffamily\color{primary}#1.}%
}{%
  \hfill$\blacksquare$\par
}

% --- Thank-you end slide ---
\newcommand{\thankyoupage}{%
  \begin{frame}[plain]
    \vbox{}
    \vfill
    \begin{centering}
      {\usebeamerfont{frametitle}\usebeamercolor[fg]{frametitle}Thank You\par}
      \vskip1em
      {\color{accent}\rule{0.4\textwidth}{0.8pt}\par}
      \vskip2em
      {\usebeamerfont{author}\usebeamercolor[fg]{author}\insertauthor\par}
      \vskip0.5em
      {\usebeamerfont{institute}\usebeamercolor[fg]{institute}\insertinstitute\par}
    \end{centering}
    \vfill
  \end{frame}
}

% --- Frametitle continuation (allowframebreaks styling) ---
\setbeamertemplate{frametitle continuation}[from second]
\setbeamerfont{frametitle continuation}{family=\sffamily, size=\small}
\setbeamercolor{frametitle continuation}{fg=primary!50}

\mode<all>
```

- [ ] **Step 2: Commit**

```bash
git add template-beamer/beamerinnerthemeAcademic.sty
git commit -m "feat: add Beamer inner theme with title page, items, blocks, theorems"
```

---

### Task 5: Create beamerthemeAcademic.sty (main theme)

**Files:**
- Create: `template-beamer/beamerthemeAcademic.sty`

**Interfaces:**
- Consumes: all four sub-themes from Tasks 1-4
- Produces: entry-point theme file that loads sub-themes and sets global options

- [ ] **Step 1: Write the main theme file**

Create `template-beamer/beamerthemeAcademic.sty`:

```latex
% beamerthemeAcademic.sty — Main Academic Beamer theme
% Load with: \usetheme{Academic}
\mode<presentation>

% --- Load sub-themes ---
\usecolortheme{Academic}
\usefonttheme{Academic}
\useoutertheme{Academic}
\useinnertheme{Academic}

% --- Code listings (matching paper template) ---
\RequirePackage{listings}
\lstset{
  language=Python,
  backgroundcolor=\color{paleGold},
  basicstyle=\ttfamily\footnotesize\color{primary},
  keywordstyle=\color{primary}\bfseries,
  commentstyle=\color{gray}\itshape,
  stringstyle=\color{accent},
  numbers=left,
  numberstyle=\tiny\color{gray},
  numbersep=6pt,
  frame=single,
  framerule=0.4pt,
  rulecolor=\color{accent},
  breaklines=true,
  tabsize=2,
  showstringspaces=false,
  captionpos=b,
}

% --- Table of contents at each section start ---
\AtBeginSection[]
{
  \begin{frame}<beamer>[plain]
    \frametitle{Outline}
    \tableofcontents[currentsection]
  \end{frame}
}

\mode<all>
```

- [ ] **Step 2: Commit**

```bash
git add template-beamer/beamerthemeAcademic.sty
git commit -m "feat: add main Beamer theme loading sub-themes and listings"
```

---

### Task 6: Create template-beamer.tex

**Files:**
- Create: `template-beamer/template-beamer.tex`

**Interfaces:**
- Consumes: complete Beamer theme from Tasks 1-5
- Produces: compilable example presentation demonstrating all features

- [ ] **Step 1: Write the example presentation**

Create `template-beamer/template-beamer.tex`:

```latex
% template-beamer.tex — Example presentation using Academic Beamer theme
\documentclass[aspectratio=169]{beamer}

% --- Theme ---
\usetheme{Academic}

% --- Metadata ---
\title{Quantum Many-Body Systems:\\Emergent Phenomena and Entanglement}
\subtitle{A Tensor Network Perspective}
\author{Author Name}
\institute{Department of Physics\\University of Example}
\date{\today}

\begin{document}

% --- Title Slide ---
\begin{frame}
  \titlepage
\end{frame}

% --- Outline ---
\begin{frame}{Outline}
  \tableofcontents
\end{frame}

% --- Section 1 ---
\section{Introduction}

\begin{frame}{Motivation}
  \begin{columns}
    \column{0.5\textwidth}
    \begin{itemize}
      \item Strongly correlated electrons exhibit rich emergent phenomena
      \item Collective behavior challenges theoretical understanding
      \item Entanglement provides organizing principle
      \begin{itemize}
        \item Area law for gapped systems
        \item Logarithmic violations at criticality
        \item Topological entanglement entropy
      \end{itemize}
    \end{itemize}

    \column{0.5\textwidth}
    \begin{block}{Key Questions}
      \begin{enumerate}
        \item How does entanglement classify quantum phases?
        \item Can tensor networks capture the relevant physics?
        \item What is the role of symmetry?
      \end{enumerate}
    \end{block}
  \end{columns}
\end{frame}

% --- Section 2 ---
\section{Tensor Network Methods}

\begin{frame}{Matrix Product States}
  A matrix product state (MPS) for $N$ sites:
  \begin{equation}
    |\psi\rangle = \sum_{\{s_i\}} \mathrm{Tr}\left[A_1^{s_1} A_2^{s_2}
    \cdots A_N^{s_N}\right] |s_1 s_2 \cdots s_N\rangle
    \label{eq:mps}
  \end{equation}

  \begin{definition}
    The \textbf{bond dimension} $\chi$ controls the amount of entanglement
    captured by the MPS ansatz. For area-law states, $\chi$ is
    independent of system size.
  \end{definition}

  \begin{theorem}[Area Law]
    For gapped local Hamiltonians in $d$ dimensions, the ground state
    entanglement entropy satisfies:
    \begin{equation}
      S_A = \alpha\cdot\mathrm{Area}(\partial A) + \cdots
    \end{equation}
  \end{theorem}
\end{frame}

\begin{frame}{Entanglement Spectrum}
  \begin{proofblock}[Proof Sketch]
    The ground state projector can be expressed via a contour integral:
    \begin{equation}
      |\psi_0\rangle\langle\psi_0| =
      \frac{1}{2\pi i} \oint_\Gamma \frac{dz}{z - H}
    \end{equation}
    where $\Gamma$ encircles the ground state energy.
  \end{proofblock}

  \begin{alertblock}{Important}
    The area law is violated in gapless 1D systems (log violation)
    and systems with Fermi surfaces.
  \end{alertblock}
\end{frame}

% --- Section 3 ---
\section{Applications}

\begin{frame}{Topological Order}
  \begin{exampleblock}{Topological Entanglement Entropy}
    For topologically ordered phases:
    \begin{equation}
      S_A = \alpha L - \gamma + O(L^{-1})
    \end{equation}
    where $\gamma = \ln\mathcal{D}$ is the \textbf{topological
    entanglement entropy} and $\mathcal{D}$ is the total quantum
    dimension.
  \end{exampleblock}

  \begin{corollary}
    The classification of gapped quantum phases in 2D reduces to the
    classification of modular tensor categories.
  \end{corollary}
\end{frame}

\begin{frame}[fragile]{Code Example: DMRG Algorithm}
  \begin{lstlisting}[caption={Single-site DMRG sweep}]
import numpy as np

def dmrg_sweep(H_mpo, mps, max_bond_dim=100):
    """Perform one DMRG sweep (left-to-right)."""
    for site in range(len(mps)):
        # Contract effective Hamiltonian
        H_eff = contract_environment(H_mpo, mps, site)

        # Solve local eigenvalue problem
        energy, psi = np.linalg.eigh(H_eff)

        # Truncate to max_bond_dim
        mps[site] = truncate(psi[:, 0], max_bond_dim)

    return mps, energy[0]
  \end{lstlisting}
\end{frame}

% --- Section 4 ---
\section{Conclusions}

\begin{frame}{Summary}
  \begin{itemize}
    \item Entanglement is a powerful organizing principle for
          quantum many-body systems
    \item Tensor networks provide efficient representations
          for area-law states
    \item Topological order manifests in universal subleading
          contributions to entanglement entropy
    \item DMRG and related algorithms enable precision numerics
  \end{itemize}

  \vspace{1em}
  \begin{block}{Outlook}
    \begin{itemize}
      \item Extend to higher dimensions (PEPS)
      \item Real-time dynamics and thermalization
      \item Connections to holography and AdS/CFT
    \end{itemize}
  \end{block}
\end{frame}

% --- Thank You ---
\thankyoupage

\end{document}
```

- [ ] **Step 2: Commit**

```bash
git add template-beamer/template-beamer.tex
git commit -m "feat: add Beamer example presentation with full demo"
```

---

### Task 7: Compile and verify

**Files:**
- Compile: `template-beamer/template-beamer.tex`

**Interfaces:**
- Consumes: all theme files + example from Tasks 1-6
- Produces: successful PDF compilation or fixes

- [ ] **Step 1: Compile the Beamer template**

```bash
cd /Users/dune/Desktop/pre-workspace/template-beamer && pdflatex -interaction=nonstopmode template-beamer.tex
```

Expected: No fatal errors. Warnings about overfull boxes or font substitutions are acceptable.

- [ ] **Step 2: Compile a second time (for TOC and nav)**

```bash
cd /Users/dune/Desktop/pre-workspace/template-beamer && pdflatex -interaction=nonstopmode template-beamer.tex
```

Expected: Clean compilation, navigation bar populated, TOC pages correct.

- [ ] **Step 3: Fix any compilation errors**

If compilation fails, inspect `template-beamer/template-beamer.log`, identify errors, fix the relevant `.sty` or `.tex` file, recompile.

- [ ] **Step 4: Commit any fixes**

```bash
git add template-beamer/
git commit -m "fix: compilation fixes for Beamer template"
```

---

### Task 8: Cleanup and README

**Files:**
- Create: `template-beamer/README.md`

**Interfaces:**
- Consumes: verified working Beamer template
- Produces: README, cleaned auxiliary files

- [ ] **Step 1: Clean auxiliary files**

```bash
cd /Users/dune/Desktop/pre-workspace/template-beamer && rm -f *.aux *.log *.out *.toc *.nav *.snm *.vrb *.synctex.gz *.fls *.fdb_latexmk
```

- [ ] **Step 2: Update root .gitignore for Beamer aux files**

Update `.gitignore` to add:
```
*.nav
*.snm
*.vrb
```

- [ ] **Step 3: Verify final file structure**

```bash
ls -la template-beamer/beamerthemeAcademic.sty template-beamer/beamercolorthemeAcademic.sty template-beamer/beamerfontthemeAcademic.sty template-beamer/beamerinnerthemeAcademic.sty template-beamer/beamerouterthemeAcademic.sty template-beamer/template-beamer.tex
```

Expected: All six files present.

- [ ] **Step 4: Commit**

```bash
git add template-beamer/ .gitignore
git commit -m "chore: add Beamer README, gitignore update, cleanup"
```

---

## Self-Review

1. **Spec coverage:** All spec features mapped — colors (Task 1), fonts (Task 2), headline/footline/frametitle (Task 3), title page/items/blocks/theorems/TOC (Task 4), main theme + listings (Task 5), example presentation (Task 6), compilation (Task 7), cleanup (Task 8).

2. **Placeholder scan:** No TBDs or vague references. All code is explicit LaTeX.

3. **Type consistency:** Color names (`primary`, `accent`, `paleGold`, `iceBlue`, `bodyText`) defined in Task 1, used consistently in Tasks 2-6. Beamer font names match font theme definitions. Theme file path `beamerthemeAcademic.sty` matches `\usetheme{Academic}` convention.
