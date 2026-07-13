# LaTeX Academic Report Template — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a custom LaTeX class (`academic-report.cls`) with `paper` and `assignment` modes, plus two example templates, using a deep navy & gold color scheme.

**Architecture:** A single `.cls` file extending `article`, loaded by thin template `.tex` files. Mode is selected via class option. All styling lives in the class file — templates only contain content.

**Tech Stack:** LaTeX2e, standard TeX Live packages: `amsmath`, `amsthm`, `amssymb`, `geometry`, `fancyhdr`, `hyperref`, `xcolor`, `titlesec`, `listings`, `biblatex`, `lmodern`, `caption`, `environ`, `tcolorbox`, `tikz`

## Global Constraints

- Class option `paper` (default) enables preprint features; `assignment` enables coursework features
- Colors: Deep Navy `1B2A4A`, Muted Gold `C8A951`, Pale Gold `F5EDD6`, Ice Blue `EEF2F7`, Near-black `1A1A1A`
- Typography: Latin Modern Roman for body/math, Latin Modern Sans for headings, Latin Modern Mono for code
- All packages must be standard TeX Live (no exotic dependencies)
- Templates must compile without errors on a standard TeX Live distribution

---

### Task 1: Create academic-report.cls — core setup

**Files:**
- Create: `academic-report.cls`

**Interfaces:**
- Produces: class `academic-report` with options `paper` (default) and `assignment`, all core packages loaded, colors defined, typography configured, page geometry set

- [ ] **Step 1: Write the class identification and option handling**

Create `academic-report.cls`:

```latex
% academic-report.cls — Custom class for physics/math academic reports
% Options: paper (default), assignment
\NeedsTeXFormat{LaTeX2e}
\ProvidesClass{academic-report}[2026/07/13 v1.0 Academic Report Class]

% --- Option parsing ---
\newif\if@assignment
\@assignmentfalse
\DeclareOption{assignment}{\@assignmenttrue}
\DeclareOption{paper}{\@assignmentfalse}
\DeclareOption*{\PassOptionsToClass{\CurrentOption}{article}}
\ExecuteOptions{paper}
\ProcessOptions\relax

% --- Base class ---
\LoadClass[12pt,a4paper]{article}
```

- [ ] **Step 2: Add package loading**

Append to `academic-report.cls`:

```latex
% --- Packages ---
\RequirePackage[utf8]{inputenc}
\RequirePackage[T1]{fontenc}
\RequirePackage{lmodern}              % Latin Modern fonts
\RequirePackage{amsmath,amsthm,amssymb}
\RequirePackage[margin=1in]{geometry}
\RequirePackage{fancyhdr}
\RequirePackage[svgnames,dvipsnames]{xcolor}
\RequirePackage{titlesec}
\RequirePackage{listings}
\RequirePackage[
  backend=biber,
  style=authoryear,
  sorting=nyt,
  maxbibnames=99,
  giveninits=true
]{biblatex}
\RequirePackage[colorlinks=true,linkcolor=accent,citecolor=accent,urlcolor=accent]{hyperref}
\RequirePackage[font={small,color=primary}]{caption}
\RequirePackage{environ}
\RequirePackage{tcolorbox}
\RequirePackage{tikz}
\tcbuselibrary{skins,breakable}
```

- [ ] **Step 3: Define colors and typography**

Append to `academic-report.cls`:

```latex
% --- Colors ---
\definecolor{primary}{HTML}{1B2A4A}      % Deep Navy
\definecolor{accent}{HTML}{C8A951}       % Muted Gold
\definecolor{paleGold}{HTML}{F5EDD6}     % Pale Gold
\definecolor{iceBlue}{HTML}{EEF2F7}      % Ice Blue
\definecolor{bodyText}{HTML}{1A1A1A}     % Near-black

% --- Typography ---
\renewcommand{\familydefault}{\rmdefault}
\color{bodyText}

% Sans-serif for headings
\renewcommand{\sfdefault}{lmss}
\renewcommand{\ttdefault}{lmtt}
```

- [ ] **Step 4: Configure page geometry and headers/footers**

Append to `academic-report.cls`:

```latex
% --- Page style ---
\fancypagestyle{mainstyle}{
  \fancyhf{}
  \fancyhead[L]{\sffamily\footnotesize\color{primary}\@title}
  \fancyhead[R]{\sffamily\footnotesize\color{primary}\thepage}
  \renewcommand{\headrulewidth}{0.4pt}
  \renewcommand{\headrule}{\hbox to\headwidth{\color{primary}\leaders\hrule height \headrulewidth\hfill}}
  \fancyfoot{}
}
\pagestyle{mainstyle}

% Plain page style (chapter pages etc.)
\fancypagestyle{plain}{
  \fancyhf{}
  \fancyhead[L]{\sffamily\footnotesize\color{primary}\@title}
  \fancyhead[R]{\sffamily\footnotesize\color{primary}\thepage}
  \renewcommand{\headrulewidth}{0.4pt}
  \renewcommand{\headrule}{\hbox to\headwidth{\color{primary}\leaders\hrule height \headrulewidth\hfill}}
  \fancyfoot{}
}
```

- [ ] **Step 5: Commit**

```bash
git add academic-report.cls
git commit -m "feat: add academic-report.cls with core setup, options, and styling"
```

---

### Task 2: Add section formatting, theorem environments, and code blocks

**Files:**
- Modify: `academic-report.cls`

**Interfaces:**
- Consumes: `primary`, `accent`, `paleGold`, `iceBlue` color definitions; `tcolorbox`, `tikz`, `titlesec`, `listings` packages from Task 1
- Produces: styled sections/subsections, theorem/lemma/definition/proof/remark/corollary environments, code listing style

- [ ] **Step 1: Add section heading formatting**

Append to `academic-report.cls` (before `\endinput`):

```latex
% --- Section formatting ---
% Section: navy sans-serif bold, gold rule below
\titleformat{\section}
  {\sffamily\Large\bfseries\color{primary}}
  {\thesection}
  {1em}
  {}
  [\vspace{-0.6em}\rule{\textwidth}{0.8pt}\color{accent}]

% Subsection: navy sans-serif, no rule
\titleformat{\subsection}
  {\sffamily\large\bfseries\color{primary}}
  {\thesubsection}
  {1em}
  {}

% Subsubsection: navy sans-serif italic
\titleformat{\subsubsection}
  {\sffamily\normalsize\bfseries\color{primary}}
  {\thesubsubsection}
  {1em}
  {}

\titlespacing*{\section}{0pt}{2ex plus 1ex minus 0.5ex}{2ex plus 0.5ex}
\titlespacing*{\subsection}{0pt}{1.5ex plus 0.5ex minus 0.3ex}{1ex plus 0.3ex}
\titlespacing*{\subsubsection}{0pt}{1.2ex plus 0.3ex minus 0.2ex}{0.8ex plus 0.2ex}
```

- [ ] **Step 2: Add theorem/definition/proof environments with tcolorbox**

Append to `academic-report.cls`:

```latex
% --- Theorem environments (ice-blue shaded box, navy left border) ---
\newtcolorbox{thmbox}[1][]{
  enhanced,
  breakable,
  colback=iceBlue,
  colframe=primary,
  left=4mm,
  right=2mm,
  top=2mm,
  bottom=2mm,
  sharp corners,
  borderline west={3pt}{0pt}{primary},
  before skip=10pt,
  after skip=10pt,
  #1
}

% Theorem
\newtheorem{theorem}{Theorem}[section]
\newenvironment{theoremblock}[1][]
  {\begin{thmbox}\begin{theorem}#1}
  {\end{theorem}\end{thmbox}}

% Lemma
\newtheorem{lemma}[theorem]{Lemma}
\newenvironment{lemmablock}[1][]
  {\begin{thmbox}\begin{lemma}#1}
  {\end{lemma}\end{thmbox}}

% Corollary
\newtheorem{corollary}[theorem]{Corollary}
\newenvironment{corollaryblock}[1][]
  {\begin{thmbox}\begin{corollary}#1}
  {\end{corollary}\end{thmbox}}

% Definition (no numbering)
\newtheorem*{definition}{Definition}
\newenvironment{definitionblock}[1][]
  {\begin{thmbox}\begin{definition}#1}
  {\end{definition}\end{thmbox}}

% Remark
\newtheorem*{remark}{Remark}
\newenvironment{remarkblock}[1][]
  {\begin{thmbox}\begin{remark}#1}
  {\end{remark}\end{thmbox}}

% Proof (gold left border, ends with qed box)
\newtcolorbox{proofbox}{
  enhanced,
  breakable,
  colback=iceBlue,
  colframe=primary,
  left=4mm,
  right=2mm,
  top=2mm,
  bottom=2mm,
  sharp corners,
  borderline west={3pt}{0pt}{accent},
  before skip=10pt,
  after skip=10pt,
}
\newenvironment{proofblock}[1][]
  {\begin{proofbox}\textbf{\sffamily\color{primary}Proof.}#1}
  {\hfill$\blacksquare$\end{proofbox}}
```

- [ ] **Step 3: Add code listing style**

Append to `academic-report.cls`:

```latex
% --- Code listings ---
\lstset{
  backgroundcolor=\color{paleGold},
  basicstyle=\ttfamily\footnotesize\color{primary},
  keywordstyle=\color{primary}\bfseries,
  commentstyle=\color{gray}\itshape,
  stringstyle=\color{accent},
  numbers=left,
  numberstyle=\tiny\color{gray},
  numbersep=8pt,
  frame=single,
  framerule=0.4pt,
  rulecolor=\color{accent},
  breaklines=true,
  breakatwhitespace=true,
  tabsize=2,
  showstringspaces=false,
  captionpos=b,
  abovecaptionskip=8pt,
  belowcaptionskip=4pt,
}
```

- [ ] **Step 4: Add equation numbering per section**

Append to `academic-report.cls`:

```latex
% --- Equation numbering per section ---
\numberwithin{equation}{section}
```

- [ ] **Step 5: Commit**

```bash
git add academic-report.cls
git commit -m "feat: add section formatting, theorem environments, code listings"
```

---

### Task 3: Add title block and mode-specific features

**Files:**
- Modify: `academic-report.cls`

**Interfaces:**
- Consumes: class option `if@assignment`, all color/style definitions from Tasks 1–2
- Produces: custom `\maketitle`, paper-mode abstract environment, assignment-mode `\course`/`\instructor`/`\duedate` commands, `problem`/`solution` environments

- [ ] **Step 1: Add custom title block with navy rules and gold accent**

Append to `academic-report.cls`:

```latex
% --- Title block ---
\renewcommand{\maketitle}{%
  \begin{center}
    % Top navy rule
    \rule{\textwidth}{1.5pt}\\[0.3em]
    % Gold accent line (shorter)
    {\color{accent}\rule{0.6\textwidth}{0.8pt}}\\[1em]
    % Title
    {\sffamily\LARGE\bfseries\color{primary}\@title\\[0.5em]}
    % Author
    {\large\color{bodyText}\@author\\[0.3em]}
    % Date
    {\footnotesize\color{gray}\@date}
    \\[0.5em]
    % Bottom navy rule
    \rule{\textwidth}{0.4pt}
  \end{center}
  \vspace{1em}
}
```

- [ ] **Step 2: Add paper-mode features (abstract, appendix)**

Append to `academic-report.cls`:

```latex
% --- Paper mode features ---
\if@assignment\else
  % Abstract environment: pale gold background, navy left border
  \newtcolorbox{abstractblock}{
    enhanced,
    breakable,
    colback=paleGold,
    colframe=primary,
    boxrule=0.4pt,
    left=4mm,
    right=2mm,
    top=2mm,
    bottom=2mm,
    sharp corners,
    borderline west={3pt}{0pt}{accent},
    before skip=12pt,
    after skip=12pt,
  }
  \renewenvironment{abstract}
    {\begin{abstractblock}\textbf{\sffamily\color{primary}Abstract.}\quad}
    {\end{abstractblock}}

  % Author/affiliation commands
  \newcommand{\affiliation}[1]{\gdef\@affiliation{#1}}
  \newcommand{\@affiliation}{}
  \let\oldauthor\author
  \renewcommand{\author}[2][]{%
    \oldauthor{#2\\\small\textcolor{gray}{#1}}%
  }
\fi
```

- [ ] **Step 3: Add assignment-mode features (metadata, problem/solution)**

Append to `academic-report.cls`:

```latex
% --- Assignment mode features ---
\if@assignment
  % Course metadata
  \newcommand{\course}[1]{\gdef\@course{#1}}
  \newcommand{\@course}{}
  \newcommand{\instructor}[1]{\gdef\@instructor{#1}}
  \newcommand{\@instructor}{}
  \newcommand{\duedate}[1]{\gdef\@duedate{#1}}
  \newcommand{\@duedate}{}

  % Submission info box
  \newcommand{\submissioninfo}{%
    \ifx\@course\@empty\else
      \begin{tcolorbox}[
        enhanced,
        colback=iceBlue,
        colframe=primary,
        boxrule=0.4pt,
        sharp corners,
        before skip=8pt,
        after skip=12pt,
      ]
        \sffamily\small\color{primary}
        \begin{tabular}{@{}ll}
          \textbf{Course:} & \@course \\
          \textbf{Instructor:} & \@instructor \\
          \textbf{Due Date:} & \@duedate \\
        \end{tabular}
      \end{tcolorbox}
    \fi
  }

  % Redefine maketitle for assignments to include submission info
  \let\oldmaketitle\maketitle
  \renewcommand{\maketitle}{%
    \oldmaketitle
    \submissioninfo
  }

  % Problem environment (numbered)
  \newcounter{problem}[section]
  \renewcommand{\theproblem}{\arabic{section}.\arabic{problem}}
  \newenvironment{problem}[1][]
    {\refstepcounter{problem}%
     \vspace{10pt}%
     \noindent\textbf{\sffamily\color{primary}Problem \theproblem.}%
     \ifx#1\empty\else\ \textit{#1}\fi%
     \par\vspace{4pt}}
    {\vspace{4pt}}

  % Solution environment (ice-blue shaded)
  \newtcolorbox{solutionblock}{
    enhanced,
    breakable,
    colback=iceBlue,
    colframe=primary,
    boxrule=0.4pt,
    left=4mm,
    right=2mm,
    top=2mm,
    bottom=2mm,
    sharp corners,
    borderline west={3pt}{0pt}{accent},
    before skip=8pt,
    after skip=8pt,
  }
  \newenvironment{solution}
    {\begin{solutionblock}\textbf{\sffamily\color{primary}Solution.}\quad}
    {\end{solutionblock}}
\fi
```

- [ ] **Step 4: Add `\endinput` at end of class file**

Append to `academic-report.cls`:

```latex
\endinput
```

- [ ] **Step 5: Commit**

```bash
git add academic-report.cls
git commit -m "feat: add title block, paper mode abstract, assignment mode features"
```

---

### Task 4: Create template-paper.tex

**Files:**
- Create: `template-paper.tex`

**Interfaces:**
- Consumes: `academic-report.cls` with `paper` option
- Produces: compilable example showing paper mode with title, abstract, sections, theorem, equation, figure, code, and bibliography

- [ ] **Step 1: Write template-paper.tex**

Create `template-paper.tex`:

```latex
% template-paper.tex — Example paper/preprint using academic-report class
\documentclass[paper]{academic-report}

% --- Metadata ---
\title{On the Emergent Properties of\\Quantum Many-Body Systems}
\author{Author Name}
\date{\today}

% --- Bibliography ---
\addbibresource{references.bib}

\begin{document}

\maketitle

\begin{abstract}
We present a unified framework for analyzing emergent phenomena in quantum
many-body systems. Using a combination of tensor network methods and field
theoretic techniques, we demonstrate how entanglement structure governs the
low-energy effective description. Our results provide new insights into the
classification of quantum phases of matter and the nature of topological order.
\end{abstract}

\section{Introduction}

The study of quantum many-body systems has revealed a rich landscape of emergent
phenomena that cannot be understood from microscopic considerations alone
\cite{witten2018}. From fractional quantum Hall states to spin liquids, the
collective behavior of strongly correlated electrons continues to challenge our
theoretical understanding.

In this work, we develop a framework based on the following principles:
\begin{enumerate}
  \item Entanglement entropy as an organizing principle for quantum phases;
  \item Tensor network states as variational ansatze capturing the relevant
        low-energy physics;
  \item Field theory descriptions emerging from the continuum limit.
\end{enumerate}

\section{Theoretical Framework}

\subsection{Tensor Network Representation}

Consider a quantum state $|\psi\rangle$ on a lattice $\Lambda$ with local
Hilbert space dimension $d$. A matrix product state (MPS) representation takes
the form:

\begin{equation}
  |\psi\rangle = \sum_{\{s_i\}} \mathrm{Tr}\left[A_1^{s_1} A_2^{s_2} \cdots A_N^{s_N}\right] |s_1 s_2 \cdots s_N\rangle
  \label{eq:mps}
\end{equation}

where each $A_i^{s_i}$ is a $\chi \times \chi$ matrix, and the bond dimension
$\chi$ controls the amount of entanglement captured by the ansatz.

\begin{definitionblock}[Entanglement Entropy]
For a bipartition of the system into regions $A$ and $B$, the entanglement
entropy is defined as:
\begin{equation}
  S_A = -\mathrm{Tr}(\rho_A \ln \rho_A)
  \label{eq:entanglement}
\end{equation}
where $\rho_A = \mathrm{Tr}_B|\psi\rangle\langle\psi|$ is the reduced density
matrix.
\end{definitionblock}

\subsection{Area Law and Its Violations}

\begin{theoremblock}[Area Law]
For gapped local Hamiltonians in $d$ spatial dimensions, the ground state
entanglement entropy satisfies:
\begin{equation}
  S_A = \alpha\, \mathrm{Area}(\partial A) + O(|\partial A|^{d-2})
  \label{eq:arealaw}
\end{equation}
where $\alpha$ is a non-universal constant.
\end{theoremblock}

\begin{proofblock}
The proof follows from the exponential decay of correlations in gapped systems
and the Lieb-Robinson bound. Consider a local Hamiltonian $H = \sum_x h_x$ with
gap $\Delta$. The ground state projector can be expressed via:
\begin{equation}
  |\psi_0\rangle\langle\psi_0| = \frac{1}{2\pi i} \oint_\Gamma \frac{dz}{z - H}
\end{equation}
where $\Gamma$ encircles the ground state energy.
\end{proofblock}

\begin{remarkblock}
The area law is violated in several important cases, including gapless systems
in one dimension (logarithmic violation) and systems with Fermi surfaces
(area $\times \log$ violation).
\end{remarkblock}

\section{Applications}

\subsection{Topological Order}

For topologically ordered phases, the entanglement entropy contains a universal
constant term $\gamma$ called the topological entanglement entropy:

\begin{equation}
  S_A = \alpha L - \gamma + O(L^{-1})
\end{equation}

This term characterizes the anyonic content of the phase and is robust against
local perturbations \cite{kitaev2006}.

\subsection{Numerical Results}

We apply our framework to the Heisenberg model on a square lattice:

\begin{figure}[htbp]
  \centering
  \begin{tikzpicture}[scale=0.8]
    \draw[->] (0,0) -- (6,0) node[right] {$J_2/J_1$};
    \draw[->] (0,0) -- (0,4) node[above] {$\Delta$};
    \draw[thick, primary] (0,0.5) -- (2,2.5);
    \draw[thick, primary, dashed] (2,2.5) -- (5,1.5);
    \node at (1,1.5) {N\'eel};
    \node at (3.5,1.8) {Spin Liquid};
    \node at (1,1) {\footnotesize ordered};
    \node at (3.5,1.3) {\footnotesize disordered};
  \end{tikzpicture}
  \caption{Phase diagram of the $J_1$-$J_2$ Heisenberg model showing the
           N\'eel ordered phase and the putative spin liquid region.}
  \label{fig:phasediagram}
\end{figure}

Figure \ref{fig:phasediagram} shows the obtained phase diagram.

\subsection{Implementation}

The following Python code computes the entanglement spectrum:

\begin{lstlisting}[language=Python, caption={Computing entanglement spectrum from MPS}]
import numpy as np

def entanglement_spectrum(mps, bond_index):
    """Compute the entanglement spectrum at a given bond.

    Args:
        mps: Matrix product state tensors
        bond_index: Index of the bond to cut

    Returns:
        Sorted entanglement spectrum (singular values)
    """
    # Contract left and right environments
    left_env = contract_left(mps, bond_index)
    right_env = contract_right(mps, bond_index + 1)

    # Compute singular value decomposition
    U, S, Vh = np.linalg.svd(left_env @ right_env)

    return np.sort(S)[::-1]
\end{lstlisting}

\section{Conclusion}

We have presented a systematic framework for understanding emergent phenomena
in quantum many-body systems. The combination of tensor network methods with
field theoretic insights provides a powerful toolkit for exploring strongly
correlated phases of matter.

\begin{corollaryblock}[Classification]
The classification of gapped quantum phases in two dimensions reduces to the
classification of modular tensor categories, providing a direct connection
between entanglement structure and topological order.
\end{corollaryblock}

\appendix
\section{Detailed Derivations}

The full derivation of the Lieb-Robinson bound used in Theorem 1 follows
standard techniques. We refer the reader to \cite{bravyi2006} for details.

\printbibliography[title={References}]

\end{document}
```

- [ ] **Step 2: Commit**

```bash
git add template-paper.tex
git commit -m "feat: add paper mode template with full example content"
```

---

### Task 5: Create template-assignment.tex

**Files:**
- Create: `template-assignment.tex`

**Interfaces:**
- Consumes: `academic-report.cls` with `assignment` option
- Produces: compilable example showing assignment mode with metadata, problems, solutions, equations, and code

- [ ] **Step 1: Write template-assignment.tex**

Create `template-assignment.tex`:

```latex
% template-assignment.tex — Example assignment using academic-report class
\documentclass[assignment]{academic-report}

% --- Metadata ---
\title{Problem Set 3: Entanglement Measures}
\author{Student Name}
\date{\today}

% --- Course Info ---
\course{PHYS 641: Advanced Quantum Mechanics}
\instructor{Prof. S. Chen}
\duedate{October 15, 2026}

\begin{document}

\maketitle

\section{Entanglement Entropy of Bipartite Systems}

\begin{problem}[Entanglement entropy of a two-spin system]
Consider two spin-$\frac{1}{2}$ particles in the singlet state:
\begin{equation}
  |\psi\rangle = \frac{1}{\sqrt{2}}\left(|\uparrow\downarrow\rangle -
  |\downarrow\uparrow\rangle\right)
\end{equation}
Compute the entanglement entropy of one spin by tracing out the other.
\end{problem}

\begin{solution}
The density matrix of the full system is:
\begin{equation}
  \rho = |\psi\rangle\langle\psi| =
  \frac{1}{2}\left(|\uparrow\downarrow\rangle\langle\uparrow\downarrow| +
  |\downarrow\uparrow\rangle\langle\downarrow\uparrow| -
  |\uparrow\downarrow\rangle\langle\downarrow\uparrow| -
  |\downarrow\uparrow\rangle\langle\uparrow\downarrow|\right)
\end{equation}

Tracing out the second spin yields the reduced density matrix:
\begin{equation}
  \rho_A = \mathrm{Tr}_B(\rho) =
  \frac{1}{2}\left(|\uparrow\rangle\langle\uparrow| +
  |\downarrow\rangle\langle\downarrow|\right) =
  \begin{pmatrix} \frac{1}{2} & 0 \\ 0 & \frac{1}{2} \end{pmatrix}
\end{equation}

The entanglement entropy is:
\begin{equation}
  S_A = -\mathrm{Tr}(\rho_A \ln \rho_A) =
  -2 \cdot \frac{1}{2}\ln\frac{1}{2} = \ln 2
\end{equation}

This is the maximum possible entanglement for a two-qubit system.
\end{solution}

\begin{problem}[Area law in the AKLT chain]
The Affleck-Kennedy-Lieb-Tasaki (AKLT) chain has the following Hamiltonian:
\begin{equation}
  H = \sum_i \left[\mathbf{S}_i \cdot \mathbf{S}_{i+1} +
  \frac{1}{3}(\mathbf{S}_i \cdot \mathbf{S}_{i+1})^2\right]
\end{equation}

Using the matrix product state representation, argue that the ground state
entanglement entropy saturates to a constant $S = 2\ln 2$ for large block
sizes, consistent with the area law in one dimension.
\end{problem}

\begin{solution}
The AKLT ground state can be written as an MPS with bond dimension $\chi = 2$.
The virtual space at each bond encodes a spin-$\frac{1}{2}$ degree of freedom,
so the Schmidt decomposition at any cut has exactly two non-zero singular
values $\lambda_1 = \lambda_2 = 1/\sqrt{2}$.

For a block of length $L$, the entanglement entropy is bounded by:
\begin{equation}
  S_A(L) \leq \ln(\chi^2) = \ln 4 = 2\ln 2
\end{equation}

For large $L$, this bound is saturated. Since the entropy is independent of
block length, it satisfies the one-dimensional area law: $S_A \sim \text{const}$
for gapped systems.
\end{solution}

\section{Computational Methods}

\begin{problem}[Numerical entanglement spectrum]
Write a short program to compute the entanglement spectrum of the transverse
field Ising model at criticality. The Hamiltonian is:
\begin{equation}
  H = -J\sum_i \left(\sigma_i^z \sigma_{i+1}^z + g\sigma_i^x\right)
\end{equation}
with $g = 1$ at the critical point.
\end{problem}

\begin{solution}
The following code uses exact diagonalization for a small chain:

\begin{lstlisting}[language=Python, caption={Exact diagonalization of TFI model}]
import numpy as np
from scipy.linalg import eigh

def tfi_hamiltonian(N, J=1.0, g=1.0):
    """Build the transverse field Ising Hamiltonian."""
    dim = 2**N
    H = np.zeros((dim, dim))

    for i in range(N):
        j = (i + 1) % N
        # sigma^z_i sigma^z_{j+1}
        for s in range(dim):
            spin_i = (s >> i) & 1
            spin_j = (s >> j) & 1
            H[s, s] += -J * (1 - 2*spin_i) * (1 - 2*spin_j)
        # g * sigma^x_i
        for s in range(dim):
            s_flip = s ^ (1 << i)
            H[s, s_flip] += -J * g

    return H

N = 10
H = tfi_hamiltonian(N, g=1.0)
energies, states = eigh(H)

# Ground state
psi_0 = states[:, 0]
print(f"Ground state energy: {energies[0]:.6f}")
print(f"Energy gap: {energies[1] - energies[0]:.6f}")
\end{lstlisting}

The entanglement spectrum can then be obtained by reshaping the ground state
vector into a matrix and performing SVD.
\end{solution}

\end{document}
```

- [ ] **Step 2: Commit**

```bash
git add template-assignment.tex
git commit -m "feat: add assignment mode template with example problems"
```

---

### Task 6: Compile templates and verify

**Files:**
- No new files; compile `template-paper.tex` and `template-assignment.tex`

**Interfaces:**
- Consumes: `academic-report.cls`, both template `.tex` files
- Produces: successful compilation (PDF outputs), or error identification and fixes

- [ ] **Step 1: Check LaTeX installation**

```bash
which pdflatex || which latex
```

If neither found, note that user needs TeX Live installed to compile.

- [ ] **Step 2: Compile template-paper.tex**

```bash
cd /Users/dune/Desktop/pre-workspace && pdflatex -interaction=nonstopmode template-paper.tex
```

Expected: No fatal errors. Warnings about missing `references.bib` are acceptable.

- [ ] **Step 3: Compile template-assignment.tex**

```bash
cd /Users/dune/Desktop/pre-workspace && pdflatex -interaction=nonstopmode template-assignment.tex
```

Expected: No fatal errors. Successful PDF output.

- [ ] **Step 4: Fix any compilation errors**

If compilation fails, inspect log files for errors, fix the class or template, recompile until clean.

- [ ] **Step 5: Commit any fixes**

```bash
git add -A
git commit -m "fix: compilation fixes for templates"
```

---

### Task 7: Final review and README

**Files:**
- Create: `README.md` (only if compilation succeeds and templates are verified)

**Interfaces:**
- Consumes: verified working templates from Task 6
- Produces: README with usage instructions, cleanup of auxiliary files

- [ ] **Step 1: Clean auxiliary files**

```bash
cd /Users/dune/Desktop/pre-workspace && rm -f *.aux *.log *.out *.toc *.bbl *.bcf *.blg *.run.xml
```

- [ ] **Step 2: Verify final file structure**

```bash
cd /Users/dune/Desktop/pre-workspace && ls -la academic-report.cls template-paper.tex template-assignment.tex
```

Expected: All three files present.

- [ ] **Step 3: Update gitignore for LaTeX aux files**

Create or update `.gitignore`:

```
*.aux
*.log
*.out
*.toc
*.bbl
*.bcf
*.blg
*.run.xml
*.synctex.gz
*.fls
*.fdb_latexmk
.DS_Store
```

- [ ] **Step 4: Commit**

```bash
git add .gitignore
git commit -m "chore: add .gitignore for LaTeX auxiliary files"
```

---

## Self-Review

1. **Spec coverage:** All spec features covered — class options (Task 1), colors/typography (Task 1), sections (Task 2), theorems (Task 2), code listings (Task 2), title block (Task 3), paper features (Task 3), assignment features (Task 3), both templates (Tasks 4–5), compilation verification (Task 6), cleanup (Task 7).

2. **Placeholder scan:** No TODOs, TBDs, or vague references. All code is explicit and complete.

3. **Type consistency:** Command names consistent across tasks — `\course`, `\instructor`, `\duedate` defined in Task 3 and used in Task 5 template. Colors `primary`, `accent`, `paleGold`, `iceBlue` defined in Task 1 and used throughout Tasks 2–5.
