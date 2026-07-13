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

% --- Section formatting ---
% Section: navy sans-serif bold, gold rule below
\titleformat{\section}
  {\sffamily\Large\bfseries\color{primary}}
  {\thesection}
  {1em}
  {}
  [\vspace{-0.6em}{\color{accent}\rule{\textwidth}{0.8pt}}]

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

% --- Equation numbering per section ---
\numberwithin{equation}{section}
\endinput
