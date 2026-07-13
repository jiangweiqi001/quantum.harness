% academic-report.cls — Custom class for physics/math academic reports
% Options: papermode (default), assignment
\NeedsTeXFormat{LaTeX2e}
\ProvidesClass{academic-report}[2026/07/13 v1.0 Academic Report Class]

% --- Option parsing ---
\newif\if@assignment
\@assignmentfalse
\DeclareOption{assignment}{\@assignmenttrue}
\DeclareOption{papermode}{\@assignmentfalse}
\DeclareOption*{\PassOptionsToClass{\CurrentOption}{article}}
\ExecuteOptions{papermode}
\ProcessOptions\relax

% --- Base class ---
\LoadClass[12pt,a4paper]{article}

% --- Packages ---
\RequirePackage[utf8]{inputenc}
\RequirePackage[T1]{fontenc}
\RequirePackage{lmodern}              % Latin Modern fonts
\RequirePackage{amsmath,amsthm,amssymb}
\RequirePackage[margin=1in]{geometry}
\setlength{\headheight}{33.5pt}
\addtolength{\topmargin}{-21.5pt}
\RequirePackage{fancyhdr}
\RequirePackage[svgnames,dvipsnames]{xcolor}
\RequirePackage{titlesec}
\RequirePackage{listings}
\RequirePackage[
  backend=biber,
  style=authoryear,
  sorting=nyt,
  maxbibnames=99,
  giveninits=true,
  uniquename=init
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
\endinput
