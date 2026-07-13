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
