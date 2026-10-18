"""Generate the anonymous LNCS (MICCAI) version from the two-column draft, so both stay in sync.
usage: python to_lncs.py && latexmk -pdf main_lncs.tex      (main limit: 8 pages + up to 2 pages of references)"""
import re
s = open('main.tex').read()
body = s[s.index('\\begin{abstract}'):]
pre = r"""% AUTO-GENERATED from main.tex by to_lncs.py -- edit main.tex, not this file.
\documentclass[runningheads]{llncs}
\usepackage[T1]{fontenc}
\usepackage{graphicx,booktabs,amsmath,amssymb,xcolor,multirow}
\usepackage[hidelinks]{hyperref}
\usepackage{tikz}
\usetikzlibrary{arrows.meta,positioning,calc}
\newcommand{\todo}[1]{\textcolor{red}{[TODO: #1]}}
\newcommand{\pmsd}[1]{{\scriptsize$\pm$#1}}
\renewcommand{\topfraction}{.9}\renewcommand{\textfraction}{.1}\renewcommand{\floatpagefraction}{.8}
\begin{document}
\title{One Advocate per Diagnosis: Robust Cheap Evidence Selection for Whole-Slide Image Classification}
\titlerunning{One Advocate per Diagnosis}
\author{Anonymized Authors}
\authorrunning{Anonymized Authors}
\institute{Anonymized Affiliations}
\maketitle
"""
body = body.replace('\\end{abstract}', '\\keywords{Whole-slide images \\and Efficient inference \\and Robustness \\and Multiple-instance learning}\n\\end{abstract}', 1)
body = body.replace('\\resizebox{\\columnwidth}', '\\resizebox{.85\\textwidth}')
body = body.replace('[width=\\columnwidth]{figs/fig_robust.pdf}', '[width=.6\\textwidth]{figs/fig_robust.pdf}')
body = body.replace('figs/fig_frontier.pdf', 'figs/fig_frontier_lncs.pdf')
body = body.replace('\\begin{figure*}[t]', '\\begin{figure}[!tb]').replace('figure*', 'figure').replace('table*', 'table')
# wide main table: shrink to text width
body = re.sub(r'(\\begin\{tabular\}\{@\{\}lcccccccc@\{\}\}.*?\\end\{tabular\})', r'\\resizebox{\\textwidth}{!}{\1}', body, count=1, flags=re.S)
body = body.replace('\\bibliographystyle{unsrt}', '\\bibliographystyle{splncs04}')
open('main_lncs.tex', 'w').write(pre + body)
print('wrote main_lncs.tex')
