#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""R36: P1 narrative restructuring + table renumbering from Table 1.

Usage:
  python scripts/_r36_p1_structure.py --check
  python scripts/_r36_p1_structure.py --apply

  T0  drop \setcounter{table}{2} so tables number from 1 (all \ref follow automatically;
      grep confirmed there are no hard-coded table numbers in either manuscript)
  P1-1 new subsection "Operating-point coordinates" (four axes) at the end of Section 2
  P1-2 one-line axis declaration at the head of the three sweep subsections
  P1-3 decision table at the end of the Conclusion
  P1-4 single definition of "resolved" in Section 4.5; delete the repeated
       "we make no equivalence claim" sentences
"""
import io
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LATEX = os.path.join(ROOT, "docs", "manuscript", "latex")
EN = os.path.join(LATEX, "main.tex")
ZH = os.path.join(LATEX, "main_zh.tex")

EN_AXES = r"""\subsection{Operating-point coordinates}
\label{sec:axes}

The comparisons above leave four quantities free, and each fixes a different term of the problem. The
first is the flux and energy, which together set the counting-noise level and therefore the variance
left for a learned prior to remove. The second is the angular span, the spacing of the measured views,
which sets the curvature that the completion must extrapolate at each rod boundary. The third is the
detector row, because the source profile makes the central row the cleanest and the edge rows the
noisiest, so the same comparison can change sign between them. The fourth is the object symmetry, which
fixes how much of a target view is already contained in a single measured view: under four-fold
symmetry a view 90\dgr{} away is an exact replica, so a projection-domain error measures proximity to
that floor rather than interpolation quality. The experiments in Section~\ref{sec:results} sweep one
axis at a time, holding the others fixed, and Section~\ref{sec:limits} states what the resulting map
does not cover."""

EN_DECISION = r"""\begin{table}[t]
  \caption{Operating-point map: what to use in each region of the coordinates defined in
Section~\ref{sec:axes}.}
  \label{tab:decision}
  \centering
  \small
  \begin{tabular}{@{}>{\raggedright\arraybackslash}p{0.30\textwidth}>{\raggedright\arraybackslash}p{0.34\textwidth}>{\raggedright\arraybackslash}p{0.24\textwidth}@{}}
    \toprule
    Condition & Recommendation & Evidence \\
    \midrule
    High flux, narrow span, imaging row & Prefer interpolation & Sections~\ref{sec:row16} and~\ref{sec:span} \\
    Low flux ($f \le 0.5$) or wide span ($\ge 6$\dgr{}) & CNN completion & Sections~\ref{sec:span} and~\ref{sec:dose} \\
    Detector-wide pool & CNN completion (variance diagnostic) & Section~\ref{sec:row16}, Table~\ref{tab:supp32} \\
    Defect detectability & Neither creates image-domain detectability & Section~\ref{sec:defect} \\
    \bottomrule
  \end{tabular}\par
  \small The rows are stated against the axes of Section~\ref{sec:axes}; each recommendation is a
property of this object, detector and energy range and is not claimed as a universal constant
(Section~\ref{sec:limits}).
\end{table}"""

ZH_AXES = r"""\subsection{工作点坐标}
\label{sec:axes}

上述比较留出了四个自由量，每一个都固定了问题的不同项。第一是通量与能量，二者共同确定计数噪声水平，从而确定留给学习先验可去除的方差。第二是角采样跨度，即实测视角的间距，它确定补全必须在每个棒边界处外推的曲率。第三是探测器行，因为源剖面使中心行最干净、边缘行最噪，同一比较在两者之间可能变号。第四是物体对称性，它固定了一个目标视角已有多少内容包含在单个实测视角中：在四重对称下，相隔 90\dgr{} 的视角是精确副本，故投影域误差衡量的是与那条地板的接近程度，而非插值质量。§\ref{sec:results} 的实验每次只扫描一个轴、固定其余各轴，§\ref{sec:limits} 则说明所得到的地图不覆盖什么。"""

ZH_DECISION = r"""\begin{table}[t]
  \caption{工作点地图：在 §\ref{sec:axes} 所定义坐标的每个区域应当采用何种规则。}
  \label{tab:decision}
  \centering
  \small
  \begin{tabular}{@{}>{\raggedright\arraybackslash}p{0.28\textwidth}>{\raggedright\arraybackslash}p{0.34\textwidth}>{\raggedright\arraybackslash}p{0.26\textwidth}@{}}
    \toprule
    条件 & 建议 & 依据 \\
    \midrule
    高通量、窄跨度、成像行 & 优先插值 & §\ref{sec:row16} 与 §\ref{sec:span} \\
    低通量（$f \le 0.5$）或宽跨度（$\ge 6$\dgr{}） & CNN 补全 & §\ref{sec:span} 与 §\ref{sec:dose} \\
    探测器全域池化 & CNN 补全（方差诊断） & §\ref{sec:row16}、表~\ref{tab:supp32} \\
    缺陷可检测性 & 两者都不创造图像域可检测性 & §\ref{sec:defect} \\
    \bottomrule
  \end{tabular}\par
  \small 各行均按 §\ref{sec:axes} 的坐标表述；每条建议都是本物体、本探测器与本能区的性质，不作为普适常数主张（§\ref{sec:limits}）。
\end{table}"""

EN_RULES = [
    # T0 table numbering
    ("T0 drop setcounter", "\\setcounter{table}{2}\n", ""),

    # P1-1 new subsection 2.5
    ("P1-1 axes subsection",
     "which leads to the masked objective of Section~\\ref{sec:method}.\n\n% =====================================================================\n\\section{Method}",
     "which leads to the masked objective of Section~\\ref{sec:method}.\n\n" + EN_AXES
     + "\n\n% =====================================================================\n\\section{Method}"),

    # P1-4 definition in 4.5
    ("P1-4 metrics definition",
     "Table~\\ref{tab:metrics} summarises the role each metric plays in the argument that follows.",
     "Table~\\ref{tab:metrics} summarises the role each metric plays in the argument that follows. "
     "A difference is described as resolved only when the view-level paired test excludes zero and the "
     "mean difference exceeds the seed-to-seed standard deviation; otherwise it is reported as not "
     "resolved, which is not an equivalence claim."),

    # P1-2 axis declarations
    ("P1-2 5.6 axis",
     "\\label{sec:all32diag}\n% =====================================================================\n\nThe comparison that decides the reconstruction",
     "\\label{sec:all32diag}\n% =====================================================================\n\n"
     "This subsection sweeps the detector-row axis; flux and span are fixed at full flux and 2\\dgr{}. "
     "The comparison that decides the reconstruction"),

    ("P1-2 5.9 axis",
     "\\label{sec:span}\n% =====================================================================\n\nTo simulate spans larger than 2\\dgr{}",
     "\\label{sec:span}\n% =====================================================================\n\n"
     "This subsection sweeps the angular-span axis; flux and row are fixed at full flux and Row~16. "
     "To simulate spans larger than 2\\dgr{}"),

    ("P1-2 5.10 axis",
     "paired confidence interval of Section~\\ref{sec:row16}.\n\nTo decouple attenuation from counting statistics",
     "paired confidence interval of Section~\\ref{sec:row16}.\n\n"
     "This subsection sweeps the flux axis; span and row are fixed at 2\\dgr{} and Row~16. "
     "To decouple attenuation from counting statistics"),

    # P1-4 delete duplicated equivalence statements
    ("P1-4 abstract dedup",
     "low-count edge rows that never enter the reconstruction. We make no equivalence claim: `not resolved' means only that the difference is smaller than the seed-to-seed variability.",
     "low-count edge rows that never enter the reconstruction."),

    ("P1-4 intro dedup",
     "and treats a mean difference smaller than the seed SD as unresolved; no equivalence claim is made.",
     "and applies the resolution criterion of Section~\\ref{sec:metrics}."),

    ("P1-4 limits dedup",
     "a difference is described as resolved only when both conditions hold, and a difference that fails either is reported as not resolved rather than as equal.",
     "applying the criterion of Section~\\ref{sec:metrics}."),

    ("P1-4 conclusion dedup",
     " We make no equivalence claim: `not resolved' means only that the difference is smaller than the seed-to-seed variability, not that the two rules are equal.",
     ""),

    # P1-3 decision table
    ("P1-3 decision table",
     "is the natural next step (Section~\\ref{sec:future}).\n",
     "is the natural next step (Section~\\ref{sec:future}).\n\n" + EN_DECISION + "\n"),
]

ZH_RULES = [
    ("T0 drop setcounter", "\\setcounter{table}{2}\n", ""),

    ("P1-1 axes subsection",
     "% =====================================================================\n\\section{方法}",
     ZH_AXES + "\n\n% =====================================================================\n\\section{方法}"),

    ("P1-4 metrics definition",
     "表~\\ref{tab:metrics} 汇总了各指标在后续论证中所扮演的角色。",
     "表~\\ref{tab:metrics} 汇总了各指标在后续论证中所扮演的角色。差异只有在逐视角配对检验不含零、"
     "且均值差超过种子间标准差时才称为“可分辨”；否则报告为“未能分辨”，后者不构成等价性主张。"),

    ("P1-2 5.6 axis",
     "\\label{sec:all32diag}\n\n决定重建的那个比较",
     "\\label{sec:all32diag}\n\n本小节扫描探测器行轴；通量与跨度固定为满通量与 2\\dgr{}。\n决定重建的那个比较"),

    ("P1-2 5.9 axis",
     "\\label{sec:span}\n% =====================================================================\n\n为模拟大于 2\\dgr{} 的跨度",
     "\\label{sec:span}\n% =====================================================================\n\n"
     "本小节扫描角采样跨度轴；通量与探测器行固定为满通量与第 16 行。\n为模拟大于 2\\dgr{} 的跨度"),

    ("P1-2 5.10 axis",
     "% §\\ref{sec:row16} 的配对置信区间。\n",
     "% §\\ref{sec:row16} 的配对置信区间。\n\n本小节扫描通量轴；跨度与探测器行固定为 2\\dgr{} 与第 16 行。\n"),

    ("P1-4 abstract dedup",
     "低计数边缘行主导。本文不主张任何等价性：“未能分辨”只意味着差异小于种子间变异。",
     "低计数边缘行主导。"),

    ("P1-4 intro dedup",
     "并把小于种子标准差的均值差视为未决；不作任何等价性声明。",
     "并采用 §\\ref{sec:metrics} 的可分辨判据。"),

    ("P1-4 limits dedup",
     "\\textbf{只有两个条件同时成立，才称该差异为可分辨}；任一条件不成立，均报告为\\textbf{未能分辨}，而非相等。",
     "两者按 §\\ref{sec:metrics} 的判据合并判定。"),

    ("P1-4 conclusion dedup",
     "本文不主张任何等价性：“未能分辨”只意味着差异小于种子间变异，而非两条规则相等。",
     ""),

    ("P1-3 decision table",
     "是自然的下一步（§\\ref{sec:future}）。\n",
     "是自然的下一步（§\\ref{sec:future}）。\n\n" + ZH_DECISION + "\n"),
]


def detect_eol(text):
    return "\r\n" if "\r\n" in text else "\n"


def run(path, rules, apply_changes):
    with io.open(path, "r", encoding="utf-8", newline="") as fh:
        text = fh.read()
    eol = detect_eol(text)
    ok = True
    new_text = text
    for name, old, new in rules:
        o = old.replace("\n", eol)
        n = new.replace("\n", eol)
        c = new_text.count(o)
        if c != 1:
            print("  [FAIL] %-32s count=%d" % (name, c))
            ok = False
            continue
        new_text = new_text.replace(o, n, 1)
        print("  [ok]   %-32s" % name)
    if apply_changes and ok:
        with io.open(path, "w", encoding="utf-8", newline="") as fh:
            fh.write(new_text)
        print("  -> written: %s" % os.path.basename(path))
    return ok


def main():
    apply_changes = "--apply" in sys.argv
    print("=== R36 %s : EN ===" % ("APPLY" if apply_changes else "CHECK"))
    a = run(EN, EN_RULES, apply_changes)
    print("=== R36 %s : ZH ===" % ("APPLY" if apply_changes else "CHECK"))
    b = run(ZH, ZH_RULES, apply_changes)
    if not (a and b):
        print("RESULT: FAILED (nothing written)")
        sys.exit(1)
    print("RESULT: OK")


if __name__ == "__main__":
    main()
