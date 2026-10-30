# -*- coding: utf-8 -*-
"""R53 ZH 落地：镜像 EN 的 §6.5 / 联合图 / tab:flawedge / §5.9 三种子 / L610 / 结论 / SI"""
import io, sys

P = 'docs/manuscript/latex/main_zh.tex'
s = io.open(P, encoding='utf-8', newline='').read().replace('\r\n', '\n')
orig = s
ed = []

def rep(old, new, tag):
    global s
    n = s.count(old)
    assert n == 1, '%s: count=%d' % (tag, n)
    s = s.replace(old, new); ed.append(tag)

# 1 graphicspath
rep('              {../../figs/p01_pipeline/}}',
    '              {../../figs/p01_pipeline/}%\n              {../../figs/p01_regime_joint/}%\n              {../../figs/p01_flaw_span_sweep/}}',
    'graphicspath')

# 2 tab:spanenergy 四档
rep("""    第 16 行 & 2 & 0.9530 & 1.0315 & 1.0377 & 1.0403 \\\\
    第 16 行 & 4 & 0.9678 & 1.0467 & 1.0755 & 1.0870 \\\\
    第 16 行 & 6 & 0.9384 & 0.9995 & 0.9851 & 0.9895 \\\\
    第 16 行 & 8 & 0.9202 & 0.9849 & 0.9833 & 0.9788 \\\\""",
    """    第 16 行 & 2 & 0.9503 & 1.0658 & 1.0831 & 1.0920 \\\\
    第 16 行 & 4 & 0.9576 & 1.0407 & 1.0648 & 1.0676 \\\\
    第 16 行 & 6 & 0.9316 & 1.0020 & 0.9912 & 0.9953 \\\\
    第 16 行 & 8 & 0.9121 & 0.9743 & 0.9581 & 0.9528 \\\\""",
    'tab-rows')

# 3 表题 3--4% -> 7--9%
rep('3--4\\%', '7--9\\%', 'tab-caption')

# 4 表注
rep('  \\small 比值低于 1 表示补全占优。汇总的 overall 比值（第 16 行：0.9769、0.9994、0.9565、0.9426；图像：0.9388、0.9429、0.9364、0.9301）见图~\\ref{fig:span}。',
    '  \\small 比值低于 1 表示补全占优。第 16 行条目为三种子均值（种子 SD 0.003--0.053）；图像域条目为单次运行的点估计。第 16 行的整体汇总比值为 0.9879、0.9890、0.9536 与 0.9303，见图~\\ref{fig:span}。',
    'tab-note')

# 5 §5.9 四值
rep('分别为 0.977、0.999、0.957、0.943', '分别为 0.988、0.989、0.954、0.930', 'L462-ratios')
rep('4\\dgr{} 处未能分辨（比值 0.999）', '4\\dgr{} 处未能分辨（比值 0.989，为打平而非分离）', 'L462-tie')
rep('补全是 20.6\\%（0.1225 到 0.1477）', '补全是 17.7\\%（0.1239 到 0.1458）', 'L462-206')
rep('分别高出 3.0、5.1、6.1 个百分点；单种子点值）', '分别高出 3.0、5.1、6.1 个百分点；点估计）', 'L462-single')

# 6 Fig 8 图注
rep('CNN/线性之比为 0.977、0.999、0.957、0.943',
    'CNN/线性之比为 0.988、0.989、0.954、0.930', 'cap-ratios')

# 7 L610 crossover
rep('$s = 6$\\dgr{} 处的交叉（§\\ref{sec:span}）标志着学习先验开始压过插值的曲率偏差。',
    '交叉点落在 $s = 6$\\dgr{} 与 $8$\\dgr{} 之间（§\\ref{sec:span}），它标志着学习先验开始压过插值的曲率偏差。',
    'L610')

# 8 结论
rep('两者之比在 2\\dgr{}、4\\dgr{}、6\\dgr{} 与 8\\dgr{} 处分别为 0.977、0.999、0.957 与 0.943',
    '两者之比在 2\\dgr{}、4\\dgr{}、6\\dgr{} 与 8\\dgr{} 处分别为 0.988、0.989、0.954 与 0.930（汇总曲线始终未穿越 1，4\\dgr{} 处是打平而非分离）',
    'L718-a')
rep('补全是 20.6\\%', '补全是 17.7\\%', 'L718-b')

# 9 插入 §6.5
anchor = '\\subsection{核燃料之外的可迁移性}'
assert s.count(anchor) == 1
new65 = r"""\subsection{工作点在缺陷数据域上的复现}
\label{sec:replicate}
% =====================================================================

把边棒缺陷物模在 3.0~MeV 下的实测视角从 180（$s = 2$\dgr{}）稀释到 30（$s = 12$\dgr{}），
每个跨度三种子、且始终使用同一批 45 个 held-out 视角，第 16 行上的补全对线性比值由
$1.156 \pm 0.008$ 单调降至 6\dgr{} 的 $1.032 \pm 0.008$，在 6\dgr{} 与 8\dgr{} 之间穿越 1，
并继续降至 12\dgr{} 的 $0.783 \pm 0.006$（图~\ref{fig:defectsi}）。补全在最密采样处比插值差
15.6\%，在最宽处好 21.7\%。交叉点标在 6.9\dgr{}：它是两点均值的直线内插，而非实测交叉，
因为没有 7\dgr{} 的运行。这些比值只在 45 个 held-out 视角上统计，与表~\ref{tab:flawedge}
的 360 待补全视角口径不可互比。

交叉不是插值失效的结果。从 $s = 2$ 到 $s = 12$\dgr{}，线性误差增长 118\%、补全增长 48\%，
而纯 $O(h^2)$ 曲率项在半跨度由 1\dgr{} 增至 6\dgr{} 时应增长 36 倍；可见该行上大部分插值误差
与跨度无关——它来自光子统计、低计数处 $-\log$ 的非线性，以及零计数像素的填充值。补全只优于
\emph{线性}插值：在 $s = 8$\dgr{} 处三次插值仍好 1.1\%，补全要到 10\dgr{} 附近才反超三次插值。

物体的作用是改变水平而非改变交叉（图~\ref{fig:regime}）。在相同能量、相同 held-out 视角下，
周期阵列在 6.14\dgr{} 交叉、缺陷物模在 6.92\dgr{} 交叉，位移不足 1 度——它由 2\dgr{} 网格上
相邻均值的线性内插得到，不应读到小数点后第二位。缺陷真正改变的是固定规则的增长率：它在
$2$--$12$\dgr{} 上把该增长率放大 $118.3/62.6 = 1.89$ 倍，在 $2$--$8$\dgr{} 上放大
$61.7/33.2 = 1.86$ 倍；而补全自身的增长被放大的倍数更小、也更不稳定（$1.78$ 对 $1.59$）。
交叉之所以出现，是因为被击败的那条规则退化得更快，而不是因为补全退化得更慢。

\begin{figure}[htbp]
  \centering
  \includegraphics[width=0.98\textwidth]{fig_regime_joint_en.png}
  \caption{二维工作点。(a) 周期阵列与缺陷物模的补全对线性第 16 行比值随采样跨度的变化，两者
均为 3.0~MeV、均基于 45 个 held-out 视角；交叉点分别在 6.14\dgr{} 与 6.92\dgr{}，位移不足
1 度。由于跨度网格以 2\dgr{} 为步长采样，该位移由相邻均值的线性内插估计，不应读到小数点后
第二位。(b) 同一批误差各自以其 2\dgr{} 处的值归一化，因而比较的是随跨度的增长而非两物体之间
的偏移。(c) 同一网格上 $1-\text{ratio}$ 的百分制度格图：缺陷行在窄跨度处落后、在宽跨度处领先，
而周期行只在 6\dgr{} 附近变号。}
  \label{fig:regime}
\end{figure}

\begin{table}[t]
  \caption{边棒空腔（棒 \texttt{edge2}，$r = 260$~px，方位角 $180$\dgr{}）在四个能量上的结果，
三种子 $\times$ 100 轮。第 16 行在全部 360 个待补全视角上评估；图像域在 45 个 held-out 测试
视角上评估。}
  \label{tab:flawedge}
  \centering
  \footnotesize
  \begin{tabular}{lccccc}
    \toprule
    能量 & 第 16 行 CNN & 第 16 行 线性 & CNN/线性 & 更优视角 & 图像域 RMSE$_\mathrm{valid}$ \\
    \midrule
    1.5~MeV & 0.1246 & 0.1258 & 0.991 & 33\% & 0.001360 \\
    3.0~MeV & 0.0692 & 0.0602 & 1.150 & 0\% & 0.000723 \\
    4.5~MeV & 0.0591 & 0.0537 & 1.102 & 1\% & 0.000607 \\
    6.0~MeV & 0.0586 & 0.0540 & 1.085 & 3\% & 0.000603 \\
    \bottomrule
  \end{tabular}\par
  \small 第 16 行误差为 $-\log$ 域逐像素掩膜 RMSE，统计于 360 个待补全视角；图像域一列基于
45 个 held-out 视角，两者不可互换。缺陷域在 3.0/4.5/6.0~MeV 上落后 8.5--15.0\%，而在通量最低的
1.5~MeV 通道翻为领先 0.9\%。
\end{table}

"""
s = s.replace(anchor, new65 + anchor); ed.append('ins-6.5')

# 10 SI fig:defectsi
old_si = '  \\label{fig:spanoracle}\n\\end{figure}\n\n\\end{document}\n'
assert s.count(old_si) == 1
new_si = r"""  \label{fig:spanoracle}
\end{figure}

\begin{figure}[htbp]
  \centering
  \includegraphics[width=0.95\textwidth]{fig_span_ratio_en.png}
  \caption{局部缺陷抬高每一个跨度上的比值，而把交叉点移动不足 1 度。3.0~MeV 下缺陷物模（边棒
空腔，三种子，45 个 held-out 视角）的第 16 行补全对线性 RMSE 比值。(a) 比值由 2\dgr{} 的
$1.156 \pm 0.008$ 单调降至 12\dgr{} 的 $0.783 \pm 0.006$；6\dgr{} 与 8\dgr{} 之间的交叉已标在
图上，其位置是相邻两均值的线性内插，而非实测交叉。(b) 每个估计量的误差各自以其 2\dgr{} 处的
值归一化，缺陷物模为实线、周期阵列为虚线：在 $2$--$8$\dgr{} 上线性误差增长 62\% 对 33\%，
补全增长 35\% 对 22\%。上面这两个是 $2$--$8$\dgr{} 的增长率；$2$--$12$\dgr{} 的增长率是另一个
量，见图~\ref{fig:regime}(b)。}
  \label{fig:defectsi}
\end{figure}

\end{document}
"""
s = s.replace(old_si, new_si); ed.append('ins-SI')

if '--apply' in sys.argv:
    io.open(P, 'w', encoding='utf-8', newline='').write(s.replace('\n', '\r\n'))
    print('APPLIED %d edits' % len(ed))
else:
    print('CHECK ok: %d' % len(ed)); [print('  -', e) for e in ed]
