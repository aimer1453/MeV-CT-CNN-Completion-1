# -*- coding: utf-8 -*-
"""R52 EN 落地：§5.9 三种子换代 / Fig 8 注 / 新 §6.5 + 联合图 + tab:flawedge / L731 / 结论 / SI fig:defectsi"""
import io, sys, re

P = 'docs/manuscript/latex/main.tex'
s = io.open(P, encoding='utf-8', newline='').read().replace('\r\n', '\n')
orig = s
edits = []

def rep(old, new, tag):
    global s
    n = s.count(old)
    assert n == 1, '%s: count=%d' % (tag, n)
    s = s.replace(old, new)
    edits.append(tag)

# 1 graphicspath
rep('{../../figs/p01_pipeline/}}',
    '{../../figs/p01_pipeline/}%\n              {../../figs/p01_regime_joint/}%\n              {../../figs/p01_flaw_span_sweep/}}',
    'graphicspath')

# 2 tab:spanenergy Row16 -> 3 seeds
rep("""    Row~16 & 2 & 0.9530 & 1.0315 & 1.0377 & 1.0403 \\\\
    Row~16 & 4 & 0.9678 & 1.0467 & 1.0755 & 1.0870 \\\\
    Row~16 & 6 & 0.9384 & 0.9995 & 0.9851 & 0.9895 \\\\
    Row~16 & 8 & 0.9202 & 0.9849 & 0.9833 & 0.9788 \\\\""",
    """    Row~16 & 2 & 0.9503 & 1.0658 & 1.0831 & 1.0920 \\\\
    Row~16 & 4 & 0.9576 & 1.0407 & 1.0648 & 1.0676 \\\\
    Row~16 & 6 & 0.9316 & 1.0020 & 0.9912 & 0.9953 \\\\
    Row~16 & 8 & 0.9121 & 0.9743 & 0.9581 & 0.9528 \\\\""",
    'tab-spanenergy-rows')

# 3 table caption 3--4% -> 7--9%
rep('the completion trails linear interpolation by 3--4\\% at 3.0, 4.5 and 6.0~MeV at the densest sampling',
    'the completion trails linear interpolation by 7--9\\% at 3.0, 4.5 and 6.0~MeV at the densest sampling',
    'tab-spanenergy-caption')

# 4 footer note: pooled + mixed-seed disclosure
rep("""  \\small Ratios below 1 favour the completion. The pooled overall ratios (Row~16: 0.9769, 0.9994, 0.9565, 0.9426; image: 0.9388, 0.9429, 0.9364, 0.9301) are reported in Fig.~\\ref{fig:span}.""",
    """  \\small Ratios below 1 favour the completion. Row-16 entries are three-seed means (seed SD 0.003--0.053); the image-domain entries are single-run point estimates. The pooled overall Row-16 ratios are 0.9879, 0.9890, 0.9536 and 0.9303, and are reported in Fig.~\\ref{fig:span}.""",
    'tab-spanenergy-note')

# 5 L551 single-run sentence
rep('Each span is a single training run, so the sweep does not separate the two rules statistically at any span;',
    'Each span is trained with three seeds, and the seed spread is quoted with the Row-16 entries of Table~\\ref{tab:spanenergy};',
    'L551')

# 6 L556 body ratios + 17.7%
rep('the sweep is single-seed throughout, so every span ratio quoted below is a point estimate and none of the span orderings is established.',
    'the Row-16 ratios are three-seed means, and the seed spread is small enough to order them at every span except 4\\dgr{}.',
    'L556a')
rep('at the 100-epoch budget is 0.977, 0.999, 0.957 and 0.943 at $s = 2, 4, 6$ and 8\\dgr{}, so the completion is ahead at 2\\dgr{}, 6\\dgr{} and 8\\dgr{} and not resolved at 4\\dgr{} (ratio 0.999);',
    'at the 100-epoch budget is 0.988, 0.989, 0.954 and 0.930 at $s = 2, 4, 6$ and 8\\dgr{}, so the pooled curve never crosses unity and 4\\dgr{} is a tie (ratio 0.989), not a separation;',
    'L556b')
rep('against 20.6\\% for the completion (0.1225 to 0.1477)',
    'against 17.7\\% for the completion (0.1239 to 0.1458)',
    'L556c')
rep('as single-seed point values).', 'as point estimates).', 'L556d')

# 7 Fig 8 caption
old_cap = s[s.index('  \\caption{Sensitivity to the angular sampling step (100-epoch budget).'):]
old_cap = old_cap[:old_cap.index('}\n')+1]
new_cap = ("""  \\caption{The pooled ratio never crosses unity, and it hides an energy-dependent sign. Row-16 projection RMSE versus sampling step $s$ on the periodic phantom, three seeds per energy, evaluated on 45 held-out views. (a) Absolute error of the three estimators for the four-energy pool; the panel beneath repeats the same comparison as the completion-to-linear ratio with parity at unity --- $0.988$ at $s = 2$\\dgr{}, $0.989$ at 4\\dgr{} (a tie, not a separation), $0.954$ at 6\\dgr{}, $0.930$ at 8\\dgr{}, $0.900$ at 10\\dgr{} and $0.834$ at 12\\dgr{}, so the pooled curve never crosses unity. (b) Image-domain RMSE against the 360-view dense reference, logarithmic axis. Cubic interpolation uses four measured views against two for linear and three for the completion, yet remains above linear at every span in (a), so raising the interpolation order does not rescue the baseline. The pool is weighted by error magnitude and dominated by the 1.5~MeV channel, where the completion wins; the per-energy decomposition, in which the 3.0~MeV slice is above unity at the three densest spans and crosses at 6.14\\dgr{}, is given in Table~\\ref{tab:spanenergy} and Fig.~\\ref{fig:regime}.}""")
rep(old_cap, new_cap, 'fig-span-caption')

# 8 L731 crossover
rep('the crossover at $s = 6$\\dgr{} (Section~\\ref{sec:span})\nmarks where a learned prior begins to outweigh the curvature bias of interpolation.',
    'the crossover, which lies between $s = 6$ and $8$\\dgr{}\n(Section~\\ref{sec:span}), marks where a learned prior begins to outweigh the curvature bias of interpolation.',
    'L731')

# 9 conclusion L879
rep('the ratio between them is 0.977, 0.999, 0.957 and 0.943 at 2\\dgr{}, 4\\dgr{}, 6\\dgr{} and 8\\dgr{} (ahead except at 4\\dgr{}, where the two are not resolved)',
    'the ratio between them is 0.988, 0.989, 0.954 and 0.930 at 2\\dgr{}, 4\\dgr{}, 6\\dgr{} and 8\\dgr{} (the pooled curve never crosses unity, and 4\\dgr{} is a tie rather than a separation)',
    'L879a')
rep('interpolation then degrades by 24.9\\% on the imaging row against 20.6\\% for the completion',
    'interpolation then degrades by 24.9\\% on the imaging row against 17.7\\% for the completion',
    'L879b')

# 10 insert new 6.5 before Applicability
anchor = '\\subsection{Applicability beyond nuclear fuel}'
assert s.count(anchor) == 1
new65 = r"""\subsection{The operating point replicates in the defect domain}
\label{sec:replicate}
% =====================================================================

Thinning the measured views of the defective phantom at 3.0~MeV from 180 ($s = 2$\dgr{}) to 30
($s = 12$\dgr{}), with three seeds and the same 45 held-out views at every span, the
completion-to-linear ratio on Row~16 falls monotonically from $1.156 \pm 0.008$ through
$1.032 \pm 0.008$ at 6\dgr{} to $0.783 \pm 0.006$ at 12\dgr{}, crossing unity between 6\dgr{} and
8\dgr{} (Fig.~\ref{fig:defectsi}). The completion is 15.6\% worse than interpolation at the densest
sampling and 21.7\% better at the widest. The crossing is drawn at 6.9\dgr{} as a straight line
through the two neighbouring means, not as a measured crossing, because no 7\dgr{} run exists.
These ratios are computed on the 45 held-out views only and are not comparable with the 360-view
entries of Table~\ref{tab:flawedge}.

The crossover is not a consequence of interpolation failing. Between $s = 2$ and $s = 12$\dgr{} the
linear error grows by 118\% and the completion's by 48\%, whereas a pure $O(h^2)$ curvature term
would grow 36-fold when the half-span goes from 1\dgr{} to 6\dgr{}; most of the interpolation error
on this row is therefore independent of the span --- photon statistics, the non-linearity of $-\log$
at low counts, and the fill value assigned to zero-count pixels. The completion is ahead of
\emph{linear} interpolation only: cubic interpolation is still 1.1\% better at $s = 8$\dgr{}, and the
completion overtakes cubic only near 10\dgr{}.

The effect of the object is a change of level rather than of crossing (Fig.~\ref{fig:regime}). At
the same energy and the same held-out views the periodic array crosses at 6.14\dgr{} and the
defective phantom at 6.92\dgr{}, a shift of less than one degree that is estimated by interpolation
between neighbouring means on a 2\dgr{} grid and should not be read to a second decimal. What the
defect changes is the growth rate of the fixed rule, which it multiplies by $118/63 = 1.89$ over
$2$--$12$\dgr{} and by $62/33 = 1.86$ over $2$--$8$\dgr{}; the completion's own growth is multiplied
by less, and less stably ($1.78$ against $1.59$). The crossover arises because the rule being beaten
degrades faster, not because the completion degrades more slowly.

\begin{figure}[htbp]
  \centering
  \includegraphics[width=0.98\textwidth]{fig_regime_joint_en.png}
  \caption{The operating point in two dimensions. (a) Completion-to-linear Row-16 ratio against
sampling span for the periodic array and the defective phantom, both at 3.0~MeV and both on 45
held-out views; the crossings are at 6.14\dgr{} and 6.92\dgr{}, a separation of less than one degree
(from 6.14\dgr{} to 6.92\dgr{}). Because the span grid is sampled at 2\dgr{} steps, this shift is
estimated by linear interpolation between the neighbouring means and should not be read to the
second decimal. (b) The same errors normalised to each estimator's value at 2\dgr{}, so that growth
with span is compared rather than the offset between objects. (c) The regime map of
$1-\text{ratio}$ in per cent on the same grid: the defect row loses at narrow spans and wins at wide
ones, while the periodic row changes sign only near 6\dgr{}.}
  \label{fig:regime}
\end{figure}

\begin{table}[t]
  \caption{Edge-rod cavity (rod \texttt{edge2}, $r = 260$~px, azimuth $180$\dgr{}) across four
energies, three seeds $\times$ 100 epochs. Row~16 is evaluated on all 360 completed views; the image
domain is evaluated on the 45 held-out test views.}
  \label{tab:flawedge}
  \centering
  \footnotesize
  \begin{tabular}{lccccc}
    \toprule
    Energy & Row~16 CNN & Row~16 linear & CNN/linear & Better views & Image RMSE$_\mathrm{valid}$ \\
    \midrule
    1.5~MeV & 0.1246 & 0.1258 & 0.991 & 33\% & 0.001360 \\
    3.0~MeV & 0.0692 & 0.0602 & 1.150 & 0\% & 0.000723 \\
    4.5~MeV & 0.0591 & 0.0537 & 1.102 & 1\% & 0.000607 \\
    6.0~MeV & 0.0586 & 0.0540 & 1.085 & 3\% & 0.000603 \\
    \bottomrule
  \end{tabular}\par
  \small Row~16 errors are per-pixel masked RMSE in the $-\log$ domain on the 360 completed views;
the image-domain column uses the 45 held-out views and is not interchangeable with it. The deficit is
8.5--15.0\% at 3.0/4.5/6.0~MeV and reverses to a 0.9\% lead at 1.5~MeV, the lowest-flux channel.
\end{table}

"""
s = s.replace(anchor, new65 + anchor)
edits.append('ins-6.5')

# 11 SI fig:defectsi after fig:spanoracle
old_si = """  \\label{fig:spanoracle}
\\end{figure}

\\end{document}"""
assert s.count(old_si) == 1
new_si = r"""  \label{fig:spanoracle}
\end{figure}

\begin{figure}[htbp]
  \centering
  \includegraphics[width=0.95\textwidth]{fig_span_ratio_en.png}
  \caption{A localized defect raises the ratio at every span and moves the crossing by less than one
degree. Completion-to-linear Row-16 RMSE ratio on the defective phantom at 3.0~MeV (edge-rod cavity,
three seeds, 45 held-out views). (a) The ratio falls monotonically from $1.156 \pm 0.008$ at 2\dgr{}
to $0.783 \pm 0.006$ at 12\dgr{}; the crossing between 6\dgr{} and 8\dgr{} is marked on the figure,
and its position is an interpolation between the two neighbouring means rather than a measured
crossing. (b) Each estimator's error normalised to its own value at 2\dgr{}, for the defective
phantom (solid) and the periodic array (dashed): over $2$--$8$\dgr{} the linear error grows by 62\%
against 33\%, and the completion's by 35\% against 22\%. These are the $2$--$8$\dgr{} growth rates;
the $2$--$12$\dgr{} rates, which are a different quantity, appear in Fig.~\ref{fig:regime}(b).}
  \label{fig:defectsi}
\end{figure}

\end{document}"""
s = s.replace(old_si, new_si)
edits.append('ins-SI')

if '--apply' in sys.argv:
    io.open(P, 'w', encoding='utf-8', newline='').write(s.replace('\n', '\r\n'))
    print('APPLIED %d edits, %d -> %d chars' % (len(edits), len(orig), len(s)))
else:
    print('CHECK ok: %d edits' % len(edits))
    for e in edits: print('  -', e)
