# -*- coding: utf-8 -*-
"""R54c: P0 五项 + 2.7 定点替换（check / --apply）。
每条替换 old 必须在文件中恰好出现 1 次（count==1），否则整批失败、不做部分写入。
NOTE: 整表替换（tab:decision）为自包含块，已把行1改动并入块内，故不设独立行1替换。
"""
import io, sys

FILES = {
    'main.tex': 'docs/manuscript/latex/main.tex',
    'main_zh.tex': 'docs/manuscript/latex/main_zh.tex',
}

EN_TAB = (
    "  \\small\n"
    "  \\begin{tabular}{@{}>{\\raggedright\\arraybackslash}p{0.30\\textwidth}>{\\raggedright\\arraybackslash}p{0.34\\textwidth}>{\\raggedright\\arraybackslash}p{0.24\\textwidth}@{}}\n"
    "    \\toprule\n"
    "    Condition & Recommendation & Evidence \\\\\n"
    "    \\midrule\n"
    "    High flux, narrow span, imaging row & Prefer interpolation & Sections~\\ref{sec:row16} and~\\ref{sec:span} \\\\\n"
    "    Low flux ($f \\le 0.5$) or wide span ($\\ge 6$\\dgr{}) & CNN completion & Sections~\\ref{sec:span} and~\\ref{sec:dose} \\\\\n"
    "    Detector-wide pool & CNN completion (variance diagnostic) & Section~\\ref{sec:row16}, Table~\\ref{tab:supp32} \\\\\n"
    "    Defect detectability & Neither creates image-domain detectability & Section~\\ref{sec:defect} \\\\\n"
    "    \\bottomrule"
)
EN_TAB_NEW = (
    "  \\footnotesize\n"
    "  \\begin{tabular}{@{}>{\\raggedright\\arraybackslash}p{0.28\\textwidth}>{\\raggedright\\arraybackslash}p{0.32\\textwidth}>{\\raggedright\\arraybackslash}p{0.22\\textwidth}@{}}\n"
    "    \\toprule\n"
    "    Condition & Recommendation & Evidence level \\\\\n"
    "    \\midrule\n"
    "    Random split, 270 rows, 1\\dgr{} median neighbour (imaging row) & Either rule; CNN $+0.97\\%$ on average & Not resolved across 7 seeds ($p = 0.23$); view-level $2.05\\%$ deficit, CI excludes zero \\\\\n"
    "    Grouped far-side (90 rows), 5 seeds, 100-epoch & Either rule & $2.21 \\pm 0.50$, 5/5 seeds $> 1$; no formal test \\\\\n"
    "    270 rows, $45$\\dgr{} extrapolation (contig270) & Either rule & $-0.8\\%$; extrapolation alone is not the failure mode (3 seeds) \\\\\n"
    "    Wide span $s = 2$\\dgr{}, uniform & Interpolation & Three seeds; cubic still competitive at 2\\dgr{} \\\\\n"
    "    Wide span $s = 8$\\dgr{}, 3.0~MeV defect & CNN completion (over linear only) & Resolved on 3 seeds ($t = 19.0$, $p = 0.003$) \\\\\n"
    "    Wide span, 1.5~MeV uniform & CNN completion & Resolved on 3 seeds ($t = 16.0$, $p = 0.004$) \\\\\n"
    "    Wide span $s = 12$\\dgr{}, uniform & CNN completion (vs linear and cubic) & Cubic overtakes at $10$\\dgr{} \\\\\n"
    "    Wide span, 3.0--6.0~MeV uniform & Either rule; favours CNN & Not resolved (largest $t = 3.46$, $p = 0.074$) \\\\\n"
    "    Low flux ($f \\le 0.5$), 3.0~MeV & CNN completion & Resolved on 3 seeds ($t \\ge 122$) \\\\\n"
    "    Full flux ($f = 1.0$), 3.0~MeV & Either rule & Not resolved ($t = 3.04$, $p = 0.094$) \\\\\n"
    "    all-32-row pool & Variance diagnostic only & Not an imaging-accuracy claim \\\\\n"
    "    Defect detectability (per-rod CNR) & Neither creates image-domain detectability & Section~\\ref{sec:defect} \\\\\n"
    "    \\bottomrule"
)

ZH_TAB = (
    "  \\small\n"
    "  \\begin{tabular}{@{}>{\\raggedright\\arraybackslash}p{0.28\\textwidth}>{\\raggedright\\arraybackslash}p{0.34\\textwidth}>{\\raggedright\\arraybackslash}p{0.26\\textwidth}@{}}\n"
    "    \\toprule\n"
    "    条件 & 建议 & 依据 \\\\\n"
    "    \\midrule\n"
    "    高通量、窄跨度、成像行 & 优先插值 & §\\ref{sec:row16} 与 §\\ref{sec:span} \\\\\n"
    "    低通量（$f \\le 0.5$）或宽跨度（$\\ge 6$\\dgr{}） & CNN 补全 & §\\ref{sec:span} 与 §\\ref{sec:dose} \\\\\n"
    "    探测器全域池化 & CNN 补全（方差诊断） & §\\ref{sec:row16}、表~\\ref{tab:supp32} \\\\\n"
    "    缺陷可检测性 & 两者都不创造图像域可检测性 & §\\ref{sec:defect} \\\\\n"
    "    \\bottomrule"
)
ZH_TAB_NEW = (
    "  \\footnotesize\n"
    "  \\begin{tabular}{@{}>{\\raggedright\\arraybackslash}p{0.28\\textwidth}>{\\raggedright\\arraybackslash}p{0.32\\textwidth}>{\\raggedright\\arraybackslash}p{0.22\\textwidth}@{}}\n"
    "    \\toprule\n"
    "    条件 & 建议 & 证据等级 \\\\\n"
    "    \\midrule\n"
    "    随机划分，270 行，1\\dgr{} 中位邻域（成像行） & 两者皆可；CNN 平均领先 $+0.97\\%$ & 7 种子未分辨（$p = 0.23$）；视角层 $2.05\\%$ 劣势，CI 排除零 \\\\\n"
    "    分组远端（90 行），5 种子，100-epoch & 两者皆可 & $2.21 \\pm 0.50$，5/5 种子 $> 1$；无正式检验 \\\\\n"
    "    270 行，45\\dgr{} 外推（contig270） & 两者皆可 & $-0.8\\%$；外推本身并非失效模式（3 种子） \\\\\n"
    "    宽跨度 $s = 2$\\dgr{}，均匀阵列 & 插值 & 三种子；2\\dgr{} 处三次插值仍具竞争力 \\\\\n"
    "    宽跨度 $s = 8$\\dgr{}，3.0 MeV 缺陷 & CNN 补全（仅对线性） & 三种子分辨（$t = 19.0$，$p = 0.003$） \\\\\n"
    "    宽跨度，1.5 MeV 均匀 & CNN 补全 & 三种子分辨（$t = 16.0$，$p = 0.004$） \\\\\n"
    "    宽跨度 $s = 12$\\dgr{}，均匀阵列 & CNN 补全（对线性与三次） & 三次在 10\\dgr{} 反超 \\\\\n"
    "    宽跨度，3.0--6.0 MeV 均匀 & 两者皆可；方向偏向 CNN & 未分辨（最大 $t = 3.46$，$p = 0.074$） \\\\\n"
    "    低通量（$f \\le 0.5$），3.0 MeV & CNN 补全 & 三种子分辨（$t \\ge 122$） \\\\\n"
    "    高通量（$f = 1.0$），3.0 MeV & 两者皆可 & 未分辨（$t = 3.04$，$p = 0.094$） \\\\\n"
    "    全 32 行池化 & 仅方差诊断 & 非成像精度主张 \\\\\n"
    "    缺陷可检测性（逐棒 CNR） & 两者都不创造图像域可检测性 & §\\ref{sec:defect} \\\\\n"
    "    \\bottomrule"
)

OPS = [
    # ---- EN ----
    ('main.tex',
     "At the 100-epoch budget it leads linear interpolation on the imaging row wherever the margin is resolved --- by 2.3\\% at 2\\dgr{} in the single headline run (not seed-resolved, Section~\\ref{sec:limits}), and by 4.4 and 5.7\\% at 6 and 8\\dgr{} (single-seed point values; the two are not resolved at 4\\dgr{}, ratio 0.9994) --- and it also leads cubic interpolation",
     "At the 100-epoch budget it leads linear interpolation on the imaging row at every span, by 1.2\\%, 1.1\\%, 4.6\\% and 7.0\\% at $s = 2$, 4, 6 and 8\\dgr{} (three-seed means; seed statistics in Table~\\ref{tab:spanenergy}). The 4\\dgr{} margin (ratio 0.989) is not resolved against the seed spread, and the seed-level test of Section~\\ref{sec:row16} does not resolve the 2\\dgr{} margin either --- and it also leads cubic interpolation"),
    ('main.tex',
     "$0.989$ at 4\\dgr{} (a tie, not a separation)",
     "$0.989$ at 4\\dgr{} (not resolved)"),
    ('main.tex',
     "CNN completion when counting noise dominates, prefer interpolation on the high-flux imaging row, and",
     "CNN completion when counting noise dominates and treat interpolation as an acceptable and cheaper fallback where it does not;"),
    ('main.tex',
     "use CNN completion when noise dominates; prefer interpolation on the high-flux central row; and treat completion as contrast-preserving, not contrast-creating.",
     "use CNN completion when noise dominates; treat interpolation as an acceptable, cheaper fallback on the high-flux central row; and treat completion as contrast-preserving, not contrast-creating."),
    ('main.tex',
     "Two further scoping statements follow. The seven-seed expansion applies to the central-row comparison at 3.0~MeV; the defect case retains three seeds. The dose sweep is at a single energy and the span sweep is single-seed; these are pilot evidence, and extending them to multiple energies and seeds is left for future work.",
     "Three scoping statements follow. The seven-seed expansion applies to the central-row comparison at 3.0~MeV; the defect case and the span sweep retain three seeds, and the completion-to-cubic comparison of Table~\\ref{tab:cubic} is still a single run per span. The dose sweep is pooled over four energies but holds the object at 3.0~MeV training. None of the sweeps covers a second phantom geometry, a non-periodic object, or a missing-rod array."),
    ('main.tex', EN_TAB, EN_TAB_NEW),
    ('main.tex',
     "\n    \\midrule\n    Defective rod, CNN (3.0~MeV) & --- & 0.000780 & --- & --- & --- \\\\\n    Defective rod, linear (3.0~MeV) & --- & 0.000736 & --- & --- & --- \\\\",
     ""),
    ('main.tex',
     "The defective-rod entries are from the 3.0~MeV case of Section~\\ref{sec:defect} at the 100-epoch checkpoint and use the \\textbf{first} Geant4 realisation (August 2026); Table~\\ref{tab:flawmulti} uses the \\textbf{third} realisation (October 2026), and absolute values are not comparable across the two; the defective-rod CNN is the network trained in-domain from scratch on the defect data (Section~\\ref{sec:defect}). ",
     "Defect-rod results are reported separately: the corner-rod cavity in Section~\\ref{sec:defect} and the edge-rod cavity in Table~\\ref{tab:flawedge}; the two Geant4 realisations (August and October 2026) are not comparable, as noted at Table~\\ref{tab:flawmulti}. "),
    # ---- ZH ----
    ('main_zh.tex',
     "为检验工作点叙述能否预测优势在何处出现，我们将角采样跨度从 2\\dgr{} 扫描到 8\\dgr{}。在密采样的 2\\dgr{} 端，线性插值凭其固定的局部平均仍具竞争力，因为它的局部平均恰与平滑信号的条件均值重合；但随着跨度变宽，其误差随它必须外推的曲率而增长。CNN 经训练习得对局部正弦图结构的表示，因此沿一条平缓得多的轨迹前进。在 100-epoch 预算下，它在成像行上、凡是边际可分辨之处都领先线性插值——2\\dgr{} 处 2.3\\%（单次正文运行，未按种子分辨，见 §\\ref{sec:limits}），6 与 8\\dgr{} 处 4.4 与 5.7\\%（单种子取值；二者在 4\\dgr{} 处未能分辨，比值 0.9994）——并且也领先于使用四个实测视角（比补全多一个）的三次插值；与更高阶规则的比较在补充材料中量化，其结论与如下读法一致：在真正稀疏的角采样下，让重建保持可用的是数据驱动的先验，而非更高阶的局部规则。",
     "为检验工作点叙述能否预测优势在何处出现，我们将角采样跨度从 2\\dgr{} 扫描到 8\\dgr{}。在密采样的 2\\dgr{} 端，线性插值凭其固定的局部平均仍具竞争力，因为它的局部平均恰与平滑信号的条件均值重合；但随着跨度变宽，其误差随它必须外推的曲率而增长。CNN 经训练习得对局部正弦图结构的表示，因此沿一条平缓得多的轨迹前进。在 100-epoch 预算下，它在成像行上、每个跨度都领先线性插值——在 $s = 2$、$4$、$6$ 与 $8$\\dgr{} 处分别领先 1.2\\%、1.1\\%、4.6\\% 与 7.0\\%（三种子均值；种子统计量见表~\\ref{tab:spanenergy}）。4\\dgr{} 处的边际（比值 0.989）相对种子散布未能分辨，而 §~\\ref{sec:row16} 的种子级检验同样未能分辨 2\\dgr{} 处的边际。并且也领先于使用四个实测视角（比补全多一个）的三次插值；与更高阶规则的比较在补充材料中量化，其结论与如下读法一致：在真正稀疏的角采样下，让重建保持可用的是数据驱动的先验，而非更高阶的局部规则。"),
    ('main_zh.tex',
     "4\\dgr{} 为 $0.989$（打平，而非分离）、",
     "4\\dgr{} 为 $0.989$（未能分辨）、"),
    ('main_zh.tex',
     "，4\\dgr{} 处是打平而非分离）（除 4\\dgr{} 处二者未能分辨外，补全均领先）",
     "，4\\dgr{} 处未能分辨（该档除外，补全均领先）"),
    ('main_zh.tex',
     "计数噪声主导时采用CNN 补全，高通量成像行上优先插值，",
     "计数噪声主导时采用 CNN 补全，否则插值是可接受的、更廉价的保底规则（fallback），"),
    ('main_zh.tex',
     "另有两点适用范围说明。七种子扩展仅适用于第 16 行在 3.0~MeV 上的比较；缺陷情形仍保留三个种子。剂量扫描只在单一能量上、跨度扫描为单种子，二者均为探索性证据；把它们扩展到多个能量与多个种子留待未来工作。",
     "另有三点适用范围说明。七种子扩展仅适用于第 16 行在 3.0~MeV 上的比较；缺陷情形与跨度扫描均保留三个种子，而补全对三次的比较（表~\\ref{tab:cubic}）仍是每跨度单次运行。剂量扫描跨四能量池化，但训练物体固定在 3.0~MeV。各扫描均未覆盖第二种物模几何、非周期物体或缺失棒阵列。"),
    ('main_zh.tex', ZH_TAB, ZH_TAB_NEW),
    ('main_zh.tex',
     "\n    \\midrule\n    缺陷棒，CNN（3.0~MeV） & --- & 0.000780 & --- & --- & --- \\\\\n    缺陷棒，线性插值（3.0~MeV） & --- & 0.000736 & --- & --- & --- \\\\",
     ""),
    ('main_zh.tex',
     "缺陷棒条目来自 §\\ref{sec:defect} 的 3.0~MeV 情形、100-epoch checkpoint，并使用\\textbf{第一批} Geant4 仿真（2026 年 8 月）；表~\\ref{tab:flawmulti} 使用\\textbf{第三批}仿真（2026 年 10 月），两表绝对值不可互比；缺陷棒 CNN 是在缺陷数据上从零开始域内训练的网络（§\\ref{sec:defect}）。",
     "缺陷棒结果单独汇报：角棒空腔见 §\\ref{sec:defect}（角落棒），边棒空腔见表~\\ref{tab:flawedge}；两批 Geant4 仿真（2026 年 8 月与 10 月）绝对值不可互比，如表~\\ref{tab:flawmulti} 注所示。"),
]

apply = '--apply' in sys.argv

# 先逐个校验 count==1（在 apply 之前统一校验，避免部分写入）
for fn, path in FILES.items():
    s = io.open(path, encoding='utf-8', newline='').read().replace('\r\n', '\n')
    for i, (f, old, new) in enumerate(OPS):
        if f != fn:
            continue
        c = s.count(old)
        if c != 1:
            print("FAIL [%s] op#%d count=%d (expected 1)" % (fn, i, c))
            idx = s.find(old[:40])
            print("  ctx:", repr(s[max(0,idx-30):idx+80]) if idx>=0 else "  (old not found)")
            sys.exit(1)
    print("[%s] check OK: %d ops, all unique" % (fn, sum(1 for f,_,_ in OPS if f==fn)))

if not apply:
    print("ALL CHECK OK")
    sys.exit(0)

for fn, path in FILES.items():
    s = io.open(path, encoding='utf-8', newline='').read().replace('\r\n', '\n')
    for (f, old, new) in OPS:
        if f == fn:
            assert s.count(old) == 1, "re-check failed for %s" % f
            s = s.replace(old, new)
    io.open(path, 'w', encoding='utf-8', newline='').write(s.replace('\n', '\r\n'))
    print("[%s] applied %d ops" % (fn, sum(1 for f,_,_ in OPS if f==fn)))
print("APPLIED")
