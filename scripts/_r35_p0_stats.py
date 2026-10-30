#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""R35: reviewer-response P0 items 1-6 (statistical conventions + realisation labels).

Usage:
  python scripts/_r35_p0_stats.py --check
  python scripts/_r35_p0_stats.py --apply

P0-1 equal-weight negative result in abstract + conclusion
P0-2 per-energy convention stated in the Table 5 note; abstract uses equal-weight only
P0-3 Fig 9 f=1.0 aggregation explained (four-energy mean vs single energy; NOT 360 vs 45 views)
P0-4 Geant4 realisation labels on Table 6 / Table 9 and in the body
P0-5 single-seed sweep downgraded to point estimates
P0-6 image-domain 6.0% stated as a point estimate with a CI crossing zero
"""
import io
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LATEX = os.path.join(ROOT, "docs", "manuscript", "latex")
EN = os.path.join(LATEX, "main.tex")
ZH = os.path.join(LATEX, "main_zh.tex")

EN_RULES = [
    # ---------------- P0-1 / P0-2 : abstract ----------------
    ("abs: equal-weight result",
     r"resolved from linear interpolation across seven training seeds (5 wins, 2 losses, p=0.23). The" + "\n"
     + r"completion's advantage is confined to where counting noise dominates --- the 1.5~MeV channel, wide",
     r"resolved from linear interpolation across seven training seeds (5 wins, 2 losses, p=0.23); under the" + "\n"
     + r"equal-weight paired analysis, which is the primary statistic of this paper, the completion trails" + "\n"
     + r"interpolation by $2.05\%$ with a $95\%$ confidence interval that excludes zero. The completion's" + "\n"
     + r"advantage is confined to where counting noise dominates --- the 1.5~MeV channel, wide"),

    ("abs: per-energy convention",
     r"Per energy the sign is positive at 1.5~MeV ($+4.33\%$)",
     r"Per energy the equal-weight sign is positive at 1.5~MeV ($+4.33\%$)"),

    # ---------------- P0-2 : Table 5 note ----------------
    ("tab:projdomain: pooling convention",
     r"three-seed statistics in Table~\ref{tab:supp32}. Pooling here is in quadrature",
     r"three-seed statistics in Table~\ref{tab:supp32}. The per-energy improvement in this table is" + "\n"
     + r"quadrature-pooled; the equal-weight paired analysis gives $+4.33$, $-6.66$, $-8.25$ and $-9.18\%$ at" + "\n"
     + r"1.5, 3.0, 4.5 and 6.0~MeV (Section~\ref{sec:row16}). Pooling here is in quadrature"),

    # ---------------- P0-1 / P0-6 : conclusion ----------------
    ("conclusion: equal-weight result",
     r"0.0875 $\pm$ 0.0017 against a seed-stable 0.0884) --- which is the expected behaviour where measurement variance is already small",
     r"0.0875 $\pm$ 0.0017 against a seed-stable 0.0884). Under the equal-weight paired analysis, which weights every held-out view alike, the completion trails by $2.05\%$ with a $95\%$ CI that excludes zero; the two summaries answer different questions and both are reported. This is the expected behaviour where measurement variance is already small"),

    ("conclusion: 6.0% not established",
     r"In the image domain the completion leads by 6.0\% on RMSE across seeds (0.001235 $\pm$ 0.000009 against 0.001314), again concentrated at the lowest flux, while interpolation retains the marginally better SSIM.",
     r"In the image domain the point estimate is a $6.0\%$ RMSE advantage across seeds (0.001235 $\pm$ 0.000009 against 0.001314), again concentrated at the lowest flux, but the paired interval over the twelve (energy, seed) units crosses zero, so the advantage is not established; interpolation retains the marginally better SSIM."),

    # ---------------- P0-3 : Fig 9 aggregation ----------------
    ("fig:dose: f=1.0 aggregation",
     r"three rules coincide (0.1101, 0.1089 and 0.1130 at $f = 1.0$); as the counts thin the CNN curve rises",
     r"three rules coincide (0.1101, 0.1089 and 0.1130 at $f = 1.0$). These $f = 1.0$ entries are equal-weight" + "\n"
     + r"means over the four energies, with the CNN averaged over three seeds, whereas the per-energy rows of" + "\n"
     + r"Table~\ref{tab:projdomain} are single-energy values from the headline run (at 3.0~MeV, 0.0884 linear" + "\n"
     + r"against 0.0912 CNN); both are scored on the same 45 held-out test views, so the difference is one of" + "\n"
     + r"aggregation and not of evaluation protocol. As the counts thin the CNN curve rises"),

    # ---------------- P0-4 : realisation labels ----------------
    ("tab:baseline: first realisation",
     r"The defective-rod entries are from the 3.0~MeV case of Section~\ref{sec:defect} at the 100-epoch checkpoint;",
     r"The defective-rod entries are from the 3.0~MeV case of Section~\ref{sec:defect} at the 100-epoch checkpoint and use the \textbf{first} Geant4 realisation (August 2026); Table~\ref{tab:flawmulti} uses the \textbf{third} realisation (October 2026), and absolute values are not comparable across the two;"),

    ("tab:flawmulti: third realisation",
     r"\footnotesize PSNR$_c$ uses one data range (0.1278, the largest of the four dense ranges) for all",
     r"\footnotesize This table uses the \textbf{third} Geant4 realisation (October 2026); the defective-rod entries of Table~\ref{tab:baseline} use the \textbf{first} realisation (August 2026), and absolute values are not comparable across the two. PSNR$_c$ uses one data range (0.1278, the largest of the four dense ranges) for all"),

    ("limits: defect realisation",
     r"The defect image-domain error is a clear win for linear interpolation at 3.0~MeV (0.000642 against 0.000724, a three-seed mean)",
     r"The defect image-domain error is a clear win for linear interpolation at 3.0~MeV on the third realisation (0.000642 against 0.000724, a three-seed mean)"),

    ("conclusion: defect realisation",
     r"On the asymmetric assembly the completion retains $99.1 \pm 1.4\%$ of the projection-domain defect contrast",
     r"On the asymmetric assembly (third Geant4 realisation) the completion retains $99.1 \pm 1.4\%$ of the projection-domain defect contrast"),

    # ---------------- P0-5 : single-seed downgrade ----------------
    ("fig:span: single-seed exploratory",
     r"and the single-seed sweep provides no seed statistics.",
     r"and the sweep is single-seed exploratory, providing no seed statistics, so no span ordering reported here is established."),

    ("5.8: image-domain point estimate",
     r"In the image domain the completion leads throughout, by 6.1, 5.7, 6.4 and 7.0\% over linear interpolation",
     r"In the image domain the point estimate favours the completion throughout, by 6.1, 5.7, 6.4 and 7.0\% over linear interpolation"),

    ("5.8: cubic 1.5 MeV point estimate",
     r"the only ordering that holds at every span is at 1.5~MeV, where the completion leads throughout (Table~\ref{tab:cubic})",
     r"the only ordering that holds at every span as a point estimate is at 1.5~MeV, where the completion is ahead at each of the four spans (Table~\ref{tab:cubic})"),

    ("tab:cubic: single-seed exploratory",
     r"the completion is a single training run per span, so the per-cell differences carry no seed statistics",
     r"the completion is a single training run per span, so this is a single-seed exploratory comparison: the per-cell differences carry no seed statistics"),

    # ---------------- P0-5 : span-sweep section opener ----------------
    ("5.8: sweep header downgrade",
     r"Figure~\ref{fig:span} reports both domains as the measured views are thinned from 180 ($s = 2$\dgr{}) to 45 ($s = 8$\dgr{}).",
     r"Figure~\ref{fig:span} reports both domains as the measured views are thinned from 180 ($s = 2$\dgr{}) to 45 ($s = 8$\dgr{}); the sweep is single-seed throughout, so every span ratio quoted below is a point estimate and none of the span orderings is established."),
]

ZH_RULES = [
    ("ZH abs: equal-weight result",
     "CNN 补全在七个训练种子下未能从线性插值中分辨出来（5 胜 2 负，p=0.23）。补全的优势仅限于计数噪声主导之处",
     "CNN 补全在七个训练种子下未能从线性插值中分辨出来（5 胜 2 负，p=0.23）；但在等权配对分析——本文的首要统计口径——下，补全落后插值 2.05\\%，其 95\\% 置信区间不含零。补全的优势仅限于计数噪声主导之处"),

    ("ZH abs: per-energy convention",
     "逐能量地，符号在 $1.5$~MeV 处为正",
     "在等权口径下，逐能量的符号在 $1.5$~MeV 处为正"),

    ("ZH tab:projdomain: pooling convention",
     "此处池化为二次方，故每个视角按其误差平方贡献",
     "本表的逐能量改善为二次方池化口径；等权配对分析在 1.5、3.0、4.5、6.0~MeV 上分别给出 $+4.33$、$-6.66$、$-8.25$、$-9.18\\%$（§\\ref{sec:row16}）。此处池化为二次方，故每个视角按其误差平方贡献"),

    ("ZH conclusion: equal-weight result",
     "0.0875 $\\pm$ 0.0017 对种子稳定的 0.0884）——这在测量方",
     "0.0875 $\\pm$ 0.0017 对种子稳定的 0.0884）。在等权配对分析下（每个留出视角等权），补全落后 $2.05\\%$，其 $95\\%$ 置信区间不含零；两种汇总回答不同问题，本文一并报告。这在测量方"),

    ("ZH conclusion: 6.0% not established",
     "在图像域，补全跨种子在 RMSE 上领先 6.0\\%（0.001235 $\\pm$ 0.000009 对 0.001314），同样集中于最低通量，而插值保持略好的 SSIM。",
     "在图像域，跨种子的点估计为补全在 RMSE 上领先 6.0\\%（0.001235 $\\pm$ 0.000009 对 0.001314），同样集中于最低通量，但十二个（能量，种子）配对单元的区间跨零，故该优势未被确立；插值保持略好的 SSIM。"),

    ("ZH conclusion: defect realisation",
     "在非对称组件上，补全保住了投影域缺陷对比度的 $99.1 \\pm 1.4\\%$",
     "在非对称组件（第三批 Geant4 仿真）上，补全保住了投影域缺陷对比度的 $99.1 \\pm 1.4\\%$"),

    ("ZH tab:baseline: first realisation",
     "缺陷棒条目来自 §\\ref{sec:defect} 的 3.0~MeV 情形、100-epoch checkpoint；",
     "缺陷棒条目来自 §\\ref{sec:defect} 的 3.0~MeV 情形、100-epoch checkpoint，并使用\\textbf{第一批} Geant4 仿真（2026 年 8 月）；表~\\ref{tab:flawmulti} 使用\\textbf{第三批}仿真（2026 年 10 月），两表绝对值不可互比；"),

    ("ZH tab:flawmulti: third realisation",
     "\\footnotesize PSNR$_c$ 对四档使用同一个 data\\_range",
     "\\footnotesize 本表使用\\textbf{第三批} Geant4 仿真（2026 年 10 月）；表~\\ref{tab:baseline} 的缺陷棒条目使用\\textbf{第一批}仿真（2026 年 8 月），两表绝对值不可互比。PSNR$_c$ 对四档使用同一个 data\\_range"),

    ("ZH limits: defect realisation",
     "缺陷图像域误差在 3.0~MeV 上仍是线性插值的明确胜出（0.000642 对 0.000724，三种子均值）",
     "缺陷图像域误差在 3.0~MeV 上仍是线性插值的明确胜出（第三批仿真，0.000642 对 0.000724，三种子均值）"),

    ("ZH fig:span: single-seed exploratory",
     "且单种子扫描不提供种子统计。",
     "且该扫描为单种子探索性结果，不提供种子统计，故此处任何跨度排序都未被确立。"),

    ("ZH 5.8: image-domain point estimate",
     "在图像域，补全全程领先：相对线性插值为 6.1、5.7、6.4、7.0\\%",
     "在图像域，点估计显示补全全程领先：相对线性插值为 6.1、5.7、6.4、7.0\\%"),

    ("ZH fig:dose: f=1.0 aggregation",
     "（$f = 1.0$ 处分别为 0.1101、0.1089 与 0.1130）；随计数",
     "（$f = 1.0$ 处分别为 0.1101、0.1089 与 0.1130）。这三个 $f = 1.0$ 取值是四能量的等权平均、CNN 再对三个种子取均值；而表~\\ref{tab:projdomain} 的逐能量行是正文运行的单能量值（3.0~MeV：线性 0.0884、CNN 0.0912）。两者都在同一批 45 个留出测试视角上评分，故差异来自聚合方式而非评测协议。随计数"),
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
            print("  [FAIL] %-42s count=%d" % (name, c))
            ok = False
            continue
        new_text = new_text.replace(o, n, 1)
        print("  [ok]   %-42s" % name)
    if apply_changes and ok:
        with io.open(path, "w", encoding="utf-8", newline="") as fh:
            fh.write(new_text)
        print("  -> written: %s" % os.path.basename(path))
    return ok


def main():
    apply_changes = "--apply" in sys.argv
    print("=== R35 %s : EN ===" % ("APPLY" if apply_changes else "CHECK"))
    a = run(EN, EN_RULES, apply_changes)
    print("=== R35 %s : ZH ===" % ("APPLY" if apply_changes else "CHECK"))
    b = run(ZH, ZH_RULES, apply_changes)
    if not (a and b):
        print("RESULT: FAILED (nothing written)")
        sys.exit(1)
    print("RESULT: OK")


if __name__ == "__main__":
    main()
