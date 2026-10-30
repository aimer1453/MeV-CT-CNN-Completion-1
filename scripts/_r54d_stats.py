# -*- coding: utf-8 -*-
"""
R54d statistical red-line rewrite (Paper 1 freeze).

Scope (per AGENT BRIEF item [2] + execution order [6]):
  - Demote the equal-weight +2.05% claim from "CI excludes zero / significant
    against completion" to "not resolved": keep the +2.05% point estimate, delete
    the "95% CI that excludes zero" / "significant" wording, and insert the
    cluster-robust view-level interval statement
    "neither the seed-level test (p=0.23) nor a cluster-robust view-level
     interval (30 deg sectors, 48 blocks) excludes zero; we therefore report the
     central row as not resolved".
  - Sync abstract / conclusion / decision table (the operative excludes-zero line
    lives in tab:decision, not tab:projdomain; tab:projdomain note is clean).
  - Fix the §5.3 logic-error clause ("the 100-epoch random value ... inherits
    this conclusion") -- remove the unsound inference, keep the 30-epoch grouped
    conclusion, and state the 100-epoch random value is a separate checkpoint.

OUT OF SCOPE (kept on purpose, not method-superiority claims):
  - L120 "excludes zero-count pixels" (data filter)
  - L236 / L1049 / L1117 "does not significantly overestimate" (negative)
  - L281 resolution-criterion definition ("resolved only when ... excludes zero")
  - L410 / L902 "not significant" (null results, desired)
  - L695 methodological description of reading a bootstrap interval
  - L946 "defect signal is significant in the measured projections" (legitimate
    detectability, |t| 13.76-24.09)

EN + ZH both rewritten to stay in sync.

Usage: python _r54d_stats.py --check | --apply
"""
import sys
import os

HERE = os.path.dirname(os.path.abspath(__file__))
LATEX = os.path.normpath(os.path.join(HERE, "..", "docs", "manuscript", "latex"))

FILES = {
    "en": os.path.join(LATEX, "main.tex"),
    "zh": os.path.join(LATEX, "main_zh.tex"),
}

# (old, new) pairs applied to BOTH files where keys are file-specific.
OPS = {
    "en": [
        # OP1 abstract L74-75
        (
            "completion trails interpolation by $2.05\\%$ with a $95\\%$ CI that excludes zero; the seed-level test\n"
            "treats the method-level difference as unresolved. The completion's",
            "completion trails interpolation by $2.05\\%$ as a point estimate, but neither the seed-level test ($p = 0.23$) nor a cluster-robust view-level interval ($30^\\circ$ sectors, 48 blocks) excludes zero; we therefore report the central row as not resolved. The completion's",
        ),
        # OP2 §5.6 L391 single headline run (drop the significance editorial)
        (
            "difference is $-0.0029$ with a paired bootstrap $95\\%$ CI of $[-0.0056,-0.0003]$ that excludes zero.",
            "difference is $-0.0029$ with a paired bootstrap $95\\%$ CI of $[-0.0056,-0.0003]$.",
        ),
        # OP3 §5.6 L398 primary statement
        (
            "The primary statistical statement of this paper is the equal-weight paired analysis over held-out views. It is the only one of the two summaries that supports a claim about accuracy on a typical view, and it places the completion $2.05\\%$ behind interpolation with a $95\\%$ CI that excludes zero: at the level of individual views the difference is significant, and it runs against the completion. The quadrature pool of Table~\\ref{tab:projdomain} is retained as a variance diagnostic and is not used as evidence of accuracy.",
            "The primary statistical statement of this paper is the equal-weight paired analysis over held-out views. It is the only one of the two summaries that supports a claim about accuracy on a typical view, and it places the completion $2.05\\%$ behind interpolation as a point estimate. However, neither the seed-level test ($p = 0.23$, seven seeds) nor a cluster-robust view-level interval ($30^\\circ$ sectors, 48 blocks) excludes zero, so we report the central row as not resolved: the observed $2.05\\%$ deficit is a point estimate whose uncertainty, after accounting for seed and view-cluster structure, is not distinguishable from zero. The quadrature pool of Table~\\ref{tab:projdomain} is retained as a variance diagnostic and is not used as evidence of accuracy.",
        ),
        # OP4 conclusion L944
        (
            "Under the equal-weight paired analysis, which weights every held-out view alike, the completion trails by $2.05\\%$ with a $95\\%$ CI that excludes zero; the two summaries answer different questions and both are reported.",
            "Under the equal-weight paired analysis, which weights every held-out view alike, the completion trails by $2.05\\%$ as a point estimate; however, neither the seed-level test ($p = 0.23$) nor a cluster-robust view-level interval ($30^\\circ$ sectors, 48 blocks) excludes zero, so the central row is reported as not resolved, and the two summaries answer different questions and both are reported.",
        ),
        # OP5 tab:decision L960
        (
            "Not resolved across 7 seeds ($p = 0.23$); view-level $2.05\\%$ deficit, CI excludes zero \\\\",
            "Not resolved across 7 seeds ($p = 0.23$); view-level $2.05\\%$ point estimate, cluster-robust interval does not exclude zero \\\\",
        ),
        # OP6 §5.3 L355 logic-error clause
        (
            "indicating the random split does not significantly overestimate performance; the 100-epoch random value of 0.3804 inherits this conclusion, because removing the rotated duplicates can only lower the metric, and the 90\\dgr{}-shift floor of 0.0232 (above) sits far below every method.",
            "indicating the random split does not significantly overestimate performance at the 30-epoch checkpoint. The 100-epoch random-split value of 0.3804 reported above is a separate checkpoint and is not pooled with the 30-epoch grouped comparison.",
        ),
        # OP7 §5.6 L400-402 per-energy "every interval excluding zero" (drop the
        # significance claim; keep the per-energy point estimates / sign).
        (
            "Per energy the sign is explicit: the completion leads by $4.33\\%$ at $1.5$~MeV and trails by $6.66$,\n$8.25$ and $9.18\\%$ at $3.0$, $4.5$ and $6.0$~MeV (paired bootstrap over the 45 views of each\nenergy, every interval excluding zero).",
            "Per energy the sign is explicit: the completion leads by $4.33\\%$ at $1.5$~MeV and trails by $6.66$,\n$8.25$ and $9.18\\%$ at $3.0$, $4.5$ and $6.0$~MeV (paired bootstrap over the 45 views of each\nenergy).",
        ),
    ],
    "zh": [
        # OP1 abstract L73
        (
            "补全落后插值 2.05\\%，其 95\\% 置信区间不含零；种子层级检验把方法层级的差异视为未决。",
            "补全落后插值 2.05\\% 作为点估计，但种子层级检验（$p = 0.23$）与顾及 $30^\\circ$ 视角扇区结构的簇稳健视角层区间（$30^\\circ$ 扇区、48 个块）均不含零，因此我们报告中心行未能分辨。",
        ),
        # OP2 §5.6 L323 single headline run
        (
            "差异为 $-0.0029$，配对 bootstrap $95\\%$ 置信区间 $[-0.0056,-0.0003]$\n不含零。",
            "差异为 $-0.0029$，配对 bootstrap $95\\%$ 置信区间 $[-0.0056,-0.0003]$。",
        ),
        # OP3 §5.6 L329 primary statement
        (
            "而它给出的是补全落后插值 $2.05\\%$、且 95\\% 置信区间不含零：在逐视角层级该差异显著，且方向不利于补全。",
            "而它给出的补全落后插值 $2.05\\%$ 只是点估计。然而，种子层级检验（$p = 0.23$，七个种子）与顾及 $30^\\circ$ 视角扇区结构的簇稳健视角层区间（$30^\\circ$ 扇区、48 个块）均不含零，因此我们报告中心行未能分辨：观测到的 $2.05\\%$ 劣势是一个点估计，在计入种子与视角簇结构后的不确定度无法与零区分。",
        ),
        # OP4 conclusion L773
        (
            "在等权配对分析下（每个留出视角等权），补全落后 $2.05\\%$，其 $95\\%$ 置信区间不含零；两种汇总回答不同问题，本文一并报告。",
            "在等权配对分析下（每个留出视角等权），补全落后 $2.05\\%$ 作为点估计；然而种子层级检验（$p = 0.23$）与顾及 $30^\\circ$ 视角扇区结构的簇稳健视角层区间（$30^\\circ$ 扇区、48 个块）均不含零，因此中心行报告为未能分辨，两种汇总回答不同问题，本文一并报告。",
        ),
        # OP5 tab:decision L788
        (
            "7 种子未分辨（$p = 0.23$）；视角层 $2.05\\%$ 劣势，CI 排除零 \\\\",
            "7 种子未分辨（$p = 0.23$）；视角层 $2.05\\%$ 点估计，簇稳健区间不含零 \\\\",
        ),
        # OP6 §5.3 L292 logic-error clause
        (
            "表明随机划分不会显著高估性能；100-epoch 的随机划分值 0.3804 继承了这一结论，因为移除旋转重复项只会降低指标，而 90\\dgr{} 平移地板 0.0232（如上）远低于每个方法。",
            "表明随机划分在 30-epoch checkpoint 下不会显著高估性能。上文报告的 100-epoch 随机划分值 0.3804 是另一个 checkpoint，不与 30-epoch 的分块比较合并。",
        ),
        # OP7 §5.6 L331 per-energy "每个区间都不含零" (drop the significance claim)
        (
            "（各档 45 个视角上的配对 bootstrap，每个区间都不含零）。跨种子时，",
            "（各档 45 个视角上的配对 bootstrap）。跨种子时，",
        ),
    ],
}


def normalize(text):
    return text.replace("\r\n", "\n").replace("\r", "\n")


def run(mode):
    for key, path in FILES.items():
        with open(path, "r", encoding="utf-8") as f:
            raw = f.read()
        text = normalize(raw)
        ops = OPS[key]
        for i, (old, new) in enumerate(ops, 1):
            n = text.count(old)
            if n != 1:
                raise SystemExit(
                    f"[{key}] OP{i}: expected exactly 1 match, found {n}.\n"
                    f"--- OLD ---\n{old}\n---"
                )
            if mode == "apply":
                text = text.replace(old, new)
        if mode == "apply":
            # restore CRLF to match the original file convention
            text = text.replace("\n", "\r\n")
            with open(path, "w", encoding="utf-8", newline="") as f:
                f.write(text)
            print(f"[{key}] applied {len(ops)} ops -> {path}")
        else:
            print(f"[{key}] check OK: {len(ops)} ops each matched exactly once")


if __name__ == "__main__":
    mode = sys.argv[1].lstrip("-") if len(sys.argv) > 1 else "check"
    if mode not in ("check", "apply"):
        raise SystemExit("usage: python _r54d_stats.py --check | --apply")
    run(mode)
