# -*- coding: utf-8 -*-
"""
R54e: central-row qualification sentence (adjudication 1) + ZH fallback desync fix.

Adjudication 1 (user-approved, evidence-backed by algorithm side): after the
seven-seed paragraph in Section sec:row16 (EN) / its ZH mirror, add ONE
qualification statement -- the near-tie under the random split presupposes the
90-degree-symmetric duplicates are in the training set; grouped split removes
them and the completion loses by a factor of 2.16 (runs/_grp_pairing_100ep.json).
Abstract and conclusion are NOT touched.

Bonus fix found during the R54c read-back: ZH conclusion still said
"偏爱插值" where the EN conclusion (and ZH abstract) use the fallback wording.
Synced to the ZH abstract's established phrase.

Adjudication 2 (Fig 9 caption 45-view sentence) required NO edit: R54b already
replaced it in both files (EN "Table~ reports 45 held-out test views whereas the
curves here pool all 180 completed views..." / ZH mirror), verified 0 residuals
of the old wording.

Usage: python _r54e_qualify.py --check | --apply
"""
import sys
import os

HERE = os.path.dirname(os.path.abspath(__file__))
LATEX = os.path.normpath(os.path.join(HERE, "..", "docs", "manuscript", "latex"))

FILES = {
    "en": os.path.join(LATEX, "main.tex"),
    "zh": os.path.join(LATEX, "main_zh.tex"),
}

# (old, new, expected_count)
OPS = {
    "en": [
        # op1: qualification paragraph after the seven-seed paragraph of sec:row16
        (
            "We therefore report the central row as not resolved at full flux, and locate the completion's value in the noise-dominated regime.",
            "We therefore report the central row as not resolved at full flux, and locate the completion's value in the noise-dominated regime.\n\nThe near-tie under the random split presupposes that the training set contains the 90\\dgr{}-symmetric duplicates of the test views; when the split removes them the completion loses to interpolation by a factor of 2.16 (Section~\\ref{sec:testset}). The 2\\dgr{} result is therefore a statement about this four-fold-symmetric testbed, not about sparse-angle completion in general.",
            1,
        ),
    ],
    "zh": [
        # op1: ZH mirror of the qualification paragraph
        (
            "因此我们报告中心行在满通量下未能分辨，并把补全的价值定位于噪声主导的工况。",
            "因此我们报告中心行在满通量下未能分辨，并把补全的价值定位于噪声主导的工况。\n\n随机划分下的近似打平，预设了训练集中含有测试视角的 90\\dgr{} 对称副本；一旦划分移除这些副本，补全便以 2.16 倍之差输给线性插值（§\\ref{sec:testset}）。因此 2\\dgr{} 的结果是对这一四重对称测试床的陈述，而非关于稀疏角度补全的一般性结论。",
            1,
        ),
        # op2: ZH conclusion fallback desync (EN conclusion uses the fallback wording;
        # ZH abstract already says 保底规则（fallback）-- align the conclusion with it)
        (
            "在高通量中心行上偏爱插值；",
            "在高通量中心行上把插值作为可接受的、更廉价的保底规则（fallback）；",
            1,
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
        for i, (old, new, cnt) in enumerate(ops, 1):
            n = text.count(old)
            if n != cnt:
                raise SystemExit(
                    f"[{key}] OP{i}: expected {cnt} match(es), found {n}.\n"
                    f"--- OLD ---\n{old}\n---"
                )
            if mode == "apply":
                text = text.replace(old, new)
        if mode == "apply":
            text = text.replace("\n", "\r\n")
            with open(path, "w", encoding="utf-8", newline="") as f:
                f.write(text)
            print(f"[{key}] applied {len(ops)} ops -> {path}")
        else:
            print(f"[{key}] check OK: {len(ops)} ops matched expected counts")


if __name__ == "__main__":
    mode = sys.argv[1].lstrip("-") if len(sys.argv) > 1 else "check"
    if mode not in ("check", "apply"):
        raise SystemExit("usage: python _r54e_qualify.py --check | --apply")
    run(mode)
