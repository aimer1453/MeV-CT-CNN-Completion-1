#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""R37 stage 1: title + abstract reframing (nuclear-specific -> high-Z MeV CT methodology).

Usage:
  python scripts/_r37_reframe_s1.py --check
  python scripts/_r37_reframe_s1.py --apply

Numbers, tables, figures and statistical criteria are untouched.
Stage 1 only: title, abstract sentence 2, abstract closing sentence.
"""
import io
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LATEX = os.path.join(ROOT, "docs", "manuscript", "latex")
EN = os.path.join(LATEX, "main.tex")
ZH = os.path.join(LATEX, "main_zh.tex")

EN_RULES = [
    ("1.1 title",
     r"\title{When Counting Noise Dominates: CNN Projection Completion for Sparse-Angle MeV CT}",
     r"\title{When Counting Noise Dominates: An Operating-Point Map for CNN Projection Completion in High-Z MeV CT}"),

    ("1.2 abstract sentence 2",
     r"We map that operating point for megavoltage CT" + "\n"
     + r"of a dense $3\times3$ uranium-dioxide fuel-rod array.",
     r"We map that operating point for high-Z, high-attenuation MeV CT, using a dense" + "\n"
     + r"$3\times3$ uranium-dioxide fuel-rod array as an extreme testbed."),

    ("1.3 abstract closing sentence",
     r"treat completion as contrast-preserving, not contrast-creating." + "\n" + r"\end{abstract}",
     r"treat completion as contrast-preserving, not contrast-creating. Because the boundary is set by" + "\n"
     + r"measurable physical quantities --- photon counts, angular span, and detector row --- the design rule is" + "\n"
     + r"expected to transfer to other high-Z MeV CT scenarios where dense angular sampling is physically" + "\n"
     + r"impossible; that transfer is argued here, not demonstrated." + "\n" + r"\end{abstract}"),
]

ZH_RULES = [
    ("ZH 1.1 title",
     r"\title{当计数噪声主导时：稀疏角度 MeV CT 的 CNN 投影补全}",
     r"\title{当计数噪声主导时：高 Z 兆伏级 CT 中 CNN 投影补全的工作点地图}"),

    ("ZH 1.2 abstract sentence 2",
     "我们为致密 3×3 二氧化铀燃料棒阵列的稀疏角度\n兆伏级 CT 绘制这一工作点地图。",
     "我们以高 Z、高衰减的兆伏级 CT 为对象绘制这一工作点地图，并把致密 3×3 二氧化铀燃料棒阵列\n作为最严苛的测试床。"),

    ("ZH 1.3 abstract closing sentence",
     "并把补全视为保住对比度而非创造\n对比度。",
     "并把补全视为保住对比度而非创造\n对比度。由于该边界由可测量的物理量——光子计数、角跨度与探测器行——决定，该设计规则有望\n迁移到其他无法进行密集角度采样的高 Z 兆伏级 CT 场景；此处只论证这种迁移，并未实验验证。"),
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
            print("  [FAIL] %-34s count=%d" % (name, c))
            ok = False
            continue
        new_text = new_text.replace(o, n, 1)
        print("  [ok]   %-34s" % name)
    if apply_changes and ok:
        with io.open(path, "w", encoding="utf-8", newline="") as fh:
            fh.write(new_text)
        print("  -> written: %s" % os.path.basename(path))
    return ok


def main():
    apply_changes = "--apply" in sys.argv
    print("=== R37-S1 %s : EN ===" % ("APPLY" if apply_changes else "CHECK"))
    a = run(EN, EN_RULES, apply_changes)
    print("=== R37-S1 %s : ZH ===" % ("APPLY" if apply_changes else "CHECK"))
    b = run(ZH, ZH_RULES, apply_changes)
    if not (a and b):
        print("RESULT: FAILED (nothing written)")
        sys.exit(1)
    print("RESULT: OK")


if __name__ == "__main__":
    main()
