#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Fig S11：C 组（1D 真单行）与 A/B 组的三方跨行对比（分组柱状 + log 纵轴）。

横轴 = 探测器行（Row 0 / 15 / 16 / 31），每组 3 根柱 = A（2D 32 行）/ B（2D 伪单行）/ C（1D 真单行）。
log 纵轴是唯一能同时容纳 0.082 与 1.829（22 倍跨度）的画法。

刻意标注的三个事实：
  1. Row 16：C 最低（0.082）—— 真单行架构在该行的精度最高，且参数只有 2D 的 34.9%；
  2. Row 15：C 与 A 打平（0.0966 vs 0.0966），而 B 崩到 0.708 —— 证明 B 的崩溃来自伪单行设置；
  3. Row 0/31：C 全面劣于 A（+388% / +347%），是训练域外的外推失败。

数据源：runs/_row1d_generalization.json（scripts/eval_row1d.py 产出）
用法：python scripts/plot_row1d_generalization.py
产出：docs/figs/p01_row1d_generalization/fig_row1d_generalization.{png,pdf}
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
J = json.loads((ROOT / "runs" / "_row1d_generalization.json").read_text(encoding="utf-8"))

ROWS = [0, 15, 16, 31]
LABELS = ["Row 0\n(low flux)", "Row 15\n(mirror of 16)", "Row 16\n(trained)", "Row 31\n(low flux)"]
SERIES = [
    ("A · 2D U-Net, 32 rows", "A_base32_3mev", "#185FA5", "#0C447C", None),
    ("B · 2D U-Net, pseudo 1-row", "B_row16_3mev", "#D85A30", "#993C1D", "\\\\\\\\"),
    ("C · 1D U-Net, true 1-row", "C_cnn1d_row16", "#2E8B57", "#1B5E3A", None),
]

fig, ax = plt.subplots(figsize=(9.2, 4.6))
x = np.arange(len(ROWS))
w = 0.26

for i, (label, key, c, ec, hatch) in enumerate(SERIES):
    vals = [J["full360"][key][str(r)]["mean"] for r in ROWS]
    errs = [J["full360"][key][str(r)]["std"] for r in ROWS]
    off = (i - 1) * w
    bars = ax.bar(x + off, vals, width=w * 0.92, color=c, edgecolor=ec, linewidth=0.7,
                  label=label, zorder=3, hatch=hatch)
    ax.errorbar(x + off, vals, yerr=errs, fmt="none", ecolor="#2b2b2b",
                elinewidth=0.9, capsize=2.5, zorder=4)
    for b, v, e in zip(bars, vals, errs):
        ax.text(b.get_x() + b.get_width() / 2, min(v * 1.10, 2.9), f"{v:.3f}",
                ha="center", va="bottom", fontsize=7.6, zorder=5)

ax.set_yscale("log")
ax.set_ylim(0.05, 3.9)
ax.set_xticks(x)
ax.set_xticklabels(LABELS, fontsize=9)
ax.set_ylabel("per-row RMSE  (−log physical domain, log scale)", fontsize=10)
ax.set_title("Single-row ablation: 2D 32-row vs 2D pseudo-single-row vs 1D true single-row\n"
             "3.0 MeV, 3 seeds, 360-view protocol", fontsize=10.5)
ax.grid(axis="y", which="both", color="0.86", linewidth=0.6, zorder=0)
# 图例放到图外下方：图内上方全是数据（Row0/31 的 1.8/1.7），放哪都会压柱
ax.legend(fontsize=8.2, loc="upper center", bbox_to_anchor=(0.5, -0.16),
          ncol=3, frameon=False)

# ---- 三处注记：一律放进「无柱」的空白带，不用长箭头（长箭头会穿过中间那根 B 柱）----
ax.text(x[0], 2.55, "C: +388% vs A\n(extrapolation)", ha="center", va="bottom",
        fontsize=7.8, color="#8a1c1c")
ax.text(x[1], 0.92, "C ≈ A on the mirror row\n0.0966 vs 0.0966 (−0.04%)", ha="center",
        va="bottom", fontsize=7.8, color="#0C447C")
ax.text(x[2] + 0.06, 0.122, "C best on its own row\n−15.3% vs A, −5.7% vs B", ha="center",
        va="bottom", fontsize=7.8, color="#1B5E3A", linespacing=1.25)
ax.text(x[3], 2.35, "C: +347% vs A\n(extrapolation)", ha="center", va="bottom",
        fontsize=7.8, color="#8a1c1c")

fig.tight_layout(pad=0.4)
OUT = ROOT / "docs" / "figs" / "p01_row1d_generalization"
OUT.mkdir(parents=True, exist_ok=True)
for ext, dpi in (("png", 320), ("pdf", None)):
    fig.savefig(OUT / f"fig_row1d_generalization.{ext}", dpi=dpi, facecolor="white",
                bbox_inches="tight")
print(f"[saved] {OUT / 'fig_row1d_generalization.png'}")
print(f"[saved] {OUT / 'fig_row1d_generalization.pdf'}")
