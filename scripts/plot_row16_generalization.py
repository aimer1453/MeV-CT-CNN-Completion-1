#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Fig S10：Row-16-only 消融的跨行泛化（双 panel + log 纵轴）。

左 panel = A 组（32 行训练），右 panel = B 组（Row-16-only 训练），
两个 panel 共享 y 轴（log），横轴为 Row 0/15/16/31。
B 的 Row 16 是训练行（加斜纹强调）。

数据源：runs/_row16_generalization.json（本仓库脚本 eval_row16_generalization.py 产出）
用法：python scripts/plot_row16_generalization.py
产出：docs/figs/p01_row16_generalization/fig_row16_generalization.{png,pdf}
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
J = json.loads((ROOT / "runs" / "_row16_generalization.json").read_text(encoding="utf-8"))

ROWS = [0, 15, 16, 31]
LABELS = ["Row 0\n(edge)", "Row 15\n(mirror of 16)", "Row 16\n(trained)", "Row 31\n(edge)"]
A = [J["table"][f"row{r}"]["A"] for r in ROWS]
B = [J["table"][f"row{r}"]["B"] for r in ROWS]
AE = [J["table"][f"row{r}"]["A_std"] for r in ROWS]
BE = [J["table"][f"row{r}"]["B_std"] for r in ROWS]

C_A, C_B = "#185FA5", "#D85A30"      # 蓝 = A 组，橙 = B 组
E_A, E_B = "#0C447C", "#993C1D"      # 深色描边

fig, axes = plt.subplots(1, 2, figsize=(10.6, 4.4), sharey=True)
for ax, vals, errs, title, c, ec in (
    (axes[0], A, AE, "A  ·  32-row training", C_A, E_A),
    (axes[1], B, BE, "B  ·  Row-16-only training", C_B, E_B),
):
    x = np.arange(len(ROWS))
    ax.bar(x, vals, width=0.62, color=c, edgecolor=ec, linewidth=0.7, zorder=3)
    ax.errorbar(x, vals, yerr=errs, fmt="none", ecolor="#2b2b2b",
                elinewidth=1.0, capsize=3, zorder=4)
    ax.set_yscale("log")
    ax.set_ylim(0.055, 3.0)
    ax.set_xticks(x)
    ax.set_xticklabels(LABELS, fontsize=8.6)
    ax.set_title(title, fontsize=11.5)
    ax.grid(axis="y", which="both", color="0.86", linewidth=0.6, zorder=0)
    for xi, v, e in zip(x, vals, errs):
        ax.text(xi, min(v * 1.14, 2.55), f"{v:.3f}", ha="center", va="bottom", fontsize=8.6)

# B 的 Row 16 是训练行：加斜纹强调
axes[1].patches[2].set_hatch("////")
axes[1].patches[2].set_edgecolor("#2b2b2b")
axes[0].set_ylabel("RMSE  (−log physical domain, log scale)", fontsize=10)

# 左 panel 标注镜像等价
axes[0].text(0.5, 0.93, "Row 15 ≈ Row 16 (Δ = 0.03%)",
             transform=axes[0].transAxes, ha="center", va="top",
             fontsize=8.8, color="0.25",
             bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="0.8", lw=0.6))

fig.tight_layout()
outdir = ROOT / "docs" / "figs" / "p01_row16_generalization"
outdir.mkdir(parents=True, exist_ok=True)
for ext in ("png", "pdf"):
    fig.savefig(outdir / f"fig_row16_generalization.{ext}", dpi=300, bbox_inches="tight")
print("[saved]", outdir / "fig_row16_generalization.png")
print("[saved]", outdir / "fig_row16_generalization.pdf")
print("[data] A:", [round(v, 6) for v in A])
print("[data] B:", [round(v, 6) for v in B])
