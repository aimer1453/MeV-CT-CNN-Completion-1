#!/usr/bin/env python
# -*- coding: utf-8 -*-
r"""
scripts/_fig_style.py —— 论文全部折线/面板图的统一风格基座。

设计约束（来自审稿重绘指令）
----------------------------
* 网络不可用，无法 pip 安装 nature-figure-style / SSCI-Plots / Icarus-Figures，
  故在此**内联**一套 Nature 风格的 rcParams + 调色板 + 小工具，零外部依赖
  （只用 numpy + matplotlib）。
* 配色统一用 **Okabe–Ito** 色盲友好 7 色；白底绘图时**去掉纯黑**，方法/域固定映射：
    - Linear 插值  -> 蓝   #0072B2
    - Cubic  插值  -> 绿   #009E73
    - CNN    补全  -> 朱红 #D55E00
    - 强调/第四色  -> 橙   #E69F00
    - 补充色        -> 天蓝 #56B4E9 / 黄 #F0E442 / 紫 #CC79A7
* 热力图/差异图统一用 RdBu_r（红=正/补全更优）或 cividis。
* Nature 风格要点：白底、细 spine（0.6 pt）、去掉 top/right spine、tick 朝内、
  小字号（多栏图 6–7 pt）、无多余边框。

用法
----
    from _fig_style import NATURE_RC, OI, METHOD_COLORS, apply_rc, panel_label, save_png
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
from matplotlib import font_manager  # noqa: F401  (kept for downstream imports)

# ----------------------------------------------------------------------------
# 1. Okabe–Ito 色盲友好调色板（白底去黑）
# ----------------------------------------------------------------------------
OI = {
    "orange":   "#E69F00",
    "skyblue":  "#56B4E9",
    "bluish":   "#009E73",
    "yellow":   "#F0E442",
    "blue":     "#0072B2",
    "vermillion": "#D55E00",
    "purple":   "#CC79A7",
}
# 顺序列表（绘图循环用）
OI_ORDER = ["#E69F00", "#56B4E9", "#009E73", "#F0E442",
            "#0072B2", "#D55E00", "#CC79A7"]

# 方法固定配色（全论文统一，同一方法必须同色）
METHOD_COLORS = {
    "linear": "#0072B2",   # 蓝
    "cubic":  "#009E73",   # 绿
    "cnn":    "#D55E00",   # 朱红
}
METHOD_LABELS = {
    "linear": "Linear interpolation",
    "cubic":  "Cubic interpolation",
    "cnn":    "CNN completion",
}
# 域固定配色
DOMAIN_COLORS = {
    "uniform": "#0072B2",  # 蓝
    "defect":  "#D55E00",  # 朱红
}

# 热力图默认
HEAT_DIVERGING = "RdBu_r"   # 红=正（补全更优）
HEAT_SEQUENTIAL = "cividis"

# ----------------------------------------------------------------------------
# 2. Nature 风格 rcParams
# ----------------------------------------------------------------------------
NATURE_RC = {
    "font.family": "DejaVu Sans",
    "font.size": 7.0,
    "axes.linewidth": 0.6,
    "axes.labelsize": 7.2,
    "axes.titlesize": 7.6,
    "xtick.labelsize": 6.4,
    "ytick.labelsize": 6.4,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "xtick.major.size": 2.6,
    "ytick.major.size": 2.6,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "xtick.minor.size": 1.4,
    "ytick.minor.size": 1.4,
    "legend.fontsize": 6.4,
    "legend.frameon": False,
    "legend.handlelength": 1.6,
    "legend.handletextpad": 0.4,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "savefig.facecolor": "white",
    "savefig.dpi": 300,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "mathtext.fontset": "dejavusans",
    "axes.spines.top": False,
    "axes.spines.right": False,
}

# 多栏小图（三联图等）用的更紧字号
NATURE_RC_SMALL = dict(NATURE_RC)
NATURE_RC_SMALL.update({
    "font.size": 6.6,
    "axes.labelsize": 6.9,
    "axes.titlesize": 7.0,
    "xtick.labelsize": 6.2,
    "ytick.labelsize": 6.2,
    "legend.fontsize": 5.6,
})


def apply_rc(small: bool = False) -> None:
    """套用 Nature 风格 rcParams。"""
    plt.rcParams.update(NATURE_RC_SMALL if small else NATURE_RC)


def panel_label(ax, text: str, x: float = -0.16, y: float = 0.98,
                fontsize: float = 8.5, weight: str = "bold") -> None:
    """在子图左上角放 (a)/(b)/(c) 之类面板标签（默认 Nature 风格的粗体小标签）。"""
    ax.text(x, y, text, transform=ax.transAxes, fontsize=fontsize,
            fontweight=weight, va="top", ha="left")


def save_png(fig, path: Path, dpi: int = 300, tight: bool = True) -> Path:
    """只输出 PNG（审稿要求 PNG only，不得出 PDF）。"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if tight:
        fig.savefig(path, dpi=dpi, bbox_inches="tight")
    else:
        fig.savefig(path, dpi=dpi)
    plt.close(fig)
    return path
