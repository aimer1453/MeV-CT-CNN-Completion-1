#!/usr/bin/env python
# -*- coding: utf-8 -*-
r"""
scripts/plot_fig_supp_all32.py —— 补充材料 Fig. S3：all-32-row vs row-16 投影域对比

正文 §5.6 只报 Row 16（唯一进入正弦图、决定重建的那一行）；全 32 行口径
（= 训练期验证指标）整体移入补充材料，本图与 Table S1 配套：

  (a) all 32 rows —— mask 加权全探测器口径，CNN 领先 52.6%（三种子 52.5%）；
  (b) row 16      —— 成像口径，两者统计上不可区分。

两栏相差约 3.5 倍：源剖面在探测器中心最强、边缘行接近零计数，all32 被
低计数行主导（那里受限的是计数噪声而非补全规则）。

数据来源（全部只读）
--------------------
  CNN 逐档 row16 / all32：
      runs/20261024_0400_cnn_bs8_lr1e-4_cnn_recon/test_metrics.json
      → projection_domain_check.<E>.by_input_phase.<A|B>.<roi>.by_split.test
      （两个输入相位按 n_valid_pixels 池化，与 p01_table_proj_domain.py 同口径）
  线性插值逐档（固定规则、与 checkpoint 无关，数值与 100-epoch 正文表逐位一致）：
      runs/_reviewer_p0/reviewer_p0_30ep.json
      → uniform.per_energy.<E>.methods.linear_interp["<roi>|random_test"].rmse_valid
  交叉校验（线性为固定规则，overall 应与多种子汇总逐位一致）：
      runs/_reviewer_p0/multiseed_summary.json → uniform.*.projection.linear_all32 / linear_row16

输出
----
  docs/figs/p01_reviewer_p0/fig_supp_all32_vs_row16.png
  docs/figs/p01_reviewer_p0/fig_supp_all32_data.csv

用法
----
  python scripts/plot_fig_supp_all32.py
"""

from __future__ import annotations

import csv
import json
import math
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import NullFormatter

ROOT = Path(__file__).resolve().parent.parent
for p in (ROOT, ROOT / "scripts"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

OUT = ROOT / "docs" / "figs" / "p01_reviewer_p0"
CNN_TM = ROOT / "runs" / "20261024_0400_cnn_bs8_lr1e-4_cnn_recon" / "test_metrics.json"
LIN_30 = ROOT / "runs" / "_reviewer_p0" / "reviewer_p0_30ep.json"
MULTISEED = ROOT / "runs" / "_reviewer_p0" / "multiseed_summary.json"

ENERGIES = ("1.5", "3", "4.5", "6")
LABELS = list(ENERGIES) + ["Overall"]
DPI = 300

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 8.0,
    "axes.linewidth": 0.7,
    "axes.labelsize": 8.0,
    "xtick.labelsize": 7.5,
    "ytick.labelsize": 7.5,
    "legend.fontsize": 6.8,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "savefig.facecolor": "white",
    "mathtext.fontset": "dejavusans",
})

C_CNN = "#B4483C"     # CNN 补全
C_LIN = "#3A6EA5"     # 线性插值


def cnn_by_energy() -> dict:
    """{energy: {roi: (rmse, n_valid_pixels)}}，两个输入相位 test split 池化。"""
    tm = json.loads(CNN_TM.read_text(encoding="utf-8"))
    pdc = tm["projection_domain_check"]
    out: dict[str, dict[str, tuple[float, int]]] = {}
    for e in ENERGIES:
        entry: dict[str, tuple[float, int]] = {}
        for roi in ("row16", "all32"):
            nv = se = 0
            for ph in ("A", "B"):
                t = pdc[e]["by_input_phase"][ph][roi]["by_split"]["test"]
                nv += t["n_valid_pixels"]
                se += t["rmse_valid"] ** 2 * t["n_valid_pixels"]
            entry[roi] = (math.sqrt(se / nv), nv)
        out[e] = entry
    return out


def linear_by_energy() -> dict:
    """线性插值（固定规则）：30ep json 与 100-epoch 正文表逐位一致，与 checkpoint 无关。"""
    d = json.loads(LIN_30.read_text(encoding="utf-8"))["uniform"]
    out: dict[str, dict[str, float]] = {}
    for e in ENERGIES:
        m = d["per_energy"][e]["methods"]["linear_interp"]
        out[e] = {roi: m[f"{roi}|random_test"]["rmse_valid"]
                  for roi in ("row16", "all32")}
    return out


def pooled(values: dict, roi: str) -> float:
    """跨能量按 n_valid_pixels 池化（与 p01_table_proj_domain.py 的 overall 同口径）。"""
    nv = se = 0
    for e in ENERGIES:
        v, n = values[e][roi]
        nv += n
        se += v * v * n
    return math.sqrt(se / nv)


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    cnn = cnn_by_energy()
    lin = linear_by_energy()

    # 交叉校验：线性插值是固定规则，overall 必须与多种子汇总逐位一致
    ms = json.loads(MULTISEED.read_text(encoding="utf-8"))["uniform"]
    ref_all32 = ms["20261022"]["projection"]["linear_all32"]
    ref_row16 = ms["20261022"]["projection"]["linear_row16"]
    lin_pool = {roi: pooled({e: {roi: (lin[e][roi], cnn[e][roi][1])}
                             for e in ENERGIES}, roi)
                for roi in ("row16", "all32")}
    for roi, ref in (("all32", ref_all32), ("row16", ref_row16)):
        # 容差 1e-6：两种池化顺序的浮点结合律差异（实测 ~2e-9），
        # 远低于正文 4 位小数的报告精度
        if abs(lin_pool[roi] - ref) > 1e-6:
            raise SystemExit(f"[FAIL] 线性插值 overall({roi}) 与多种子汇总不一致: "
                             f"{lin_pool[roi]} vs {ref}")

    series = {
        ("all32", "lin"): [lin[e]["all32"] for e in ENERGIES] + [lin_pool["all32"]],
        ("all32", "cnn"): [cnn[e]["all32"][0] for e in ENERGIES] + [pooled(cnn, "all32")],
        ("row16", "lin"): [lin[e]["row16"] for e in ENERGIES] + [lin_pool["row16"]],
        ("row16", "cnn"): [cnn[e]["row16"][0] for e in ENERGIES] + [pooled(cnn, "row16")],
    }

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.0), sharex=True)
    width = 0.34
    x = np.arange(len(LABELS))
    for ax, roi, title in ((axes[0], "all32",
                            "(a) all 32 rows — detector-wide (training-time metric)"),
                           (axes[1], "row16",
                            "(b) row 16 — the row that enters the sinogram")):
        for i, (key, col, lab) in enumerate((("lin", C_LIN, "Linear interp."),
                                             ("cnn", C_CNN, "CNN completion"))):
            vals = series[(roi, key)]
            ax.bar(x + (i - 0.5) * width, vals, width * 0.9, color=col,
                   label=lab, linewidth=0, zorder=3)
        ax.set_yscale("log")
        # 2026-10-29 (R36): 副刻度标签一律关掉。面板 (b) 的 ylim 跨不过一个数量级，
        # matplotlib 会把副刻度标成 "3x10^-1" / "6x10^-2" 这类长标签，比 (a) 的
        # "10^0 / 10^-1" 长一倍，横向挤压刻度区（审稿人报告的"刻度溢出"即此）。
        ax.yaxis.set_minor_formatter(NullFormatter())
        ax.set_xticks(x)
        ax.set_xticklabels(LABELS)
        ax.set_xlabel("Energy (MeV)")
        ax.grid(axis="y", which="major", color="#DDDDDD", linewidth=0.6, zorder=0)
        ax.grid(axis="y", which="minor", color="#EEEEEE", linewidth=0.4, zorder=0)
        ax.set_axisbelow(True)
        ax.set_title(title, fontsize=8.0, pad=4)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        ax.set_xlim(-0.55, len(LABELS) - 0.45)

    axes[0].set_ylabel("Masked RMSE ($-\\log$ domain)")
    axes[0].set_ylim(2.0e-2, 2.2)
    axes[1].set_ylim(5.0e-2, 3.0e-1)
    # 2026-10-29 (R36): 两个面板统一用短十进小数刻度，密度降到 4-6 个，
    # 既避免 (b) 的尾数形式溢出，也让两栏刻度风格一致。
    axes[0].set_yticks([0.02, 0.05, 0.1, 0.2, 0.5, 1.0])
    axes[0].set_yticklabels(["0.02", "0.05", "0.1", "0.2", "0.5", "1"])
    axes[1].set_yticks([0.05, 0.1, 0.2, 0.3])
    axes[1].set_yticklabels(["0.05", "0.1", "0.2", "0.3"])
    h, lb = axes[0].get_legend_handles_labels()
    fig.legend(h, lb, loc="lower center", ncol=2, frameon=False,
               bbox_to_anchor=(0.5, 0.215), handlelength=1.2, columnspacing=1.8)

    a_lin, a_cnn = series[("all32", "lin")][-1], series[("all32", "cnn")][-1]
    r_lin, r_cnn = series[("row16", "lin")][-1], series[("row16", "cnn")][-1]
    txt = (f"Overall, all 32 rows: {a_cnn:.4f} (CNN) vs {a_lin:.4f} (linear) — "
           f"{100 * (1 - a_cnn / a_lin):.1f}% lead (three-seed mean 52.5%)\n"
           f"Overall, row 16: {r_cnn:.4f} vs {r_lin:.4f} — "
           f"{100 * (1 - r_cnn / r_lin):.1f}% single-run margin, not seed-resolved\n"
           f"All-32 / row-16 error ratio: {a_cnn / r_cnn:.1f}× (CNN), "
           f"{a_lin / r_lin:.1f}× (linear) — low-count edge rows dominate the 32-row mean.")
    # 3 行、每行 <=115 字符：旧版第 3 行 180+ 字符宽于画布，savefig 未用 tight bbox，
    # 两端被直接裁掉（"All-" 与行尾丢失），审稿人放大即读成乱码。
    fig.text(0.5, 0.008, txt, ha="center", va="bottom", fontsize=7.2,
             color="#333333", linespacing=1.45)
    fig.subplots_adjust(left=0.085, right=0.995, top=0.88, bottom=0.385, wspace=0.16)

    p = OUT / "fig_supp_all32_vs_row16.png"
    # bbox_inches="tight"：兜底防止底部文案被画布裁切（旧版裁掉了两端字符）
    fig.savefig(p, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] {p}")

    # 副产物：落盘 csv，README 里逐格可查
    with open(OUT / "fig_supp_all32_data.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["group", "roi", "linear_interp", "cnn_completion",
                    "cnn_lead_pct", "cnn_n_valid_pixels"])
        for k, g in enumerate(LABELS):
            for roi in ("all32", "row16"):
                lv = series[(roi, "lin")][k]
                cv = series[(roi, "cnn")][k]
                n = "" if g == "Overall" else cnn[ENERGIES[k]][roi][1]
                w.writerow([g, roi, f"{lv:.6f}", f"{cv:.6f}",
                            f"{100 * (1 - cv / lv):+.2f}", n])
    print(f"[OK] {OUT / 'fig_supp_all32_data.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
