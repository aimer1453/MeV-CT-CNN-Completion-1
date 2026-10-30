#!/usr/bin/env python
# -*- coding: utf-8 -*-
r"""
scripts/plot_p01_reviewer_p0.py —— 审稿意见 P0 的两张图

图 1（fig_sector_split.png）
    分块划分（0–90 训练 / 90–180 验证 / 180–360 测试）对 `rmse_valid` 的影响。
    同一 best.pt、同一批 180 个测试视角，只换"哪些视角属于 test"。
    附 shift90 泄漏下限：pred(θ) = 实测(θ+90)，量化"90° 对称把目标视角确定到什么程度"。
    两栏：row16（进正弦图那一行）/ all32（全 32 行，训练期验证口径）。y 轴对数。

图 2（fig_train_curves_30ep.png）
    30 epoch 的验证曲线（4 档能量 + 池化），均匀数据与缺陷数据各一栏，
    5 epoch pilot 的终点用虚线标出，直观显示"是否还在降"。

数据来源（只读）
----------------
  runs/_reviewer_p0/reviewer_p0_30ep.json   ← scripts/p01_reviewer_p0.py
  runs/_reviewer_p0/reviewer_p0_5ep.json    ← 同上（5 epoch 权重）
  runs/20261023_2332_cnn_bs8_lr1e-4/train_log.csv   （均匀 30ep）
  runs/20261023_2327_cnn_bs8_lr1e-4/train_log.csv   （缺陷 30ep，3.0 MeV）

输出
----
  docs/figs/p01_reviewer_p0/fig_sector_split.png
  docs/figs/p01_reviewer_p0/fig_train_curves_30ep.png

用法
----
  conda run -n ct_full --no-capture-output python scripts/plot_p01_reviewer_p0.py
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

ROOT = Path(__file__).resolve().parent.parent
for p in (ROOT, ROOT / "scripts"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

OUT = ROOT / "docs" / "figs" / "p01_reviewer_p0"
P0 = ROOT / "runs" / "_reviewer_p0"
UNIF_30 = ROOT / "runs" / "20261023_2332_cnn_bs8_lr1e-4"
FLAW_30 = ROOT / "runs" / "20261023_2327_cnn_bs8_lr1e-4"
UNIF_5 = ROOT / "runs" / "20261022_2352_cnn_bs8_lr1e-4"
FLAW_5 = ROOT / "runs" / "20261023_2239_cnn_bs8_lr1e-4"

ENERGIES = ("1.5", "3", "4.5", "6")
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

C_CNN_R = "#D55E00"     # CNN 随机划分（Okabe–Ito 朱红）
C_CNN_S = "#E69F00"     # CNN 分块划分（Okabe–Ito 橙）
C_LIN = "#0072B2"       # 线性插值（Okabe–Ito 蓝）
C_SHIFT = "#7A7A7A"     # shift90 泄漏下限（中性灰保留）


# ------------------------------------------------------------------ 图 1
def fig_sector_split() -> Path:
    d30 = json.loads((P0 / "reviewer_p0_30ep.json").read_text(encoding="utf-8"))["uniform"]
    d5 = json.loads((P0 / "reviewer_p0_5ep.json").read_text(encoding="utf-8"))["uniform"]

    labels = list(ENERGIES) + ["Overall"]

    def get(blk, group: str, roi: str, series: str) -> float:
        """group: 能量字符串或 "overall"；roi: row16/all32；series: cnn_r/cnn_s/lin/shift。"""
        meth, sub = {"cnn_r": ("cnn", "random_test"),
                     "cnn_s": ("cnn", "sector_test"),
                     "lin": ("linear_interp", "sector_test"),
                     "shift": ("shift90", "sector_test")}[series]
        if group == "overall":
            return blk["overall"][meth][f"{roi}|{sub}"]
        return blk["per_energy"][group]["methods"][meth][f"{roi}|{sub}"]["rmse_valid"]

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.0), sharex=True)
    series = [("cnn_r", C_CNN_R, "CNN, random split"),
              ("cnn_s", C_CNN_S, "CNN, sector split"),
              ("lin", C_LIN, "Linear interp., sector split"),
              ("shift", C_SHIFT, "90$^\\circ$-shift copy (oracle)")]
    width = 0.20
    x = np.arange(len(labels))

    for ax, roi, title in ((axes[0], "row16", "(a) row 16 — the row that enters the sinogram"),
                           (axes[1], "all32", "(b) all 32 rows — training-time metric")):
        for i, (key, col, lab) in enumerate(series):
            vals = []
            for k in labels[:-1]:
                vals.append(get(d30, k, roi, key))
            vals.append(get(d30, "overall", roi, key))
            ax.bar(x + (i - 1.5) * width, vals, width * 0.9, color=col,
                   label=lab, linewidth=0, zorder=3)
        ax.set_yscale("log")
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.set_xlabel("Energy (MeV)")
        ax.grid(axis="y", which="major", color="#DDDDDD", linewidth=0.6, zorder=0)
        ax.grid(axis="y", which="minor", color="#EEEEEE", linewidth=0.4, zorder=0)
        ax.set_axisbelow(True)
        ax.set_title(title, fontsize=8.0, pad=4)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)

    axes[0].set_ylabel("Masked RMSE ($-\\log$ domain)")
    axes[0].set_ylim(2.5e-4, 1.1)
    axes[1].set_ylim(9e-3, 3.0)
    h, lb = axes[0].get_legend_handles_labels()
    fig.legend(h, lb, loc="lower center", ncol=4, frameon=False,
               bbox_to_anchor=(0.5, 0.135), handlelength=1.2, columnspacing=1.6)

    # 关键数字：随机 → 分块 的相对变化（30 epoch，Overall）
    r = get(d30, "overall", "all32", "cnn_r")
    s = get(d30, "overall", "all32", "cnn_s")
    sh = get(d30, "overall", "all32", "shift")
    txt = (f"CNN all-32 overall: random {r:.4f} $\\rightarrow$ sector {s:.4f} "
           f"({100*(s-r)/r:+.1f}%)\nsymmetry floor (90$^\\circ$ shift): {sh:.4f} "
           f"— {r/sh:.0f}$\\times$ below CNN")
    fig.text(0.5, 0.005, txt, ha="center", va="bottom", fontsize=6.8, color="#333333")

    for ax in axes:
        ax.set_xlim(-0.55, len(labels) - 0.45)
    fig.subplots_adjust(left=0.085, right=0.995, top=0.88, bottom=0.315, wspace=0.16)
    p = OUT / "fig_sector_split.png"
    fig.savefig(p, dpi=DPI)
    plt.close(fig)
    print(f"[OK] {p}")

    # 副产物：把随机→分块的差值落成 csv，README 里逐格可查
    rows = []
    for k in labels:
        kind = "overall" if k == "Overall" else k
        for roi in ("row16", "all32"):
            a = get(d30, kind, roi, "cnn_r")
            b = get(d30, kind, roi, "cnn_s")
            rows.append({"split": "30ep", "roi": roi, "group": k,
                         "cnn_random": f"{a:.6f}", "cnn_sector": f"{b:.6f}",
                         "rel_change_pct": f"{100*(b-a)/a:+.2f}",
                         "linear_sector": f"{get(d30, kind, roi, 'lin'):.6f}",
                         "shift90_sector": f"{get(d30, kind, roi, 'shift'):.6f}"})
            a5 = get(d5, kind, roi, "cnn_r")
            b5 = get(d5, kind, roi, "cnn_s")
            rows.append({"split": "5ep", "roi": roi, "group": k,
                         "cnn_random": f"{a5:.6f}", "cnn_sector": f"{b5:.6f}",
                         "rel_change_pct": f"{100*(b5-a5)/a5:+.2f}",
                         "linear_sector": f"{get(d5, kind, roi, 'lin'):.6f}",
                         "shift90_sector": f"{get(d5, kind, roi, 'shift'):.6f}"})
    with open(OUT / "fig_sector_split_data.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    return p


# ------------------------------------------------------------------ 图 2
def read_log(run_dir: Path) -> dict:
    """返回 {energy: [(epoch, rmse_valid), ...]}，只取 split=val。"""
    out: dict[str, list[tuple[int, float]]] = {}
    p = run_dir / "train_log.csv"
    with open(p, newline="", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            if r["split"] != "val":
                continue
            v = float(r["rmse_valid"])
            if not math.isfinite(v):
                continue
            out.setdefault(r["energy_MeV"], []).append((int(r["epoch"]), v))
    for k in out:
        out[k].sort()
    return out


def fig_train_curves() -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.0))
    cfg = [(axes[0], UNIF_30, "(a) Uniform phantom, 30 epochs", ENERGIES),
           (axes[1], FLAW_30, "(b) Defect phantom (3.0 MeV), 30 epochs", ("3",))]
    cols = {"1.5": "#0072B2", "3": "#009E73", "4.5": "#E69F00", "6": "#D55E00"}
    for ax, run_dir, title, ens in cfg:
        logs = read_log(run_dir)
        for e in ens:
            xs = [t[0] for t in logs[e]]
            ys = [t[1] for t in logs[e]]
            ax.plot(xs, ys, color=cols[e], linewidth=1.1, label=f"{e} MeV")
            ax.plot(xs[-1], ys[-1], "o", color=cols[e], markersize=3.0)
            ax.annotate(f"{ys[-1]:.4f}", (xs[-1], ys[-1]), textcoords="offset points",
                        xytext=(3, 2), fontsize=6.2, color=cols[e])
        # 5 epoch pilot 的终点（虚线），说明"pilot 只到哪"
        pilot = FLAW_5 if run_dir is FLAW_30 else UNIF_5
        pl = read_log(pilot)
        for e in ens:
            if e in pl and pl[e]:
                ax.axhline(pl[e][-1][1], color=cols[e], linewidth=0.7,
                           linestyle=(0, (3, 2)), alpha=0.55, zorder=1)
        ax.set_yscale("log")
        ax.set_xlabel("Epoch")
        ax.set_title(title, fontsize=8.0, pad=4)
        ax.grid(True, which="major", color="#DDDDDD", linewidth=0.6)
        ax.grid(True, which="minor", color="#F0F0F0", linewidth=0.4)
        ax.set_axisbelow(True)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    axes[0].set_ylabel("Validation masked RMSE\n($-\\log$ domain)")
    axes[0].set_ylim(0.20, 1.25)
    axes[1].set_ylim(0.32, 1.65)
    axes[0].set_xlim(-1.0, 33.0)
    axes[1].set_xlim(-1.0, 36.0)
    h, lb = axes[0].get_legend_handles_labels()
    fig.legend(h, lb, loc="upper center", ncol=4, frameon=False,
               bbox_to_anchor=(0.5, 0.995), handlelength=1.4, columnspacing=1.8,
               fontsize=6.8)
    fig.text(0.5, 0.005,
             "Dashed horizontal lines: end of the corresponding 5-epoch pilot.",
             ha="center", va="bottom", fontsize=6.6, color="#333333")
    fig.subplots_adjust(left=0.115, right=0.995, top=0.80, bottom=0.20, wspace=0.20)
    p = OUT / "fig_train_curves_30ep.png"
    fig.savefig(p, dpi=DPI)
    plt.close(fig)
    print(f"[OK] {p}")
    return p


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    fig_sector_split()
    fig_train_curves()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
