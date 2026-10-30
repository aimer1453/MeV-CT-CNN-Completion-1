#!/usr/bin/env python
# -*- coding: utf-8 -*-
r"""
scripts/plot_p02_span_oracle.py —— Fig. 13：角采样跨度扫描 + oracle 参照地板

R11-6 要求的三曲线折线图：
  x = 采样跨度 s（2/4/6/8°）
  y = 图像域 rmse_valid（对 dense 360 视角 FBP，FOV r≤600）
  曲线：Linear（实线）/ CNN（实线）/ Oracle floor（虚线）

**oracle 地板的定义（必须与图一起说清）**
把真实 dense 投影按 720/s 视角网格重新采样（s=4→180、s=6→120、s=8→90 视角，
即"补全后"网格），缺失位置**填真值**，走**同一套 hamming FBP** 重建。
s=2 时补全后网格 = 360 = 全测量，地板构造上恒为 0，**不画**。

**它不是下界（重要，必须写进图注）**：CNN 的 rmse 在 s=4/6/8 都**低于** oracle
（比值 0.99 / 0.95 / 0.92）。两个原因：
  1. 真值含泊松噪声，oracle 用的是**同一个含噪样本**，不是无噪声期望；
  2. 图像域 FOV 内无掩膜，而零计数像素在真值里是 5.50 的 −ln 填充离群值
     （1.5 MeV 占 17.4%），oracle 把这份噪声原样搬进重建。
自洽校验：oracle(s=4) == fbp_sparse(s=2) == 0.003039；oracle(s=8) == fbp_sparse(s=4) == 0.005277。

数据源
------
  runs/_reviewer_p0/sweep100_s{2,4,6,8}.json   （CNN 100 epoch 主结果）
  runs/_reviewer_p0/oracle_floor.json          （R12-1 已交付）

输出
----
  docs/figs/p02_step_sweep/angular_span_sweep_paper.{png,pdf}
  docs/figs/p02_step_sweep/angular_span_sweep_paper.csv
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "runs" / "_reviewer_p0"
FIG = ROOT / "docs" / "figs" / "p02_step_sweep"
DPI = 300
STEPS = (2, 4, 6, 8)
ENERGIES = ("1.5", "3", "4.5", "6")

COL = {"linear": "#0072B2", "cnn": "#D55E00", "oracle": "#6E6E6E"}
LBL = {"linear": "Linear interpolation (2 views)",
       "cnn": "CNN completion (100 epochs)",
       "oracle": "Oracle floor (true views, same FBP)"}
STYLE = {"oracle": (0, (4, 2))}
ECOL = {"1.5": "#E8B4AB", "3": "#AFC4DC", "4.5": "#9FCFC4", "6": "#C9C2E0"}

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 7.4,
    "axes.linewidth": 0.7,
    "axes.labelsize": 7.6,
    "xtick.labelsize": 6.8,
    "ytick.labelsize": 6.8,
    "legend.fontsize": 5.9,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "savefig.facecolor": "white",
    "mathtext.fontset": "dejavusans",
})


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="100", help="CNN 主结果的预算标签")
    ap.add_argument("--out", default="angular_span_sweep_paper")
    a = ap.parse_args()

    FIG.mkdir(parents=True, exist_ok=True)
    tag = a.tag
    cn: dict[str, dict] = {}
    lin: dict[str, dict] = {}
    for s in STEPS:
        d = json.loads((SRC / f"sweep{tag}_s{s}.json").read_text(encoding="utf-8"))
        cn[s] = d["image"]["cnn"]
        lin[s] = d["image"]["linear"]
    orc = json.loads((SRC / "oracle_floor.json").read_text(encoding="utf-8"))

    steps = [s for s in STEPS if s in cn]
    xs = np.arange(len(steps))
    orc_steps = [s for s in steps if str(s) in orc["per_step"]
                 and orc["per_step"][str(s)]["overall"]["rmse_valid"] > 0]

    fig, axes = plt.subplots(1, 2, figsize=(6.9, 2.75),
                             gridspec_kw={"width_ratios": [1.55, 1.0], "wspace": 0.30})

    # ================= 左栏：三条主曲线 =================
    ax = axes[0]
    # 背景：逐能量淡线
    for e in ENERGIES:
        yc = [cn[s]["per_energy"][e]["rmse_valid"] for s in steps]
        yl = [lin[s]["per_energy"][e]["rmse_valid"] for s in steps]
        ax.plot(xs, yl, lw=0.7, color=ECOL[e], alpha=0.85, zorder=2)
        ax.plot(xs, yc, lw=0.7, color=ECOL[e], alpha=0.85, zorder=2,
                ls=(0, (1.2, 1.2)))
    bg, = ax.plot([], [], lw=0.7, color="#B0B0B0",
                  label="Per-energy (1.5 / 3.0 / 4.5 / 6.0 MeV)")
    h_cnn, = ax.plot(xs, [cn[s]["overall"]["rmse_valid"] for s in steps],
                     marker="o", ms=3.6, lw=1.35, color=COL["cnn"],
                     label=LBL["cnn"], zorder=5)
    h_lin, = ax.plot(xs, [lin[s]["overall"]["rmse_valid"] for s in steps],
                     marker="s", ms=3.2, lw=1.15, color=COL["linear"],
                     label=LBL["linear"], zorder=4)
    h_orc = None
    if orc_steps:
        h_orc, = ax.plot([steps.index(s) for s in orc_steps],
                         [orc["per_step"][str(s)]["overall"]["rmse_valid"]
                          for s in orc_steps],
                         marker="^", ms=3.4, lw=1.15, color=COL["oracle"],
                         ls=STYLE["oracle"], label=LBL["oracle"], zorder=3)
        # 逐能量 oracle 淡线
        for e in ENERGIES:
            yo = [orc["per_step"][str(s)]["per_energy"][e]["rmse_valid"]
                  for s in orc_steps]
            ax.plot([steps.index(s) for s in orc_steps], yo, lw=0.7,
                    color="#C8C8C8", alpha=0.9, zorder=1)
    for i, s in enumerate(steps):
        ax.annotate(f"{cn[s]['overall']['rmse_valid']:.5f}",
                    (xs[i], cn[s]["overall"]["rmse_valid"]),
                    textcoords="offset points", xytext=(0, -14), fontsize=5.6,
                    color=COL["cnn"], ha="center")
        ax.annotate(f"{lin[s]['overall']['rmse_valid']:.5f}",
                    (xs[i], lin[s]["overall"]["rmse_valid"]),
                    textcoords="offset points", xytext=(0, 9), fontsize=5.6,
                    color=COL["linear"], ha="center")
    # 两条必要注记（短句，不是 caption 段落）
    ax.text(0.985, 0.045, "CNN below linear at every span\n"
                          "oracle floor is not a lower bound",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=5.6,
            color="#333333", linespacing=1.3)
    ax.set_yscale("log")
    ax.set_xticks(xs); ax.set_xticklabels([str(s) for s in steps])
    ax.set_xlabel("Sampling step $s$ (deg)")
    ax.set_ylabel("RMSE vs. dense reference (image domain)")
    ax.set_title("(a) Absolute error grows with $s$", fontsize=7.2, pad=4)
    ax.grid(color="#EDEDED", lw=0.5); ax.set_axisbelow(True)
    ax.margins(x=0.12, y=0.20)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.legend(loc="upper left", frameon=True, framealpha=0.95, borderpad=0.28,
              handletextpad=0.32, borderaxespad=0.32,
              handles=[h_lin, h_cnn] + ([h_orc, bg] if h_orc is not None else [bg]))

    # ================= 右栏：CNN / oracle 比值 =================
    ax = axes[1]
    ratios = []
    for s in orc_steps:
        f = orc["per_step"][str(s)]["overall"]["rmse_valid"]
        ratios.append(cn[s]["overall"]["rmse_valid"] / f)
    ax.axhline(1.0, color="#333333", lw=0.8, ls=(0, (4, 2)), zorder=2)
    ax.plot(range(len(orc_steps)), ratios, marker="o", ms=4.0, lw=1.25,
            color=COL["cnn"], zorder=4)
    for i, (s, r) in enumerate(zip(orc_steps, ratios)):
        ax.annotate(f"{r:.3f}", (i, r), textcoords="offset points",
                    xytext=(0, 8), fontsize=6.0, color=COL["cnn"], ha="center")
    ax.set_xticks(range(len(orc_steps)))
    ax.set_xticklabels([str(s) for s in orc_steps])
    ax.set_xlabel("Sampling step $s$ (deg)")
    ax.set_ylabel("CNN RMSE / oracle floor")
    ax.set_title("(b) CNN sits below the floor", fontsize=7.2, pad=4)
    ax.set_ylim(min(ratios) - 0.07, max(ratios) + 0.07)
    ax.grid(color="#EDEDED", lw=0.5); ax.set_axisbelow(True)
    ax.text(0.5, 0.05, "below 1 : CNN is more accurate than\n"
                       "the true-view oracle on the same grid",
            transform=ax.transAxes, ha="center", va="bottom", fontsize=5.5,
            color="#333333", linespacing=1.25)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)

    fig.subplots_adjust(left=0.085, right=0.985, top=0.878, bottom=0.175)

    # 仅输出 PNG（审稿要求 PNG only，禁止 PDF）
    out = []
    p = FIG / f"{a.out}.png"
    fig.savefig(p, dpi=DPI)
    out.append(p)
    plt.close(fig)

    with open(FIG / f"{a.out}.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["step_deg", "energy_MeV", "method", "rmse_valid",
                    "psnr_common", "ssim"])
        for s in steps:
            for e in ENERGIES:
                for m, dd in (("cnn", cn[s]["per_energy"][e]),
                              ("linear", lin[s]["per_energy"][e])):
                    w.writerow([s, e, m, f"{dd['rmse_valid']:.6f}",
                                f"{dd.get('psnr_common', ''):.4f}" if "psnr_common" in dd else "",
                                f"{dd['ssim']:.6f}"])
            for m, dd in (("cnn", cn[s]["overall"]), ("linear", lin[s]["overall"])):
                w.writerow([s, "overall", m, f"{dd['rmse_valid']:.6f}",
                            f"{dd.get('psnr_common', ''):.4f}" if "psnr_common" in dd else "",
                            f"{dd['ssim']:.6f}"])
            if s in orc_steps:
                w.writerow([s, "overall", "oracle_floor",
                            f"{orc['per_step'][str(s)]['overall']['rmse_valid']:.6f}",
                            f"{orc['per_step'][str(s)]['overall']['psnr_common']:.4f}",
                            f"{orc['per_step'][str(s)]['overall']['ssim']:.6f}"])
                for e in ENERGIES:
                    dd = orc["per_step"][str(s)]["per_energy"][e]
                    w.writerow([s, e, "oracle_floor", f"{dd['rmse_valid']:.6f}",
                                f"{dd.get('psnr_common', ''):.4f}" if "psnr_common" in dd else "",
                                f"{dd['ssim']:.6f}"])

    print(f"[CNN budget] {tag} epochs")
    print("  CNN/oracle floor: " + " ".join(f"s={s}:{r:.4f}" for s, r in zip(orc_steps, ratios)))
    print("  CNN 领先 linear: " + " ".join(
        f"s={s}:{(1-cn[s]['overall']['rmse_valid']/lin[s]['overall']['rmse_valid'])*100:.2f}%"
        for s in steps))
    for p in out:
        print(f"[OK] {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
