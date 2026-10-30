#!/usr/bin/env python
# -*- coding: utf-8 -*-
r"""
scripts/plot_p02_budget.py —— 训练预算敏感性（为什么"row16 劣势"是欠训产物）

三栏：
  (a) 投影域 row16 RMSE：CNN 随预算下降，线性插值恒为常数 → 两条线相交
  (b) CNN/linear 比值：从 1.052 单调降到 0.977，穿过 1
  (c) 图像域 RMSE 的相对领先：2.6% → 4.6% → 6.1%

数据（全部实测，s=2°，同一 test 视角、同一口径）
------------------------------------------------
  30 ep  : run 20261023_2332  runs/_reviewer_p0/sweep_s2.json
  60 ep  : run 20261024_0323  runs/_reviewer_p0/sweep60_s2.json
  100 ep : run 20261024_0400  runs/_reviewer_p0/sweep100_s2.json

输出
----
  docs/figs/p02_budget/fig_budget_en.{png,pdf}
  docs/figs/p02_budget/fig_budget.csv
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "runs" / "_reviewer_p0"
FIG = ROOT / "docs" / "figs" / "p02_budget"
DPI = 300

BUDGETS = (30, 60, 100)
TAGS = {30: "", 60: "60", 100: "100"}

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 7.4,
    "axes.linewidth": 0.7,
    "axes.labelsize": 7.6,
    "xtick.labelsize": 6.8,
    "ytick.labelsize": 6.8,
    "legend.fontsize": 6.2,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "savefig.facecolor": "white",
    "mathtext.fontset": "dejavusans",
})


def main() -> int:
    FIG.mkdir(parents=True, exist_ok=True)
    rows = []
    for b in BUDGETS:
        d = json.loads((SRC / f"sweep{TAGS[b]}_s2.json").read_text(encoding="utf-8"))
        p = d["projection"]; im = d["image"]
        rows.append({
            "epochs": b,
            "row16_cnn": p["cnn"]["row16"]["overall"]["rmse_valid"],
            "row16_linear": p["linear"]["row16"]["overall"]["rmse_valid"],
            "all32_cnn": p["cnn"]["all32"]["overall"]["rmse_valid"],
            "all32_linear": p["linear"]["all32"]["overall"]["rmse_valid"],
            "img_cnn": im["cnn"]["overall"]["rmse_valid"],
            "img_linear": im["linear"]["overall"]["rmse_valid"],
            "ssim_cnn": im["cnn"]["overall"]["ssim"],
            "ssim_linear": im["linear"]["overall"]["ssim"],
            "run_id": d["cnn_run_id"],
        })

    xs = np.arange(len(BUDGETS))
    r16 = np.array([r["row16_cnn"] for r in rows])
    lin = np.array([r["row16_linear"] for r in rows])
    rat = r16 / lin
    gain = (1.0 - np.array([r["img_cnn"] for r in rows])
            / np.array([r["img_linear"] for r in rows])) * 100.0

    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.5),
                             gridspec_kw={"wspace": 0.36})

    ax = axes[0]
    ax.plot(xs, r16, marker="o", ms=4.0, lw=1.2, color="#B4483C", label="CNN", zorder=3)
    ax.plot(xs, lin, marker="s", ms=3.6, lw=1.0, color="#3A6EA5",
            label="Linear interpolation", zorder=3)
    for i in (0, len(BUDGETS) - 1):
        # 标签放两条线的外侧面：高的往上、低的往下，避免交点处交叉
        hi_cnn = r16[i] >= lin[i]
        for arr, c, hi in ((r16, "#B4483C", hi_cnn), (lin, "#3A6EA5", not hi_cnn)):
            dy = 8 if hi else -13
            h = "left" if i == 0 else "right"
            ax.annotate(f"{arr[i]:.4f}", (xs[i], arr[i]),
                        textcoords="offset points", xytext=(4 if i == 0 else -4, dy),
                        fontsize=5.4, color=c, ha=h)
    ax.set_xticks(xs); ax.set_xticklabels([str(b) for b in BUDGETS])
    ax.set_xlabel("Training epochs")
    ax.set_ylabel("RMSE$_{\\mathrm{row16}}$ (test views)")
    ax.set_title("(a) Projection domain, row 16", fontsize=7.2, pad=4)
    ax.legend(loc="upper right", frameon=True, framealpha=0.94, borderpad=0.3,
              handletextpad=0.35, borderaxespad=0.35)
    ax.grid(color="#EDEDED", lw=0.5); ax.set_axisbelow(True)
    ax.margins(x=0.16, y=0.17)
    for s_ in ("top", "right"):
        ax.spines[s_].set_visible(False)

    ax = axes[1]
    ax.axhline(1.0, color="#333333", lw=0.8, ls=(0, (4, 2)), zorder=2)
    ax.plot(xs, rat, marker="o", ms=4.0, lw=1.2, color="#B4483C", zorder=3)
    for i, v in enumerate(rat):
        ax.annotate(f"{v:.4f}", (xs[i], v), textcoords="offset points",
                    xytext=(0, 8), fontsize=5.6, color="#B4483C", ha="center")
    ax.text(0.5, 0.06, "$<$1 : CNN closer to truth", transform=ax.transAxes,
            ha="center", va="bottom", fontsize=5.6, color="#333333")
    ax.set_xticks(xs); ax.set_xticklabels([str(b) for b in BUDGETS])
    ax.set_xlabel("Training epochs")
    ax.set_ylabel("CNN / linear  (RMSE$_{\\mathrm{row16}}$)")
    ax.set_title("(b) Crosses unity between 60 and 100", fontsize=7.2, pad=4)
    ax.grid(color="#EDEDED", lw=0.5); ax.set_axisbelow(True)
    ax.margins(x=0.16, y=0.24)
    for s_ in ("top", "right"):
        ax.spines[s_].set_visible(False)

    ax = axes[2]
    ax.bar(xs, gain, 0.55, color="#B4483C", linewidth=0, zorder=3)
    for i, v in enumerate(gain):
        ax.text(i, v + 0.12, f"{v:.1f}%", ha="center", fontsize=6.2)
    ax.set_xticks(xs); ax.set_xticklabels([str(b) for b in BUDGETS])
    ax.set_xlabel("Training epochs")
    ax.set_ylabel("Image-domain RMSE lead (%)")
    ax.set_title("(c) Image domain, vs. linear", fontsize=7.2, pad=4)
    ax.set_ylim(0, max(gain) * 1.35)
    ax.grid(axis="y", color="#EDEDED", lw=0.5); ax.set_axisbelow(True)
    for s_ in ("top", "right"):
        ax.spines[s_].set_visible(False)

    fig.subplots_adjust(left=0.075, right=0.985, top=0.868, bottom=0.175)

    out = []
    for ext in ("png", "pdf"):
        p = FIG / f"fig_budget_en.{ext}"
        fig.savefig(p, dpi=DPI)
        out.append(p)
    plt.close(fig)

    with open(FIG / "fig_budget.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["epochs", "run_id", "row16_cnn", "row16_linear", "ratio",
                    "all32_cnn", "all32_linear", "img_cnn", "img_linear",
                    "img_lead_pct", "ssim_cnn", "ssim_linear"])
        for r, g in zip(rows, gain):
            w.writerow([r["epochs"], r["run_id"], f"{r['row16_cnn']:.6f}",
                        f"{r['row16_linear']:.6f}", f"{r['row16_cnn']/r['row16_linear']:.6f}",
                        f"{r['all32_cnn']:.6f}", f"{r['all32_linear']:.6f}",
                        f"{r['img_cnn']:.6f}", f"{r['img_linear']:.6f}", f"{g:.4f}",
                        f"{r['ssim_cnn']:.6f}", f"{r['ssim_linear']:.6f}"])

    print("  row16 CNN/linear: " + " ".join(f"{b}ep={v:.4f}" for b, v in zip(BUDGETS, rat)))
    print("  图像域领先: " + " ".join(f"{b}ep={g:.2f}%" for b, g in zip(BUDGETS, gain)))
    for p in out:
        print(f"[OK] {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
