#!/usr/bin/env python
# -*- coding: utf-8 -*-
r"""
scripts/plot_cnr_multien_en.py —— Fig S6（缺陷空腔 CNR）换代版

为什么要有这个脚本
------------------
原 `fig11_cnr_en.{png,pdf}`（在 `plot_p01_dilution_en.py` 里）只有 **3.0 MeV 单档**
的五方法柱状图。缺陷数据换代到 `3_3_flaw_1_1`（带包壳）之后，Table 8 报的是
**四能量 × 3 种子**的 CNR，图与表对不上：读者在图上只看到 3.0 MeV 的 1.29，
看不到"1.5 → 3.0 上升、3.0 之后饱和回落"这个**换代后新出现的非单调形态**，
而这条形态恰恰是论文要改写的结论之一。

本图把 (a) 换成 **四能量分组柱状图**（五条重建路线 × 四档能量，Rose 判据线 5），
CNN 一串即 0.96 / 1.29 / 1.21 / 1.29；(b) 保留九根棒中心值（3.0 MeV，seed 20261022），
因为 Fig S6 caption 里"中心棒天生低于角棒"那句是在这一面板上查实并**反号**的。

数据来源（不重算任何东西，只读既有产物）
----------------------------------------
- `runs/_flaw11_multi_summary.json` 的 `per_energy[<E>]["cnr"][<method>]`（mean/std/values）
- `runs/<RUN_ID_3.0>_flaw_step3/image_metrics.json` 的 `defect_cnr.cnn_completed.rod_centres`

用法
----
  python scripts/plot_cnr_multien_en.py
  python scripts/plot_cnr_multien_en.py --summary runs/_flaw11_multi_summary.json \
      --rod-run 20261028_0022_cnn_bs8_lr1e-4_flaw11_3.0 \
      --out-dir docs/figs/p01_flaw_step3_indomain_100ep
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
sys_path = ROOT / "scripts"
import sys  # noqa: E402
for p in (ROOT, sys_path):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from data.csv_paths import resolve_csv  # noqa: E402,F401  （保持与同目录脚本同口径）

SUMMARY = ROOT / "runs" / "_flaw11_multi_summary.json"
ROD_RUN = "20261028_0022_cnn_bs8_lr1e-4_flaw11_3.0"
OUT_DIR = ROOT / "docs" / "figs" / "p01_flaw_step3_indomain_100ep"
DPI = 300

METHODS = [("dense", "dense ref."), ("cnn_completed", "CNN"),
           ("linear_interp", "linear"), ("fbp_sparse", "FBP"),
           ("sart_sparse", "SART")]
COLS = {"dense": "#8A8A8A", "cnn_completed": "#B4483C", "linear_interp": "#3A6EA5",
        "fbp_sparse": "#7FA8C9", "sart_sparse": "#C4A35A"}


def main() -> int:
    ap = argparse.ArgumentParser(description="Fig S6 缺陷 CNR 四能量分组柱状图")
    ap.add_argument("--summary", default=str(SUMMARY))
    ap.add_argument("--rod-run", default=ROD_RUN,
                    help="九根棒中心值取自哪个 run（默认 3.0 MeV seed 20261022）")
    ap.add_argument("--out-dir", default=str(OUT_DIR))
    a = ap.parse_args()

    summ = json.loads(Path(a.summary).read_text(encoding="utf-8"))
    per_e = summ["per_energy"]
    energies = sorted(per_e.keys(), key=float)
    if len(energies) < 2:
        raise SystemExit(f"[FAIL] 汇总里只有 {len(energies)} 档能量，画不了分组柱状图")

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.42),
                             gridspec_kw={"width_ratios": [1.35, 1.0]})

    # ---------------- (a) CNR vs energy，五条路线分组 ----------------
    ax = axes[0]
    n_m = len(METHODS)
    n_e = len(energies)
    group_w = 0.78
    bw = group_w / n_m
    xs = np.arange(n_e)
    for j, (key, lab) in enumerate(METHODS):
        off = (j - (n_m - 1) / 2.0) * bw
        means = [per_e[e]["cnr"][key]["mean"] for e in energies]
        sds = [per_e[e]["cnr"][key]["std"] for e in energies]
        ax.bar(xs + off, means, bw * 0.94, color=COLS[key], linewidth=0,
               label=lab, zorder=3)
        # 只给 CNN 画误差线（其余四路与种子无关，std ~1e-9，画出来是噪声）
        if key == "cnn_completed":
            ax.errorbar(xs + off, means, yerr=sds, fmt="none", ecolor="#333333",
                        elinewidth=0.7, capsize=2.0, zorder=4)
    ax.axhline(5.0, color="#333333", lw=0.8, ls=(0, (4, 2)), zorder=5)
    ax.text(0.985, 5.08, "Rose criterion (CNR = 5)", fontsize=5.6,
            ha="right", va="bottom", color="#333333",
            bbox=dict(boxstyle="round,pad=0.20", fc="white", ec="none"), zorder=6)
    # CNN 一串的数字标在柱顶（这就是正文要用的 0.96 / 1.29 / 1.21 / 1.29）
    j_cnn = [k for k, _ in METHODS].index("cnn_completed")
    off_cnn = (j_cnn - (n_m - 1) / 2.0) * bw
    for i, e in enumerate(energies):
        v = per_e[e]["cnr"]["cnn_completed"]["mean"]
        ax.text(xs[i] + off_cnn, v + 0.09, f"{v:.2f}", ha="center",
                fontsize=5.9, color="#B4483C", zorder=7)
    ax.set_xticks(xs)
    ax.set_xticklabels([f"{float(e):g}" for e in energies])
    ax.set_xlabel("Energy (MeV)")
    ax.set_ylim(0, 6.2)
    ax.set_ylabel("Cavity CNR")
    ax.set_title("(a) Cavity contrast-to-noise ratio, four energies",
                 fontsize=7.2, pad=3)
    ax.grid(axis="y", color="#EDEDED", lw=0.5)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    # legend 放在 2.6 以上的空白带（最高柱 SART 2.43），不会压柱；
    # 误差线说明挪到**轴外**底部——放在轴内任何角落都会压到 4.5/6.0 组的柱。
    ax.legend(loc="upper left", ncol=2, frameon=False, fontsize=5.6,
              handlelength=1.3, columnspacing=0.9, labelspacing=0.2)

    # ---------------- (b) 九根棒中心值 ----------------
    ax = axes[1]
    p = ROOT / "runs" / f"{a.rod_run}_flaw_step3" / "image_metrics.json"
    if not p.is_file():
        raise SystemExit(f"[FAIL] 缺少 {p}")
    tm = json.loads(p.read_text(encoding="utf-8"))
    rods = tm["defect_cnr"]["cnn_completed"]["rod_centres"]
    names = [r["rod"] for r in rods]
    mu = [r["mu_centre"] for r in rods]
    isd = [bool(r["is_defect_rod"]) for r in rods]
    xr = np.arange(len(rods))
    ax.bar(xr, mu, 0.62, color=["#B4483C" if d else "#9E9E9E" for d in isd],
           linewidth=0, zorder=3)
    for i, v in enumerate(mu):
        ax.text(i, v + 0.00012, f"{v:.4f}", ha="center", fontsize=5.6, rotation=90)
    ax.set_xticks(xr)
    ax.set_xticklabels(names, rotation=38, ha="right")
    ax.set_ylim(0, max(mu) * 1.32)
    ax.set_ylabel("Rod centre value ($\\mu$)")
    ax.set_title(f"(b) Centre value of the nine rods (CNN, {float(energies[1]):g} MeV)",
                 fontsize=7.2, pad=3)
    ax.grid(axis="y", color="#EDEDED", lw=0.5)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.text(0.985, 0.96, "red = rod containing the cavity", transform=ax.transAxes,
            fontsize=5.9, va="top", ha="right", color="#B4483C",
            bbox=dict(boxstyle="round,pad=0.20", fc="white", ec="none"))

    fig.text(0.30, 0.045,
             "error bars: SD over 3 seeds (CNN only; the other four routes "
             "are seed-invariant)",
             fontsize=5.1, ha="center", va="bottom", color="#555555")
    fig.subplots_adjust(left=0.075, right=0.988, top=0.885, bottom=0.32, wspace=0.30)
    out_dir = Path(a.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out = []
    for ext in ("png", "pdf"):
        q = out_dir / f"fig11_cnr_multien_en.{ext}"
        fig.savefig(q, dpi=DPI)
        out.append(q)
    plt.close(fig)
    for q in out:
        print(f"[OK] {q}")

    # 顺手把 (a) 面板的数字打出来，供 README / 交接文档引用
    print("\n[CNR 面板数据]")
    for key, lab in METHODS:
        row = "  %-14s" % lab + "  ".join(
            f"{float(e):g}: {per_e[e]['cnr'][key]['mean']:.2f}"
            + (f"±{per_e[e]['cnr'][key]['std']:.2f}"
               if per_e[e]["cnr"][key]["std"] > 1e-6 else "")
            for e in energies)
        print(row)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
