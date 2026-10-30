#!/usr/bin/env python
"""P0-2 出图：CNN vs linear 的配对差与 bootstrap 95% CI（森林图）。"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
RUNS = ROOT / "runs"


def main() -> int:
    src = RUNS / "_p02_bootstrap.json"
    if not src.is_file():
        raise SystemExit(f"[FATAL] 缺 {src}，先跑 p02_bootstrap.py")
    d = json.loads(src.read_text(encoding="utf-8"))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "font.size": 9, "axes.linewidth": 0.8, "savefig.dpi": 220, "figure.dpi": 220,
        "text.color": "#e8e8e8", "axes.labelcolor": "#e8e8e8",
        "xtick.color": "#e8e8e8", "ytick.color": "#e8e8e8", "axes.edgecolor": "#e8e8e8",
        "figure.facecolor": "#1b1b1b", "axes.facecolor": "#1b1b1b",
        "legend.framealpha": 0.22, "legend.edgecolor": "#888888",
    })
    FG = "#e8e8e8"
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 3.9),
                             gridspec_kw={"width_ratios": [1.35, 1.0]})

    # (a) 逐能量 + 池化
    ax = axes[0]
    labs, mus, los, his, cols = [], [], [], [], []
    for ekey, v in d["row16"]["by_energy"].items():
        c = v["cnn_vs_linear"]
        labs.append(f"{ekey} MeV  (n={v['n_samples']})")
        mus.append(c["mean_diff"]); los.append(c["ci_lo"]); his.append(c["ci_hi"])
        cols.append("#ff5c5c" if c["mean_diff"] < 0 else "#4ea1ff")
    p = d["row16"]["pooled"]
    labs.append(f"pooled  (n={p['n']})")
    mus.append(p["mean_diff"]); los.append(p["ci_lo"]); his.append(p["ci_hi"])
    cols.append("#ffd24a")
    y = np.arange(len(labs))[::-1]
    ax.axvline(0, color=FG, lw=0.9, alpha=0.6)
    for yy, m, lo, hi, c in zip(y, mus, los, his, cols):
        ax.plot([lo, hi], [yy, yy], "-", color=c, lw=2.0, solid_capstyle="round")
        ax.plot([m], [yy], "o", color=c, ms=7, mec="#111111", mew=0.6)
    ax.set_yticks(y); ax.set_yticklabels(labs, fontsize=8)
    ax.set_xlabel(r"$\Delta$row16 RMSE = CNN $-$ linear   (paired, 10 000 bootstraps)")
    ax.set_title("(a) projection domain, centre row 16\n"
                 r"left of 0 = CNN better", fontsize=9.5)
    ax.grid(axis="x", alpha=0.18)
    ax.set_xlim(min(los) * 1.25 - abs(min(los)) * 0.05, max(his) * 1.35)

    # (b) 图像域（单元 = 能量×种子）
    ax = axes[1]
    c = d["image"]["cnn_vs_linear"]
    ax.axvline(0, color=FG, lw=0.9, alpha=0.6)
    ax.plot([c["ci_lo"], c["ci_hi"]], [1, 1], "-", color="#ffd24a", lw=2.6,
            solid_capstyle="round")
    ax.plot([c["mean_diff"]], [1], "o", color="#ffd24a", ms=9, mec="#111111", mew=0.6)
    units = [u["cnn"] - u["linear"] for u in d["image"].get("per_unit", [])]
    rng = np.random.default_rng(0)
    ax.plot(units, 0.0 + rng.uniform(-0.07, 0.07, len(units)), "|",
            color="#4ea1ff", ms=10, alpha=0.9, mew=1.4)
    ax.set_yticks([1, 0])
    ax.set_yticklabels([f"pooled\n(n={d['image']['n_units']})",
                        f"single units\n({len(units)})"], fontsize=8)
    ax.set_ylim(-0.45, 1.5)
    ax.set_xlabel(r"$\Delta$image RMSE = CNN $-$ linear")
    ax.set_title("(b) image domain (unit = energy $\\times$ seed)\n"
                 "left of 0 = CNN better", fontsize=9.5)
    ax.grid(axis="x", alpha=0.18)

    txt = (f"row16: {p['rel_cnn_vs_linear_pct']:+.2f} %  "
           f"$d_z$={p['cohen_dz']:+.2f} [{p['cohen_dz_ci'][0]:+.2f},{p['cohen_dz_ci'][1]:+.2f}]\n"
           f"image: {d['image']['rel_cnn_vs_linear_pct']:+.2f} %  "
           f"$d_z$={c['cohen_dz']:+.2f}")
    fig.text(0.5, -0.02, txt, ha="center", fontsize=8.2, color=FG)

    fig.tight_layout()
    out = ROOT / "docs" / "figs" / "p02_bootstrap" / "fig_bootstrap_ci.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    for ext in (".png", ".pdf"):
        fig.savefig(out.with_suffix(ext), bbox_inches="tight",
                    facecolor=fig.get_facecolor())
    print(f"[fig] {out.relative_to(ROOT)} (+ .pdf)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
