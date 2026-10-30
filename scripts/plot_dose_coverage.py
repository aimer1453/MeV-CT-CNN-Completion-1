#!/usr/bin/env python
"""P0-1 配套图：coverage vs 剂量因子 f（row 16 与全 32 行）。"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
RUNS = ROOT / "runs"


def main() -> int:
    src = RUNS / "_dose_coverage.json"
    if not src.is_file():
        raise SystemExit(f"[FATAL] 缺 {src}，先跑 scripts/dose_coverage.py")
    d = json.loads(src.read_text(encoding="utf-8"))
    fs = sorted(float(k) for k in d["factors"])
    a32 = np.array([d["factors"][f"{f:g}"]["_all"]["valid_coverage"] for f in fs]) * 100
    r16 = np.array([d["factors"][f"{f:g}"]["_all"]["valid_coverage_row16"] for f in fs]) * 100

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
    fig, ax = plt.subplots(figsize=(5.4, 3.9))

    ax.axhline(100, color=FG, ls=":", lw=0.9, alpha=0.35)
    ax.semilogx(fs, a32, "-o", color="#4ea1ff", lw=2.2, ms=7, mec="#111111", mew=0.6,
                label="all 32 detector rows")
    ax.semilogx(fs, r16, "-s", color="#ff5c5c", lw=2.2, ms=7, mec="#111111", mew=0.6,
                label="Row 16 (the row that is reconstructed)")
    fmax = max(fs)
    for x, y in zip(fs, a32):
        right = x == fmax
        ax.annotate(f"{y:.1f}%", (x, y), textcoords="offset points",
                    xytext=(-9, -6) if right else (0, -14),
                    ha="right" if right else "center",
                    fontsize=7.6, color="#4ea1ff")
    for x, y in zip(fs, r16):
        right = x == fmax
        ax.annotate(f"{y:.1f}%", (x, y), textcoords="offset points",
                    xytext=(-9, 9) if right else (0, 10),
                    ha="right" if right else "center",
                    fontsize=7.6, color="#ff5c5c")

    ax.set_xlabel("dose factor $f$  (fraction of the $2\\times10^{6}$ primaries per view)")
    ax.set_ylabel("valid coverage [%]")
    ax.set_ylim(0, 108)
    from matplotlib.ticker import NullFormatter
    ax.set_xticks(fs)
    ax.set_xticklabels([f"{f:g}" for f in fs])
    ax.xaxis.set_minor_formatter(NullFormatter())     # 关掉 log 轴的 2x10^-1 之类次刻度标签
    ax.set_xlim(min(fs) / 1.35, max(fs) * 1.25)
    ax.grid(alpha=0.16)
    ax.legend(fontsize=7.8, loc="lower left")
    ax.set_title("Valid coverage vs dose: the imaging row and the\n"
                 "detector-wide pool diverge below $f=1$", fontsize=9.5)

    out = ROOT / "docs" / "figs" / "p01_dose_coverage" / "fig_coverage_vs_f.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    for ext in (".png", ".pdf"):
        fig.savefig(out.with_suffix(ext), bbox_inches="tight",
                    facecolor=fig.get_facecolor())
    print(f"[fig] {out.relative_to(ROOT)} (+ .pdf)")

    # 顺带打印比值，供正文用
    for f, r, a in zip(fs, r16, a32):
        print(f"  f={f:<6g} row16 {r:6.2f}%   all32 {a:6.2f}%   ratio {r / a:5.2f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
