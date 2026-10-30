#!/usr/bin/env python
# -*- coding: utf-8 -*-
r"""
scripts/plot_norm_stats.py —— 生成 3 张 norm/coverage 对比图（灰阶，论文可用）
================================================================================
输入：docs/norm_stats_proposal.json（由 scripts/compute_norm_stats.py 生成）
输出（统一放 docs/figs/norm_stats/ 子目录）：
    docs/figs/norm_stats/norm_distribution.png      -log 分布（分能量 + 全局均值/±std + ZERO_LOG_FILL）
    docs/figs/norm_stats/coverage_comparison.png    两口径 coverage 对比
    docs/figs/norm_stats/norm_input_vs_target.png   输入并集 vs target 的 mean/std 对比 + Δ 阈值判定

约定：
  - 只用 matplotlib，不依赖 seaborn，不做主题美化
  - 灰阶 + 线型/填充纹理区分，便于黑白打印
  - 每张图有标题、轴标签、单位、图例、数值标注
  - 图放 docs/figs/norm_stats/ 以便入库（runs/**/*.png 被 .gitignore 忽略）

用法：
  conda run -n ct_full python scripts/plot_norm_stats.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import rcParams

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "docs" / "norm_stats_proposal.json"
FIGD = ROOT / "docs" / "figs" / "norm_stats"

ENERGIES = ("1.5", "3", "4.5", "6")
GRAYS = ("#000000", "#555555", "#888888", "#bbbbbb")
STYLES = ("-", "--", "-.", ":")

#: 双语输出：zh -> _analysis.png（自己看），en -> _paper.png（论文用）
LANGS: tuple[str, ...] = ("zh", "en")
SUFFIX: dict[str, str] = {"zh": "analysis", "en": "paper"}
FONTS: dict[str, list[str]] = {
    "zh": ["Microsoft YaHei", "SimHei", "DejaVu Sans"],
    "en": ["DejaVu Sans", "Arial"],          # 英文版禁止挂中文字体，否则会回退
}

#: 图上文案。中英一一对应，中文标题 <= 30 字符，英文标题 <= 60 字符。
T: dict[str, dict[str, str]] = {
    "zh": {
        "title_dist": "排除无效像素后的 -log 分布",
        "title_cov": "覆盖率双口径对比",
        "title_cmp": "输入文件并集 vs target 的 -log 统计对比",
        "x_log": "-log 值（无量纲）",
        "y_density": "概率密度（1/无量纲单位）",
        "x_energy": "能量档 (MeV)",
        "y_value": "数值（-log 值）",
        "y_coverage": "覆盖率（0-1，无量纲）",
        "sub_mean": "均值",
        "sub_std": "标准差",
        "leg_global_mean": "全局 mean = {v:.4f}",
        "leg_global_band": "全局 {lab} = {v:.4f}",
        "band_lo": "mean - std",
        "band_hi": "mean + std",
        "leg_off": "（在横轴范围外）",
        "leg_zero": "ZERO_LOG_FILL = {v:.2f}",
        "leg_cov": "coverage（资产口径）",
        "leg_vcov": "valid_coverage（评估口径）",
        "leg_target": "target（1080 文件）",
        "leg_union": "输入文件并集（{n} 文件去重）",
        "ok": "OK",
        "exceeds": "超阈",
        "cap_dist": "n = 1080 target（train）| 有效像素 {nv:,} / {nt:,} | "
                    "mean = {mean:.4f}，std = {std:.4f}，zero_log_fill = {zf:.2f} | "
                    "横轴截断至 [0, 8]：<0 共 {below:,}，>8 共 {above:,}",
        "cap_cov": "n = 1080 target（train，逐样本均值）| "
                   "coverage = I_target > 0 / 总像素 | "
                   "valid_coverage = (I_target > 0) & (empty_proj > 0) / 总像素",
        "cap_cmp": "target = 1080 个 train target 文件 | 输入文件并集 = {n} 个去重后的 "
                   "ch_prev/ch0/ch_next 文件 | 阈值：|Δmean| > {thm} 或 |Δstd|/std > {ths:.0%} | "
                   "{verdict}",
        "verdict_block": "存在超阈项",
        "verdict_ok": "全部在阈值内",
    },
    "en": {
        "title_dist": "Distribution of -log Values by Energy",
        "title_cov": "Coverage vs Valid Coverage",
        "title_cmp": "Input Union vs Target Statistics",
        "x_log": "-log Value",
        "y_density": "Probability Density",
        "x_energy": "Energy (MeV)",
        "y_value": "Value (-log)",
        "y_coverage": "Coverage (0-1)",
        "sub_mean": "Mean",
        "sub_std": "Standard Deviation",
        "leg_global_mean": "Global Mean = {v:.4f}",
        "leg_global_band": "Global {lab} = {v:.4f}",
        "band_lo": "Mean - Std",
        "band_hi": "Mean + Std",
        "leg_off": " (off-axis)",
        "leg_zero": "Zero-Fill Value = {v:.2f}",
        "leg_cov": "Coverage (asset scope)",
        "leg_vcov": "Valid Coverage (eval scope)",
        "leg_target": "Target",
        "leg_union": "Input Union",
        "ok": "OK",
        "exceeds": "exceeds",
        "cap_dist": "n = 1080 targets (train) | valid pixels = {nv:,} / {nt:,} | "
                    "mean = {mean:.4f}, std = {std:.4f}, zero_log_fill = {zf:.2f} | "
                    "x clipped to [0, 8]: {below:,} below 0, {above:,} above 8",
        "cap_cov": "n = 1080 targets (train, per-sample mean) | "
                   "coverage = I_target > 0 / total pixels | "
                   "valid_coverage = (I_target > 0) & (empty_proj > 0) / total pixels",
        "cap_cmp": "target = 1080 train target files | input_union = {n} deduplicated "
                   "ch_prev/ch0/ch_next files | threshold: |delta mean| > {thm} or "
                   "|delta std| / std > {ths:.0%} | {verdict}",
        "verdict_block": "BLOCK: threshold exceeded",
        "verdict_ok": "all within threshold",
    },
}



def setup_style(lang: str) -> None:
    """按语言切换字体。英文版只挂 DejaVu/Arial，杜绝中文回退。"""
    if lang not in LANGS:
        raise ValueError(f"未知 lang: {lang!r}，应为 {LANGS}")
    rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": FONTS[lang],
        "axes.unicode_minus": False,
        "figure.dpi": 110,
        "savefig.dpi": 200,
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "grid.linestyle": ":",
        "axes.edgecolor": "black",
    })


def caption(fig, text: str) -> None:
    """元数据统一放图底部单行小字（方案 A）：灰色、字号 7、无边框。"""
    fig.text(0.5, -0.02, text, ha="center", va="top", fontsize=7, color="#555555")



def fig_norm_distribution(d: dict, lang: str) -> Path:
    t = T[lang]
    g = d["global"]
    gv = g["valid"]
    fig, ax = plt.subplots(figsize=(8.6, 5.4))

    for i, e in enumerate(ENERGIES):
        h = d["per_energy"][e]["hist"]
        edges = np.asarray(h["bin_edges"])
        counts = np.asarray(h["counts"], dtype=float)
        n_valid = d["per_energy"][e]["n_valid"]
        bw = edges[1] - edges[0]
        dens = counts / (n_valid * bw)              # histogram density，不是 count
        centers = (edges[:-1] + edges[1:]) / 2
        ax.plot(centers, dens, color=GRAYS[i], ls=STYLES[i], lw=1.5, label=f"{e} MeV")

    # 三条统计线统一用点划线（-.）竖线形态；靠 lw 与图例文案区分彼此
    ax.axvline(gv["mean"], color="black", lw=1.4, ls="-.",
               label=t["leg_global_mean"].format(v=gv["mean"]))
    for k, lab in ((-1, t["band_lo"]), (1, t["band_hi"])):
        x = gv["mean"] + k * gv["std"]
        off = x < 0 or x > 8
        ax.axvline(x, color="black", lw=1.0, ls="-.",
                   label=t["leg_global_band"].format(lab=lab, v=x)
                   + (t["leg_off"] if off else ""))
    ax.axvline(g["zero_log_fill"], color="black", lw=1.6, ls="-.",
               label=t["leg_zero"].format(v=g["zero_log_fill"]))

    ax.set_xlim(0, 8)
    ax.set_xlabel(t["x_log"])
    ax.set_ylabel(t["y_density"])
    ax.set_title(t["title_dist"], fontweight="bold")
    ax.legend(fontsize=7.2, loc="upper right", framealpha=0.92)

    over = sum(d["per_energy"][e]["hist"]["above"] for e in ENERGIES)
    below = sum(d["per_energy"][e]["hist"]["below"] for e in ENERGIES)
    caption(fig, t["cap_dist"].format(
        nv=g["n_valid"], nt=g["n_total"], mean=gv["mean"], std=gv["std"],
        zf=g["zero_log_fill"], below=below, above=over))

    out = FIGD / f"norm_distribution_{SUFFIX[lang]}.png"
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


def fig_coverage(d: dict, lang: str) -> Path:
    t = T[lang]
    labels = list(ENERGIES) + ["ALL"]
    cov = [(d["per_energy"][e] if e != "ALL" else d["global"])["coverage"] for e in labels]
    vcov = [(d["per_energy"][e] if e != "ALL" else d["global"])["valid_coverage"] for e in labels]

    x = np.arange(len(labels))
    w = 0.36
    fig, ax = plt.subplots(figsize=(8.2, 5.0))
    b1 = ax.bar(x - w / 2, cov, w, color="#3a3a3a", edgecolor="black", lw=0.8,
                label=t["leg_cov"])
    b2 = ax.bar(x + w / 2, vcov, w, color="white", edgecolor="black", lw=0.8, hatch="///",
                label=t["leg_vcov"])

    for rect, v in list(zip(b1, cov)) + list(zip(b2, vcov)):
        ax.text(rect.get_x() + rect.get_width() / 2, v + 0.012, f"{v:.4f}",
                ha="center", va="bottom", fontsize=8)

    ax.set_ylim(0, 1.30)
    ax.set_xticks(x, [f"{l} MeV" if l != "ALL" else "ALL" for l in labels])
    ax.set_xlabel(t["x_energy"])
    ax.set_ylabel(t["y_coverage"])
    ax.set_title(t["title_cov"], fontweight="bold")
    ax.legend(fontsize=7.8, loc="upper left", framealpha=0.95)
    caption(fig, t["cap_cov"])

    out = FIGD / f"coverage_comparison_{SUFFIX[lang]}.png"
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


def fig_input_vs_target(d: dict, lang: str) -> Path:
    t = T[lang]
    labels = list(ENERGIES) + ["ALL"]
    c = d["comparison"]
    tgt_m = [c[e]["target_mean"] for e in labels]
    in_m = [c[e]["input_mean"] for e in labels]
    tgt_s = [c[e]["target_std"] for e in labels]
    in_s = [c[e]["input_std"] for e in labels]
    th_m = d["meta"]["thresholds"]["delta_mean"]
    th_s = d["meta"]["thresholds"]["delta_std_rel"]
    n_union = d["input_union"]["ALL"]["n_files"]

    x = np.arange(len(labels))
    w = 0.36
    fig, axes = plt.subplots(1, 2, figsize=(12.0, 5.6))

    for ax, tm, im, title in (
        (axes[0], tgt_m, in_m, t["sub_mean"]),
        (axes[1], tgt_s, in_s, t["sub_std"]),
    ):
        b1 = ax.bar(x - w / 2, tm, w, color="#3a3a3a", edgecolor="black", lw=0.8,
                    label=t["leg_target"])
        b2 = ax.bar(x + w / 2, im, w, color="white", edgecolor="black", lw=0.8, hatch="///",
                    label=t["leg_union"].format(n=n_union))
        top = max(max(tm), max(im))
        ax.set_ylim(0, top * 1.55)          # 留出数值标注与图例的头部空间
        for rect, v in list(zip(b1, tm)) + list(zip(b2, im)):
            ax.text(rect.get_x() + rect.get_width() / 2, v, f"{v:.4f}",
                    ha="center", va="bottom", fontsize=7.0)
        ax.set_xticks(x, [l if l != "ALL" else "ALL" for l in labels])
        ax.set_xlabel(t["x_energy"], labelpad=6)
        ax.set_ylabel(t["y_value"])
        ax.set_title(title, fontweight="bold")
        ax.legend(fontsize=7.6, loc="upper left", framealpha=0.95)

    # 每个能量档的 Δ 与阈值判定（超阈用红字）
    for i, e in enumerate(labels):
        dm = c[e]["delta_mean"]
        ds = c[e]["delta_std_rel"]
        axes[0].text(x[i], -0.155,
                     f"Δmean={dm:+.4f}\n{t['exceeds'] if abs(dm) > th_m else t['ok']}",
                     ha="center", va="top", fontsize=7.2,
                     color="red" if abs(dm) > th_m else "#333333",
                     transform=axes[0].get_xaxis_transform())
        axes[1].text(x[i], -0.155,
                     f"Δstd={ds:+.2%}\n{t['exceeds'] if abs(ds) > th_s else t['ok']}",
                     ha="center", va="top", fontsize=7.2,
                     color="red" if abs(ds) > th_s else "#333333",
                     transform=axes[1].get_xaxis_transform())

    blocked = d["blocked"]
    fig.suptitle(t["title_cmp"], fontweight="bold", fontsize=11.5, y=1.02)
    caption(fig, t["cap_cmp"].format(
        n=n_union, thm=th_m, ths=th_s,
        verdict=t["verdict_block"] if blocked else t["verdict_ok"]))

    out = FIGD / f"norm_input_vs_target_{SUFFIX[lang]}.png"
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="norm/coverage 三张图（双语双份输出）")
    ap.add_argument("--lang", choices=LANGS, default=None,
                    help="只出某一语言；缺省两种都出（analysis=zh, paper=en）")
    args = ap.parse_args(argv)

    langs = (args.lang,) if args.lang else LANGS
    d = json.loads(SRC.read_text(encoding="utf-8"))
    FIGD.mkdir(parents=True, exist_ok=True)
    outs = []
    for lang in langs:
        setup_style(lang)
        outs += [fig_norm_distribution(d, lang), fig_coverage(d, lang),
                 fig_input_vs_target(d, lang)]
    for p in outs:
        print(f"已生成 {p}  ({p.stat().st_size/1024:.1f} KB)")
    print(f"共 {len(outs)} 个文件（{len(langs)} 种语言 × 3 张图）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
