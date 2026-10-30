#!/usr/bin/env python
"""P0-1 · Fig 9 主面板 + 能量分解小多图（row16 与 all32 双口径）。

读 runs/_dose_eval_f{1,0.5,0.25,0.125}.json，画 RMSE vs 剂量分数 f（对数轴）。

主面板（Fig 9）：三条线 CNN / linear / cubic；CNN 带 3-种子误差线。
  - 线性/三次插值为确定性基线（与 CNN 种子无关），故无种子方差、只画实线。
  - CNN 标记 = 4 能量 × 3 种子均值（跨能量再取均值）；误差棒 = 4 能量上 3-种子
    std 的平均值（仅反映种子不确定性，不含能量间真实差异）。
补充图（Fig S）：4 能量各一子图，展示能量分辨行为（row16 与 all32 同页）。

输出：docs/figs/p01_dose_rmse/{fig_dose_rmse_row16.png, fig_dose_rmse_all32.png,
                              fig_dose_rmse_by_energy.png, README.md,
                              fig_dose_rmse_summary.json}
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter

import _fig_style as fs
fs.apply_rc()

ROOT = Path(__file__).resolve().parent.parent
RUNS = ROOT / "runs"
FIGDIR = ROOT / "docs/figs/p01_dose_rmse"
FACTORS = [1.0, 0.5, 0.25, 0.125]
ENERGIES = [1.5, 3.0, 4.5, 6.0]
FX = np.array(FACTORS)
COL = fs.METHOD_COLORS
LBL = fs.METHOD_LABELS


def load(factor: float) -> dict:
    p = RUNS / f"_dose_eval_f{factor:g}.json"
    if not p.is_file():
        raise SystemExit(f"[FATAL] 缺 {p}")
    return json.loads(p.read_text(encoding="utf-8"))


def cnn_run_ids(d: dict) -> list[str]:
    return [k for k in d["energies"]["1.5"]["recon_domain"].keys()
            if k not in ("linear_interp", "cubic_interp")]


def collect() -> dict:
    """factor -> energy -> {linear, cubic, cnn(np.array of 3 seeds),
                            linear_a32, cubic_a32, cnn_a32(np.array)}"""
    out = {}
    for f in FACTORS:
        d = load(f)
        rids = cnn_run_ids(d)
        per_e = {}
        for e in ENERGIES:
            ek = f"{e:g}"
            rd = d["energies"][ek]["recon_domain"]
            per_e[e] = dict(
                linear=rd["linear_interp"]["row16"],
                cubic=rd["cubic_interp"]["row16"],
                cnn=np.array([rd[r]["row16"] for r in rids]),
                linear_a32=rd["linear_interp"]["all32"],
                cubic_a32=rd["cubic_interp"]["all32"],
                cnn_a32=np.array([rd[r]["all32"] for r in rids]),
            )
        out[f] = per_e
    return out


def _collect_panel(data: dict, key: str):
    """返回某口径（row16 / all32）下三条曲线的绘图数据。

    key = "cnn" (Row 16) or "cnn_a32" (all 32 rows)；基线键随之取 linear/cubic
    或 linear_a32/cubic_a32。
    * CNN 点 = 4 能量 × 3 种子均值（跨能量再取均值）；种子 SD = 4 能量上 3-种子
      std 的均值（仅反映种子不确定性）。
    * 线性/三次插值确定性（与种子无关），只画实线均值。
    """
    base = "" if key == "cnn" else "_a32"
    pts_c, err_c, lin, cub = [], [], [], []
    for f in FACTORS:
        pe = data[f]
        cm = np.array([pe[e][key].mean() for e in ENERGIES])
        cs = np.array([pe[e][key].std(ddof=1) for e in ENERGIES])
        pts_c.append(cm.mean())
        err_c.append(cs.mean())
        lin.append(np.mean([pe[e]["linear" + base] for e in ENERGIES]))
        cub.append(np.mean([pe[e]["cubic" + base] for e in ENERGIES]))
    return (np.array(pts_c), np.array(err_c), np.array(lin), np.array(cub))


def _draw_panel(ax, pts_c, err_c, lin, cub, ylabel: str, with_f1_arrow: bool) -> None:
    ax.set_xscale("log")
    # CNN 带 3-种子 SD 作为填充带（替代误差棒，更清爽）；填充不单独入图例
    ax.fill_between(FX, pts_c - err_c, pts_c + err_c, color=COL["cnn"],
                    alpha=0.18, lw=0, zorder=2)
    ax.plot(FX, pts_c, "o-", color=COL["cnn"], lw=1.6, ms=5.0, zorder=4,
            label="CNN completion")
    ax.plot(FX, lin, "s--", color=COL["linear"], lw=1.6, ms=4.5, zorder=3,
            label=LBL["linear"])
    ax.plot(FX, cub, "^:", color=COL["cubic"], lw=1.6, ms=4.5, zorder=3,
            label=LBL["cubic"])
    ax.set_ylabel(ylabel)
    ax.grid(True, which="both", ls=":", alpha=0.4)
    ax.set_axisbelow(True)
    ax.set_xticks(FACTORS)
    ax.get_xaxis().set_major_formatter(plt.matplotlib.ticker.ScalarFormatter())
    ax.xaxis.set_minor_formatter(NullFormatter())
    # 数据派生标注（非硬编码）：CNN 点数值
    for x, y in zip(FX, pts_c):
        ax.annotate(f"{y:.4f}", (x, y), textcoords="offset points",
                    xytext=(0, 8), fontsize=6.0, ha="center", color=COL["cnn"])
    if with_f1_arrow:
        # f=1.0（全剂量基准）用从空白区指向数据点的箭头标出，文字放在曲线下方空白处
        i0 = list(FACTORS).index(1.0)
        ax.annotate("full dose  $f$=1.0", xy=(FX[i0], pts_c[i0]),
                    xytext=(0.42, 0.20), fontsize=6.2, color="#333333",
                    ha="center", va="center",
                    arrowprops=dict(arrowstyle="-|>", color="#333333", lw=0.7,
                                    shrinkA=1, shrinkB=3))


def _panel_two_stack(data: dict, fname: str) -> dict:
    """Fig 9 主图：上下两栏共享 x（剂量分数，对数轴），(a) Row 16、(b) all 32。

    图内不写图号/标题/caption —— 图号与题注由 LaTeX 侧承担，避免"烤字"。
    """
    p16 = _collect_panel(data, "cnn")
    p32 = _collect_panel(data, "cnn_a32")
    fig, axes = plt.subplots(2, 1, figsize=(6.8, 5.2), sharex=True,
                            gridspec_kw=dict(hspace=0.10),
                            constrained_layout=True)
    _draw_panel(axes[0], *p16, "centre-row (Row 16) RMSE in $-\\log$ domain",
                with_f1_arrow=True)
    _draw_panel(axes[1], *p32, "all-32-row RMSE in $-\\log$ domain",
                with_f1_arrow=False)
    axes[1].set_xlabel("dose fraction $f$  (log scale)")
    fs.panel_label(axes[0], "(a)")
    fs.panel_label(axes[1], "(b)")
    # 单一图例置于上栏右上空白处（该处无数据），仅 3 个方法项
    axes[0].legend(loc="upper right", fontsize=6.2, frameon=False, ncol=1)
    return fs.save_png(fig, FIGDIR / fname, dpi=300)


def _small_multiples(data: dict, key: str, fname: str) -> None:
    """SI 小多图：4 能量各一子图（图内同样不写图号/总标题）。"""
    fig, axes = plt.subplots(2, 2, figsize=(7.4, 5.6))
    for ax, e in zip(axes.ravel(), ENERGIES):
        cx = [data[f][e][key].mean() for f in FACTORS]
        ce = [data[f][e][key].std(ddof=1) for f in FACTORS]
        lx = [data[f][e]["linear" if key == "cnn" else "linear_a32"] for f in FACTORS]
        ux = [data[f][e]["cubic" if key == "cnn" else "cubic_a32"] for f in FACTORS]
        ax.set_xscale("log")
        ax.errorbar(FX, cx, yerr=ce, fmt="o-", color=COL["cnn"], label="CNN",
                    capsize=3, lw=1.4, ms=4)
        ax.plot(FX, lx, "s--", color=COL["linear"], label="linear", lw=1.4, ms=4)
        ax.plot(FX, ux, "^:", color=COL["cubic"], label="cubic", lw=1.4, ms=4)
        ax.set_title(f"{e:g} MeV", fontsize=7.2)
        ax.grid(True, which="both", ls=":", alpha=0.4)
        ax.set_axisbelow(True)
        ax.legend(frameon=False, fontsize=6.2)
        ax.set_xticks(FACTORS)
        ax.get_xaxis().set_major_formatter(plt.matplotlib.ticker.ScalarFormatter())
        ax.xaxis.set_minor_formatter(NullFormatter())
    fig.tight_layout()
    fs.save_png(fig, FIGDIR / fname, dpi=300)


def write_readme(summary: dict, data: dict) -> None:
    r16 = summary["row16"]
    a32 = summary["all32"]
    lines = ["# Fig 9 · 剂量曲线（P0-1）", ""]
    lines.append("## 1. 数据来源")
    lines.append("- 评测 JSON：`runs/_dose_eval_f{1,0.5,0.25,0.125}.json`"
                 "（`scripts/dose_eval.py` 产出）。")
    lines.append("- f=1.0 用原始规范档 CSV `sparse_targets.csv` + `config.json` + 规范三种子"
                 " `20261024_0400/1633/1805`；f<1 用泊松降采样数据"
                 " `sparse_targets_dose_f<f>.csv` + `config_dose_f<f>.json` + 各自三种子。")
    lines.append("- 每档：CNN（3 视角 U-Net）vs 线性插值 vs 三次插值，4 能量 × 3 种子。"
                 "确定性基线（线性/三次）与 CNN 种子无关。")
    lines.append("")
    lines.append("## 2. 逐图说明")
    lines.append("- `fig_dose_rmse.png`（Fig 9 主图，两栏共享 x）：上栏 (a) 中心行 Row 16 "
                 "RMSE vs f（对数轴），下栏 (b) 全 32 行探测器域 RMSE；三法对比，"
                 "CNN 带 3-种子 SD 填充带，f=1.0 用箭头标出。")
    lines.append("- `fig_dose_rmse_by_energy.png`（Fig S）：4 能量各一子图，能量分辨行为。")
    lines.append("")
    lines.append("## 3. 关键数字表")
    lines.append("| f | CNN row16 | linear row16 | CNN all32 | linear all32 |")
    lines.append("|---|---|---|---|---|")
    for i, f in enumerate(FACTORS):
        lines.append(f"| {f:g} | {r16['cnn_point'][i]:.4f}±{r16['cnn_seed_spread'][i]:.4f}"
                     f" | {r16['linear'][i]:.4f} | {a32['cnn_point'][i]:.4f}±{a32['cnn_seed_spread'][i]:.4f}"
                     f" | {a32['linear'][i]:.4f} |")
    lines.append("")
    lines.append("> CNN 误差棒 = 4 能量上 3-种子 std 的均值（仅种子不确定性）。"
                 "完整 improvement% 表见同目录 `improvement_table.md`。")
    lines.append("")
    lines.append("## 4. 物理背景分析")
    lines.append("- Row 16 是探测器竖直剖面的**峰顶**（`empty_proj` 中心行均值约 22.5、"
                 "边缘约 0.36，近 60×），计数充足；±1° 视角的线性插值本身已很准，"
                 "CNN 的相对收益被插值误差下限吃掉 → Row 16 上二者基本打平（甚至 CNN 略差）。")
    lines.append("- 峰顶以外的行计数近乎为零，`-log` 动态范围极大，线性插值无解；"
                 "训练先验可补出与邻行一致的形状 → 全 32 行口径 CNN 领先 2–3×。")
    lines.append("- 剂量越低，峰顶行优势越被压掉、外圈行问题越主导；两套口径差距会缩小"
                 "（这正是 P0-1 剂量曲线要量化的）。")
    lines.append("")
    lines.append("## 5. 口径与陷阱")
    lines.append("- Row 16 与 all32 都是 per-pixel mask RMSE，**分母不同，绝对值不可互比**；"
                 "只能各自比较方法间差异。")
    lines.append("- 线性/三次插值是确定性规则（与 checkpoint/种子无关），故无种子方差；"
                 "CNN 的误差棒才反映训练种子抖动。")
    lines.append("- f=1.0 与 f<1 数据来自不同剂量档（泊松降采样），Row 16 上 CNN 是否显著"
                 "优于线性，须看同档 paired 统计（§5.6 bootstrap），不能只看点估计。")
    lines.append("- 与 Table 5（方差加权 squared pool）和 §5.6 bootstrap（等权算术平均）"
                 "的池化差异见 `docs/handoff/algo_delivered.md` 第二十二轮 ③——两口径均正确，不可混用。")
    lines.append("")
    lines.append("## 6. 已知遗留")
    lines.append("- f<1 三档的逐视角 bootstrap 尚未重跑（仅 f=1.0 有 §5.6 置信区间）；"
                 "低剂量端的显著性需补 bootstrap。")
    lines.append("- 图像域 dose 曲线未在本图给出（P0-1 焦点在投影域两条口径），如需可补。")
    (FIGDIR / "README.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    FIGDIR.mkdir(parents=True, exist_ok=True)
    data = collect()
    # Fig 9 主图：上下两栏共享 x（Row 16 / all 32）；图内不烤图号/标题
    p = _panel_two_stack(data, "fig_dose_rmse.png")
    # SI：4 能量各一子图
    _small_multiples(data, "cnn", "fig_dose_rmse_by_energy.png")
    r16 = dict(cnn_point=list(_collect_panel(data, "cnn")[0]),
               cnn_seed_spread=list(_collect_panel(data, "cnn")[1]),
               linear=list(_collect_panel(data, "cnn")[2]),
               cubic=list(_collect_panel(data, "cnn")[3]))
    a32 = dict(cnn_point=list(_collect_panel(data, "cnn_a32")[0]),
               cnn_seed_spread=list(_collect_panel(data, "cnn_a32")[1]),
               linear=list(_collect_panel(data, "cnn_a32")[2]),
               cubic=list(_collect_panel(data, "cnn_a32")[3]))
    summary = {"row16": r16, "all32": a32,
               "factors": FACTORS, "energies": ENERGIES}
    (FIGDIR / "fig_dose_rmse_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8")
    write_readme(summary, data)
    print(f"[fig9] 主图 {p.name} + by_energy PNG + README 已写入 {FIGDIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
