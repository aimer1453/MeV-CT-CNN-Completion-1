#!/usr/bin/env python
# -*- coding: utf-8 -*-
r"""
scripts/plot_regime_joint.py —— 二维工作点的联合展示图（180 mm × 80 mm，三栏）

回答的问题
----------
"补全什么时候该用、什么时候不该用"不是一维问题，而是 **(域, 跨度)** 二维的：
同一套 CNN，在周期阵列上 2° 就略胜插值，在非周期（缺陷）阵列上要拉到 6.9° 才追平。

三栏
----
(a) 比值图：CNN / linear 的 row-16 RMSE 比值，均匀域 vs 缺陷域，y=1 为打平线；
    ★ 标缺陷域的穿越点 $s^*$，双箭头标两域的"追平跨度差" $\Delta s$。
(b) 归一化退化率：每个方法除以它自己在 2° 的 RMSE，把两个域放到同一无量纲坐标。
    读法：线越陡 = 该估计量对视角抽稀越不耐受。缺陷域的 linear 最陡。
(c) 离散制度格：2 行（域）× 6 列（跨度），格子填 $1-\text{ratio}$（>0 = 补全更优，
    橙；<0 = 插值更优，蓝），零等值线即"该用/不该用"的边界。

口径（必须与正文一致）
----------------------
* 两个域都是 **45 held-out test 视角 / masked / row-16 / 3 种子（ddof=0）**。
* 缺陷域 = 边棒 `3_3_flaw_1_2`，只有 3.0 MeV 一档。
* 均匀域 = `3_3` 主域，**必须取 3.0 MeV 单档切片**（`per_energy["3"]`），
  **不能用 `overall` 四能量池化** —— 缺陷域没有池化可言，池化 vs 切片是同图不同物。
  池化会造出"均匀域 2° 就赢、全程无穿越"的假象，成因是 1.5 MeV 只占 25% 像素
  却占池化 MSE 的 66.6%（RMSE 0.202 vs 其余 0.073–0.094，平方后压倒），
  而 1.5 MeV 恰是 CNN 唯一赢的档。切片后均匀域 **6.14° 穿越**，与缺陷域 6.92° 同量级。
* 真源：`runs/_flaw12_span_summary.json`（缺陷域）、
  `runs/_reviewer_p0/sweep100_3seed_s*.json`（均匀域，读 `per_energy["3"]`）；
  均匀域缺档时才回退 `runs/_r34_span_sweep.json`（该文件只有池化，回退时会在图上告警）。
* **禁止**与 360 待补全视角的跨能量表横比。

关于 $\Delta s$
---------------
$\Delta s$ = 两个域各自穿越 1.0 的跨度之差（均匀 6.14° → 缺陷 6.92°）。

**图上只写 "< 1°"，不写 0.78°**：两个穿越点都是 2° 网格上的相邻点线性内插，
缺陷域三种子各自的 $s^*$ 为 6.78 / 6.90 / 7.04°（极差 0.26°），且 0.78° 只相当于
0.39 个列间距，印出两位小数会让读者以为精确到 0.01°。精确值由 CSV 与图注承载，
图注同时声明"位移远小于池化暗示的 5°，且小于 1°"。

用法
----
    python scripts/plot_regime_joint.py
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm, to_hex
from matplotlib.patches import FancyArrowPatch

ROOT = Path(__file__).resolve().parent.parent
RUNS = ROOT / "runs"
FIG = ROOT / "docs" / "figs" / "p01_regime_joint"
DPI = 300
STEPS = (2, 4, 6, 8, 10, 12)

COL = {"linear": "#0072B2", "cubic": "#009E73", "cnn": "#D55E00"}
LBL = {"linear": "Linear", "cubic": "Cubic", "cnn": "CNN"}
DOMCOL = {"uniform": "#0072B2", "defect": "#D55E00"}

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 6.6,
    "axes.linewidth": 0.6,
    "axes.labelsize": 6.9,
    "xtick.labelsize": 6.2,
    "ytick.labelsize": 6.2,
    "legend.fontsize": 5.6,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "savefig.facecolor": "white",
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "mathtext.fontset": "dejavusans",
})


def load_defect() -> dict[int, dict]:
    p = RUNS / "_flaw12_span_summary.json"
    if not p.is_file():
        return {}
    j = json.loads(p.read_text(encoding="utf-8"))
    out: dict[int, dict] = {}
    for k, a in j.get("per_step", {}).items():
        out[int(k)] = {
            "cnn": a["row16_cnn"]["mean"],
            "linear": a["row16_linear"]["mean"],
            "cubic": a["row16_cubic"]["mean"],
            "ratio": a["row16_ratio"]["mean"],
            "ratio_sd": a["row16_ratio"]["sd"],
        }
    return out


ENERGY = "3"          # 两域必须同能量：缺陷域只有 3.0 MeV，均匀域取同档切片


def load_uniform() -> dict[int, dict]:
    """均匀域 row-16 的 **3.0 MeV 切片**（不是四能量池化）。

    为什么必须切片：缺陷域（边棒 `3_3_flaw_1_2`）只有 3.0 MeV 一档，拿四能量
    池化曲线去比是同图不同物。更关键的是池化会被 1.5 MeV 支配 —— s=2° 时
    1.5 MeV 只占 25% 像素却占池化 MSE 的 66.6%（RMSE 0.202 vs 其余 0.073–0.094，
    平方后压倒），而 1.5 MeV 恰好是 CNN 唯一赢的档（0.9503），其余三档全输
    （1.0658/1.0831/1.0920）。于是池化曲线在 2° 显示"CNN 已赢"、全程无交叉，
    切片曲线在 6.14° 交叉 —— 前者是 1.5 MeV 单档效应，不是可比的结论。

    读 `runs/_reviewer_p0/sweep100_3seed_s*.json` 的
    `projection.<m>.row16.per_energy["3"].rmse_valid`（3 种子汇总，6 档齐全）。
    """
    out: dict[int, dict] = {}
    for p in sorted((ROOT / "runs" / "_reviewer_p0").glob("sweep100_3seed_s*.json")):
        s = int(p.stem.split("_s")[-1])
        j = json.loads(p.read_text(encoding="utf-8"))
        proj = j["projection"]
        out[s] = {m: proj[m]["row16"]["per_energy"][ENERGY]["rmse_valid"]
                  for m in ("cnn", "linear", "cubic")}
    if not out:
        raise SystemExit("缺 runs/_reviewer_p0/sweep100_3seed_s*.json（3 种子汇总）")
    return out


def ratio_of(d: dict) -> float:
    """比值一律现算 cnn/linear，不依赖数据源里是否带 ratio 字段。"""
    return d["cnn"] / d["linear"]


def cross_span(dom: dict[int, dict]) -> float | None:
    """比值穿越 1.0 的跨度（相邻两点线性内插）；全程同侧则返回 None。"""
    ss = sorted(s for s in STEPS if s in dom)
    for a, b in zip(ss, ss[1:]):
        ra, rb = ratio_of(dom[a]), ratio_of(dom[b])
        if (ra - 1.0) * (rb - 1.0) < 0:
            return a + (b - a) * (ra - 1.0) / (ra - rb)
    return None


def main() -> int:
    defe = load_defect()
    uni = load_uniform()
    if not defe:
        raise SystemExit("缺 runs/_flaw12_span_summary.json")
    FIG.mkdir(parents=True, exist_ok=True)

    fig = plt.figure(figsize=(7.09, 3.15))          # 180 mm × 80 mm
    gs = fig.add_gridspec(1, 3, width_ratios=[1.02, 1.0, 1.22], wspace=0.34,
                          left=0.072, right=0.905, top=0.885, bottom=0.185)

    # ================= (a) 比值图 =================
    ax = fig.add_subplot(gs[0, 0])
    ss_u = sorted(s for s in STEPS if s in uni)
    ss_d = sorted(s for s in STEPS if s in defe)
    ax.axhline(1.0, color="#555555", lw=0.8, zorder=1)
    if ss_u:
        ax.plot(ss_u, [ratio_of(uni[s]) for s in ss_u], "-s", color=DOMCOL["uniform"],
                lw=1.0, ms=3.0, mfc="white", mew=0.9, zorder=3, label="Uniform array")
    ax.plot(ss_d, [defe[s]["ratio"] for s in ss_d], "-o", color=DOMCOL["defect"],
            lw=1.35, ms=3.4, zorder=4, label="Defect (edge rod)")

    s_d = cross_span(defe)
    s_u = cross_span(uni) if ss_u else None
    if s_d is not None:
        ax.axvline(s_d, color="#666666", lw=0.65, ls=(0, (2.2, 1.8)), zorder=2)
        ax.annotate("$\\bigstar$", (s_d, 1.0), fontsize=8.4, color="#1A1A1A",
                    ha="center", va="center", zorder=7)
        # 注：缺陷域穿越点的精确值由 CSV 与图注承载，图上只放 ★ 标记，
        # 不写 "6.9°" 之类会误导为池化暗示精度的数字（见 §53R 红线）。
    if s_u is not None:
        # 均匀域的穿越点：同一条 y=1 线上，两者只差 0.78°，竖线用虚线区分、
        # 穿越点用空心方块标（与曲线的空心方块同一语义），不另加文字标签
        # —— 0.78° 在 1.2–12.8 的轴上只占 7% 宽，两个标签必然互压。
        ax.axvline(s_u, color=DOMCOL["uniform"], lw=0.55, ls=(0, (1.4, 1.6)),
                   alpha=0.85, zorder=2)
        ax.plot([s_u], [1.0], marker="s", ms=3.2, mfc="none",
                mec=DOMCOL["uniform"], mew=0.8, zorder=6)

    # Δs：两个域各自追平插值的跨度之差（均匀 6.14° → 缺陷 6.92°）。
    # 图上只写 "< 1°"：两个穿越点都是 2° 网格上的内插，缺陷域三种子的 s* 极差就有
    # 0.26°，而 0.78° 只相当于 0.39 个列间距 —— 印 "0.8" 会让读者当成 0.80。
    # 精确值（0.78°）由 CSV 与图注承载，图上只做"小于 1°"这个稳健定性断言。
    #
    # 排版：0.78° 在轴上只有约 3.8 mm 宽，比 "Δs<1°" 这串字还窄。若把文字居中放在
    # 箭头正上方，它会横跨并盖住两个箭头头部（放大后糊成一个黑块）。所以文字右移到
    # 箭头之外（x ≈ 7.2–8.5、y ≈ 0.82 区域无曲线），箭头本身也缩到 ms=3.6 以免头部互压。
    if s_d is not None and s_u is not None:
        ds = s_d - s_u
        y_arrow = 0.820
        ax.add_patch(FancyArrowPatch((s_u, y_arrow), (s_d, y_arrow),
                                     arrowstyle="<|-|>", mutation_scale=3.6,
                                     lw=0.6, color="#333333", zorder=5,
                                     shrinkA=0.8, shrinkB=0.8))
        lbl = ("$\\Delta s<1^\\circ$" if abs(ds) < 1.0
               else f"$\\Delta s\\approx{ds:.1f}^\\circ$")
        ax.annotate(lbl, (s_d, y_arrow), textcoords="offset points",
                    xytext=(4.0, 0), fontsize=5.6, color="#333333", ha="left",
                    va="center")

    # "哪一侧更好"由 y=1 参考线与图注承担，图上不再放斜体小字（会撞图例与刻度）
    ax.set_ylim(0.75, 1.22)
    ax.set_xlim(1.2, 12.8)
    ax.set_xticks(ss_d)
    ax.set_xticklabels([f"{s}°" for s in ss_d])
    ax.set_xlabel("Sampling span $s$")
    ax.set_ylabel("Row-16 RMSE ratio (CNN / linear)")
    ax.set_title("(a) Where completion overtakes", fontsize=7.0, pad=3.5)
    ax.grid(alpha=0.16, lw=0.45)
    ax.set_axisbelow(True)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.legend(loc="upper right", frameon=True, framealpha=0.95, borderpad=0.3,
              handletextpad=0.4, borderaxespad=0.35)

    # ================= (b) 归一化退化率 =================
    ax = fig.add_subplot(gs[0, 1])
    for dom, name, ls in ((uni, "Uniform", (0, (3.6, 2.0))), (defe, "Defect", "-")):
        ss = sorted(s for s in STEPS if s in dom)
        if len(ss) < 2:
            continue
        for m in ("linear", "cubic", "cnn"):
            yv = np.array([dom[s][m] for s in ss], dtype=float)
            yv = yv / yv[0]
            ax.plot(ss, yv, ls=ls, marker=("o" if name == "Defect" else "s"),
                    color=COL[m], lw=(1.3 if m == "cnn" else 0.95),
                    ms=(3.0 if m == "cnn" else 2.6),
                    mfc=(COL[m] if name == "Defect" else "white"), mew=0.9,
                    zorder=(4 if m == "cnn" else 3))
    # 端点增幅：缺陷域
    for m, dy, ha in (("linear", 3, "right"), ("cnn", -10, "right")):
        ss = sorted(s for s in STEPS if s in defe)
        yv = np.array([defe[s][m] for s in ss], dtype=float)
        ax.annotate(f"{(yv[-1] / yv[0] - 1) * 100:+.0f}%", (ss[-1], yv[-1] / yv[0]),
                    textcoords="offset points", xytext=(-2, dy), fontsize=5.7,
                    color=COL[m], ha=ha, va="bottom" if dy > 0 else "top",
                    path_effects=[pe.withStroke(linewidth=1.6, foreground="white")])
    h = [plt.Line2D([], [], color=COL[m], lw=(1.3 if m == "cnn" else 0.95),
                    marker="o", ms=2.8, label=LBL[m]) for m in ("linear", "cubic", "cnn")]
    l1 = ax.legend(handles=h, loc="upper left", frameon=False, handletextpad=0.4)
    ax.add_artist(l1)                      # 第一个图例必须 add_artist，否则会被第二个顶掉
    ax.legend(handles=[plt.Line2D([], [], color="#555555", ls="-", lw=1.0, label="Defect"),
                       plt.Line2D([], [], color="#555555", ls=(0, (3.6, 2.0)), lw=1.0,
                                  label="Uniform")],
              loc="lower right", frameon=False, handletextpad=0.4)
    # (a)(c) 的刻度带度数符号，(b) 必须一致，否则三联图看起来像两套坐标
    ax.set_xticks(list(STEPS))
    ax.set_xticklabels([f"{s}°" for s in STEPS])
    ax.set_xlim(1.2, 12.8)
    ax.set_xlabel("Sampling span $s$")
    ax.set_ylabel("RMSE / RMSE($s$=2°)")
    ax.set_title("(b) Normalised degradation", fontsize=7.0, pad=3.5)
    ax.grid(alpha=0.16, lw=0.45)
    ax.set_axisbelow(True)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.margins(y=0.12)

    # ================= (c) 制度格 =================
    ax = fig.add_subplot(gs[0, 2])
    gain = np.full((2, len(STEPS)), np.nan)        # 分数 (1 - ratio)
    for j, s in enumerate(STEPS):
        if s in uni:
            gain[0, j] = 1.0 - ratio_of(uni[s])
        if s in defe:
            gain[1, j] = 1.0 - defe[s]["ratio"]
    # 差异图统一用 RdBu_r（红 = 补全更优 >0），刻度以**百分数**呈现，
    # vmin/vmax = ±20%，零等值线居中。颜色条标 "1−ratio (%)"。
    cmap = plt.get_cmap("RdBu_r")
    lim = 20.0                                      # 单位：百分点
    norm = TwoSlopeNorm(vmin=-lim, vcenter=0.0, vmax=lim)
    masked = np.ma.masked_invalid(gain * 100.0)      # 转百分数再上色
    im = ax.imshow(masked, cmap=cmap, norm=norm, aspect="auto", origin="upper",
                   extent=(-0.5, len(STEPS) - 0.5, 1.5, -0.5))

    # 列的"数据坐标"（与 imshow 的 extent 一致），零等值线要用它定位
    x_of = lambda s: (s - STEPS[0]) / (STEPS[1] - STEPS[0])
    xs_cross = [x_of(s) for s in (s_d, s_u) if s is not None]

    for i in range(2):
        for j in range(len(STEPS)):
            if np.isnan(gain[i, j]):
                ax.text(j, i, "n/a", ha="center", va="center", fontsize=5.2,
                        color="#999999", zorder=9)
                continue
            v = gain[i, j]
            # 四舍五入到整数后若为 0，不写 "−0"/"+0"（-0.2% 会印成 "-0"）
            lab = "0" if abs(v * 100) < 0.5 else f"{v * 100:+.0f}"
            # 零等值线会穿过的格子：均匀域 s=6° 那格的值是 0，而零穿越点 6.14°
            # 距格心只有 0.14°，等值线正好劈在数字右半边。仅抬高 zorder 不够
            # （黑虚线仍压在笔画上），所以给这一格加**同色底衬**把线在字后挖空，
            # 这也是等值线标注的常规做法。其它格子不变，避免整片出现色块。
            near = any(abs(xc - j) < 0.34 for xc in xs_cross)
            bbox = None
            if near:
                bbox = dict(boxstyle="square,pad=0.20", ec="none",
                            fc=to_hex(cmap(norm(v * 100.0))))
            ax.text(j, i, lab, ha="center", va="center", fontsize=5.6, zorder=9,
                    bbox=bbox,
                    color=("#0B2E4A" if v < 0 else "#4A1A00"),
                    fontweight=("bold" if abs(v) > 0.10 else "normal"))

    # 零等值线（分段折线）：**两行现在都有穿越**，所以这条线是真实存在而非伪造。
    # 每行内用它自己内插出的穿越点画竖虚线，两行之间按线性连接 —— 2 行网格上
    # 边界只在"域"方向有两个采样点，连线是线性内插（不是拟合），已在图注说明。
    if s_d is not None and s_u is not None:
        xd, xu = x_of(s_d), x_of(s_u)
        seg = [((xu, 0.5), (xu, -0.5)), ((xu, 0.5), (xd, 0.5)), ((xd, 0.5), (xd, 1.5))]
        for (x0, y0), (x1, y1) in seg:                  # 白色描边，压在格子上仍可读
            ax.plot([x0, x1], [y0, y1], color="white", lw=1.9, zorder=5,
                    solid_capstyle="butt")
        for (x0, y0), (x1, y1) in seg:
            ax.plot([x0, x1], [y0, y1], color="#111111", lw=0.75,
                    ls=(0, (2.2, 1.8)), zorder=6)

    ax.set_xticks(range(len(STEPS)))
    ax.set_xticklabels([f"{s}°" for s in STEPS])
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["Uniform", "Defect"])
    ax.set_xlabel("Sampling span $s$")
    ax.tick_params(length=0)
    for sp in ax.spines.values():
        sp.set_linewidth(0.5)
        sp.set_color("#B0B0B0")
    ax.set_title("(c) Regime map  ($1-$ratio, %)", fontsize=7.0, pad=3.5)

    # 颜色条：单位百分点，label "1−ratio (%)"，置于 (c) 右侧
    cbar = fig.colorbar(im, ax=ax, fraction=0.055, pad=0.03)
    cbar.set_label("1−ratio (%)", fontsize=6.4)
    cbar.ax.tick_params(labelsize=5.6)
    cbar.locator = plt.matplotlib.ticker.MaxNLocator(5)
    cbar.update_ticks()

    # 仅输出 PNG（审稿要求 PNG only，禁止 PDF）
    p = FIG / "fig_regime_joint_en.png"
    fig.savefig(p, dpi=DPI)
    plt.close(fig)

    print(f"[defect]  s* = {s_d if s_d is None else round(s_d, 2)}")
    print(f"[uniform] s* = {cross_span(uni)}  (None = 全程同侧，无穿越)")
    print(f"[OK] {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
