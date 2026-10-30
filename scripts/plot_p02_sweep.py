#!/usr/bin/env python
# -*- coding: utf-8 -*-
r"""
scripts/plot_p02_sweep.py —— 角度采样跨度扫描的定稿折线图（Fig 8，均匀域）

横轴：采样跨度 s（2°/4°/6°/8°/10°/12°，实际有几档画几档）。
曲线（主图默认 3 条，见 --with-ref 才恢复第 4 条）：
    linear      ½·[L(θ−h) + L(θ+h)]                              —— 用 2 个实测视角
    cubic       9/16·[L(θ−h)+L(θ+h)] − 1/16·[L(θ−3h)+L(θ+3h)]    —— 用 4 个实测视角
    CNN         主结果，预算由 --tag 决定（默认 100 epoch）
    fbp_sparse  不补全的 FBP，只在图像域给出（浅灰虚线，线端标注，不进图例）

三次插值是**被优待**的基线：它用 4 个实测视角，比 CNN 的输入多一个
（CNN 只给 θ−3h、θ−h、θ+h 三帧）。CNN 若仍能赢，结论更强。

版式（2026-10-30 按图表专家意见重构）
------------------------------------
    ┌─────────────┬─────────────┐
    │ (a) 投影域  │ (b) 图像域  │
    │   row 16    │   RMSE      │
    ├─────────────┤             │
    │ CNN/linear  │  （跨两行） │
    │  比值窄条   │             │
    └─────────────┴─────────────┘
* 窄条高 12 mm，与 (a) **共享 x 轴**，只画 CNN/linear 比值 + y=1 参考线。
  它回答"实测视角被抽稀时，补全相对插值的领先是收窄还是扩大"，替代原来挤在
  (a) 右上角的 inset（inset 遮住了 2°–4° 的真实数据点）。
* SSIM 面板（旧 (c)）移出主图 —— `--with-ssim` 仅供补充材料。
* 画布底部原有两行说明文字（"The absolute error rises …"），已移入图注。
  图内不再有脱离坐标系的散文，避免 PDF 文本层里出现成片长句。
* 配色 Okabe–Ito（色盲友好）：CNN #D55E00、linear #0072B2、cubic #009E73。

静默改图意的防线（P0 修复）
---------------------------
旧版把 `--tag` 默认设为 "60"、`--ref` 默认 ""（=30 ep），且模块级 `LBL` 里
硬写着 "CNN completion (60 epochs)"。于是**不带参数重跑**会把主曲线从 100 ep
静默切成 60 ep，而 CSV 的 `method` 列只写 `cnn`，读表的人无从分辨。
现在：
  1. `--tag` 默认 "100_3seed"（定稿的 3 种子汇总），`--ref` 默认 "60"；
     **不带参数重跑 = 复现定稿图**，不会退回单 run；
  2. 若 `--tag` 与 `--ref` 相同 → 直接退出（否则两条曲线同源，图意错误）；
  3. 图例文字一律由预算实数生成，模块常量不再写死 epoch；
     定稿 tag 的文字登记在 `BUDGET_LBL`，避免出现 "100_3seed epochs" 这类半截话；
  4. CSV 每行追加 `model_budget_epochs` 列（linear/cubic/fbp 留空），
     落到表里的每个 CNN 数值都自带预算标签。

数据源
------
  runs/_reviewer_p0/sweep{tag}_s{2,4,6,8,10,12}.json   主结果（含 cubic）
      └ 定稿 tag = "100_3seed"（由 scripts/agg_sweep_3seed.py 聚合 3 个 run 写出）
      └ "100" 是**单 run**（seed 20261022），已不作为默认，仅在复现旧图时用
  runs/_reviewer_p0/sweep{ref}_s*.json                 参照预算（仅 --with-ref）

用法
----
  python scripts/plot_p02_sweep.py                      # 定稿图（无需参数）
  python scripts/plot_p02_sweep.py --tag 100 --ref 60    # 单 run 版本（对照用）
  python scripts/plot_p02_sweep.py --out fig_step_sweep_supp --with-ssim

输出
----
  docs/figs/p02_step_sweep/<out>.{png,pdf,csv}
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
from matplotlib import ticker as mticker
from matplotlib.lines import Line2D

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "runs" / "_reviewer_p0"
FIG = ROOT / "docs" / "figs" / "p02_step_sweep"
DPI = 300
STEPS = (2, 4, 6, 8, 10, 12)

# 预算标签在 main() 里由 --tag/--ref 赋值；这里只给默认值，
# 任何遗漏都会在 validate_budget() 里报错，而不是悄悄沿用 60。
TAG = "100_3seed"
REF = "60"
TAG_LBL = "100 epochs, 3 seeds"
REF_LBL = "60"

# 预算 tag → 图例文字。只登记**定稿在用**的 tag，其余按 "xxx epochs" 兜底。
BUDGET_LBL = {"100_3seed": "100 epochs, 3 seeds"}

SERIES = ("linear", "cubic", "cnn", "cnn_ref", "fbp_sparse")

# Okabe–Ito 色盲友好调色板（标准 8 色中的三主色 + 参照紫 + 中性灰）
COL = {"linear": "#0072B2",     # blue
       "cubic": "#009E73",      # bluish green
       "cnn": "#D55E00",        # vermillion
       "cnn_ref": "#CC79A7",    # reddish purple（参照预算，虚线）
       "fbp_sparse": "#9A9A9A"}  # grey
STYLE = {"cnn_ref": (0, (4, 2)), "fbp_sparse": (0, (2.4, 1.6))}

BASE_LBL = {"linear": "Linear interpolation (2 views)",
            "cubic": "Cubic interpolation (4 views)",
            "fbp_sparse": "FBP, no completion"}
LBL: dict[str, str] = {}


def validate_budget(tag_lbl: str | None = None, ref_lbl: str | None = None) -> None:
    """把"预算写死"这类静默错误挡在绘图之前。"""
    global TAG_LBL, REF_LBL
    TAG_LBL = (tag_lbl or BUDGET_LBL.get(TAG) or TAG or "30").strip()
    REF_LBL = (ref_lbl or BUDGET_LBL.get(REF) or REF or "30").strip()
    if TAG.strip() == REF.strip():
        raise SystemExit(
            f"[FATAL] --tag 与 --ref 相同（{TAG!r}）：两条 CNN 曲线会同源，图意错误。")
    # 标签里若已写明 epoch（如 "100 epochs, 3 seeds"）就原样用，否则补上单位，
    # 避免出现 "100, 3 seeds epochs" 这种半截话。
    if "epoch" not in TAG_LBL.lower():
        TAG_LBL = f"{TAG_LBL} epochs"
    if "epoch" not in REF_LBL.lower():
        REF_LBL = f"{REF_LBL} epochs"
    LBL.update(BASE_LBL)
    LBL["cnn"] = f"CNN completion ({TAG_LBL})"
    LBL["cnn_ref"] = f"CNN completion ({REF_LBL})"


plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 7.4,
    "axes.linewidth": 0.7,
    "axes.labelsize": 7.6,
    "xtick.labelsize": 6.8,
    "ytick.labelsize": 6.8,
    "legend.fontsize": 6.0,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "savefig.facecolor": "white",
    "pdf.fonttype": 42,          # TrueType 嵌入：文本层可提取、不转曲
    "ps.fonttype": 42,
    "mathtext.fontset": "dejavusans",
})


def load(tag: str) -> dict[int, dict]:
    out = {}
    for s in STEPS:
        p = SRC / f"sweep{tag}_s{s}.json"
        if p.is_file():
            out[s] = json.loads(p.read_text(encoding="utf-8"))
    return out


def series(data: dict[int, dict], ref: dict[int, dict], getter,
           steps: list[int]) -> dict[str, list[float]]:
    """cnn_ref 从 ref（参照预算）取，其余从 data 取。"""
    out: dict[str, list[float]] = {}
    for m in SERIES:
        src = ref if m == "cnn_ref" else data
        key = "cnn" if m == "cnn_ref" else m
        vals = []
        ok = True
        for s in steps:
            if s not in src:
                ok = False
                break
            v = getter(src[s], key)
            if v is None:
                ok = False
                break
            vals.append(v)
        if ok:
            out[m] = vals
    return out


def g_proj(d, m):
    return d["projection"].get(m, {}).get("row16", {}).get("overall", {}).get("rmse_valid")


def g_rmse(d, m):
    return d["image"].get(m, {}).get("overall", {}).get("rmse_valid")


def g_ssim(d, m):
    return d["image"].get(m, {}).get("overall", {}).get("ssim")


def budget_of(m: str) -> str:
    if m == "cnn":
        return TAG_LBL
    if m == "cnn_ref":
        return REF_LBL
    return ""


def style_axes(ax, xlab: list[str], xs) -> None:
    ax.set_xticks(xs)
    ax.set_xticklabels(xlab)
    ax.xaxis.set_minor_locator(mticker.NullLocator())   # x 轴无 minor，杜绝多余标签
    ax.set_xlim(-0.45, len(xs) - 0.55)
    ax.grid(color="#EDEDED", lw=0.5)
    ax.set_axisbelow(True)
    for s_ in ("top", "right"):
        ax.spines[s_].set_visible(False)


def main() -> int:
    global TAG, REF
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="100_3seed",
                    help="主结果预算：'' =30ep, '60', '100'（单 run）, "
                         "'100_3seed'（定稿，3 种子汇总）；对应 sweep{tag}_s*.json")
    ap.add_argument("--ref", default="60",
                    help="参照预算（默认 60ep；仅 --with-ref 时画出）")
    ap.add_argument("--out", default="fig_step_sweep_en")
    ap.add_argument("--with-ref", action="store_true",
                    help="画出参照预算曲线（默认不画：图例只留 3 条，预算说明进图注）")
    ap.add_argument("--with-ssim", action="store_true",
                    help="附加 SSIM 面板（补充材料用；主图不放）")
    ap.add_argument("--tag-label", default=None,
                    help="图例里显示的预算文字（默认取 --tag；3 种子汇总时传 '100, 3 seeds'）")
    ap.add_argument("--ref-label", default=None, help="同 --tag-label，作用于参照曲线")
    a = ap.parse_args()
    TAG, REF = a.tag, a.ref
    validate_budget(a.tag_label, a.ref_label)

    FIG.mkdir(parents=True, exist_ok=True)
    data = load(TAG)
    ref = load(REF)
    if not data:
        raise SystemExit(f"没有 sweep{TAG}_s*.json，先跑 scripts/p02_sweep_eval.py --tag {TAG}")
    steps = [s for s in STEPS if s in data]
    if len(steps) < len(STEPS):
        print(f"[warn] 只找到 s={steps}，其余档缺失")
    xs = np.arange(len(steps))
    xlab = [f"{s}" for s in steps]

    ncol = 3 if a.with_ssim else 2
    fig = plt.figure(figsize=(6.9 if ncol == 2 else 8.4, 3.05))
    gs = fig.add_gridspec(2, ncol, height_ratios=[1.0, 0.26],
                          width_ratios=[1.0] * ncol,
                          hspace=0.10, wspace=0.30)
    ax_a = fig.add_subplot(gs[0, 0])
    ax_strip = fig.add_subplot(gs[1, 0], sharex=ax_a)
    ax_b = fig.add_subplot(gs[:, 1])
    ax_c = fig.add_subplot(gs[:, 2]) if a.with_ssim else None

    rows_out: list[list] = []

    def draw(ax, getter, methods, logy: bool, title: str, ylab: str, loc: str,
             annotate: bool = True):
        vals = series(data, ref, getter, steps)
        for m in methods:
            if m not in vals:
                continue
            ax.plot(xs, vals[m], marker="o", ms=3.2, lw=(1.35 if m == "cnn" else 1.0),
                    color=COL[m], label=LBL[m], zorder=(4 if m == "cnn" else 3),
                    ls=STYLE.get(m, "-"))
            if (not annotate) or m in ("cnn_ref", "cubic", "fbp_sparse"):
                continue          # 虚线 / 参照基线不加数字，靠图例与 CSV 承载
            v = vals[m][-1]
            txt = f"{v:.4f}" if v < 0.01 else f"{v:.3f}"
            # 是否标在点上方：要跟"该面板所有曲线的末端值"比，不能只比 linear/cnn —
            # 否则 linear 恒被判为最高，标签会压到 cubic 的点上（0.157 vs 0.158）。
            hi = v >= max(vals[o][-1] for o in vals)
            ax.annotate(txt, (xs[-1], v), textcoords="offset points",
                        xytext=(-4, 8 if hi else -12), fontsize=5.4,
                        color=COL[m], ha="right", va="center")
        if logy:
            ax.set_yscale("log")
            ax.yaxis.set_minor_formatter(mticker.NullFormatter())
        style_axes(ax, xlab, xs)
        ax.set_ylabel(ylab)
        ax.set_title(title, fontsize=7.2, pad=4)
        ax.margins(y=0.16)
        panel = "Projection" if getter is g_proj else ("Image" if getter is g_rmse else "SSIM")
        for i, s in enumerate(steps):
            for m in methods:
                if m in vals:
                    rows_out.append([s, 360 // s, panel, m, vals[m][i], budget_of(m)])
        return vals

    proj_methods = ["linear", "cubic", "cnn"] + (["cnn_ref"] if a.with_ref else [])
    v_proj = draw(ax_a, g_proj, proj_methods, False,
                  "(a) Projection domain, row 16",
                  "RMSE (test views, $-\\ln$ units)", "upper left")
    ax_a.tick_params(labelbottom=False)          # x 标签由下方窄条承担
    ax_a.legend(loc="upper left", frameon=True, framealpha=0.94, borderpad=0.28,
                handletextpad=0.36, borderaxespad=0.30)

    img_methods = (["linear", "cubic", "cnn"] + (["cnn_ref"] if a.with_ref else [])
                   + ["fbp_sparse"])
    # (b) 末端三条曲线在 s>=10 挤在 6% 的区间里，标数值会互相压住；
    # 该面板的作用是"FBP 欠采样主导"的背景，精确值由 CSV 与图注承载。
    v_img = draw(ax_b, g_rmse, img_methods, True,
                 "(b) Image domain RMSE", "RMSE vs. dense reference", "lower right",
                 annotate=False)
    # fbp_sparse 不进图例（图例只留 3 条主方法），改为线端小字标注
    if "fbp_sparse" in v_img:
        ax_b.annotate("FBP, no completion", (xs[-1], v_img["fbp_sparse"][-1]),
                      textcoords="offset points", xytext=(-3, 9), fontsize=5.2,
                      color=COL["fbp_sparse"], ha="right", va="center")
    ax_b.legend(handles=[Line2D([], [], color=COL[m], lw=(1.35 if m == "cnn" else 1.0),
                                marker="o", ms=3.2, label=LBL[m])
                         for m in ("linear", "cubic", "cnn")
                         + (("cnn_ref",) if a.with_ref else ())],
                loc="lower right", frameon=True, framealpha=0.94, borderpad=0.28,
                handletextpad=0.36, borderaxespad=0.30)

    # ---- 窄条：CNN / linear 比值，与 (a) 共享 x 轴
    r = [v_proj["cnn"][i] / v_proj["linear"][i] for i in range(len(steps))]
    ax_strip.plot(xs, r, marker="o", ms=2.8, lw=1.1, color=COL["cnn"], zorder=3)
    ax_strip.axhline(1.0, color="#666666", lw=0.75, ls=(0, (3, 2)), zorder=2)
    lo = min(min(r), 1.0)
    hi = max(max(r), 1.0)
    pad = 0.26 * (hi - lo if hi > lo else 0.05)
    ax_strip.set_ylim(lo - pad, hi + pad)
    # 窄条只有 12 mm：y 轴只留 y=1 与数据最小值两个刻度，数值交给线上的两端标注
    ax_strip.set_yticks([round(lo, 2), 1.0])
    for i in (0, len(steps) - 1):
        # 起点贴着 y=1 参考线 → 标在点下方；终点接近下边界 → 标在点上方
        first = i == 0
        ax_strip.annotate(f"{r[i]:.3f}", (xs[i], r[i]), textcoords="offset points",
                          xytext=(0, -5 if first else 5), fontsize=5.2, color=COL["cnn"],
                          ha="left" if first else "right", va="top" if first else "bottom")
    style_axes(ax_strip, xlab, xs)
    ax_strip.set_xlabel("Sampling step $s$ (deg)", labelpad=2)
    ax_strip.set_ylabel("ratio", fontsize=5.4, labelpad=3)
    ax_strip.tick_params(axis="y", labelsize=5.2, length=1.6, pad=1.4, width=0.5)
    ax_strip.tick_params(axis="x", labelsize=6.4, length=2.2, pad=1.6)
    for sp in ("left", "bottom"):
        ax_strip.spines[sp].set_linewidth(0.5)

    if ax_c is not None:
        draw(ax_c, g_ssim, img_methods, False,
             "(c) Image domain SSIM", "SSIM vs. dense reference", "upper right")

    # ---- 交叉点竖线只在 (a)、(a') 上标，色随被判超的基线
    for base in ("linear", "cubic"):
        if base not in v_proj or "cnn" not in v_proj:
            continue
        win = [steps[i] for i in range(len(steps))
               if v_proj["cnn"][i] < v_proj[base][i]]
        if win and steps.index(win[0]) > 0:
            i = steps.index(win[0])
            for ax_ in (ax_a, ax_strip):
                ax_.axvline(i - 0.5, color=COL[base], lw=0.7, ls=(0, (2, 2)),
                            alpha=0.65, zorder=1)
            ax_a.annotate(f"CNN < {base}\nfrom $s$={win[0]}$^\\circ$",
                          (i - 0.5, ax_a.get_ylim()[0]),
                          xytext=(2, 4), textcoords="offset points",
                          fontsize=5.0, color=COL[base], ha="left", va="bottom")

    fig.subplots_adjust(left=0.088, right=0.978, top=0.915, bottom=0.135)

    out = []
    for ext in ("png", "pdf"):
        p = FIG / f"{a.out}.{ext}"
        fig.savefig(p, dpi=DPI)
        out.append(p)
    plt.close(fig)

    # ---- CSV：每行自带预算标签，方法列与图例一一对应
    with open(FIG / f"{a.out}.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["step_deg", "n_measured_views", "panel", "method", "value",
                    "model_budget_epochs"])
        w.writerows(rows_out)

    # 标签里已含单位（"100 epochs, 3 seeds"），这里不能再补 " epochs"
    print(f"[budget] 主曲线 = {TAG_LBL}；参照 = {REF_LBL}"
          f"{'（已画出）' if a.with_ref else '（未画出，仅计入图注）'}")
    for getter, name in ((g_proj, "row16"), (g_rmse, "image RMSE")):
        vv = series(data, ref, getter, steps)
        print(f"  [{name}] " + " | ".join(f"{m}={vv[m][-1]:.6f} (s={steps[-1]})"
                                          for m in SERIES if m in vv))
        for base in ("linear", "cubic"):
            if base in vv and "cnn" in vv:
                print(f"      CNN/{base}: " + " ".join(
                    f"{vv['cnn'][i] / vv[base][i]:.4f}" for i in range(len(steps))))
    for p in out:
        print(f"[OK] {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
