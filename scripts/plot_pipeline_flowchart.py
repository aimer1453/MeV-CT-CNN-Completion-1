#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Fig 1 —— 稀疏角 CT 补全管线流程图（中英双语，学术风）。

三列布局：Data Preparation / Conditioning & U-Net / Training & Evaluation。
虚线圆角模块框 + 极淡底色；白底节点 + 细黑边；标准几何形状
（圆角矩形 = 起止，平行四边形 = 数据/输出，矩形 = 处理/U-Net，菱形 = 判断）。
直角连线；Converged? 的 No 分支绕右-下外侧回到 U-Net。

**按最终物理尺寸出图**：figsize = 6.30×3.48 in = \textwidth，LaTeX 缩放比 1.0，
图内字号 7.5–9 pt 即印刷字号（旧版 11.97 in 宽被缩到 0.526，等效只剩 4.5–5.7 pt）。

只依赖 matplotlib。用法：
    python scripts/plot_pipeline_flowchart.py            # 出图
    python scripts/plot_pipeline_flowchart.py --fit      # 出图并报告文字溢出
产出：docs/figs/p01_pipeline/pipeline_flowchart.{pdf,png}
      docs/figs/p01_pipeline/pipeline_flowchart_zh.{pdf,png}
"""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyBboxPatch, Polygon  # noqa: E402

OUT = __import__("pathlib").Path(__file__).resolve().parent.parent / "docs" / "figs" / "p01_pipeline"

EDGE = "#2b2b2b"
ARROW = "#1a1a1a"
GRAY = "#5a5a5a"
C_M1 = "#F0F8FF"   # aliceblue
C_M2 = "#F0FFF0"   # honeydew
C_M3 = "#FFF5EE"   # seashell

# 画布（数据坐标）
XL = (0.0, 14.25)
YL = (-0.05, 7.72)

# 最终印刷尺寸（inch）：\textwidth = 455.24 pt = 6.32 in，故取 6.30×3.48。
# 图内字号即印刷字号（7.5–9 pt），不再被 \includegraphics 二次缩小。
FIG_W = 6.30
FIG_H = 3.48

# 三列顶端节点的统一基准线（视觉对齐）：模块一 Start / 模块二 Target Angle /
# 模块三 Masked MSE Loss 的中心同在 TOP_Y，形成一条横向基准线。
TOP_Y = 6.55

_FIT: list = []   # --fit 时记录节点文字与形状，供 _report_fit 量化检测
NODE: dict = {}   # 节点几何登记表：连线端点一律由几何推导，不手写坐标


def add_cluster(ax, x0, y0, w, h, color, title, fs):
    p = FancyBboxPatch((x0, y0), w, h,
                       boxstyle="round,pad=0.0,rounding_size=0.16",
                       fc=color, ec="#b9b9b9", lw=0.7, ls=(0, (4, 2.5)), zorder=1)
    ax.add_patch(p)
    ax.text(x0 + 0.14, y0 + h - 0.14, title, ha="left", va="top",
            fontsize=fs, fontweight="bold", color="#141414", zorder=6)
    _FIT.append(dict(text=title, cx=x0 + w / 2, cy=y0 + h, w=w - 0.28, h=0.42,
                     shape="title", fs=fs, artist=ax.texts[-1]))


def add_node(ax, cx, cy, w, h, text, shape, fs):
    if shape == "round":
        ax.add_patch(FancyBboxPatch(
            (cx - w / 2, cy - h / 2), w, h,
            boxstyle="round,pad=0.0,rounding_size=0.14",
            fc="white", ec=EDGE, lw=0.75, zorder=4))
    elif shape == "box":
        ax.add_patch(FancyBboxPatch(
            (cx - w / 2, cy - h / 2), w, h,
            boxstyle="square,pad=0.0", fc="white", ec=EDGE, lw=0.75, zorder=4))
    elif shape == "para":
        sk = h * 0.42
        pts = [(cx - w / 2 - sk, cy - h / 2), (cx + w / 2 - sk, cy - h / 2),
               (cx + w / 2 + sk, cy + h / 2), (cx - w / 2 + sk, cy + h / 2)]
        ax.add_patch(Polygon(pts, closed=True, fc="white", ec=EDGE, lw=0.75, zorder=4))
    elif shape == "diamond":
        pts = [(cx - w / 2, cy), (cx, cy + h / 2), (cx + w / 2, cy), (cx, cy - h / 2)]
        ax.add_patch(Polygon(pts, closed=True, fc="white", ec=EDGE, lw=0.75, zorder=4))
    ax.text(cx, cy, text, ha="center", va="center", fontsize=fs, zorder=5,
            color="#111111", linespacing=1.2)
    _FIT.append(dict(text=text, cx=cx, cy=cy, w=w, h=h, shape=shape, fs=fs,
                     artist=ax.texts[-1]))


def node(ax, name, cx, cy, w, h, text, shape, fs):
    """登记几何 + 画节点。连线用 top/bot/left/right 取值，不写死坐标。"""
    add_node(ax, cx, cy, w, h, text, shape, fs)
    NODE[name] = dict(cx=cx, cy=cy, w=w, h=h, shape=shape)


def top(n):
    return NODE[n]["cy"] + NODE[n]["h"] / 2


def bot(n):
    return NODE[n]["cy"] - NODE[n]["h"] / 2


def left(n):
    return NODE[n]["cx"] - NODE[n]["w"] / 2


def right(n):
    return NODE[n]["cx"] + NODE[n]["w"] / 2


def ymid(n):
    return NODE[n]["cy"]


def arrow(ax, pts, label=None, lpos=None, color=ARROW, lw=0.8, ls="-",
          fs=7.5, rot=0, bbox_fc="white"):
    dls = ls if isinstance(ls, tuple) else ((0, (4, 3)) if ls == "--" else "-")
    if len(pts) == 2:
        ax.annotate("", xy=pts[1], xytext=pts[0],
                    arrowprops=dict(arrowstyle="-|>", color=color, lw=lw, ls=dls,
                                    shrinkA=0, shrinkB=0, mutation_scale=8), zorder=3)
    else:
        ax.plot([p[0] for p in pts], [p[1] for p in pts], color=color, lw=lw, ls=dls,
                solid_capstyle="round", solid_joinstyle="miter", zorder=3)
        ax.annotate("", xy=pts[-1], xytext=pts[-2],
                    arrowprops=dict(arrowstyle="-|>", color=color, lw=lw,
                                    shrinkA=0, shrinkB=0, mutation_scale=8), zorder=3)
    if label and lpos:
        # 标签压线时必须用「不透明 + 模块底色」的底框：alpha<1 会让线从字里透出来成灰影
        ax.text(lpos[0], lpos[1], label, fontsize=fs, ha="center", va="center",
                color="#333333", rotation=rot, zorder=7,
                bbox=dict(boxstyle="round,pad=0.18", fc=bbox_fc, ec="none", alpha=1.0))


def _report_fit(fig, ax, lang):
    """量化检测：每个节点的文字外接框是否超出其形状（半尺寸画布下的主要风险）。"""
    fig.canvas.draw()
    inv = ax.transData.inverted()
    print(f"\n[FIT {lang}] 节点文字占框比例（w>0.96 或 h>0.92 判溢出）")
    print(f"  {'node':34s} {'shape':7s} {'fs':>4s} {'w_ratio':>8s} {'h_ratio':>8s}  verdict")
    bad = 0
    for r in _FIT:
        bb = r["artist"].get_window_extent(fig.canvas.get_renderer()).transformed(inv)
        tw, th = bb.width, bb.height
        if r["shape"] == "title":                    # 模块标题只判宽度
            wx, hy = tw / r["w"], th / 0.42
        elif r["shape"] == "diamond":                 # 菱形：按 (tw/w)+(th/h) <= 1 判定
            wx = (tw / 2) / (r["w"] / 2) + (th / 2) / (r["h"] / 2)
            hy = wx
        else:                                        # 矩形 / 圆角 / 平行四边形
            wx, hy = tw / r["w"], th / r["h"]
        flag = "OVERFLOW" if (wx > 0.96 or hy > 0.92) else "ok"
        bad += flag != "ok"
        first = r["text"].split("\n")[0][:30]
        print(f"  {first:34s} {r['shape']:7s} {r['fs']:4.1f} {wx:8.2f} {hy:8.2f}  {flag}")
    print(f"  -> {len(_FIT)} nodes, {bad} overflow")
    _FIT.clear()


STR = {
    "en": dict(
        t1="1   Data Preparation", t2="2   Conditioning & U-Net",
        t3="3   Training & Evaluation",
        start="Start", geant="Geant4 Dual-phase\n360° Scan",
        extract="Extract 32×1200\nProjections", context="Context\n[θ−3°, θ−1°, θ+1°]",
        split="Split   270 / 45 / 45", target="Target Angle\nsinθ, cosθ",
        inp="5-Channel Input", unet="32×1200 U-Net\n(4 downsampling)",
        pred="Predicted Projection", loss="Masked MSE Loss\n(non-zero pixels)",
        conv="Converged?", rmse="Proj-domain RMSE\n(Row 16 vs 32)",
        image="Image-domain Metrics\n(FBP → RMSE/PSNR)",
        abl="Single-row\nablation",
        l_ch02="ch 0–2", l_ch34="ch 3–4", l_bat="batches", l_yes="Yes",
        l_no="No", l_san="sanity check", fs=8.0, fst=7.5,
    ),
    "zh": dict(
        t1="1   数据准备", t2="2   条件注入与 U-Net", t3="3   训练与评估",
        start="开始", geant="Geant4 双相位扫描\n(360° 投影)",
        extract="提取 32×1200 投影", context="实测邻角上下文\n[θ−3°, θ−1°, θ+1°]",
        split="划分   270 / 45 / 45", target="目标角度\nsinθ, cosθ",
        inp="5 通道输入", unet="32×1200 U-Net\n(4 次下采样)",
        pred="预测投影", loss="掩膜均方误差损失\n(非零像素)",
        conv="是否收敛？", rmse="投影域误差评估\n(Row 16 vs 32)",
        image="图像域重建评估\n(FBP → RMSE/PSNR)",
        abl="单行\n消融",
        l_ch02="通道 0–2", l_ch34="通道 3–4", l_bat="批次", l_yes="是",
        l_no="否", l_san="一致性检查", fs=7.8, fst=7.5,
    ),
}


def draw(lang: str, measure: bool = False):
    s = STR[lang]
    fs, fst = s["fs"], s["fst"]
    if lang == "zh":
        plt.rcParams["font.family"] = "sans-serif"
        plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
    else:
        plt.rcParams["font.family"] = "sans-serif"
        plt.rcParams["font.sans-serif"] = ["Arial", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["pdf.fonttype"] = 42
    plt.rcParams["ps.fonttype"] = 42

    NODE.clear()
    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))
    ax.set_xlim(*XL)
    ax.set_ylim(*YL)
    ax.set_aspect("equal")
    ax.axis("off")

    # ---------- 三个模块框 ----------
    add_cluster(ax, 0.15, 0.75, 3.85, 6.66, C_M1, s["t1"], fst)
    add_cluster(ax, 4.45, 0.75, 3.90, 6.66, C_M2, s["t2"], fst)
    add_cluster(ax, 8.85, 0.75, 5.00, 6.66, C_M3, s["t3"], fst)

    # ---------- 节点几何 ----------
    # 半尺寸画布下字号相对放大 ~1.58 倍，节点必须相应加高加宽（实测见 --fit）。
    # 三列顶端（start / target / loss）中心统一在 TOP_Y。
    X1 = 2.05
    node(ax, "start", X1, TOP_Y, 1.95, 0.47, s["start"], "round", fs)
    node(ax, "geant", X1, 5.395, 3.30, 0.86, s["geant"], "box", fs - 0.3)
    node(ax, "extract", X1, 4.035, 2.90, 0.86, s["extract"], "para", fs - 0.3)
    node(ax, "split", X1, 2.845, 2.85, 0.52, s["split"], "para", fs - 0.3)
    node(ax, "context", X1, 1.655, 2.95, 0.86, s["context"], "para", fs - 0.5)

    X2 = 6.40
    node(ax, "target", X2, TOP_Y, 2.85, 0.86, s["target"], "para", fs - 0.3)
    node(ax, "input", X2, 4.99, 2.55, 0.56, s["inp"], "para", fs - 0.3)
    node(ax, "unet", X2, 3.43, 3.35, 0.86, s["unet"], "box", fs - 0.3)
    node(ax, "pred", X2, 1.89, 3.10, 0.52, s["pred"], "para", fs - 0.3)

    X3 = 11.35
    node(ax, "loss", X3, TOP_Y, 3.40, 0.86, s["loss"], "box", fs - 0.3)
    node(ax, "conv", X3, 5.16, 3.00, 1.15, s["conv"], "diamond", fs - 0.2)
    node(ax, "rmse", X3, 3.77, 3.05, 0.86, s["rmse"], "para", fs - 0.4)
    node(ax, "image", X3, 2.525, 3.60, 0.86, s["image"], "para", fs - 0.4)
    node(ax, "abl", X3, 1.28, 3.70, 0.86, s["abl"], "box", fs - 0.3)

    # ---------- 模块一内部（竖直流） ----------
    arrow(ax, [(X1, bot("start")), (X1, top("geant"))])
    arrow(ax, [(X1, bot("geant")), (X1, top("extract"))])
    arrow(ax, [(X1, bot("extract")), (X1, top("split"))])
    arrow(ax, [(X1, bot("split")), (X1, top("context"))])

    # ---------- 跨模块：Context -> 5-Channel Input ----------
    # 标签贴在竖线中段（rot=90 竖排）：走线通道仅 0.45 单位(~14pt)宽，
    # 横排的 "ch 0–2"(~24pt) 必然压到两侧模块虚线框，只能顺线竖排。
    _g = 4.22                                   # 模块一/二之间的走线通道
    arrow(ax, [(right("context"), ymid("context")), (_g, ymid("context")),
               (_g, ymid("input")), (left("input"), ymid("input"))],
          label=s["l_ch02"], lpos=(_g, (ymid("context") + ymid("input")) / 2), rot=90)

    # ---------- 模块二内部 ----------
    # ch 3–4 标签居中在竖线本身（x=X2）：与 ch 0–2 同一口径——标签中心落在指向线上，
    # 白底框顺线破线。模块二内部宽度充裕（3.90 单位），横排即可，无需竖排。
    arrow(ax, [(X2, bot("target")), (X2, top("input"))],
          label=s["l_ch34"], lpos=(X2, (bot("target") + top("input")) / 2),
          bbox_fc=C_M2)
    arrow(ax, [(X2, bot("input")), (X2, top("unet"))])
    arrow(ax, [(X2, bot("unet")), (X2, top("pred"))])

    # ---------- Predicted Projection -> Loss（直角，走模块二/三之间的通道） ----------
    _g2 = 8.62
    arrow(ax, [(right("pred"), ymid("pred")), (_g2, ymid("pred")),
               (_g2, ymid("loss")), (left("loss"), ymid("loss"))])

    # ---------- 模块三内部 ----------
    arrow(ax, [(X3, bot("loss")), (X3, top("conv"))])
    arrow(ax, [(X3, bot("conv")), (X3, top("rmse"))],
          label=s["l_yes"], lpos=(11.75, (bot("conv") + top("rmse")) / 2), bbox_fc=C_M3)
    arrow(ax, [(X3, bot("rmse")), (X3, top("image"))])
    arrow(ax, [(X3, bot("image")), (X3, top("abl"))],
          label=s["l_san"], lpos=(12.10, (bot("image") + top("abl")) / 2), fs=7.0,
          color=GRAY, lw=0.7, ls="--", bbox_fc=C_M3)

    # ---------- No 分支：绕右侧 + 底部，从左侧回到 U-Net ----------
    arrow(ax, [(right("conv"), ymid("conv")), (13.98, ymid("conv")), (13.98, 0.38),
               (4.62, 0.38), (4.62, ymid("unet")), (left("unet"), ymid("unet"))],
          label=s["l_no"], lpos=(13.15, ymid("conv") + 0.16), bbox_fc=C_M3)

    if measure:
        _report_fit(fig, ax, lang)

    # 让内容铺满画布：PDF 的 MediaBox 即 figsize，LaTeX 里缩放比可为 1.0
    # （不用 bbox_inches="tight"，否则会多出 ~0.1 in 边距、缩放比掉到 0.98）
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
    OUT.mkdir(parents=True, exist_ok=True)
    stem = "pipeline_flowchart" if lang == "en" else "pipeline_flowchart_zh"
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"{stem}.{ext}", dpi=400, facecolor="white")
    plt.close(fig)
    print(f"[saved] {OUT / (stem + '.pdf')}")
    print(f"[saved] {OUT / (stem + '.png')}")


if __name__ == "__main__":
    import sys

    _measure = "--fit" in sys.argv
    draw("en", measure=_measure)
    draw("zh", measure=_measure)
