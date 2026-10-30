#!/usr/bin/env python
# -*- coding: utf-8 -*-
r"""
scripts/plot_method_figures.py —— R6：方法流程图 + 数据生成与划分图（中英双语）

内容规格唯一来源
----------------
`docs/handoff/algo_requests.md` 的 **R6 小节**（论文 agent 撰写，2026-10-23）。
本脚本**逐字照抄**那里的方框文字，不改写、不增删环节。中文版由同一张表给出，
两版共用同一套几何布局（`LAYOUT`），只有文字不同 —— 满足 R6 §6.3「中英一致」。

**不要复用旧 `scan.py` 流程图**（R6 §6.0 列了六处过时）：本脚本不含
`bright-line` / `correct_empty_proj` / `interference` / `Parse Energy.data`，
重建框写 `FBP / SART`，且把 CNN 补全作为方法主体画进主链。

两张图
------
  Fig. 1（P0，竖向，6 框，单栏 8.5 cm）：互补相位划分 → 输入上下文 → U-Net →
        掩膜 MSE 损失 → 交错拼接 → 重建与评估
  Fig. 2（P1，横向，5 框，双栏 17.5 cm）：Geant4 仿真 → 空扫 → 投影导出 →
        双相位划分 → 分层划分

尺寸即印刷尺寸
--------------
figure 的 inches 直接取**最终印刷宽度**（Fig1 3.346 in = 8.5 cm，Fig2 6.890 in = 17.5 cm），
坐标轴单位也用 inch。这样**字号 pt 就是印刷 pt**，R6「最终印刷尺寸下 ≥ 7 pt」
是可被本脚本直接断言的，而不是靠肉眼猜。

灰度安全
--------
所有方框同一浅灰填充 `#f2f2f2` + 黑边，**不靠颜色区分环节**；黑白打印后仅靠
框序与箭头方向区分（R6 §6.3）。

输出
----
  docs/figs/<run_id>/method_pipeline_{analysis,paper}.png / .pdf
  docs/figs/<run_id>/data_generation_{analysis,paper}.png / .pdf

用法
----
  conda run -n ct_full python scripts/plot_method_figures.py
  conda run -n ct_full python scripts/plot_method_figures.py --dpi 200
  conda run -n ct_full python scripts/plot_method_figures.py --no-title   # 若论文侧要求去掉图内标题
"""

from __future__ import annotations

import argparse
import sys
import unicodedata
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt                      # noqa: E402
from matplotlib import rcParams                      # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
RUN_ID = "20261022_2352_cnn_bs8_lr1e-4"
FIG_DIR = ROOT / "docs" / "figs" / RUN_ID

#: 双语后缀，与 `scripts/plot_norm_stats.py` 一致（zh -> _analysis，en -> _paper）
LANGS: tuple[str, ...] = ("zh", "en")
SUFFIX: dict[str, str] = {"zh": "analysis", "en": "paper"}

#: 英文字体：R6 要求 Times New Roman 或 Helvetica。Windows 无 Helvetica；
#: Times New Roman 实测含全部所需字形（θ ° × → ₂ −），故取 serif 路线。
FONTS: dict[str, list[str]] = {
    "zh": ["Microsoft YaHei", "DejaVu Sans"],
    "en": ["Times New Roman"],
}

#: 字号（pt）。R6 要求最终印刷尺寸下 ≥ 7 pt —— 本脚本 figure 尺寸即印刷尺寸，
#: 故这里的 pt 就是印刷 pt；`MIN_PT` 是自检下限。
PT_TITLE = 9.5
PT_BOX_TITLE = 8.5
PT_BOX_SUB = 7.5
MIN_PT = 7.0

FILL = "#f2f2f2"
EDGE = "black"
EDGE_LW = 0.8            # pt，R6 要求框线 ≥ 0.5 pt
ARROW_LW = 0.9
ROUND = 0.055            # inch，圆角半径


@dataclass(frozen=True)
class Box:
    title: str
    sub: tuple[str, ...]


@dataclass(frozen=True)
class Layout:
    """几何全部用 inch；figure 尺寸 == 印刷尺寸。"""
    width: float
    margin_x: float
    margin_top: float
    margin_bottom: float
    box_w: float
    box_h: float
    gap: float
    title_h: float          # 图标题占用的高度（--no-title 时置 0）
    #: 框内标题中心的相对高度（0=底边，1=顶边）
    title_frac: float = 0.70
    #: 副标题块的锚点相对高度；配合 `sub_va` 使用
    sub_frac: float = 0.30
    #: 副标题垂直对齐：竖向图用 center（行数相近，居中更稳）；
    #: 横向图行数 2~4 不等，用 top 让所有副标题从同一高度起排，避免参差
    sub_va: str = "center"


# ------------------------------------------------------------------ 内容（R6 逐字）
FIG1: dict[str, tuple[str, tuple[Box, ...]]] = {
    "en": ("Method pipeline", (
        Box("Complementary phase assignment",
            ("360 views at 1°: phase A = even views,", "phase B = odd views")),
        Box("Input context (5 channels)",
            ("θ − 3°, θ − 1°, θ + 1° projections", "+ sin θ, cos θ maps")),
        Box("U-Net",
            ("5 → 1 channel; predicts the projection", "at the target angle θ")),
        Box("Masked MSE loss",
            ("computed only on pixels with", "non-zero measured counts")),
        Box("Interleave",
            ("predicted + measured views", "→ dense 360-view sinogram")),
        Box("Reconstruction and scoring",
            ("FBP / SART → rmse_valid", "in the line-integral domain")),
    )),
    "zh": ("方法流程", (
        Box("互补相位划分",
            ("360 个视角 @1°：A 相位 = 偶数视角，", "B 相位 = 奇数视角")),
        Box("输入上下文（5 通道）",
            ("θ−3°、θ−1°、θ+1° 投影", "+ sin θ、cos θ 图")),
        Box("U-Net",
            ("5 → 1 通道，", "预测目标角 θ 的投影")),
        Box("掩膜 MSE 损失",
            ("仅在非零计数的像素上计算",)),
        Box("交错拼接",
            ("预测视角 + 已测视角", "→ 360 视角稠密正弦图")),
        Box("重建与评估",
            ("FBP / SART → 线积分域 rmse_valid",)),
    )),
}

FIG2: dict[str, tuple[str, tuple[Box, ...]]] = {
    "en": ("Data generation and split", (
        Box("Geant4 simulation",
            ("3 × 3 UO₂ rod array;", "1.5, 3.0, 4.5, 6.0 MeV;", "1° steps over 360°")),
        Box("Empty scan",
            ("same geometry without", "the assembly →", "normalisation and", "validity mask")),
        Box("Projection export",
            ("one 32 × 1200 array", "per view")),
        Box("Dual-phase split",
            ("phase A = even views,", "phase B = odd views;", "180 each")),
        Box("Stratified split",
            ("per energy:", "270 train / 45 val /", "45 test")),
    )),
    "zh": ("数据生成与划分", (
        Box("Geant4 仿真",
            ("3 × 3 UO₂ 棒阵；", "1.5/3.0/4.5/6.0 MeV；", "360° 内 1° 步进")),
        Box("空扫",
            ("去掉组件的同几何扫描", "→ 归一化与有效像素掩膜")),
        Box("投影导出",
            ("每个视角一个", "32 × 1200 数组")),
        Box("双相位划分",
            ("A 相位 = 偶数视角，", "B 相位 = 奇数视角，", "各 180")),
        Box("分层划分",
            ("每档能量：", "270 训练 / 45 验证 /", "45 测试")),
    )),
}

#: 两张图的几何。width 与 R6 指定的栏宽一致（8.5 cm / 17.5 cm）。
LAYOUTS: dict[str, Layout] = {
    "method_pipeline": Layout(width=8.5 / 2.54, margin_x=0.10, margin_top=0.10,
                              margin_bottom=0.10, box_w=8.5 / 2.54 - 0.20,
                              box_h=0.62, gap=0.22, title_h=0.30),
    "data_generation": Layout(width=17.5 / 2.54, margin_x=0.10, margin_top=0.10,
                              margin_bottom=0.10, box_w=(17.5 / 2.54 - 0.20 - 4 * 0.16) / 5,
                              box_h=1.02, gap=0.16, title_h=0.30,
                              title_frac=0.845, sub_frac=0.70, sub_va="top"),
}


def setup_style(lang: str) -> None:
    if lang not in LANGS:
        raise ValueError(f"未知 lang: {lang!r}")
    # font.family 直接给**列表**，才能启用 matplotlib 的逐字形回退
    # （U+2082 下标 2 在 Microsoft YaHei 里缺失，回退到 DejaVu Sans 才不出现豆腐块；
    #   若写成 family="sans-serif" + font.sans-serif=[...]，实测**不会**回退）
    rcParams.update({
        "font.family": FONTS[lang],
        "axes.unicode_minus": False,
        "figure.dpi": 110,
        "savefig.dpi": 200,
        "text.color": "black",
        "pdf.fonttype": 42,          # Type 42 = TrueType，期刊要求可编辑/可嵌入
        "ps.fonttype": 42,
    })


def has_cjk(s: str) -> bool:
    return any(unicodedata.name(ch, "").startswith("CJK") or
               "\u4e00" <= ch <= "\u9fff" for ch in s)


def add_box(ax, x0: float, y0: float, w: float, h: float, box: Box,
            title_frac: float, sub_frac: float, sub_va: str = "center") -> list:
    ax.add_patch(FancyBboxPatch(
        (x0, y0), w, h,
        boxstyle=f"round,pad=0,rounding_size={ROUND}",
        linewidth=EDGE_LW, edgecolor=EDGE, facecolor=FILL, zorder=2,
    ))
    t = ax.text(x0 + w / 2, y0 + h * title_frac, box.title,
                ha="center", va="center", fontsize=PT_BOX_TITLE,
                fontweight="bold", zorder=3)
    s = ax.text(x0 + w / 2, y0 + h * sub_frac, "\n".join(box.sub),
                ha="center", va=sub_va, fontsize=PT_BOX_SUB,
                linespacing=1.30, zorder=3)
    return [t, s]


def add_arrow(ax, p0: tuple[float, float], p1: tuple[float, float]) -> None:
    ax.add_patch(FancyArrowPatch(
        p0, p1, arrowstyle="-|>", mutation_scale=9,
        linewidth=ARROW_LW, color="black", shrinkA=0, shrinkB=0, zorder=1,
    ))


def draw_vertical(ax, boxes: tuple[Box, ...], lay: Layout, fig_title: str,
                  show_title: bool) -> tuple[list, list]:
    n = len(boxes)
    h = lay.margin_bottom + n * lay.box_h + (n - 1) * lay.gap \
        + lay.margin_top + (lay.title_h if show_title else 0.0)
    ax.set_xlim(0, lay.width)
    ax.set_ylim(0, h)
    ax.set_aspect("equal")
    ax.axis("off")

    texts: list = []
    if show_title:
        texts.append(ax.text(lay.width / 2, h - lay.margin_top - lay.title_h / 2,
                             fig_title, ha="center", va="center",
                             fontsize=PT_TITLE, fontweight="bold"))

    rects: list[tuple[float, float, float, float]] = []
    y = h - lay.margin_top - (lay.title_h if show_title else 0.0)
    for i, b in enumerate(boxes):
        y0 = y - lay.box_h
        texts += add_box(ax, lay.margin_x, y0, lay.box_w, lay.box_h, b,
                         lay.title_frac, lay.sub_frac, lay.sub_va)
        rects.append((lay.margin_x, y0, lay.box_w, lay.box_h))
        if i < n - 1:
            cx = lay.width / 2
            add_arrow(ax, (cx, y0), (cx, y0 - lay.gap))
        y = y0 - lay.gap
    return texts, rects


def draw_horizontal(ax, boxes: tuple[Box, ...], lay: Layout, fig_title: str,
                    show_title: bool) -> tuple[list, list]:
    n = len(boxes)
    h = lay.margin_bottom + lay.box_h + lay.margin_top + (lay.title_h if show_title else 0.0)
    ax.set_xlim(0, lay.width)
    ax.set_ylim(0, h)
    ax.set_aspect("equal")
    ax.axis("off")

    texts: list = []
    if show_title:
        texts.append(ax.text(lay.width / 2, h - lay.margin_top - lay.title_h / 2,
                             fig_title, ha="center", va="center",
                             fontsize=PT_TITLE, fontweight="bold"))

    y0 = lay.margin_bottom
    rects: list[tuple[float, float, float, float]] = []
    x = lay.margin_x
    for i, b in enumerate(boxes):
        texts += add_box(ax, x, y0, lay.box_w, lay.box_h, b,
                         lay.title_frac, lay.sub_frac, lay.sub_va)
        rects.append((x, y0, lay.box_w, lay.box_h))
        if i < n - 1:
            cy = y0 + lay.box_h / 2
            add_arrow(ax, (x + lay.box_w, cy), (x + lay.box_w + lay.gap, cy))
        x += lay.box_w + lay.gap
    return texts, rects


def build(fig_key: str, lang: str, show_title: bool):
    """返回 (fig, texts, lay, boxes)。几何对两种语言完全相同。"""
    content = (FIG1 if fig_key == "method_pipeline" else FIG2)[lang]
    fig_title, boxes = content
    lay = LAYOUTS[fig_key]

    if fig_key == "method_pipeline":
        h = lay.margin_bottom + len(boxes) * lay.box_h + (len(boxes) - 1) * lay.gap \
            + lay.margin_top + (lay.title_h if show_title else 0.0)
        fig, ax = plt.subplots(figsize=(lay.width, h))
        texts, rects = draw_vertical(ax, boxes, lay, fig_title, show_title)
    else:
        h = lay.margin_bottom + lay.box_h + lay.margin_top + (lay.title_h if show_title else 0.0)
        fig, ax = plt.subplots(figsize=(lay.width, h))
        texts, rects = draw_horizontal(ax, boxes, lay, fig_title, show_title)
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
    return fig, ax, texts, rects, boxes


# ------------------------------------------------------------------ 自检
#: R6 §6.4 禁用词（旧 scan.py 流程图的遗留概念，出现在图里即不可用）
FORBIDDEN: tuple[str, ...] = (
    "interference", "bright-line", "Bright-line", "correct_empty_proj",
    "Energy.data", "empty_proj", "empty-beam",
)

#: R6 §6.4 必须出现的术语（缺一即说明图与正文不一致）
REQUIRED: dict[str, dict[str, tuple[str, ...]]] = {
    "method_pipeline": {
        "en": ("sin θ", "cos θ", "Masked MSE", "FBP / SART", "θ − 3°"),
        "zh": ("sin θ", "cos θ", "MSE", "FBP / SART", "θ−3°"),
    },
    "data_generation": {
        "en": ("3 × 3 UO₂", "1° steps over 360°", "270 train / 45 val /", "180 each"),
        "zh": ("3 × 3 UO₂", "1° 步进", "270 训练 / 45 验证 /", "各 180"),
    },
}


def check_tokens(fig_key: str, lang: str, texts: list) -> None:
    joined = "\n".join(t.get_text() for t in texts)
    low = joined.lower()
    for bad in FORBIDDEN:
        if bad.lower() in low:
            raise AssertionError(f"[{fig_key}/{lang}] 出现禁用词 {bad!r}（R6 §6.4）")
    for must in REQUIRED[fig_key][lang]:
        if must not in joined:
            raise AssertionError(f"[{fig_key}/{lang}] 缺少必需术语 {must!r}（R6 §6.4）")


def verify(fig, ax, texts: list, rects: list, boxes: tuple[Box, ...], lay: Layout,
           fig_key: str, lang: str) -> dict:
    """实测每个 Text 的渲染外接框，断言：框内、字号达标、标题与副标题不重叠、
    英文版零中文、图内无 Fig 编号。返回可打印的自检表。"""
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    inv = ax.transData.inverted()

    def to_data(bb):
        """把像素 bbox 转成数据坐标 (x0, y0, x1, y1)。"""
        (x0, y0) = inv.transform((bb.x0, bb.y0))
        (x1, y1) = inv.transform((bb.x1, bb.y1))
        return (min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1))

    report = {"lang": lang, "fig": fig_key, "boxes": [], "warnings": []}

    # 0) R6 §6.4 词表校验
    check_tokens(fig_key, lang, texts)

    # 1) 英文版不得出现任何 CJK 字符（R6 §6.3）
    if lang == "en":
        for t in texts:
            if has_cjk(t.get_text()):
                raise AssertionError(f"英文版出现中文字符: {t.get_text()!r}")

    # 2) 图内不得出现 Fig 编号（R6 §6.3）
    for t in texts:
        label = t.get_text()
        if "Fig." in label or "Fig " in label:
            raise AssertionError(f"图内出现 Fig 编号: {label!r}")

    # 3) 字号下限（印刷尺寸 == figure 尺寸，故 pt 即印刷 pt）
    for t in texts:
        if t.get_fontsize() < MIN_PT - 1e-9:
            raise AssertionError(f"字号 {t.get_fontsize()} < {MIN_PT} pt: {t.get_text()!r}")

    # 4) 逐框：两个 Text 都必须在框内，且互不重叠
    #    texts 的顺序是 [图标题(可选)] + 每框(标题, 副标题)
    off = len(texts) - 2 * len(boxes)
    pad = 0.03                                    # inch，左右内边距
    for i, b in enumerate(boxes):
        bx, by, bw, bh = rects[i]
        t_title, t_sub = texts[off + 2 * i], texts[off + 2 * i + 1]
        tx0, ty0, tx1, ty1 = to_data(t_title.get_window_extent(renderer=renderer))
        sx0, sy0, sx1, sy1 = to_data(t_sub.get_window_extent(renderer=renderer))

        w_text = max(tx1 - tx0, sx1 - sx0)
        limit_w = bw - 2 * pad
        ok_w = w_text <= limit_w + 1e-9

        # 垂直：两个文字块都要在 [by, by+bh] 内，且标题底 > 副标题顶
        ok_v = (ty0 >= by - 1e-9 and ty1 <= by + bh + 1e-9
                and sy0 >= by - 1e-9 and sy1 <= by + bh + 1e-9)
        ok_gap = ty0 >= sy1 - 1e-9
        ok = ok_w and ok_v and ok_gap

        report["boxes"].append({
            "idx": i + 1, "title": b.title,
            "w_text_in": round(w_text, 4), "limit_in": round(limit_w, 4),
            "y_title": (round(ty0 - by, 4), round(ty1 - by, 4)),
            "y_sub": (round(sy0 - by, 4), round(sy1 - by, 4)),
            "box_h_in": round(bh, 4),
            "fits": bool(ok),
        })
        if not ok_w:
            report["warnings"].append(
                f"框{i + 1}「{b.title}」文字宽 {w_text:.4f} > 可用 {limit_w:.4f} in")
        if not ok_v:
            report["warnings"].append(
                f"框{i + 1}「{b.title}」文字越出框高 {bh:.3f} in")
        if not ok_gap:
            report["warnings"].append(
                f"框{i + 1}「{b.title}」标题与副标题重叠")
    return report


def save(fig, fig_key: str, lang: str, dpi: int, want_pdf: bool) -> list[Path]:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    out: list[Path] = []
    p_png = FIG_DIR / f"{fig_key}_{SUFFIX[lang]}.png"
    fig.savefig(p_png, dpi=dpi, bbox_inches="tight", pad_inches=0.02,
                facecolor="white")
    out.append(p_png)
    if want_pdf:
        p_pdf = FIG_DIR / f"{fig_key}_{SUFFIX[lang]}.pdf"
        # 去掉 CreationDate / ModDate，否则每次重跑 PDF 都变、git 永远脏
        fig.savefig(p_pdf, bbox_inches="tight", pad_inches=0.02, facecolor="white",
                    metadata={"CreationDate": None, "ModDate": None,
                              "Creator": "scripts/plot_method_figures.py",
                              "Producer": "matplotlib"})
        out.append(p_pdf)
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="R6：方法流程图 + 数据生成图（中英双语）")
    ap.add_argument("--dpi", type=int, default=300,
                    help="PNG 分辨率。用户指定 200，R6 要求 >=300，默认取 300")
    ap.add_argument("--no-title", action="store_true", help="不画图内标题（只留框与箭头）")
    ap.add_argument("--no-pdf", action="store_true", help="不额外输出矢量 PDF")
    args = ap.parse_args(argv)

    show_title = not args.no_title
    want_pdf = not args.no_pdf
    all_paths: list[Path] = []
    reports: list[dict] = []

    for fig_key in ("method_pipeline", "data_generation"):
        for lang in LANGS:
            setup_style(lang)
            fig, ax, texts, rects, boxes = build(fig_key, lang, show_title)
            rep = verify(fig, ax, texts, rects, boxes, LAYOUTS[fig_key], fig_key, lang)
            paths = save(fig, fig_key, lang, args.dpi, want_pdf)
            plt.close(fig)
            reports.append(rep)
            all_paths += paths

    # ---------------- 打印自检表
    print("=" * 92)
    for rep in reports:
        lay = LAYOUTS[rep["fig"]]
        print(f"[{rep['fig']}] lang={rep['lang']}  box_w={lay.box_w:.3f} in  "
              f"框数={len(rep['boxes'])}  字号: 标题{PT_BOX_TITLE}/副标题{PT_BOX_SUB} pt "
              f"(下限 {MIN_PT})")
        for b in rep["boxes"]:
            flag = "OK  " if b["fits"] else "OVER"
            print(f"   {flag} 框{b['idx']} {b['title'][:26]:<26} "
                  f"宽 {b['w_text_in']:.4f}/{b['limit_in']:.4f} in  "
                  f"标题 y {b['y_title']}  副标题 y {b['y_sub']}  框高 {b['box_h_in']}")
        if rep["warnings"]:
            print("   !!! " + "; ".join(rep["warnings"]))
    print("-" * 92)
    over = [w for r in reports for w in r["warnings"]]
    print(f"溢出告警：{len(over)} 条")
    for p in all_paths:
        print(f"已写 {p.relative_to(ROOT)}  ({p.stat().st_size:,} B)")
    print("=" * 92)
    return 1 if over else 0


if __name__ == "__main__":
    raise SystemExit(main())
