#!/usr/bin/env python
# -*- coding: utf-8 -*-
r"""
scripts/plot_recon_triptych.py —— R3：重建切片三联图（sparse / CNN 补全 / dense GT）

规格来源
--------
`docs/handoff/algo_requests.md` 的 **R3 小节**（论文 agent 撰写）：
三联图，左 sparse（未补全）、中 CNN 补全、右 dense GT；**同一角度、同一色窗**；
英文标注，论文风格（无中文、标题 ≤60 字符）。
建议附残差图 `recon_residual_paper.png`；每个能量各一张（1.5 MeV 最关键）。

产出
----
  docs/figs/20261022_2352_cnn_bs8_lr1e-4/recon_compare_paper.png
      **主图**：1.5 MeV（全文最差档）放大三联 1×3，裁剪到棒阵区，双栏宽 6.89 in。
  docs/figs/20261022_2352_cnn_bs8_lr1e-4/recon_compare_energies_paper.png
      四档全览 4×3（每档一行：sparse / CNN completed / dense reference），全 FOV。
  docs/figs/20261022_2352_cnn_bs8_lr1e-4/recon_residual_paper.png
      残差图 2×4（每行 2 档 × 2 类型）：|sparse − dense| 与 |CNN − dense|。
  以上三图均另存同名 .pdf（矢量）。

**"同一角度"的落实**
--------------------
本项目的重建是**单行平面重建**（探测器第 16 行，1200 个探测元），
不存在"选哪个角度切片"的问题。真正需要对齐的是：
  1. 同一个探测器行（RECON_ROW = 16）——三者都取这一行；
  2. 同一个坐标系与同一个 FOV 圆；
  3. 同一色窗（同一行内三图共用 vmin/vmax）。
  故 R3 的"同一角度"在本项目等价于"同一探测器行 + 同色窗"。

**色窗**
--------
用 dense 参考的稳健分位 `[p0.5, p99.5]`（FOV 内），不是 min/max：
1.5 MeV 的 dense 在像素 (600,600) 有孤立热点（0.2303，而 p99.9 只有 0.0248，
是 p99.9 的 9.3 倍），min/max 会把整幅图压成全黑。热点在**四档都有**
（最大点都在中心 30 px 内、dense 参考自带），是线积分集中点，不是 CNN 伪影。

**尺寸即印刷尺寸**
------------------
figure 的 inches 直接取最终印刷宽度（双栏 17.5 cm = 6.890 in），
所以字号 pt 就是印刷 pt，可被直接断言 ≥ 7 pt。

用法
----
  conda run -n ct_full python scripts/plot_recon_triptych.py
  conda run -n ct_full python scripts/plot_recon_triptych.py --dpi 300
"""

from __future__ import annotations

import argparse
import sys
import textwrap
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt                       # noqa: E402
import numpy as np                                    # noqa: E402
from matplotlib import rcParams                       # noqa: E402
from matplotlib.patches import Circle                 # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
for p in (ROOT, ROOT / "scripts"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import baseline_recon as br                           # noqa: E402

RUN_ID = "20261022_2352_cnn_bs8_lr1e-4"
FIG_DIR = ROOT / "docs" / "figs" / RUN_ID
ENERGIES = ("1.5", "3", "4.5", "6")
PHASE = "A"          # 显示用：实测 A 相位、预测 B 相位

FIG_W_IN = 17.5 / 2.54          # 双栏宽 6.890 in
PT_PANEL, PT_COLH, PT_ROWH, MIN_PT = 7.0, 8.0, 8.0, 7.0
CAP_WRAP = 112                  # 图注每行字符数（按 6.89 in 宽、7.6 pt 估）

ZOOM = slice(180, 1020)         # 棒阵显著区实测 x/y ∈ [255,945]，取 [180,1020] 留边

CAP_MAIN = (
    "Zoom of the rod array at 1.5 MeV, the worst case in the study. The same grey window is "
    "applied to all three panels. The sparse reconstruction carries angle-dependent streak "
    "artefacts across the whole field of view, whereas the CNN-completed reconstruction "
    "reproduces the rod boundaries of the dense reference. Grey windows are set to [p1, p99] "
    "of the dense reference of each energy; the data itself is normalised, the window is "
    "chosen to make details visible.")
CAP_ALL = (
    "Reconstruction of the 3 x 3 UO2 rod array from detector row 16 (1200 bins). Left: "
    "filtered back-projection from the 180 measured views only. Centre: the same FBP after "
    "the CNN completes the 180 missing views. Right: FBP from all 360 measured views (dense "
    "reference). Each row shares one grey window. Grey windows are set to [p1, p99] of the "
    "dense reference of each energy; the data itself is normalised, the window is chosen "
    "to make details visible.")
CAP_RES = (
    "Absolute reconstruction error inside the field of view, against the dense reference of the "
    "same energy. The sparse and CNN panels of a given energy share one colour scale, set to the "
    "99.5 percentile of the larger of the two error maps, so the two can be compared directly. "
    "The concentric rings of the sparse reconstruction are the signature of the 2 degree "
    "angular undersampling; they are removed by the completion.")


def font_setup() -> None:
    rcParams.update({
        "font.family": ["Times New Roman"],   # 英文版；直接给列表才逐字形回退
        "axes.unicode_minus": True,
        "savefig.dpi": 300,
    })


def load(e: str, kind: str) -> np.ndarray:
    if kind == "dense":
        p = ROOT / "runs" / f"{RUN_ID}_cnn_recon" / "recon" / f"dense_{e}_fbp_hamming.npy"
    elif kind == "cnn":
        p = ROOT / "runs" / f"{RUN_ID}_cnn_recon" / "recon" / f"cnn_{e}_{PHASE}.npy"
    elif kind == "sparse":
        p = ROOT / "runs" / f"{RUN_ID}_fbp_sparse" / "recon" / f"fbp_{e}_{PHASE}.npy"
    else:
        raise ValueError(kind)
    return np.load(p).astype(np.float64)


def window(dense: np.ndarray, fov: np.ndarray) -> tuple[float, float]:
    v = dense[fov]
    return float(np.percentile(v, 1.0)), float(np.percentile(v, 99.0))


def n_cap_lines(text: str) -> int:
    return len(textwrap.wrap(text, width=CAP_WRAP))


def save(fig, stem: str, dpi: int) -> list[Path]:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    out: list[Path] = []
    p_png = FIG_DIR / f"{stem}.png"
    fig.savefig(p_png, dpi=dpi, bbox_inches="tight", pad_inches=0.04, facecolor="white")
    out.append(p_png)
    p_pdf = FIG_DIR / f"{stem}.pdf"
    fig.savefig(p_pdf, bbox_inches="tight", pad_inches=0.04, facecolor="white",
                metadata={"CreationDate": None, "ModDate": None,
                          "Creator": "scripts/plot_recon_triptych.py",
                          "Producer": "matplotlib"})
    out.append(p_pdf)
    plt.close(fig)
    return out


#: 2026-10-23 用户裁定：图内不再印说明性长句（与 LaTeX caption 重复、排版难看）。
#: 由 main 的 --with-caption 打开；默认关闭，caption 只在 LaTeX 里出现。
CAPTION_ON = False


def cap_space(text: str, fontsize: float) -> float:
    """图注占用的高度（英寸）；关闭时为 0，图面自动收紧。"""
    if not CAPTION_ON:
        return 0.0
    return n_cap_lines(text) * fontsize * 1.3 / 72.0


def put_caption(fig, text: str, fontsize: float) -> None:
    """图注：按**英寸**排行距，避免与图面重叠（fig.text 的 y 是 figure 比例）。"""
    lines = textwrap.wrap(text, width=CAP_WRAP)
    fig_h = fig.get_size_inches()[1]
    line_h_in = fontsize * 1.3 / 72.0
    y0 = 0.010 / fig_h
    for k, line in enumerate(lines):
        fig.text(0.5, y0 + (len(lines) - 1 - k) * line_h_in / fig_h, line,
                 ha="center", va="bottom", fontsize=fontsize)


def style_panel(ax, *, circle: bool = True) -> None:
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_box_aspect(1)
    if circle:
        ax.add_patch(Circle((600, 600), 600, fill=False, edgecolor="white",
                            linewidth=0.5, alpha=0.85))
    for s in ax.spines.values():
        s.set_linewidth(0.5); s.set_color("#444444")


# ------------------------------------------------------------------ 主图
def fig_main(dpi: int) -> list[Path]:
    fov = br.fov_mask()
    e = "1.5"
    dense, cnn, sparse = load(e, "dense"), load(e, "cnn"), load(e, "sparse")
    vmin, vmax = window(dense, fov)

    panel = 2.05                                   # in；3×2.05 + 边距 ≈ 6.89
    cap_h = cap_space(CAP_MAIN, PT_PANEL + 0.6)
    fig_h = 0.55 + panel + 0.26 + cap_h + 0.16
    fig, axes = plt.subplots(1, 3, figsize=(FIG_W_IN, fig_h))
    titles = ["(a) Sparse, 180 views", "(b) CNN completed, 360 views",
              "(c) Dense reference, 360 views"]
    for ci, img in enumerate((sparse, cnn, dense)):
        ax = axes[ci]
        ax.imshow(img[ZOOM, ZOOM], cmap="gray", vmin=vmin, vmax=vmax,
                  interpolation="nearest")
        style_panel(ax, circle=False)
        ax.set_title(titles[ci], fontsize=PT_COLH, pad=4)
    fig.subplots_adjust(left=0.015, right=0.995, top=1.0 - 0.42 / fig_h,
                        bottom=(cap_h + 0.10) / fig_h, wspace=0.04)
    if CAPTION_ON:
        put_caption(fig, CAP_MAIN, PT_PANEL + 0.6)
    return save(fig, "recon_compare_paper", dpi)


# ------------------------------------------------------------------ 四档全览
def fig_all(dpi: int) -> list[Path]:
    fov = br.fov_mask()
    panel = 1.34
    cap_h = cap_space(CAP_ALL, PT_PANEL + 0.6)
    fig_h = 0.55 + len(ENERGIES) * (panel + 0.24) + cap_h + 0.16
    fig, axes = plt.subplots(len(ENERGIES), 3, figsize=(FIG_W_IN, fig_h))
    titles = ["(a) Sparse, 180 views", "(b) CNN completed, 360 views",
              "(c) Dense reference, 360 views"]
    for ri, e in enumerate(ENERGIES):
        dense, cnn, sparse = load(e, "dense"), load(e, "cnn"), load(e, "sparse")
        vmin, vmax = window(dense, fov)
        for ci, img in enumerate((sparse, cnn, dense)):
            ax = axes[ri, ci]
            ax.imshow(img, cmap="gray", vmin=vmin, vmax=vmax, interpolation="nearest")
            style_panel(ax)
            if ri == 0:
                ax.set_title(titles[ci], fontsize=PT_COLH, pad=4)
            if ci == 0:
                ax.text(-0.06, 0.5, f"{e} MeV", transform=ax.transAxes,
                        ha="center", va="center", fontsize=PT_ROWH, rotation=90)
    fig.subplots_adjust(left=0.05, right=0.995, top=1.0 - 0.42 / fig_h,
                        bottom=(cap_h + 0.10) / fig_h, wspace=0.05, hspace=0.26)
    if CAPTION_ON:
        put_caption(fig, CAP_ALL, PT_PANEL + 0.6)
    return save(fig, "recon_compare_energies_paper", dpi)


# ------------------------------------------------------------------ 残差图
def fig_residual(dpi: int) -> list[Path]:
    fov = br.fov_mask()
    panel = 1.55
    cap_h = cap_space(CAP_RES, PT_PANEL + 0.6)
    fig_h = 0.55 + 2 * (panel + 0.24) + cap_h + 0.16
    fig, axes = plt.subplots(2, 4, figsize=(FIG_W_IN, fig_h))
    for k, e in enumerate(ENERGIES):
        r, c = divmod(k, 2)                 # 第 k 档 → 第 r 行、第 c 组（每组 2 列）
        dense, cnn, sparse = load(e, "dense"), load(e, "cnn"), load(e, "sparse")
        errs = [np.abs(sparse - dense), np.abs(cnn - dense)]
        # 同一档的 sparse / CNN 两格**共用一个色标**（取两者 99.5 分位的较大值），
        # 否则两格各自定标会把"变好了"这件事在视觉上抹平
        emax = max(float(np.nanpercentile(np.where(fov, er, np.nan), 99.5))
                   for er in errs)
        for j, er in enumerate(errs):
            ax = axes[r, 2 * c + j]
            err_in = np.where(fov, er, np.nan)
            ax.imshow(err_in, cmap="hot", vmin=0.0, vmax=emax,
                      interpolation="nearest")
            style_panel(ax)
            if r == 0:
                ax.set_title("|Sparse - dense|" if j == 0 else "|CNN - dense|",
                             fontsize=PT_COLH, pad=4)
            if j == 0:
                ax.text(-0.07, 0.5, f"{e} MeV", transform=ax.transAxes,
                        ha="center", va="center", fontsize=PT_ROWH, rotation=90)
            ax.text(0.5, -0.035, f"RMSE {np.sqrt(np.nanmean(err_in ** 2)):.2e}",
                    transform=ax.transAxes, ha="center", va="top", fontsize=PT_PANEL)
    fig.subplots_adjust(left=0.05, right=0.995, top=1.0 - 0.42 / fig_h,
                        bottom=(cap_h + 0.10) / fig_h, wspace=0.10, hspace=0.26)
    if CAPTION_ON:
        put_caption(fig, CAP_RES, PT_PANEL + 0.6)
    return save(fig, "recon_residual_paper", dpi)


def main(argv: list[str] | None = None) -> int:
    global CAPTION_ON
    ap = argparse.ArgumentParser(description="R3：重建切片三联图 + 残差图")
    ap.add_argument("--dpi", type=int, default=300)
    ap.add_argument("--with-caption", action="store_true",
                    help="在图内印长句图注（默认关闭，caption 交给 LaTeX）")
    ap.add_argument("--run-id", default=None, help="覆盖 RUN_ID（图输出到 docs/figs/<run_id>/）")
    args = ap.parse_args(argv)
    CAPTION_ON = bool(args.with_caption)
    global RUN_ID, FIG_DIR
    if args.run_id:
        RUN_ID = args.run_id
        FIG_DIR = ROOT / "docs" / "figs" / RUN_ID
        FIG_DIR.mkdir(parents=True, exist_ok=True)
    font_setup()
    for fn in (fig_main, fig_all, fig_residual):
        outs = fn(args.dpi)
        print("  " + "  ".join(str(p.relative_to(ROOT)) for p in outs))
    for stem in ("recon_compare_paper", "recon_compare_energies_paper",
                 "recon_residual_paper"):
        for ext in ("png", "pdf"):
            p = FIG_DIR / f"{stem}.{ext}"
            if not p.is_file() or p.stat().st_size == 0:
                raise RuntimeError(f"输出缺失或为空: {p}")
    print("OK  R3 三图已生成（英文版，无中文）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
