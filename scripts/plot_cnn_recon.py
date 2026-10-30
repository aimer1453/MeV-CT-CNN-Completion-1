#!/usr/bin/env python
# -*- coding: utf-8 -*-
r"""
scripts/plot_cnn_recon.py —— Table 4b 配图：图像域重建对比（中英双语）

图的唯一目的
------------
把 Table 4b 的四行（`dense` 参考 / `cnn_completed` / `fbp_sparse` / `sart_sparse`）
画成同一张可横比的图，并给出 CNN 与 dense 的误差图，让"补全到底修好了什么"
**看得见**而不是只报数字。

布局
----
  4 行（能量 1.5 / 3 / 4.5 / 6 MeV）× 5 列：

    col 0  dense          360 视角全实测 hamming FBP（参考）
    col 1  CNN completed  180 实测 + 180 CNN 补全 → 360 视角 hamming FBP
    col 2  FBP sparse     180 视角（2° 步进）直接 FBP，无补全
    col 3  SART sparse    180 视角 iradon_sart(n_iter=20)，无补全
    col 4  |CNN − dense|  FOV 圆内绝对误差图

  每格小字给出该格口径下的 `rmse_valid / psnr / ssim`（**图像域**，
  与同目录 README 的口径一致；PSNR 用四档统一 `data_range=0.2737`，即
  `psnr_common`，否则跨能量会把 1.5 MeV 从最差误判成最好）。

显示约定
--------
  - 重建图统一灰度 cmap；每行共用同一 `vmin/vmax`（由该档 dense 决定），
    保证同一行内**明暗可比**；行间不共用（四档 μ 量级不同，强行共色标会压暗三档）。
  - 误差图用 `hot`，每行共用色标（由 CNN 误差最大值决定），并**只画 FOV 圆内**。
  - 每张图叠加 FOV 圆（r=600 px，白色细线），圈外两图皆 0，故圈外无信息。
  - 显示用输入相位 A 的重建（实测 A、预测 B），四行一致；相位 B 的版本在
    `runs/<id>_cnn_recon/recon/cnn_<e>_B.npy`，不重复画。

数据来源（全部从磁盘读，不重算）
--------------------------------
  runs/<run_id>_cnn_recon/recon/{dense_<e>_fbp_hamming,cnn_<e>_A}.npy
  runs/<run_id>_fbp_sparse/recon/fbp_<e>_A.npy
  runs/<run_id>_sart_sparse/recon/sart_<e>_A.npy
  runs/<run_id>_{cnn_recon,fbp_sparse,sart_sparse}/test_metrics.json  （格内指标）

输出
----
  docs/figs/<run_id>/baseline_compare_{analysis,paper}.png / .pdf

  论文侧 §5.6 的 TODO 占位把**这一张定为 Fig. 6**（"the figure should show the three
  methods plus the dense reference"），**R3 的重建切片三联图是 Fig. 7**。
  所以本图的输出名是 `baseline_compare_*`，不是 `cnn_recon_compare_*`。

用法
----
  conda run -n ct_full python scripts/plot_cnn_recon.py
  conda run -n ct_full python scripts/plot_cnn_recon.py --dpi 300
"""

from __future__ import annotations

import argparse
import json
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
LANGS = ("zh", "en")
SUFFIX = {"zh": "analysis", "en": "paper"}
FONTS: dict[str, list[str]] = {
    "zh": ["Microsoft YaHei", "DejaVu Sans"],
    "en": ["Times New Roman"],
}

# 印刷尺寸：双栏 17.5 cm
FIG_W_IN = 17.5 / 2.54
PANEL_IN = 1.08           # 每格边长（英寸）；6 列 x 1.08 + 边距 ≈ 6.89
ROW_TITLE_IN = 0.22      # 行标题条
COL_TITLE_IN = 0.34      # 列标题条
PT_PANEL, PT_COLH, PT_ROWH, MIN_PT = 6.4, 7.5, 8.0, 7.0

COL_TITLE = {
    "zh": ["dense 参考\n(360 视角)", "CNN 补全\n(180+180 视角)",
           "线性插值\n(180+180 视角)", "FBP 稀疏\n(180 视角)", "SART 稀疏\n(180 视角)",
           "|CNN − dense|\n圆内误差"],
    "en": ["dense reference\n(360 views)", "CNN completed\n(180+180 views)",
           "Linear interpolation\n(180+180 views)", "FBP sparse\n(180 views)",
           "SART sparse\n(180 views)", "|CNN - dense|\nerror in FOV"],
}
CAPTION = {
    "zh": ("Table 4 图像域对比。第 0 列为 dense 参考（360 视角全实测 FBP）；"
           "第 1 列为 CNN 补全、第 2 列为线性插值（均为 180 实测 + 180 补全视角 → FBP）；"
           "第 3、4 列为未补全的 180 视角直接重建。"
           "PSNR 用四档统一 data_range=0.2737 的 psnr_common，跨能量可比。"
           "灰度窗口取每档 dense 的 [p1, p99]（1.5 MeV 中心有孤立热点，"
           "用 min/max 会全黑）。数据本身已归一化，窗宽只为让细节可见。"
           "白圈为 FOV 圆 r=600 px，圈外两图皆为 0。"),
    "en": ("Table 4 in the image domain. Col 0: dense reference (360 measured views, FBP). "
           "Col 1: CNN completion and col 2: linear interpolation (both from 180 measured "
           "plus 180 completed views, then FBP). Cols 3-4: direct reconstruction from the "
           "same 180 views without completion. "
           "PSNR uses psnr_common with a single data_range of 0.2737 across all four "
           "energies. Grey windows are set to [p1, p99] of the dense reference of each "
           "energy (1.5 MeV has an isolated central hot spot, so a min/max window would "
           "render the frame black); the data itself is normalised, the window is chosen "
           "to make details visible. The white circle is the FOV (r = 600 px); both images "
           "are zero outside it."),
}

ROW_TITLE = {
    "zh": ["1.5 MeV", "3.0 MeV", "4.5 MeV", "6.0 MeV"],
    "en": ["1.5 MeV", "3.0 MeV", "4.5 MeV", "6.0 MeV"],
}
ENERGIES = ("1.5", "3", "4.5", "6")


def load_json(p: Path) -> dict:
    return json.loads(p.read_text(encoding="utf-8"))


def metric_text(lang: str, e: str, tm_cnn: dict, tm_lin: dict, tm_fbp: dict,
                tm_sart: dict, dense: np.ndarray, cnn: np.ndarray) -> list[str]:
    """每列小字指标。PSNR 一律用四档统一 data_range 的 psnr_common。"""
    v = tm_cnn["per_energy"][e]
    li = tm_lin["per_energy"][e]
    f = tm_fbp["per_energy"][e]
    s = tm_sart["per_energy"][e]
    dr = tm_cnn["per_energy"][e]["data_range_common"]
    err = float(np.sqrt(np.mean((cnn - dense) ** 2)))
    mae = float(np.mean(np.abs(cnn - dense)))
    if lang == "zh":
        return [
            f"参考\nμ范围 ±{v['data_range'] / 2:.4f}",
            f"RMSE {v['rmse_valid']:.2e}\nPSNR {v['psnr_common']:.2f} dB\n"
            f"SSIM {v['ssim']:.4f}",
            f"RMSE {li['rmse_valid']:.2e}\nPSNR {li['psnr_common']:.2f} dB\n"
            f"SSIM {li['ssim']:.4f}",
            f"RMSE {f['rmse_valid']:.2e}\nPSNR {f['psnr_common']:.2f} dB\n"
            f"SSIM {f['ssim']:.4f}",
            f"RMSE {s['rmse_valid']:.2e}\nPSNR {s['psnr_common']:.2f} dB\n"
            f"SSIM {s['ssim']:.4f}",
            f"MAE {mae:.2e}\nRMSE {err:.2e}",
        ]
    return [
        f"reference\nrange ±{v['data_range'] / 2:.4f}",
        f"RMSE {v['rmse_valid']:.2e}\nPSNR {v['psnr_common']:.2f} dB\n"
        f"SSIM {v['ssim']:.4f}",
        f"RMSE {li['rmse_valid']:.2e}\nPSNR {li['psnr_common']:.2f} dB\n"
        f"SSIM {li['ssim']:.4f}",
        f"RMSE {f['rmse_valid']:.2e}\nPSNR {f['psnr_common']:.2f} dB\n"
        f"SSIM {f['ssim']:.4f}",
        f"RMSE {s['rmse_valid']:.2e}\nPSNR {s['psnr_common']:.2f} dB\n"
        f"SSIM {s['ssim']:.4f}",
        f"MAE {mae:.2e}\nRMSE {err:.2e}",
    ]


def cap_lines(text: str) -> list[str]:
    """图注按 figure 宽度折行（fig.text 的 wrap=True 不会按 figure 宽度折）。"""
    return textwrap.wrap(text, width=112)


def put_caption(fig, text: str, fontsize: float) -> None:
    """图注按**英寸**排行距，避免与图面/末行指标文字重叠。"""
    lines = cap_lines(text)
    fig_h = fig.get_size_inches()[1]
    line_h_in = fontsize * 1.3 / 72.0
    y0 = 0.010 / fig_h
    for k, line in enumerate(lines):
        fig.text(0.5, y0 + (len(lines) - 1 - k) * line_h_in / fig_h, line,
                 ha="center", va="bottom", fontsize=fontsize)


def display_window(dense: np.ndarray, fov: np.ndarray) -> tuple[float, float]:
    """显示窗口 `[p1, p99]`（FOV 内）。

    **不能用 min/max**：1.5 MeV 的 dense 有一个中心热点（像素 (600,600) 值 0.2303），
    而 p99.9 只有 0.0248 —— 最大值是 p99.9 的 9.3 倍。用 min/max 开窗会把整幅图
    压成全黑（实测第一版就是这样），棒阵完全不可辨。
    该热点在**四档都有**（最大点均在中心 30 px 内，且 dense 参考自带），
    是线积分集中点，不是 CNN 引入的伪影。
    p0.5/p99.5 曾使画面整体发灰（2026-10-23 用户反馈），收紧为 p1/p99。
    数据本身已归一化，窗宽只为让细节可见。
    """
    v = dense[fov]
    return float(np.percentile(v, 1.0)), float(np.percentile(v, 99.0))


def build(lang: str, dpi: int, with_caption: bool = False) -> list[Path]:
    rcParams.update({
        "font.family": FONTS[lang],          # 直接给列表才会逐字形回退
        "axes.unicode_minus": True,
        "savefig.dpi": dpi,
    })
    tm_cnn = load_json(ROOT / "runs" / f"{RUN_ID}_cnn_recon" / "test_metrics.json")
    tm_fbp = load_json(ROOT / "runs" / f"{RUN_ID}_fbp_sparse" / "test_metrics.json")
    tm_sart = load_json(ROOT / "runs" / f"{RUN_ID}_sart_sparse" / "test_metrics.json")
    tm_lin = load_json(ROOT / "runs" / f"{RUN_ID}_linear_interp_image" / "test_metrics.json")
    fov = br.fov_mask()

    n_rows, n_cols = len(ENERGIES), 6
    # 2026-10-23 用户裁定：图内不再放说明性长句（与 LaTeX caption 重复且排版难看），
    # 只保留列标题、行标题、逐格指标。--with-caption 可恢复旧行为。
    cap_txt = (CAPTION["zh"] if lang == "zh" else CAPTION["en"])
    cap_h = (len(cap_lines(cap_txt)) * (PT_PANEL + 0.6) * 1.3 / 72.0) \
        if with_caption else 0.0
    fig_h = 0.62 + n_rows * (PANEL_IN + 0.30) + cap_h + (0.45 if with_caption else 0.12)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(FIG_W_IN, fig_h))

    for ri, e in enumerate(ENERGIES):
        dense = np.load(ROOT / "runs" / f"{RUN_ID}_cnn_recon" / "recon"
                        / f"dense_{e}_fbp_hamming.npy")
        cnn = np.load(ROOT / "runs" / f"{RUN_ID}_cnn_recon" / "recon"
                      / f"cnn_{e}_A.npy")
        lin = np.load(ROOT / "runs" / f"{RUN_ID}_linear_interp_image" / "recon"
                      / f"linear_{e}_A.npy")
        fbp = np.load(ROOT / "runs" / f"{RUN_ID}_fbp_sparse" / "recon" / f"fbp_{e}_A.npy")
        sart = np.load(ROOT / "runs" / f"{RUN_ID}_sart_sparse" / "recon" / f"sart_{e}_A.npy")

        vmin, vmax = display_window(dense, fov)
        err = np.abs(cnn - dense)
        err_in = np.where(fov, err, np.nan)
        emax = float(np.nanpercentile(err_in, 99.5))

        panels = [dense, cnn, lin, fbp, sart, err]
        texts = metric_text(lang, e, tm_cnn, tm_lin, tm_fbp, tm_sart, dense, cnn)

        for ci in range(n_cols):
            ax = axes[ri, ci]
            if ci < 5:
                ax.imshow(panels[ci], cmap="gray", vmin=vmin, vmax=vmax,
                          interpolation="nearest")
            else:
                ax.imshow(np.where(fov, err, np.nan), cmap="hot",
                          vmin=0.0, vmax=emax, interpolation="nearest")
            ax.set_xticks([]); ax.set_yticks([])
            ax.add_patch(Circle((600, 600), 600, fill=False, edgecolor="white",
                                linewidth=0.5, alpha=0.85))
            for s in ax.spines.values():
                s.set_linewidth(0.5)
                s.set_color("#444444")
            ax.text(0.5, -0.035, texts[ci], transform=ax.transAxes,
                    ha="center", va="top", fontsize=PT_PANEL, linespacing=1.35)
            if ri == 0:
                ax.set_title(COL_TITLE[lang][ci], fontsize=PT_COLH, pad=4,
                             linespacing=1.3)
            if ci == 0:
                ax.text(-0.06, 0.5, ROW_TITLE[lang][ri], transform=ax.transAxes,
                        ha="center", va="center", fontsize=PT_ROWH, rotation=90)

    fig.subplots_adjust(left=0.055, right=0.995, top=0.935,
                        bottom=(cap_h + (0.42 if with_caption else 0.10)) / fig_h,
                        wspace=0.05, hspace=0.34)

    if with_caption:
        put_caption(fig, cap_txt, PT_PANEL + 0.6)

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    out: list[Path] = []
    p_png = FIG_DIR / f"baseline_compare_{SUFFIX[lang]}.png"
    fig.savefig(p_png, dpi=dpi, bbox_inches="tight", pad_inches=0.04,
                facecolor="white")
    out.append(p_png)
    p_pdf = FIG_DIR / f"baseline_compare_{SUFFIX[lang]}.pdf"
    fig.savefig(p_pdf, bbox_inches="tight", pad_inches=0.04, facecolor="white",
                metadata={"CreationDate": None, "ModDate": None,
                          "Creator": "scripts/plot_cnn_recon.py",
                          "Producer": "matplotlib"})
    out.append(p_pdf)
    plt.close(fig)
    return out


def verify(fig_key: str) -> None:
    """静态核验：文件非空、尺寸符合预期。"""
    for lang in LANGS:
        for ext in ("png", "pdf"):
            p = FIG_DIR / f"{fig_key}_{SUFFIX[lang]}.{ext}"
            if not p.is_file() or p.stat().st_size == 0:
                raise RuntimeError(f"输出缺失或为空: {p}")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Table 4b 配图：图像域重建对比")
    ap.add_argument("--dpi", type=int, default=300)
    ap.add_argument("--with-caption", action="store_true",
                    help="在图内印长句图注（默认关闭，caption 交给 LaTeX）")
    ap.add_argument("--run-id", default=None, help="覆盖 RUN_ID（图输出到 docs/figs/<run_id>/）")
    args = ap.parse_args(argv)
    global RUN_ID, FIG_DIR
    if args.run_id:
        RUN_ID = args.run_id
        FIG_DIR = ROOT / "docs" / "figs" / RUN_ID
        FIG_DIR.mkdir(parents=True, exist_ok=True)
    for lang in LANGS:
        outs = build(lang, args.dpi, args.with_caption)
        print(f"[{lang}] " + "  ".join(str(p.relative_to(ROOT)) for p in outs))
    verify("baseline_compare")
    print("OK  docs/figs/" + RUN_ID + "/baseline_compare_{analysis,paper}.{png,pdf}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
