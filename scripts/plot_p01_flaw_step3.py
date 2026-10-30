#!/usr/bin/env python
# -*- coding: utf-8 -*-
r"""
scripts/plot_p01_flaw_step3.py —— P0-1 第三步配图（缺陷数据域内训练）

五张图，对应三个待回答问题：

  Fig. A  flaw_proj_angle_rmse.png
          投影域：45 个 test 视角的 rmse 随目标角变化（叠加缺陷正弦轨迹），
          以及 rmse 与"缺陷运动速度 |dcos/dθ|"的散点+拟合
          → 回答"缺陷附近的视角误差是否更大"（答：不更大）

  Fig. B  flaw_proj_error_map.png
          投影域：|预测 − 真值| 的 (预测角 × 探测器列) 热图 + 列剖面，
          列剖面上叠加 |真值 L| 的列均值
          → 说明误差跟的是**物体结构**（|L| 大的列），不是缺陷轨迹

  Fig. C  flaw_recon_compare.png
          图像域：dense / CNN 补全 / 线性插值 / FBP 稀疏 / SART 稀疏 五法并列，
          第二行为缺陷位置放大（同色窗）

  Fig. D  flaw_cnr.png
          图像域：缺陷棒与其余 8 根棒的**同等**峰值搜索对照 + CNR 柱状

  Fig. E  flaw_proj_matched_filter.png
          投影域轨迹匹配滤波：flaw−uniform 沿缺陷轨迹取带 vs 带外，
          并用 90°/180°/270° 旋转的"伪轨迹"做零假设对照
          → 缺陷能不能被检出（投影域能，图像域不能）

数据来源（全部从 runs/<RUN_ID>_flaw_step3/ 与 sparse_targets_flaw.csv 读）
--------------------------------------------------------------------------
  proj_angle.csv / proj_error_map_{A,B}.npy / proj_matched_filter_curve.npy
  image_metrics.json / recon/{dense,cnn_completed,linear_interp,fbp_sparse,sart_sparse}_A.npy

显示约定
--------
  - 重建图灰度窗取 dense 的 [p1, p99]（与 Fig. 6/7 一致）。
  - 同一列上下两行共用色标，保证放大图与全图可横比。
  - 缺陷棒位置用十字标出；FOV 圆 r=600 px 用白细线。

用法
----
  conda run -n ct_full --no-capture-output python scripts/plot_p01_flaw_step3.py
"""

from __future__ import annotations

import csv
import json
import math
import sys
from pathlib import Path

import numpy as np

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt                       # noqa: E402
from matplotlib import rcParams                       # noqa: E402
from matplotlib.patches import Circle                 # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
for p in (ROOT, ROOT / "scripts"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

RUN_ID = "20261023_2239_cnn_bs8_lr1e-4"
SRC = ROOT / "runs" / f"{RUN_ID}_flaw_step3"
FIG = ROOT / "docs" / "figs" / "p01_flaw_step3"
FIG.mkdir(parents=True, exist_ok=True)

rcParams.update({
    "font.family": ["Microsoft YaHei", "DejaVu Sans"],
    "axes.unicode_minus": False,
    "font.size": 8.5,
    "axes.titlesize": 9.0,
    "axes.labelsize": 8.5,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 7.2,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
})

TRAJ = {"amp": 367.1, "offset": 601.9, "phase_deg": 224.7}
METHODS = ("dense", "cnn_completed", "linear_interp", "fbp_sparse", "sart_sparse")
TITLE = {"dense": "dense 参考\n(360 视角全实测)",
         "cnn_completed": "CNN 补全\n(180+180)",
         "linear_interp": "线性插值\n(180+180)",
         "fbp_sparse": "FBP 稀疏\n(180 视角)",
         "sart_sparse": "SART 稀疏\n(180 视角)"}


def traj_col(a):
    return TRAJ["offset"] + TRAJ["amp"] * np.cos(np.radians(a + TRAJ["phase_deg"]))


def meta() -> dict:
    return json.loads((SRC / "image_metrics.json").read_text(encoding="utf-8"))


# ------------------------------------------------------------------ Fig. A
def fig_angle() -> Path:
    rows = list(csv.DictReader(open(SRC / "proj_angle.csv", encoding="utf-8")))
    ang = np.array([float(r["target_angle_deg"]) for r in rows])
    rv = np.array([float(r["rmse_valid"]) for r in rows])
    sp = np.array([float(r["defect_speed_au"]) for r in rows])
    pj = meta()["projection_domain"]

    fig, axes = plt.subplots(1, 2, figsize=(7.1, 2.75))

    ax = axes[0]
    ax.plot(ang, rv, "o-", ms=3.2, lw=0.9, color="#1f4e79", label="test 视角 rmse")
    ax.axhline(rv.mean(), ls="--", lw=0.8, color="#888")
    ax.set_xlabel("目标角 θ (deg)")
    ax.set_ylabel("投影域 rmse_valid")
    ax.set_xlim(-5, 364)
    ax.set_title(f"(a) 45 个 test 视角：均值 {rv.mean():.3f}，"
                 f"std {rv.std(ddof=1):.3f}（起伏仅 {rv.std(ddof=1) / rv.mean() * 100:.1f}%）",
                 loc="left")
    ax2 = ax.twinx()
    ax2.plot(np.arange(360), traj_col(np.arange(360)), lw=0.8, color="#c0504d", alpha=0.7)
    ax2.set_ylabel("缺陷投影列 col(θ) (px)", color="#c0504d")
    ax2.tick_params(axis="y", colors="#c0504d")
    ax2.set_ylim(-50, 1250)
    ax.legend(loc="upper right")

    ax = axes[1]
    ax.plot(sp, rv, "o", ms=3.4, color="#1f4e79", alpha=0.85)
    k, b = np.polyfit(sp, rv, 1)
    xs = np.linspace(0, 1, 50)
    c = pj["corr_rmse_vs_defect_speed"]
    ax.plot(xs, k * xs + b, "-", lw=1.0, color="#c0504d",
            label=f"r={c['pearson_r']:+.3f}, t={c['t']:+.2f} (df={c['df']})")
    ax.set_xlabel("缺陷运动速度 |dcos(θ+224.7°)/dθ| (a.u.)")
    ax.set_ylabel("投影域 rmse_valid")
    ax.set_title("(b) 误差与缺陷运动速度无关（|t|<1 不显著）", loc="left")
    ax.legend(loc="upper left")

    fig.tight_layout()
    p = FIG / "flaw_proj_angle_rmse.png"
    fig.savefig(p)
    fig.savefig(p.with_suffix(".pdf"))
    plt.close(fig)
    print(f"写出 {p}")
    return p


# ------------------------------------------------------------------ Fig. B
def fig_errmap() -> Path:
    import p01_flaw_step3_eval as ev

    errA = np.load(SRC / "proj_error_map_A.npy")
    errB = np.load(SRC / "proj_error_map_B.npy")
    met = meta()
    pj = met["projection_domain"]["per_view_row16"]
    rows = [r for r in ev.read_rows(ev.FLAW_CSV)]
    sino, _theta, _empty = ev.build_sinogram(rows, ev.FLAW_ROOT, 5.10)
    gt_col = np.abs(sino).mean(axis=0)

    fig = plt.figure(figsize=(7.1, 4.6))
    gs = fig.add_gridspec(2, 3, width_ratios=[1.05, 1.0, 0.05],
                          height_ratios=[1.0, 1.0], hspace=0.55, wspace=0.30)

    vmax = float(np.percentile(np.abs(np.concatenate([errA.ravel(), errB.ravel()])), 99.5))
    for r, (err, angs, tag, ph) in enumerate((
            (errA, np.arange(1, 360, 2), "输入相位 A → 预测奇数角", "A"),
            (errB, np.arange(0, 360, 2), "输入相位 B → 预测偶数角", "B"))):
        ax = fig.add_subplot(gs[r, 0])
        im = ax.imshow(np.abs(err), aspect="auto", origin="lower", cmap="magma",
                       vmin=0, vmax=vmax, extent=[0, 1200, 0, 360])
        aa = np.linspace(0, 359, 720)
        ax.plot(traj_col(aa), aa, "-", lw=0.9, color="#00e5ff", alpha=0.9,
                label="缺陷轨迹")
        ax.set_xlim(0, 1200)
        ax.set_ylim(0, 360)
        ax.set_ylabel("预测角 θ (deg)")
        ax.set_title(f"({'ab'[r]}) |预测−真值|  row16  {tag}", loc="left", fontsize=8.2)
        ax.legend(loc="upper right", framealpha=0.8, fontsize=6.4)

        axp = fig.add_subplot(gs[r, 1])
        ee = np.abs(err).mean(axis=0)
        axp.plot(np.arange(1200), ee, lw=0.7, color="#1f4e79", label="列均值 |误差|")
        axp.set_xlim(0, 1200)
        axp.set_ylim(0, max(ee.max() * 1.35, 1e-6))
        axp.set_xlabel("探测器列 (px)")
        axp.set_ylabel("列均值 |误差|", color="#1f4e79")
        axp.tick_params(axis="y", colors="#1f4e79")
        axg = axp.twinx()
        axg.plot(np.arange(1200), gt_col, lw=0.7, ls="--", color="#777", alpha=0.9,
                 label="列均值 |真值 L|")
        axg.set_ylabel("列均值 |L|", color="#777")
        axg.tick_params(axis="y", colors="#777")
        rr = float(np.corrcoef(ee, gt_col)[0, 1])
        axp.set_title(f"({'cd'[r]}) 列剖面：误差与 |L| 同步，r={rr:+.3f}"
                      f"（轨迹带/带外 相对误差比 {pj[ph]['rel_ratio_on_off']:.2f}）",
                      loc="left", fontsize=8.2)
        if r == 0:
            cax = fig.add_subplot(gs[:, 2])
            cb = fig.colorbar(im, cax=cax)
            cb.set_label("|预测 − 真值|  (-log 域)")

    fig.suptitle("补全误差跟的是物体投影结构（|L| 大的列），不是缺陷轨迹："
                 "沿轨迹的相对误差反而更低", fontsize=8.5, y=0.995)
    p = FIG / "flaw_proj_error_map.png"
    fig.savefig(p)
    fig.savefig(p.with_suffix(".pdf"))
    plt.close(fig)
    print(f"写出 {p}")
    return p


# ------------------------------------------------------------------ Fig. C
def fig_recon() -> Path:
    imgs = {m: np.load(SRC / "recon" / ("dense.npy" if m == "dense" else f"{m}_A.npy"))
            for m in METHODS}
    met = meta()
    cnr = met["defect_cnr"]
    r_px = cnr["dense"]["in_image"]["defect_rod_r_px"]
    az = cnr["dense"]["in_image"]["defect_rod_az_img_deg"]
    cy = 600.0 - r_px * math.sin(math.radians(az))
    cx = 600.0 + r_px * math.cos(math.radians(az))
    half = 130
    y0, y1 = int(cy - half), int(cy + half)
    x0, x1 = int(cx - half), int(cx + half)

    dense = imgs["dense"]
    vmin, vmax = np.percentile(dense, [1.0, 99.0])

    fig, axes = plt.subplots(2, len(METHODS), figsize=(7.1, 3.4))
    for c, m in enumerate(METHODS):
        pm = met["per_method"][m]
        axes[0, c].imshow(imgs[m], cmap="gray", vmin=vmin, vmax=vmax,
                          interpolation="nearest")
        axes[0, c].add_patch(Circle((600, 600), 600, fill=False, ec="w", lw=0.4, alpha=0.6))
        axes[0, c].plot([cx], [cy], "+", ms=6, mew=0.9, color="#00e5ff")
        axes[0, c].set_title(f"{TITLE[m]}\nrmse {pm['rmse_valid']:.4f}  "
                             f"SSIM {pm['ssim']:.3f}", fontsize=6.6)
        axes[0, c].axis("off")

        axes[1, c].imshow(imgs[m][y0:y1, x0:x1], cmap="gray", vmin=vmin, vmax=vmax,
                          interpolation="nearest")
        cm = cnr.get(m, {})
        tagtxt = (f"CNR {cm['in_image']['cnr']:.2f}"
                  if cm.get("status") != "missing reference" else "CNR —")
        axes[1, c].set_xlabel(f"{tagtxt}", fontsize=7.0, labelpad=2, color="#333")
        axes[1, c].set_xticks([]); axes[1, c].set_yticks([])
    fig.suptitle("缺陷数据（3.0 MeV，单缺陷 (i,j)=(−1,+1)）图像域五法对比；"
                 f"灰度窗 = dense 的 [p1, p99]，下行为缺陷棒 260×260 放大", fontsize=8.5)
    fig.tight_layout(rect=(0, 0, 1, 0.94), h_pad=1.0)
    p = FIG / "flaw_recon_compare.png"
    fig.savefig(p)
    fig.savefig(p.with_suffix(".pdf"))
    plt.close(fig)
    print(f"写出 {p}")
    return p


# ------------------------------------------------------------------ Fig. D
def fig_cnr() -> Path:
    cnr = meta()["defect_cnr"]
    ms = [m for m in METHODS if cnr.get(m, {}).get("status") != "missing reference"]

    fig, axes = plt.subplots(1, 2, figsize=(7.1, 3.15),
                             gridspec_kw={"width_ratios": [0.85, 1.75]})
    ax = axes[0]
    vals = [cnr[m]["in_image"]["cnr"] for m in ms]
    bars = ax.bar(np.arange(len(ms)), vals, color="#1f4e79", width=0.62)
    ax.set_xticks(np.arange(len(ms)))
    ax.set_xticklabels([m.replace("_", "\n") for m in ms], fontsize=6.6)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.04, f"{v:.2f}",
                ha="center", fontsize=7)
    ax.set_ylim(0, max(vals) * 1.30)
    ax.set_ylabel("CNR = (μ_ref − μ_bubble) / σ", fontsize=7.6)
    ax.set_title(f"(a) 缺陷空腔 CNR（Rose 判据 5，均未达）", loc="left", fontsize=8.0)

    ax = axes[1]
    names = [r["rod"] for r in cnr[ms[0]]["rod_centres"]]
    idx = np.arange(len(names))
    width = 0.16
    for i, m in enumerate(ms):
        rc = cnr[m]["rod_centres"]
        norm = np.mean([r["mu_centre"] for r in rc if not r["is_defect_rod"]])
        ys = [r["mu_centre"] / norm for r in rc]
        ax.bar(idx + (i - (len(ms) - 1) / 2) * width, ys, width=width,
               label=m.replace("_", " "))
    ax.set_xticks(idx)
    ax.set_xticklabels(names, fontsize=6.4, rotation=30)
    ax.set_ylabel("中心均值 / 正常棒均值", fontsize=7.6)
    ax.set_ylim(0, 1.55)
    d = cnr["dense"]["vs_noflaw"]
    ax.set_title(f"(b) 9 根棒中心值：只有 corner1 塌陷；"
                 f"差值图副口径排名 {d['defect_rank_among_9']}/9", loc="left", fontsize=8.0)
    ax.legend(fontsize=6.0, ncol=5, loc="upper center", framealpha=0.95,
              columnspacing=0.9, handlelength=1.0)

    fig.tight_layout()
    p = FIG / "flaw_cnr.png"
    fig.savefig(p)
    fig.savefig(p.with_suffix(".pdf"))
    plt.close(fig)
    print(f"写出 {p}")
    return p


# ------------------------------------------------------------------ Fig. E
def fig_mf() -> Path:
    mf = meta()["projection_matched_filter"]
    dl = np.load(SRC / "proj_matched_filter_curve.npy")
    ang = np.arange(360)

    fig, axes = plt.subplots(1, 2, figsize=(7.1, 2.75),
                             gridspec_kw={"width_ratios": [1.0, 1.0]})
    ax = axes[0]
    ax.plot(ang, dl, lw=0.8, color="#1f4e79", label="沿实测轨迹")
    ax.axhline(0, lw=0.7, color="k", ls=":")
    ax.axhline(mf["by_offset"]["offset_0deg"]["mean_dlog"], lw=0.8, ls="--",
               color="#c0504d")
    ax.set_xlabel("投影角 θ (deg)")
    ax.set_ylabel("ΔL(θ) = 轨迹带内 − 带外")
    ax.set_xlim(0, 359)
    ax.set_title(f"(a) 沿轨迹的投影差（带 ±{mf['band_halfwidth_px']:.0f} px）；"
                 f"峰值角 {mf['peak_angle_deg']:.0f}°", loc="left")
    ax.legend(loc="upper right")

    ax = axes[1]
    labs, mus, sems = [], [], []
    for k, v in mf["by_offset"].items():
        labs.append(k.replace("offset_", "").replace("deg", "°"))
        mus.append(v["mean_dlog"])
        sems.append(v["sem"])
    ax.bar(np.arange(len(mus)), mus, yerr=sems, capsize=3,
           color=["#c0504d", "#1f4e79", "#8faadc", "#8faadc"], width=0.6)
    ax.axhline(0, lw=0.7, color="k")
    for i, (m, s) in enumerate(zip(mus, sems)):
        t = m / s if s > 0 else 0.0
        ax.text(i, m + (s if m >= 0 else -s) * 1.6 + (0.0004 if m >= 0 else -0.0009),
                f"t={t:+.1f}", ha="center", fontsize=7)
    ax.set_xticks(np.arange(len(mus)))
    ax.set_xticklabels(labs)
    ax.set_ylabel("ΔL 均值（360 角）")
    d4 = mf["d4_contrast"]
    ax.set_title(f"(b) D4 对照：90/180/270° 伪轨迹基线一致（极差 "
                 f"{mf['by_offset']['offset_90deg']['mean_dlog'] - mf['by_offset']['offset_270deg']['mean_dlog']:+.4f}）；"
                 f"0° 偏离基线 {d4['offset0_minus_pseudo_baseline']:+.4f}（t={d4['t_stat']:+.1f}）",
                 loc="left", fontsize=7.8)

    fig.tight_layout()
    p = FIG / "flaw_proj_matched_filter.png"
    fig.savefig(p)
    fig.savefig(p.with_suffix(".pdf"))
    plt.close(fig)
    print(f"写出 {p}")
    return p


def main() -> int:
    import argparse

    ap = argparse.ArgumentParser(description="缺陷数据第三步配图")
    ap.add_argument("--run-id", default=None,
                    help="覆盖 RUN_ID，从 runs/<run_id>_flaw_step3/ 读数")
    ap.add_argument("--suffix", default="",
                    help="输出目录后缀，如 _30ep → docs/figs/p01_flaw_step3_30ep/")
    args = ap.parse_args()

    global RUN_ID, SRC, FIG
    RUN_ID = args.run_id or RUN_ID
    SRC = ROOT / "runs" / f"{RUN_ID}_flaw_step3"
    FIG = ROOT / "docs" / "figs" / f"p01_flaw_step3{args.suffix}"
    FIG.mkdir(parents=True, exist_ok=True)
    print(f"[源] {SRC}\n[出] {FIG}", flush=True)

    fig_angle()
    fig_errmap()
    fig_recon()
    fig_cnr()
    fig_mf()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
