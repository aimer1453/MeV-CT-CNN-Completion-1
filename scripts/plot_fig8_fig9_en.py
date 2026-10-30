#!/usr/bin/env python
# -*- coding: utf-8 -*-
r"""
scripts/plot_fig8_fig9_en.py —— 论文用英文版 Fig 8（缺陷重建）与 Fig 9（三源匹配滤波）

Fig 8：缺陷数据重建，上排全视野、下排缺陷附近 260×260 放大，
       放大窗内用白圈标出**空腔位置**（r≈20 px = 0.20 cm，由径向半高反推）。
       图内不放说明性长句，只留方法名与逐格指标。

Fig 9：同一匹配滤波算子在三种数据来源上的结果
       (a) ΔL(θ)：实测 / 线性补全 / CNN 补全，另叠加实测的 90/180/270° 三条伪轨迹（零假设）
       (b) c(θ) = ΔL_0 − mean(ΔL_90, ΔL_180, ΔL_270)：D4 校正后的逐角缺陷信号
       (c) 对比度 ± sem 与 t 值柱状图

**阴影区 = 缺陷可见的两瓣（不是旧的 ±18° 单弧）**
两瓣由 ΔL(θ) 曲线按"安静区 mean + 3σ"阈值自动判定（与
`scripts/plot_p01_dilution_en.py` 同一算法、同一阈值），实测为
**106–164° 与 288–342°，共 114/360 = 31.7%**。
旧的 `141 ± 18°` 单弧（37 视角）是早期误判，已作废。

数据来源（只读）
----------------
  runs/<run_id>_flaw_step3/recon/*.npy              （Fig 8）
  runs/<run_id>_flaw_step3/image_metrics.json       （Fig 8 的 CNR/RMSE）
  runs/<run_id>_flaw_step3/mf_methods.json          （Fig 9）
  runs/<run_id>_flaw_step3/mf_methods_curves.npz

用法
----
  python scripts/plot_fig8_fig9_en.py                       # 默认 100 epoch 定稿版
  python scripts/plot_fig8_fig9_en.py --run-id 20261023_2327_cnn_bs8_lr1e-4 \
                                     --out-dir p01_flaw_step3_30ep

输出
----
  docs/figs/<out-dir>/fig8_defect_recon_en.{png,pdf}
  docs/figs/<out-dir>/fig9_matched_filter_en.{png,pdf}
  docs/figs/<out-dir>/fig9_matched_filter.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle

ROOT = Path(__file__).resolve().parent.parent
for p in (ROOT, ROOT / "scripts"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import baseline_recon as br                       # noqa: E402

RUN_ID = "20261024_0436_cnn_bs8_lr1e-4"       # 100 epoch，定稿版
OUT_DIR = "p01_flaw_step3_100ep"
DPI = 300

TRAJ = {"amp": 367.1, "offset": 601.9, "phase_deg": 224.7}
AZ_DEFECT_IMG = TRAJ["phase_deg"] - 90.0        # 134.7°
R_DEFECT_PX = TRAJ["amp"]                       # 367.1
CY = 600.0 - R_DEFECT_PX * math.sin(math.radians(AZ_DEFECT_IMG))
CX = 600.0 + R_DEFECT_PX * math.cos(math.radians(AZ_DEFECT_IMG))
R_CAVITY = 20.0                                 # px = 0.20 cm
ZOOM_HALF = 130

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 7.6,
    "axes.linewidth": 0.7,
    "axes.labelsize": 7.8,
    "xtick.labelsize": 7.0,
    "ytick.labelsize": 7.0,
    "legend.fontsize": 6.6,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "savefig.facecolor": "white",
    "mathtext.fontset": "dejavusans",
})

METHODS = [("dense", "dense reference\n(360 views, all measured)"),
           ("cnn_completed", "CNN completion\n(180 + 180)"),
           ("linear_interp", "Linear interp.\n(180 + 180)"),
           ("fbp_sparse", "FBP, sparse\n(180 views)"),
           ("sart_sparse", "SART, sparse\n(180 views)")]
C_MEAS, C_LIN, C_CNN = "#222222", "#3A6EA5", "#B4483C"


def load(p: Path) -> dict:
    return json.loads(p.read_text(encoding="utf-8"))


def visible_lobes(dl: np.ndarray, min_len: int = 5) -> tuple[list[tuple[int, int]], float]:
    """缺陷可见的角向支撑：先按全均值粗定，再按"安静区 mean + 3σ"重定。

    与 scripts/plot_p01_dilution_en.py 逐字同一算法。必须相对安静区噪声判，
    不能相对 0 —— ΔL 含两次仿真间的全局乘性差（约 −0.198），用 0 当阈值会严重
    低估支撑。返回 (两瓣 [(起, 止)], 稀释倍数)。
    """
    def runs(m: np.ndarray) -> list[tuple[int, int]]:
        out = []
        dd = np.diff(np.r_[0, m.astype(int), 0])
        for s, e in zip(np.where(dd == 1)[0], np.where(dd == -1)[0]):
            if e - s >= min_len:
                out.append((int(s), int(e - 1)))
        return out

    lobes = runs(dl > dl.mean())
    quiet = np.ones(dl.size, dtype=bool)
    for s, e in lobes:
        quiet[s:e + 1] = False
    thr = float(dl[quiet].mean()) + 3.0 * float(dl[quiet].std(ddof=1))
    lobes = runs(dl > thr)
    n_sig = sum(e - s + 1 for s, e in lobes)
    return lobes, 360.0 / max(n_sig, 1)


# ------------------------------------------------------------------ Fig 8
def fig8(SRC: Path, FIG: Path) -> list[Path]:
    tm = load(SRC / "image_metrics.json")
    dense = np.load(SRC / "recon" / "dense.npy")
    fov = br.fov_mask()
    vmin, vmax = np.percentile(dense[fov], [1.0, 99.0])

    fig, axes = plt.subplots(2, len(METHODS), figsize=(7.2, 3.95))
    for ci, (key, title) in enumerate(METHODS):
        img = dense if key == "dense" else np.load(SRC / "recon" / f"{key}_A.npy")

        ax = axes[0, ci]
        ax.imshow(img, cmap="gray", vmin=vmin, vmax=vmax, interpolation="nearest")
        ax.add_patch(Circle((CX, CY), R_CAVITY, fill=False, ec="#39FF14", lw=0.8))
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(title, fontsize=7.2, pad=3)
        for s in ax.spines.values():
            s.set_visible(False)

        ax = axes[1, ci]
        y0, y1 = int(CY) - ZOOM_HALF, int(CY) + ZOOM_HALF
        x0, x1 = int(CX) - ZOOM_HALF, int(CX) + ZOOM_HALF
        ax.imshow(img[y0:y1, x0:x1], cmap="gray", vmin=vmin, vmax=vmax,
                  interpolation="nearest", extent=(x0, x1, y1, y0))
        ax.add_patch(Circle((CX, CY), R_CAVITY, fill=False, ec="#39FF14", lw=0.9))
        ax.set_xticks([]); ax.set_yticks([])
        for s in ax.spines.values():
            s.set_linewidth(0.5); s.set_color("#999999")

        m = tm["per_method"][key]
        cnr = tm["defect_cnr"][key]["in_image"]["cnr"]
        if key == "dense":
            txt = f"CNR {cnr:.2f}"
        else:
            txt = f"RMSE {m['rmse_valid']:.2e}\nSSIM {m['ssim']:.3f}\nCNR {cnr:.2f}"
        ax.text(0.5, -0.045, txt, transform=ax.transAxes, ha="center", va="top",
                fontsize=6.3, linespacing=1.35)

    fig.text(0.5, 0.004, "cavity marked by the open circle ($r\\approx$ 20 px = 0.20 cm)",
             ha="center", va="bottom", fontsize=6.3, color="#444444")
    fig.subplots_adjust(left=0.005, right=0.995, top=0.915, bottom=0.185, wspace=0.04,
                        hspace=0.10)
    out = []
    for ext in ("png", "pdf"):
        p = FIG / f"fig8_defect_recon_en.{ext}"
        fig.savefig(p, dpi=DPI)
        out.append(p)
    plt.close(fig)
    return out


# ------------------------------------------------------------------ Fig 9
def fig9(SRC: Path, FIG: Path) -> list[Path]:
    mf = load(SRC / "mf_methods.json")
    z = np.load(SRC / "mf_methods_curves.npz")
    ang = np.arange(360.0)
    null = np.vstack([z["measured_off90"], z["measured_off180"], z["measured_off270"]])
    n_mean = null.mean(axis=0)
    n_sd = null.std(axis=0, ddof=1)
    # 2026-10-28（换代裁决）：**不再在图上标"缺陷可见瓣/视角数**。
    # 直接作差的正弦图 ΔL 是两次独立仿真之差，包壳补齐后被边缘对齐残差主导，
    # 瓣结构在两批之间不稳健；可检测性一律以 D4 匹配滤波为准（同图内伪轨迹做基线）。
    # 因此 (a)(b) 不再画 lobe span，也不再印 "n=x/360"。
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.62))

    # (a) ΔL(θ) + D4 零假设带
    ax = axes[0]
    ax.fill_between(ang, n_mean - n_sd, n_mean + n_sd, color="#8E8E8E", alpha=0.28, lw=0,
                    label="null band ($90/180/270^\\circ$)")
    ax.plot(ang, n_mean, color="#6E6E6E", lw=0.7, ls=(0, (4, 2)))
    ax.plot(ang, z["measured_off0"], color=C_MEAS, lw=1.15, label="measured (true)")
    ax.plot(ang, z["linear_off0"], color=C_LIN, lw=1.0, ls=(0, (5, 1.6)),
            label="linear interp.")
    ax.plot(ang, z["cnn_off0"], color=C_CNN, lw=1.0, ls=(0, (1, 1.3)), label="CNN")
    ax.set_xlim(0, 360); ax.set_xticks([0, 90, 180, 270, 360])
    ax.set_xlabel("Projection angle $\\theta$ (deg)")
    ax.set_ylabel("$\\Delta L(\\theta)$ over the trajectory band")
    ax.set_title("(a) Trajectory integral vs $D_4$ null", fontsize=7.4, pad=3)
    ax.grid(color="#E9E9E9", lw=0.5); ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.set_ylim(-0.37, 0.31)
    # 面板内不再放 legend（会压住零假设带与曲线），改由 fig 顶部共享 legend 承担。

    # (b) D4 校正后
    ax = axes[1]
    styles = (("measured", C_MEAS, "-"), ("linear", C_LIN, (0, (5, 1.6))),
              ("cnn", C_CNN, (0, (1, 1.3))))
    for key, col, ls in styles:
        ax.plot(ang, z[f"{key}_d4"], color=col, lw=1.0, ls=ls)
    ax.axhline(0.0, color="#999999", lw=0.6)
    lines = []
    for key, col, _ in styles:
        d4 = mf[key]["d4_curve"]
        t = mf[key]["d4_contrast"]["t_stat"]
        nm = {"measured": "measured", "linear": "linear", "cnn": "CNN"}[key]
        lines.append(f"{nm:<9s} {d4['mean']:+.4f}   $t={t:+.2f}$")
    # 数字框：先给顶部留出 45% 的空白带，再把框固定在右上角 —— 这样框不压任何曲线。
    # （原先写死左下角会压住 θ≈0–60 的曲线；自动挑角落仍会压 1.4%，故直接造空带。）
    _curves = np.vstack([z[f"{k}_d4"] for k, _, _ in styles])
    _y0, _y1 = _curves.min(), _curves.max()
    _span = (_y1 - _y0) or 1.0
    _ya = _y0 - 0.08 * _span
    _yb = _y1 + 0.45 * _span
    _frac = float(np.mean((_curves >= _ya + 0.70 * (_yb - _ya)) & (_curves <= _yb)))
    print(f"[fig9] 顶部留白带 = 45% span；数字框位于该带内，覆盖曲线点比例 {_frac:.1%}")
    ax.text(0.97, 0.97, "\n".join(lines), transform=ax.transAxes, fontsize=5.9,
            va="top", ha="right", family="DejaVu Sans Mono", linespacing=1.5,
            bbox=dict(boxstyle="round,pad=0.28", fc="white", ec="#CCCCCC", lw=0.5))
    ax.set_ylim(_ya, _yb)
    ax.set_xlim(0, 360); ax.set_xticks([0, 90, 180, 270, 360])
    ax.set_xlabel("Projection angle $\\theta$ (deg)")
    ax.set_ylabel("$c(\\theta)=\\Delta L_0-\\overline{\\Delta L_{\\rm null}}$")
    ax.set_title("(b) Defect signal after $D_4$ correction", fontsize=7.4, pad=3)
    ax.grid(color="#E9E9E9", lw=0.5); ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)

    # (c) 对比度柱状
    ax = axes[2]
    keys = ("measured", "linear", "cnn")
    labs = ("measured", "linear\ninterp.", "CNN")
    cols = (C_MEAS, C_LIN, C_CNN)
    vals = [-mf[k]["d4_contrast"]["offset0_minus_pseudo_baseline"] for k in keys]
    sems = [mf[k]["d4_contrast"]["sem"] for k in keys]
    xs = np.arange(3)
    ax.bar(xs, vals, 0.56, color=cols, linewidth=0, zorder=3)
    ax.errorbar(xs, vals, yerr=sems, fmt="none", ecolor="#333333", elinewidth=0.8,
                capsize=2.5, zorder=4)
    kept = [100.0] + [100 * v / vals[0] for v in vals[1:]]
    # t 值不再浮在柱顶（三个柱间距只有 1/3 面板宽，6.4pt 的 "$t=-13.76$" 会首尾相撞），
    # 改挂到 x 轴刻度标签的第二行，横向空间与刻度同宽、绝不会互相挤压。
    _tick2 = []
    for i, k in enumerate(keys):
        t = mf[k]["d4_contrast"]["t_stat"]
        _tick2.append(f"{labs[i]}\n$t={t:+.2f}$")
        ax.text(xs[i], 0.0035, f"{kept[i]:.1f}%", ha="center", fontsize=6.2,
                color="white")
    # CNN 与 linear 的差是否显著（逐角配对 t）
    pt = mf["paired_tests"].get("d4: cnn_minus_linear")
    if pt is not None:
        # 拆成两行：单行 ~52 字符在 1/3 面板宽里会被右边界裁掉。
        ax.text(0.5, 0.965,
                "CNN vs linear (paired, $n$=360):\n"
                f"$\\Delta c$={pt['mean_diff']:+.4f}, $t$={pt['t_stat']:+.2f}"
                + (" (n.s.)" if abs(pt["t_stat"]) < 1.96 else ""),
                transform=ax.transAxes, ha="center", va="top", fontsize=5.4,
                color="#444444", linespacing=1.35)
    ax.set_xticks(xs); ax.set_xticklabels(_tick2, fontsize=6.0)
    ax.tick_params(axis="x", pad=1.5)
    ax.set_ylim(0, max(vals) * 1.34)
    ax.set_ylabel("Defect contrast $|c|$")
    ax.set_title("(c) Contrast retained", fontsize=7.4, pad=3)
    ax.grid(axis="y", color="#E9E9E9", lw=0.5); ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)

    # 共享 legend 放在图顶一行（ncol=4）——面板内 legend 必然压数据，面板外才干净。
    _h, _l = axes[0].get_legend_handles_labels()
    fig.legend(_h, _l, loc="upper center", bbox_to_anchor=(0.5, 0.995), ncol=4,
               frameon=False, fontsize=6.2, handlelength=1.6, columnspacing=1.1,
               labelspacing=0.2)
    fig.subplots_adjust(left=0.088, right=0.985, top=0.795, bottom=0.235, wspace=0.46)
    out = []
    for ext in ("png", "pdf"):
        p = FIG / f"fig9_matched_filter_en.{ext}"
        fig.savefig(p, dpi=DPI)
        out.append(p)
    plt.close(fig)

    with open(FIG / "fig9_matched_filter.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["source", "offset0_mean", "pseudo_baseline",
                    "d4_contrast", "sem", "t_stat", "retained_pct",
                    "arc_inside_minus_outside"])
        for k in keys:
            d4 = mf[k]["d4_contrast"]
            arc = mf[k].get("arc_band", {})
            w.writerow([k, f"{d4['offset0_mean']:.6f}",
                        f"{d4['pseudo_baseline']:.6f}",
                        f"{d4['offset0_minus_pseudo_baseline']:.6f}",
                        f"{d4['sem']:.6f}", f"{d4['t_stat']:.4f}",
                        f"{100 * (-d4['offset0_minus_pseudo_baseline']) / vals[0]:.2f}",
                        f"{arc.get('inside_minus_outside', float('nan')):.6f}"])
        # 瓣信息仍落盘（可追溯），但**图上不再标**：直接作差的 ΔL 含两次仿真的
        # 边缘对齐残差，瓣结构在两批数据间不稳健，可检测性以 D4 为准。
        _lobes, _dil = visible_lobes(z["measured_off0"])
        _n = sum(e - s + 1 for s, e in _lobes)
        w.writerow([])
        w.writerow(["# NOTE", "lobe structure below is from the raw defect-minus-uniform "
                    "difference and is NOT robust across simulation batches; "
                    "detectability is assessed with the D4 statistic above"])
        w.writerow(["lobe_start_deg", "lobe_end_deg", "n_views"])
        for s, e in _lobes:
            w.writerow([s, e, e - s + 1])
        w.writerow([])
        w.writerow(["n_lobe_total", _n, "dilution_x", f"{_dil:.2f}"])
    return out


def main() -> int:
    global RUN_ID, OUT_DIR
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-id", default=RUN_ID)
    ap.add_argument("--out-dir", default=OUT_DIR)
    a = ap.parse_args()
    RUN_ID, OUT_DIR = a.run_id, a.out_dir
    SRC = ROOT / "runs" / f"{RUN_ID}_flaw_step3"
    FIG = ROOT / "docs" / "figs" / OUT_DIR
    if not SRC.is_dir():
        raise SystemExit(f"缺目录：{SRC}（先跑 scripts/p01_flaw_step3_eval.py --run-id {RUN_ID}）")
    FIG.mkdir(parents=True, exist_ok=True)
    for p in fig8(SRC, FIG) + fig9(SRC, FIG):
        print(f"[OK] {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
