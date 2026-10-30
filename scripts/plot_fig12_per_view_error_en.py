#!/usr/bin/env python
# -*- coding: utf-8 -*-
r"""
scripts/plot_fig12_per_view_error_en.py —— 英文版 Fig 12（逐视角完成误差图，CNN vs 线性）

为什么要有这张图
----------------
§5.7 的负结论是"CNN 的 MSE 收缩使其比线性插值略降对比度"。审稿人最直接的追问是：
"是不是只有缺陷所在的少数视角变差了？"本图把 360 个视角的完成误差逐列摊开，答案是
**不是**——CNN 在 360/360 个视角上逐视角 RMSE 都高于线性插值，且瓣内/瓣外的劣化幅度
几乎相同（1.12× vs 1.15×）。缺陷瓣不是 CNN 的"失手区"，而是它整体过平滑的一部分。

三栏
----
  (a) |e_cnn(θ, x)|，θ 为 0..359 全部目标视角，x 为 1200 探测器列
  (b) |e_lin(θ, x)|，同一色标（同一 mask、同一 fill）
  (c) 逐视角 RMSE 散点：横轴线性、纵轴 CNN，y=x 参考线；点按"是否落在信号瓣内"着色

口径（务必与 §5.6 的表区分，二者不可混用）
------------------------------------------
  数据集    3_3_flaw（3.0 MeV 缺陷数据），单能量
  误差域    原始 -log 域（不做归一化）；零计数像素按 config 的 zero_log_fill 填
  行        RECON_ROW = 16（唯一进入正弦图的那一行）
  视角      全部 360 个目标视角（不是 45 个 test 视角）
  mask      目标投影 > 0 且 empty_proj > 0，逐视角逐像素
  CNN 误差  runs/<RUN_ID>_flaw_step3/proj_error_map_{A,B}.npy
            （输入相位 A → 目标相位 B 的 180 行；B → A 的 180 行；并集覆盖 360 视角）
  线性误差  0.5·[L(θ−1) + L(θ+1)] − L(θ)，两侧都是实测视角

  因此本图的 abs rmse（CNN 0.0818 / lin 0.0714）与 §5.6 表的 0.1319 / 0.1292
  不构成直接对照：后者是四档能量归一化域的 overall 池化、且只含 test 视角。

数据来源（只读）
----------------
  runs/<RUN_ID>_flaw_step3/proj_error_map_{A,B}.npy
  sparse_targets_flaw.csv + 3_3_flaw/ + 3_3/ + runs/<RUN_ID>/config.json

输出
----
  docs/figs/p01_flaw_step3_30ep/fig12_per_view_error_en.{png,pdf}
  docs/figs/p01_flaw_step3_30ep/fig12_per_view_rmse.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
for p in (ROOT, ROOT / "scripts"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from data.csv_paths import resolve_csv              # noqa: E402  CSV 统一落点
import baseline_recon as br                       # noqa: E402
import p01_flaw_step3_eval as S                   # noqa: E402

RUN_ID = "20261024_0436_cnn_bs8_lr1e-4"
SRC = ROOT / "runs" / f"{RUN_ID}_flaw_step3"
FIG = ROOT / "docs" / "figs" / "p01_flaw_step3_100ep"
DPI = 300
BAND_PX = 60.0
VMAX = 0.30          # 与 p99(|e|)≈0.28 对齐，两栏共用
LOBES = [(106, 164), (288, 342)]      # Fig 10 的 3σ 判据结果

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 7.4,
    "axes.linewidth": 0.7,
    "axes.labelsize": 7.6,
    "xtick.labelsize": 6.8,
    "ytick.labelsize": 6.8,
    "legend.fontsize": 6.2,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "savefig.facecolor": "white",
    "mathtext.fontset": "dejavusans",
})


def collect() -> dict:
    global LOBES
    # 缺陷几何从本 run 的 image_metrics.json 读回：step3 已把 --defect-rod /
    # --traj-json 的裁决结果写进 defect_geometry。fig12 是独立进程，若沿用模块级
    # 角棒常量会在边棒批次把"轨迹带内/带外"算反（amp 差 107 px、phase 差 45°）。
    # 旧批次没有该字段时静默回退，保证角棒 Fig S8 逐位可复现。
    _mpath = SRC / "image_metrics.json"
    if _mpath.is_file():
        _g = (json.loads(_mpath.read_text(encoding="utf-8"))
              .get("defect_geometry") or {})
        if _g.get("traj"):
            S.TRAJ = dict(_g["traj"])
        if _g.get("defect_rod") and _g["defect_rod"] != "corner1":
            _c = (-S.TRAJ["phase_deg"]) % 360.0
            _half = 29
            LOBES[:] = [(int(round(c - _half)) % 360, int(round(c + _half)) % 360)
                        for c in (_c, (_c + 180.0) % 360.0)]
            print(f"[fig12] 非角棒缺陷 {_g['defect_rod']}：TRAJ={S.TRAJ} "
                  f"动态 LOBES={LOBES}", flush=True)
    else:
        print("[fig12][WARN] 无 image_metrics.json，沿用角棒默认 TRAJ/LOBES",
              flush=True)

    # 数据表与数据根都从 run 自己反推：config.data.targets 记的是表名（csv_data/ 内的
    # 裸文件名），数据根由首行的 target_file 前缀给出（3_3_flaw 或 3_3_flaw_1_1）。
    # 这样换数据批次只需改 --run-id，不会漏改常量而把 empty_proj 取成旧档。
    _cfg = json.loads((br.RUNS_DIR / RUN_ID / "config.json").read_text(encoding="utf-8"))
    S.FLAW_CSV = resolve_csv(_cfg["data"]["targets"])
    rows = S.read_rows(S.FLAW_CSV)
    S.FLAW_ROOT = (br.ROOT / rows[0]["target_file"]).parent.parent.parent
    rows.sort(key=lambda r: float(r["target_angle_deg"]))
    cfg = S.json.loads((br.RUNS_DIR / RUN_ID / "config.json").read_text(encoding="utf-8"))
    fill = float(cfg["norm"]["zero_log_fill"])

    sino, theta, empty = S.build_sinogram(rows, S.FLAW_ROOT, fill)
    if sino.shape[0] != 360:
        raise SystemExit(f"期望 360 视角，实得 {sino.shape[0]}")

    mask = np.zeros_like(sino, dtype=bool)
    for i, r in enumerate(rows):
        proj = np.load(br.ROOT / r["target_file"]).astype(np.float32)
        mask[i] = (proj[br.RECON_ROW] > 0) & (empty[br.RECON_ROW] > 0)

    by_phase = {p: [r for r in rows if r["phase"] == p] for p in ("A", "B")}
    e_cnn = np.full_like(sino, np.nan)
    for in_phase, missing in (("A", "B"), ("B", "A")):
        e = np.load(SRC / f"proj_error_map_{in_phase}.npy")
        if e.shape[0] != len(by_phase[missing]):
            raise SystemExit(f"误差图 {e.shape} 与相位 {missing} 行数不符")
        for i, r in enumerate(by_phase[missing]):
            e_cnn[int(float(r["target_angle_deg"]))] = e[i]
    if np.isnan(e_cnn).any():
        raise SystemExit("误差图未覆盖全部 360 视角")

    e_lin = np.empty_like(sino)
    for a in range(360):
        e_lin[a] = 0.5 * (sino[(a - 1) % 360] + sino[(a + 1) % 360]) - sino[a]

    def rms(a: np.ndarray, m: np.ndarray) -> float:
        return math.sqrt(float(np.mean(a[m] ** 2))) if m.any() else float("nan")

    r_cnn = np.array([rms(e_cnn[a], mask[a]) for a in range(360)])
    r_lin = np.array([rms(e_lin[a], mask[a]) for a in range(360)])

    angles = np.arange(360.0)
    traj = S.TRAJ["offset"] + S.TRAJ["amp"] * np.cos(
        np.radians(angles + S.TRAJ["phase_deg"]))
    ins = np.zeros(360, dtype=bool)
    for s, e in LOBES:
        if s <= e:
            ins[s:e + 1] = True
        else:                      # 跨 0° 的瓣（边棒中心 180°/0° 必然出现）
            ins[s:] = True
            ins[:e + 1] = True

    # 轨迹带内（缺陷信号真正所在）的逐视角 rmse
    cols = np.arange(sino.shape[1], dtype=float)
    band = np.abs(cols[None, :] - traj[:, None]) <= BAND_PX
    rb_cnn = np.array([rms(e_cnn[a], mask[a] & band[a]) for a in range(360)])
    rb_lin = np.array([rms(e_lin[a], mask[a] & band[a]) for a in range(360)])

    return {"sino": sino, "mask": mask, "e_cnn": e_cnn, "e_lin": e_lin,
            "r_cnn": r_cnn, "r_lin": r_lin, "rb_cnn": rb_cnn, "rb_lin": rb_lin,
            "traj": traj, "ins": ins, "n_views": 360}


def fig12(d: dict) -> list[Path]:
    mask, ins = d["mask"], d["ins"]
    lobes = LOBES
    win = d["r_cnn"] < d["r_lin"]

    fig = plt.figure(figsize=(7.5, 2.92))
    gs = fig.add_gridspec(2, 3, width_ratios=[1.52, 1.52, 1.02],
                          height_ratios=[1.0, 0.05], wspace=0.40, hspace=0.42,
                          left=0.085, right=0.978, top=0.875, bottom=0.245)

    ims = []
    for j, (key, tag) in enumerate((("e_cnn", "(a) CNN completion"),
                                    ("e_lin", "(b) Linear completion"))):
        ax = fig.add_subplot(gs[0, j])
        e = np.abs(np.ma.masked_where(~mask, d[key]))
        im = ax.imshow(e, aspect="auto", cmap="inferno", vmin=0.0, vmax=VMAX,
                       extent=(0, e.shape[1], 360, 0), interpolation="nearest")
        ims.append(im)
        ax.plot(d["traj"], np.arange(360.0), color="#31D6DA", lw=0.65, ls=(0, (3, 2)))
        for s, e2 in lobes:
            ax.axhline(s, color="#FFFFFF", lw=0.5, ls=(0, (1, 2)), alpha=0.85)
            ax.axhline(e2, color="#FFFFFF", lw=0.5, ls=(0, (1, 2)), alpha=0.85)
        ax.set_xlabel("Detector column (px)")
        if j == 0:
            ax.set_ylabel("Projection angle $\\theta$ (deg)")
        ax.set_xticks([0, 400, 800, 1200])
        ax.set_yticks([0, 90, 180, 270, 360])
        ax.set_title(tag, fontsize=7.2, pad=3)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)

    cax = fig.add_subplot(gs[1, 0:2])
    cb = fig.colorbar(ims[0], cax=cax, orientation="horizontal")
    cb.set_label("$|\\,L_\\mathrm{completion} - L_\\mathrm{truth}\\,|$   (row 16, $-\\ln$ units)",
                 fontsize=6.4, labelpad=2)
    cb.ax.tick_params(labelsize=6.0, length=2)
    cb.outline.set_linewidth(0.5)
    # 只保留上面两栏共用的色标：删掉重复的 im
    ims[1].set_clim(0.0, VMAX)

    # ---- (c) 逐视角 RMSE 散点
    ax = fig.add_subplot(gs[:, 2])
    lim = max(d["r_lin"].max(), d["r_cnn"].max()) * 1.04
    ax.plot([0, lim], [0, lim], color="#666666", lw=0.7, ls=(0, (4, 2)), zorder=2)
    ax.scatter(d["r_lin"][~ins], d["r_cnn"][~ins], s=7, c="#9E9E9E",
               linewidths=0, alpha=0.85, zorder=3, label="outside lobes")
    ax.scatter(d["r_lin"][ins], d["r_cnn"][ins], s=8, c="#E8912B",
               linewidths=0, alpha=0.9, zorder=4, label="inside lobes")
    ax.set_xlim(0, lim); ax.set_ylim(0, lim)
    ax.set_xlabel("RMSE$\\,$linear (per view)")
    ax.set_ylabel("RMSE$\\,$CNN (per view)")
    ax.set_title("(c) Every view, one point", fontsize=7.2, pad=3)
    ax.grid(color="#EDEDED", lw=0.5); ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.legend(loc="lower right", frameon=True, framealpha=0.92, borderpad=0.3,
              handletextpad=0.35, borderaxespad=0.3)
    ax.text(0.03, 0.93,
            f"above $y=x$ in {int((~win).sum())} of {win.size} views\n"
            f"ratio CNN/linear: {d['r_cnn'][ins].mean()/d['r_lin'][ins].mean():.2f} in-lobe\n"
            f"{d['r_cnn'][~ins].mean()/d['r_lin'][~ins].mean():.2f} outside the lobes",
            transform=ax.transAxes, ha="left", va="top", fontsize=5.9,
            bbox=dict(boxstyle="round,pad=0.26", fc="white", ec="#CCCCCC", lw=0.5))

    out = []
    for ext in ("png", "pdf"):
        p = FIG / f"fig12_per_view_error_en.{ext}"
        fig.savefig(p, dpi=DPI)
        out.append(p)
    plt.close(fig)

    with open(FIG / "fig12_per_view_rmse.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["angle_deg", "rmse_cnn", "rmse_linear", "ratio_cnn_over_linear",
                    "rmse_cnn_band", "rmse_linear_band", "in_signal_lobe"])
        for a in range(360):
            w.writerow([a, f"{d['r_cnn'][a]:.6f}", f"{d['r_lin'][a]:.6f}",
                        f"{d['r_cnn'][a]/d['r_lin'][a]:.6f}",
                        f"{d['rb_cnn'][a]:.6f}", f"{d['rb_lin'][a]:.6f}", int(ins[a])])

    print(f"[fig12] CNN better in {int(win.sum())}/{win.size} views; "
          f"ratio in-lobe={d['r_cnn'][ins].mean()/d['r_lin'][ins].mean():.4f} "
          f"out={d['r_cnn'][~ins].mean()/d['r_lin'][~ins].mean():.4f}; "
          f"band-rmse in-lobe {d['rb_cnn'][ins].mean():.6f}/{d['rb_lin'][ins].mean():.6f}")
    return out


def main(argv: list[str] | None = None) -> int:
    import argparse

    global RUN_ID, SRC, FIG
    ap = argparse.ArgumentParser(description="Fig 12 逐视角误差图（英文版）")
    ap.add_argument("--run-id", default=RUN_ID,
                    help="缺陷域 run id（默认 100ep 零样本 20261024_0436）")
    ap.add_argument("--out-dir", default=None,
                    help="输出目录（默认 docs/figs/p01_flaw_step3_100ep）")
    args = ap.parse_args(argv)
    RUN_ID = args.run_id
    SRC = ROOT / "runs" / f"{RUN_ID}_flaw_step3"
    FIG = (ROOT / args.out_dir if args.out_dir
           else ROOT / "docs" / "figs" / "p01_flaw_step3_100ep")
    FIG.mkdir(parents=True, exist_ok=True)
    print(f"[fig12] RUN_ID={RUN_ID}  SRC={SRC}  FIG={FIG}", flush=True)
    d = collect()
    for p in fig12(d):
        print(f"[OK] {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
