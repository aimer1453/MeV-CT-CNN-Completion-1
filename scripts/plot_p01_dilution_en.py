#!/usr/bin/env python
# -*- coding: utf-8 -*-
r"""
scripts/plot_p01_dilution_en.py —— 英文版 Fig 10（稀释效应）与 Fig 11（空腔 CNR）

Fig 10（证明"缺陷信息在投影域、被 FBP 稀释"的关键直观材料）
  (a) 缺陷差分正弦图 ΔL(θ, col) = L_flaw − L_uniform：缺陷只落在一条窄正弦带上
  (b) 逐视角带内−带外对比度 ΔL(θ)：两瓣共 114/360（31.7%）的视角带信号，
      故等权反投影稀释 3.2–3.7×（旧稿"±18°/36 视角/10×"作废，见 §5 更正块）
  (c) 逐视角完成误差（相对量）：瓣内并不比瓣外大 → 图像域 CNR 低不是模型在异常视角失手

Fig 11（缺陷 CNR）
  (a) 五种方法的空腔 CNR（主口径：腔内 vs 同棒外环），附 Rose 判据 5
  (b) 9 根棒中心值，标出缺陷棒（值被空腔打掉 96%）

数据来源（只读）
----------------
  runs/<flaw_run>_flaw_step3/image_metrics.json         （CNR、9 棒中心值）
  runs/<flaw_run>_flaw_step3/proj_error_map_{A,B}.npy   （逐视角完成误差，行序 = CSV 中该相位行序）
  runs/<flaw_run>_flaw_step3/proj_angle.csv             （45 个 test 视角的误差，未用）
  sparse_targets_flaw.csv + 3_3_flaw/ + 3_3/            （现算 ΔL 与世界对比度）

输出
----
  docs/figs/p01_flaw_step3_30ep/fig10_dilution_en.{png,pdf}
  docs/figs/p01_flaw_step3_30ep/fig11_cnr_en.{png,pdf}
  docs/figs/p01_flaw_step3_30ep/fig10_per_view.csv
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
import p01_flaw_step3_eval as S                    # noqa: E402

RUN_ID = "20261024_0436_cnn_bs8_lr1e-4"
SRC = ROOT / "runs" / f"{RUN_ID}_flaw_step3"
FIG = ROOT / "docs" / "figs" / "p01_flaw_step3_100ep"
DPI = 300
ARC_HALF = 18.0

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


def load(p: Path) -> dict:
    return json.loads(p.read_text(encoding="utf-8"))


# ------------------------------------------------------------------ 逐视角数据
def per_view() -> dict:
    """返回对齐到 0..359 的：ΔL 差分栈、逐视角带对比度、逐视角相对误差。"""
    # 数据表与数据根都从 run 自己反推：config.data.targets 记的是表名（csv_data/ 内的
    # 裸文件名），数据根由首行的 target_file 前缀给出（3_3_flaw 或 3_3_flaw_1_1）。
    # 这样换数据批次只需改 --run-id，不会漏改常量而把 empty_proj 取成旧档。
    _cfg = load(br.RUNS_DIR / RUN_ID / "config.json")
    S.FLAW_CSV = resolve_csv(_cfg["data"]["targets"])
    rows = S.read_rows(S.FLAW_CSV)
    S.FLAW_ROOT = (br.ROOT / rows[0]["target_file"]).parent.parent.parent
    rows.sort(key=lambda r: float(r["target_angle_deg"]))
    cfg = load(br.RUNS_DIR / RUN_ID / "config.json")["norm"]
    fill = float(cfg["zero_log_fill"])

    # 缺陷侧 + 无缺陷侧，同 fill
    sino_f, theta_f, empty_f = S.build_sinogram(rows, S.FLAW_ROOT, fill)
    rows_u = [r for r in S.read_rows(br.SPARSE_TARGETS) if r["energy_MeV"] == S.ENERGY]
    rows_u.sort(key=lambda r: float(r["target_angle_deg"]))
    sino_u, theta_u, _ = S.build_sinogram(rows_u, ROOT / "3_3", fill)
    if not np.array_equal(np.round(theta_f).astype(int), np.round(theta_u).astype(int)):
        raise SystemExit("两侧角度序不一致")
    d = sino_f - sino_u                                  # (360,1200) 缺陷信号

    # 逐视角 mask（与训练/评估一致）
    mask = np.zeros_like(sino_f, dtype=bool)
    for i, r in enumerate(rows):
        proj = np.load(br.ROOT / r["target_file"]).astype(np.float32)
        mask[i] = (proj[br.RECON_ROW] > 0) & (empty_f[br.RECON_ROW] > 0)

    # 逐视角完成误差：proj_error_map_{A,B}.npy 的行序 = CSV 中该相位（目标相位）行的顺序
    by_phase = {p: [r for r in rows if r["phase"] == p] for p in ("A", "B")}
    err = np.full_like(sino_f, np.nan)
    # 输入相位 A → 预测目标相位 B；输入相位 B → 预测目标相位 A
    for in_phase, missing in (("A", "B"), ("B", "A")):
        e = np.load(SRC / f"proj_error_map_{in_phase}.npy")
        ang = [int(float(r["target_angle_deg"])) for r in by_phase[missing]]
        if e.shape[0] != len(ang):
            raise SystemExit(f"误差图 {e.shape} 与相位 {missing} 行数 {len(ang)} 不符")
        for i, a in enumerate(ang):
            err[a] = e[i]
    if np.isnan(err).any():
        raise SystemExit("误差图未覆盖全部 360 视角")

    # 逐视角带内−带外对比度（与匹配滤波同式）
    cols = np.arange(d.shape[1], dtype=np.float64)
    angles = np.arange(360.0)
    traj = S.TRAJ["offset"] + S.TRAJ["amp"] * np.cos(
        np.radians(angles + S.TRAJ["phase_deg"]))
    dl = np.empty(360)
    for a in range(360):
        m = np.abs(cols - traj[a]) <= 60.0
        dl[a] = d[a][m].mean() - d[a][~m].mean()

    # 逐视角绝对误差、相对误差（= rmse / rms(L) 同 mask）、以及 |L| 的 rms
    rmse_abs = np.empty(360)
    rms_l = np.empty(360)
    rel = np.empty(360)
    for a in range(360):
        m = mask[a]
        rmse_abs[a] = math.sqrt(np.mean(err[a][m] ** 2))
        rms_l[a] = math.sqrt(np.mean(sino_f[a][m] ** 2))
        rel[a] = rmse_abs[a] / rms_l[a]

    # 缺陷可见的角向支撑：必须相对"安静区"的噪声来判，不能相对 0
    # （ΔL 含两次仿真之间的全局乘性差，整体偏移约 −0.198，用 0 当阈值会严重低估支撑）
    def runs(mask: np.ndarray, min_len: int = 5) -> list[tuple[int, int]]:
        out = []
        dd = np.diff(np.r_[0, mask.astype(int), 0])
        for s, e in zip(np.where(dd == 1)[0], np.where(dd == -1)[0]):
            if e - s >= min_len:
                out.append((int(s), int(e - 1)))
        return out

    base = float(dl.mean())
    lobes = runs(dl > base)                       # 第一遍：粗定两瓣
    quiet = np.ones(360, dtype=bool)
    for s, e in lobes:
        quiet[s:e + 1] = False
    q_mu = float(dl[quiet].mean())
    q_sd = float(dl[quiet].std(ddof=1))
    thr = q_mu + 3.0 * q_sd                       # 3σ 判据
    lobes = runs(dl > thr)                        # 第二遍：按噪声阈值重定
    n_sig = sum(e - s + 1 for s, e in lobes)
    n_half = int((dl > base + 0.5 * (float(dl.max()) - base)).sum())
    dil = sorted([360.0 / n_sig, 360.0 / max(n_half, 1)])

    return {"d": d, "dl": dl, "rel": rel, "rmse_abs": rmse_abs, "rms_l": rms_l,
            "base": base, "quiet_mean": q_mu, "quiet_sd": q_sd, "threshold": thr,
            "peak": float(dl.max()), "lobes": lobes, "n_sig": n_sig,
            "n_half": n_half, "dilution_range": (dil[0], dil[1]),
            "angles": angles, "traj": traj, "mask": mask}


# ------------------------------------------------------------------ Fig 10
def fig10(pv: dict) -> list[Path]:
    ang = pv["angles"]
    d = pv["d"]
    dl = pv["dl"]
    rel = pv["rel"]
    lobes = pv["lobes"]
    n_sig, n_half = pv["n_sig"], pv["n_half"]
    dil_lo, dil_hi = pv["dilution_range"]
    dev = np.abs(((ang - 141.0 + 180.0) % 360.0) - 180.0)
    ins = dev <= ARC_HALF

    fig = plt.figure(figsize=(7.2, 2.72))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.55, 1.0, 1.0], wspace=0.34,
                          left=0.058, right=0.988, top=0.889, bottom=0.225)

    # (a) 差分正弦图 + 轨迹 + 两个可见瓣的角向位置
    ax = fig.add_subplot(gs[0, 0])
    vmax = float(np.percentile(np.abs(d), 99.5))
    ax.imshow(d, aspect="auto", cmap="RdBu_r", vmin=-vmax, vmax=vmax,
              extent=(0, d.shape[1], 360, 0), interpolation="nearest")
    ax.plot(pv["traj"], ang, color="#111111", lw=0.7, ls=(0, (3, 2)))
    for s, e in lobes:
        ax.axhspan(s, e, color="#FFD54F", alpha=0.30, lw=0)
    ax.set_xlabel("Detector column (px)")
    ax.set_ylabel("Projection angle $\\theta$ (deg)")
    ax.set_title("(a) Defect difference $\\Delta L(\\theta,\\,x)$", fontsize=7.2, pad=3)
    ax.set_xticks([0, 400, 800, 1200])
    ax.text(0.02, 0.97, "dashed: defect trajectory", transform=ax.transAxes, fontsize=5.6,
            va="top", ha="left", color="#111111")
    ax.text(0.02, 0.055, "shaded: lobes that carry signal", transform=ax.transAxes,
            fontsize=5.6, va="bottom", ha="left", color="#8A6D00")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)

    # (b) 逐视角带对比度
    ax = fig.add_subplot(gs[0, 1])
    for s, e in lobes:
        ax.axvspan(s, e, color="#FFD54F", alpha=0.35, lw=0)
    ax.plot(ang, dl, color="#222222", lw=0.8)
    ax.axhline(pv["base"], color="#B4483C", lw=0.8, ls=(0, (4, 2)))
    ax.set_xlim(0, 360); ax.set_xticks([0, 90, 180, 270, 360])
    ax.set_xlabel("Projection angle $\\theta$ (deg)")
    ax.set_ylabel("$\\Delta L(\\theta)$ band $-$ off-band")
    ax.set_title(f"(b) Signal spans {n_sig} of 360 views", fontsize=7.2, pad=3)
    ax.grid(color="#EDEDED", lw=0.5); ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.text(0.5, 0.05,
            f"lobes {lobes[0][0]}$-${lobes[0][1]}$^\\circ$ and "
            f"{lobes[1][0]}$-${lobes[1][1]}$^\\circ$\n"
            f"$\\Rightarrow$ dilution {dil_lo:.1f}$-${dil_hi:.1f}$\\times$",
            transform=ax.transAxes, ha="center", va="bottom", fontsize=5.8,
            bbox=dict(boxstyle="round,pad=0.26", fc="white", ec="#CCCCCC", lw=0.5))

    # (c) 逐视角相对误差
    ax = fig.add_subplot(gs[0, 2])
    for s, e in lobes:
        ax.axvspan(s, e, color="#FFD54F", alpha=0.35, lw=0)
    ax.plot(ang, rel, color="#B4483C", lw=0.8)
    ax.axhline(float(rel.mean()), color="#666666", lw=0.7, ls=(0, (4, 2)))
    ax.set_xlim(0, 360); ax.set_xticks([0, 90, 180, 270, 360])
    ax.set_xlabel("Projection angle $\\theta$ (deg)")
    ax.set_ylabel("RMSE$_\\mathrm{view}$ / RMS$(|L|)$")
    ax.set_title("(c) No error excess in the lobes", fontsize=7.2, pad=3)
    ax.grid(color="#EDEDED", lw=0.5); ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    r = float(np.corrcoef(pv["rmse_abs"], pv["rms_l"])[0, 1])
    ax.text(0.5, 0.955,
            f"in-lobe / out-lobe = {rel[ins].mean() / rel[~ins].mean():.2f}\n"
            f"abs. error $\\propto$ $|L|$,  $r={r:+.2f}$",
            transform=ax.transAxes, ha="center", va="top", fontsize=5.9,
            bbox=dict(boxstyle="round,pad=0.26", fc="white", ec="#CCCCCC", lw=0.5))

    out = []
    for ext in ("png", "pdf"):
        p = FIG / f"fig10_dilution_en.{ext}"
        fig.savefig(p, dpi=DPI)
        out.append(p)
    plt.close(fig)

    with open(FIG / "fig10_per_view.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["angle_deg", "dl_band_minus_offband", "rmse_row16",
                    "rms_abs_L", "relative_completion_error", "in_defect_lobe"])
        for a in range(360):
            w.writerow([a, f"{dl[a]:.6f}", f"{pv['rmse_abs'][a]:.6f}",
                        f"{pv['rms_l'][a]:.6f}", f"{rel[a]:.6f}", int(ins[a])])
    print(f"[data] lobes={lobes}  n_sig={n_sig}  n_half={n_half}  "
          f"dilution={dil_lo:.2f}-{dil_hi:.2f}  "
          f"rel in/out={rel[ins].mean():.4f}/{rel[~ins].mean():.4f}  "
          f"corr(abs err, |L|)={r:+.3f}")
    return out


def fig11() -> list[Path]:
    tm = load(SRC / "image_metrics.json")
    methods = [("dense", "dense ref."), ("cnn_completed", "CNN"),
               ("linear_interp", "linear"), ("fbp_sparse", "FBP"), ("sart_sparse", "SART")]
    cols = {"dense": "#8A8A8A", "cnn_completed": "#B4483C", "linear_interp": "#3A6EA5",
            "fbp_sparse": "#7FA8C9", "sart_sparse": "#C4A35A"}

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.42),
                             gridspec_kw={"width_ratios": [1.0, 1.25]})
    ax = axes[0]
    vals = [tm["defect_cnr"][k]["in_image"]["cnr"] for k, _ in methods]
    xs = np.arange(len(methods))
    ax.bar(xs, vals, 0.6, color=[cols[k] for k, _ in methods], linewidth=0, zorder=3)
    ax.axhline(5.0, color="#333333", lw=0.8, ls=(0, (4, 2)))
    ax.text(0.02, 5.08, "Rose criterion (CNR = 5)", fontsize=5.9,
            ha="left", va="bottom", color="#333333",
            bbox=dict(boxstyle="round,pad=0.20", fc="white", ec="none"))
    for i, v in enumerate(vals):
        ax.text(i, v + 0.07, f"{v:.2f}", ha="center", fontsize=6.3)
    ax.set_xticks(xs)
    ax.set_xticklabels([n for _, n in methods], rotation=18, ha="right")
    ax.set_ylim(0, 5.9)
    ax.set_ylabel("Cavity CNR")
    ax.set_title("(a) Cavity contrast-to-noise ratio", fontsize=7.2, pad=3)
    ax.grid(axis="y", color="#EDEDED", lw=0.5); ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)

    ax = axes[1]
    rods = tm["defect_cnr"]["cnn_completed"]["rod_centres"]
    names = [r["rod"] for r in rods]
    mu = [r["mu_centre"] for r in rods]
    isd = [bool(r["is_defect_rod"]) for r in rods]
    xr = np.arange(len(rods))
    ax.bar(xr, mu, 0.62, color=["#B4483C" if d else "#9E9E9E" for d in isd],
           linewidth=0, zorder=3)
    for i, v in enumerate(mu):
        ax.text(i, v + 0.00012, f"{v:.4f}", ha="center", fontsize=5.6, rotation=90)
    ax.set_xticks(xr); ax.set_xticklabels(names, rotation=38, ha="right")
    ax.set_ylim(0, max(mu) * 1.32)
    ax.set_ylabel("Rod centre value ($\\mu$)")
    ax.set_title("(b) Centre value of the nine rods (CNN)", fontsize=7.2, pad=3)
    ax.grid(axis="y", color="#EDEDED", lw=0.5); ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.text(0.985, 0.96, "red = rod containing the cavity", transform=ax.transAxes,
            fontsize=5.9, va="top", ha="right", color="#B4483C",
            bbox=dict(boxstyle="round,pad=0.20", fc="white", ec="none"))

    fig.subplots_adjust(left=0.075, right=0.988, top=0.885, bottom=0.30, wspace=0.30)
    out = []
    for ext in ("png", "pdf"):
        p = FIG / f"fig11_cnr_en.{ext}"
        fig.savefig(p, dpi=DPI)
        out.append(p)
    plt.close(fig)
    return out


def main() -> int:
    global RUN_ID, SRC, FIG
    ap = argparse.ArgumentParser(description="Fig 10（稀释效应）+ Fig 11（空腔 CNR）")
    ap.add_argument("--run-id", default=RUN_ID,
                    help="缺陷 run（默认 100 epoch 定稿版）")
    ap.add_argument("--out-dir", default=None,
                    help="输出目录名（默认 docs/figs/<run_id 去掉 _cnn_bs8_lr1e-4>_flaw_step3_100ep）")
    a = ap.parse_args()
    RUN_ID = a.run_id
    SRC = ROOT / "runs" / f"{RUN_ID}_flaw_step3"
    if a.out_dir:
        FIG = ROOT / "docs" / "figs" / a.out_dir
    else:
        tag = RUN_ID.replace("_cnn_bs8_lr1e-4", "")
        FIG = ROOT / "docs" / "figs" / f"p01_flaw_step3_{tag}"
    if not SRC.is_dir():
        raise SystemExit(f"缺目录：{SRC}")
    FIG.mkdir(parents=True, exist_ok=True)
    pv = per_view()
    for p in fig10(pv) + fig11():
        print(f"[OK] {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
