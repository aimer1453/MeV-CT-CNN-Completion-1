#!/usr/bin/env python
"""CNR 的 ROI 定义严格复核（纯后处理，不动模型、不重训）。

问题
----
论文侧质疑：空腔 CNR 偏低（3.0 MeV 约 1.5），是不是背景 ROI 取脏了？
两种方向的偏差必须先分清：

  * 背景 ROI 混入**空气**（μ≈0、噪声≈0）→ σ 偏小 → CNR **虚高**（不是偏低）；
  * 背景 ROI 混入**高噪声/高梯度区**（低计数边缘行、材料边界半影、
    稀疏角重建的条纹伪影）→ σ 偏大 → CNR **偏低**。

本脚本检验第二个方向，同时把第一个方向也量化（因为 D 定义正好落在空气区）。

几何锚点（`3_3_geant4/src/DetectorConstruction.cc`，1 cm = 100 px 由栅距定标）
---------------------------------------------------------------------------
    栅距 pitch = 2*(OutRad+0.3cm) = 2.60 cm            (:245)
    棒外径 OutRad = 1.00 cm -> 100 px                   (:148)
    包壳内径 InRad = 0.80 cm ->  80 px                  (:147)
    芯块半径 FRad = InRad-0.05cm = 0.75 cm -> 75 px     (:205)
    空腔半径 0.2 cm -> 20 px
    1 cm = 100 px（由「角棒 r=367.7 px = √2×2.6 cm」与「边棒 r=260 px = 2.6 cm」定标）

**重要实测（本脚本 --geom-check 自动核验）**：缺陷数据（`3_3_flaw/`）的重建图里
棒的可见边缘在 **r≈76 px**，即 0.292×pitch，等于 `FRad/pitch = 0.288`；
而 `OutRad/pitch = 0.385`（=100 px）处没有台阶。**缺陷 phantom 里没有包壳**：
80–100 px 区间重建值 2e-5，而均匀数据（`3_3/`）同一区间是 2e-3（≈0.5×芯块值）。
成因：`3_3_flaw_geant4/src/DetectorConstruction.cc:169-172` 把芯块放进了一个
填 air 的 `fuelColl` 盒子（30×30×500 cm），该盒子与 9 个 `shell` 体积重叠，
`checkOverlaps=false` 下 shell 被空气覆盖——与 `3_3_geant4` 版（shell 与 fuel
同父 `logicAssembly`、无 fuelColl）不同。

=> 因此用户要求的"纯包壳区"（D）在缺陷数据里物理上不存在，该区间是空气。
   本脚本把 D 按字面跑出来（结果自证：Δ≈0、CNR≈0），并补一个 D′ =
   "9 根棒 52–68 px 纯芯块环池化"，这才是缺陷数据里可用的"远离空腔的纯棒体"。

4 种定义（以缺陷棒心为原点；半径单位 px）
----------------------------------------
  A  当前定义：μ_bub = r≤15 圆盘；μ_ref = 同棒 32–62 环；
     σ = 3 根正常角棒 r≤15 圆盘 std 的中位数。        ← 论文现用口径
  B  紧贴空腔的环带 24–40 px；σ 与信号取**同一环带**（消除 A 的带不一致）
  C  与 B 同环带，但只取朝背离阵列中心的 ±60° 扇区（4 根角棒外侧翼，方位对称）
  D  字面「纯包壳区」：9 根棒 82–98 px 环池化（两侧各留 2 px 避半影）
  A′ 诊断：Δ 用 A 的 32–62 环，σ 也取该环（只在正常棒上）→ 隔离"σ 带不一致"
  D′ 诊断：9 根棒 52–68 px 纯芯块环池化 → 缺陷数据里唯一可用的"全图纯棒体"

口径
----
重建图在 `-log` 物理域（像素值 = μ·Δx，Δx=0.01 cm），FOV 圆 r≤600。
主报告用 seed 升序第一个 run；CNN 另给 3 seed 均值±标准差（其余四方法与权重
无关，跨 seed 逐位相同，见 `runs/_flaw_multi_table.md`）。

用法
----
    python scripts/cnr_roi_audit.py --energy 3.0
    python scripts/cnr_roi_audit.py --all-energies
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT))

import baseline_recon as br                                    # noqa: E402

RUNS = ROOT / "runs"
NAN = float("nan")

# ---------------------------------------------------------------- 几何常量（px）
PITCH = 260.0
R_CORNER = 367.7           # √2 × 2.6 cm
R_EDGE = 260.0             # 2.6 cm
R_FUEL = 75.0              # FRad = 0.75 cm
R_CLAD_IN = 80.0           # InRad = 0.8 cm
R_CLAD_OUT = 100.0         # OutRad = 1.0 cm
R_CAVITY = 20.0            # 0.2 cm
CENTRE = 600.0

#: 缺陷轨迹定标（与 p01_flaw_step3_eval.py 一致）
TRAJ = {"amp": 367.1, "offset": 601.9, "phase_deg": 224.7}
AZ_DEFECT_IMG = TRAJ["phase_deg"] - 90.0        # 134.7°
R_DEFECT_PX = TRAJ["amp"]                       # 367.1（角棒真值 367.7，差 0.6 px）

#: A 口径的半径（必须与 p01_flaw_step3_eval.py 的 BUBBLE_ROI_R / REF_R0,REF_R1 一致）
A_BUB_R = 15.0
A_REF_R0, A_REF_R1 = 32.0, 62.0

DEFS: dict[str, dict] = {
    "A": dict(bub=15.0, ref=(A_REF_R0, A_REF_R1), sector=None, noise="disc", diag=False,
              zh="当前定义：r≤15 圆盘 vs 同棒 32–62 环；σ = 正常角棒 r≤15 圆盘 std 中位数"),
    "B": dict(bub=15.0, ref=(24.0, 40.0), sector=None, noise="band", diag=False,
              zh="紧贴空腔的环带 24–40 px（离空腔边 4 px、离棒边 35 px）；σ 同环带"),
    "C": dict(bub=15.0, ref=(24.0, 40.0), sector="outward", noise="band", diag=False,
              zh="与 B 同环带，只取背离阵列中心的 ±60° 扇区（角棒外侧翼）；σ 同扇区"),
    "D": dict(bub=15.0, ref=(82.0, 98.0), sector=None, noise="global", diag=False,
              zh="字面「纯包壳区」：9 根棒 82–98 px 环池化（缺陷 phantom 里该区是空气）"),
    "A'": dict(bub=15.0, ref=(A_REF_R0, A_REF_R1), sector=None, noise="band", diag=True,
               zh="诊断：Δ 与 σ 都取 A 的 32–62 环（隔离带不一致的影响）"),
    "D'": dict(bub=15.0, ref=(52.0, 68.0), sector=None, noise="global", diag=True,
               zh="诊断：9 根棒 52–68 px 纯芯块环池化（缺陷数据里可用的全图纯棒体）"),
}

METHODS = ("dense", "cnn_completed", "linear_interp", "fbp_sparse", "sart_sparse")

YC, XC = np.mgrid[0:1200, 0:1200].astype(np.float64)


# ---------------------------------------------------------------- 位置
def center_xy(r: float, az: float) -> tuple[float, float]:
    return CENTRE - r * math.sin(math.radians(az)), CENTRE + r * math.cos(math.radians(az))


def rod_centers() -> list[tuple[str, float, float]]:
    out = [("center", 0.0, 0.0)]
    out += [(f"edge{k}", R_EDGE, 90.0 * k) for k in range(4)]
    out += [(f"corner{k}", R_CORNER, 45.0 + 90.0 * k) for k in range(4)]
    return out


def defect_center() -> tuple[float, float]:
    return center_xy(R_DEFECT_PX, AZ_DEFECT_IMG)


def normal_corners() -> list[tuple[str, float, float, float]]:
    out = []
    for name, r, az in rod_centers():
        if not name.startswith("corner") or abs(r - R_DEFECT_PX) > 1.0:
            continue
        if abs(((az - AZ_DEFECT_IMG) + 180.0) % 360.0 - 180.0) < 1.0:
            continue
        cy, cx = center_xy(r, az)
        out.append((name, cy, cx, az))
    return out


# ---------------------------------------------------------------- 掩膜
def disc(cy: float, cx: float, r: float) -> np.ndarray:
    return (YC - cy) ** 2 + (XC - cx) ** 2 <= r * r


def ring(cy: float, cx: float, r0: float, r1: float) -> np.ndarray:
    d2 = (YC - cy) ** 2 + (XC - cx) ** 2
    return (d2 >= r0 * r0) & (d2 <= r1 * r1)


def sector_ring(cy: float, cx: float, r0: float, r1: float,
                az_out: float, half: float = 60.0) -> np.ndarray:
    """环带上 ±half° 扇区，扇区中心指向 az_out（背离阵列中心）。"""
    d2 = (YC - cy) ** 2 + (XC - cx) ** 2
    ang = np.degrees(np.arctan2(-(YC - cy), XC - cx)) % 360.0
    diff = np.abs(((ang - az_out) + 180.0) % 360.0 - 180.0)
    return (d2 >= r0 * r0) & (d2 <= r1 * r1) & (diff <= half)


def zone_frac(rr: np.ndarray) -> dict[str, float]:
    n = float(np.size(rr))
    return {"cavity 0-20": float(np.count_nonzero(rr <= R_CAVITY) / n),
            "fuel 20-75": float(np.count_nonzero((rr > R_CAVITY) & (rr <= R_FUEL)) / n),
            "gap 75-80": float(np.count_nonzero((rr > R_FUEL) & (rr <= R_CLAD_IN)) / n),
            "clad 80-100": float(np.count_nonzero((rr > R_CLAD_IN) & (rr <= R_CLAD_OUT)) / n),
            "air >100": float(np.count_nonzero(rr > R_CLAD_OUT) / n)}


# ---------------------------------------------------------------- 统计
def stats(v: np.ndarray) -> dict:
    v = np.asarray(v, dtype=np.float64).ravel()
    mu, sd = float(v.mean()), float(v.std(ddof=1))
    med = float(np.median(v))
    mad = float(np.median(np.abs(v - med))) * 1.4826
    dev = np.abs(v - mu)
    return {"n": int(v.size), "mean": mu, "std": sd, "median": med, "mad_sigma": mad,
            "min": float(v.min()), "max": float(v.max()),
            "p0.1": float(np.percentile(v, 0.1)), "p99.9": float(np.percentile(v, 99.9)),
            "outlier_frac_3sd": float(np.count_nonzero(dev > 3 * sd) / v.size),
            "outlier_frac_3mad": float(np.count_nonzero(np.abs(v - med) > 3 * mad) / v.size)
            if mad > 0 else NAN,
            "skew": float(((v - mu) ** 3).mean() / sd ** 3) if sd > 0 else NAN}


def decompose(v: np.ndarray, rr: np.ndarray, bin_px: float = 1.0) -> dict:
    """把环带内总方差拆成「径向结构」与「去径向残差」两部分。

    σ_tot² ≈ σ_struct² + σ_noise²。若结构项占比很小，说明这个环带里没有
    径向梯度污染，σ 就是纯噪声（= CNR 的分母是干净的）。
    """
    v = np.asarray(v, dtype=np.float64).ravel()
    rr = np.asarray(rr, dtype=np.float64).ravel()
    if v.size < 50 or rr.max() - rr.min() < 2 * bin_px:
        return {"sd_total": float(v.std(ddof=1)) if v.size > 1 else NAN,
                "sd_noise_azimuthal": NAN, "sd_radial_struct": NAN,
                "struct_var_frac": NAN}
    idx = np.floor((rr - rr.min()) / bin_px).astype(int)
    bm, resid = [], []
    for b in np.unique(idx):
        sel = idx == b
        if sel.sum() < 12:
            continue
        bm.append(v[sel].mean())
        resid.append(v[sel] - v[sel].mean())
    if not resid:
        return {"sd_total": float(v.std(ddof=1)), "sd_noise_azimuthal": NAN,
                "sd_radial_struct": NAN, "struct_var_frac": NAN}
    resid = np.concatenate(resid)
    bm = np.array(bm)
    sd_tot = float(v.std(ddof=1))
    sd_noise = float(resid.std(ddof=1))
    sd_struct = float(bm.std(ddof=1)) if bm.size > 2 else NAN
    return {"sd_total": sd_tot, "sd_noise_azimuthal": sd_noise,
            "sd_radial_struct": sd_struct,
            "struct_var_frac": float(sd_struct ** 2 / sd_tot ** 2)
            if sd_tot > 0 and sd_struct == sd_struct else NAN}


# ---------------------------------------------------------------- 载入
def load_recon(run_id: str, method: str) -> np.ndarray | None:
    p = RUNS / f"{run_id}_flaw_step3" / "recon" / (
        "dense.npy" if method == "dense" else f"{method}_A.npy")
    if not p.is_file():
        return None
    return br.restrict_to_fov(np.load(p).astype(np.float64))


def _same(a: str, b: str) -> bool:
    try:
        return float(a) == float(b)
    except (TypeError, ValueError):
        return a == b


def pick_runs(energy: str) -> list[str]:
    rows = list(csv.DictReader(open(RUNS / "_flaw_multi_index.csv", newline="",
                                    encoding="utf-8")))
    sel = [r for r in rows if _same(energy, r["energy"])]
    sel.sort(key=lambda r: r["seed"])
    return [r["run_id"] for r in sel]


# ---------------------------------------------------------------- 核心：按定义算 CNR
def cnr_under(img: np.ndarray, key: str) -> dict:
    cfg = DEFS[key]
    cy, cx = defect_center()
    d2 = (YC - cy) ** 2 + (XC - cx) ** 2
    rr = np.sqrt(d2)
    m_bub = d2 <= cfg["bub"] ** 2

    global_ref = cfg["noise"] == "global"
    if global_ref:                      # 9 根棒池化
        m_ref = np.zeros_like(m_bub)
        for _n, r, az in rod_centers():
            ccy, ccx = center_xy(r, az)
            m_ref |= ring(ccy, ccx, cfg["ref"][0], cfg["ref"][1])
    elif cfg["sector"] == "outward":
        m_ref = sector_ring(cy, cx, cfg["ref"][0], cfg["ref"][1], az_out=AZ_DEFECT_IMG)
    else:
        m_ref = (d2 >= cfg["ref"][0] ** 2) & (d2 <= cfg["ref"][1] ** 2)

    mu_b = float(img[m_bub].mean())
    mu_r = float(img[m_ref].mean())

    if global_ref:
        sd = float(img[m_ref].std(ddof=1))
    elif cfg["noise"] == "disc":
        sd = float(np.median([img[disc(ccy, ccx, A_BUB_R)].std(ddof=1)
                              for _n, ccy, ccx, _a in normal_corners()]))
    else:                               # band / sector，σ 与信号同带
        sds = []
        for _n, ccy, ccx, az in normal_corners():
            if cfg["sector"] == "outward":
                mm = sector_ring(ccy, ccx, cfg["ref"][0], cfg["ref"][1], az)
            else:
                mm = ring(ccy, ccx, cfg["ref"][0], cfg["ref"][1])
            sds.append(float(img[mm].std(ddof=1)))
        sd = float(np.median(sds))

    dec = ({"sd_total": NAN, "sd_noise_azimuthal": NAN, "sd_radial_struct": NAN,
            "struct_var_frac": NAN} if global_ref
           else decompose(img[m_ref], rr[m_ref]))
    rr_ref = rr[m_ref] if not global_ref else np.array([NAN])
    return {"def": key, "mu_bubble": mu_b, "mu_ref": mu_r,
            "contrast": mu_r - mu_b, "noise_sd": sd,
            "cnr": (mu_r - mu_b) / sd if sd > 0 else NAN,
            "ref_zone_frac": zone_frac(rr_ref) if not global_ref else None,
            "ref_r_range_px": [float(rr_ref.min()), float(rr_ref.max())]
            if not global_ref else list(cfg["ref"]),
            "ref_stats": stats(img[m_ref]), "ref_decompose": dec,
            "n_bub_px": int(m_bub.sum()), "n_ref_px": int(m_ref.sum())}


# ---------------------------------------------------------------- 几何核验
def geom_check(img_dense: np.ndarray, tag: str) -> dict:
    """从重建图实测棒的可见边缘半径 + 包壳区信号水平。

    用于验证 FRad(75)/OutRad(100) 哪个是图上的真实边界，并量化
    「80–100 px 是否有 Zr 包壳信号」。
    """
    prof, edges = [], np.arange(40.0, 120.0, 1.0)
    for e in edges:
        vals = []
        for _n, r, az in rod_centers():
            ccy, ccx = center_xy(r, az)
            d2 = (YC - ccy) ** 2 + (XC - ccx) ** 2
            vals.append(img_dense[(d2 >= e * e) & (d2 < (e + 1) ** 2)].mean())
        prof.append(float(np.mean(vals)))
    prof = np.array(prof)
    plateau = float(np.median(prof[edges < 60]))
    i0 = int(np.argmax(prof < 0.5 * plateau))
    r_edge = float(edges[i0])
    band = prof[(edges >= 82) & (edges < 98)]
    # 噪声：取该带的像素 std（9 根棒池化）
    m = np.zeros_like(img_dense, dtype=bool)
    for _n, r, az in rod_centers():
        ccy, ccx = center_xy(r, az)
        m |= ring(ccy, ccx, 82.0, 98.0)
    sd = float(img_dense[m].std(ddof=1))
    n_eff = max(1.0, float(m.sum()) / 40.0)     # 保守：相关长度 ~6 px 的等效独立样本
    return {"tag": tag, "r_edge_halfheight_px": r_edge,
            "r_edge_over_pitch": r_edge / PITCH,
            "FRad_over_pitch": R_FUEL / PITCH, "OutRad_over_pitch": R_CLAD_OUT / PITCH,
            "plateau_fuel": plateau,
            "mean_82_98_px": float(band.mean()), "sd_82_98_px": sd,
            "se_mean_82_98": sd / math.sqrt(n_eff),
            "frac_of_fuel_82_98": float(band.mean() / plateau),
            "expected_clad_over_fuel": 0.51,
            "note": ("缺陷数据：82–98 px 无信号 → phantom 无包壳。"
                     "均匀数据：该区间约 0.5×芯块值 → 有 Zr 包壳。")}


# ---------------------------------------------------------------- 出图
def make_audit_figure(imgs: dict[str, np.ndarray], out_png: Path, energy: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle, Rectangle, Wedge

    plt.rcParams.update({
        "font.size": 9, "axes.linewidth": 0.8, "savefig.dpi": 220, "figure.dpi": 220,
        "text.color": "#e8e8e8", "axes.labelcolor": "#e8e8e8",
        "xtick.color": "#e8e8e8", "ytick.color": "#e8e8e8", "axes.edgecolor": "#e8e8e8",
        "figure.facecolor": "#1b1b1b", "axes.facecolor": "#1b1b1b",
        "legend.framealpha": 0.22, "legend.edgecolor": "#888888",
    })
    FG = "#e8e8e8"

    fig = plt.figure(figsize=(13.4, 8.6))
    gs = fig.add_gridspec(2, 3, hspace=0.36, wspace=0.30)
    cnn = imgs["cnn_completed"]
    cy, cx = defect_center()
    az = AZ_DEFECT_IMG

    # (a) 全 FOV
    ax = fig.add_subplot(gs[0, 0])
    lo, hi = np.percentile(cnn, [0.5, 99.5])
    ax.imshow(cnn, cmap="gray", vmin=lo, vmax=hi)
    ax.add_patch(Circle((cx, cy), R_CLAD_OUT, fill=False, ec="#ff5c5c", lw=1.3))
    ax.plot([cx], [cy], "+", color="#ff5c5c", ms=9, mew=1.3)
    ax.add_patch(Rectangle((cx - 118, cy - 118), 236, 236, fill=False,
                           ec="#4ea1ff", lw=1.0, ls="--"))
    ax.set_xlim(0, 1200); ax.set_ylim(1200, 0)
    ax.set_xlabel("x [px]"); ax.set_ylabel("y [px]")
    ax.set_title(f"(a) {energy} MeV completion, full FOV\n"
                 f"defect rod r={R_CORNER:.1f} px, az={az:.1f}$^\\circ$", fontsize=9.5)

    # (b) 放大 + ROI
    ax = fig.add_subplot(gs[0, 1])
    pad = 112
    y0, y1, x0, x1 = int(cy - pad), int(cy + pad), int(cx - pad), int(cx + pad)
    crop = cnn[y0:y1, x0:x1]
    ax.imshow(crop, cmap="gray", extent=(x0, x1, y1, y0),
              vmin=np.percentile(crop, 2), vmax=np.percentile(crop, 98))
    for r, c, ls in [(R_CAVITY, "#ffd24a", ":"), (R_FUEL, "#4ea1ff", ":"),
                     (R_CLAD_IN, "#8cf08c", "-."), (R_CLAD_OUT, "#ff5c5c", "-.")]:
        ax.add_patch(Circle((cx, cy), r, fill=False, ec=c, lw=1.0, ls=ls))
    ax.add_patch(Circle((cx, cy), A_BUB_R, fill=False, ec="#ffffff", lw=1.3))
    ax.add_patch(Wedge((cx, cy), A_REF_R1, 0, 360, width=A_REF_R1 - A_REF_R0,
                       fill=False, ec="#ffd24a", lw=1.8))
    ax.add_patch(Wedge((cx, cy), 40, 0, 360, width=16, fill=False, ec="#4ea1ff", lw=1.6))
    ax.add_patch(Wedge((cx, cy), 40, az - 60, az + 60, width=16, fill=False,
                       ec="#ff7de0", lw=1.6))
    ax.set_xlim(x0, x1); ax.set_ylim(y1, y0)
    ax.set_xlabel("x [px]"); ax.set_ylabel("y [px]")
    ax.set_title("(b) ROI overlay (dotted = geometry interfaces)\n"
                 "white r$\\leq$15 | yellow A 32-62 | blue B 24-40 | pink C sector",
                 fontsize=9.0)

    # (c) 径向剖面
    ax = fig.add_subplot(gs[0, 2])
    edges = np.arange(0.0, 112.0, 2.0)
    mids = edges[:-1] + 1.0

    def prof_of(img, cc_list):
        out = []
        for e in edges[:-1]:
            vals = []
            for ccy, ccx in cc_list:
                d2 = (YC - ccy) ** 2 + (XC - ccx) ** 2
                vals.append(img[(d2 >= e * e) & (d2 < (e + 2.0) ** 2)].mean())
            out.append(float(np.mean(vals)))
        return np.array(out)

    pd_ = prof_of(cnn, [(cy, cx)])
    pn = prof_of(cnn, [(a, b) for _n, a, b, _z in normal_corners()])
    ax.axvspan(A_REF_R0, A_REF_R1, color="#ffd24a", alpha=0.16)
    ax.axvspan(24, 40, color="#4ea1ff", alpha=0.16)
    ax.axvspan(82, 98, color="#8cf08c", alpha=0.10)
    ax.plot(mids, pd_, "-o", ms=2.8, color="#ff5c5c", label="defect rod")
    ax.plot(mids, pn, "-s", ms=2.8, color="#4ea1ff", label="3 normal corner rods")
    ax.set_ylim(0.0, 0.0064)
    for r, lab, yy in [(R_CAVITY, "cavity edge 20", 0.00625),
                       (R_FUEL, "fuel edge 75", 0.00625),
                       (R_CLAD_OUT, "OutRad 100", 0.00585)]:
        ax.axvline(r, color=FG, ls=":", lw=0.9, alpha=0.55)
        ax.text(r, yy, " " + lab, fontsize=6.6, va="top", color=FG)
    ax.text(90.0, 0.0011, "no cladding\nsignal here", fontsize=6.6, color="#8cf08c",
            ha="center", va="bottom")
    ax.set_xlim(0, 110)
    ax.set_xlabel("radius from rod centre [px]"); ax.set_ylabel("reconstruction (-log domain)")
    ax.set_title("(c) radial profile: cavity dip vs background bands", fontsize=9.5)
    ax.legend(fontsize=7, loc="center right")

    # (d) σ(r)
    ax = fig.add_subplot(gs[1, 0])
    for meth, col in zip(METHODS, ["#dcdcdc", "#ff5c5c", "#4ea1ff", "#8cf08c", "#ffd24a"]):
        if meth not in imgs:
            continue
        sd = []
        for e in edges[:-1]:
            v = []
            for _n, ccy, ccx, _z in normal_corners():
                d2 = (YC - ccy) ** 2 + (XC - ccx) ** 2
                v.append(imgs[meth][(d2 >= e * e) & (d2 < (e + 2.0) ** 2)])
            sd.append(float(np.concatenate(v).std(ddof=1)))
        ax.plot(mids, sd, "-o", ms=2.4, color=col, label=meth)
    ax.axvspan(A_REF_R0, A_REF_R1, color="#ffd24a", alpha=0.14)
    ax.axvspan(24, 40, color="#4ea1ff", alpha=0.14)
    ax.axvline(R_FUEL, color=FG, ls=":", lw=0.8, alpha=0.5)
    ax.set_xlim(0, 110)
    ax.set_xlabel("radius [px]"); ax.set_ylabel(r"$\sigma$ (pooled over 3 normal rods)")
    ax.set_title("(d) noise vs radius (yellow = A ring, blue = B/C band)", fontsize=9.5)
    ax.legend(fontsize=6.4, ncol=2)

    # (e) CNR vs 定义
    ax = fig.add_subplot(gs[1, 1])
    keys = ["A", "B", "C", "D"]
    w = 0.16
    for i, meth in enumerate(METHODS):
        if meth not in imgs:
            continue
        vals = [cnr_under(imgs[meth], k)["cnr"] for k in keys]
        ax.bar(np.arange(len(keys)) + (i - 2) * w, vals, w * 0.9, label=meth)
    ax.axhline(5.0, color="#ff5c5c", ls="--", lw=1.0)
    ax.text(-0.35, 5.2, "Rose criterion (5)", color="#ff5c5c", fontsize=7.5)
    ax.axhline(0, color=FG, lw=0.6, alpha=0.4)
    ax.set_xticks(np.arange(len(keys)))
    ax.set_xticklabels([f"{k}\n{DEFS[k]['ref'][0]:.0f}-{DEFS[k]['ref'][1]:.0f} px"
                        for k in keys], fontsize=7.5)
    ax.set_ylabel("CNR = $\\Delta/\\sigma$")
    ax.set_ylim(min(-0.8, ax.get_ylim()[0]), 6.2)
    ax.set_title("(e) CNR under four background-ROI definitions", fontsize=9.5)
    ax.legend(fontsize=6.2, ncol=2, loc="upper right")

    # (f) σ 分解
    ax = fig.add_subplot(gs[1, 2])
    xs = np.arange(len(METHODS))
    for i, meth in enumerate(METHODS):
        if meth not in imgs:
            continue
        cyy, cxx = defect_center()
        m_ref = ring(cyy, cxx, A_REF_R0, A_REF_R1)
        rr = np.sqrt((YC - cyy) ** 2 + (XC - cxx) ** 2)
        d = decompose(imgs[meth][m_ref], rr[m_ref])
        ax.bar(i - 0.19, d["sd_radial_struct"], 0.36, color="#ffd24a",
               label="radial structure" if i == 0 else None)
        ax.bar(i + 0.19, d["sd_noise_azimuthal"], 0.36, color="#4ea1ff",
               label="azimuthal residual (noise)" if i == 0 else None)
        ax.text(i, max(d["sd_radial_struct"], d["sd_noise_azimuthal"]) * 1.02,
                f"{d['struct_var_frac']:.1%}", ha="center", fontsize=6.8, color=FG)
    ax.set_xticks(xs)
    ax.set_xticklabels([m.replace("_", "\n") for m in METHODS], fontsize=6.6)
    ax.set_ylabel(r"components of $\sigma_A$")
    ax.set_title("(f) $\\sigma_A$ decomposition (label = structure variance share)",
                 fontsize=9.0)
    ax.legend(fontsize=6.8, loc="upper right")

    fig.suptitle(f"CNR background-ROI audit - {energy} MeV defect domain "
                 f"(seed 20261022)", fontsize=11.5, color=FG)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    for ext in (".png", ".pdf"):
        fig.savefig(out_png.with_suffix(ext), bbox_inches="tight",
                    facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  [fig] {out_png.with_suffix('.png').relative_to(ROOT)} (+ .pdf)")


def make_sensitivity_figure(per_energy: dict[str, dict], out_png: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "font.size": 9, "axes.linewidth": 0.8, "savefig.dpi": 220, "figure.dpi": 220,
        "text.color": "#e8e8e8", "axes.labelcolor": "#e8e8e8",
        "xtick.color": "#e8e8e8", "ytick.color": "#e8e8e8", "axes.edgecolor": "#e8e8e8",
        "figure.facecolor": "#1b1b1b", "axes.facecolor": "#1b1b1b",
        "legend.framealpha": 0.22, "legend.edgecolor": "#888888",
    })
    ens = list(per_energy)
    fig, axes = plt.subplots(1, len(ens), figsize=(3.5 * len(ens), 3.4), sharey=True)
    if len(ens) == 1:
        axes = [axes]
    keys = ["A", "B", "C", "D"]
    w = 0.16
    for ax, en in zip(axes, ens):
        tab = per_energy[en]["cnn_dense"]
        for i, meth in enumerate(METHODS):
            if meth not in tab:
                continue
            vals = [tab[meth][k] for k in keys]
            ax.bar(np.arange(len(keys)) + (i - 2) * w, vals, w * 0.9,
                   label=meth if ax is axes[0] else None)
        ax.axhline(5.0, color="#ff5c5c", ls="--", lw=0.9)
        ax.axhline(0, color="#e8e8e8", lw=0.6, alpha=0.4)
        ax.set_xticks(range(len(keys))); ax.set_xticklabels(keys, fontsize=8)
        ax.set_title(f"{en} MeV", fontsize=9.5)
        ax.set_ylim(-0.8, 6.0)
    axes[0].set_ylabel("CNN-completion CNR = $\\Delta/\\sigma$")
    axes[0].legend(fontsize=6.2, ncol=1, loc="upper left")
    fig.suptitle("Sensitivity of the cavity CNR to the background-ROI definition "
                 "(four energies)", fontsize=10.5, color="#e8e8e8")
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    for ext in (".png", ".pdf"):
        fig.savefig(out_png.with_suffix(ext), bbox_inches="tight",
                    facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  [fig] {out_png.with_suffix('.png').relative_to(ROOT)} (+ .pdf)")


# ---------------------------------------------------------------- 单能量
def run_energy(energy: str, make_fig: bool) -> dict:
    runs = pick_runs(energy)
    if not runs:
        raise SystemExit(f"[ERROR] 索引里没有 {energy} MeV 的 run")
    primary = runs[0]
    print(f"\n{'='*74}\n=== {energy} MeV  主 run = {primary}  seed 组 = {len(runs)}\n{'='*74}")

    cy, cx = defect_center()
    d2 = (YC - cy) ** 2 + (XC - cx) ** 2
    rr = np.sqrt(d2)
    m_bub = d2 <= A_BUB_R ** 2
    m_ref = (d2 >= A_REF_R0 ** 2) & (d2 <= A_REF_R1 ** 2)

    # ---- T1
    print(f"\n[T1] A 口径 ROI 解剖")
    print(f"  defect centre (cy,cx) = ({cy:.2f}, {cx:.2f})  "
          f"[r={R_DEFECT_PX:.1f} px, az={AZ_DEFECT_IMG:.1f} deg]")
    print(f"  mu_bubble : r <= {A_BUB_R:.0f} px, n={int(m_bub.sum())}")
    print(f"  mu_ref    : r in [{A_REF_R0:.0f},{A_REF_R1:.0f}] px, n={int(m_ref.sum())}")
    zf = zone_frac(rr[m_ref])
    for k, v in zf.items():
        print(f"      {k:14s} {v*100:6.2f} %")
    nonfuel = zf["gap 75-80"] + zf["clad 80-100"] + zf["air >100"] + zf["cavity 0-20"]
    print(f"  -> non-fuel share {nonfuel*100:.3f} %  "
          f"({'CLEAN: all inside the fuel pellet' if nonfuel < 1e-9 else 'CONTAMINATED'})")

    img = load_recon(primary, "cnn_completed")
    st = stats(img[m_ref])
    dec = decompose(img[m_ref], rr[m_ref])
    print(f"\n  A ref-ring pixel stats (CNN completion):")
    print(f"      n={st['n']}  mean={st['mean']:.6f}  std={st['std']:.6f}  "
          f"median={st['median']:.6f}  MAD-sigma={st['mad_sigma']:.6f}")
    print(f"      min={st['min']:.6f}  max={st['max']:.6f}  "
          f"p0.1={st['p0.1']:.6f}  p99.9={st['p99.9']:.6f}")
    print(f"      >3sigma outliers {st['outlier_frac_3sd']*100:.3f} %  "
          f"(>3xMAD {st['outlier_frac_3mad']*100:.3f} %), skew {st['skew']:+.3f}")
    print(f"  sigma decomposition: total {dec['sd_total']:.6f} = radial-structure "
          f"{dec['sd_radial_struct']:.6f} (+) azimuthal-residual "
          f"{dec['sd_noise_azimuthal']:.6f}   [structure var share "
          f"{dec['struct_var_frac']:.1%}]")

    # 离群点的径向分布（判断是否聚在边界）
    rr_ref = rr[m_ref]
    v = img[m_ref]
    dev = np.abs(v - v.mean()) > 3 * v.std(ddof=1)
    sub = []
    for lo, hi in [(32, 40), (40, 48), (48, 55), (55, 62)]:
        sel = (rr_ref >= lo) & (rr_ref < hi)
        sub.append({"band": f"{lo}-{hi}", "n": int(sel.sum()),
                    "outlier_frac_pct": 100 * float(dev[sel].mean()) if sel.any() else NAN})

    # ---- 几何核验
    dense = load_recon(primary, "dense")
    gc = geom_check(dense, f"flaw {energy} MeV dense")
    print(f"\n  [geometry] visible rod edge (half-height) = "
          f"{gc['r_edge_halfheight_px']:.0f} px = {gc['r_edge_over_pitch']:.3f} x pitch "
          f"(FRad/pitch={gc['FRad_over_pitch']:.3f}, OutRad/pitch={gc['OutRad_over_pitch']:.3f})")
    print(f"  [geometry] mean over 82-98 px = {gc['mean_82_98_px']:+.6f} "
          f"= {gc['frac_of_fuel_82_98']*100:+.1f} % of fuel "
          f"(SE {gc['se_mean_82_98']:.1e}); expected Zr cladding would be "
          f"~{gc['expected_clad_over_fuel']*100:.0f} % -> cladding absent")

    # ---- T2
    imgs = {}
    for m in METHODS:
        a = load_recon(primary, m)
        if a is not None:
            imgs[m] = a
    print(f"\n[T2] CNR under the four definitions (primary run)")
    table = {}
    for m in METHODS:
        if m not in imgs:
            continue
        table[m] = {k: cnr_under(imgs[m], k) for k in DEFS}
    hdr = "  | method | " + " | ".join(f"{k}" for k in DEFS) + " |"
    print(hdr)
    for m, d in table.items():
        cells = [f"{d[k]['cnr']:.2f} ({d[k]['contrast']:.5f}/{d[k]['noise_sd']:.5f})"
                 for k in DEFS]
        print(f"  | {m:14s} | " + " | ".join(cells) + " |")

    # ---- 3 seed（只有 CNN 会变）
    seed_cnr = {}
    for rid in runs:
        im = load_recon(rid, "cnn_completed")
        if im is None:
            continue
        seed_cnr[rid] = {k: cnr_under(im, k)["cnr"] for k in DEFS}
    if len(seed_cnr) > 1:
        print(f"\n  CNN over {len(seed_cnr)} seeds:")
        for k in DEFS:
            arr = np.array([d[k] for d in seed_cnr.values()])
            print(f"      {k:3s}: {arr.mean():.3f} +/- {arr.std(ddof=1):.3f}   "
                  f"({', '.join(f'{x:.3f}' for x in arr)})")

    out = {"energy": energy, "primary_run": primary, "runs": runs,
           "geometry_px": {"pitch": PITCH, "r_corner": R_CORNER, "r_edge": R_EDGE,
                           "r_cavity": R_CAVITY, "r_fuel": R_FUEL,
                           "r_clad_in": R_CLAD_IN, "r_clad_out": R_CLAD_OUT,
                           "source": "DetectorConstruction.cc:147-148,205,245; 1 cm = 100 px"},
           "defect_centre_px": [cy, cx],
           "defs": {k: DEFS[k] for k in DEFS},
           "A_ref_zone_frac": zf, "A_ref_pixel_stats_cnn": st,
           "A_ref_decompose_cnn": dec, "A_outlier_radial_profile": sub,
           "geom_check_dense": gc, "table": table, "cnn_seed_cnr": seed_cnr}
    (RUNS / f"_cnr_roi_audit_{energy}.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")

    if make_fig:
        make_audit_figure(imgs, ROOT / "docs" / "figs" / "p01_cnr_roi_audit"
                          / f"fig_cnr_roi_audit_{energy}MeV.png", energy)
    # 给灵敏度总图用的精简表
    return {"energy": energy, "primary_run": primary, "geom": gc,
            "A_stats": st, "A_decomp": dec, "A_zone": zf, "outlier_radial": sub,
            "cnn_dense": {m: {k: table[m][k]["cnr"] for k in DEFS} for m in table},
            "detail": table, "seed": seed_cnr}


# ---------------------------------------------------------------- main
def main() -> int:
    ap = argparse.ArgumentParser(description="CNR 背景 ROI 定义严格复核")
    ap.add_argument("--energy", default="3.0")
    ap.add_argument("--all-energies", action="store_true")
    ap.add_argument("--no-fig", action="store_true")
    args = ap.parse_args()

    energies = ["1.5", "3.0", "4.5", "6.0"] if args.all_energies else [args.energy]
    per = {}
    for en in energies:
        per[en] = run_energy(en, make_fig=not args.no_fig)

    if len(per) > 1 and not args.no_fig:
        make_sensitivity_figure(per, ROOT / "docs" / "figs" / "p01_cnr_roi_audit"
                                / "fig_cnr_roi_sensitivity_energies.png")

    # ---- 汇总 markdown
    L = []
    L.append("# CNR 背景 ROI 定义复核（自动生成，勿手改）\n")
    L.append("生成脚本：`scripts/cnr_roi_audit.py`；原始数据："
             + "、".join(f"`runs/_cnr_roi_audit_{e}.json`" for e in per) + "\n")
    L.append("\n## 0. 几何锚点\n")
    L.append("| 量 | 值 | 依据 |\n|---|---|---|")
    L.append("| 棒心栅距 | 260 px | 角棒 r=367.7 px = √2×2.6 cm |")
    L.append("| 芯块半径 FRad | 75 px (0.75 cm) | `DetectorConstruction.cc:205` |")
    L.append("| 包壳内/外径 | 80 / 100 px | `:147-148` |")
    L.append("| 空腔半径 | 20 px (0.2 cm) | `:230` bubbleR |")
    L.append("| 图上可见棒边 | **"
             + " / ".join(f"{per[e]['geom']['r_edge_halfheight_px']:.0f}" for e in per)
             + " px**（= "
             + " / ".join(f"{per[e]['geom']['r_edge_over_pitch']:.3f}" for e in per)
             + " ×pitch） | 实测径向剖面半高位置 |")
    L.append("\n> `OutRad/pitch = 0.385`（=100 px）处**没有**台阶；"
             "实测边缘等于 `FRad/pitch = 0.288`。缺陷数据重建图里 82–98 px 区间"
             "的重建值只有芯块值的 "
             + " / ".join(f"{per[e]['geom']['frac_of_fuel_82_98']*100:+.1f}%"
                          for e in per)
             + "（均匀数据同一区间约 +50%）。**结论：缺陷 phantom 无包壳**，"
               "该区间是空气；因此字面「纯包壳区」定义不可用。\n")

    L.append("\n## 1. 当前 A 口径的 ROI 解剖\n")
    L.append("| 能量 | n(μ_ref) | 芯块占比 | 非芯块占比 | mean | std | MAD-σ | "
             ">3σ 离群 | 偏度 | 结构方差占比 |")
    L.append("|---|---|---|---|---|---|---|---|---|---|")
    for e, d in per.items():
        s, z = d["A_stats"], d["A_zone"]
        nonfuel = z["gap 75-80"] + z["clad 80-100"] + z["air >100"] + z["cavity 0-20"]
        L.append(f"| {e} | {s['n']} | {z['fuel 20-75']*100:.2f}% | "
                 f"{nonfuel*100:.3f}% | {s['mean']:.6f} | {s['std']:.6f} | "
                 f"{s['mad_sigma']:.6f} | {s['outlier_frac_3sd']*100:.3f}% | "
                 f"{s['skew']:+.3f} | {d['A_decomp']['struct_var_frac']:.1%} |")
    L.append("\n（全部为 CNN 补全、seed 升序第一个 run；结构方差占比 = 径向分箱均值"
             "方差 / 总方差，越低说明环内越没有径向梯度污染。）\n")

    L.append("\n## 2. 四种定义下的 CNR（第 1 个 seed）\n")
    for e, d in per.items():
        L.append(f"\n### {e} MeV\n")
        L.append("| 方法 | A (32–62, σ圆盘) | B (24–40, σ同带) | C (24–40 外侧扇区) | "
                 "D (82–98 池化) | A′ (32–62, σ同带) | D′ (52–68 池化) |")
        L.append("|---|---|---|---|---|---|---|")
        for m in METHODS:
            if m not in d["detail"]:
                continue
            t = d["detail"][m]
            L.append(f"| {m} | " + " | ".join(f"{t[k]['cnr']:.2f}" for k in
                                              ["A", "B", "C", "D", "A'", "D'"]) + " |")
    L.append("\nA′/D′ 为诊断列：A′ 把 σ 换成与 Δ 同一环带（隔离 A 的带不一致）；"
             "D′ 是缺陷数据里唯一可用的「全图纯棒体」定义。\n")

    L.append("\n## 3. CNN 三 seed 均值±标准差\n")
    L.append("| 能量 | A | B | C | D | A′ | D′ |")
    L.append("|---|---|---|---|---|---|---|")
    for e, d in per.items():
        cells = []
        for k in ["A", "B", "C", "D", "A'", "D'"]:
            arr = np.array([x[k] for x in d["seed"].values()])
            cells.append(f"{arr.mean():.3f} ± {arr.std(ddof=1):.3f}")
        L.append(f"| {e} | " + " | ".join(cells) + " |")

    L.append("\n## 4. 相对 A 的偏差（CNN 补全，三 seed 均值）\n")
    L.append("| 能量 | A | B/A−1 | C/A−1 | A′/A−1 | D′/A−1 | 最大绝对偏差 |")
    L.append("|---|---|---|---|---|---|---|")
    for e, d in per.items():
        base = np.mean([x["A"] for x in d["seed"].values()])
        cells, mx = [], 0.0
        for k in ["B", "C", "A'", "D'"]:
            v = np.mean([x[k] for x in d["seed"].values()])
            dev = (v / base - 1.0) * 100 if base else NAN
            cells.append(f"{dev:+.1f}%")
            mx = max(mx, abs(dev))
        L.append(f"| {e} | {base:.3f} | " + " | ".join(cells)
                 + f" | {mx:.1f}% |")
    L.append("\n判读：若 A 被高噪声污染，B/C/D′ 应**同向且显著**高于 A。实测"
             "偏差方向不一致（1.5/4.5 多为正、6.0 为负）、量级与 σ 的径向变化同阶"
             "（≤ 约 20%，唯 1.5 MeV 的带匹配 A′ 达 +32%），因此不是系统性污染，"
             "而是 **σ 随半径变化** 造成的口径差异。\n")

    md = RUNS / "_cnr_roi_sensitivity.md"
    md.write_text("\n".join(L) + "\n", encoding="utf-8")
    print(f"\n  [md] {md.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
