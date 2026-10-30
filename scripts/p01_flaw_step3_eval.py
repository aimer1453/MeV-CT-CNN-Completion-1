#!/usr/bin/env python
# -*- coding: utf-8 -*-
r"""
scripts/p01_flaw_step3_eval.py —— P0-1 第三步：**缺陷数据域内训练**的完整评估

回答三个问题（用户 2026-10-23 22:40 指派）
-------------------------------------------
  Q1  缺陷附近的视角，补全误差是否更大？
      → 投影域：CNN 预测误差的 (角度 × 列) 分布，与缺陷正弦轨迹叠加比对；
        并对 train.py 给出的 45 个 test 视角做误差-几何相关性检验。
  Q2  重建图上缺陷是否可检测？
      → 图像域：以"同方法、无缺陷数据"的重建为参照，在缺陷棒位置算对比度
        与 CNR（Δ/σ），并把其余 8 根棒作为对照。
  Q3  CNN 相对线性插值 / FBP 稀疏 / SART 稀疏谁更好？
      → 五方法同表（dense / cnn / linear / fbp / sart），同 dense 参考、同滤镜、
        同 FOV、同 SSIM 裁剪。

数据口径
--------
  - 缺陷数据：`sparse_targets_flaw.csv`（360 行，270/45/45，seed 20261022），
    盘上 3_3_flaw/fuel_ct_scan_z0.0_20260804_195102/projections/proj_aXXXX.0_z+0.npy。
  - **不读 angles.csv**：`angles.csv` 是均匀数据那套（file 列带 `3_3/` 前缀），
    直接用它会把均匀投影当成缺陷投影。本脚本一律从 `sparse_targets_flaw.csv`
    的 `target_file` / `target_angle_deg` 取角度与文件，一对一行。
  - 单缺陷位置**以代码为准**：(i, j) = (−1, +1) → 世界坐标 (−2.6, +2.6) cm
    （缺陷版 DetectorConstruction.cc `(i == -1 && j == 1) ? SolidFuelWithBubble`）。
    投影域无法区分 90° 的整数倍（4 根角棒 D4 等价），故图像上的**绝对方位**
    由数据拟合确定，代码给的是"哪一根棒"。
  - norm 取自 `runs/20261023_2239_cnn_bs8_lr1e-4/config.json`（缺陷数据重算：
    0.5174 / 0.8145 / 5.10）。基线不经过网络，重建全程在 -log 物理域。

参照（无缺陷数据）的重建
------------------------
  图像域 CNR 需要一个"同方法、同口径、无缺陷"的对照图。直接复用均匀数据那一轮
  已缓存的重建（`runs/20261022_2352_cnn_bs8_lr1e-4_*/recon/`），每个数据集用**它
  自己训练/调参的模型**（缺陷数据用缺陷模型、均匀数据用均匀模型），差别就只剩缺陷
  本身。唯一口径差异：均匀侧 zero_log_fill=5.50、缺陷侧 5.10（各自 config），
  该差异是全局常数项，在"缺陷 ROI − 背景环"的差分里抵消，详见输出 JSON 的
  `uniform_reference.fill_mismatch_note`。

输出
----
  runs/<RUN_ID>_flaw_step3/
      config.json  proj_angle.csv  proj_error_map.npy  image_metrics.json
      image_per_sample.csv  defect_cnr.json  recon/{method}_{phase}.npy
  docs/figs/p01_flaw_step3/   （由 scripts/plot_p01_flaw_step3.py 生成）

用法
----
  conda run -n ct_full --no-capture-output python scripts/p01_flaw_step3_eval.py
"""

from __future__ import annotations

import csv
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from data.csv_paths import resolve_csv  # noqa: E402  CSV 统一落点 csv_data/
for p in (ROOT, ROOT / "scripts"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import baseline_recon as br                       # noqa: E402
from cnn_recon import predict_views, write_subset_csv  # noqa: E402

RUN_ID = "20261023_2239_cnn_bs8_lr1e-4"          # 缺陷数据域内训练那一轮
# 无缺陷参照 run：必须与论文当前代次一致（2026-10-24 起为 100ep 的 20261024_0400）。
# 旧的 5ep 参照 20261022_2352_cnn_bs8_lr1e-4 已作废，禁止再用于 vs_noflaw 口径。
REF_RUN_ID = "20261024_0400_cnn_bs8_lr1e-4"      # 均匀数据那一轮（作无缺陷参照，100ep）
#: 默认禁止"用非缺陷表训练的 run 做缺陷域投影域 all32 统计"（跨域混引）。
#: 确需对该模型做缺陷域零样本评估时，命令行传 --allow-domain-mismatch，
#: 此时投影域 all32 段置 null，只保留缺陷域上算出的 row16 / 图像域 / CNR / 匹配滤波。
ALLOW_DOMAIN_MISMATCH = False
FLAW_CSV = resolve_csv("sparse_targets_flaw.csv")
FLAW_ROOT = ROOT / "3_3_flaw"
ENERGY = "3"
OUT_DIR = ROOT / "runs" / f"{RUN_ID}_flaw_step3"
RECON_DIR = OUT_DIR / "recon"

SART_N_ITER, SART_RELAX = 20, 0.15
FILTER = "hamming"
BATCH = 8

#: 9 根棒的图像半径（像素）。栅距 2.6 cm、M_eff≈1、1 cm→100 px（探测器像素 0.01 cm）
R_CORNER = 367.7      # 4 根角棒 √2×2.6 cm
R_EDGE = 260.0        # 4 根边棒 2.6 cm
NAN = float("nan")

#: 缺陷空腔的 ROI（实测：缺陷棒中心重建值 0.0003，正常棒 0.0045，半高处 r≈20 px）
BUBBLE_ROI_R = 15.0
REF_R0, REF_R1 = 32.0, 62.0

#: 缺陷正弦轨迹（实测定标，见 docs/handoff/algo_delivered.md）
#:   col(θ) = 601.9 + 367.1·cos(θ + 224.7°)，RMS 残差 35.2 px
#: ⚠ 默认值是**角棒 (i,j)=(-1,+1)** 专用。换缺陷位置必须传 --traj-json 覆盖，
#:   否则投影域匹配滤波会沿错误轨迹积分（边棒 amp 差 107 px，D4 直接失效）。
TRAJ = {"amp": 367.1, "offset": 601.9, "phase_deg": 224.7}
TRAJ_SOURCE = "module_default (corner rod, 3_3_flaw_1_1)"
DEFECT_ROD = "corner1"       # 角棒默认；边棒批次用 --defect-rod edge2 覆盖

METHODS = ("dense", "cnn_completed", "linear_interp", "fbp_sparse", "sart_sparse")
METHOD_ZH = {"dense": "dense 参考（360 视角全实测）",
             "cnn_completed": "CNN 补全（180 实测 + 180 补全）",
             "linear_interp": "线性插值（180 实测 + 180 补全）",
             "fbp_sparse": "FBP 稀疏（180 视角，无补全）",
             "sart_sparse": "SART 稀疏（180 视角，无补全）"}


# ------------------------------------------------------------------ 数据
def read_rows(path: Path, split: str | None = None) -> list[dict]:
    if not path.is_file():
        raise br.ReconSpecError(f"缺少 {path}")
    with open(path, newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    if split is not None:
        rows = [r for r in rows if r["split"] == split]
    rows.sort(key=lambda r: float(r["target_angle_deg"]))
    return rows


def check_angles(rows: list[dict], tag: str) -> np.ndarray:
    ang = np.array([float(r["target_angle_deg"]) for r in rows])
    if rows and len(rows) == 360:
        if not np.array_equal(ang.astype(np.int64), np.arange(360)):
            raise br.ReconSpecError(f"[{tag}] 角度不是 0..359 的整数序列")
    return ang


def build_sinogram(rows: list[dict], data_root: Path, fill: float
                   ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """返回 (sino(360,1200) row16 的 -log、theta(360,)、empty(32,1200))。"""
    scan_dirs = {r["scan_dir"] for r in rows}
    if len(scan_dirs) != 1:
        raise br.ReconSpecError(f"出现多个 scan_dir: {scan_dirs}")
    empty = br.load_empty(data_root, scan_dirs.pop())
    sino = np.empty((len(rows), br.OUTPUT_SIZE), dtype=np.float64)
    theta = np.empty(len(rows), dtype=np.float64)
    for i, r in enumerate(rows):
        proj = np.load(br.ROOT / r["target_file"]).astype(np.float32)
        if proj.shape != (32, br.OUTPUT_SIZE):
            raise br.ReconSpecError(f"投影形状 {proj.shape} != (32,1200): {r['target_file']}")
        L = br.log_ratio(proj, empty).astype(np.float64)
        sino[i] = np.where(np.isfinite(L), L, float(fill))[br.RECON_ROW]
        theta[i] = float(r["target_angle_deg"])
    return sino, theta, empty


def log_stack(rows: list[dict], empty: np.ndarray, fill: float) -> np.ndarray:
    """按 rows 顺序返回全 32 行的 -log 栈 (N,32,1200)，用于投影域误差图。"""
    out = np.empty((len(rows), 32, br.OUTPUT_SIZE), dtype=np.float32)
    for i, r in enumerate(rows):
        proj = np.load(br.ROOT / r["target_file"]).astype(np.float32)
        L = br.log_ratio(proj, empty).astype(np.float64)
        out[i] = np.where(np.isfinite(L), L, float(fill)).astype(np.float32)
    return out


# ------------------------------------------------------------------ 图像域
def recon_all(rows: list[dict], sino: np.ndarray, theta: np.ndarray,
              in_phase: str, by_phase: dict[str, list[dict]], header: list[str],
              model, mean: float, std: float, fill: float, device,
              cache: bool = True,
              baselines: dict[str, np.ndarray] | None = None
              ) -> tuple[dict[str, np.ndarray], np.ndarray]:
    """对一个输入相位返回 4 张补全类重建 + 预测误差栈。

    `baselines`：若给出（键为 `linear_interp_{phase}` / `fbp_sparse_{phase}` /
    `sart_sparse_{phase}`），则跳过这三个重建而直接复用。它们**与模型权重无关**
    （只依赖实测正弦图与几何），跨种子完全相同，重算纯属浪费 GPU 时间——
    3 个种子 × 2 相位 × 20 次 SART 迭代是本脚本最大的耗时项（每次 ~4.5 min）。
    """
    missing = br.opposite_phase(in_phase)
    idx = {int(a): i for i, a in enumerate(np.round(theta).astype(np.int64))}
    ridge = {in_phase: br.select_phase(theta, in_phase),
             missing: br.select_phase(theta, missing)}
    for p in (in_phase, missing):
        if int(ridge[p].sum()) != 180:
            raise br.ReconSpecError(f"相位 {p} 期望 180 视角，实得 {int(ridge[p].sum())}")

    out: dict[str, np.ndarray] = {}
    err: np.ndarray = np.zeros((0, br.OUTPUT_SIZE), dtype=np.float32)

    # ---- CNN 补全
    t0 = time.time()
    csv_p = write_subset_csv(ENERGY, missing, header, by_phase[missing])
    pred, _mask = predict_views(model, csv_p, mean, std, fill, BATCH, device,
                                expected_files=[r["target_file"] for r in by_phase[missing]])
    sino_cnn = sino.copy()
    gt_row = np.empty((len(by_phase[missing]), br.OUTPUT_SIZE), dtype=np.float32)
    err = np.empty((len(by_phase[missing]), br.OUTPUT_SIZE), dtype=np.float32)
    for i, r in enumerate(by_phase[missing]):
        a = int(float(r["target_angle_deg"]))
        pr = pred[i, 0, br.RECON_ROW, :].astype(np.float64)
        sino_cnn[idx[a]] = pr
        gt_row[i] = sino[idx[a]].astype(np.float32)
        err[i] = (pr - sino[idx[a]]).astype(np.float32)
    out["cnn_completed"] = br.restrict_to_fov(br.recon_fbp(sino_cnn, theta, FILTER))
    print(f"    cnn_completed {time.time() - t0:.1f}s", flush=True)

    if baselines is None:
        # ---- 线性插值补全（ch0/ch_next 两点均值，与 baseline_linear_image 逐字同式）
        t0 = time.time()
        sino_lin = sino.copy()
        for r in by_phase[missing]:
            a = int(float(r["target_angle_deg"]))
            lo = sino[idx[(a - 1) % 360]]
            hi = sino[idx[(a + 1) % 360]]
            sino_lin[idx[a]] = 0.5 * (lo + hi)
        out["linear_interp"] = br.restrict_to_fov(br.recon_fbp(sino_lin, theta, FILTER))
        print(f"    linear_interp {time.time() - t0:.1f}s", flush=True)

        # ---- 稀疏直重建（无补全，只用实测 180 视角）
        t0 = time.time()
        sel = ridge[in_phase]
        out["fbp_sparse"] = br.restrict_to_fov(br.recon_fbp(sino[sel], theta[sel], FILTER))
        print(f"    fbp_sparse {time.time() - t0:.1f}s", flush=True)
        t0 = time.time()
        out["sart_sparse"] = br.restrict_to_fov(
            br.recon_sart(sino[sel], theta[sel], SART_N_ITER, SART_RELAX,
                          log_prefix=f"[{in_phase}] "))
        print(f"    sart_sparse {time.time() - t0:.1f}s", flush=True)
    else:
        t0 = time.time()
        for m in ("linear_interp", "fbp_sparse", "sart_sparse"):
            out[m] = br.restrict_to_fov(baselines[f"{m}_{in_phase}"])
        print(f"    三个基线复用（种子无关） {time.time() - t0:.1f}s", flush=True)

    if cache:
        RECON_DIR.mkdir(parents=True, exist_ok=True)
        for k, v in out.items():
            np.save(RECON_DIR / f"{k}_{in_phase}.npy", v.astype(np.float32))
    return out, err


# ------------------------------------------------------------------ 缺陷定位 / CNR
def _disc_mean_crop(crop: np.ndarray, cy: float, cx: float, rad: float) -> float:
    yy, xx = np.mgrid[0:crop.shape[0], 0:crop.shape[1]]
    m = (yy - cy) ** 2 + (xx - cx) ** 2 <= rad * rad
    return float(crop[m].mean())


def rod_centers() -> list[tuple[str, float, float]]:
    """9 根棒的 (名字, 半径 px, 图像方位角 deg)，方位角按图像数组极角定义。

    世界坐标（`pos(i*spacing, j*spacing, 0)`，spacing=2.6 cm，1 cm=100 px）：
        角棒 (±2.6, ±2.6) → r=√2·260=367.7 px，世界方位 45°+90k
        边棒 (±2.6, 0) / (0, ±2.6) → r=260 px，世界方位 0°+90k
    图像方位 = 世界方位 − 90°（实测标定，见交接文档附录）。减 90° 后
    {45+90k} 与 {90k} 两个方位角集合各自不变，故这里直接按世界方位角给
    图像方位角不会错位；但**不能把角棒的方位写成 0°+90k**（那样 4 个采样点
    会落进棒间空隙，得到 4 个几乎相同的假读数，已实测踩过）。
    """
    out = [("center", 0.0, 0.0)]
    for k in range(4):
        out.append((f"edge{k}", R_EDGE, 0.0 + 90.0 * k))
    for k in range(4):
        out.append((f"corner{k}", R_CORNER, 45.0 + 90.0 * k))
    return out


#: ⚠ 历史巧合式，勿推广到其它缺陷位置。
#:   原注释写"世界方位 224.7° − 90° = 134.7° → 角棒 135°"，但用两个独立锚点核验：
#:     角棒 phase=224.7° 实测最暗棒 corner1 az=135°；边棒 phase=180.2° 实测最暗棒 edge2 az=180°。
#:   两者共同满足 **az_img = −phase (mod 360)**；而 `phase − 90` 仅在 phase≡45 (mod 180)
#:   （即 45°/225°）时与之等价，角棒 224.7° 恰好落在这个巧合点上，故一直未暴露。
#:   边棒 phase=180.2° 时该式给出 90.2°（指向正常棒 edge1），与真值差 90°。
#:   因此新批次一律用 --defect-rod 从 rod_centers() 查表定位，不依赖任何坐标推导。
AZ_DEFECT_IMG = TRAJ["phase_deg"] - 90.0
R_DEFECT_PX = TRAJ["amp"]


def bubble_metrics(img: np.ndarray) -> dict:
    """**主 CNR 口径**：缺陷棒中心的空腔 vs 同一根棒的外环，噪声取 3 根正常棒的
    同尺寸中心圆盘标准差（同图、同位置族、均匀区）。

    为什么不用"与无缺陷数据做差"作主口径：两次仿真是独立随机序列，差值图里
    混着 RNG 漂移（实测各棒 ~±0.0004），会把本来很干净的信号稀释掉。直接的
    图像内对比度才是审稿人能复核的数。
    """
    cy = 600.0 - R_DEFECT_PX * math.sin(math.radians(AZ_DEFECT_IMG))
    cx = 600.0 + R_DEFECT_PX * math.cos(math.radians(AZ_DEFECT_IMG))
    yy, xx = np.mgrid[0:img.shape[0], 0:img.shape[1]]
    d2 = (yy - cy) ** 2 + (xx - cx) ** 2

    m_bub = d2 <= BUBBLE_ROI_R ** 2
    m_ref = (d2 >= REF_R0 ** 2) & (d2 <= REF_R1 ** 2)
    mu_b = float(img[m_bub].mean())
    mu_r = float(img[m_ref].mean())

    # 噪声：与缺陷棒**同半径族**的 3 根正常棒，取 std 的中位数。
    # 角棒批次自动选另外 3 根角棒（与旧实现逐位一致，保护 Table 8）；
    # 边棒批次选另外 3 根边棒。旧实现硬编码 name.startswith("corner")，
    # 在边棒下 σ 会恒为 NaN，且 Δ 与 σ 落在不同半径带，必须改为按半径匹配。
    #
    # ⚠ center 缺陷例外：缺陷棒位于转轴（R_DEFECT_PX≈0），r=0 的棒**只有 center
    # 自己**，同半径族再剔除缺陷棒后必然为空 → σ 恒 NaN（实测 12 个中心棒 run 全部
    # 命中）。此时改用 4 根完好角棒（corner 族）作噪声基——中心棒批 4 根角棒全完好，
    # 且与主口径「σ 取正常角棒」一致。边棒/角棒批次逻辑逐位不变，Table 8 不受影响。
    if R_DEFECT_PX < 1.0:
        noise_rods = [(n, r, a) for n, r, a in rod_centers()
                      if abs(r - R_CORNER) <= 1.0]
    else:
        noise_rods = [(n, r, a) for n, r, a in rod_centers()
                      if abs(r - R_DEFECT_PX) <= 1.0
                      and abs(((a - AZ_DEFECT_IMG) + 180.0) % 360.0 - 180.0) >= 1.0]
    sds = []
    for name, r, az in noise_rods:
        cy2 = 600.0 - r * math.sin(math.radians(az))
        cx2 = 600.0 + r * math.cos(math.radians(az))
        m = (yy - cy2) ** 2 + (xx - cx2) ** 2 <= BUBBLE_ROI_R ** 2
        sds.append(float(img[m].std(ddof=1)))
    sd = float(np.median(sds)) if sds else NAN

    # 空腔半径：沿径向找"半高"位置
    r_edges = np.arange(2.0, 45.0, 1.0)
    prof = np.array([float(img[(d2 >= (e - 1) ** 2) & (d2 < e ** 2)].mean())
                     for e in r_edges])
    half = 0.5 * (mu_b + mu_r)
    above = np.where(prof >= half)[0]
    r_bub_px = float(r_edges[above[0]] - 0.5) if above.size else NAN

    _sig = ("4 根完好角棒（center 缺陷位于转轴，无同半径正常棒，改用角棒族）"
            if R_DEFECT_PX < 1.0 else
            f"与缺陷棒同半径族（r≈{R_DEFECT_PX:.1f} px）的 3 根正常棒")

    return {"mu_bubble": mu_b, "mu_reference": mu_r,
            "contrast": mu_r - mu_b,
            "noise_sd": sd,
            "cnr": (mu_r - mu_b) / sd if sd > 0 else NAN,
            "bubble_radius_px": r_bub_px,
            "bubble_radius_cm": r_bub_px / 100.0 if r_bub_px == r_bub_px else NAN,
            "defect_rod_az_img_deg": AZ_DEFECT_IMG,
            "defect_rod_r_px": R_DEFECT_PX,
            "noise_reference_rods": [name for name, _r, _az in noise_rods],
            "definition": (f"μ_bubble = 缺陷棒中心 r≤{BUBBLE_ROI_R:.0f} px 圆盘均值；"
                           f"μ_ref = 同棒 {REF_R0:.0f}–{REF_R1:.0f} px 环均值；"
                           f"σ = {_sig}同尺寸中心圆盘 std 的中位数；"
                           "bubble_radius 由径向半高位置确定")}


def rod_centre_values(img: np.ndarray, rad: float = BUBBLE_ROI_R) -> list[dict]:
    """9 根棒中心圆盘的重建均值（主图的对照面板用）。"""
    yy, xx = np.mgrid[0:img.shape[0], 0:img.shape[1]]
    out = []
    for name, r, az in rod_centers():
        cy = 600.0 - r * math.sin(math.radians(az))
        cx = 600.0 + r * math.cos(math.radians(az))
        m = (yy - cy) ** 2 + (xx - cx) ** 2 <= rad * rad
        out.append({"rod": name, "mu_centre": float(img[m].mean()),
                    "is_defect_rod": bool(
                        abs(r - R_DEFECT_PX) < 1.0 and
                        abs(((az - AZ_DEFECT_IMG) + 180.0) % 360.0 - 180.0) < 1.0)})
    return out


def rod_peak(diff: np.ndarray, r: float, az: float, roi_r: float = 25.0,
             search_r: float = 60.0, step: int = 4) -> dict:
    """在一根棒的足迹内做**同等**的峰值搜索（9 根棒同一套参数，避免只挑缺陷棒）。"""
    cy = 600.0 - r * math.sin(math.radians(az))
    cx = 600.0 + r * math.cos(math.radians(az))
    pad = int(search_r + roi_r) + 2
    y0, y1 = int(round(cy)) - pad, int(round(cy)) + pad + 1
    x0, x1 = int(round(cx)) - pad, int(round(cx)) + pad + 1
    crop = diff[max(0, y0):y1, max(0, x0):x1]
    oy, ox = max(0, y0), max(0, x0)
    loc_cy, loc_cx = cy - oy, cx - ox

    best = {"abs": -1.0, "v": NAN, "dy": 0, "dx": 0}
    rng = range(-int(search_r), int(search_r) + 1, step)
    for dy in rng:
        for dx in rng:
            if dy * dy + dx * dx > search_r * search_r:
                continue
            v = _disc_mean_crop(crop, loc_cy + dy, loc_cx + dx, roi_r)
            if abs(v) > best["abs"]:
                best = {"abs": abs(v), "v": v, "dy": int(dy), "dx": int(dx)}
    return {"rod_r_px": r, "rod_az_img_deg": az, "roi_radius_px": roi_r, **best}


def bg_stats(diff: np.ndarray, mask: np.ndarray) -> tuple[float, float]:
    yy, xx = np.mgrid[0:diff.shape[0], 0:diff.shape[1]]
    bg = mask.copy()
    for _name, r, az in rod_centers():
        cy = 600.0 - r * math.sin(math.radians(az))
        cx = 600.0 + r * math.cos(math.radians(az))
        bg &= (yy - cy) ** 2 + (xx - cx) ** 2 > 130.0 ** 2
    return float(diff[bg].mean()), float(diff[bg].std(ddof=1))


def defect_cnr(diff: np.ndarray, mask: np.ndarray) -> dict:
    """逐棒同等峰值搜索 → 缺陷棒是否突出（CNR = Δ / σ_bg）。"""
    mu_bg, sd_bg = bg_stats(diff, mask)
    per_rod = []
    for name, r, az in rod_centers():
        pk = rod_peak(diff, r, az)
        is_defect = (abs(r - R_DEFECT_PX) < 1.0) and \
            (abs(((az - AZ_DEFECT_IMG) + 180.0) % 360.0 - 180.0) < 1.0)
        per_rod.append({"rod": name, "is_defect_rod": bool(is_defect), **pk,
                        "delta_vs_bg": pk["v"] - mu_bg,
                        "cnr": (pk["v"] - mu_bg) / sd_bg if sd_bg > 0 else NAN})
    def_row = next(r for r in per_rod if r["is_defect_rod"])
    others = [r for r in per_rod if not r["is_defect_rod"]]
    rank = 1 + sum(1 for r in others if r["abs"] > def_row["abs"])
    return {
        "defect_rod_az_img_deg": AZ_DEFECT_IMG,
        "defect_rod_r_px": R_DEFECT_PX,
        "defect_rod_note": (
            f"ROI 由 --defect-rod={DEFECT_ROD} 查 rod_centers() 定位"
            f"（r={R_DEFECT_PX:.1f} px, az_img={AZ_DEFECT_IMG:.1f}°）；"
            f"投影域轨迹来自 {TRAJ_SOURCE}。旧实现以 phase−90 反推图像方位，"
            "该式仅在角棒 phase≈225° 时碰巧成立，已弃用；"
            "差 90° 的整数倍在投影域不可分辨，「哪一根棒」以图像域最暗棒锚定为准"),
        "bg_mean": mu_bg, "bg_std": sd_bg,
        "defect_peak": def_row["v"],
        "defect_delta_vs_bg": def_row["delta_vs_bg"],
        "cnr": def_row["cnr"],
        "defect_rank_among_9": rank,
        "control_abs_mean": float(np.mean([r["abs"] for r in others])),
        "control_abs_max": float(np.max([r["abs"] for r in others])),
        "contrast_ratio": def_row["abs"] / max(1e-12, float(np.mean([r["abs"] for r in others]))),
        "per_rod": per_rod,
    }


def proj_matched_filter(sino_flaw: np.ndarray, fill: float,
                        band_px: float = 60.0) -> dict:
    """投影域**轨迹匹配滤波**：flaw − uniform 的正弦图差，沿缺陷轨迹取带 vs 带外。

    这是"缺陷到底能不能被检出"的判据：缺陷在单张重建图上被 10° 弧宽的稀释
    掩盖（见 CNR），但在投影域沿轨迹做定向积分可以在 360 个角度上累加。
    两个数据源用**同一个 fill**（缺陷侧裁决值），零计数像素两侧填同一常数，
    差分里严格为 0，不会伪造信号。

    零假设对照：把假定轨迹人为旋转 90°/180°/270° 各做一次同样的积分。
    真缺陷只在 offset=0 时应给出显著非零值（4 根角棒 D4 等价，90° 的整数倍
    在投影域不可分辨，故 90° 也会显著——这一点本身即为标定的直接证据）。
    """
    def _same_energy(a: str, b: str) -> bool:
        """能量标签按数值比较。

        坑：均匀数据表 `sparse_targets.csv` 的 energy_MeV 写作 `3`，而缺陷数据表
        （多能量扩展后）写作 `3.0`；用字符串相等过滤会得到 0 行，报
        "均匀数据 3.0 MeV 期望 360 行，实得 0"。
        """
        try:
            return float(a) == float(b)
        except (TypeError, ValueError):
            return a == b

    rows_u = [r for r in read_rows(br.SPARSE_TARGETS)
              if _same_energy(r["energy_MeV"], ENERGY)]
    if len(rows_u) != 360:
        raise br.ReconSpecError(f"均匀数据 {ENERGY} MeV 期望 360 行，实得 {len(rows_u)}")
    check_angles(rows_u, "uniform")
    print("构建无缺陷正弦图（同 fill）...", flush=True)
    sino_u, theta_u, _ = build_sinogram(rows_u, ROOT / "3_3", fill)
    if not np.array_equal(np.round(theta_u).astype(np.int64), np.arange(360)):
        raise br.ReconSpecError("均匀数据角度不是 0..359")
    d = sino_flaw - sino_u                                  # (360,1200)

    cols = np.arange(br.OUTPUT_SIZE)
    angles = np.arange(360.0)
    out: dict[str, dict] = {}
    curves: dict[int, np.ndarray] = {}
    for off in (0, 90, 180, 270):
        traj = TRAJ["offset"] + TRAJ["amp"] * np.cos(
            np.radians(angles + TRAJ["phase_deg"] + off))
        dl = np.empty(360)
        for a in range(360):
            m = np.abs(cols - traj[a]) <= band_px
            dl[a] = d[a][m].mean() - d[a][~m].mean()
        curves[off] = dl
        mu = float(dl.mean())
        sd = float(dl.std(ddof=1))
        sem = sd / math.sqrt(len(dl))
        out[f"offset_{off}deg"] = {
            "mean_dlog": mu, "std": sd, "sem": sem,
            "t_stat": mu / sem if sem > 0 else NAN,
            "n_angles": len(dl),
            "note": "dlog = 轨迹带内均值 − 带外均值（单角），再对 360 角取平均",
        }
    # 弧宽：以拟合相位为中心，dl 随角度偏离的衰减
    peak_a = float(np.argmax(curves[0]))
    dev = np.abs(((angles - peak_a + 180.0) % 360.0) - 180.0)
    bins = np.arange(0, 91, 5.0)
    prof = [{"abs_dev_deg": float(bins[i]),
             "mean_dlog": float(curves[0][(dev >= bins[i]) & (dev < bins[i] + 5)].mean()),
             "n_angles": int(((dev >= bins[i]) & (dev < bins[i] + 5)).sum())}
            for i in range(len(bins) - 1)]
    np.save(OUT_DIR / "proj_matched_filter_curve.npy", curves[0])

    # ---- D4 对照：offset 0 相对 90/180/270 三条"伪轨迹"的基线偏移。
    # 物模 4 重旋转对称 ⇒ 三条伪轨迹给出的基线应彼此一致（实测 −0.0795/−0.0789/
    # −0.0796，差异 <0.001）；offset 0 的偏离量里，数据集间的全局乘性差
    # （两次仿真统计量差异）被消掉，剩下的就是缺陷本身。
    base = [out[f"offset_{o}deg"]["mean_dlog"] for o in (90, 180, 270)]
    base_sem = [out[f"offset_{o}deg"]["sem"] for o in (90, 180, 270)]
    b0 = float(np.mean(base))
    b0_sem = float(np.sqrt(np.mean([s * s for s in base_sem])) / math.sqrt(len(base)) *
                   math.sqrt(len(base)))          # 三条伪轨迹的平均的标准误
    b0_sem = float(np.sqrt(sum(s * s for s in base_sem)) / len(base))
    d0 = out["offset_0deg"]["mean_dlog"]
    s0 = out["offset_0deg"]["sem"]
    contrast = d0 - b0
    contrast_sem = math.sqrt(s0 * s0 + b0_sem * b0_sem)
    out["d4_contrast"] = {
        "offset0_minus_pseudo_baseline": contrast,
        "sem": contrast_sem,
        "t_stat": contrast / contrast_sem if contrast_sem > 0 else NAN,
        "pseudo_baseline": b0,
        "pseudo_baseline_spread": float(np.ptp(base)),
        "note": ("物模 4 重旋转对称 ⇒ 90/180/270 三条伪轨迹的基线应一致（实测极差<0.001）；"
                 "offset0 相对该基线的偏离即缺陷信号，已消去两次仿真之间的全局乘性差"),
    }
    return {"by_offset": {k: v for k, v in out.items() if k.startswith("offset_")},
            "d4_contrast": out["d4_contrast"], "arc_profile": prof,
            "peak_angle_deg": peak_a, "band_halfwidth_px": band_px,
            "definition": ("ΔL(θ) = mean_{|col−traj(θ)|≤60}(flaw−uniform) − "
                           "mean_{其余列}(flaw−uniform)，两侧同 fill")}


# ------------------------------------------------------------------ main
def main(argv: list[str] | None = None) -> int:
    import argparse

    # global 必须先于任何对 ENERGY 的读取（argparse 的 default=ENERGY 就是读取），
    # 否则 Python 报 SyntaxError: name 'ENERGY' is used prior to global declaration
    global RUN_ID, OUT_DIR, RECON_DIR, REF_RUN_ID, ALLOW_DOMAIN_MISMATCH, \
        ENERGY, FLAW_CSV, FLAW_ROOT, TRAJ, TRAJ_SOURCE, \
        AZ_DEFECT_IMG, R_DEFECT_PX, DEFECT_ROD

    ap = argparse.ArgumentParser(description="P0-1 第三步：缺陷数据域内训练评估")
    ap.add_argument("--reuse", action="store_true",
                    help="复用 runs/<RUN_ID>_flaw_step3/recon 与 proj_error_map_*.npy"
                         "（只重算指标与 CNR，不重跑 SART）")
    ap.add_argument("--run-id", default=None,
                    help="覆盖模块常量 RUN_ID（如换 30 epoch 权重 "
                         "20261023_2327_cnn_bs8_lr1e-4）；输出目录随之改为 "
                         "runs/<run_id>_flaw_step3/")
    ap.add_argument("--ref-run-id", default=None,
                    help="无缺陷参照 run（vs_noflaw 口径）。默认 100ep 的 "
                         "20261024_0400_cnn_bs8_lr1e-4；换代次时必须同步换")
    ap.add_argument("--allow-domain-mismatch", action="store_true",
                    help="允许对非缺陷表训练的 run 做缺陷域评估（零样本场景）。"
                         "此时投影域 all32 段置 null，只保留缺陷域上算出的 "
                         "row16 / 图像域 / CNR / 匹配滤波")
    ap.add_argument("--energy", default=ENERGY,
                    help="能量档字符串（与数据表 energy_MeV 列一致：1.5 / 3 / 4.5 / 6）。"
                         "多能量扩展：决定无缺陷参照的文件名（dense_<E>_fbp_hamming.npy 等）"
                         "与均匀正弦图的能量过滤")
    ap.add_argument("--flaw-csv", default=None,
                    help="缺陷稀疏角数据表（默认 sparse_targets_flaw.csv；"
                         "多能量时传 sparse_targets_flaw_<E>.csv）")
    ap.add_argument("--defect-rod", default=None,
                    help="缺陷棒名（rod_centers() 的名字：center/edge0..3/corner0..3）。"
                         "由查表给出图像域 ROI 的 (r, az)，绕开 AZ_DEFECT_IMG=phase-90 "
                         "这个只在角棒成立的巧合式。边棒批次传 edge2。"
                         "不传则沿用角棒默认（保护 Table 8 可复现）")
    ap.add_argument("--traj-json", default=None,
                    help="空腔轨迹定标 json（localize_flaw_cavity.py 产物）。按能量数值"
                         "匹配记录，用其 amp/offset/phase_deg 覆盖模块级 TRAJ，"
                         "供投影域匹配滤波 D4 使用。与 --defect-rod 独立，建议同时传")
    ap.add_argument("--baseline-from", default=None, dest="baseline_from",
                    help="复用另一 run 的 step3 基线重建（dense / linear_interp / "
                         "fbp_sparse / sart_sparse）——这四项与模型权重无关、跨种子"
                         "逐位相同；只有 CNN 补全按当前 RUN_ID 实算。用于多种子扫描时"
                         "省掉每种子 ~4.5 min 的 SART")
    args = ap.parse_args(argv)

    if args.energy != ENERGY:
        ENERGY = args.energy
        print(f"[覆盖] ENERGY={ENERGY}", flush=True)
    if args.flaw_csv:
        FLAW_CSV = resolve_csv(args.flaw_csv)
        print(f"[覆盖] FLAW_CSV={FLAW_CSV.name}", flush=True)
    if args.run_id:
        RUN_ID = args.run_id
        OUT_DIR = ROOT / "runs" / f"{RUN_ID}_flaw_step3"
        RECON_DIR = OUT_DIR / "recon"
        print(f"[覆盖] RUN_ID={RUN_ID}  OUT_DIR={OUT_DIR}", flush=True)
    if args.ref_run_id:
        REF_RUN_ID = args.ref_run_id
        print(f"[覆盖] REF_RUN_ID={REF_RUN_ID}", flush=True)
    if args.allow_domain_mismatch:
        ALLOW_DOMAIN_MISMATCH = True
        print("[覆盖] ALLOW_DOMAIN_MISMATCH=True（投影域 all32 段将置 null）", flush=True)

    # ---- 缺陷几何覆盖（边棒批次必需；不传时逐位保持角棒默认）
    if args.defect_rod:
        hits = [(n, r, az) for n, r, az in rod_centers() if n == args.defect_rod]
        if len(hits) != 1:
            raise br.ReconSpecError(
                f"--defect-rod={args.defect_rod!r} 不在 rod_centers()："
                f"{[n for n, _, _ in rod_centers()]}")
        DEFECT_ROD, R_DEFECT_PX, AZ_DEFECT_IMG = hits[0][0], hits[0][1], hits[0][2]
        print(f"[覆盖] DEFECT_ROD={DEFECT_ROD}  R_DEFECT_PX={R_DEFECT_PX:.1f} "
              f"AZ_DEFECT_IMG={AZ_DEFECT_IMG:.1f}°（查表定位，不依赖 phase−90）",
              flush=True)
    if args.traj_json:
        _p = Path(args.traj_json)
        if not _p.is_absolute():
            _p = ROOT / _p
        if not _p.is_file():
            raise br.ReconSpecError(f"--traj-json 不存在: {_p}")
        _d = json.loads(_p.read_text(encoding="utf-8"))
        _key = next((k for k in _d if not k.startswith("_")
                     and abs(float(k) - float(ENERGY)) < 1e-9), None)
        if _key is None:
            raise br.ReconSpecError(
                f"--traj-json 无 {ENERGY} MeV 记录（现有 {list(_d)}）")
        _r = _d[_key]
        TRAJ = {"amp": float(_r["amp"]), "offset": float(_r["offset"]),
                "phase_deg": float(_r["phase_deg"])}
        TRAJ_SOURCE = f"{_p.relative_to(ROOT)}#{_key}"
        print(f"[覆盖] TRAJ={TRAJ}  来源={TRAJ_SOURCE}", flush=True)
        _derived = (-TRAJ["phase_deg"]) % 360.0
        _delta = abs(((_derived - AZ_DEFECT_IMG) + 180.0) % 360.0 - 180.0)
        print(f"[几何核验] 投影拟合推得 az_img=−phase={_derived:.2f}°，"
              f"当前 ROI az_img={AZ_DEFECT_IMG:.2f}°，差 {_delta:.2f}°"
              + ("  ⚠ 不一致，请核对 --defect-rod 与定标结果" if _delta > 5.0 else "  ✓"),
              flush=True)
        print(f"[几何核验] 图像域独立锚点 darkest_rod="
              f"{_r.get('image_domain', {}).get('darkest_rod')}", flush=True)

    for p in (OUT_DIR, RECON_DIR):
        p.mkdir(parents=True, exist_ok=True)

    cfg = json.loads((br.RUNS_DIR / RUN_ID / "config.json").read_text(encoding="utf-8"))
    norm = cfg["norm"]
    mean, std, fill = float(norm["norm_mean"]), float(norm["norm_std"]), \
        float(norm["zero_log_fill"])
    print(f"norm: mean={mean} std={std} fill={fill}", flush=True)

    rows = read_rows(FLAW_CSV)
    check_angles(rows, "flaw")
    # 数据根目录由数据表自己反推（`3_3_flaw/...` 或 `3_3_flaw_1_1/...`），
    # 不再硬编码，避免新数据到位后忘记改常量而把 empty_proj 取错档。
    _roots = {(ROOT / r["target_file"]).parent.parent.parent for r in rows}
    if len(_roots) != 1:
        raise br.ReconSpecError(f"数据表跨多个数据根目录: {_roots}")
    FLAW_ROOT = _roots.pop()
    _sd = {(ROOT / r["target_file"]).parent.parent.name for r in rows}
    if _sd != {rows[0]["scan_dir"]}:
        raise br.ReconSpecError(f"scan_dir 列与文件路径不一致: {_sd}")
    print(f"[数据根] {FLAW_ROOT.relative_to(ROOT)}  scan_dir={rows[0]['scan_dir']}", flush=True)
    header = list(rows[0].keys())
    by_phase = {p: [r for r in rows if r["phase"] == p] for p in ("A", "B")}
    for p in ("A", "B"):
        if len(by_phase[p]) != 180:
            raise br.ReconSpecError(f"相位 {p} 期望 180 行，实得 {len(by_phase[p])}")

    recon: dict[str, dict[str, np.ndarray]] = {}
    errs: dict[str, np.ndarray] = {}
    # 正弦图很便宜（0.3 s，360 个 npy），无论如何都建：相对误差口径要用到 |gt|
    print("构建缺陷正弦图 ...", flush=True)
    t0 = time.time()
    sino, theta, empty = build_sinogram(rows, FLAW_ROOT, fill)
    print(f"  完成 {time.time() - t0:.1f}s   shape={sino.shape} "
          f"min={sino.min():+.4f} max={sino.max():+.4f}", flush=True)

    if args.reuse:
        print("[reuse] 从磁盘读取已算重建与投影误差图", flush=True)
        dense = br.restrict_to_fov(np.load(RECON_DIR / "dense.npy"))
        for in_phase in ("A", "B"):
            recon[in_phase] = {
                m: br.restrict_to_fov(np.load(RECON_DIR / f"{m}_{in_phase}.npy"))
                for m in METHODS if m != "dense"}
            errs[in_phase] = np.load(OUT_DIR / f"proj_error_map_{in_phase}.npy")
    else:
        # ---- dense 参考
        t0 = time.time()
        if args.baseline_from:
            src = ROOT / "runs" / f"{args.baseline_from}_flaw_step3" / "recon"
            if not (src / "dense.npy").is_file():
                raise br.ReconSpecError(f"--baseline-from 指向的目录缺 dense.npy：{src}")
            dense = br.restrict_to_fov(np.load(src / "dense.npy"))
            baseline_arrays = {
                f"{m}_{ph}": np.load(src / f"{m}_{ph}.npy")
                for m in ("linear_interp", "fbp_sparse", "sart_sparse")
                for ph in ("A", "B")}
            print(f"dense 与三个基线复用自 {args.baseline_from} "
                  f"（种子无关，与重算逐位相同）{time.time() - t0:.1f}s", flush=True)
        else:
            dense = br.restrict_to_fov(br.recon_fbp(sino, theta, FILTER))
            baseline_arrays = None
            print(f"dense 360 视角 FBP {time.time() - t0:.1f}s", flush=True)
        np.save(RECON_DIR / "dense.npy", dense.astype(np.float32))

        # ---- CNN
        import torch

        from models.unet import UNet

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        torch.backends.cudnn.benchmark = False
        model = UNet().to(device)
        ckpt = torch.load(br.RUNS_DIR / RUN_ID / "best.pt", map_location=device,
                          weights_only=False)
        model.load_state_dict(ckpt["model"])
        model.eval()
        print(f"best.pt: epoch={ckpt['epoch']} val_rmse_valid_phys="
              f"{ckpt['val_rmse_valid_phys']:.6f}  device={device}", flush=True)

        # ---- 两相位各四类重建
        for in_phase in ("A", "B"):
            print(f"[输入相位 {in_phase}]", flush=True)
            r, e = recon_all(rows, sino, theta, in_phase, by_phase, header, model,
                             mean, std, fill, device, baselines=baseline_arrays)
            recon[in_phase] = r
            errs[in_phase] = e
        # 投影域误差图（两个输入相位各一，用于"缺陷附近视角"分析）
        for in_phase in ("A", "B"):
            np.save(OUT_DIR / f"proj_error_map_{in_phase}.npy", errs[in_phase])

    # ---- 图像域指标（逐 test 样本，方法 × 相位）
    mask = br.fov_mask()
    test_rows = [r for r in rows if r["split"] == "test"]
    if len(test_rows) != 45:
        raise br.ReconSpecError(f"test 期望 45 行，实得 {len(test_rows)}")
    per_sample: list[dict] = []
    for r in test_rows:
        in_phase = br.opposite_phase(r["phase"])
        for meth in METHODS:
            img = dense if meth == "dense" else recon[in_phase][meth]
            m = br.image_metrics(dense, img, mask)
            per_sample.append({
                "method": meth, "energy_MeV": ENERGY,
                "target_angle_deg": float(r["target_angle_deg"]),
                "target_phase": r["phase"], "input_phase": in_phase,
                "n_valid_pixels": m["n_valid_pixels"],
                "rmse_valid": m["rmse_valid"], "rmse_full": m["rmse_full"],
                "psnr": m["psnr"], "ssim": m["ssim"],
            })
    with open(OUT_DIR / "image_per_sample.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(per_sample[0].keys()))
        w.writeheader()
        w.writerows(per_sample)

    # 池化（与 cnn_recon.write_outputs 同式：按 n_valid_pixels 加权 MSE）
    per_method: dict[str, dict] = {}
    for meth in METHODS:
        sub = [s for s in per_sample if s["method"] == meth]
        n = len(sub)
        nv = sum(s["n_valid_pixels"] for s in sub)
        mse_v = sum(s["rmse_valid"] ** 2 * s["n_valid_pixels"] for s in sub) / nv
        per_method[meth] = {
            "method_zh": METHOD_ZH[meth], "n_samples": n,
            "domain": "image",
            "rmse_valid": math.sqrt(mse_v),
            "rmse_full": math.sqrt(sum(s["rmse_full"] ** 2 for s in sub) / n),
            "psnr": float(np.mean([s["psnr"] for s in sub])),
            "ssim": float(np.mean([s["ssim"] for s in sub])),
            "data_range": float(dense.max() - dense.min()),
        }
    # 按输入相位拆开（同图重复计数，仅用于内部核对，不作主指标）
    per_phase: dict[str, dict] = {}
    for meth in METHODS:
        for ph in ("A", "B"):
            sub = [s for s in per_sample if s["method"] == meth and s["input_phase"] == ph]
            if not sub:
                continue
            nv = sum(s["n_valid_pixels"] for s in sub)
            mse_v = sum(s["rmse_valid"] ** 2 * s["n_valid_pixels"] for s in sub) / nv
            per_phase[f"{meth}|{ph}"] = {"n_samples": len(sub),
                                         "rmse_valid": math.sqrt(mse_v)}

    # ---- 图像域 CNR：缺陷重建 vs 同方法无缺陷参照
    # 参照**文件名**里的能量段用紧凑写法：`3` 而不是 `3.0`（`{3.0:g}` → `3`，
    # `{1.5:g}` → `1.5`）。缺陷表可以写 `3.0`，但均匀参照 recon 目录里存的是
    # `cnn_3_A.npy`；直接用 ENERGY 拼会得到 `cnn_3.0_A.npy` → 全部参照"缺失"，
    # CNR 整块报废（3.0 MeV 档踩过这个坑，1.5/4.5/6 恰好两种写法一致才没暴露）。
    try:
        ET = f"{float(ENERGY):g}"
    except ValueError:
        ET = ENERGY
    ref_paths = {
        "dense": br.RUNS_DIR / f"{REF_RUN_ID}_cnn_recon" / "recon" / f"dense_{ET}_fbp_hamming.npy",
        "cnn_completed": br.RUNS_DIR / f"{REF_RUN_ID}_cnn_recon" / "recon" / f"cnn_{ET}_A.npy",
        "linear_interp": br.RUNS_DIR / f"{REF_RUN_ID}_linear_interp_image" / "recon" / f"linear_{ET}_A.npy",
        "fbp_sparse": br.RUNS_DIR / f"{REF_RUN_ID}_fbp_sparse" / "recon" / f"fbp_{ET}_A.npy",
        "sart_sparse": br.RUNS_DIR / f"{REF_RUN_ID}_sart_sparse" / "recon" / f"sart_{ET}_A.npy",
    }
    cnr: dict[str, dict] = {}
    for meth, p in ref_paths.items():
        if not p.is_file():
            cnr[meth] = {"status": "missing reference", "path": str(p.relative_to(ROOT))}
            continue
        ref = br.restrict_to_fov(np.load(p))
        cur = dense if meth == "dense" else recon["A"][meth]
        d = br.restrict_to_fov(cur - ref)
        cnr[meth] = {"status": "ok",
                     "in_image": bubble_metrics(cur),          # 主口径
                     "vs_noflaw": defect_cnr(d, mask),         # 副口径
                     "rod_centres": rod_centre_values(cur)}

    # ---- 投影域：train.py 给的 45 个 test 视角误差 + 几何相关性
    # 口径守卫：这一段读的是 runs/<RUN_ID>/test_per_sample.csv，即**训练时那个数据表**的
    # test split。只有 RUN_ID 确实用 sparse_targets_flaw.csv 训练过，它才是缺陷测试集；
    # 否则（例如拿均匀数据训练的模型做零样本评估）它就是另一个域的 test split，
    # 数字不可与缺陷域数字并列。默认直接报错，确需跨域评估时传
    # --allow-domain-mismatch，此时该段整体置 null 并显式标注。
    run_cfg = json.loads((br.RUNS_DIR / RUN_ID / "config.json").read_text(encoding="utf-8"))
    trained_on = (run_cfg.get("data") or {}).get("targets")
    mismatch = trained_on != FLAW_CSV.name
    if mismatch and not ALLOW_DOMAIN_MISMATCH:
        raise br.ReconSpecError(
            f"[FATAL] runs/{RUN_ID}/config.json 的 data.targets={trained_on!r}，"
            f"不是 {FLAW_CSV.name!r}。\n"
            f"        该 run 的训练数据表不是缺陷表，其 test_per_sample.csv 属于别的域，\n"
            f"        投影域 all32 数字不可与缺陷域并列（跨域混引）。\n"
            f"        若确需对该模型做缺陷域零样本评估，请加 --allow-domain-mismatch，\n"
            f"        此时投影域 all32 段会置 null，只保留缺陷域上算出的 row16 / 图像域 / CNR。")
    if mismatch:
        print(f"[WARN] data.targets={trained_on!r} 非缺陷表："
              f"投影域 all32 段置 null（--allow-domain-mismatch）", flush=True)

    src = br.RUNS_DIR / RUN_ID / "test_per_sample.csv"
    with open(src, newline="", encoding="utf-8") as fh:
        train_ps = list(csv.DictReader(fh))
    if mismatch:
        train_ps = []
    ang = np.array([float(r["target_angle_deg"]) for r in train_ps])
    rv = np.array([float(r["rmse_valid"]) for r in train_ps])
    col_def = TRAJ["offset"] + TRAJ["amp"] * np.cos(
        np.radians(ang + TRAJ["phase_deg"]))
    speed = np.abs(np.sin(np.radians(ang + TRAJ["phase_deg"])))   # |dcos/dθ|
    offset = np.abs(col_def - 600.0)
    proj_angle = []
    for i in range(len(ang)):
        proj_angle.append({"target_angle_deg": float(ang[i]),
                           "defect_col_px": float(col_def[i]),
                           "abs_offset_px": float(offset[i]),
                           "defect_speed_au": float(speed[i]),
                           "rmse_valid": float(rv[i])})
    if train_ps:
        with open(OUT_DIR / "proj_angle.csv", "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=list(proj_angle[0].keys()))
            w.writeheader()
            w.writerows(proj_angle)
    else:
        (OUT_DIR / "proj_angle.csv").write_text(
            "# 空：该 run 的训练数据表不是缺陷表，投影域 test 视角不适用\n",
            encoding="utf-8")

    def corr(a: np.ndarray, b: np.ndarray) -> dict:
        r = float(np.corrcoef(a, b)[0, 1])
        n = len(a)
        t = r * math.sqrt((n - 2) / max(1e-12, 1 - r * r))
        return {"pearson_r": r, "t": t, "n": n,
                "df": n - 2, "note": "双侧 t 检验；|t|>2.02 即 p<0.05 (df=43)"}

    # ---- 逐视角 row16 误差（比 train.py 的 32 行 mask 加权更贴近缺陷所在的行）
    def traj_col(a: np.ndarray) -> np.ndarray:
        return TRAJ["offset"] + TRAJ["amp"] * np.cos(np.radians(a + TRAJ["phase_deg"]))

    view_stats: dict[str, dict] = {}
    for in_phase in ("A", "B"):
        e = errs[in_phase].astype(np.float64)
        angs = (np.arange(1, 360, 2) if in_phase == "A" else np.arange(0, 360, 2))
        if e.shape[0] != len(angs):
            raise br.ReconSpecError(
                f"相位 {in_phase} 误差图 {e.shape[0]} 行 != {len(angs)} 个预测视角")
        per_view = np.sqrt((e ** 2).mean(axis=1))
        sp = np.abs(np.sin(np.radians(angs + TRAJ["phase_deg"])))
        off = np.abs(traj_col(angs.astype(float)) - 600.0)
        on_tr, off_tr, rel_on, rel_off = [], [], [], []
        for i, a in enumerate(angs):
            c = int(round(float(traj_col(float(a)))))
            m = np.zeros(br.OUTPUT_SIZE, dtype=bool)
            m[max(0, c - 60):min(br.OUTPUT_SIZE, c + 61)] = True
            ae = np.abs(e[i])
            ag = np.abs(sino[int(a)])
            on_tr.append(float(ae[m].mean()))
            off_tr.append(float(ae[~m].mean()))
            # 相对误差：先按区域把 |误差| 与 |真值| 各自求和再相除，避免逐像素除零
            rel_on.append(float(ae[m].sum() / max(1e-12, ag[m].sum())))
            rel_off.append(float(ae[~m].sum() / max(1e-12, ag[~m].sum())))
        view_stats[in_phase] = {
            "n_views": len(angs),
            "rmse_row16_mean": float(per_view.mean()),
            "rmse_row16_std": float(per_view.std(ddof=1)),
            "corr_rmse_vs_defect_speed": corr(per_view, sp),
            "corr_rmse_vs_defect_offset": corr(per_view, off),
            "mean_abs_err_on_trajectory": float(np.mean(on_tr)),
            "mean_abs_err_off_trajectory": float(np.mean(off_tr)),
            "ratio_on_off": float(np.mean(on_tr) / np.mean(off_tr)),
            "rel_err_on_trajectory": float(np.mean(rel_on)),
            "rel_err_off_trajectory": float(np.mean(rel_off)),
            "rel_ratio_on_off": float(np.mean(rel_on) / np.mean(rel_off)),
            "definition": ("逐预测视角、只取 row16；on-trajectory = 缺陷轨迹列 ±60 px；"
                           "rel_err = Σ|误差| / Σ|真值|（分区域求和后相除）"),
        }

    # ---- linear_interp 的 row16 基线（补算，与上面 CNN 的 row16 并列）
    # 背景：`linear_interp` 原先只在**图像域**作为 baseline 出现，投影域 row16
    # 只算了 CNN 的误差（`errs` 来自模型预测）。审稿口径要求 row16 上 CNN vs
    # linear 并列，故在此用与 `recon_all()` 中线性插值补全**逐字同式**的两点均值
    # `0.5*(sino[a-1] + sino[a+1])` 补出 linear 的 row16 误差图。
    # 注意：`sino` 已由 build_sinogram 取成 row16 单行，形状 (360, 1200)。
    idx_lin = {int(a): i for i, a in enumerate(np.round(theta).astype(np.int64))}
    view_stats_lin: dict[str, dict] = {}
    for in_phase in ("A", "B"):
        angs_l = (np.arange(1, 360, 2) if in_phase == "A"
                  else np.arange(0, 360, 2))
        e_l = np.empty((len(angs_l), br.OUTPUT_SIZE), dtype=np.float64)
        for i, a in enumerate(angs_l):
            lo = sino[idx_lin[(a - 1) % 360]]
            hi = sino[idx_lin[(a + 1) % 360]]
            e_l[i] = 0.5 * (lo + hi) - sino[idx_lin[a]]
        per_view_l = np.sqrt((e_l ** 2).mean(axis=1))
        view_stats_lin[in_phase] = {
            "n_views": len(angs_l),
            "rmse_row16_mean": float(per_view_l.mean()),
            "rmse_row16_std": float(per_view_l.std(ddof=1)),
            "definition": ("逐预测视角、只取 row16；linear_interp = "
                           "0.5*(sino[a-1]+sino[a+1])，与 recon_all() 中线性插值"
                           "补全逐字同式（该式在 recon_all 内未保留误差图，故在此重算）"),
        }

    if mismatch:
        proj_stats = {
            "n_test_views": 0,
            "rmse_valid_mean": None,
            "rmse_valid_std": None,
            "rmse_valid_min": None,
            "rmse_valid_max": None,
            "corr_rmse_vs_defect_speed": None,
            "corr_rmse_vs_defect_offset": None,
            "test_view_source": (
                f"null：runs/{RUN_ID} 的 data.targets={trained_on!r} 非缺陷表，"
                "其 test_per_sample.csv 属于别的域，禁止与缺陷域数字并列"),
            "domain_mismatch": True,
            "trained_on": trained_on,
            "per_view_row16": view_stats,
            "per_view_row16_linear": view_stats_lin,
        }
    else:
        proj_stats = {"n_test_views": len(ang),
                      "rmse_valid_mean": float(rv.mean()),
                      "rmse_valid_std": float(rv.std(ddof=1)),
                      "rmse_valid_min": float(rv.min()),
                      "rmse_valid_max": float(rv.max()),
                      "corr_rmse_vs_defect_speed": corr(rv, speed),
                      "corr_rmse_vs_defect_offset": corr(rv, offset),
                      "test_view_source": "train.py 的 test_per_sample.csv（全 32 行 mask 加权）",
                      "domain_mismatch": False,
                      "trained_on": trained_on,
                      "per_view_row16": view_stats,
                      "per_view_row16_linear": view_stats_lin}

    print("投影域轨迹匹配滤波（缺陷检出判据）...", flush=True)
    mf = proj_matched_filter(sino, fill)
    for k, v in mf["by_offset"].items():
        print(f"  {k:14s} ΔL={v['mean_dlog']:+.5f} ± {v['sem']:.5f}  "
              f"t={v['t_stat']:+.2f}", flush=True)
    d4 = mf["d4_contrast"]
    print(f"  D4 对照：offset0 − 伪轨迹基线 = {d4['offset0_minus_pseudo_baseline']:+.5f} "
          f"± {d4['sem']:.5f}  t={d4['t_stat']:+.2f}", flush=True)

    metrics = {
        "run_id": f"{RUN_ID}_flaw_step3",
        "reference_run_id": RUN_ID,
        "domain": "image",
        "task": (f"缺陷数据（{FLAW_ROOT.name}，{ENERGY} MeV，单缺陷棒 {DEFECT_ROD}）"
                 "域内训练后的评估"),
        "denoise_note": "本文件所有 rmse_valid/psnr/ssim 均为**图像域**，与投影域不可同表",
        "dense_reference": "360 视角全实测 hamming FBP，与各方法同代码、同 FOV、同裁剪",
        "per_method": per_method,
        "per_method_by_input_phase": per_phase,
        "projection_domain": proj_stats,
        "projection_matched_filter": mf,
        "defect_geometry": {
            "defect_rod": DEFECT_ROD,
            "rod_radius_px": R_DEFECT_PX,
            "rod_az_img_deg": AZ_DEFECT_IMG,
            "traj": dict(TRAJ),
            "traj_source": TRAJ_SOURCE,
            "az_mapping_note": (
                "模块级 AZ_DEFECT_IMG = phase−90 只在角棒 phase≈225° 时碰巧成立"
                "（角棒 224.7°→134.7° 对；边棒 180.2°→90.2° 错 90°）。"
                "本批次 ROI 由 --defect-rod 查 rod_centers() 定位（图像域最暗棒锚定），"
                "D4 轨迹由 --traj-json 的投影域拟合给出，两者互不依赖。"),
        },
        "uniform_reference": {
            "run_id": REF_RUN_ID,
            "paths": {k: str(v.relative_to(ROOT)) for k, v in ref_paths.items()},
            "fill_mismatch_note": ("均匀侧 zero_log_fill=5.50、缺陷侧 5.10（各自 config 的裁决值）。"
                                   "该差异只影响零计数像素、且在重建后表现为全局常数项，"
                                   "在 CNR 的'缺陷 ROI − 背景环'差分里抵消；"
                                   "故不重算均匀侧重建。"),
        },
        "defect_cnr": cnr,
        "table_note_en": ("Image-domain metrics from 180 measured plus 180 completed views; "
                          "the completed set is not restricted to the held-out split."),
        "git_commit": br.git_info()[0], "git_dirty": br.git_info()[1],
    }
    (OUT_DIR / "image_metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2, default=float) + "\n",
        encoding="utf-8")

    cfg_out = {
        "run_id": f"{RUN_ID}_flaw_step3",
        "method": "step3_eval",
        "domain": "image",
        "data": {"csv": str(FLAW_CSV.name), "n_rows": len(rows),
                 "split": {"train": 270, "val": 45, "test": 45},
                 "data_root": str(FLAW_ROOT.relative_to(ROOT)),
                 "energy_MeV": ENERGY,
                 "defect": {"rod_ij": [-1, 1], "world_cm": [-2.6, 2.6],
                            "source": "缺陷版 DetectorConstruction.cc（代码为准）"}},
        "norm": {"norm_mean": mean, "norm_std": std, "zero_log_fill": fill},
        "recon": {"filter_name": FILTER, "circle": True,
                  "sart_n_iter": SART_N_ITER, "sart_relaxation": SART_RELAX,
                  "fov_restriction": "restrict_to_fov（圆外置零）"},
        "trajectory_fit": TRAJ,
        "trajectory_validation": {
            "source": "scripts/localize_flaw_cavity.py + runs/_flaw_cavity_localization.json",
            "conclusion": ("四档能量（1.5/3/4.5/6 MeV）的 dense 360 视角 FBP 里，9 根棒中心 "
                           "r<=15 px 圆盘均值最暗的一根恒为 corner1（r=367.7 px, az=135deg），"
                           "TRAJ 指向同一位置 → 多能量可直接复用，无需逐能量重标定"),
            "projection_fit_note": ("投影域正弦拟合在 1.5 MeV 因宽影重叠偏移到 amp=354.7 px、"
                                    "残差 62.8 px（其余三档 amp=367.2~368.9、残差 7.7~7.8 px），"
                                    "图像域证据优先，故不改 TRAJ"),
        },
        "baseline_reuse_from": args.baseline_from,
        "geometry_px": {"r_corner": R_CORNER, "r_edge": R_EDGE,
                        "note": "栅距 2.6 cm、M_eff≈1、1 cm=100 px"},
        "methods": list(METHODS),
        "git_commit": br.git_info()[0], "git_dirty": br.git_info()[1],
    }
    (OUT_DIR / "config.json").write_text(
        json.dumps(cfg_out, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    # ---- 控制台摘要
    print("\n=== 图像域五项指标（45 个 test 视角池化）===")
    print(f"{'method':16s} {'rmse_valid':>11s} {'rmse_full':>10s} {'psnr':>7s} {'ssim':>7s}")
    for meth in METHODS:
        m = per_method[meth]
        print(f"{meth:16s} {m['rmse_valid']:11.6f} {m['rmse_full']:10.6f} "
              f"{m['psnr']:7.2f} {m['ssim']:7.4f}")
    print("\n=== 缺陷 CNR（图像域，主口径：腔内 vs 同棒外环）===")
    for meth in METHODS:
        c = cnr.get(meth, {})
        if c.get("status") != "ok":
            print(f"{meth:16s} 参照缺失")
            continue
        im = c["in_image"]
        nf = c["vs_noflaw"]
        print(f"{meth:16s} μ_bub={im['mu_bubble']:.5f} μ_ref={im['mu_reference']:.5f} "
              f"Δ={im['contrast']:.5f} σ={im['noise_sd']:.5f} CNR={im['cnr']:7.2f}  "
              f"空腔半径 {im['bubble_radius_px']:.0f}px "
              f"| 副口径(vs 无缺陷) CNR={nf['cnr']:+.2f} 排名 {nf['defect_rank_among_9']}/9 "
              f"比值 {nf['contrast_ratio']:.1f}×")
    print("\n=== 投影域 test 视角误差 ===")
    if proj_stats["rmse_valid_mean"] is None:
        print(f"[跨域] {proj_stats['test_view_source']}")
    else:
        print(f"rmse_valid mean={proj_stats['rmse_valid_mean']:.6f} "
              f"std={proj_stats['rmse_valid_std']:.6f} "
              f"[{proj_stats['rmse_valid_min']:.6f}, {proj_stats['rmse_valid_max']:.6f}]")
        for k in ("corr_rmse_vs_defect_speed", "corr_rmse_vs_defect_offset"):
            c = proj_stats[k]
            print(f"{k}: r={c['pearson_r']:+.3f}  t={c['t']:+.2f}  n={c['n']}")
    print("--- 逐视角 row16（更贴近缺陷那一行）---")
    for ph, v in view_stats.items():
        print(f"[{ph}] rmse_row16 mean={v['rmse_row16_mean']:.4f} "
              f"std={v['rmse_row16_std']:.4f} | "
              f"绝对误差 轨迹带 {v['mean_abs_err_on_trajectory']:.4f} / 带外 "
              f"{v['mean_abs_err_off_trajectory']:.4f} = {v['ratio_on_off']:.3f} | "
              f"相对误差 轨迹带 {v['rel_err_on_trajectory']:.4f} / 带外 "
              f"{v['rel_err_off_trajectory']:.4f} = {v['rel_ratio_on_off']:.3f} | "
              f"r(speed)={v['corr_rmse_vs_defect_speed']['pearson_r']:+.3f} "
              f"(t={v['corr_rmse_vs_defect_speed']['t']:+.2f})")
    print("--- 逐视角 row16：linear_interp 基线（与上面 CNN 并列）---")
    for ph, v in view_stats_lin.items():
        print(f"[{ph}] linear rmse_row16 mean={v['rmse_row16_mean']:.4f} "
              f"std={v['rmse_row16_std']:.4f}  n_views={v['n_views']}")
    print(f"\n输出目录: {OUT_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
