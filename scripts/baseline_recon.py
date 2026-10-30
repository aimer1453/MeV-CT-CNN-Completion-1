#!/usr/bin/env python
# -*- coding: utf-8 -*-
r"""
scripts/baseline_recon.py —— 图像域重建基线（FBP / SART）

任务定位
--------
本脚本回答的是「**不做正弦图补全、直接拿稀疏投影重建**」这条传统路线能做到多好，
用于论文 §4.6 Table 4 / §5.6 Fig. 5 与传统方法的对照。

它**不是**投影域基线。投影域基线见 `scripts/baseline_linear.py`。

几何与数据口径
--------------
  - 探测器阵列 `(32, 1200)`：**32 是探测器行（sensor_y），1200 是探测器列（sensor_x）**。
    角度轴在**文件名**里，不在数组里。本脚本取 `recon_row = 16`（取自
    `results/scan_parameters.json`）得到每角度 `(1200,)`，360 角度拼成正弦图 `(360, 1200)`。
  - 360 个独立角度、1° 步进（`angles.csv` 每档 360 行，实测步进恒为 1.0）。
  - `iradon` 要求输入形状 `(n_detector, n_angles)`，故传入 `sino.T = (1200, n_angles)`。
  - `empty_proj.npy` 在 `{scan_dir}/results/empty_proj.npy`，**不在 scan 根目录**。

两套重建（这是本脚本的全部内容）
--------------------------------
  dense 参考 : 360 角度（1° 步进）→ FBP  → (1200, 1200)   ← 论文里「理想全采样」
  sparse    : 180 角度（2° 步进）→ FBP 或 SART → (1200, 1200)

  稀疏集取 **输入相位**，不是 target 相位。`sparse_targets.csv` 的 `phase` 列是 **target**
  相位（A=偶数角，B=奇数角），model 的输入通道恒为**相反相位**，故
  输入相位 = `opposite(target_phase)`。这一点与 `data/dataset.py` 完全一致，
  本脚本只读 CSV 的 `phase` 列，不重新推导。

  不合并 phase A + B 作为稀疏输入（避免与 dense 的 360 角语义混淆）。

域与指标（**必须连同 `domain: "image"` 一起引用**）
--------------------------------------------------
  - 物理域：`L = -log((I+1e-10)/(E+1e-10))`，与 `data/dataset.py` 的 `log_ratio` 同式。
    无效像素 `~isfinite(L)` 在 **-log 域**填 `zero_log_fill`（=5.50），与 CNN 同口径。
    重建全程在 -log 域，**不做归一化**（基线不经过网络）。
  - `rmse_valid`：FOV（重建圆，`r <= 600`）上的 RMSE —— 图像域，**与投影域 rmse_valid 不同域**。
  - `rmse_full` ：全部 1200×1200 像素。
  - **图像域关系与投影域相反**：投影域恒有 `rmse_full >= rmse_valid`；图像域因为
    `iradon(circle=True)` 把圆外**两张图都置成精确 0**，圆外误差贡献为 0，故必然
    `rmse_full <= rmse_valid`，且 `rmse_full == rmse_valid * sqrt(n_valid/n_full)`。
    沿用投影域断言会直接误判为失败（实测已踩过）。因此图像域 `rmse_full` 被圆外零区
    稀释，**只作格式占位，不建议进论文**。
  - **FOV 限制必须显式做一次**：`iradon(circle=True)` 自动把圆外置零；但 `iradon_sart`
    **没有 `circle` 参数**，圆外会留下 ART 残余（实测 max|diff| ≈ 4e-5）。故两种方法的
    重建结果统一过 `restrict_to_fov()`，保证支撑域一致。
  - `psnr`      ：`10*log10(data_range^2 / mse_valid)`，`data_range = ref.max()-ref.min()`。
  - `ssim`      ：在**中心 848×848 裁剪**上算。848 = int(1200/√2)，该方框半对角 599.6 < 600，
                  全部落在重建圆内，避免圆外恒零区把 SSIM 抬虚高。

为什么不用 `results/postprocessing/recon_original_fbp_row16.npy` 当 dense 参考
------------------------------------------------------------------------------
那是一套独立管线的产物（与本脚本 FBP 在 FOV 内相关系数约 0.68，且被 clip 到 >=0）。
若 sparse 用本脚本、dense 用那套，则差异里混进了**管线差异**，无法归因到采样密度。
故 dense 参考由本脚本用**同一份代码、同一 filter、同一填充**重算，保证
「sparse vs dense 只差角度采样数」。

输出（`runs/<cnn_run_id>_<method>/`）
-------------------------------------
  config.json          method / filter / sart 配置 / 域与指标定义 / git
  train_log.csv        14 列，每档一行，epoch=-1，split=test
  test_metrics.json    结构同 CNN + `domain` / `reference` / `metric_definition`
  test_per_sample.csv  11 列，180 行
  recon/               可用 `--save-recon` 落盘（.npy 已被 .gitignore 忽略）

用法
----
  conda run -n ct_full python scripts/baseline_recon.py --methods fbp
  conda run -n ct_full python scripts/baseline_recon.py --methods sart --sart-iter 20
  conda run -n ct_full python scripts/baseline_recon.py --methods fbp,sart --reuse
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT))
from data.csv_paths import resolve_csv  # noqa: E402  CSV 统一落点 csv_data/

from data.dataset import LOG_EPS, log_ratio  # noqa: E402

SPARSE_TARGETS = resolve_csv("sparse_targets.csv")
ANGLES_CSV = resolve_csv("angles.csv")
RUNS_DIR = ROOT / "runs"
DEFAULT_CNN_RUN_ID = "20261022_2352_cnn_bs8_lr1e-4"

ENERGIES: tuple[str, ...] = ("1.5", "3", "4.5", "6")
PHASES: tuple[str, ...] = ("A", "B")
NAN = float("nan")

#: 角度 → 相位：A = 偶数角，B = 奇数角（与 `sparse_targets.csv` 的 phase 列同义）
PHASE_PARITY: dict[str, int] = {"A": 0, "B": 1}

#: 取哪一行探测器做重建（来自 `results/scan_parameters.json` 的 recon_row）
RECON_ROW = 16
#: 输出图像边长 = 探测器列数
OUTPUT_SIZE = 1200
#: iradon(circle=True) 的判据：`xpr**2 + ypr**2 > radius**2` 置零，radius = output_size//2
CIRCLE_RADIUS = OUTPUT_SIZE // 2
#: SSIM 用中心正方形边长，保证整个方框落在重建圆内
SSIM_CROP = int(OUTPUT_SIZE / math.sqrt(2.0))     # 848

LOG_COLUMNS: tuple[str, ...] = (
    "epoch", "split", "energy_MeV", "n_samples", "coverage", "valid_coverage",
    "rmse_valid", "rmse_full", "psnr", "ssim", "n_invariant_warn", "lr",
    "loss_train", "loss_val",
)
PER_SAMPLE_COLUMNS: tuple[str, ...] = (
    "energy_MeV", "target_angle_deg", "split", "n_valid_pixels",
    "coverage", "valid_coverage", "rmse_valid", "rmse_full", "psnr", "ssim",
    "rmse_invariant_warn",
)

RMSE_TOL = 1e-9


class ReconSpecError(RuntimeError):
    """输入或口径不满足时抛出（fail fast，不做兜底）。"""


# ------------------------------------------------------------------ 小工具
def git_info() -> tuple[str, bool]:
    def _run(*args: str) -> str:
        try:
            return subprocess.run(("git", *args), cwd=ROOT, capture_output=True,
                                  text=True, check=False).stdout.strip()
        except OSError:
            return ""

    return _run("rev-parse", "HEAD") or "UNKNOWN", bool(_run("status", "--porcelain"))


def fov_mask() -> np.ndarray:
    """与 `iradon(circle=True)` 完全一致的圆盘掩膜（含边界）。

    源码判据：`xpr, ypr = mgrid[:N,:N] - N//2`，`xpr**2 + ypr**2 > radius**2` 置零，
    故有效区 = `xpr**2 + ypr**2 <= radius**2`。
    """
    xpr, ypr = np.mgrid[:OUTPUT_SIZE, :OUTPUT_SIZE] - CIRCLE_RADIUS
    return (xpr * xpr + ypr * ypr) <= CIRCLE_RADIUS ** 2


def ssim_crop_slices() -> tuple[slice, slice]:
    lo = (OUTPUT_SIZE - SSIM_CROP) // 2
    return slice(lo, lo + SSIM_CROP), slice(lo, lo + SSIM_CROP)


def load_angle_table(energy: str, data_root: Path) -> list[dict]:
    """按 `angles.csv` 取该能量的 360 行（角度 + 文件），不重新推导文件名。

    注意：`angles.csv` 的 `file` 列**已含 `3_3/` 前缀**（相对仓库根），而 `scan_dir` 列
    **不含**该前缀，故投影文件从 `ROOT` 解析、empty_proj 从 `data_root` 解析。
    """
    if not ANGLES_CSV.is_file():
        raise ReconSpecError(f"缺少 {ANGLES_CSV}")
    with open(ANGLES_CSV, newline="", encoding="utf-8") as fh:
        rows = [r for r in csv.DictReader(fh) if r["energy_MeV"] == energy]
    if not rows:
        raise ReconSpecError(f"angles.csv 中没有 energy_MeV={energy} 的行")
    rows.sort(key=lambda r: float(r["angle_deg"]))
    angs = np.array([float(r["angle_deg"]) for r in rows])
    if len(rows) != 360:
        raise ReconSpecError(f"[{energy} MeV] angles.csv 期望 360 行，实得 {len(rows)}")
    steps = np.unique(np.round(np.diff(angs), 6))
    if steps.size != 1 or abs(steps[0] - 1.0) > 1e-9:
        raise ReconSpecError(f"[{energy} MeV] 角度步进期望恒为 1.0，实得 {steps}")
    for r in rows:
        if not (ROOT / r["file"]).is_file():
            raise ReconSpecError(f"[{energy} MeV] 缺少投影文件 {ROOT / r['file']}")
    return rows


def load_empty(data_root: Path, scan_dir: str) -> np.ndarray:
    p = data_root / scan_dir / "results" / "empty_proj.npy"
    if not p.is_file():
        raise ReconSpecError(f"缺少 empty_proj: {p}")
    return np.load(p)


def build_sinogram(energy: str, data_root: Path, fill: float) -> tuple[np.ndarray, np.ndarray]:
    """返回 `(sino, theta)`：`sino` 形状 `(360, 1200)`，`theta` 形状 `(360,)`，单位度。"""
    rows = load_angle_table(energy, data_root)
    scan_dirs = {r["scan_dir"] for r in rows}
    if len(scan_dirs) != 1:
        raise ReconSpecError(f"[{energy} MeV] angles.csv 内出现多个 scan_dir: {scan_dirs}")
    empty = load_empty(data_root, scan_dirs.pop())

    sino = np.empty((len(rows), OUTPUT_SIZE), dtype=np.float64)
    theta = np.empty(len(rows), dtype=np.float64)
    for i, r in enumerate(rows):
        proj = np.load(ROOT / r["file"]).astype(np.float32)
        if proj.shape != (32, OUTPUT_SIZE):
            raise ReconSpecError(f"[{energy} MeV] 投影形状期望 (32,1200)，实得 {proj.shape}: {r['file']}")
        L = log_ratio(proj, empty).astype(np.float64)
        sino[i] = np.where(np.isfinite(L), L, float(fill))[RECON_ROW]
        theta[i] = float(r["angle_deg"])
    return sino, theta


def select_phase(theta: np.ndarray, phase: str) -> np.ndarray:
    """选择某个相位的视角下标。A = 偶数角，B = 奇数角。"""
    parity = PHASE_PARITY[phase]
    return np.isclose(np.mod(np.round(theta).astype(np.int64), 2), parity)


def opposite_phase(phase: str) -> str:
    if phase not in PHASE_PARITY:
        raise ReconSpecError(f"未知相位 {phase!r}")
    return "B" if phase == "A" else "A"


# ------------------------------------------------------------------ 重建
def recon_fbp(sino: np.ndarray, theta: np.ndarray, filter_name: str) -> np.ndarray:
    from skimage.transform import iradon

    return iradon(sino.T, theta=theta, filter_name=filter_name,
                  output_size=OUTPUT_SIZE, circle=True, interpolation="linear")


def recon_sart(sino: np.ndarray, theta: np.ndarray, n_iter: int,
               relaxation: float, log_prefix: str = "") -> np.ndarray:
    """SART，`clip=None`（与 CNN 同 -log 域口径，允许负值）。

    第 0 次迭代 image=None（零初值），之后用上一次结果热启动，共 `n_iter` 次。
    """
    from skimage.transform import iradon_sart

    img: np.ndarray | None = None
    t_start = time.time()
    for k in range(n_iter):
        img = iradon_sart(sino.T, theta=theta, image=img, clip=None,
                          relaxation=relaxation)
        if k == 0 or (k + 1) % 5 == 0 or k + 1 == n_iter:
            el = time.time() - t_start
            print(f"    {log_prefix}iter {k + 1:>2}/{n_iter}  {el:6.1f}s "
                  f"(min {img.min():+.5f} max {img.max():+.5f})", flush=True)
    assert img is not None
    return img


def restrict_to_fov(img: np.ndarray) -> np.ndarray:
    """把重建圆外置零。

    FBP 用 `circle=True` 已自动置零（本函数对它等价于 no-op）；**`iradon_sart` 没有
    `circle` 参数**，圆外会留下 ART 的残余值（实测 max|diff| ~4e-5），必须显式置零，
    否则两张图的支撑域不一致，`rmse_full` 与 `rmse_valid` 的关系会被污染。
    """
    return np.where(fov_mask(), img, 0.0)


def get_dense_reference(energy: str, data_root: Path, fill: float, filter_name: str,
                        cache_dir: Path, reuse: bool) -> np.ndarray:
    p = cache_dir / f"dense_{energy}_fbp_{filter_name}.npy"
    if reuse and p.is_file():
        print(f"  [reuse] {p.name}")
        return restrict_to_fov(np.load(p))
    sino, theta = build_sinogram(energy, data_root, fill)
    print(f"  [{energy} MeV] dense 360 视角重建 ...", flush=True)
    t0 = time.time()
    img = restrict_to_fov(recon_fbp(sino, theta, filter_name))
    print(f"  [{energy} MeV] dense 完成 {time.time() - t0:.1f}s "
          f"(min {img.min():+.5f} max {img.max():+.5f})", flush=True)
    np.save(p, img.astype(np.float32))
    return img


def get_sparse_recon(method: str, energy: str, phase: str, data_root: Path, fill: float,
                     filter_name: str, sart_iter: int, sart_relax: float,
                     cache_dir: Path, reuse: bool) -> np.ndarray:
    tag = f"{method}_{energy}_{phase}"
    p = cache_dir / f"{tag}.npy"
    if reuse and p.is_file():
        print(f"  [reuse] {p.name}")
        return restrict_to_fov(np.load(p))
    sino, theta = build_sinogram(energy, data_root, fill)
    sel = select_phase(theta, phase)
    if int(sel.sum()) != 180:
        raise ReconSpecError(f"[{energy} MeV] 相位 {phase} 期望 180 视角，实得 {int(sel.sum())}")
    print(f"  [{energy} MeV / phase {phase}] {method.upper()} 稀疏 {int(sel.sum())} 视角 ...",
          flush=True)
    t0 = time.time()
    if method == "fbp":
        img = recon_fbp(sino[sel], theta[sel], filter_name)
    elif method == "sart":
        img = recon_sart(sino[sel], theta[sel], sart_iter, sart_relax,
                         log_prefix=f"[{energy}/{phase}] ")
    else:
        raise ReconSpecError(f"未知 method {method!r}")
    img = restrict_to_fov(img)
    print(f"  [{energy} MeV / phase {phase}] {method} 完成 {time.time() - t0:.1f}s",
          flush=True)
    np.save(p, img.astype(np.float32))
    return img


# ------------------------------------------------------------------ 指标
def image_metrics(ref: np.ndarray, cmp_: np.ndarray, mask: np.ndarray) -> dict[str, float]:
    """图像域 RMSE / PSNR / SSIM。`ref` 为 dense 参考。

    与投影域**相反**的一条关系（务必不要沿用投影域的断言）：
    投影域恒有 `rmse_full >= rmse_valid`；图像域因 `iradon(circle=True)` 把圆外
    **两张图都置为精确 0**，圆外误差贡献为 0，故必然 `rmse_full <= rmse_valid`，
    且满足 `rmse_full == rmse_valid * sqrt(n_valid / n_full)`。
    本函数断言的是**后一条**，并额外校验圆外确实逐点相等。
    """
    from skimage.metrics import structural_similarity

    sq = (cmp_ - ref) ** 2
    n_valid = int(mask.sum())
    n_full = int(sq.size)
    se_valid = float(sq[mask].sum())
    se_full = float(sq.sum())
    rmse_valid = math.sqrt(se_valid / n_valid)
    rmse_full = math.sqrt(se_full / n_full)

    # 圆外必须逐点相等（circle=True 置零），否则说明参考与比较的圆参数不一致
    out_max = float(np.abs(cmp_[~mask]).max()) if (~mask).any() else 0.0
    if out_max != 0.0:
        raise ReconSpecError(
            f"FOV 圆外存在非零差异（max|diff|={out_max:.3e}），"
            f"说明 circle 参数或重建尺寸不一致")
    expect = rmse_valid * math.sqrt(n_valid / n_full)
    if abs(rmse_full - expect) > 1e-9 * max(1.0, expect):
        raise ReconSpecError(
            f"rmse_full ({rmse_full:.12f}) != rmse_valid*sqrt(n_valid/n_full) "
            f"({expect:.12f})，圆外零误差关系不成立")

    ref_rng = float(ref.max() - ref.min())
    if not ref_rng > 0:
        raise ReconSpecError("dense 参考取值范围退化，data_range<=0")
    psnr = 10.0 * math.log10(ref_rng * ref_rng / (se_valid / n_valid)) if se_valid > 0 else NAN

    sy, sx = ssim_crop_slices()
    # gaussian_weights 会显著变慢且对 1200x1200 无必要，保持默认 uniform
    ssim = float(structural_similarity(ref[sy, sx], cmp_[sy, sx], data_range=ref_rng))
    return {"rmse_valid": rmse_valid, "rmse_full": rmse_full,
            "psnr": psnr, "ssim": ssim, "data_range": ref_rng,
            "n_valid_pixels": n_valid,
            "out_of_fov_max_abs_diff": out_max}


def load_test_rows() -> list[dict]:
    if not SPARSE_TARGETS.is_file():
        raise ReconSpecError(f"缺少 {SPARSE_TARGETS}")
    with open(SPARSE_TARGETS, newline="", encoding="utf-8") as fh:
        rows = [r for r in csv.DictReader(fh) if r["split"] == "test"]
    if not rows:
        raise ReconSpecError("sparse_targets.csv 中 split=test 没有样本")
    per = {e: sum(1 for r in rows if r["energy_MeV"] == e) for e in ENERGIES}
    if any(v == 0 for v in per.values()):
        raise ReconSpecError(f"test 集分能量不完整: {per}")
    for r in rows:
        if r["phase"] not in PHASE_PARITY:
            raise ReconSpecError(f"未知 phase: {r['phase']!r} ({r['target_file']})")
    return rows


# ------------------------------------------------------------------ 主流程
def run_method(method: str, cnn_run_id: str, data_root: Path, fill: float,
               filter_name: str, sart_iter: int, sart_relax: float,
               reuse: bool) -> dict:
    out_dir = RUNS_DIR / f"{cnn_run_id}_{method}_sparse"
    cache_dir = out_dir / "recon"
    cache_dir.mkdir(parents=True, exist_ok=True)
    mask = fov_mask()

    # ---- 1. dense 参考（每档一张，360 视角 FBP）
    print(f"[{method}] 阶段 1/3：dense 参考重建", flush=True)
    dense = {e: get_dense_reference(e, data_root, fill, filter_name, cache_dir, reuse)
             for e in ENERGIES}

    # ---- 2. 稀疏重建（每档 × 每相位一张）
    print(f"[{method}] 阶段 2/3：稀疏 180 视角重建（4 档 × 2 相位 = 8 张）", flush=True)
    sparse = {(e, p): get_sparse_recon(method, e, p, data_root, fill, filter_name,
                                       sart_iter, sart_relax, cache_dir, reuse)
              for e in ENERGIES for p in PHASES}

    # ---- 3. 逐样本指标：按 target 相位取"相反相位"的稀疏重建
    print(f"[{method}] 阶段 3/3：逐样本指标", flush=True)
    rows = load_test_rows()
    per_sample: list[dict] = []
    for r in rows:
        e = r["energy_MeV"]
        if e not in ENERGIES:
            raise ReconSpecError(f"未知能量: {e!r}")
        in_phase = opposite_phase(r["phase"])
        m = image_metrics(dense[e], sparse[(e, in_phase)], mask)
        per_sample.append({
            "energy_MeV": e,
            "target_angle_deg": float(r["target_angle_deg"]),
            "split": "test",
            "n_valid_pixels": m["n_valid_pixels"],
            "coverage": NAN,          # 图像域不涉及投影覆盖率，写 NaN 以示不适用
            "valid_coverage": NAN,
            "rmse_valid": m["rmse_valid"],
            "rmse_full": m["rmse_full"],
            "psnr": m["psnr"],
            "ssim": m["ssim"],
            "rmse_invariant_warn": False,
            "input_phase": in_phase,
            "target_phase": r["phase"],
        })

    # ---- 分能量汇总（sqrt(sum(se)/sum(n))，与 EnergyAccumulator 同式）
    #: 跨全部能量统一的 data_range，用于 `psnr_common`：消除「逐档 data_range 不同」
    #: 导致的跨档不可比问题（1.5 MeV 的 data_range 是 6 MeV 的 5.3 倍，逐档 PSNR 会把
    #: 绝对误差大但相对误差小的档判为更好）。
    dr_common = max(float(dense[e].max() - dense[e].min()) for e in ENERGIES)
    per_energy: dict[str, dict] = {}
    for e in ENERGIES:
        sub = [s for s in per_sample if s["energy_MeV"] == e]
        n = len(sub)
        nv = sum(s["n_valid_pixels"] for s in sub)
        se_v = sum(s["rmse_valid"] ** 2 * s["n_valid_pixels"] for s in sub)
        se_f = sum(s["rmse_full"] ** 2 for s in sub) / n * n  # 每样本全像素数相同
        mse_v = se_v / nv
        per_energy[e] = {
            "n_samples": n,
            "coverage": NAN,
            "valid_coverage": NAN,
            "rmse_valid": math.sqrt(mse_v),
            "rmse_full": math.sqrt(sum(s["rmse_full"] ** 2 for s in sub) / n),
            "psnr": float(np.mean([s["psnr"] for s in sub])),
            "psnr_common": 10.0 * math.log10(dr_common * dr_common / mse_v),
            "ssim": float(np.mean([s["ssim"] for s in sub])),
            "data_range": float(dense[e].max() - dense[e].min()),
            "data_range_common": dr_common,
            "_se_valid": se_v,
            "_se_full": se_f,
            "_n_valid": nv,
        }

    n_all = sum(v["n_samples"] for v in per_energy.values())
    n_v = sum(v["_n_valid"] for v in per_energy.values())
    se_v = sum(v["_se_valid"] for v in per_energy.values())
    overall = {
        "n_samples": n_all,
        "coverage": NAN,
        "valid_coverage": NAN,
        "rmse_valid": math.sqrt(se_v / n_v),
        "rmse_full": math.sqrt(sum(v["_se_full"] for v in per_energy.values()) / n_all),
        "psnr": float(np.mean([v["psnr"] for v in per_energy.values()])),
        "psnr_common": float(np.mean([v["psnr_common"] for v in per_energy.values()])),
        "ssim": float(np.mean([v["ssim"] for v in per_energy.values()])),
    }
    # 图像域的正确关系是 rmse_full <= rmse_valid（圆外两图皆精确为 0）
    if overall["rmse_full"] > overall["rmse_valid"] + RMSE_TOL:
        raise ReconSpecError("图像域总体 rmse_full > rmse_valid，圆外零误差假设被破坏")
    for v in per_energy.values():
        v.pop("_se_valid", None)
        v.pop("_se_full", None)
        v.pop("_n_valid", None)

    return {"method": method, "out_dir": out_dir, "per_sample": per_sample,
            "per_energy": per_energy, "overall": overall,
            "filter_name": filter_name, "sart_iter": sart_iter,
            "sart_relax": sart_relax, "fill": fill}


# ------------------------------------------------------------------ 落盘
def write_outputs(res: dict, cnn_run_id: str) -> dict[str, Path]:
    out_dir: Path = res["out_dir"]
    method: str = res["method"]
    out_dir.mkdir(parents=True, exist_ok=True)
    commit, dirty = git_info()
    cfg_cnn = json.loads((RUNS_DIR / cnn_run_id / "config.json").read_text(encoding="utf-8"))

    method_note = {
        "fbp": "稀疏 180 视角（2° 步进）+ skimage iradon 滤波反投影；无正弦图补全",
        "sart": "稀疏 180 视角（2° 步进）+ skimage iradon_sart 迭代重建；无正弦图补全",
    }[method]

    config = {
        "run_id": out_dir.name,
        "method": f"{method}_sparse",
        "domain": "image",
        "method_note": method_note,
        "reference_run_id": cnn_run_id,
        "git_commit": commit,
        "git_dirty": dirty,
        "geometry": {
            "sensor_x": OUTPUT_SIZE, "sensor_y": 32, "recon_row": RECON_ROW,
            "output_size": [OUTPUT_SIZE, OUTPUT_SIZE],
            "sinogram_shape": [360, OUTPUT_SIZE],
            "note": "32 是探测器行（sensor_y），1200 是探测器列（sensor_x）；角度轴在文件名里",
        },
        "sampling": {
            "dense": {"n_views": 360, "step_deg": 1.0, "source": "angles.csv 全 360 角"},
            "sparse": {"n_views": 180, "step_deg": 2.0,
                       "phase_rule": "A=偶数角，B=奇数角；输入相位 = opposite(target phase)",
                       "why_not_merge": "不合并 A+B，避免与 dense 360 角语义混淆"},
        },
        "recon": {"filter_name": res["filter_name"],
                  "sart_n_iter": res["sart_iter"] if method == "sart" else None,
                  "sart_relaxation": res["sart_relax"] if method == "sart" else None,
                  "sart_clip": None if method == "sart" else None,
                  "clip_note": "clip=None：与 CNN 同 -log 域口径，允许负值" if method == "sart" else None,
                  "fov_restriction": "重建圆外一律置零。FBP 由 circle=True 自动完成；"
                                     "iradon_sart 无 circle 参数，由 restrict_to_fov 显式置零"
                                     "（否则圆外 ART 残余 ~4e-5 会污染 rmse_full）"},
        "norm": {"norm_scope": None,
                 "note": "基线不经过网络，重建全程在 -log 物理域，不做归一化/反归一化"},
        "log_eps": LOG_EPS,
        "zero_log_fill": res["fill"],
        "metric_definition": {
            "domain": "image",
            "rmse_valid": f"FOV 重建圆（xpr^2+ypr^2<={CIRCLE_RADIUS}^2，"
                          f"{int(fov_mask().sum())} 像素）上的 RMSE，图像域",
            "rmse_full": f"全部 {OUTPUT_SIZE}x{OUTPUT_SIZE} 像素上的 RMSE",
            "psnr": "10*log10(data_range^2/mse_valid)，data_range = dense.max()-dense.min()"
                    "（逐档不同，跨档不可比）",
            "psnr_common": "同上但 data_range 取**四档统一**的 max-min（=1.5 MeV 那档的 0.2737），"
                           "用于跨能量比较；逐档 psnr 会把 data_range 大的档判得更好",
            "ssim": f"中心 {SSIM_CROP}x{SSIM_CROP} 裁剪上的 SSIM（该方框半对角 599.6 < 600，"
                    f"全部在重建圆内，避免圆外恒零区抬高 SSIM）",
            "reference": "dense = 360 视角 1° 步进 FBP，由本脚本同代码重算",
            "domain_invariant": "图像域恒有 rmse_full <= rmse_valid，且 "
                                "rmse_full = rmse_valid*sqrt(n_valid/n_full)，"
                                "因为 iradon(circle=True) 把圆外两图都置零；"
                                "该关系与投影域相反，不要照搬投影域断言",
            "cross_domain_warning": "本文件的 rmse_valid 是**图像域**，与 CNN 的投影域 "
                                    "rmse_valid 不同域、数值不可同表比较",
        },
        "data": {"targets": "sparse_targets.csv", "split": "test",
                 "n_samples": res["overall"]["n_samples"],
                 "energies_MeV": list(ENERGIES)},
    }
    p_cfg = out_dir / "config.json"
    p_cfg.write_text(json.dumps(config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    # ---- train_log.csv：14 列，每档一行，epoch=-1
    p_log = out_dir / "train_log.csv"
    with open(p_log, "w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=LOG_COLUMNS)
        w.writeheader()
        for e in ENERGIES:
            v = res["per_energy"][e]
            w.writerow({
                "epoch": -1, "split": "test", "energy_MeV": e,
                "n_samples": v["n_samples"],
                "coverage": "NaN", "valid_coverage": "NaN",
                "rmse_valid": f"{v['rmse_valid']:.6f}",
                "rmse_full": f"{v['rmse_full']:.6f}",
                "psnr": f"{v['psnr']:.6f}",
                "ssim": f"{v['ssim']:.6f}",
                "n_invariant_warn": 0, "lr": "NaN",
                "loss_train": "NaN", "loss_val": "NaN",
            })

    # ---- test_per_sample.csv：11 列（额外 2 列 phase 只进 json，不进 csv 保持列数锁定）
    p_ps = out_dir / "test_per_sample.csv"
    with open(p_ps, "w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=PER_SAMPLE_COLUMNS)
        w.writeheader()
        for r in res["per_sample"]:
            out = {k: r[k] for k in PER_SAMPLE_COLUMNS}
            for k in ("coverage", "valid_coverage"):
                out[k] = "NaN"
            for k in ("rmse_valid", "rmse_full", "psnr", "ssim"):
                out[k] = f"{out[k]:.6f}"
            out["target_angle_deg"] = f"{float(out['target_angle_deg']):g}"
            w.writerow(out)

    # ---- test_metrics.json
    tm = {
        "run_id": out_dir.name,
        "method": f"{method}_sparse",
        "domain": "image",
        "git_commit": commit,
        "git_dirty": dirty,
        "reference_run_id": cnn_run_id,
        "reference_definition": "dense = 360 视角 1° 步进 FBP（同代码、同 filter、同填充）",
        "best_epoch": None,
        "best_val_rmse_valid_phys": None,
        "best_note": "重建基线无训练过程，不涉及 best 权重",
        "invariant_warn": {"n_warn": 0, "n_samples": res["overall"]["n_samples"],
                           "ratio": 0.0,
                           "note": "图像域关系与投影域相反：恒有 rmse_full <= rmse_valid"
                                   "（iradon(circle=True) 把圆外两图都置零，圆外误差为 0），"
                                   "且 rmse_full == rmse_valid*sqrt(n_valid/n_full)；"
                                   "校验在 image_metrics 内直接 raise，不降级为告警"},
        "overall": res["overall"],
        "per_energy": res["per_energy"],
        "per_sample": {f"{r['energy_MeV']}@{r['target_angle_deg']:g}": {
            "target_phase": r["target_phase"],
            "input_phase": r["input_phase"],
            "rmse_valid": r["rmse_valid"], "rmse_full": r["rmse_full"],
            "psnr": r["psnr"], "ssim": r["ssim"]} for r in res["per_sample"]},
        "cross_domain_warning": "domain=image：rmse_valid 为图像域，禁止与投影域 "
                                "rmse_valid 同表比较；coverage/valid_coverage 在图像域不适用，填 NaN",
    }
    p_tm = out_dir / "test_metrics.json"
    p_tm.write_text(json.dumps(tm, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    return {"config": p_cfg, "train_log": p_log,
            "test_per_sample": p_ps, "test_metrics": p_tm}


# ------------------------------------------------------------------ main
def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="图像域重建基线（FBP / SART）")
    ap.add_argument("--cnn-run-id", default=DEFAULT_CNN_RUN_ID)
    ap.add_argument("--data-root", default="3_3",
                    help="scan 根目录，只用于定位 empty_proj（grid 的 file 列已含 3_3/ 前缀）")
    ap.add_argument("--methods", default="fbp,sart", help="逗号分隔，可选 fbp / sart")
    ap.add_argument("--filter-name", default="hamming")
    ap.add_argument("--sart-iter", type=int, default=20)
    ap.add_argument("--sart-relax", type=float, default=0.15)
    ap.add_argument("--reuse", action="store_true", help="复用 recon/ 下已有 npy")
    args = ap.parse_args(argv)

    methods = [m.strip() for m in args.methods.split(",") if m.strip()]
    bad = [m for m in methods if m not in ("fbp", "sart")]
    if bad:
        print(f"[FAIL] 未知 method: {bad}", file=sys.stderr)
        return 2

    try:
        cfg_p = RUNS_DIR / args.cnn_run_id / "config.json"
        if not cfg_p.is_file():
            raise ReconSpecError(f"缺少 {cfg_p}")
        cfg = json.loads(cfg_p.read_text(encoding="utf-8"))
        norm = cfg.get("norm") or {}
        if norm.get("zero_log_fill") is None:
            raise ReconSpecError("config.json 的 norm.zero_log_fill 缺失，禁止使用默认值")
        fill = float(norm["zero_log_fill"])

        data_root = ROOT / args.data_root
        results = []
        for method in methods:
            t0 = time.time()
            print("=" * 96)
            print(f"[{method}] 开始（zero_log_fill={fill}  filter={args.filter_name}）", flush=True)
            res = run_method(method, args.cnn_run_id, data_root, fill, args.filter_name,
                             args.sart_iter, args.sart_relax, args.reuse)
            paths = write_outputs(res, args.cnn_run_id)
            res["_paths"] = paths
            res["_elapsed"] = time.time() - t0
            results.append(res)
    except ReconSpecError as exc:
        print(f"[FAIL] {exc}", file=sys.stderr)
        return 2
    except Exception as exc:                        # noqa: BLE001
        import traceback
        traceback.print_exc()
        print(f"[FAIL] 异常: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 3

    for res in results:
        method = res["method"]
        print("=" * 96)
        print(f"method = {method}_sparse   done in {res['_elapsed'] / 60:.1f} min   "
              f"domain = image")
        print("-" * 96)
        print(f"{'energy':>8} {'n':>4} {'rmse_valid':>12} {'rmse_full':>12} "
              f"{'psnr':>9} {'psnr_common':>12} {'ssim':>9}")
        for e in ENERGIES:
            v = res["per_energy"][e]
            print(f"{e:>8} {v['n_samples']:>4} {v['rmse_valid']:>12.6f} "
                  f"{v['rmse_full']:>12.6f} {v['psnr']:>9.4f} "
                  f"{v['psnr_common']:>12.4f} {v['ssim']:>9.6f}")
        o = res["overall"]
        print(f"{'overall':>8} {o['n_samples']:>4} {o['rmse_valid']:>12.6f} "
              f"{o['rmse_full']:>12.6f} {o['psnr']:>9.4f} "
              f"{o['psnr_common']:>12.4f} {o['ssim']:>9.6f}")
        print("-" * 96)
        low = [e for e in ENERGIES if res["per_energy"][e]["psnr"] < 10.0]
        print(f"PSNR < 10 dB 的档：{low if low else '无'}")
        for k, p in res["_paths"].items():
            print(f"已写 {k:>16}: {p.relative_to(ROOT)}  ({p.stat().st_size:,} B)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
