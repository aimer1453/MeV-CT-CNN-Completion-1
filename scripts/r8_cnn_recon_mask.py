#!/usr/bin/env python
# -*- coding: utf-8 -*-
r"""
scripts/r8_cnn_recon_mask.py —— R8：CNN 补全重建 + mask 后处理 + 前后对比指标

任务（论文 §5.7 唯一缺口）
------------------------
1. 用 `best.pt` 对**全部 180 个缺失视角**做 CNN 补全，补齐 360 视角；
2. 用 **smooth_empty(2.0) + SIRT(20 iter) + fw(5) + row16** 重建（与
   `3_3_geant4/quick_reconstruct.py` 逐项同口径）；
3. 把重建图写成符合 `mask_reconstruction.py` 命名规范的 PNG，**驱动原版脚本**
   跑完整流程（掩膜 → 膨胀 → 去环 → 区域增强）；
4. 算增强前/后的 Ring_Index / CNR / Uniformity / SNR，四档全做；
5. 结果存 `runs/<run_id>_cnn_recon_mask/test_metrics.json`。

口径要点（易踩坑，务必读）
--------------------------
**A. 两个 -log 域的换算（本脚本最关键的一步）**

CNN 在 **raw empty** 域训练/预测：`data/dataset.py::log_ratio` 用
`-log((I+1e-10)/(E_raw+1e-10))`；而 `quick_reconstruct.py` 的
`compute_sinogram_smooth_empty` 用 `-log(I/max(E_smoothed,1e-10))`，
`E_smoothed = smooth_empty_proj(E_raw, sigma=2.0)`（沿探测器 Z 逐列高斯平滑）。
**两者相差一个逐列因子，不能直接混用。** 换算（精确）::

    L_smooth = L_raw + log(max(E_smoothed[row], 1e-10) / (E_raw[row] + 1e-10))

- 实测视角：用 `compute_sinogram_smooth_empty` 的原式从原始投影算；
- 预测视角：CNN 输出在 raw 域，加上同一个逐列 delta 换算到 smooth 域。

**B. 补全正弦图的构造（sinogram 域集成，非图像平均）**

两个输入相位各补全一次：A→实测A+预测B、B→实测B+预测A。180 个实测视角在两者中
逐元素相同；180 个预测视角是不同的模型输入。**本脚本对两个补全正弦图取平均**，
再做一次环校正 + SIRT —— 不引入任意相位选择，也用满全部信息。
图像域平均（先各自重建再平均）会额外引入平滑，**没有采用**。

**C. 背景 ROI 必须改用灰色带（原始定义在后处理图上退化）**

`quick_reconstruct.py::select_background_roi` 用四角 50×50 作背景，但
`mask_reconstruction.py::apply_mask_to_array` 把掩膜黑区（占 80.5%，含 FOV
圆外全部像素）**逐像素置零** → 增强后四角全零 → Uniformity=0、CNR/SNR 爆到
~9.8e7（1.5 MeV 实测）。**这是废值，不是"变好了"。**
本脚本背景 ROI 改用**灰色带**（`|mask-0.5|<0.15`，包壳/边缘材料）：被掩膜保留、
前后都非零、物理意义清晰。原始四角口径数字一并记在 `roi_sanity_check` 供审计，
**不进主表**。

**D. 指标是显示域（0~1），不是物理域**

整条 mask 后处理链（p2/p98 归一化 → 三段量化掩膜 → 去环 → 置零/平滑 → 线性增强
到目标均值 0.45/0.95）都在 8 bit 显示域运行。before/after 是显示域对比，
**不能与 quick_reconstruct.py 在物理域 .npy 上算的指标混用**。

**E. 增强目标即指标 → 部分指标存在循环性（必须写明）**

`enhance_by_mask` 把灰色带强制映射到 mean=0.45/std=0.07，增强后 Uniformity
≈ 0.156 是**构造出来的**，不是独立验证；CNR/SNR 提升同理。
**Ring_Index 反而变差**（1.5 MeV 实测 1.38→1.72），因为材料边界被锐化。
论文引用必须带上这条 caveat。

用法
----
  conda run -n ct_full python scripts/r8_cnn_recon_mask.py
  conda run -n ct_full python scripts/r8_cnn_recon_mask.py --limit-energy 1.5
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
for p in (ROOT, ROOT / "scripts"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import baseline_recon as br                              # noqa: E402
import cnn_recon                                          # noqa: E402  复用 predict_views
from data.dataset import LOG_EPS                          # noqa: E402
from models.unet import UNet                              # noqa: E402

QR_PATH = ROOT / "3_3_geant4" / "quick_reconstruct.py"
TMP_ROOT = ROOT / "tmp_r8_postprocessing"
DEFAULT_CNN_RUN_ID = "20261022_2352_cnn_bs8_lr1e-4"
RECON_BASENAME = "recon_smooth_empty_sigma2.0_sirt_ringfw_l5_row16"

SIGMA = 2.0
SIRT_ITER = 20
FW_LEVEL = 5
ROW = br.RECON_ROW            # 16

NAN = float("nan")
EN = ("1.5", "3", "4.5", "6")


# ------------------------------------------------------------------ 工具
def _load_quick_reconstruct():
    """按路径加载 quick_reconstruct.py，复用其 smooth_empty / 指标实现。

    该模块顶层 import tomopy 失败会 sys.exit(1)，故只允许在 ct_full 下运行。
    """
    spec = importlib.util.spec_from_file_location("quick_reconstruct", QR_PATH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def resolve_mask_script() -> Path:
    """找含 correct_ring_by_material 的 mask_reconstruction.py（完整流程版）。"""
    for c in sorted(ROOT.glob("3_3/energy_scan_*/mask_reconstruction.py")):
        txt = c.read_text(encoding="utf-8", errors="ignore")
        if "def correct_ring_by_material" in txt:
            return c
    raise br.ReconSpecError("找不到含 correct_ring_by_material 的 mask_reconstruction.py")


def git_info() -> tuple[str, bool]:
    def _run(*args: str) -> str:
        try:
            return subprocess.run(("git", *args), cwd=ROOT, capture_output=True,
                                  text=True, check=False).stdout.strip()
        except OSError:
            return ""

    return _run("rev-parse", "HEAD") or "UNKNOWN", bool(_run("status", "--porcelain"))


# ------------------------------------------------------------------ 指标
def _gray_band(mask: np.ndarray) -> np.ndarray:
    return np.abs(mask - 0.5) < 0.15


def _corners_orig() -> np.ndarray:
    m = np.zeros((1200, 1200), dtype=bool)
    m[:50, :50] = True
    m[:50, -50:] = True
    m[-50:, :50] = True
    m[-50:, -50:] = True
    return m


def _fuel_roi() -> np.ndarray:
    m = np.zeros((1200, 1200), dtype=bool)
    m[570:630, 570:630] = True          # 中心 60x60，与 quick_reconstruct.compute_cnr 同
    return m


def ring_artifact_index(image: np.ndarray) -> float:
    """与 quick_reconstruct.py::ring_artifact_index 逐字同式。"""
    h, w = image.shape
    y, x = np.ogrid[:h, :w]
    r = np.sqrt((x - w / 2) ** 2 + (y - h / 2) ** 2).astype(int)
    max_r = min(h, w) // 2
    means = [image[r == i].mean() for i in range(1, max_r) if (r == i).sum() > 0]
    if len(means) < 2:
        return 0.0
    rd = np.diff(means)
    return float(rd.std() / (np.abs(rd).mean() + 1e-8))


def image_metrics(img: np.ndarray, mask: np.ndarray) -> dict:
    """Uniformity / CNR / SNR / Ring_Index（显示域 0~1）。

    灰色带 ROI 进主表；四角原始口径另存 sanity check（会退化，仅作审计）。
    """
    fu = _fuel_roi()
    out: dict = {"Ring_Index": ring_artifact_index(img)}
    for name, sel in (("gray_band", _gray_band(mask)), ("corners_orig", _corners_orig())):
        if not sel.any():
            out[name] = {"n_px": 0}
            continue
        v = img[sel]
        fv = img[fu]
        out[name] = {
            "n_px": int(sel.sum()),
            "bg_mean": float(v.mean()),
            "bg_std": float(v.std()),
            "fuel_mean": float(fv.mean()),
            "Uniformity": float(v.std() / (v.mean() + 1e-8)),
            "CNR": float(abs(fv.mean() - v.mean()) / (v.std() + 1e-8)),
            "SNR": float(fv.mean() / (v.std() + 1e-8)),
        }
    return out


# ------------------------------------------------------------------ 数据
def load_model(cnn_run_id: str, batch: int) -> tuple:
    import torch

    cfg_p = br.RUNS_DIR / cnn_run_id / "config.json"
    ckpt_p = br.RUNS_DIR / cnn_run_id / "best.pt"
    for p in (cfg_p, ckpt_p):
        if not p.is_file():
            raise br.ReconSpecError(f"缺少 {p}")
    cfg = json.loads(cfg_p.read_text(encoding="utf-8"))
    norm = cfg.get("norm") or {}
    for k in ("norm_mean", "norm_std", "zero_log_fill"):
        if norm.get(k) is None:
            raise br.ReconSpecError(f"config.json 的 norm.{k} 缺失，禁止用默认值")
    mean, std, fill = (float(norm["norm_mean"]), float(norm["norm_std"]),
                       float(norm["zero_log_fill"]))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = False
    model = UNet().to(device)
    ckpt = torch.load(ckpt_p, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"])
    info = {"best_epoch": int(ckpt["epoch"]),
            "best_val_rmse_valid_phys": float(ckpt["val_rmse_valid_phys"]),
            "device": str(device), "batch": int(batch),
            "norm_mean": mean, "norm_std": std, "zero_log_fill": fill}
    return model, info


# ------------------------------------------------------------------ 单档
def run_energy(energy: str, model, info: dict, qr, mask_script: Path) -> dict:
    import torch
    from torch.utils.data import DataLoader

    with open(br.SPARSE_TARGETS, newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        header = list(reader.fieldnames or [])
        rows = [r for r in reader if r["energy_MeV"] == energy]
    if len(rows) != 360:
        raise br.ReconSpecError(f"[{energy} MeV] 期望 360 行，实得 {len(rows)}")
    by_phase = {p: [r for r in rows if r["phase"] == p] for p in ("A", "B")}
    for p in ("A", "B"):
        if len(by_phase[p]) != 180:
            raise br.ReconSpecError(
                f"[{energy} MeV] 相位 {p} 期望 180 行，实得 {len(by_phase[p])}")

    ang_rows = br.load_angle_table(energy, ROOT / "3_3")
    scan_dirs = {r["scan_dir"] for r in ang_rows}
    if len(scan_dirs) != 1:
        raise br.ReconSpecError(f"[{energy} MeV] angles.csv 多个 scan_dir: {scan_dirs}")
    empty = br.load_empty(ROOT / "3_3", scan_dirs.pop())
    if empty.shape != (32, br.OUTPUT_SIZE):
        raise br.ReconSpecError(f"[{energy} MeV] empty 形状 {empty.shape}")
    theta = np.array([float(r["angle_deg"]) for r in ang_rows], dtype=np.float64)
    idx_of_angle = {int(a): i for i, a in enumerate(np.round(theta).astype(np.int64))}

    # ---- 实测正弦图：smooth 域（quick_reconstruct 原式）
    e_sm_row = np.maximum(qr.smooth_empty_proj(empty, SIGMA)[ROW, :], 1e-10)
    sino_smooth_meas = np.empty((360, br.OUTPUT_SIZE), dtype=np.float64)
    for i, r in enumerate(ang_rows):
        proj = np.load(ROOT / r["file"]).astype(np.float64)
        if proj.shape != (32, br.OUTPUT_SIZE):
            raise br.ReconSpecError(f"[{energy} MeV] 投影形状 {proj.shape}: {r['file']}")
        I = np.maximum(proj[ROW, :], 1e-10)
        sino_smooth_meas[i] = -np.log(I / e_sm_row)
    # raw 域 → smooth 域的逐列换算因子
    delta = np.log(e_sm_row) - np.log(empty[ROW, :].astype(np.float64) + LOG_EPS)

    # ---- CNN 补全（两个输入相位），结果在 sinogram 域平均
    device = torch.device(info["device"])
    tmp_csv_dir = ROOT / ".workbuddy" / "tmp" / "r8_cnn_recon_mask"
    tmp_csv_dir.mkdir(parents=True, exist_ok=True)
    acc = np.zeros((360, br.OUTPUT_SIZE), dtype=np.float64)
    n_pred = np.zeros(360, dtype=np.int64)
    phase_detail: dict[str, dict] = {}

    for in_phase in ("A", "B"):
        missing = br.opposite_phase(in_phase)
        t0 = time.time()
        csv_p = tmp_csv_dir / f"{energy}_{missing}.csv"
        with open(csv_p, "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=header)
            w.writeheader()
            for r in by_phase[missing]:
                rr = dict(r)
                rr["split"] = "test"
                w.writerow(rr)

        pred, mask = cnn_recon.predict_views(
            model, csv_p, info["norm_mean"], info["norm_std"],
            info["zero_log_fill"], info["batch"], device,
            expected_files=[r["target_file"] for r in by_phase[missing]])
        if pred.shape[0] != len(by_phase[missing]):
            raise br.ReconSpecError(
                f"[{energy} MeV] 预测 {pred.shape[0]} 个视角，"
                f"期望 {len(by_phase[missing])}")

        sino_phase = sino_smooth_meas.copy()
        for i, r in enumerate(by_phase[missing]):
            a = int(float(r["target_angle_deg"]))
            # CNN 输出在 raw 域 → 加 delta 换算到 smooth 域
            sino_phase[idx_of_angle[a]] = pred[i, 0, ROW, :].astype(np.float64) + delta
        acc += sino_phase
        n_pred += 1
        phase_detail[in_phase] = {
            "input_phase": in_phase, "predicted_phase": missing,
            "n_predicted_views": int(len(by_phase[missing])),
            "seconds": round(time.time() - t0, 2),
        }

    sino_comp = acc / n_pred[:, None]          # 实测视角两份相同 → 平均后不变
    if not np.all(np.isfinite(sino_comp)):
        raise br.ReconSpecError(f"[{energy} MeV] 补全正弦图含 NaN/inf")

    # ---- 环校正 fw(5) + SIRT(20)
    # 注意：qr.ring_correction 内部会自己加 np.newaxis 变 3D，这里必须传 2D
    sino_fw = qr.ring_correction(sino_comp.astype(np.float32),
                                 method="fw", level=FW_LEVEL)
    recon = qr.reconstruct_sirt(sino_fw, theta, num_iter=SIRT_ITER)
    recon = np.maximum(recon, 0).astype(np.float64)
    if recon.shape != (br.OUTPUT_SIZE, br.OUTPUT_SIZE):
        raise br.ReconSpecError(f"[{energy} MeV] 重建形状 {recon.shape}")

    # ---- 写成 mask_reconstruction.py 需要的 PNG，驱动原版脚本
    work = TMP_ROOT / energy
    if work.exists():
        shutil.rmtree(work)
    (work / "results" / "postprocessing").mkdir(parents=True, exist_ok=True)
    (work / "results" / "po_result").mkdir(parents=True, exist_ok=True)
    png_path = work / "results" / "postprocessing" / f"{RECON_BASENAME}.png"
    img_norm = qr.normalize_image(recon, 2, 98)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.imsave(png_path, img_norm, cmap="gray", vmin=0, vmax=1)
    np.save(work / "results" / "postprocessing" / f"{RECON_BASENAME}.npy",
            recon.astype(np.float32))

    # 复制原版脚本并只改三个 I/O 路径（函数逻辑一律不动）
    # 注意：注入的路径必须用 POSIX 风格（正斜杠），否则 Windows 反斜杠会被
    # Python 字符串转义（如 "\1.5" → "\x01.5"），导致 WinError 123。
    script = work / "mask_reconstruction.py"
    txt = mask_script.read_text(encoding="utf-8")
    png_posix = png_path.as_posix()
    in_posix = (work / "results" / "postprocessing").as_posix()
    out_posix = (work / "results" / "po_result").as_posix()
    old_paths = [
        ('mask_source_image = "./results/postprocessing/'
         'recon_smooth_empty_sigma2.0_sirt_ringfw_l5_row16.png"',
         f'mask_source_image = "{png_posix}"'),
        ('input_folder = "./results/postprocessing"',
         f'input_folder = "{in_posix}"'),
        ('output_folder = "./results/po_result"',
         f'output_folder = "{out_posix}"'),
    ]
    for old, new in old_paths:
        if old not in txt:
            raise br.ReconSpecError(f"mask 脚本中未找到待替换路径: {old[:60]}")
        txt = txt.replace(old, new, 1)
    script.write_text(txt, encoding="utf-8")

    t0 = time.time()
    proc = subprocess.run(
        [sys.executable, str(script)], cwd=str(work),
        capture_output=True, text=True, check=False)
    if proc.returncode != 0:
        raise br.ReconSpecError(
            f"[{energy} MeV] mask 后处理脚本退出码 {proc.returncode}\n"
            f"stdout 尾部:\n{proc.stdout[-2000:]}\nstderr 尾部:\n{proc.stderr[-2000:]}")
    mask_seconds = round(time.time() - t0, 2)

    # ---- 读回产物，算前后指标
    from PIL import Image
    po = work / "results" / "po_result"
    mask_img = np.array(Image.open(po / "generated_mask_dilated.png").convert("L"),
                        dtype=np.float64) / 255.0

    def _load(name: str) -> np.ndarray:
        p = po / f"{RECON_BASENAME}{name}"
        if not p.is_file():
            raise br.ReconSpecError(f"[{energy} MeV] 缺少后处理产物 {p.name}")
        return np.array(Image.open(p).convert("L"), dtype=np.float64) / 255.0

    before = np.array(Image.open(png_path).convert("L"), dtype=np.float64) / 255.0
    stages = {
        "before_any_postprocess": before,
        "after_ring_and_mask": _load("_masked_denoised.png"),
        "after_enhance": _load("_enhanced_final.png"),
    }
    metrics = {k: image_metrics(v, mask_img) for k, v in stages.items()}

    return {
        "energy": energy,
        "recon": recon,
        "png": png_path,
        "work_dir": work,
        "mask_seconds": mask_seconds,
        "phase_detail": phase_detail,
        "metrics": metrics,
        "mask_label_fraction": {
            "black": float((mask_img < 0.15).mean()),
            "gray": float((np.abs(mask_img - 0.5) < 0.15).mean()),
            "white": float((mask_img >= 0.85).mean()),
        },
        "recon_stats": {
            "min": float(recon.min()), "max": float(recon.max()),
            "mean": float(recon.mean()),
            "nonzero_fraction": float((recon > 0).mean()),
        },
    }


# ------------------------------------------------------------------ 落盘
def write_outputs(results: list[dict], cnn_run_id: str, info: dict,
                  mask_script: Path) -> Path:
    out_dir = br.RUNS_DIR / f"{cnn_run_id}_cnn_recon_mask"
    out_dir.mkdir(parents=True, exist_ok=True)
    commit, dirty = git_info()

    config = {
        "run_id": out_dir.name,
        "method": "cnn_completed_mask_postprocess",
        "domain": "display_0_1",
        "cnn_run_id": cnn_run_id,
        "best_pt": f"runs/{cnn_run_id}/best.pt",
        "best_epoch": info["best_epoch"],
        "best_val_rmse_valid_phys": info["best_val_rmse_valid_phys"],
        "device": info["device"],
        "energies": list(EN),
        "completion": {
            "n_views_total": 360,
            "n_predicted_views_per_phase": 180,
            "input_phases": ["A", "B"],
            "aggregation": "mean of the two phase-complementary completed "
                           "sinograms (sinogram domain, not image domain)",
        },
        "reconstruction": {
            "sinogram": "smooth_empty(sigma=2.0) row 16, -log(I/max(E_smooth,1e-10))",
            "predicted_view_domain_conversion":
                "L_smooth = L_raw + log(max(E_smooth[16],1e-10)/(E_raw[16]+1e-10))",
            "ring_correction": {"method": "fw", "level": FW_LEVEL,
                                "impl": "tomopy.remove_stripe_fw"},
            "algorithm": "sirt", "num_iter": SIRT_ITER,
            "sinogram_order": True,
            "post": "np.maximum(recon, 0), no FOV circle masking",
            "source": "3_3_geant4/quick_reconstruct.py (functions reused)",
        },
        "mask_postprocess": {
            "script": str(mask_script.relative_to(ROOT)),
            "steps": ["3-level quantized mask (low=0.33, high=0.66)",
                      "dilate_gray_white", "correct_ring_by_material (threshold=2.0)",
                      "apply_mask_to_array (sigma=1.0)", "enhance_by_mask"],
            "note": "原版脚本只改三个 I/O 路径，函数逻辑一律未动",
            "work_dir": str(TMP_ROOT.relative_to(ROOT)),
        },
        "metrics": {
            "definitions": {
                "Uniformity": "std/mean of background ROI",
                "CNR": "|mean(fuel 60x60) - mean(bg)| / std(bg)",
                "SNR": "mean(fuel 60x60) / std(bg)",
                "Ring_Index": "std(diff)/mean(abs(diff)) of radial means, "
                              "same formula as quick_reconstruct.py",
            },
            "background_roi": "gray band (|mask-0.5|<0.15); the original "
                              "four-corner ROI degenerates to all-zero after "
                              "masking (see roi_sanity_check)",
            "domain": "display 0..1 (8-bit PNG), NOT physical",
        },
        "git_commit": commit, "git_dirty": dirty,
    }
    (out_dir / "config.json").write_text(
        json.dumps(config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    # ---- train_log.csv（R8 无训练，记一行推理日志）
    n_pred_total = sum(len(r["phase_detail"]) * 180 for r in results)
    with open(out_dir / "train_log.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["epoch", "split", "energy_MeV", "n_predicted_views", "note"])
        for r in results:
            w.writerow([info["best_epoch"], "inference", r["energy"],
                        len(r["phase_detail"]) * 180,
                        "R8 is inference-only (no training); best.pt reused from "
                        f"{cnn_run_id}"])
        w.writerow([info["best_epoch"], "inference", "all", n_pred_total,
                    "2 input phases x 180 predicted views per energy"])

    # ---- test_metrics.json
    per_energy = {}
    for r in results:
        e = r["energy"]
        m = r["metrics"]
        per_energy[e] = {
            "n_completed_sinograms": 2,
            "n_predicted_views_total": 360,
            "recon_stats": r["recon_stats"],
            "mask_label_fraction": r["mask_label_fraction"],
            "mask_postprocess_seconds": r["mask_seconds"],
            "metrics": m,
            "before_after": {
                "Uniformity": {
                    "before": m["before_any_postprocess"]["gray_band"]["Uniformity"],
                    "after": m["after_enhance"]["gray_band"]["Uniformity"],
                },
                "CNR": {
                    "before": m["before_any_postprocess"]["gray_band"]["CNR"],
                    "after": m["after_enhance"]["gray_band"]["CNR"],
                },
                "SNR": {
                    "before": m["before_any_postprocess"]["gray_band"]["SNR"],
                    "after": m["after_enhance"]["gray_band"]["SNR"],
                },
                "Ring_Index": {
                    "before": m["before_any_postprocess"]["Ring_Index"],
                    "after": m["after_enhance"]["Ring_Index"],
                },
            },
        }

    def _agg(key: str, sub: str) -> float:
        """四档平均：before_after[key] = {"before": float, "after": float}。"""
        vals = [per_energy[e]["before_after"][key][sub] for e in EN]
        return float(np.mean(vals))

    overall = {
        "n_energies": len(EN),
        "Uniformity": {"before": _agg("Uniformity", "before"),
                       "after": _agg("Uniformity", "after")},
        "CNR": {"before": _agg("CNR", "before"),
                "after": _agg("CNR", "after")},
        "SNR": {"before": _agg("SNR", "before"),
                "after": _agg("SNR", "after")},
        "Ring_Index": {"before": _agg("Ring_Index", "before"),
                       "after": _agg("Ring_Index", "after")},
    }

    tm = {
        "method": "cnn_completed_mask_postprocess",
        "domain": "display_0_1",
        "cnn_run_id": cnn_run_id,
        "best_epoch": info["best_epoch"],
        "per_energy": per_energy,
        "overall": overall,
        "roi_sanity_check": {
            "why_changed":
                "原口径四角 50x50 背景 ROI 在增强后全零（apply_mask_to_array 把掩膜"
                "黑区逐像素置零，黑区占 80.5% 且含 FOV 圆外全部像素），导致 "
                "Uniformity=0、CNR/SNR ~9.8e7 的废值。主表改用灰色带 ROI。",
            "corners_orig_after_enhance": {
                e: per_energy[e]["metrics"]["after_enhance"]["corners_orig"]
                for e in EN},
            "gray_band_definition": "|mask - 0.5| < 0.15（包壳/边缘材料）",
        },
        "circularity_caveat":
            "enhance_by_mask 把灰色带强制映射到 mean=0.45/std=0.07，因此增强后 "
            "Uniformity≈0.156、CNR/SNR 的提升是构造结果，不是独立验证；"
            "Ring_Index 反而变差（材料边界被锐化）。引用时必须附带此条。",
        "train_test_gap_note":
            "R7 已实测投影域 train/test 无差距（best.pt 仅 epoch=4 未收敛）；"
            "R2 收敛后需重测。",
        "artifacts": {
            "work_dir": str(TMP_ROOT.relative_to(ROOT)),
            "per_energy_outputs": {
                e: str((TMP_ROOT / e / "results" / "po_result").relative_to(ROOT))
                for e in EN},
        },
    }
    (out_dir / "test_metrics.json").write_text(
        json.dumps(tm, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return out_dir


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="R8：CNN 补全重建 + mask 后处理 + 前后指标")
    ap.add_argument("--cnn-run-id", default=DEFAULT_CNN_RUN_ID)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--limit-energy", default=None, help="只跑某一档，如 1.5")
    args = ap.parse_args(argv)

    t_all = time.time()
    try:
        import tomopy                                      # noqa: F401
        model, info = load_model(args.cnn_run_id, args.batch)
        qr = _load_quick_reconstruct()
        mask_script = resolve_mask_script()
        print(f"best.pt: epoch={info['best_epoch']} "
              f"val_rmse_valid_phys={info['best_val_rmse_valid_phys']:.6f} "
              f"device={info['device']} batch={info['batch']}", flush=True)
        print(f"mask 脚本（完整流程版）: {mask_script.relative_to(ROOT)}", flush=True)

        energies = (args.limit_energy,) if args.limit_energy else EN
        results: list[dict] = []
        for e in energies:
            print(f"[{e} MeV] 开始", flush=True)
            results.append(run_energy(e, model, info, qr, mask_script))
            m = results[-1]["metrics"]
            g = m["after_enhance"]["gray_band"]
            print(f"  重建 max={results[-1]['recon_stats']['max']:.6f} "
                  f"非零占比={results[-1]['recon_stats']['nonzero_fraction']:.4f} | "
                  f"增强后 Uniformity={g['Uniformity']:.4f} CNR={g['CNR']:.3f} "
                  f"SNR={g['SNR']:.3f} Ring={m['after_enhance']['Ring_Index']:.4f}",
                  flush=True)
        print(f"四档完成，用时 {time.time() - t_all:.1f}s", flush=True)

        if args.limit_energy:
            print("[提示] --limit-energy 只做单档试跑，未写 test_metrics.json")
            return 0

        out_dir = write_outputs(results, args.cnn_run_id, info, mask_script)
    except br.ReconSpecError as exc:
        print(f"[FAIL] {exc}", file=sys.stderr)
        return 2
    except Exception as exc:                                 # noqa: BLE001
        import traceback
        traceback.print_exc()
        print(f"[FAIL] 异常: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 3

    print("=" * 96)
    print("R8：CNN 补全重建 → mask 后处理，增强前/后（灰色带 ROI，显示域 0~1）")
    print("-" * 96)
    print(f"{'energy':>8} {'Uni_bef':>9} {'Uni_aft':>9} {'CNR_bef':>9} {'CNR_aft':>9} "
          f"{'SNR_bef':>9} {'SNR_aft':>9} {'Ring_bef':>10} {'Ring_aft':>10}")
    for e in EN:
        v = json.loads((out_dir / "test_metrics.json").read_text(encoding="utf-8"))
        b = v["per_energy"][e]["before_after"]
        print(f"{e:>8} {b['Uniformity']['before']:>9.4f} {b['Uniformity']['after']:>9.4f} "
              f"{b['CNR']['before']:>9.3f} {b['CNR']['after']:>9.3f} "
              f"{b['SNR']['before']:>9.3f} {b['SNR']['after']:>9.3f} "
              f"{b['Ring_Index']['before']:>10.4f} {b['Ring_Index']['after']:>10.4f}")
    o = v["overall"]
    print("-" * 96)
    print(f"{'overall':>8} {o['Uniformity']['before']:>9.4f} {o['Uniformity']['after']:>9.4f} "
          f"{o['CNR']['before']:>9.3f} {o['CNR']['after']:>9.3f} "
          f"{o['SNR']['before']:>9.3f} {o['SNR']['after']:>9.3f} "
          f"{o['Ring_Index']['before']:>10.4f} {o['Ring_Index']['after']:>10.4f}")
    for f in ("config.json", "train_log.csv", "test_metrics.json"):
        p = out_dir / f
        print(f"已写 {f:>18}: {p.relative_to(ROOT)}  ({p.stat().st_size:,} B)")
    print("=" * 96)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
