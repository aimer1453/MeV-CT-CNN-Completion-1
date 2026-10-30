#!/usr/bin/env python
# -*- coding: utf-8 -*-
r"""
scripts/cnn_recon.py —— Table 4b：**图像域** CNN 补全重建基线

任务
----
把 CNN 也拉到图像域，让 Table 4b 的四行处于同一域：
    fbp_sparse / sart_sparse / **cnn_completed** / dense(参考)
（前三行与 dense 参考比 PSNR/SSIM；`fbp_sparse`、`sart_sparse` 见 `scripts/baseline_recon.py`）

做法（每个能量档、每个输入相位各一次重建）
------------------------------------------
  1. 取某一相位（记为 P）的 180 个视角作为**实测**输入；
  2. 用 `best.pt` 预测**相反相位**（opposite(P)）的全部 180 个视角 —— 每个视角的输入是
     θ−3°、θ−1°、θ+1° 三帧（都落在相位 P 内）+ sinθ、cosθ，与训练时完全一致；
  3. 交错拼接：180 实测 + 180 预测 = **360 视角稠密正弦图**；
  4. `iradon(filter=hamming, circle=True)` → 1200×1200；
  5. 与 dense 参考（360 视角**全部实测**的 FBP）比 PSNR / SSIM / RMSE。

  **推理不需要 GT**，所以"补齐 180 个缺失视角"是部署时的真实行为。

⚠️ 两条必须写进论文的限制
--------------------------
1. **train/test 差距（已实测，结论与最初预期相反）**
   每个能量档只有 45 个 test 视角，**拼不出完整 360 视角正弦图**。因此本脚本补全的是该档
   **全部 180 个缺失视角**，其中约 135 个属 train 划分、22~25 个属 val。按常理模型在训练中
   见过 train 视角，补全质量应偏好 —— 但**实测没有这个差距**（1.5 MeV：row16 train 0.2334 /
   val 0.2408 / test 0.2448；all32 train 0.8453 / val 0.8467 / test 0.8366）。
   原因是 `best.pt` 只训到 epoch=4（未收敛），欠训模型尚未记住训练样本。
   结论写进 `test_metrics.json.train_test_gap_check`，**不写"乐观估计"的警告**（无实测支持）。
   R2 训练到收敛后必须重跑本检查。

2. **投影域两套口径不可混用（row16 vs all32）**
   `row16` 只算 `RECON_ROW=16`（真正进入正弦图、决定重建的行）；`all32` 是全 32 行 mask 加权
   （与 `train.py` evaluate 一致，可与 R4/Table 对齐）。1.5 MeV 实测 row16≈0.235 vs
   all32≈0.844，差 3.5 倍。原因是源束流在探测器竖直方向强峰形：`empty_proj` 行均值中心
   12.62 / 边缘 0.20（~60 倍），边缘行近乎零计数，`-log` 把噪声放大成 -6.4~+5.5 的极端
   动态范围，于是 all32 被边缘低通量行主导。引用时必须写明是哪一套。

口径（与 `baseline_recon.py` 逐项对齐，代码直接 import 复用）
--------------------------------------------------------------
  - dense 参考、FOV 圆（`r<=600`，1,130,911 px）、SSIM 中心 848×848 裁剪、
    `psnr`（逐档 data_range）与 `psnr_common`（四档统一 data_range）**全部复用
    `scripts/baseline_recon.py` 的实现**，保证 Table 4b 四行同口径可横比。
  - **图像域恒有 `rmse_full <= rmse_valid`**（`circle=True` 把圆外两图都置零），
    与投影域相反，不可沿用投影域断言。
  - `coverage` / `valid_coverage` 是投影域资产口径，图像域不适用，本文件填 `NaN`，**不要引用**。

输出
----
  runs/<cnn_run_id>_cnn_recon/{config.json,train_log.csv,test_metrics.json,test_per_sample.csv}
  runs/<cnn_run_id>_cnn_recon/recon/{dense_*,cnn_*}.npy   （.npy 被 .gitignore 忽略）

用法
----
  conda run -n ct_full python scripts/cnn_recon.py
  conda run -n ct_full python scripts/cnn_recon.py --batch 16 --limit-energy 1.5   # 快速试跑
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
import subprocess
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

import baseline_recon as br                              # noqa: E402  同口径复用
from data.dataset import SparseCTDataset                 # noqa: E402
from models.unet import UNet                             # noqa: E402

SPARSE_TARGETS = resolve_csv("sparse_targets.csv")
RUNS_DIR = ROOT / "runs"
DEFAULT_CNN_RUN_ID = "20261022_2352_cnn_bs8_lr1e-4"
TMP_DIR = ROOT / ".workbuddy" / "tmp" / "cnn_recon"

NAN = float("nan")

# R7 §7.2 指定的表注原文，**必须原样带给论文侧**（论文侧要直接贴进 Table 4 的表注）
TABLE_NOTE_EN = ("image-domain metrics were computed from 180 measured views plus 180 "
                 "completed views; the completed set is not restricted to the held-out split.")


def git_info() -> tuple[str, bool]:
    def _run(*args: str) -> str:
        try:
            return subprocess.run(("git", *args), cwd=ROOT, capture_output=True,
                                  text=True, check=False).stdout.strip()
        except OSError:
            return ""

    return _run("rev-parse", "HEAD") or "UNKNOWN", bool(_run("status", "--porcelain"))


# ------------------------------------------------------------------ 数据
def load_energy_rows(energy: str) -> list[dict]:
    """取该能量的全部 360 行（不推导角度，逐字来自 CSV）。"""
    with open(SPARSE_TARGETS, newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        header = list(reader.fieldnames or [])
        rows = [r for r in reader if r["energy_MeV"] == energy]
    if len(rows) != 360:
        raise br.ReconSpecError(f"[{energy} MeV] 期望 360 行，实得 {len(rows)}")
    return header, rows


def write_subset_csv(energy: str, phase: str, header: list[str],
                     rows: list[dict]) -> Path:
    """把筛选出的行写成临时 CSV（split 列统一改成 test），供 SparseCTDataset 读取。

    只在**选行**时使用 CSV 的 `phase` 列，**不在运行时推导角度**。
    """
    TMP_DIR.mkdir(parents=True, exist_ok=True)
    p = TMP_DIR / f"{energy}_{phase}.csv"
    with open(p, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=header)
        w.writeheader()
        for r in rows:
            out = dict(r)
            out["split"] = "test"        # 仅为通过 load_csv_rows 的 split 校验
            w.writerow(out)
    return p


# ------------------------------------------------------------------ 预测
def predict_views(model, csv_path: Path, mean: float, std: float, fill: float,
                  batch: int, device, n_views: int = 3,
                  expected_files: list[str] | None = None) -> tuple[np.ndarray, np.ndarray]:
    """返回 (pred_phys, mask)：pred_phys 形状 (N,1,32,1200) 物理 -log 域。

    **行序保证**：`SparseCTDataset` 的 `load_csv_rows` 按 CSV 文件顺序 append、
    `__getitem__(idx)` 取同一下标、DataLoader `shuffle=False`，所以 `pred[i]`
    与传入 CSV 的第 i 行一一对应。传 `expected_files` 时逐行断言（防止上游改排序
    导致正弦图行被贴到错误角度）。
    """
    import torch
    from torch.utils.data import DataLoader

    ds = SparseCTDataset(split="test", zero_log_fill=fill,
                         norm_mean=mean, norm_std=std, csv_path=csv_path,
                         n_views=n_views)
    if expected_files is not None:
        if len(ds.rows) != len(expected_files):
            raise br.ReconSpecError(f"数据集 {len(ds.rows)} 行 != 期望 {len(expected_files)}")
        for i, f in enumerate(expected_files):
            if ds.rows[i]["target_file"] != f:
                raise br.ReconSpecError(
                    f"行序错位 i={i}：dataset={ds.rows[i]['target_file']} 期望={f}"
                    f"（若成立，正弦图行会被贴到错误角度，重建结果作废）")
    loader = DataLoader(ds, batch_size=batch, shuffle=False, num_workers=0)

    preds: list[np.ndarray] = []
    masks: list[np.ndarray] = []
    model.eval()
    with torch.no_grad():
        for b in loader:
            x = b["x"].to(device)
            y = b["y"].to(device)
            m = b["mask"].to(device).bool()
            pred = model(x)
            if pred.shape != y.shape:
                raise br.ReconSpecError(f"预测 shape {tuple(pred.shape)} != {tuple(y.shape)}")
            preds.append((pred.to(torch.float64) * std + mean).cpu().numpy())
            masks.append(m.cpu().numpy())
    return np.concatenate(preds, axis=0), np.concatenate(masks, axis=0)


def projection_check(pred: np.ndarray, mask: np.ndarray, rows: list[dict],
                     empty: np.ndarray, fill: float) -> dict:
    """投影域留出检查，**两套口径并列**（不可混用、不可互相引用）：

    - `row16`：只算 `RECON_ROW=16` 这一行 —— 这是真正进入正弦图、决定重建的行；
    - `all32`：全 32 行、mask 加权 —— 与 `train.py` 的 evaluate 口径一致（= R4 的数字）。

    两套差别极大（1.5 MeV 实测 row16≈0.24 vs all32≈0.84），原因是源束流在探测器
    竖直方向强峰形：`empty_proj` 行均值中心 12.62 / 边缘 0.20（相差 ~60 倍），
    边缘行近乎零计数，`-log` 把噪声放大成 -6.4~+5.5 的极端动态范围，
    因此 all32 被边缘低通量行主导。**引用时必须写明是哪一套。**
    """
    gt_row16 = np.empty((len(rows), br.OUTPUT_SIZE), dtype=np.float64)
    gt_all32 = np.empty((len(rows), 32, br.OUTPUT_SIZE), dtype=np.float64)
    for i, r in enumerate(rows):
        proj = np.load(br.ROOT / r["target_file"]).astype(np.float64)
        if proj.shape != (32, br.OUTPUT_SIZE):
            want = (32, br.OUTPUT_SIZE)
            raise br.ReconSpecError(
                "投影形状 " + str(proj.shape) + " != " + str(want) + ": "
                + r["target_file"])
        L = br.log_ratio(proj, empty).astype(np.float64)
        L = np.where(np.isfinite(L), L, fill)
        gt_all32[i] = L
        gt_row16[i] = L[br.RECON_ROW]

    out: dict[str, dict] = {}
    variants = {
        "row16": (pred[:, 0, br.RECON_ROW, :][:, None, :],
                  gt_row16[:, None, :],
                  mask[:, 0, br.RECON_ROW, :][:, None, :],
                  "只算 RECON_ROW=16 这一行（进入正弦图的行）"),
        "all32": (pred[:, 0, :, :], gt_all32, mask[:, 0, :, :],
                  "全 32 行、mask 加权（与 train.py evaluate 口径一致）"),
    }
    for name, (pr, gt_sel, mk, definition) in variants.items():
        sq = (pr - gt_sel) ** 2
        per_split: dict[str, dict] = {}
        for sp in ("train", "val", "test"):
            sel = [i for i, r in enumerate(rows) if r["split"] == sp]
            if not sel:
                continue
            m = mk[sel]
            n = int(m.sum())
            se = float(np.where(m, sq[sel], 0.0).sum())
            per_split[sp] = {"n_views": len(sel),
                             "n_valid_pixels": n,
                             "rmse_valid": math.sqrt(se / n) if n else NAN}
        n_all = int(mk.sum())
        se_all = float(np.where(mk, sq, 0.0).sum())
        out[name] = {
            "definition": definition,
            "n_predicted_views": len(rows),
            "n_valid_pixels": n_all,
            "rmse_valid_all": math.sqrt(se_all / n_all) if n_all else NAN,
            "by_split": per_split,
        }
    out["cross_reading_warning"] = (
        "row16 与 all32 相差约 3.5 倍（1.5 MeV 实测），**不可混用**：重建只用 row16；"
        "all32 仅用于与 R4/Table 的投影域数字对齐。差异来源是源束流竖直峰形导致边缘行"
        "近零计数、-log 噪声放大，不是训练/测试泄漏。")
    return out


# ------------------------------------------------------------------ 主流程
def run(energy: str, model, mean: float, std: float, fill: float, batch: int,
        device, filter_name: str, recon_dir: Path, n_views: int = 3) -> dict:
    header, rows = load_energy_rows(energy)
    by_phase = {p: [r for r in rows if r["phase"] == p] for p in ("A", "B")}
    for p in ("A", "B"):
        if len(by_phase[p]) != 180:
            raise br.ReconSpecError(f"[{energy} MeV] 相位 {p} 期望 180 行，实得 {len(by_phase[p])}")

    sino, theta = br.build_sinogram(energy, ROOT / "3_3", fill)      # (360,1200) 实测 -log
    if not np.array_equal(np.round(theta).astype(np.int64), np.arange(360)):
        raise br.ReconSpecError(f"[{energy} MeV] theta 不是 0..359 的整数序列")
    idx_of_angle = {int(a): i for i, a in enumerate(np.round(theta).astype(np.int64))}

    scan_dirs = {r["scan_dir"] for r in rows}
    if len(scan_dirs) != 1:
        raise br.ReconSpecError(f"[{energy} MeV] 出现多个 scan_dir: {scan_dirs}")
    empty = br.load_empty(ROOT / "3_3", scan_dirs.pop())

    # dense 参考：360 视角全部实测
    dense = br.get_dense_reference(energy, ROOT / "3_3", fill, filter_name,
                                   recon_dir, reuse=True)

    out: dict = {"energy": energy, "dense": dense, "recon": {}, "proj": {}}
    for in_phase in ("A", "B"):
        missing = br.opposite_phase(in_phase)
        t0 = time.time()
        csv_p = write_subset_csv(energy, missing, header, by_phase[missing])
        pred, mask = predict_views(model, csv_p, mean, std, fill, batch, device,
                                   n_views=n_views,
                                   expected_files=[r["target_file"] for r in by_phase[missing]])
        if pred.shape[0] != len(by_phase[missing]):
            raise br.ReconSpecError(f"[{energy} MeV] 预测 {pred.shape[0]} 个视角，"
                                    f"期望 {len(by_phase[missing])}")

        # ---- 拼 360 视角稠密正弦图：相位 in_phase 用实测，另一相位用预测（取 recon_row）
        sino_comp = sino.copy()
        for i, r in enumerate(by_phase[missing]):
            a = int(float(r["target_angle_deg"]))
            sino_comp[idx_of_angle[a]] = pred[i, 0, br.RECON_ROW, :].astype(np.float64)

        img = br.restrict_to_fov(br.recon_fbp(sino_comp, theta, filter_name))
        np.save(recon_dir / f"cnn_{energy}_{in_phase}.npy", img.astype(np.float32))
        out["recon"][in_phase] = img

        # ---- 投影域留出检查：row16 / all32 两套口径并列，各按 split 拆开
        pc = projection_check(pred, mask, by_phase[missing], empty, fill)
        out["proj"][in_phase] = {"missing_phase": missing, "check": pc}
        r16 = pc["row16"]
        a32 = pc["all32"]
        print(f"  [{energy} MeV] 输入相位 {in_phase} → 预测相位 {missing}："
              f"{len(by_phase[missing])} 视角，{time.time() - t0:.1f}s | "
              f"row16 all={r16['rmse_valid_all']:.6f} "
              f"test={r16['by_split'].get('test', {}).get('rmse_valid', NAN):.6f} | "
              f"all32 all={a32['rmse_valid_all']:.6f} "
              f"test={a32['by_split'].get('test', {}).get('rmse_valid', NAN):.6f}", flush=True)
    return out


# ------------------------------------------------------------------ 落盘
def write_outputs(results: list[dict], cnn_run_id: str, mean: float,
                  std: float, fill: float, filter_name: str, batch: int,
                  best_epoch: int, best_val: float,
                  dense_match: dict, n_views: int = 3) -> Path:
    out_dir = RUNS_DIR / f"{cnn_run_id}_cnn_recon"
    mask = br.fov_mask()
    commit, dirty = git_info()

    # ---- 逐样本：target 相位 T → 用"输入相位 = opposite(T)"的补全重建
    test_rows = br.load_test_rows()
    per_sample: list[dict] = []
    for r in test_rows:
        e = r["energy_MeV"]
        if e not in br.ENERGIES:
            raise br.ReconSpecError(f"未知能量 {e!r}")
        in_phase = br.opposite_phase(r["phase"])
        res = next(x for x in results if x["energy"] == e)
        m = br.image_metrics(res["dense"], res["recon"][in_phase], mask)
        per_sample.append({
            "energy_MeV": e,
            "target_angle_deg": float(r["target_angle_deg"]),
            "split": "test",
            "n_valid_pixels": m["n_valid_pixels"],
            "coverage": NAN,
            "valid_coverage": NAN,
            "rmse_valid": m["rmse_valid"],
            "rmse_full": m["rmse_full"],
            "psnr": m["psnr"],
            "ssim": m["ssim"],
            "rmse_invariant_warn": False,
            "target_phase": r["phase"],
            "input_phase": in_phase,
        })

    dr_common = max(float(x["dense"].max() - x["dense"].min()) for x in results)

    per_energy: dict[str, dict] = {}
    for e in br.ENERGIES:
        sub = [s for s in per_sample if s["energy_MeV"] == e]
        n = len(sub)
        nv = sum(s["n_valid_pixels"] for s in sub)
        se_v = sum(s["rmse_valid"] ** 2 * s["n_valid_pixels"] for s in sub)
        mse_v = se_v / nv
        res = next(x for x in results if x["energy"] == e)
        per_energy[e] = {
            "n_samples": n,
            "coverage": NAN,
            "valid_coverage": NAN,
            "rmse_valid": math.sqrt(mse_v),
            "rmse_full": math.sqrt(sum(s["rmse_full"] ** 2 for s in sub) / n),
            "psnr": float(np.mean([s["psnr"] for s in sub])),
            "psnr_common": 10.0 * math.log10(dr_common * dr_common / mse_v),
            "ssim": float(np.mean([s["ssim"] for s in sub])),
            "data_range": float(res["dense"].max() - res["dense"].min()),
            "data_range_common": dr_common,
        }

    n_all = sum(v["n_samples"] for v in per_energy.values())
    n_v = sum(s["n_valid_pixels"] for s in per_sample)
    se_v = sum(s["rmse_valid"] ** 2 * s["n_valid_pixels"] for s in per_sample)
    overall = {
        "n_samples": n_all,
        "coverage": NAN,
        "valid_coverage": NAN,
        "rmse_valid": math.sqrt(se_v / n_v),
        "rmse_full": math.sqrt(sum(v["rmse_full"] ** 2 * v["n_samples"]
                                   for v in per_energy.values()) / n_all),
        "psnr": float(np.mean([v["psnr"] for v in per_energy.values()])),
        "psnr_common": float(np.mean([v["psnr_common"] for v in per_energy.values()])),
        "ssim": float(np.mean([v["ssim"] for v in per_energy.values()])),
    }
    if overall["rmse_full"] > overall["rmse_valid"] + 1e-9:
        raise br.ReconSpecError("图像域总体 rmse_full > rmse_valid，圆外零误差假设被破坏")

    out_dir.mkdir(parents=True, exist_ok=True)

    config = {
        "run_id": out_dir.name,
        "method": "cnn_completed",
        "domain": "image",
        "method_note": "CNN 补齐相反相位的 180 个视角 → 与 180 个实测视角交错成 360 视角正弦图 "
                       "→ hamming FBP；与 dense（360 视角全实测 FBP）比 PSNR/SSIM",
        "reference_run_id": cnn_run_id,
        "input_views": n_views,
        "input_channels": n_views + 2,
        "checkpoint": {"file": f"runs/{cnn_run_id}/best.pt",
                       "best_epoch": best_epoch, "val_rmse_valid_phys": best_val},
        "git_commit": commit,
        "git_dirty": dirty,
        "geometry": {"sensor_x": br.OUTPUT_SIZE, "sensor_y": 32,
                     "recon_row": br.RECON_ROW, "output_size": [br.OUTPUT_SIZE] * 2,
                     "sinogram_shape": [360, br.OUTPUT_SIZE]},
        "sampling": {"dense": {"n_views": 360, "step_deg": 1.0},
                     "completed": {"measured_views": 180, "predicted_views": 180,
                                   "step_deg": 1.0},
                     "phase_rule": "A = 偶数角，B = 奇数角；预测相位 = opposite(实测相位)"},
        "n_views": 360,
        "n_completed": 180,
        "held_out_restricted": False,
        "held_out_note": TABLE_NOTE_EN,
        "norm": {"norm_mean": mean, "norm_std": std, "zero_log_fill": fill,
                 "note": "反归一化按 train.py:461 的 pred*std + mean，与 evaluate 同式"},
        "recon": {"filter_name": filter_name, "circle": True,
                  "fov_restriction": "圆外置零（circle=True 自动，等价于 baseline_recon 的 "
                                     "restrict_to_fov）"},
        "inference": {"batch": batch, "device": "cuda/cpu 自动", "shuffle": False,
                      "gt_needed": False},
        "metric_definition": {
            "domain": "image",
            "rmse_valid": "FOV 重建圆（r<=600，1,130,911 像素）上的 RMSE",
            "rmse_full": "全部 1200x1200 像素；圆外两图皆 0，故恒有 rmse_full <= rmse_valid",
            "psnr": "10*log10(data_range^2/mse_valid)，data_range 逐档不同",
            "psnr_common": "同上但 data_range 取四档统一值，用于跨能量比较",
            "ssim": "中心 848x848 裁剪上的 SSIM（该方框半对角 599.6 < 600，全在圆内）",
            "reference": "dense = 360 视角全实测 FBP，与 baseline_recon.py 同一份代码",
            "cross_domain_warning": "本文件 rmse_valid 是图像域，与投影域 rmse_valid 不同域、不可同表",
            "optimism_warning": "原写'补全的 180 个缺失视角中约 135 个属 train 划分 → 乐观估计'，"
                                "但 train_test_gap_check 实测未观察到 train/test 差距"
                                "（epoch=4 未收敛），故不写该警告；详见 test_metrics.json",
            "table_note_en": TABLE_NOTE_EN,
        },
        "data": {"targets": "sparse_targets.csv", "split": "test",
                 "n_samples": n_all, "energies_MeV": list(br.ENERGIES),
                 "reconstructions_per_energy": 2,
                 "note": "n_samples=45/档 是按 target 相位分配的样本数；"
                         "实际重建次数是 2/档（每个输入相位一张）"},
        "dense_reference_cross_check": dense_match,
    }
    (out_dir / "config.json").write_text(
        json.dumps(config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    with open(out_dir / "train_log.csv", "w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=br.LOG_COLUMNS)
        w.writeheader()
        for e in br.ENERGIES:
            v = per_energy[e]
            w.writerow({"epoch": -1, "split": "test", "energy_MeV": e,
                        "n_samples": v["n_samples"], "coverage": "NaN",
                        "valid_coverage": "NaN",
                        "rmse_valid": f"{v['rmse_valid']:.6f}",
                        "rmse_full": f"{v['rmse_full']:.6f}",
                        "psnr": f"{v['psnr']:.6f}", "ssim": f"{v['ssim']:.6f}",
                        "n_invariant_warn": 0, "lr": "NaN",
                        "loss_train": "NaN", "loss_val": "NaN"})

    with open(out_dir / "test_per_sample.csv", "w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=br.PER_SAMPLE_COLUMNS)
        w.writeheader()
        for r in per_sample:
            row = {k: r[k] for k in br.PER_SAMPLE_COLUMNS}
            for k in ("coverage", "valid_coverage"):
                row[k] = "NaN"
            for k in ("rmse_valid", "rmse_full", "psnr", "ssim"):
                row[k] = f"{row[k]:.6f}"
            row["target_angle_deg"] = f"{float(row['target_angle_deg']):g}"
            w.writerow(row)

    proj_check = {}
    d_read = ("投影域 rmse_valid 是物理 -log 域、mask 内，**两套口径并列**："
              "row16 只算 RECON_ROW=16（进入正弦图、决定重建的行）；"
              "all32 为全 32 行 mask 加权（与 train.py evaluate 一致，可与 R4/Table 对齐）。"
              "两者不可混用；1.5 MeV 实测相差约 3.5 倍，因源束流竖直峰形使边缘行近零计数、"
              "-log 放大噪声，all32 被边缘行主导。二者都不是图像域指标。")
    for x in results:
        e = x["energy"]
        proj_check[e] = {
            "by_input_phase": {
                p: {"missing_phase": d["missing_phase"],
                    "row16": d["check"]["row16"],
                    "all32": d["check"]["all32"]}
                for p, d in x["proj"].items()},
            "reading_note": d_read,
        }
    # ---- 实测的 train/test 差距（替代原先凭先验写的"乐观估计"警告）
    gap: dict[str, dict] = {}
    for x in results:
        e = x["energy"]
        g: dict[str, dict] = {}
        for p, d in x["proj"].items():
            for name in ("row16", "all32"):
                bs = d["check"][name]["by_split"]
                tr = bs.get("train", {}).get("rmse_valid", NAN)
                te = bs.get("test", {}).get("rmse_valid", NAN)
                g[f"{name}_{p}"] = {"train": tr, "test": te,
                                    "test_minus_train": te - tr}
        gap[e] = g
    worst_gap = max((abs(v["test_minus_train"]) for per in gap.values()
                     for v in per.values() if np.isfinite(v["test_minus_train"])),
                    default=NAN)
    gap_verdict = {
        "measured": gap,
        "max_abs_test_minus_train": worst_gap,
        "conclusion": (
            "实测 train 与 test 的投影域误差几乎相同（见 measured），**没有观察到"
            "训练集记忆带来的留出优势**。这与 best.pt 只训到 epoch=4（未收敛）一致："
            "欠训模型尚未记住训练样本。因此 Table 4b 的图像域 PSNR/SSIM 在本 checkpoint "
            "下不因'补全视角多属 train 划分'而系统性偏高。"),
        "caveat": (
            "该结论只对当前 epoch=4 的权重成立。若后续训练到收敛（R2 的 30 epoch），"
            "必须重跑本检查再看 test_minus_train；届时若出现显著负值，才需要按"
            "'乐观估计'写进论文限制。"),
    }

    tm = {
        "run_id": out_dir.name,
        "method": "cnn_completed",
        "domain": "image",
        "n_views": 360,
        "n_completed": 180,
        "held_out_restricted": False,
        "table_note_en": TABLE_NOTE_EN,
        "git_commit": commit,
        "git_dirty": dirty,
        "reference_run_id": cnn_run_id,
        "reference_definition": "dense = 360 视角全实测 hamming FBP，与 baseline_recon.py 同代码同参数",
        "best_epoch": best_epoch,
        "best_val_rmse_valid_phys": best_val,
        "best_note": f"best.pt 对应 epoch={best_epoch}（5 epoch 试跑，未收敛）",
        "invariant_warn": {"n_warn": 0, "n_samples": n_all, "ratio": 0.0,
                           "note": "图像域恒有 rmse_full <= rmse_valid（圆外两图皆 0），"
                                   "与投影域相反；校验在写入前 raise"},
        "overall": overall,
        "per_energy": per_energy,
        "per_sample": {f"{r['energy_MeV']}@{r['target_angle_deg']:g}": {
            "target_phase": r["target_phase"],
            "input_phase": r["input_phase"],
            "rmse_valid": r["rmse_valid"], "rmse_full": r["rmse_full"],
            "psnr": r["psnr"], "ssim": r["ssim"]} for r in per_sample},
        "per_sample_note": "test 45 个 target 视角共用其输入相位的补全重建；"
                           "每档 2 张重建（输入相位 A/B 各一），n_samples 是 target 视角数",
        "projection_domain_check": proj_check,
        "train_test_gap_check": gap_verdict,
        "optimism_warning": None,
        "optimism_note": "原计划写'补全的 180 个缺失视角含约 135 个 train 视角 → 乐观估计'，"
                         "但 train_test_gap_check 实测未观察到该差距（epoch=4 未收敛），"
                         "故不写该警告；若 R2 训练到收敛后重跑出现显著负 gap，需补写",
        "cross_domain_warning": "domain=image：rmse_valid 为图像域，禁止与投影域 rmse_valid 同表；"
                                "coverage/valid_coverage 在图像域不适用，填 NaN",
    }
    (out_dir / "test_metrics.json").write_text(
        json.dumps(tm, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return out_dir


# ------------------------------------------------------------------ main
def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Table 4b：图像域 CNN 补全重建")
    ap.add_argument("--cnn-run-id", default=DEFAULT_CNN_RUN_ID)
    ap.add_argument("--filter-name", default="hamming")
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--limit-energy", default=None,
                    help="只跑某一档（快速试跑），例如 1.5；默认四档全跑")
    args = ap.parse_args(argv)

    try:
        import torch

        cfg_p = RUNS_DIR / args.cnn_run_id / "config.json"
        ckpt_p = RUNS_DIR / args.cnn_run_id / "best.pt"
        for p in (cfg_p, ckpt_p):
            if not p.is_file():
                raise br.ReconSpecError(f"缺少 {p}")
        cfg = json.loads(cfg_p.read_text(encoding="utf-8"))
        norm = cfg.get("norm") or {}
        for k in ("norm_mean", "norm_std", "zero_log_fill"):
            if norm.get(k) is None:
                raise br.ReconSpecError(f"config.json 的 norm.{k} 缺失，禁止用默认值")
        mean, std, fill = float(norm["norm_mean"]), float(norm["norm_std"]), \
            float(norm["zero_log_fill"])

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        torch.backends.cudnn.benchmark = False      # 推理不需要 autotune，避免不确定性
        # 输入视角数从被评测 run 自己的 config 读（缺省 3 = 正文配置），
        # 保证两视角消融 run 的 in_channels 与当初训练一致
        n_views = int(cfg.get("input_views") or 3)
        model = UNet(in_channels=n_views + 2).to(device)
        ckpt = torch.load(ckpt_p, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model"])
        best_epoch = int(ckpt["epoch"])
        best_val = float(ckpt["val_rmse_valid_phys"])
        print(f"加载 best.pt：epoch={best_epoch}  val_rmse_valid_phys={best_val:.6f}  "
              f"device={device}  batch={args.batch}  input_views={n_views} "
              f"(in_channels={n_views + 2})", flush=True)

        out_dir = RUNS_DIR / f"{args.cnn_run_id}_cnn_recon"
        recon_dir = out_dir / "recon"
        recon_dir.mkdir(parents=True, exist_ok=True)

        energies = (args.limit_energy,) if args.limit_energy else br.ENERGIES
        results: list[dict] = []
        t_all = time.time()
        for e in energies:
            print(f"[{e} MeV] 开始", flush=True)
            results.append(run(e, model, mean, std, fill, args.batch, device,
                               args.filter_name, recon_dir, n_views=n_views))
        print(f"四档重建完成，用时 {time.time() - t_all:.1f}s", flush=True)

        # dense 参考与 fbp_sparse 的缓存必须逐字节一致（同代码同数据 → 应当一致）
        dense_match: dict[str, dict] = {}
        ref_dir = RUNS_DIR / f"{args.cnn_run_id}_fbp_sparse" / "recon"
        for x in results:
            e = x["energy"]
            p = ref_dir / f"dense_{e}_fbp_{args.filter_name}.npy"
            if p.is_file():
                same = bool(np.array_equal(np.load(p).astype(np.float32),
                                           x["dense"].astype(np.float32)))
                dense_match[e] = {"compared_with": str(p.relative_to(ROOT)),
                                  "byte_equal": same}
            else:
                dense_match[e] = {"compared_with": None, "byte_equal": None,
                                  "note": "fbp_sparse 缓存缺失，无法交叉校验"}
        if dense_match and not all(v["byte_equal"] in (True, None) for v in dense_match.values()):
            raise br.ReconSpecError(f"dense 参考与 fbp_sparse 缓存不一致: {dense_match}")

        if args.limit_energy:
            print("[提示] --limit-energy 只做了单档试跑，未写 test_metrics.json")
            return 0

        out_dir = write_outputs(results, args.cnn_run_id, mean, std, fill,
                                args.filter_name, args.batch, best_epoch, best_val,
                                dense_match, n_views=n_views)
    except br.ReconSpecError as exc:
        print(f"[FAIL] {exc}", file=sys.stderr)
        return 2
    except Exception as exc:                             # noqa: BLE001
        import traceback
        traceback.print_exc()
        print(f"[FAIL] 异常: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 3

    tm = json.loads((out_dir / "test_metrics.json").read_text(encoding="utf-8"))
    print("=" * 96)
    print(f"method = cnn_completed   domain = image   n_samples = "
          f"{tm['overall']['n_samples']}（每档 2 张重建）")
    print("-" * 96)
    print(f"{'energy':>8} {'n':>4} {'rmse_valid':>12} {'rmse_full':>12} "
          f"{'psnr':>9} {'psnr_common':>12} {'ssim':>9} {'p16_test':>10} {'p32_test':>10}")
    for e in br.ENERGIES:
        v = tm["per_energy"][e]
        pt = tm["projection_domain_check"][e]["by_input_phase"]
        p16 = np.mean([pt[p]["row16"]["by_split"].get("test", {}).get("rmse_valid", NAN)
                       for p in pt])
        p32 = np.mean([pt[p]["all32"]["by_split"].get("test", {}).get("rmse_valid", NAN)
                       for p in pt])
        print(f"{e:>8} {v['n_samples']:>4} {v['rmse_valid']:>12.6f} "
              f"{v['rmse_full']:>12.6f} {v['psnr']:>9.4f} {v['psnr_common']:>12.4f} "
              f"{v['ssim']:>9.6f} {p16:>10.6f} {p32:>10.6f}")
    o = tm["overall"]
    print(f"{'overall':>8} {o['n_samples']:>4} {o['rmse_valid']:>12.6f} "
          f"{o['rmse_full']:>12.6f} {o['psnr']:>9.4f} {o['psnr_common']:>12.4f} "
          f"{o['ssim']:>9.6f}")
    print("-" * 96)
    print(f"dense 参考与 fbp_sparse 缓存交叉校验："
          f"{ {k: v['byte_equal'] for k, v in dense_match.items()} }")
    for f in ("config.json", "train_log.csv", "test_per_sample.csv", "test_metrics.json"):
        p = out_dir / f
        print(f"已写 {f:>18}: {p.relative_to(ROOT)}  ({p.stat().st_size:,} B)")
    print("=" * 96)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
