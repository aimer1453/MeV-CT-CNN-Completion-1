#!/usr/bin/env python
# -*- coding: utf-8 -*-
r"""
train.py —— 稀疏角 CT 补全训练 / 评估（新电脑从零重建，不参考旧机器任何代码）

==================== 已裁决口径（违反即视为该次运行无效） ====================
1. 优化器 Adam，lr = 1e-4（固定，不提供 CLI 覆盖），warmup = 50 **optimizer step**
   （按 step 计、不按 epoch 计；batch 降级时 warmup 仍固定 50 step，不按比例缩放）。
2. epochs 默认 5（读 config.json，可用 --epochs 覆盖并写入 run 记录）。
3. AMP **关闭**：全程不出现 autocast / GradScaler。
4. per-pixel mask loss：零计数像素不参与，分母 `mask.sum() + eps`，eps = 1e-8。
5. 评估反归一化到物理域（-log 域）后算 rmse_valid 与 rmse_full，且断言
   rmse_full >= rmse_valid（逐样本 + 分能量池化 + 全量池化，三层都查）。
6. best.pt 按 **val_rmse_valid_phys** 保存。
7. 分 1.5 / 3 / 4.5 / 6.0 MeV 记录 coverage、valid_coverage、rmse_valid、rmse_full。
8. psnr / ssim 一律填 NaN（口径待定，当前不作主指标）。
9. train_log.csv 严格 13 列，每 epoch 每档一行（train 4 行 + val 4 行）：
   epoch,split,energy_MeV,n_samples,coverage,valid_coverage,rmse_valid,rmse_full,
   psnr,ssim,lr,loss_train,loss_val
10. test 阶段加载 best.pt，落盘 test_per_sample.csv 与 test_metrics.json，
    并注明 best 对应的 epoch 与 val_rmse_valid_phys。

==================== batch 处理（C3′） ====================
- config.json 的 `batch` 为 "PENDING" 时：**必须**显式传 `--batch`，否则 SystemExit。
  绝不使用任何默认 batch。
- config 的 `batch` 已是数字时：不传 `--batch` 就用 config；传了且不一致 → SystemExit。
- 本次 run 的 `runs/<run_id>/config.json` 副本记录：
  最终生效 batch、CLI 是否传入 --batch、取值来源。
- smoke 定案后由人工回填全局 config.json；回填前本脚本对 PENDING 一律 fail-fast。

==================== 用法 ====================
# 正式训练（batch 已回填后）
venv/Scripts/python.exe train.py

# 正式训练（batch 仍 PENDING，必须显式传）
venv/Scripts/python.exe train.py --batch 8

# 结构/流程 smoke（只跑 1 个 train batch + 1 个 val batch，不写 runs/）
venv/Scripts/python.exe train.py --smoke --batch 8

# 模型结构自检（不涉及训练）
venv/Scripts/python.exe -m models.unet --inspect
==================================================================
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import platform
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from data.dataset import PROJECT_ROOT, RECON_ROW, SparseCTDataset, load_csv_rows
from models.unet import UNet

# ------------------------------------------------------------------ 常量
CONFIG_PATH = PROJECT_ROOT / "config.json"
RUNS_DIR = PROJECT_ROOT / "runs"
METHOD = "cnn"

# 缺陷数据域内训练（P0-1 第三步）用 --csv 覆盖；默认口径不变。
CSV_PATH: str = "sparse_targets.csv"

#: Row-16-only 消融开关。默认 False = 论文正文口径（全 32 行），由 --single-row 打开。
#: 打开后 SparseCTDataset 会把投影输入的行信息压缩到 RECON_ROW，网络结构与参数量不变。
SINGLE_ROW: bool = False

#: 训练损失口径。默认 masked = 论文正文口径（零计数像素不参与），由 --loss-mode
#: unmasked 打开做消融。**评估侧 metrics 恒为 masked 物理域 RMSE，不受此开关影响**，
#: 即"改训练目标、不改评估口径"，便于与主表逐位对照。
LOSS_MODE: str = "masked"

LOG_COLUMNS = [
    "epoch", "split", "energy_MeV", "n_samples", "coverage", "valid_coverage",
    "rmse_valid", "rmse_full", "psnr", "ssim", "n_invariant_warn", "lr",
    "loss_train", "loss_val",
]

PER_SAMPLE_COLUMNS = [
    "energy_MeV", "target_angle_deg", "split", "n_valid_pixels",
    "coverage", "valid_coverage", "rmse_valid", "rmse_full", "psnr", "ssim",
    "rmse_invariant_warn",
]

ENERGIES: tuple[float, ...] = (1.5, 3.0, 4.5, 6.0)
NAN = float("nan")

MASK_EPS_DEFAULT = 1e-8
RMSE_TOL = 1e-9
#: 某 epoch 逐样本 rmse_full < rmse_valid 的告警比例上限；超过即停止并上报
WARN_RATIO_LIMIT = 0.05


class TrainSpecError(RuntimeError):
    """配置或运行时不满足已裁决口径时抛出（fail fast，不做兜底）。"""


# ================================================================== 小工具
def fmt_lr(lr: float) -> str:
    """1e-4 -> '1e-4'（目录命名按 docs/run_artifacts_spec.md 的示例写法）。"""
    mant, exp = f"{lr:.0e}".split("e")
    return f"{mant}e{int(exp)}"


def energy_key(e: float) -> str:
    """1.5 -> '1.5'，3.0 -> '3'，4.5 -> '4.5'，6.0 -> '6'。"""
    return f"{e:g}"


def git_info() -> tuple[str, bool]:
    def _run(*args: str) -> str:
        return subprocess.run(
            ["git", *args], cwd=PROJECT_ROOT, capture_output=True, text=True, check=False
        ).stdout.strip()

    commit = _run("rev-parse", "HEAD") or "UNKNOWN"
    dirty = bool(_run("status", "--porcelain"))
    return commit, dirty


def env_info() -> dict[str, Any]:
    info: dict[str, Any] = {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "numpy": np.__version__,
        "cuda": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",
    }
    if torch.cuda.is_available():
        free, total = torch.cuda.mem_get_info()
        info["total_memory_bytes"] = int(total)
        info["free_memory_at_start_bytes"] = int(free)
        info["compute_capability"] = "%d.%d" % torch.cuda.get_device_capability(0)
    return info


def load_config(path: Path = CONFIG_PATH) -> dict[str, Any]:
    if not path.is_file():
        raise SystemExit(f"[FATAL] 找不到 {path}")
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def seed_everything(seed: int) -> None:
    import random

    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # 数据增强/随机性来源只有 shuffle，这里不追求 bitwise 复现
    torch.backends.cudnn.benchmark = True


# ================================================================== batch 裁决
def resolve_batch(cfg: dict[str, Any], cli_batch: int | None) -> tuple[int, str, bool]:
    """按 C3′ 解析 batch。

    返回 (生效 batch, 来源标记, CLI 是否传入)。
    - config 为 "PENDING" 且 CLI 未传 → SystemExit
    - config 为数字且 CLI 未传   → 用 config
    - config 为数字且 CLI 传了且不一致 → SystemExit
    """
    cfg_batch = cfg.get("batch")
    pending = isinstance(cfg_batch, str) and cfg_batch.strip().upper() == "PENDING"

    if pending:
        if cli_batch is None:
            raise SystemExit(
                "[FATAL] config.json 的 batch 仍为 \"PENDING\"。\n"
                "        原因：待 --smoke 从 8 起、OOM 降 4/2 实测确定后回填。\n"
                "        回填前必须显式传 --batch，本脚本不提供任何默认 batch。\n"
                "        例：venv/Scripts/python.exe train.py --batch 8"
            )
        return int(cli_batch), "cli_over_pending", True

    if not isinstance(cfg_batch, int) or isinstance(cfg_batch, bool) or cfg_batch <= 0:
        raise SystemExit(
            f"[FATAL] config.json 的 batch 非法: {cfg_batch!r}（应为正整数或 \"PENDING\"）"
        )

    if cli_batch is None:
        return cfg_batch, "config", False

    if int(cli_batch) != cfg_batch:
        raise SystemExit(
            f"[FATAL] --batch {cli_batch} 与 config.json 的 batch {cfg_batch} 不一致。\n"
            f"        batch 已在 config 中定案；若确需改动，请先改 config.json 并提交。"
        )
    return cfg_batch, "cli_consistent", True


def resolve_lr(cfg: dict[str, Any]) -> float:
    """lr 固定 1e-4，只读 config，不提供 CLI 覆盖。"""
    lr = cfg.get("lr")
    if isinstance(lr, str) or not isinstance(lr, (int, float)) or not math.isfinite(float(lr)):
        raise SystemExit(f"[FATAL] config.json 的 lr 非法: {lr!r}（应为有限正数）")
    if float(lr) <= 0:
        raise SystemExit(f"[FATAL] config.json 的 lr 必须为正，收到 {lr!r}")
    if abs(float(lr) - 1e-4) > 1e-12:
        print(f"[WARN] config.json 的 lr = {lr}，裁决值为 1e-4；以 config 为准")
    return float(lr)


def resolve_norm(cfg: dict[str, Any]) -> tuple[float, float, float]:
    """读全局 norm；zero_fill_normalized 一律现算，不读已有字段。"""
    norm = cfg.get("norm") or {}
    for k in ("norm_mean", "norm_std", "zero_log_fill"):
        if norm.get(k) is None:
            raise SystemExit(f"[FATAL] config.json 缺少 norm.{k}，禁止填默认值")
    mean = float(norm["norm_mean"])
    std = float(norm["norm_std"])
    fill = float(norm["zero_log_fill"])
    if not math.isfinite(std) or std <= 0:
        raise SystemExit(f"[FATAL] norm_std 必须为正有限数，收到 {std!r}")
    if norm.get("norm_scope") != "global":
        raise SystemExit(
            f"[FATAL] 当前只支持 norm_scope=global，收到 {norm.get('norm_scope')!r}"
        )
    return mean, std, fill


# ================================================================== 数据
class IndexedDataset(Dataset):
    """给 SparseCTDataset 的返回值补一个 idx，供评估时按样本归集。

    不改 data/dataset.py，只做包装。
    """

    def __init__(self, ds: SparseCTDataset) -> None:
        self.ds = ds

    def __len__(self) -> int:
        return len(self.ds)

    def __getitem__(self, i: int) -> dict[str, Any]:
        item = self.ds[i]
        item["idx"] = int(i)
        return item


def collate(batch: Sequence[dict[str, Any]]) -> dict[str, torch.Tensor]:
    """numpy -> torch，并断言 dtype 与有限性（数据集侧已保证，这里再兜一层）。"""
    x = torch.from_numpy(np.stack([b["x"] for b in batch]))
    y = torch.from_numpy(np.stack([b["y"] for b in batch]))
    mask = torch.from_numpy(np.stack([b["mask"] for b in batch]))

    if x.dtype != torch.float32 or y.dtype != torch.float32:
        raise TrainSpecError(f"x/y dtype 必须是 float32，收到 {x.dtype}/{y.dtype}")
    if mask.dtype != torch.bool:
        raise TrainSpecError(f"mask dtype 必须是 bool，收到 {mask.dtype}")
    if x.shape[1] not in (4, 5) or y.shape[1] != 1:
        # 5 = 三视角 + sin/cos（正文配置）；4 = 两视角消融 + sin/cos
        raise TrainSpecError(f"通道数错误：x={tuple(x.shape)} y={tuple(y.shape)}")
    if x.shape[-2:] != (32, 1200) or y.shape[-2:] != (32, 1200):
        raise TrainSpecError(f"空间尺寸错误：x={tuple(x.shape)} y={tuple(y.shape)}")
    if not torch.isfinite(x).all() or not torch.isfinite(y).all():
        raise TrainSpecError("x 或 y 含 NaN/Inf（归一化链路有 bug）")

    return {
        "x": x,
        "y": y,
        "mask": mask,
        "idx": torch.tensor([b["idx"] for b in batch], dtype=torch.long),
        "energy": torch.tensor(
            [float(b["meta"]["energy_MeV"]) for b in batch], dtype=torch.float64
        ),
        "angle": torch.tensor(
            [float(b["meta"]["target_angle_deg"]) for b in batch], dtype=torch.float64
        ),
    }


class CoverageCache:
    """coverage / valid_coverage 按 idx 缓存（首次 eval 时读盘，后续复用）。"""

    def __init__(self, ds: SparseCTDataset) -> None:
        self.ds = ds
        self._cache: dict[int, tuple[float, float]] = {}

    def get(self, idx: int) -> tuple[float, float]:
        if idx not in self._cache:
            self._cache[idx] = self.ds.coverage_and_valid_of(idx)
        return self._cache[idx]


def build_dataset(split: str, norm: tuple[float, float, float],
                  n_views: int = 3) -> SparseCTDataset:
    mean, std, fill = norm
    return SparseCTDataset(split=split, zero_log_fill=fill, norm_mean=mean, norm_std=std,
                           csv_path=CSV_PATH, n_views=n_views, single_row=SINGLE_ROW)


def build_loader(
    ds: SparseCTDataset,
    batch: int,
    shuffle: bool,
    num_workers: int,
    device: torch.device,
) -> DataLoader:
    return DataLoader(
        IndexedDataset(ds),
        batch_size=batch,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate,
        pin_memory=(device.type == "cuda"),
        drop_last=False,
    )


# ================================================================== loss / 指标
def masked_mse(
    pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor, eps: float
) -> torch.Tensor:
    """per-pixel mask MSE：零计数像素不参与，分母 mask.sum() + eps。"""
    m = mask.to(pred.dtype)
    sq = (pred - target) ** 2
    return (sq * m).sum() / (m.sum() + eps)


def masked_mse_per_sample(
    pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor, eps: float
) -> torch.Tensor:
    """逐样本 masked MSE（float64 累加），用于按能量归集训练 loss。"""
    m = mask.to(torch.float64)
    sq = (pred.to(torch.float64) - target.to(torch.float64)) ** 2
    num = (sq * m).sum(dim=(1, 2, 3))
    den = m.sum(dim=(1, 2, 3)) + eps
    return num / den


def unmasked_mse(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """全像素 MSE（消融用）。零计数像素在 target 里是 zero_fill_normalized 伪值，
    故该损失会驱动网络去拟合填充常数——这正是本消融要检验的副作用。
    **仅用于训练 loss**，评估口径始终 masked。"""
    return ((pred - target) ** 2).mean()


def unmasked_mse_per_sample(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """逐样本全像素 MSE（float64 累加），与 unmasked_mse 配套用于 loss 归集。"""
    sq = (pred.to(torch.float64) - target.to(torch.float64)) ** 2
    return sq.flatten(1).mean(dim=1)


def train_loss(pred: torch.Tensor, target: torch.Tensor,
               mask: torch.Tensor, eps: float) -> torch.Tensor:
    return (masked_mse(pred, target, mask, eps) if LOSS_MODE == "masked"
            else unmasked_mse(pred, target))


def train_loss_per_sample(pred: torch.Tensor, target: torch.Tensor,
                          mask: torch.Tensor, eps: float) -> torch.Tensor:
    return (masked_mse_per_sample(pred, target, mask, eps) if LOSS_MODE == "masked"
            else unmasked_mse_per_sample(pred, target))


class EnergyAccumulator:
    """按能量归集：样本数、像素数、平方误差和、coverage 和、loss 和。"""

    def __init__(self) -> None:
        self.d: dict[str, dict[str, float]] = {
            energy_key(e): {
                "n_samples": 0.0,
                "n_valid_px": 0.0,
                "n_full_px": 0.0,
                "se_valid": 0.0,
                "se_full": 0.0,
                "coverage_sum": 0.0,
                "valid_coverage_sum": 0.0,
                "loss_sum": 0.0,
                "loss_n": 0.0,
            }
            for e in ENERGIES
        }

    def add_sample(
        self,
        energy: float,
        n_valid: int,
        n_full: int,
        se_valid: float,
        se_full: float,
        coverage: float,
        valid_coverage: float,
    ) -> None:
        slot = self.d.get(energy_key(energy))
        if slot is None:
            raise TrainSpecError(f"未知能量 {energy!r}，不在 {ENERGIES} 内")
        slot["n_samples"] += 1
        slot["n_valid_px"] += n_valid
        slot["n_full_px"] += n_full
        slot["se_valid"] += se_valid
        slot["se_full"] += se_full
        slot["coverage_sum"] += coverage
        slot["valid_coverage_sum"] += valid_coverage

    def add_loss(self, energy: float, loss: float, weight: int = 1) -> None:
        slot = self.d[energy_key(energy)]
        slot["loss_sum"] += float(loss) * weight
        slot["loss_n"] += weight

    def summary(self) -> dict[str, dict[str, float]]:
        out: dict[str, dict[str, float]] = {}
        for e, s in self.d.items():
            if s["n_samples"] == 0:
                continue
            rmse_valid = math.sqrt(s["se_valid"] / s["n_valid_px"]) if s["n_valid_px"] else NAN
            rmse_full = math.sqrt(s["se_full"] / s["n_full_px"]) if s["n_full_px"] else NAN
            if not (math.isnan(rmse_valid) or math.isnan(rmse_full)):
                if not (rmse_full >= rmse_valid - RMSE_TOL):
                    raise TrainSpecError(
                        f"[{e} MeV] rmse_full ({rmse_full:.9f}) < rmse_valid ({rmse_valid:.9f})，"
                        f"违反硬性关系，本次运行无效"
                    )
            out[e] = {
                "n_samples": int(s["n_samples"]),
                "n_valid_px": int(s["n_valid_px"]),
                "n_full_px": int(s["n_full_px"]),
                "coverage": s["coverage_sum"] / s["n_samples"],
                "valid_coverage": s["valid_coverage_sum"] / s["n_samples"],
                "rmse_valid": rmse_valid,
                "rmse_full": rmse_full,
                "loss": (s["loss_sum"] / s["loss_n"]) if s["loss_n"] else NAN,
            }
        return out


def pooled(per_energy: dict[str, dict[str, float]]) -> dict[str, float]:
    """跨能量池化（按像素加权），用于 test_metrics 的 overall。"""
    n = sum(v["n_samples"] for v in per_energy.values())
    nv = sum(v["n_valid_px"] for v in per_energy.values())
    nf = sum(v["n_full_px"] for v in per_energy.values())
    # 从 rmse 反推平方误差和，避免重复累计
    se_valid = sum((v["rmse_valid"] ** 2) * v["n_valid_px"] for v in per_energy.values())
    se_full = sum((v["rmse_full"] ** 2) * v["n_full_px"] for v in per_energy.values())
    rv = math.sqrt(se_valid / nv) if nv else NAN
    rf = math.sqrt(se_full / nf) if nf else NAN
    if not (math.isnan(rv) or math.isnan(rf)):
        if not (rf >= rv - RMSE_TOL):
            raise TrainSpecError(f"overall rmse_full ({rf}) < rmse_valid ({rv})，违反硬性关系")
    return {
        "n_samples": int(n),
        "coverage": sum(v["coverage"] * v["n_samples"] for v in per_energy.values()) / n if n else NAN,
        "valid_coverage": sum(v["valid_coverage"] * v["n_samples"] for v in per_energy.values()) / n if n else NAN,
        "rmse_valid": rv,
        "rmse_full": rf,
    }


# ================================================================== 评估
@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    norm: tuple[float, float, float],
    eps: float,
    cov_cache: CoverageCache,
    collect_per_sample: bool = False,
) -> tuple[dict[str, dict[str, float]], list[dict[str, Any]], dict[str, dict[str, Any]]]:
    """评估一轮：按能量归集 rmse / coverage。

    逐样本 `rmse_full >= rmse_valid` 违反时**只告警、不中断**：
    记 `n_invariant_warn` 计数，并保存首个告警样本的 idx / 两个 rmse / energy / angle。
    硬断言保留在分能量池化（`EnergyAccumulator.summary`）与 overall 池化（`pooled`）里。

    返回 (per_energy_summary, per_sample_rows, warn_by_energy)。
    """
    mean, std, _ = norm
    model.eval()
    acc = EnergyAccumulator()
    rows: list[dict[str, Any]] = []
    warn: dict[str, dict[str, Any]] = {
        energy_key(e): {"n_warn": 0, "n_samples": 0, "first": None} for e in ENERGIES
    }

    for batch in loader:
        x = batch["x"].to(device, non_blocking=True)
        y = batch["y"].to(device, non_blocking=True)
        mask = batch["mask"].to(device, non_blocking=True)

        pred = model(x)
        if pred.shape != y.shape:
            raise TrainSpecError(f"预测 shape {tuple(pred.shape)} != 目标 {tuple(y.shape)}")

        # ---- 反归一化到物理域（-log 域）
        pred_phys = pred.to(torch.float64) * std + mean
        tgt_phys = y.to(torch.float64) * std + mean
        sq = (pred_phys - tgt_phys) ** 2

        m = mask.bool()
        per_valid_se = sq.masked_fill(~m, 0.0).flatten(1).sum(dim=1)
        per_valid_n = m.flatten(1).sum(dim=1)
        per_full_se = sq.flatten(1).sum(dim=1)
        per_full_n = torch.full_like(per_full_se, float(sq[0].numel()))

        rv = torch.sqrt(per_valid_se / per_valid_n.clamp_min(1))
        rf = torch.sqrt(per_full_se / per_full_n.clamp_min(1))
        # 逐样本违反 → 告警标记（不 raise）
        bad_flag = rf < rv - RMSE_TOL

        loss_i = masked_mse_per_sample(pred, y, mask, eps)

        for i in range(len(batch["idx"])):
            idx = int(batch["idx"][i])
            energy = float(batch["energy"][i])
            cov, vcov = cov_cache.get(idx)
            mask_mean = float(m[i].to(torch.float64).mean())
            if SINGLE_ROW:
                # Row-16-only 消融下这条检查**有意**不适用：
                # 正常口径 mask = (I_target>0)&(empty>0)，故 mask.mean() 必等于 valid_coverage；
                # 消融口径 mask 是"只监督 row16"的行选择器（占 1/32），与 valid_coverage（行内有效像素率）语义不同，
                # 因此跳过。cov>=vcov 与 rmse_full>=rmse_valid 两条硬断言仍保留。
                pass
            elif abs(vcov - mask_mean) > 1e-6:
                raise TrainSpecError(
                    f"idx={idx} 的 valid_coverage ({vcov:.8f}) != mask.mean() ({mask_mean:.8f})，"
                    f"评估口径与 mask 不一致"
                )
            if cov < vcov - 1e-12:
                raise TrainSpecError(f"idx={idx} 的 coverage ({cov}) < valid_coverage ({vcov})")

            n_valid = int(per_valid_n[i])
            acc.add_sample(
                energy,
                n_valid=n_valid,
                n_full=int(per_full_n[i]),
                se_valid=float(per_valid_se[i]),
                se_full=float(per_full_se[i]),
                coverage=cov,
                valid_coverage=vcov,
            )
            acc.add_loss(energy, float(loss_i[i]), weight=1)

            ek = energy_key(energy)
            slot = warn.setdefault(ek, {"n_warn": 0, "n_samples": 0, "first": None})
            slot["n_samples"] += 1
            is_warn = bool(bad_flag[i])
            if is_warn:
                slot["n_warn"] += 1
                if slot["first"] is None:
                    slot["first"] = {
                        "idx": idx,
                        "energy": energy,
                        "target_angle_deg": float(batch["angle"][i]),
                        "rmse_valid": float(rv[i]),
                        "rmse_full": float(rf[i]),
                    }

            if collect_per_sample:
                rows.append({
                    "energy_MeV": ek,
                    "target_angle_deg": float(batch["angle"][i]),
                    "split": loader.dataset.ds.split,  # type: ignore[attr-defined]
                    "n_valid_pixels": n_valid,
                    "coverage": cov,
                    "valid_coverage": vcov,
                    "rmse_valid": float(rv[i]),
                    "rmse_full": float(rf[i]),
                    "psnr": NAN,
                    "ssim": NAN,
                    "rmse_invariant_warn": is_warn,
                })

    return acc.summary(), rows, warn


# ================================================================== 可视化 dump
#: 每个能量档要导出的目标角度（度）；取 test 集里最接近的实际角度
VIS_ANGLES: tuple[float, ...] = (0.0, 45.0, 90.0, 135.0)


def _angle_dist(a: float, b: float) -> float:
    """0~359 环形角度距离。"""
    d = abs(a - b) % 360.0
    return min(d, 360.0 - d)


def pick_vis_indices(ds: SparseCTDataset) -> dict[int, dict[str, Any]]:
    """每个能量档 × 每个目标角度，挑一个最接近的 test 样本。

    角度按环形距离匹配；已选过的样本不再重复占用，避免同一 idx 出现两次。
    """
    rows = ds.rows
    picked: dict[int, dict[str, Any]] = {}
    used: set[int] = set()
    for e in ENERGIES:
        idxs = [i for i, r in enumerate(rows) if float(r["energy_MeV"]) == e]
        if not idxs:
            continue
        for want in VIS_ANGLES:
            cand = [i for i in idxs if i not in used] or idxs
            best = min(cand, key=lambda i: _angle_dist(float(rows[i]["target_angle_deg"]), want))
            used.add(best)
            actual = float(rows[best]["target_angle_deg"])
            picked[best] = {
                "energy_key": energy_key(e),
                "energy_MeV": float(e),
                "target_angle_deg": actual,
                "angle_wanted": want,
                "angle_offset": _angle_dist(actual, want),
            }
    return picked


def dump_vis(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    norm: tuple[float, float, float],
    out_path: Path,
) -> Path:
    """把选定样本的 pred / gt / mask 导出为 npz（物理域 -log 值，未归一化）。

    只做前向，不参与任何指标计算，也不改变训练/评估逻辑。
    """
    mean, std, fill = norm
    ds = loader.dataset.ds                       # type: ignore[attr-defined]
    picked = pick_vis_indices(ds)
    if not picked:
        raise TrainSpecError("vis dump：未能挑出任何样本")

    preds: list[np.ndarray] = []
    gts: list[np.ndarray] = []
    msks: list[np.ndarray] = []
    meta: list[dict[str, Any]] = []

    model.eval()
    with torch.no_grad():
        for batch in loader:
            idx_list = [int(v) for v in batch["idx"]]
            sel = [i for i, k in enumerate(idx_list) if k in picked]
            if not sel:
                continue
            x = batch["x"].to(device, non_blocking=True)
            y = batch["y"].to(device, non_blocking=True)
            pred = model(x)
            pred_phys = pred[sel].to(torch.float64) * std + mean
            tgt_phys = y[sel].to(torch.float64) * std + mean
            m = batch["mask"][sel].bool()
            for j, li in enumerate(sel):
                preds.append(pred_phys[j].cpu().numpy())
                gts.append(tgt_phys[j].cpu().numpy())
                msks.append(m[j].cpu().numpy())
                meta.append(picked[idx_list[li]])

    if not preds:
        raise TrainSpecError("vis dump：遍历 test loader 后没有匹配到任何样本")

    order = sorted(range(len(meta)),
                   key=lambda k: (meta[k]["energy_MeV"], meta[k]["target_angle_deg"]))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_path,
        pred=np.stack([preds[k] for k in order]).astype(np.float32),
        gt=np.stack([gts[k] for k in order]).astype(np.float32),
        mask=np.stack([msks[k] for k in order]).astype(bool),
        energy=np.array([meta[k]["energy_MeV"] for k in order], dtype=np.float64),
        angle=np.array([meta[k]["target_angle_deg"] for k in order], dtype=np.float64),
        # 附加元信息（不属于用户指定字段，便于下游脚本自检）
        domain=np.array("physical_-log（未归一化；无效像素填 zero_log_fill）"),
        zero_log_fill=np.float64(fill),
        norm_mean=np.float64(mean),
        norm_std=np.float64(std),
        angle_wanted=np.array([meta[k]["angle_wanted"] for k in order], dtype=np.float64),
        angle_offset=np.array([meta[k]["angle_offset"] for k in order], dtype=np.float64),
    )
    print(f"vis dump：{len(order)} 个样本  ->  {out_path}（{out_path.stat().st_size:,} B）")
    for k in order:
        m = meta[k]
        print(f"  {m['energy_key']:>5} MeV  angle={m['target_angle_deg']:>5.0f}°"
              f"（目标 {m['angle_wanted']:>5.0f}°，偏差 {m['angle_offset']:.0f}°）")
    return out_path


def run_dump_only(args: argparse.Namespace) -> int:
    """跳过训练，直接用 runs/<RUN_ID>/best.pt 跑 test + dump。

    不写 test_metrics.json / test_per_sample.csv，避免覆盖已有实验记录。
    """
    run_dir = RUNS_DIR / args.dump_only
    best_path = run_dir / "best.pt"
    cfg_path = run_dir / "config.json"
    for p in (best_path, cfg_path):
        if not p.is_file():
            raise TrainSpecError(f"缺少 {p}")

    cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    norm_cfg = cfg.get("norm") or {}
    for k in ("norm_mean", "norm_std", "zero_log_fill"):
        if norm_cfg.get(k) is None:
            raise TrainSpecError(f"{cfg_path.name} 的 norm.{k} 缺失")
    norm = (float(norm_cfg["norm_mean"]), float(norm_cfg["norm_std"]),
            float(norm_cfg["zero_log_fill"]))
    # 输入视角数从该 run 自己的 config 读，保证 dump-only 与当初训练的通道数一致
    n_views = int(cfg.get("input_views") or 3)

    device = torch.device(
        "cuda" if torch.cuda.is_available() and args.device != "cpu" else "cpu"
    )
    model = UNet(in_channels=n_views + 2).to(device)
    ckpt = torch.load(best_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"])
    print(f"[dump-only] run={args.dump_only}  device={device}")
    print(f"[dump-only] best.pt: epoch={ckpt['epoch']}  "
          f"val_rmse_valid_phys={float(ckpt['val_rmse_valid_phys']):.6f}")
    print(f"[dump-only] norm: mean={norm[0]}  std={norm[1]}  zero_log_fill={norm[2]}  "
          f"input_views={n_views} (in_channels={n_views + 2})")

    batch = args.batch if args.batch is not None else int(cfg.get("batch") or 8)
    ds_test = build_dataset("test", norm, n_views=n_views)
    test_loader = build_loader(ds_test, batch, shuffle=False,
                               num_workers=args.num_workers, device=device)
    print(f"[dump-only] test n={len(ds_test)}  batch={batch}")
    dump_vis(model, test_loader, device, norm, run_dir / "vis_samples.npz")
    return 0


# ================================================================== 记录
class RunWriter:
    """runs/<run_id>/ 下的 config.json / train_log.csv / test_metrics.json。"""

    def __init__(self, run_id: str, cfg: dict[str, Any]) -> None:
        self.run_id = run_id
        self.dir = RUNS_DIR / run_id
        self.dir.mkdir(parents=True, exist_ok=True)
        self.cfg = cfg
        self.log_path = self.dir / "train_log.csv"
        self._log_fh = None
        self._log_w = None

    # ---- config 副本
    def write_config(self, snapshot: dict[str, Any]) -> Path:
        p = self.dir / "config.json"
        with open(p, "w", encoding="utf-8") as fh:
            json.dump(snapshot, fh, ensure_ascii=False, indent=2)
            fh.write("\n")
        return p

    # ---- train_log.csv
    def open_log(self) -> None:
        self._log_fh = open(self.log_path, "w", encoding="utf-8", newline="")
        self._log_w = csv.DictWriter(self._log_fh, fieldnames=LOG_COLUMNS)
        self._log_w.writeheader()
        self._log_fh.flush()

    def write_log_rows(self, rows: Iterable[dict[str, Any]]) -> None:
        assert self._log_w is not None
        for r in rows:
            self._log_w.writerow({k: r[k] for k in LOG_COLUMNS})
        self._log_fh.flush()  # type: ignore[union-attr]

    def close_log(self) -> None:
        if self._log_fh is not None:
            self._log_fh.close()
            self._log_fh = None
            self._log_w = None

    # ---- 其它
    def write_json(self, name: str, obj: Any) -> Path:
        p = self.dir / name
        with open(p, "w", encoding="utf-8") as fh:
            json.dump(obj, fh, ensure_ascii=False, indent=2)
            fh.write("\n")
        return p

    def write_per_sample(self, rows: Sequence[dict[str, Any]]) -> Path:
        p = self.dir / "test_per_sample.csv"
        with open(p, "w", encoding="utf-8", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=PER_SAMPLE_COLUMNS)
            w.writeheader()
            for r in rows:
                out = {k: r[k] for k in PER_SAMPLE_COLUMNS}
                # psnr/ssim 统一写成 "NaN"（与 train_log.csv 一致，避免 "nan" 大小写不一）
                for k in ("psnr", "ssim"):
                    v = out.get(k)
                    if isinstance(v, float) and math.isnan(v):
                        out[k] = "NaN"
                # 指标统一 6 位小数，避免科学计数法混入
                for k in ("coverage", "valid_coverage", "rmse_valid", "rmse_full"):
                    if isinstance(out.get(k), float):
                        out[k] = f"{out[k]:.6f}"
                out["target_angle_deg"] = f"{float(out['target_angle_deg']):g}"
                w.writerow(out)
        return p


def make_run_id(batch: int, lr: float, now: datetime | None = None,
                method: str = METHOD) -> str:
    now = now or datetime.now()
    return f"{now:%Y%m%d_%H%M}_{method}_bs{batch}_lr{fmt_lr(lr)}"


def build_run_config(
    base: dict[str, Any],
    run_id: str,
    batch: int,
    batch_source: str,
    cli_batch_provided: bool,
    lr: float,
    epochs: int,
    norm: tuple[float, float, float],
    mask_eps: float,
    steps_per_epoch: int,
    warmup: int,
    n_parameters: int | None = None,
    epochs_cli_provided: bool = False,
    n_views: int = 3,
    extra: dict[str, Any] | None = None,
    csv_path: str | None = None,
    split_sizes: dict[str, int] | None = None,
    energies: tuple[float, ...] | None = None,
    seed: int | None = None,
    seed_source: str = "config",
) -> dict[str, Any]:
    """生成 run 目录内的 config 快照。

    `zero_fill_normalized` 必须由 norm 现算（规范硬约束），不复制原字段。

    `n_views` 决定输入通道布局（3 = θ−3°/θ−1°/θ+1° + sin/cos，5 通道；
    2 = 消融，丢弃 θ−3°，4 通道），config 快照里的 `data.input_channels`
    与顶层 `input_views` / `input_channels` 必须如实反映实际装配顺序，
    不能恒抄 5 通道版（否则评测侧 run_dump_only / cnn_recon.py 会按 3 视角
    建模，加载 4 通道权重时形状不匹配）。

    `csv_path` / `split_sizes` / `energies` 必须反映**实际生效的数据表**
    （默认主表 sparse_targets.csv / 1080-180-180 / 四档）。
    缺陷域内训练（--csv sparse_targets_flaw.csv）时由调用方显式传入，
    防止 run 记录把缺陷数据误记成主表（跨代混引的一种形式）。
    """
    mean, std, fill = norm
    commit, dirty = git_info()
    # config 里只记**文件名**（不带 csv_data/ 前缀），与历史 run 的记录格式一致；
    # 实际路径由 data/csv_paths.resolve_csv 解析（2026-10-27 起统一落点 csv_data/）。
    csv_name = Path(csv_path or "sparse_targets.csv").name

    # 输入通道布局随 n_views 变：2 视角消融丢掉 θ−3°（ch_prev_file）左上下文，
    # 装配顺序变为 [ch0_file, ch_next_file, sin, cos]，与 data/dataset.py:__getitem__ 一致
    if n_views == 3:
        ch_layout = {
            "ch0": "ch0_file (theta_i = target-1deg, 三元组中心)",
            "ch1": "ch_prev_file (theta_{i-1} = target-3deg, 左侧额外上下文)",
            "ch2": "ch_next_file (theta_{i+1} = target+1deg)",
            "ch3": "sin(theta_target) 广播图",
            "ch4": "cos(theta_target) 广播图",
        }
    elif n_views == 2:
        ch_layout = {
            "ch0": "ch0_file (theta_i = target-1deg)",
            "ch1": "ch_next_file (theta_{i+1} = target+1deg)",
            "ch2": "sin(theta_target) 广播图",
            "ch3": "cos(theta_target) 广播图",
            "_dropped": "ch_prev_file (target-3deg) 左上下文（两视角消融）",
        }
    else:
        raise TrainSpecError(f"n_views 只支持 2/3，得到 {n_views}")

    cfg = {
        "run_id": run_id,
        "method": METHOD,
        "git_commit": commit,
        "git_dirty": dirty,
        "seed": seed if seed is not None else base.get("seed"),
        "seed_source": seed_source,
        "batch": batch,
        "batch_source": batch_source,
        "cli_batch_provided": cli_batch_provided,
        "cli_batch_value": batch if cli_batch_provided else None,
        "lr": lr,
        "epochs": epochs,
        "epochs_effective": epochs,
        "epochs_cli_provided": epochs_cli_provided,
        "epochs_source": "cli" if epochs_cli_provided else "config",
        "epoch_start": 0,
        "warmup": warmup,
        "warmup_unit": "optimizer_step",
        "warmup_note": "按 optimizer step 计，不按 epoch；batch 降级时不按比例缩放",
        "steps_per_epoch": steps_per_epoch,
        "warmup_end_step_index": warmup - 1,
        "optimizer": "adam",
        "amp": False,
        "amp_note": "全程不启用 autocast / GradScaler（已裁决关闭）",
        "norm": {
            "norm_mean": mean,
            "norm_std": std,
            "norm_scope": "global",
            "zero_log_fill": fill,
            "zero_fill_normalized": (fill - mean) / std,
            "zero_fill_normalized_derivation": (
                f"(zero_log_fill - norm_mean) / norm_std = "
                f"({fill} - {mean}) / {std} = {(fill - mean) / std:.6f}（运行时现算）"
            ),
            "zero_fill_normalized_usage": "仅供人工核对；运行时代码由三值现算，不读本字段",
            "domain": "-log((I + 1e-10) / (E + 1e-10))",
            "fill_order_note": "填充发生在归一化之前的 -log 域；归一化域填充值不是常数 5.5",
            "recomputed": True,
            "source": (
                f"{csv_name} split=train 的 target，排除 (I<=0)|(E<=0) 像素后统计；"
                "val/test 直接复用"
            ),
            "norm_per_energy_reference": {
                "_note": "仅参考，不用于训练。训练只使用上面的全局 norm。",
                "1.5": {"mean": 0.6315, "std": 1.18, "zero_log_fill": 6.3},
                "3": {"mean": 0.6891, "std": 0.93, "zero_log_fill": 5.6},
                "4.5": {"mean": 0.7547, "std": 0.8104, "zero_log_fill": 4.9},
                "6": {"mean": 0.7875, "std": 0.7729, "zero_log_fill": 4.5},
            },
        },
        "coverage_definition": {
            "coverage": "I_target > 0 像素数 / 总像素数（资产口径）",
            "valid_coverage": "(I_target > 0) & (empty_proj > 0) 像素数 / 总像素数（评估口径）",
            "rmse_valid_uses": "mask = (I_target > 0) & (empty_proj > 0)",
        },
        "mask": {
            "per_pixel": True,
            "loss_mode": LOSS_MODE,
            "zero_count_excluded_from_loss": (LOSS_MODE == "masked"),
            "denominator": ("mask.sum() + eps" if LOSS_MODE == "masked"
                            else "pred.numel()（全像素，消融口径）"),
            "eps": mask_eps,
            "loss_mode_note": (
                "默认 masked；unmasked 为消融分支，零计数像素的 zero_fill 伪值参与训练。"
                "评估侧 rmse_valid/rmse_full 两种模式恒为 masked 口径，不受影响。"
            ),
        },
        "metric": {
            "primary": "val_rmse_valid_phys",
            "also_report": ["rmse_full", "coverage", "valid_coverage"],
            "invariant": "rmse_full >= rmse_valid",
            "psnr_ssim": "NaN（口径待定，当前不作主指标）",
        },
        "data": {
            "targets": csv_name,
            "manifest": "train_manifest_first_run.csv",
            "angles": "angles.csv",
            "energies_MeV": list(energies if energies is not None else ENERGIES),
            "split_sizes": split_sizes or {"train": 1080, "val": 180, "test": 180},
            "input_channels": ch_layout,
            "single_row_ablation": bool(SINGLE_ROW),
        },
        "model": {
            **(base.get("model") or {}),
            "n_parameters_actual": n_parameters,
            "structure": "enc DoubleConv 32/64/128/256, 4x maxpool; dec ConvT+concat+单 Conv3x3; head 1x1",
            "note": "参数量为实测值（models/unet.py --inspect）",
        },
        "env": env_info(),
    }
    if extra:
        cfg["runtime"] = extra
        # input_views / input_channels / input_views_note 必须同时落在 config **顶层**：
        # train.py:run_dump_only 与 scripts/cnn_recon.py 只读顶层 cfg.get("input_views")，
        # 只藏在 runtime 里会静默回退成 3 视角 → 5 通道模型加载 4 通道 best.pt 失败。
        # （顶层 input_channels 为 int 总通道数；布局字典在 data.input_channels，两者不冲突）
        for _k in ("input_views", "input_channels", "input_views_note", "loss_mode"):
            if _k in extra:
                cfg[_k] = extra[_k]
    return cfg


# ================================================================== smoke
def run_smoke(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    norm: tuple[float, float, float],
    lr: float,
    warmup: int,
    eps: float,
) -> int:
    """结构/流程 smoke：1 个 train batch（前向+反向+1 step）+ 1 个 val batch。

    不写 runs/、不保存权重、不评估全量。仅用于确认显存、shape、loss 有限性、
    反归一化正确性与 rmse 关系。
    """
    mean, std, fill = norm
    print("=" * 96)
    print("SMOKE（不写 runs/、不保存权重、不跑全量）")
    print("=" * 96)

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        free, total = torch.cuda.mem_get_info()
        print(f"mem_get_info 基线   : free={free:,} B ({free / 1024**3:.2f} GiB)  "
              f"total={total:,} B ({total / 1024**3:.2f} GiB)  "
              f"已占用={(total - free) / 1024**3:.2f} GiB")

    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min((s + 1) / max(warmup, 1), 1.0)
    )

    # ---------------- train batch ----------------
    model.train()
    # 1) 数据加载单独计时（首个 batch 走磁盘 IO + -log + 归一化）
    t_load = time.perf_counter()
    batch = next(iter(train_loader))
    load_ms = (time.perf_counter() - t_load) * 1000
    print(f"\n实际 batch         : {batch['x'].shape[0]}")
    print(f"x shape / dtype     : {tuple(batch['x'].shape)} / {batch['x'].dtype}")
    print(f"y shape / dtype     : {tuple(batch['y'].shape)} / {batch['y'].dtype}")
    print(f"mask shape / dtype  : {tuple(batch['mask'].shape)} / {batch['mask'].dtype}  "
          f"覆盖率={float(batch['mask'].float().mean()):.4f}")
    print(f"首个 batch 数据加载 : {load_ms:.1f} ms（含 npy 读取 + -log + 归一化）")

    x = batch["x"].to(device, non_blocking=True)
    y = batch["y"].to(device, non_blocking=True)
    mask = batch["mask"].to(device, non_blocking=True)

    # 2) 预热一次（cuDNN 算法选择 + kernel 首次编译），不计入稳态耗时
    t_warm = time.perf_counter()
    opt.zero_grad(set_to_none=True)
    warm_loss = train_loss(model(x), y, mask, eps)
    warm_loss.backward()
    opt.zero_grad(set_to_none=True)
    if device.type == "cuda":
        torch.cuda.synchronize()
    warm_ms = (time.perf_counter() - t_warm) * 1000

    # 3) 稳态计时：前向 + 反向 + 1 optimizer step
    t0 = time.perf_counter()
    opt.zero_grad(set_to_none=True)
    pred = model(x)
    loss = train_loss(pred, y, mask, eps)
    loss.backward()
    opt.step()
    sched.step()
    if device.type == "cuda":
        torch.cuda.synchronize()
    dt = (time.perf_counter() - t0) * 1000

    grad_norm = math.sqrt(
        sum(float(p.grad.detach().pow(2).sum()) for p in model.parameters() if p.grad is not None)
    )
    print(f"预热（不计时）      : {warm_ms:.1f} ms（cuDNN 算法选择 + kernel 首次编译）")
    print(f"稳态 前向+反向+1step: OK  ({dt:.1f} ms)")
    loss_v = float(loss.detach())
    print(f"loss（{LOSS_MODE} MSE）: {loss_v:.6e}   有限={math.isfinite(loss_v)}")
    print(f"梯度范数            : {grad_norm:.6e}   有限={math.isfinite(grad_norm)}")
    if not torch.isfinite(loss.detach()).all():
        print("[FAIL] loss 非有限")
        return 1
    if not math.isfinite(grad_norm) or grad_norm == 0.0:
        print("[FAIL] 梯度非有限或全零")
        return 1

    # ---------------- 反归一化正确性 ----------------
    print("\n[反归一化正确性]")
    probe = torch.tensor([-1.0, 0.0, 0.7202, 3.0, fill], dtype=torch.float64)
    probe_norm = (probe - mean) / std
    probe_back = probe_norm * std + mean
    err = float((probe_back - probe).abs().max())
    print(f"  denorm(norm(x)) == x 最大误差 : {err:.3e}  {'OK' if err < 1e-9 else 'FAIL'}")
    if err >= 1e-9:
        print("[FAIL] 反归一化往返误差过大")
        return 1

    fill_norm_calc = (fill - mean) / std
    print(f"  zero_log_fill={fill} -> 归一化域现算 {fill_norm_calc:.6f}")

    y_invalid = y[~mask]
    if y_invalid.numel() > 0:
        if SINGLE_ROW:
            # Row-16-only 消融下这条检查**有意**不适用：
            # mask 已收缩到 row16，而 y 保留真实 32 行 —— 后者是为了让训练期的硬断言
            # 「rmse_full >= rmse_valid」继续成立（rmse_full 需要统计未被监督行的真实误差）。
            # 这些行是"故意不监督的区域"，不是"零计数无效像素"，故跳过本项。
            print("  无效像素填充值检查 : 跳过（Row-16-only 模式：mask 已收缩到 row16，"
                  "其余行 y 保留真实投影以维持 rmse_full >= rmse_valid）")
        else:
            back = y_invalid.to(torch.float64) * std + mean
            delta = float((back - fill).abs().max())
            print(f"  无效像素反归一化回 zero_log_fill 最大偏差 : {delta:.3e}  "
                  f"{'OK' if delta < 1e-6 else 'FAIL'}")
            if delta >= 1e-6:
                print("[FAIL] 无效像素填充值反归一化不一致")
                return 1
    else:
        print("  本 batch 无无效像素（跳过该项）")

    # ---------------- 双口径 RMSE ----------------
    with torch.no_grad():
        pred_phys = pred.to(torch.float64) * std + mean
        tgt_phys = y.to(torch.float64) * std + mean
        sq = (pred_phys - tgt_phys) ** 2
        m = mask.bool()
        rv = math.sqrt(float(sq[m].sum()) / max(int(m.sum()), 1))
        rf = math.sqrt(float(sq.sum()) / sq.numel())
    print("\n[双口径物理域 RMSE]")
    print(f"  rmse_valid : {rv:.6f}   (有效像素 {int(mask.sum()):,})")
    print(f"  rmse_full  : {rf:.6f}   (全像素 {mask.numel():,})")
    print(f"  断言 rmse_full >= rmse_valid : {'OK' if rf >= rv - RMSE_TOL else 'FAIL'}")
    if rf < rv - RMSE_TOL:
        print("[FAIL] 违反 rmse_full >= rmse_valid")
        return 1

    # ---------------- val batch ----------------
    print("\n[val batch]")
    model.eval()
    vb = next(iter(val_loader))
    with torch.no_grad():
        vpred = model(vb["x"].to(device))
    print(f"  val x shape : {tuple(vb['x'].shape)}  ->  pred {tuple(vpred.shape)}")
    print(f"  pred 有限性 : {bool(torch.isfinite(vpred).all())}")

    # ---------------- 显存峰值 ----------------
    if device.type == "cuda":
        peak_alloc = torch.cuda.max_memory_allocated(device)
        peak_res = torch.cuda.max_memory_reserved(device)
        free2, total = torch.cuda.mem_get_info()
        print("\n[显存]")
        print(f"  峰值 allocated : {peak_alloc:,} B ({peak_alloc / 1024**3:.2f} GiB)")
        print(f"  峰值 reserved  : {peak_res:,} B ({peak_res / 1024**3:.2f} GiB)")
        print(f"  结束后 free    : {free2:,} B ({free2 / 1024**3:.2f} GiB)")

    params = sum(p.numel() for p in model.parameters())
    print(f"\n参数总量 : {params:,} ({params / 1e6:.3f} M)")
    print("\n[OK] smoke 全项通过（结构 / 前反向 / loss 有限 / 反归一化 / 双口径 RMSE）")
    return 0


# ================================================================== 训练主流程
def train(args: argparse.Namespace) -> int:
    global CSV_PATH, SINGLE_ROW, LOSS_MODE
    if getattr(args, "csv", None):
        CSV_PATH = str(args.csv)
        print(f"[CSV] 使用 --csv 覆盖数据表: {CSV_PATH}")
    if getattr(args, "single_row", False):
        SINGLE_ROW = True
        print(f"[ROW16] Row-16-only 消融：投影输入只保留第 {RECON_ROW} 行，"
              f"mask 收缩到该行（网络结构与参数量完全不变）")
    _lm = getattr(args, "loss_mode", "masked") or "masked"
    if _lm not in ("masked", "unmasked"):
        raise TrainSpecError(f"--loss-mode 仅允许 masked/unmasked，收到 {_lm!r}")
    LOSS_MODE = _lm
    if LOSS_MODE != "masked":
        print(f"[LOSS] 消融口径 loss_mode={LOSS_MODE}：全像素 MSE 参与训练；"
              f"评估侧 rmse_valid/rmse_full 仍为 masked 口径，不受影响")
    cfg_path = Path(args.config) if getattr(args, "config", None) else CONFIG_PATH
    base_cfg = load_config(cfg_path)
    norm = resolve_norm(base_cfg)
    mean, std, fill = norm
    lr = resolve_lr(base_cfg)
    cli_seed = getattr(args, "seed", None)
    seed = int(cli_seed if cli_seed is not None else base_cfg.get("seed", 20261022))
    seed_source = "cli" if cli_seed is not None else "config"
    n_views = int(getattr(args, "input_views", 3) or 3)
    if n_views not in (2, 3):
        raise TrainSpecError(f"--input-views 仅允许 2 或 3，收到 {n_views}")
    epochs = int(args.epochs if args.epochs is not None else base_cfg.get("epochs", 5))
    warmup = int(base_cfg.get("warmup", 50))
    mask_eps = float((base_cfg.get("mask") or {}).get("eps", MASK_EPS_DEFAULT))

    batch, batch_source, cli_batch_provided = resolve_batch(base_cfg, args.batch)

    device = torch.device(
        "cuda" if torch.cuda.is_available() and args.device != "cpu" else "cpu"
    )
    seed_everything(seed)

    commit_full, git_dirty_now = git_info()
    gpu_free = gpu_total = None
    if device.type == "cuda":
        gpu_free, gpu_total = torch.cuda.mem_get_info()

    print("=" * 96)
    print(f"稀疏角 CT 补全训练   method={METHOD}  device={device}")
    print(f"batch={batch}（来源 {batch_source}，CLI 传入={cli_batch_provided}）  "
          f"lr={lr:g}  epochs={epochs}  warmup={warmup} step  AMP=off")
    print(f"norm: mean={mean}  std={std}  zero_log_fill={fill}  "
          f"zero_fill_norm(现算)={(fill - mean) / std:.6f}  scope={base_cfg['norm']['norm_scope']}")
    print(f"mask: per_pixel=True  eps={mask_eps}  loss_mode={LOSS_MODE}"
          + ("  零计数像素不参与 loss" if LOSS_MODE == "masked"
             else "  ⚠ 消融：零计数像素（zero_fill 伪值）参与训练 loss"))
    print(f"seed={seed}（来源 {seed_source}）  git={commit_full[:7]}  git_dirty={git_dirty_now}")
    if device.type == "cuda":
        print(f"gpu: {torch.cuda.get_device_name(0)}  "
              f"free={gpu_free:,} B ({gpu_free / 1024**3:.2f} GiB) / "
              f"total={gpu_total:,} B ({gpu_total / 1024**3:.2f} GiB)")
        if gpu_free < 2 * 1024**3:
            print(f"[WARN] GPU free {gpu_free / 1024**3:.2f} GiB < 2 GiB，"
                  f"仍按 batch={batch} 启动（smoke 实测峰值 reserved 1.12 GiB）")
    print("=" * 96)

    ds_train = build_dataset("train", norm, n_views=n_views)
    ds_val = build_dataset("val", norm, n_views=n_views)
    print(f"样本数：train={len(ds_train)}  val={len(ds_val)}  "
          f"input_views={n_views}（in_channels={n_views + 2}）")

    nw = args.num_workers
    train_loader = build_loader(ds_train, batch, shuffle=True, num_workers=nw, device=device)
    # eval 用 shuffle=False 的顺序 loader，保证 idx 归集与日志可复现
    train_eval_loader = build_loader(ds_train, batch, shuffle=False, num_workers=nw, device=device)
    val_loader = build_loader(ds_val, batch, shuffle=False, num_workers=nw, device=device)

    model = UNet(in_channels=n_views + 2).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"模型：UNet  in_channels={n_views + 2}  params={n_params:,} ({n_params / 1e6:.3f} M)")

    steps_per_epoch = len(train_loader)   # drop_last=False → ceil(1080/batch)
    print(f"steps_per_epoch = {steps_per_epoch}   "
          f"（ceil({len(ds_train)}/{batch})，drop_last=False）")
    print(f"warmup={warmup} step → 占首 epoch {warmup / steps_per_epoch:.1%}，"
          f"结束于 0-based step {warmup - 1}")
    print("=" * 96)

    cov_train = CoverageCache(ds_train)
    cov_val = CoverageCache(ds_val)

    # ---------------- smoke 分支 ----------------
    if args.smoke:
        try:
            return run_smoke(
                model, train_loader, val_loader, device,
                norm=norm, lr=lr, warmup=warmup, eps=mask_eps,
            )
        except torch.cuda.OutOfMemoryError as exc:
            _report_oom(batch, exc)
            return 3

    # ---------------- 正式训练 ----------------
    # 两视角消融用 method="cnn_2view" 占命名槽位，保持
    # runs/<时间>_<method>_bs<batch>_lr<lr>/ 模式可解析
    run_id = make_run_id(batch, lr, method=METHOD if n_views == 3 else f"{METHOD}_2view")
    _tag = (getattr(args, "tag", None) or "").strip()
    if _tag:
        run_id = f"{run_id}_{_tag}"
    writer = RunWriter(run_id, base_cfg)
    if _tag:
        print(f"[TAG] run_id 追加后缀: _{_tag}")

    # 审计轨迹必须如实反映实际生效的数据表（--csv 覆盖时不得记成主表）
    try:
        split_sizes = {s: len(load_csv_rows(CSV_PATH, s)) for s in ("train", "val", "test")}
        energies = tuple(sorted({float(r["energy_MeV"]) for r in
                                 load_csv_rows(CSV_PATH, "train")}))
    except Exception as exc:  # 清点失败不应阻断训练，但必须显式告知
        print(f"[WARN] 数据表清点失败（config 快照按主表默认值记录）: {exc}")
        split_sizes, energies = None, None

    snapshot = build_run_config(
        base_cfg, run_id, batch=batch, batch_source=batch_source,
        cli_batch_provided=cli_batch_provided, lr=lr, epochs=epochs,
        norm=norm, mask_eps=mask_eps, steps_per_epoch=steps_per_epoch, warmup=warmup,
        n_parameters=n_params, epochs_cli_provided=(args.epochs is not None),
        n_views=n_views,
        csv_path=CSV_PATH, split_sizes=split_sizes, energies=energies,
        seed=seed, seed_source=seed_source,
        extra={
            "num_workers": nw,
            "loss_mode": LOSS_MODE,
            "loss_mode_note": (
                "masked = 论文正文口径（零计数像素不参与训练 loss）；"
                "unmasked = 消融（全像素 MSE，拟合 zero_fill 伪值）。"
                "两种模式的评估指标均为 masked 物理域 RMSE，可直接对照。"
            ),
            "input_views": n_views,
            "input_channels": n_views + 2,
            "input_views_note": (
                "3 = θ−3°/θ−1°/θ+1° + sin/cos（论文正文配置，5 通道）；"
                "2 = θ−1°/θ+1° + sin/cos（两视角消融，4 通道，丢弃 θ−3° 左上下文）。"
                "监督 y 与 mask 两配置完全一致。"
            ),
            "train_batches_per_epoch": steps_per_epoch,
            "warmup_end_step_index": warmup - 1,
            "warmup_note": (
                f"warmup 固定 {warmup} step；本 run 每 epoch {steps_per_epoch} step，"
                f"warmup 在第 1 epoch 的第 {warmup} 步结束（0-based 索引 {warmup - 1}）"
            ),
            "psnr_ssim": "NaN（口径待定，当前不作主指标，5 epoch 后按论文需要再补）",
        },
    )
    cfg_path = writer.write_config(snapshot)
    writer.open_log()
    print(f"run_id = {run_id}")
    print(f"记录目录 = {writer.dir}")
    print(f"config 副本 = {cfg_path}")

    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min((s + 1) / max(warmup, 1), 1.0)
    )

    best_path = writer.dir / "best.pt"
    best = {"epoch": -1, "val_rmse_valid": float("inf")}
    global_step = 0

    for epoch in range(epochs):
        t0 = time.perf_counter()
        model.train()
        loss_acc = EnergyAccumulator()
        n_step = 0

        for batch_data in train_loader:
            x = batch_data["x"].to(device, non_blocking=True)
            y = batch_data["y"].to(device, non_blocking=True)
            mask = batch_data["mask"].to(device, non_blocking=True)

            opt.zero_grad(set_to_none=True)
            pred = model(x)
            loss = train_loss(pred, y, mask, mask_eps)
            if not torch.isfinite(loss):
                raise TrainSpecError(f"epoch {epoch} step {global_step}: loss 非有限 ({loss})")
            loss.backward()
            opt.step()
            sched.step()

            per_i = train_loss_per_sample(pred.detach(), y, mask, mask_eps)
            for i in range(len(batch_data["idx"])):
                loss_acc.add_loss(float(batch_data["energy"][i]), float(per_i[i]), weight=1)

            global_step += 1
            n_step += 1

        if device.type == "cuda":
            torch.cuda.synchronize()
        train_sec = time.perf_counter() - t0

        # ---- 评估（train / val 都按能量评估）
        train_stats, _, train_warn = evaluate(
            model, train_eval_loader, device, norm, mask_eps, cov_train
        )
        val_stats, _, val_warn = evaluate(
            model, val_loader, device, norm, mask_eps, cov_val
        )
        cur_lr = float(opt.param_groups[0]["lr"])

        # 训练侧 masked loss 按能量归集（每样本 weight=1）
        tr_loss: dict[str, float] = {}
        for e in ENERGIES:
            k = energy_key(e)
            n = loss_acc.d[k]["loss_n"]
            tr_loss[k] = (loss_acc.d[k]["loss_sum"] / n) if n else NAN

        # ---- 14 列日志：每档一行
        rows: list[dict[str, Any]] = []
        for e in ENERGIES:
            key = energy_key(e)
            if key in train_stats:
                rows.append(_log_row(
                    epoch, "train", e, train_stats[key], tr_loss[key], NAN, cur_lr,
                    n_invariant_warn=train_warn.get(key, {}).get("n_warn", 0),
                ))
            if key in val_stats:
                rows.append(_log_row(
                    epoch, "val", e, val_stats[key], NAN, val_stats[key]["loss"], cur_lr,
                    n_invariant_warn=val_warn.get(key, {}).get("n_warn", 0),
                ))
        writer.write_log_rows(rows)

        # ---- 逐样本 rmse 告警比例检查（> 5% 立即停止并上报）
        n_warn, n_tot = _warn_totals(train_warn, val_warn)
        ratio = (n_warn / n_tot) if n_tot else 0.0
        if n_warn:
            _print_invariant_warn(epoch, train_warn, val_warn, ratio)
        if ratio > WARN_RATIO_LIMIT:
            print(
                f"\n[STOP] epoch {epoch} 逐样本 rmse_full < rmse_valid 告警比例 "
                f"{ratio:.2%} > 阈值 {WARN_RATIO_LIMIT:.0%}（{n_warn}/{n_tot}）\n"
                f"       已停止训练，train_log.csv 保留到本 epoch。等待裁决后再继续。"
            )
            writer.close_log()
            return 4

        # ---- best.pt 按 val_rmse_valid_phys
        val_all = pooled(val_stats)
        v = float(val_all["rmse_valid"])
        marker = ""
        if v < best["val_rmse_valid"]:
            best = {"epoch": epoch, "val_rmse_valid": v}
            torch.save(
                {
                    "model": model.state_dict(),
                    "epoch": epoch,
                    "val_rmse_valid_phys": v,
                    "val_rmse_full_phys": float(val_all["rmse_full"]),
                    "run_id": run_id,
                    "norm": {"norm_mean": mean, "norm_std": std, "zero_log_fill": fill},
                    "batch": batch,
                    "lr": lr,
                },
                best_path,
            )
            marker = "  <- best"

        print(
            f"[epoch {epoch}] step={n_step} lr={cur_lr:.3e} "
            f"train_loss={_wavg(train_stats, 'loss'):.6e} "
            f"val_rmse_valid={v:.6f} val_rmse_full={float(val_all['rmse_full']):.6f} "
            f"warn={n_warn}/{n_tot} {train_sec:.1f}s{marker}"
        )

    writer.close_log()
    print(f"\nbest: epoch={best['epoch']}  val_rmse_valid_phys={best['val_rmse_valid']:.6f}")
    print(f"权重: {best_path}（*.pt 不入库）")

    # ---------------- test ----------------
    if not best_path.is_file():
        raise TrainSpecError("best.pt 未生成，无法进入 test 阶段")

    ckpt = torch.load(best_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"])
    best_epoch = int(ckpt["epoch"])
    best_val = float(ckpt["val_rmse_valid_phys"])
    print(f"\n[test] 加载 best.pt：epoch={best_epoch}  val_rmse_valid_phys={best_val:.6f}")

    ds_test = build_dataset("test", norm, n_views=n_views)
    test_loader = build_loader(ds_test, batch, shuffle=False, num_workers=nw, device=device)
    test_stats, per_sample, test_warn = evaluate(
        model, test_loader, device, norm, mask_eps, CoverageCache(ds_test),
        collect_per_sample=True,
    )
    t_nwarn, t_ntot = _warn_totals(test_warn, {})
    if t_nwarn:
        _print_invariant_warn(-1, test_warn, {}, (t_nwarn / t_ntot) if t_ntot else 0.0)

    ps = writer.write_per_sample(per_sample)
    overall = pooled(test_stats)
    test_metrics = {
        "run_id": run_id,
        "git_commit": snapshot["git_commit"],
        "git_dirty": snapshot["git_dirty"],
        "best_epoch": best_epoch,
        "best_val_rmse_valid_phys": best_val,
        "best_note": f"test 阶段加载 best.pt，该权重对应 epoch={best_epoch}，val_rmse_valid_phys={best_val:.6f}",
        "invariant_warn": {
            "n_warn": t_nwarn,
            "n_samples": t_ntot,
            "ratio": (t_nwarn / t_ntot) if t_ntot else 0.0,
            "note": "逐样本 rmse_full < rmse_valid 的告警数（不中断）；池化行仍硬断言",
        },
        "overall": {
            "n_samples": overall["n_samples"],
            "coverage": overall["coverage"],
            "valid_coverage": overall["valid_coverage"],
            "rmse_valid": overall["rmse_valid"],
            "rmse_full": overall["rmse_full"],
            "psnr": NAN,
            "ssim": NAN,
        },
        "per_energy": {
            e: {
                "n_samples": v["n_samples"],
                "coverage": v["coverage"],
                "valid_coverage": v["valid_coverage"],
                "rmse_valid": v["rmse_valid"],
                "rmse_full": v["rmse_full"],
                "psnr": NAN,
                "ssim": NAN,
            }
            for e, v in test_stats.items()
        },
        "psnr_ssim_note": "PSNR/SSIM 口径待定，当前不作主指标；test_per_sample.csv 中同样为 NaN",
    }
    mp = writer.write_json("test_metrics.json", test_metrics)

    print(f"test_per_sample.csv: {ps}（{len(per_sample)} 行）")
    print(f"test_metrics.json  : {mp}")
    print(f"\n[test overall] n={overall['n_samples']}  "
          f"rmse_valid={overall['rmse_valid']:.6f}  rmse_full={overall['rmse_full']:.6f}")
    for e, v in test_stats.items():
        print(f"  {e:>5} MeV  n={v['n_samples']:>3}  "
              f"coverage={v['coverage']:.4f}  valid_coverage={v['valid_coverage']:.4f}  "
              f"rmse_valid={v['rmse_valid']:.6f}  rmse_full={v['rmse_full']:.6f}")

    # ---- 可选：导出可视化样本（只做前向，不影响上面的任何指标）
    if args.dump_vis:
        dump_vis(model, test_loader, device, norm, writer.dir / "vis_samples.npz")

    print("\n[OK] 训练 + test 完成")
    return 0


def _log_row(
    epoch: int,
    split: str,
    energy: float,
    stats: dict[str, float],
    loss_train: float,
    loss_val: float,
    lr: float,
    n_invariant_warn: int = 0,
) -> dict[str, Any]:
    return {
        "epoch": epoch,
        "split": split,
        "energy_MeV": energy_key(energy),
        "n_samples": stats["n_samples"],
        "coverage": f"{stats['coverage']:.6f}",
        "valid_coverage": f"{stats['valid_coverage']:.6f}",
        "rmse_valid": f"{stats['rmse_valid']:.6f}",
        "rmse_full": f"{stats['rmse_full']:.6f}",
        "psnr": "NaN",
        "ssim": "NaN",
        "n_invariant_warn": int(n_invariant_warn),
        "lr": f"{lr:.6e}",
        "loss_train": "NaN" if loss_train is None or not math.isfinite(loss_train) else f"{loss_train:.6e}",
        "loss_val": "NaN" if loss_val is None or not math.isfinite(loss_val) else f"{loss_val:.6e}",
    }


def _wavg(stats: dict[str, dict[str, float]], key: str) -> float:
    n = sum(v["n_samples"] for v in stats.values())
    if not n:
        return NAN
    return sum(v[key] * v["n_samples"] for v in stats.values() if math.isfinite(v[key])) / n


def _warn_totals(
    train_warn: dict[str, dict[str, Any]], val_warn: dict[str, dict[str, Any]]
) -> tuple[int, int]:
    """返回 (告警总数, 样本总数)，train + val 合并统计。"""
    n_warn = sum(w["n_warn"] for w in list(train_warn.values()) + list(val_warn.values()))
    n_tot = sum(w["n_samples"] for w in list(train_warn.values()) + list(val_warn.values()))
    return int(n_warn), int(n_tot)


def _print_invariant_warn(
    epoch: int,
    train_warn: dict[str, dict[str, Any]],
    val_warn: dict[str, dict[str, Any]],
    ratio: float,
) -> None:
    """打印逐样本不变式告警摘要（含首个违反样本的完整定位信息）。"""
    n_warn, n_tot = _warn_totals(train_warn, val_warn)
    parts = [
        f"{tag}/{ek}MeV {w['n_warn']}/{w['n_samples']}"
        for tag, group in (("train", train_warn), ("val", val_warn))
        for ek, w in group.items()
        if w["n_warn"]
    ]
    print(
        f"  [WARN] epoch {epoch}: 逐样本 rmse_full < rmse_valid 告警 "
        f"{n_warn}/{n_tot} = {ratio:.3%}（阈值 {WARN_RATIO_LIMIT:.0%}）"
        + ("　分档：" + "  ".join(parts) if parts else "")
    )
    shown = 0
    for tag, group in (("train", train_warn), ("val", val_warn)):
        for ek, w in group.items():
            f = w["first"]
            if not f or shown >= 2:
                continue
            print(
                f"         首个告警 [{tag}] energy={f['energy']:g}MeV "
                f"target_angle={f['target_angle_deg']:g}deg idx={f['idx']}  "
                f"rmse_valid={f['rmse_valid']:.9f} > rmse_full={f['rmse_full']:.9f}"
            )
            shown += 1


def _report_oom(batch: int, exc: BaseException) -> None:
    lower = {8: 4, 4: 2, 2: None}.get(batch)
    hint = (
        f"建议改传 --batch {lower} 重试"
        if lower
        else "已到最小档 2，仍 OOM：需排查其他进程占用或减小模型"
    )
    print(f"\n[OOM] batch={batch} 显存不足：{exc}")
    print(f"      {hint}")
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        free, total = torch.cuda.mem_get_info()
        print(f"      当前 free={free:,} B ({free / 1024**3:.2f} GiB) / total={total / 1024**3:.2f} GiB")


# ================================================================== CLI
def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="稀疏角 CT 补全训练（batch 为 PENDING 时必须显式 --batch）"
    )
    ap.add_argument("--batch", type=int, default=None,
                    help="batch size。config 为 PENDING 时必填；config 已定案时须与之一致")
    ap.add_argument("--epochs", type=int, default=None,
                    help="覆盖 config 的 epochs（默认读 config，5）")
    ap.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    ap.add_argument("--num-workers", type=int, default=0,
                    help="DataLoader workers（Windows 建议 0）")
    ap.add_argument("--smoke", action="store_true",
                    help="只跑 1 个 train batch + 1 个 val batch，不写 runs/、不存权重")
    ap.add_argument("--dump-vis", action="store_true",
                    help="test 阶段末尾导出 runs/<run_id>/vis_samples.npz"
                         "（每档 4 个角度：0/45/90/135，取最接近的 test 样本）")
    ap.add_argument("--dump-only", default=None, metavar="RUN_ID",
                    help="跳过训练，直接用 runs/<RUN_ID>/best.pt 跑 test 并 dump；"
                         "不写 test_metrics.json / test_per_sample.csv")
    ap.add_argument("--csv", default=None, metavar="PATH",
                    help="数据表路径（默认 sparse_targets.csv）。缺陷数据域内训练用 "
                         "sparse_targets_flaw.csv；不改默认口径")
    ap.add_argument("--config", default=None, metavar="PATH",
                    help="配置路径（默认 config.json）。缺陷数据需用重算 norm 的 "
                         "config_flaw.json；不改默认口径")
    ap.add_argument("--seed", type=int, default=None,
                    help="覆盖 config 的 seed（多种子统计用）。默认读 config.json 的 seed；"
                         "传入时写入 run 快照的 seed / seed_source")
    ap.add_argument("--tag", default=None, metavar="TAG",
                    help="run_id 追加后缀（如 dose0.5），仅用于区分 runs/ 目录名；"
                         "method 字段不变。P0-1 泊松降采样实验用")
    ap.add_argument("--input-views", type=int, default=3, choices=[2, 3],
                    help="输入实测视角数。3 = θ−3°/θ−1°/θ+1° + sin/cos（论文正文配置，"
                         "5 通道，默认）；2 = θ−1°/θ+1° + sin/cos（两视角消融，4 通道，"
                         "丢弃 θ−3° 左上下文）。监督 y 与 mask 两种配置完全一致")
    ap.add_argument("--single-row", action="store_true",
                    help="Row-16-only 消融（默认关闭）：输入的投影通道只保留第 "
                         f"{RECON_ROW} 行、其余置零，并把 per-pixel mask 收缩到该行，"
                         "使监督梯度 100%% 来自重建那一行。UNet 结构与参数量完全不变，"
                         "用于检验「跨行冗余是否是 CNN 优势的来源」")
    ap.add_argument("--loss-mode", choices=["masked", "unmasked"], default="masked",
                    help="训练损失口径。masked = 论文正文口径（零计数像素不参与，默认）；"
                         "unmasked = 消融（全像素 MSE，会拟合 zero_fill 伪值）。"
                         "两种模式的评估指标均为 masked 物理域 RMSE，口径不变")
    return ap.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        if args.dump_only:
            return run_dump_only(args)
        return train(args)
    except TrainSpecError as exc:
        print(f"\n[FATAL] {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
