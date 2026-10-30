#!/usr/bin/env python
# -*- coding: utf-8 -*-
r"""C 组消融：1D U-Net 只在 Row 16 上训练（真正的单行架构）

与既有 A / B 组的唯一差别
-------------------------
  A 组：2D U-Net，32x1200，全 32 行训练          （run 目录后缀 _base32_3mev）
  B 组：2D U-Net，32x1200，输入塌缩到 Row 16     （后缀 _row16_3mev，伪单行）
  C 组：1D U-Net，5x1200，只喂 Row 16            （本脚本，方法名 cnn1d）

为什么还要单独跑 C 组
--------------------
B 组把 1 行塞进 32 行的壳里（其余 31 行无输入、无监督），退化是架构性的必然。
C 组的张量里根本没有行轴，参数量 681,345（A 组的 34.9%），
"单行"由架构保证，不再靠抹值假装。

超参数刻意与 A/B 组逐项照搬（同 CSV、同 split、同 batch/lr/epochs/warmup/优化器/loss 口径），
唯一自变量是「架构维度」，否则对比不成立。

用法
----
    python scripts/train_row1d.py --seed 20261022 --epochs 100
    python scripts/train_row1d.py --seed 20261023 --epochs 100
    python scripts/train_row1d.py --seed 20261024 --epochs 100

产出：runs/<YYYYMMDD_HHMM>_cnn1d_bs<batch>_lr<lr>_row16_3mev_s<seed>/
        config.json / train_log.csv / test_metrics.json / test_per_sample.csv / best.pt
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT))
from data.csv_paths import resolve_csv  # noqa: E402  CSV 统一落点 csv_data/

from data.dataset import RECON_ROW, SparseCTDataset, load_csv_rows  # noqa: E402
from models.unet1d import UNet1D  # noqa: E402
# 复用 train.py 的**纯函数**：loss、播种、配置读取、环境信息。
# 这样 C 组的 loss 口径与 A/B 组逐字相同（masked MSE，per-pixel mask，分母 mask.sum()+eps）。
from train import (env_info, fmt_lr, git_info, load_config,  # noqa: E402
                   masked_mse, seed_everything)

CSV_NAME = resolve_csv("sparse_targets_3MeV.csv")
ENERGY = 3.0
ROWS_DIR = ROOT / "runs"

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
NAN = float("nan")


class Row1DSpecError(RuntimeError):
    """口径被破坏时抛错（fail fast）。"""


def _masked_mse_ps(pred: torch.Tensor, target: torch.Tensor,
                   mask: torch.Tensor, eps: float) -> torch.Tensor:
    """逐样本 masked MSE（float64 累加）。

    与 `train.py::masked_mse_per_sample` **数学完全一致**，唯一差别是归约维度：
    原函数写死 `dim=(1,2,3)`（4 维张量专用），这里取 `range(1, ndim)`
    —— 对 3 维的 1D 张量即 (1,2)，对 4 维张量仍是 (1,2,3)，两者等价。
    这样 C 组的 loss 与 A/B 组仍走同一条口径，不需要改 train.py（避免动到正文流水线）。
    """
    m = mask.to(torch.float64)
    sq = (pred.to(torch.float64) - target.to(torch.float64)) ** 2
    dims = tuple(range(1, sq.dim()))
    num = (sq * m).sum(dim=dims)
    den = m.sum(dim=dims) + eps
    return num / den


# ================================================================== 数据
class RowSliceDataset(Dataset):
    """把 SparseCTDataset 的 (C,32,1200) 沿行轴切出单行 -> (C,1200)。

    刻意**复用** SparseCTDataset：归一化、log_ratio、零计数填充、mask 定义与 A/B 组
    完全同一条代码路径，区别只在于最后切一行。因此任何差异都不可能来自数据侧。
    """

    def __init__(self, ds: SparseCTDataset, row: int) -> None:
        self.ds = ds
        self.row = int(row)

    def __len__(self) -> int:
        return len(self.ds)

    def __getitem__(self, i: int) -> dict[str, Any]:
        it = self.ds[i]
        x = it["x"][:, self.row, :]           # (C, 1200)
        y = it["y"][0, self.row][None]        # (1, 1200)
        m = it["mask"][0, self.row][None]     # (1, 1200)
        if x.shape[-1] != 1200 or y.shape != (1, 1200):
            raise Row1DSpecError(f"切片后形状异常 x={x.shape} y={y.shape}")
        return {
            "x": np.ascontiguousarray(x, dtype=np.float32),
            "y": np.ascontiguousarray(y, dtype=np.float32),
            "mask": np.ascontiguousarray(m, dtype=bool),
            "idx": int(i),
            "energy": float(it["meta"]["energy_MeV"]),
            "angle": float(it["meta"]["target_angle_deg"]),
            "split": it["meta"]["split"],
        }


def collate(batch: Sequence[dict[str, Any]]) -> dict[str, Any]:
    x = torch.from_numpy(np.stack([b["x"] for b in batch]))
    y = torch.from_numpy(np.stack([b["y"] for b in batch]))
    mask = torch.from_numpy(np.stack([b["mask"] for b in batch]))
    if x.dim() != 3 or x.shape[1] != 5:
        raise Row1DSpecError(f"x 形状必须是 (B,5,1200)，收到 {tuple(x.shape)}")
    if y.shape != (x.shape[0], 1, 1200) or mask.shape != y.shape:
        raise Row1DSpecError(f"y/mask 形状错误 y={tuple(y.shape)} mask={tuple(mask.shape)}")
    if mask.dtype != torch.bool:
        raise Row1DSpecError(f"mask dtype 必须是 bool，收到 {mask.dtype}")
    if not torch.isfinite(x).all() or not torch.isfinite(y).all():
        raise Row1DSpecError("x 或 y 含 NaN/Inf（归一化链路有 bug）")
    return {
        "x": x, "y": y, "mask": mask,
        "idx": torch.tensor([b["idx"] for b in batch], dtype=torch.long),
        "energy": torch.tensor([b["energy"] for b in batch], dtype=torch.float64),
        "angle": torch.tensor([b["angle"] for b in batch], dtype=torch.float64),
        "split": [b["split"] for b in batch],
    }


class CoverageCache:
    """32 行口径的 coverage / valid_coverage（数据级统计，与模型无关），按 idx 缓存。"""

    def __init__(self, ds: SparseCTDataset) -> None:
        self.ds = ds
        self._c: dict[int, tuple[float, float]] = {}

    def get(self, idx: int) -> tuple[float, float]:
        if idx not in self._c:
            self._c[idx] = self.ds.coverage_and_valid_of(idx)
        return self._c[idx]


# ================================================================== 评估
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    norm: tuple[float, float, float],
    eps: float,
    cov_cache: CoverageCache,
    collect_per_sample: bool = False,
) -> tuple[dict[str, float], list[dict[str, Any]]]:
    """单行评估：rmse_valid（mask 内）/ rmse_full（整行 1200 像素）。均反归一化到 -log 物理域。"""
    mean, std, _ = norm
    model.eval()
    se_v = se_f = 0.0
    n_v = n_f = 0
    loss_sum = 0.0
    loss_n = 0
    covs: list[float] = []
    vcovs: list[float] = []
    rows: list[dict[str, Any]] = []

    with torch.no_grad():
        for b in loader:
            x = b["x"].to(device, non_blocking=True)
            y = b["y"].to(device, non_blocking=True)
            mask = b["mask"].to(device, non_blocking=True)
            pred = model(x)
            if pred.shape != y.shape:
                raise Row1DSpecError(f"预测 shape {tuple(pred.shape)} != 目标 {tuple(y.shape)}")

            pred_phys = pred.to(torch.float64) * std + mean
            tgt_phys = y.to(torch.float64) * std + mean
            sq = (pred_phys - tgt_phys) ** 2
            m = mask.bool()
            se_v += float(sq.masked_fill(~m, 0.0).sum())
            n_v += int(m.sum())
            se_f += float(sq.sum())
            n_f += int(sq.numel())

            li = _masked_mse_ps(pred, y, mask, eps)
            loss_sum += float(li.sum())
            loss_n += int(li.numel())

            for i in range(len(b["idx"])):
                idx = int(b["idx"][i])
                cov, vcov = cov_cache.get(idx)
                covs.append(cov)
                vcovs.append(vcov)
                if collect_per_sample:
                    mm = m[i, 0]
                    nv = int(mm.sum())
                    rows.append({
                        "energy_MeV": f"{float(b['energy'][i]):g}",
                        "target_angle_deg": float(b["angle"][i]),
                        "split": b["split"][i],
                        "n_valid_pixels": nv,
                        "coverage": cov,
                        "valid_coverage": vcov,
                        "rmse_valid": float(torch.sqrt(
                            ((pred_phys[i, 0] - tgt_phys[i, 0]) ** 2 * mm).sum()
                            / max(nv, 1))),
                        "rmse_full": float(torch.sqrt(
                            (pred_phys[i, 0] - tgt_phys[i, 0]).pow(2).mean())),
                        "psnr": NAN, "ssim": NAN,
                        "rmse_invariant_warn": False,
                    })

    stats = {
        "n_samples": int(len(covs)),
        "coverage": float(np.mean(covs)) if covs else NAN,
        "valid_coverage": float(np.mean(vcovs)) if vcovs else NAN,
        "rmse_valid": math.sqrt(se_v / n_v) if n_v else NAN,
        "rmse_full": math.sqrt(se_f / n_f) if n_f else NAN,
        "loss": (loss_sum / loss_n) if loss_n else NAN,
        "n_valid_px": n_v,
        "n_full_px": n_f,
    }
    return stats, rows


# ================================================================== 运行器
class RunWriter:
    def __init__(self, run_id: str) -> None:
        self.run_id = run_id
        self.dir = ROWS_DIR / run_id
        if self.dir.exists():
            raise Row1DSpecError(f"run 目录已存在，拒绝复用: {self.dir}")
        self.dir.mkdir(parents=True)
        self._log = (self.dir / "train_log.csv").open("w", encoding="utf-8", newline="")
        import csv
        self._csv = csv.DictWriter(self._log, fieldnames=LOG_COLUMNS)
        self._csv.writeheader()

    def log_rows(self, rows: Sequence[dict[str, Any]]) -> None:
        for r in rows:
            self._csv.writerow(r)
        self._log.flush()

    def write_json(self, name: str, obj: Any) -> Path:
        p = self.dir / name
        p.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")
        return p

    def write_per_sample(self, rows: Sequence[dict[str, Any]]) -> Path:
        import csv
        p = self.dir / "test_per_sample.csv"
        with p.open("w", encoding="utf-8", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=PER_SAMPLE_COLUMNS)
            w.writeheader()
            for r in rows:
                w.writerow(r)
        return p

    def close(self) -> None:
        self._log.close()


def _log_row(epoch: int, split: str, st: dict[str, float], lr: float,
             loss_train: float, loss_val: float) -> dict[str, Any]:
    return {
        "epoch": epoch, "split": split, "energy_MeV": f"{ENERGY:g}",
        "n_samples": st["n_samples"], "coverage": st["coverage"],
        "valid_coverage": st["valid_coverage"], "rmse_valid": st["rmse_valid"],
        "rmse_full": st["rmse_full"], "psnr": NAN, "ssim": NAN,
        "n_invariant_warn": 0, "lr": lr,
        "loss_train": loss_train, "loss_val": loss_val,
    }


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="C 组：1D U-Net 单行（Row 16）训练")
    ap.add_argument("--seed", type=int, default=20261022)
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--row", type=int, default=RECON_ROW)
    ap.add_argument("--csv", default=CSV_NAME)
    ap.add_argument("--config", default="config.json")
    ap.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    ap.add_argument("--num-workers", type=int, default=0)
    args = ap.parse_args(argv)

    cfg = load_config(ROOT / args.config)
    n = cfg["norm"]
    norm = (float(n["norm_mean"]), float(n["norm_std"]), float(n["zero_log_fill"]))
    mean, std, fill = norm
    warmup = int(cfg.get("warmup", 50))
    eps = float((cfg.get("mask") or {}).get("eps", 1e-8))

    device = torch.device("cuda" if (torch.cuda.is_available() and args.device != "cpu") else "cpu")
    seed_everything(args.seed)

    ds = {s: SparseCTDataset(split=s, zero_log_fill=fill, norm_mean=mean, norm_std=std,
                             n_views=3, csv_path=args.csv, data_root=ROOT)
          for s in ("train", "val", "test")}
    for s in ds:
        e_vals = {round(float(r["energy_MeV"]), 6) for r in ds[s].rows}
        if e_vals != {ENERGY}:
            raise Row1DSpecError(f"{s} 的能量不是单一 {ENERGY} MeV: {sorted(e_vals)}")

    dl_train = DataLoader(RowSliceDataset(ds["train"], args.row), batch_size=args.batch,
                          shuffle=True, num_workers=args.num_workers, collate_fn=collate,
                          pin_memory=(device.type == "cuda"), drop_last=False)
    dl_train_eval = DataLoader(RowSliceDataset(ds["train"], args.row), batch_size=args.batch,
                               shuffle=False, num_workers=args.num_workers, collate_fn=collate,
                               pin_memory=(device.type == "cuda"), drop_last=False)
    dl_val = DataLoader(RowSliceDataset(ds["val"], args.row), batch_size=args.batch,
                        shuffle=False, num_workers=args.num_workers, collate_fn=collate,
                        pin_memory=(device.type == "cuda"), drop_last=False)
    dl_test = DataLoader(RowSliceDataset(ds["test"], args.row), batch_size=args.batch,
                         shuffle=False, num_workers=args.num_workers, collate_fn=collate,
                         pin_memory=(device.type == "cuda"), drop_last=False)

    model = UNet1D(in_channels=5, out_channels=1).to(device)
    n_params = model.n_parameters()
    steps = len(dl_train)

    commit, dirty = git_info()
    stamp = datetime.now().strftime("%Y%m%d_%H%M")
    run_id = f"{stamp}_cnn1d_bs{args.batch}_lr{fmt_lr(args.lr)}_row{args.row}_3mev_s{args.seed}"
    print("=" * 92)
    print(f"C 组消融：1D U-Net 单行训练   device={device}   run_id={run_id}")
    print(f"arch=UNet1D  in=(5,1200) -> out=(1,1200)  params={n_params:,} ({n_params / 1e6:.3f} M)  "
          f"（2D A/B 组为 1,950,017，本模型 {n_params / 1_950_017:.1%}）")
    print(f"data={args.csv}  row={args.row}  split=270/45/45  "
          f"train={len(ds['train'])} val={len(ds['val'])} test={len(ds['test'])}")
    print(f"batch={args.batch}  lr={args.lr:g}  epochs={args.epochs}  warmup={warmup} step  "
          f"steps/epoch={steps}  optimizer=Adam  AMP=off  seed={args.seed}")
    print(f"norm: mean={mean} std={std} zero_log_fill={fill}  "
          f"fill_norm(现算)={(fill - mean) / std:.6f}")
    print(f"loss: masked MSE，per-pixel mask（I>0 且 E>0），eps={eps}，与 A/B 组同一函数")
    print(f"git={commit[:7]} dirty={dirty}")
    print("=" * 92)

    cov_train = CoverageCache(ds["train"])
    cov_val = CoverageCache(ds["val"])
    cov_test = CoverageCache(ds["test"])

    writer = RunWriter(run_id)
    snapshot = {
        "run_id": run_id,
        "method": "cnn1d",
        "ablation_group": "C_row1d_true_single_row",
        "ablation_note": (
            "C 组：1D U-Net（无行轴，输入 (5,1200)），只训练 Row 16。"
            "对照 A 组 cnn/base32_3mev（2D，32 行输入）与 B 组 cnn/row16_3mev（2D，输入塌缩到 Row 16）。"
            "三组同 CSV、同 split、同 batch/lr/epochs/warmup/优化器/loss 口径，唯一自变量是架构维度。"
        ),
        "git_commit": commit, "git_dirty": dirty,
        "seed": args.seed, "seed_source": "cli",
        "batch": args.batch, "lr": args.lr, "epochs": args.epochs, "warmup": warmup,
        "warmup_unit": "step",
        "optimizer": "adam",
        "scheduler": "LambdaLR(min((step+1)/warmup, 1.0))",
        "amp": False,
        "norm": dict(n),
        "mask": dict(cfg.get("mask") or {}),
        "metric": {
            "primary": "非零像素 -log 物理域 RMSE（rmse_valid，mask=I>0 且 E>0，仅 Row 16 的 1200 像素）",
            "secondary": "rmse_full = 整行 1200 像素（含零计数像素）",
            "coverage_note": (
                "train_log.csv 的 coverage / valid_coverage 沿用 32x1200 口径"
                "（数据级统计，与模型无关，便于与 A/B 组日志同列横比）；"
                "**模型指标一律只在 Row 16 的 1200 像素上算**"
            ),
        },
        "data": {
            "csv": str(args.csv),
            "row": args.row,
            "split_sizes": {s: len(load_csv_rows(args.csv, s)) for s in ("train", "val", "test")},
            "energies": [ENERGY],
            "input_channels": ["θ−3°", "θ−1°", "θ+1°", "sinθ", "cosθ"],
        },
        "model": {"name": "UNet1D", "in_channels": 5, "out_channels": 1, "length": 1200,
                  "enc_channels": [32, 64, 128, 256], "n_downsample": 4,
                  "n_parameters": n_params,
                  "n_parameters_2d_unet": 1_950_017,
                  "param_ratio_vs_2d": round(n_params / 1_950_017, 6)},
        "env": env_info(),
    }
    writer.write_json("config.json", snapshot)

    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min((s + 1) / max(warmup, 1), 1.0))
    best = {"epoch": -1, "val_rmse": float("inf")}
    best_path = writer.dir / "best.pt"
    gstep = 0

    for epoch in range(args.epochs):
        t0 = time.perf_counter()
        model.train()
        lsum, ln = 0.0, 0
        for b in dl_train:
            x = b["x"].to(device, non_blocking=True)
            y = b["y"].to(device, non_blocking=True)
            mask = b["mask"].to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            pred = model(x)
            loss = masked_mse(pred, y, mask, eps)
            if not torch.isfinite(loss):
                raise Row1DSpecError(f"epoch {epoch} step {gstep}: loss 非有限 ({loss})")
            loss.backward()
            opt.step()
            sched.step()
            lsum += float(loss) * len(b["idx"])
            ln += len(b["idx"])
            gstep += 1

        if device.type == "cuda":
            torch.cuda.synchronize()
        sec = time.perf_counter() - t0
        tr_stats, _ = evaluate(model, dl_train_eval, device, norm, eps, cov_train)
        va_stats, _ = evaluate(model, dl_val, device, norm, eps, cov_val)
        cur_lr = float(opt.param_groups[0]["lr"])

        writer.log_rows([
            _log_row(epoch, "train", tr_stats, cur_lr, lsum / max(ln, 1), NAN),
            _log_row(epoch, "val", va_stats, cur_lr, NAN, va_stats["loss"]),
        ])

        v = va_stats["rmse_valid"]
        marker = ""
        if v < best["val_rmse"]:
            best = {"epoch": epoch, "val_rmse": v}
            torch.save({"model": model.state_dict(), "epoch": epoch,
                        "val_rmse_valid_phys": v, "run_id": run_id, "arch": "UNet1D",
                        "row": args.row,
                        "norm": {"norm_mean": mean, "norm_std": std, "zero_log_fill": fill}},
                       best_path)
            marker = "  <- best"
        print(f"[epoch {epoch:>3}] step={gstep} lr={cur_lr:.3e} "
              f"train_loss={lsum / max(ln, 1):.6e} val_rmse_valid={v:.6f} "
              f"train_rmse={tr_stats['rmse_valid']:.6f} {sec:.1f}s{marker}", flush=True)

    writer.close()
    print(f"\nbest: epoch={best['epoch']}  val_rmse_valid_phys={best['val_rmse']:.6f}")

    ckpt = torch.load(best_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"])
    te_stats, per_sample = evaluate(model, dl_test, device, norm, eps, cov_test,
                                    collect_per_sample=True)
    ps = writer.write_per_sample(per_sample)
    tm = {
        "run_id": run_id, "arch": "UNet1D", "row": args.row,
        "ablation_group": "C_row1d_true_single_row",
        "git_commit": commit, "git_dirty": dirty,
        "best_epoch": int(ckpt["epoch"]),
        "best_val_rmse_valid_phys": float(ckpt["val_rmse_valid_phys"]),
        "n_parameters": n_params,
        "overall": {k: te_stats[k] for k in
                    ("n_samples", "coverage", "valid_coverage", "rmse_valid", "rmse_full")},
        "row16_note": (
            "rmse_valid / rmse_full 均为 Row 16 单行（1200 像素）口径；"
            "coverage / valid_coverage 为 32x1200 数据级口径，便于与 A/B 组横比"
        ),
        "psnr_ssim_note": "本消融不涉及图像域指标，psnr/ssim 记 NaN",
    }
    mp = writer.write_json("test_metrics.json", tm)
    print(f"\n[test] n={te_stats['n_samples']}  rmse_valid={te_stats['rmse_valid']:.6f}  "
          f"rmse_full={te_stats['rmse_full']:.6f}")
    print(f"per_sample: {ps}")
    print(f"metrics   : {mp}")
    print(f"\n[OK] C 组 run 完成: {writer.dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
