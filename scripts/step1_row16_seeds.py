#!/usr/bin/env python
# -*- coding: utf-8 -*-
r"""Step 1 侦察：中心行（Row 16）CNN 1D vs 线性插值，7 种子方差侦察。

设计要点
--------
- `sparse_targets_3MeV.csv` 的 split 列是**固定**的（build 时写死），
  `SparseCTDataset` 只按 split 列过滤，seed 不参与切分；
  因此 7 个种子（20261022..28）共用**完全相同**的 45 角度 test 集。
- 线性插值基线是确定性的、与 seed 无关，故 7 种子对应**同一个** L 值；
  唯一的方差来源是 CNN 的权重初始化 + 训练打乱顺序。
- 这恰好是检验「3 种子打平是否方差波动」的正确口径：CNN 散布 vs 固定 L。

指标（与 train_row1d.py::evaluate 逐项对齐）
-------------------------------------------
- -log 物理域、per-pixel mask（I_target>0 & empty_proj>0）RMSE；
- Row 16 是中心高通量行，mask = 100% 有效，rmse_valid == rmse_full；
- 这里统一报 rmse_valid（主指标）。

产出
----
- 线性基线 L（3MeV test 集，row16）
- 逐种子 CNN row16 RMSE、Δ = CNN - L
- 7 种子 CNN 的 mean ± SD
- CNN 赢 L 的次数（CNN < L 即赢，越低越好）
- boxplot 数据（min/q1/median/q3/max/全量 Δ 列表）

用法
----
    python scripts/step1_row16_seeds.py
"""
from __future__ import annotations

import csv
import json
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
RUNS = ROOT / "runs"
sys.path.insert(0, str(ROOT))
from data.csv_paths import resolve_csv  # noqa: E402
from data.dataset import LOG_EPS, log_ratio, valid_mask  # noqa: E402

CSV = resolve_csv("sparse_targets_3MeV.csv")
SEEDS = [20261022, 20261023, 20261024, 20261025, 20261026, 20261027, 20261028]
ENERGY = 3.0
ROW = 16
NAN = float("nan")


def load_empty(scan_dir: Path, cache: dict) -> np.ndarray:
    if scan_dir not in cache:
        p = scan_dir / "results" / "empty_proj.npy"
        if not p.is_file():
            raise RuntimeError(f"缺 empty_proj: {p}")
        cache[scan_dir] = np.load(p)
    return cache[scan_dir]


def to_phys(rel_file: str, empty: np.ndarray, fill: float) -> np.ndarray:
    img = np.load(Path(rel_file)).astype(np.float32)
    L = log_ratio(img, empty).astype(np.float64)
    return np.where(np.isfinite(L), L, float(fill))


def linear_row16(fill: float) -> float:
    """线性插值基线：3MeV test 集，Row 16 单行，mask RMSE（-log 物理域）。"""
    rows = [r for r in csv.DictReader(open(CSV, newline="", encoding="utf-8"))
            if r["split"] == "test" and abs(float(r["energy_MeV"]) - ENERGY) < 1e-9]
    cache: dict = {}
    se = 0.0
    n = 0
    for r in rows:
        scan = Path(r["target_file"]).parent.parent
        empty = load_empty(scan, cache)
        L_t = to_phys(r["target_file"], empty, fill)[ROW]
        L_0 = to_phys(r["ch0_file"], empty, fill)[ROW]
        L_n = to_phys(r["ch_next_file"], empty, fill)[ROW]
        pred = (L_0 + L_n) / 2.0
        I_t = np.load(Path(r["target_file"])).astype(np.float32)[ROW]
        mask = valid_mask(I_t, empty[ROW])
        d = pred - L_t
        se += float((d[mask] ** 2).sum())
        n += int(mask.sum())
    return math.sqrt(se / n) if n else NAN


def cnn_row16(seed: int) -> float | None:
    cand = sorted(p.name for p in RUNS.glob(f"*_cnn1d_*_row16_3mev_s{seed}")
                 if (p / "best.pt").is_file())
    if not cand:
        return None
    # 取最新一个（同 seed 不应有多个；若有取字典序最大）
    rid = cand[-1]
    tm = json.loads((RUNS / rid / "test_metrics.json").read_text(encoding="utf-8"))
    return float(tm["overall"]["rmse_valid"]), rid


def main() -> int:
    # fill 取 config.json（3MeV 全局 norm 的 zero_log_fill）
    cfg = json.loads((ROOT / "config.json").read_text(encoding="utf-8"))
    fill = float(cfg["norm"]["zero_log_fill"])
    print(f"[fill] zero_log_fill = {fill}")

    L = linear_row16(fill)
    print(f"[linear] 3MeV test row16 RMSE = {L:.6f}")

    cnn = {}
    missing = []
    for s in SEEDS:
        v = cnn_row16(s)
        if v is None:
            missing.append(s)
            print(f"[seed {s}] CNN run 未找到（待训练）")
        else:
            rmse, rid = v
            cnn[s] = rmse
            print(f"[seed {s}] CNN row16 RMSE = {rmse:.6f}   Δ={rmse - L:+.6f}   ({rid})")

    if missing:
        print(f"\n[状态] 仍缺 {len(missing)} 个 CNN run（种子 {missing}），"
              f"先训练再重跑本脚本做完整报告。")
        # 即便不全，也把已得部分写出，便于边训边看
    avail = [s for s in SEEDS if s in cnn]
    if len(avail) >= 2:
        arr = np.array([cnn[s] for s in avail])
        mean = float(arr.mean())
        sd = float(arr.std(ddof=1))
        print(f"\n=== 已就绪 {len(avail)} 种子 CNN row16 RMSE ===")
        print(f"mean = {mean:.6f}   SD = {sd:.6f}")
        wins = sum(1 for s in avail if cnn[s] < L)
        losses = sum(1 for s in avail if cnn[s] >= L)
        print(f"CNN 赢 linear：{wins} 胜 / {losses} 负（共 {len(avail)}）")
        deltas = {s: cnn[s] - L for s in avail}
        darr = np.array(list(deltas.values()))
        print("\n逐种子 Δ = CNN - linear:")
        for s in avail:
            print(f"  seed {s}: Δ = {deltas[s]:+.6f}")
        print(f"\nΔ 分布（boxplot 数据，{len(avail)} 点）:")
        print(f"  all   = {sorted(round(float(x), 6) for x in darr)}")
        print(f"  min   = {float(darr.min()):.6f}")
        print(f"  q1    = {float(np.percentile(darr, 25)):.6f}")
        print(f"  median= {float(np.median(darr)):.6f}")
        print(f"  q3    = {float(np.percentile(darr, 75)):.6f}")
        print(f"  max   = {float(darr.max()):.6f}")

    # 落盘（无论是否齐全，便于后续汇总）
    out = {
        "energy_MeV": ENERGY, "row": ROW, "fill": fill,
        "linear_row16_rmse": L,
        "seeds": {str(s): cnn.get(s) for s in SEEDS},
        "n_cnn_available": len(avail),
        "missing_seeds": missing,
    }
    if len(avail) >= 2:
        arr = np.array([cnn[s] for s in avail])
        darr = np.array([cnn[s] - L for s in avail])
        out["cnn_mean"] = float(arr.mean())
        out["cnn_sd"] = float(arr.std(ddof=1))
        out["cnn_wins_vs_linear"] = int(sum(1 for s in avail if cnn[s] < L))
        out["cnn_losses_vs_linear"] = int(sum(1 for s in avail if cnn[s] >= L))
        out["delta"] = {str(s): cnn[s] - L for s in avail}
        out["delta_boxplot"] = {
            "min": float(darr.min()), "q1": float(np.percentile(darr, 25)),
            "median": float(np.median(darr)), "q3": float(np.percentile(darr, 75)),
            "max": float(darr.max()), "all": [float(x) for x in darr],
        }
    p = RUNS / "_step1_row16_seeds.json"
    p.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n[json] {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
