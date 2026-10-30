#!/usr/bin/env python
# -*- coding: utf-8 -*-
r"""
scripts/make_rand90_csv.py —— R34 任务 2 控制实验：90 行**随机**训练集

要分离的两个变量
----------------
near45 相对 contig270 同时改了两件事：
  (a) 训练行数 270 → 90            （样本量）
  (b) 训练角度收缩到单一扇区 0–89°  （角度覆盖）
本 CSV 把 (b) 去掉、只保留 (a)：
  · test  **完全沿用 near45**：90–134°（逐位相同，test 难度不变）
  · train 从 315 个非 test 角中**随机抽 90 个**（不是扇区）
  · val   剩余的 225 个
因此若结果回落到 ≈ random（0.0875），说明崩溃来自「角度覆盖」；
若仍 ≈ near45（0.1374），说明崩溃来自「训练量」。

采样口径
--------
· 抽样种子固定为 SPLIT_SEED=20261022，且**三个模型种子共用同一份抽样**
  （train 行集合逐行相同），唯一变化量是模型初始化 —— 与 R34 其余对照一致。
· 单次实现（single realization）。90/315 的抽样对「test 角到最近 train 角距离」
  影响很小（期望最大间隙 ~4°），远小于要检测的 53% 效应，故不做多次抽样。
· ⚠ val 组成必然改变：near45 的 val 是 135–359°；本 CSV 的 val 是随机 225 角，
  含紧邻 test 的角度。这与原始 random split 的 val 语义一致（val 也是散落全圆），
  但对 best.pt 的选择有影响，论文侧引用时需说明。

用法
----
    python scripts/make_rand90_csv.py --energy 3.0 --n-train 90
        # -> csv_data/sparse_targets_3MeV_rand90.csv（360 行：train 90 / val 225 / test 45）
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from data.csv_paths import resolve_csv  # noqa: E402

SPLIT_SEED = 20261022

ap = argparse.ArgumentParser(description="生成 90 行随机训练集（test 沿用 near45）")
ap.add_argument("--energy", type=float, default=3.0)
ap.add_argument("--n-train", type=int, default=90)
ap.add_argument("--mode", default="iid", choices=("iid", "even"),
                help="iid = 从非 test 角中均匀随机抽（默认）；"
                     "even = 在非 test 角上等间隔铺满全圆，并强制纳入 test 块两端 89°/135°")
ap.add_argument("--src", default=None, help="源表（默认 near45，以继承其 test 角）")
ap.add_argument("--out", default=None)
args = ap.parse_args()

src = resolve_csv(args.src or f"sparse_targets_{args.energy:g}MeV_near45.csv")
if not src.is_file():
    raise SystemExit(f"[FATAL] 源表不存在: {src}")
out_name = args.out or (f"sparse_targets_{args.energy:g}MeV_rand90.csv" if args.mode == "iid"
                        else f"sparse_targets_{args.energy:g}MeV_even90.csv")
out_path = resolve_csv(out_name, for_write=True)

rows = list(csv.DictReader(open(src, newline="", encoding="utf-8")))
if not rows:
    raise SystemExit(f"[FATAL] 源表为空: {src}")

ang = {int(round(float(r["target_angle_deg"]))): r for r in rows}
test_ang = sorted(a for a, r in ang.items() if r["split"] == "test")
pool = sorted(a for a in ang if a not in set(test_ang))
if len(pool) != len(ang) - len(test_ang):
    raise SystemExit("[FATAL] 角度去重失败（源表有重复 target_angle_deg）")
if args.n_train > len(pool):
    raise SystemExit(f"[FATAL] n_train={args.n_train} 超过可抽角数 {len(pool)}")

if args.mode == "iid":
    rng = np.random.default_rng(SPLIT_SEED)
    train_ang = sorted(int(x) for x in rng.permutation(np.array(pool))[:args.n_train])
else:
    # even：在 315 个非 test 角上等间隔取点（保证全圆密度一致），
    # 再强制纳入 test 块两端（89° / 135°），使 test 块被**两侧夹住**。
    idx = np.linspace(0, len(pool) - 1, args.n_train).round().astype(int)
    sel = {pool[i] for i in idx}
    for a in (test_ang[0] - 1, test_ang[-1] + 1):
        if a in ang and a not in set(test_ang):
            sel.add(a)
    if len(sel) > args.n_train:                     # 去重后多出的按最密处删
        while len(sel) > args.n_train:
            arr = np.array(sorted(sel))
            gaps = np.diff(np.concatenate([arr, arr[:1] + 360]))
            sel.discard(int(arr[int(gaps.argmin())]))
    train_ang = sorted(sel)
    if len(train_ang) != args.n_train:
        raise SystemExit(f"[FATAL] even 模式选出 {len(train_ang)} 个角，应为 {args.n_train}")
train_set = set(train_ang)

for a, r in ang.items():
    r["split"] = "test" if a in set(test_ang) else ("train" if a in train_set else "val")

counts: dict[str, int] = {}
for r in ang.values():
    counts[r["split"]] = counts.get(r["split"], 0) + 1

with open(out_path, "w", newline="", encoding="utf-8") as fh:
    w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
    w.writeheader()
    w.writerows([ang[a] for a in sorted(ang)])

# —— 覆盖度诊断：test 角到最近 train 角的环形距离 ——
tr = np.array(train_ang)
te = np.array(test_ang)
d = np.abs(te[:, None] - tr[None, :])
d = np.minimum(d, 360 - d)
nn = d.min(axis=1)          # 每个 test 角到最近 train 角的环形距离
print(f"[rand90] src     = {src}")
print(f"[rand90] out     = {out_path}")
print(f"[rand90] rows    = {len(ang)}   " + "  ".join(f"{k}={v}" for k, v in sorted(counts.items())))
print(f"[rand90] test 角 = {test_ang[0]}–{test_ang[-1]}（沿用 near45，逐位相同）")
print(f"[rand90] train 角数 = {len(train_ang)}；test→最近 train 距离 "
      f"median={np.median(nn):.1f}°  mean={nn.mean():.1f}°  max={nn.max():.1f}°")
print(f"[rand90] 其中 test 角被 train 两侧夹住（bracketed）的比例 = "
      f"{float((nn <= 1.0).mean()) * 100:.1f}%")
print(f"[rand90] train 内 test 扇区内的 train 角数 = "
      f"{int(sum(1 for a in train_ang if 90 <= a <= 134))} / 45")
tr_gap = np.diff(np.concatenate([tr, tr[:1] + 360]))
print(f"[rand90] train 角最大间隙 = {tr_gap.max()}°（中位 {np.median(tr_gap):.1f}°）")
