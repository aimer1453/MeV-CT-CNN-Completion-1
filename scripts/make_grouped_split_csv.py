#!/usr/bin/env python
# -*- coding: utf-8 -*-
r"""
scripts/make_grouped_split_csv.py —— 按角度扇区划分（grouped split）生成数据表副本

为什么需要
----------
随机划分下，同一个张角 θ 与它的 90° 旋转副本 θ+90 / θ+180 / θ+270 会被打散到
train/val/test 三处；物模具备四重旋转对称时，这三处的投影携带同一份信息，
test 误差因此被低估。grouped split 把连续的角度扇区整块划给同一子集：
    train 0–89° / val 90–179° / test 180–359°
使 θ 与其旋转副本尽量落在同一子集（本划分下 train+val 覆盖 0–179°，
test 覆盖 180–359°，即 train 与 val 互为 90° 旋转邻域）。

用法
----
    python scripts/make_grouped_split_csv.py --energy 3.0
        # -> sparse_targets_3MeV_grouped.csv（360 行：train 90 / val 90 / test 180）
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from data.csv_paths import resolve_csv  # noqa: E402  CSV 统一落点 csv_data/

ap = argparse.ArgumentParser(description="按角度扇区生成 grouped split 数据表")
ap.add_argument("--energy", type=float, required=True, help="能量（MeV），如 3.0")
ap.add_argument("--src", default=None, help="源数据表（默认 sparse_targets_<E>MeV.csv）")
ap.add_argument("--out", default=None, help="输出文件名（默认自动命名）")
ap.add_argument("--train", type=int, nargs=2, default=(0, 90), metavar=("LO", "HI"))
ap.add_argument("--val", type=int, nargs=2, default=(90, 180), metavar=("LO", "HI"))
ap.add_argument("--test", type=int, nargs=2, default=(180, 360), metavar=("LO", "HI"))
args = ap.parse_args()

src = resolve_csv(args.src or f"sparse_targets_{args.energy:g}MeV.csv")
if not src.is_file():
    raise SystemExit(f"[FATAL] 源表不存在: {src}")
out_name = args.out or f"sparse_targets_{args.energy:g}MeV_grouped.csv"
out_path = resolve_csv(out_name, for_write=True)

rows = list(csv.DictReader(open(src, newline="", encoding="utf-8")))
if not rows:
    raise SystemExit(f"[FATAL] 源表为空: {src}")


def which(a: int) -> str:
    for name, (lo, hi) in (("train", tuple(args.train)),
                           ("val", tuple(args.val)),
                           ("test", tuple(args.test))):
        if lo <= a < hi:
            return name
    raise SystemExit(f"[FATAL] 角度 {a} 未落入任何扇区")


for r in rows:
    r["split"] = which(int(round(float(r["target_angle_deg"]))) % 360)

counts: dict[str, int] = {}
for r in rows:
    counts[r["split"]] = counts.get(r["split"], 0) + 1

with open(out_path, "w", newline="", encoding="utf-8") as fh:
    w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
    w.writeheader()
    w.writerows(rows)

print(f"[grouped] src  = {src}")
print(f"[grouped] out  = {out_path}")
print(f"[grouped] rows = {len(rows)}   " +
      "  ".join(f"{k}={v}" for k, v in sorted(counts.items())))
print(f"[grouped] 扇区 train={tuple(args.train)} val={tuple(args.val)} test={tuple(args.test)}")
