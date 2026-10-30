#!/usr/bin/env python
"""从主数据表抽取**单一能量档**的行，生成数据表副本。

用途
----
Row-16-only 消融需要在「同能量、同样本数」条件下与 32 行基线对照。
主表 `sparse_targets.csv` 含 4 个能量（1440 行），抽成单档后用 `--csv` 喂给 train.py。

用法
----
    python scripts/make_single_energy_csv.py --energy 3.0
        # -> sparse_targets_3MeV.csv（360 行：train 270 / val 45 / test 45）
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from data.csv_paths import resolve_csv  # noqa: E402  CSV 统一落点 csv_data/
ap = argparse.ArgumentParser(description="抽取单一能量档的数据表")
ap.add_argument("--energy", type=float, required=True, help="能量（MeV），如 3.0")
ap.add_argument("--src", default="sparse_targets.csv", help="源数据表")
ap.add_argument("--out", default=None, help="输出文件名（默认自动命名）")
args = ap.parse_args()

src = resolve_csv(args.src)
if src.name == "data_manifest.csv":
    raise SystemExit("[FATAL] 禁止把 data_manifest.csv 当数据源")
if not src.is_file():
    raise SystemExit(f"[FATAL] 源表不存在: {src}")

rows_all = list(csv.DictReader(open(src, newline="", encoding="utf-8")))
if not rows_all:
    raise SystemExit(f"[FATAL] 源表为空: {src}")

keep = [r for r in rows_all if abs(float(r["energy_MeV"]) - args.energy) < 1e-9]
if not keep:
    have = sorted({float(r["energy_MeV"]) for r in rows_all})
    raise SystemExit(f"[FATAL] 源表没有 {args.energy} MeV，可用: {have}")

n_per_split: dict[str, int] = {}
for r in keep:
    n_per_split[r["split"]] = n_per_split.get(r["split"], 0) + 1

expected = {"train": 270, "val": 45, "test": 45}
if n_per_split != expected:
    raise SystemExit(
        f"[FATAL] 划分数量异常 {n_per_split}，期望 {expected}。"
        "单能量表必须是完整 360 视角（dose_eval 要求角度为 0..359 整数序列）"
    )

out_name = args.out or f"sparse_targets_{args.energy:g}MeV.csv"
out = resolve_csv(out_name, for_write=True)
with open(out, "w", newline="", encoding="utf-8") as fh:
    w = csv.DictWriter(fh, fieldnames=list(rows_all[0].keys()))
    w.writeheader()
    w.writerows(keep)

print(f"已写出 {out_name}")
print(f"  能量 {args.energy:g} MeV，样本数 {len(keep)}，划分 {n_per_split}")
print(f"  scan_dir 唯一值: {sorted({r['scan_dir'] for r in keep})}")
