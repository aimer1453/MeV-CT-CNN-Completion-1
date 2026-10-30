#!/usr/bin/env python
# -*- coding: utf-8 -*-
r"""
build_train_manifest.py
=======================
从 data_manifest.csv 派生出训练专用清单 train_manifest_first_run.csv。

【口径（用户裁决 C2）】
  - data_manifest.csv  = 全量资产守护清单（16 档 5760 行），dataset 禁止读取。
  - train_manifest_first_run.csv = 第一段训练用清单，只含 in_first_run=1 的
    1.5 / 3.0 / 4.5 / 6.0 MeV 四档，共 1440 行。
  - dataset 只允许读 train_manifest_first_run.csv 与 angles.csv。

【为什么不让 dataset 直读 data_manifest.csv】
  一旦 dataset 指向全量清单，扩档就会变成"改一行 filter"的隐式行为，
  16 档会在无人察觉时进入第一段训练，违反"先验证方法再扩数据"的分段原则。
  用独立文件把"这一段只训 4 档"变成物理约束。

【split 列不在这里】
  270/45/45 的 train/val/test 划分属于稀疏三元组构造，落在 sparse_targets.csv。

用法：
  python scripts/build_train_manifest.py
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from data.csv_paths import resolve_csv  # noqa: E402  CSV 统一落点 csv_data/
SRC = resolve_csv("data_manifest.csv")
OUT = resolve_csv("train_manifest_first_run.csv", for_write=True)

FIRST_RUN_ENERGIES = (1.5, 3.0, 4.5, 6.0)
FIELDS = ["file", "scan_dir", "energy_MeV", "angle_deg", "size_bytes", "md5"]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=str(SRC))
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args()

    src, out = Path(args.src), Path(args.out)
    if not src.is_file():
        print(f"[FATAL] 源清单不存在: {src}", file=sys.stderr)
        return 2

    rows: list[dict] = []
    with open(src, newline="", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            if r.get("in_first_run") != "1":
                continue
            rows.append({k: r[k] for k in FIELDS})

    rows.sort(key=lambda r: (float(r["energy_MeV"]), float(r["angle_deg"])))

    problems: list[str] = []
    per_e: dict[str, int] = {}
    for r in rows:
        per_e[r["energy_MeV"]] = per_e.get(r["energy_MeV"], 0) + 1
        if float(r["energy_MeV"]) not in FIRST_RUN_ENERGIES:
            problems.append(f"混入非首轮能量: {r}")
    for e in FIRST_RUN_ENERGIES:
        key = f"{e:g}"
        if per_e.get(key) != 360:
            problems.append(f"{key} MeV 档行数 {per_e.get(key)} != 360")

    with open(out, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)

    print("=" * 74)
    print(f"{'energy_MeV':>12}{'rows':>8}{'bytes':>16}{'MiB':>10}")
    print("=" * 74)
    for e in FIRST_RUN_ENERGIES:
        key = f"{e:g}"
        sub = [r for r in rows if r["energy_MeV"] == key]
        b = sum(int(r["size_bytes"]) for r in sub)
        print(f"{key:>12}{len(sub):>8}{b:>16}{b/1024/1024:>10.1f}")
    print("=" * 74)
    total_b = sum(int(r["size_bytes"]) for r in rows)
    print(f"{'TOTAL':>12}{len(rows):>8}{total_b:>16}{total_b/1024/1024:>10.1f}")
    print(f"\n输出: {out}")

    if problems:
        print(f"\n[FAIL] {len(problems)} 个问题：")
        for p in problems[:20]:
            print("  ! " + p)
        return 1
    if len(rows) != 1440:
        print(f"\n[FAIL] 总行数 {len(rows)} != 1440")
        return 1
    print("[OK] 1440 行，4 档各 360 行")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
