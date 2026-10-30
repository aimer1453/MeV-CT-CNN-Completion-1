#!/usr/bin/env python
# -*- coding: utf-8 -*-
r"""
scripts/p02_make_step_csv.py —— 生成不同角度采样跨度 s 的 sparse_targets_s<s>.csv

为什么需要
----------
审稿人的核心追问是"2° 下 CNN 打不过线性插值，那是不是方法本身没价值？"。
破局实验 = 把采样跨度 s 从 2° 拉到 4°/6°/8°，看两条曲线是否交叉。
不同 s 需要不同的输入三元组，因此要为每档 s 生成一份新的数据表。

角度方案的严格推广（s=2 必须精确复现现有 sparse_targets.csv）
------------------------------------------------------------
现有 s=2：目标 θ，输入 ch0=θ−1、ch_prev=θ−3、ch_next=θ+1。
写成"半跨 h = s/2"的形式：
    ch0    = θ − h        （左端点，与 ch_next 一起把 θ 夹住）
    ch_next= θ + h
    ch_prev= θ − 3h       （左侧再退一格的额外上下文，不是 bracketing 左端点）
因此线性插值基线恒为  ½·[L(θ−h) + L(θ+h)]，h = s/2。

一致性（保证输入一定落在"已测"集合里）
--------------------------------------
按 θ mod s 分陪集。目标陪集 c 的输入角度 ≡ c + s/2 (mod s)，
于是 (c, c+s/2) 互为"实测/待补"的一对，共 s 个方向、s/2 个划分。
每个 θ 在且仅在"目标陪集 = θ mod s"那一个方向里当一次目标
→ 360 个角度全部用上，训练样本数恒为 1440，与 s=2 完全相同，
  可直接沿用原 split（同 seed、同划分，无泄漏差异）。

重建时的正弦图规模
------------------
某方向：实测陪集 360/s 个视角 + 补全陪集 360/s 个视角 = 720/s 个视角，
即补全后正弦图的角间距为 s/2 度（s=2 → 360 视角/1°；s=8 → 90 视角/4°）。
三种方法（fbp_sparse / linear / cnn）在同一 s 下用同一套视角，口径一致。

输出
----
  sparse_targets_s2.csv  （自检用：必须与 sparse_targets.csv 逐行一致）
  sparse_targets_s4.csv / s6 / s8
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from data.csv_paths import resolve_csv  # noqa: E402  CSV 统一落点 csv_data/
ANGLES_CSV = resolve_csv("angles.csv")
BASE_CSV = resolve_csv("sparse_targets.csv")
ENERGIES = ("1.5", "3", "4.5", "6")

FIELDS = ("energy_MeV", "scan_dir", "target_angle_deg", "target_file", "phase",
          "ch0_angle_deg", "ch0_file", "ch_prev_angle_deg", "ch_prev_file",
          "ch_next_angle_deg", "ch_next_file", "split")


def load_angles() -> dict[str, dict[int, dict[str, str]]]:
    """energy -> angle -> {file, scan_dir}。angles.csv 的 file 列已含 3_3/ 前缀。"""
    if not ANGLES_CSV.is_file():
        raise SystemExit(f"缺少 {ANGLES_CSV}")
    out: dict[str, dict[int, dict[str, str]]] = {}
    with open(ANGLES_CSV, newline="", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            if r["energy_MeV"] not in ENERGIES:
                continue
            a = int(round(float(r["angle_deg"])))
            out.setdefault(r["energy_MeV"], {})[a] = {
                "file": r["file"], "scan_dir": r["scan_dir"]}
    for e in ENERGIES:
        if e not in out or len(out[e]) != 360:
            raise SystemExit(f"[{e} MeV] angles.csv 期望 360 行，实得 {len(out.get(e, {}))}")
    return out


def load_split() -> dict[tuple[str, int], str]:
    """沿用原 sparse_targets.csv 的 split 划分（同 seed，不重新打乱）。"""
    if not BASE_CSV.is_file():
        raise SystemExit(f"缺少 {BASE_CSV}")
    out: dict[tuple[str, int], str] = {}
    with open(BASE_CSV, newline="", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            out[(r["energy_MeV"], int(round(float(r["target_angle_deg"]))))] = r["split"]
    if len(out) != 1440:
        raise SystemExit(f"sparse_targets.csv 期望 1440 行，实得 {len(out)}")
    return out


def build(step: int, angles: dict, split: dict) -> list[dict]:
    if step % 2 != 0:
        raise SystemExit(f"step 必须为偶数，收到 {step}")
    h = step // 2                     # 半跨：ch0 = θ−h, ch_next = θ+h, ch_prev = θ−3h
    rows: list[dict] = []
    for e in ENERGIES:
        tbl = angles[e]
        for theta in range(360):
            c = theta % step
            rec: dict[str, str] = {
                "energy_MeV": e,
                "scan_dir": tbl[theta]["scan_dir"],
                "target_angle_deg": str(theta),
                "target_file": tbl[theta]["file"],
                "phase": f"C{c}",
                "split": split[(e, theta)],
            }
            for key, off in (("ch0", -h), ("ch_prev", -3 * h), ("ch_next", +h)):
                a = (theta + off) % 360
                rec[f"{key}_angle_deg"] = str(a)
                rec[f"{key}_file"] = tbl[a]["file"]
                # 输入必须落在"已测陪集" (c + h) mod step
                if a % step != (c + h) % step:
                    raise SystemExit(
                        f"[s={step}] θ={theta} 的 {key} 角 {a} 落在陪集 {a % step}，"
                        f"期望 {(c + h) % step}")
            rows.append(rec)
    rows.sort(key=lambda r: (float(r["energy_MeV"]), int(r["target_angle_deg"])))
    return rows


def write(rows: list[dict], path: Path) -> None:
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(FIELDS))
        w.writeheader()
        w.writerows(rows)


def verify_s2(path: Path) -> bool:
    """s=2 的生成结果必须与现有 sparse_targets.csv 逐行一致（角度+文件+split）。"""
    def key(r: dict) -> tuple:
        return (r["energy_MeV"], int(float(r["target_angle_deg"])))

    with open(BASE_CSV, newline="", encoding="utf-8") as fh:
        base = {key(r): r for r in csv.DictReader(fh)}
    with open(path, newline="", encoding="utf-8") as fh:
        new = {key(r): r for r in csv.DictReader(fh)}
    if set(base) != set(new):
        return False
    for k in base:
        for col in ("target_file", "ch0_angle_deg", "ch0_file",
                    "ch_prev_angle_deg", "ch_prev_file",
                    "ch_next_angle_deg", "ch_next_file", "split"):
            if base[k][col] != new[k][col]:
                return False
    return True


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="生成不同采样跨度的 sparse_targets_s<s>.csv")
    ap.add_argument("--steps", default="2,4,6,8",
                    help="要生成的跨度列表，逗号分隔（默认 2,4,6,8）")
    args = ap.parse_args(argv)

    angles = load_angles()
    split = load_split()
    for s in (int(x) for x in args.steps.split(",")):
        rows = build(s, angles, split)
        path = resolve_csv(f"sparse_targets_s{s}.csv", for_write=True)
        write(rows, path)
        n_tr = sum(1 for r in rows if r["split"] == "train")
        extra = ""
        if s == 2:
            ok = verify_s2(path)
            extra = f"  [自检 vs sparse_targets.csv: {'一致' if ok else '★不一致★'}]"
            if not ok:
                raise SystemExit("s=2 未能复现现有数据表，角度方案推广有误，中止")
        print(f"[OK] {path.name}  rows={len(rows)}  train={n_tr}  "
              f"半跨h={s // 2}°  补全后角间距={s / 2:.1f}°  视角数={720 // s}{extra}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
