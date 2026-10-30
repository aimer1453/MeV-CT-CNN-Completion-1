#!/usr/bin/env python
# -*- coding: utf-8 -*-
r"""
build_sparse_targets.py
=======================
生成 sparse_targets.csv：稀疏角补全任务的训练三元组表。

【角度公式（顾问最终裁决，不得改动）】
    ch0_angle    = (target - 1) mod 360
    ch_next_angle = (target + 1) mod 360
    ch_prev_angle = (target - 3) mod 360

  几何含义注释（口径确认 1 原文）：
    "覆盖环上 4°、相邻已知角间隔 2°、target 距 ch0/ch_next 各 1°、ch_prev 距 ch0 2°"
  !! span 4° 是三元组在环上覆盖的总范围（target-3 → target+1），
     不是对称 bracketing 的左右已知角距离。
  !! 禁止改成 ch0=target-3、ch_next=target+3。
  !! 禁止在 CSV 新增 span 列。

【ch_prev 的物理角色（口径确认 2）】
  ch_prev 是额外左侧上下文，不是 bracketing 左端点。
  bracketing 左端点 = ch0，右端点 = ch_next，target ∈ [ch0, ch_next) 顺时针。
  dataset 输入通道映射固定：
      ch_prev -> θ_{i-1}
      ch0     -> θ_i  （三元组中心，距 target 最近的稀疏角）
      ch_next -> θ_{i+1}
      target  -> 输出
  !! 禁止把 ch_prev 当 bracketing 左端点，禁止重新定义 ch0。

【相位】
  稀疏间隔 2°，互补双相位：phase A = 偶数角 {0,2,...,358}，phase B = 奇数角 {1,3,...,359}。
  本 CSV 的 phase 列 = **target 所属相位**。
  由于 ch0/prev/next 与 target 相差奇数度，已知角必然属于**相反相位**，自检第 7 项强制校验。

【wrap】
  所有 mod 360 已在生成时算完。dataset 只读 CSV，禁止运行时再推导 prev/next 或做 mod 360。

【split】
  顾问裁决：固定种子随机分层。
    每个能量档内，把 target_angle_deg 的 0..359 用**固定种子**随机打乱，
    打乱后前 270 = train、中 45 = val、后 45 = test；4 档各自独立切分。
  种子派生：base_seed = 20261022（--seed 控制），每档用 f"{seed}:{energy:g}"
    作为子种子，因此 (a) 单一种子可完全复现，(b) 四档的排列互不相同。
  !! seed 控制的是 split 划分；另有独立的 --sample-seed 只控制抽查哪 10 个 npy。
  不使用固定区间（0-269/270-314/315-359 已被裁决废弃）。

【自检 11 项，任一失败 -> 退出码 1，不写 csv】
用法：
  python scripts/build_sparse_targets.py --dry-run    # 只自检
  python scripts/build_sparse_targets.py              # 自检通过后写 sparse_targets.csv
"""

from __future__ import annotations

import argparse
import csv
import random
import re
import struct
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from data.csv_paths import resolve_csv  # noqa: E402  CSV 统一落点 csv_data/
DATA_ROOT = ROOT / "3_3"
ANGLES_CSV = resolve_csv("angles.csv")
TRAIN_MANIFEST = resolve_csv("train_manifest_first_run.csv")
OUT = resolve_csv("sparse_targets.csv", for_write=True)

FIRST_RUN_ENERGIES = (1.5, 3.0, 4.5, 6.0)

FIELDS = [
    "energy_MeV", "scan_dir",
    "target_angle_deg", "target_file", "phase",
    "ch0_angle_deg", "ch0_file",
    "ch_prev_angle_deg", "ch_prev_file",
    "ch_next_angle_deg", "ch_next_file",
    "split",
]

# 固定核对例子（顾问给出，生成后必须逐条命中）
FIXED_CASES = {
    0: (357, 359, 1),
    1: (358, 0, 2),
    2: (359, 1, 3),
    358: (355, 357, 359),
    359: (356, 358, 0),
}

SPLIT_SIZES = (("train", 270), ("val", 45), ("test", 45))


def phase_of(angle: int) -> str:
    """A = 偶数角，B = 奇数角"""
    return "A" if angle % 2 == 0 else "B"


def triple(target: int) -> tuple[int, int, int]:
    """(ch_prev, ch0, ch_next) —— 唯一角度公式入口"""
    return ((target - 3) % 360, (target - 1) % 360, (target + 1) % 360)


def make_splits(seed: int, energy: float) -> dict[int, str]:
    """固定种子随机分层：0..359 打乱后 270/45/45。

    子种子 f"{seed}:{energy:g}" 使四档排列互不相同，但整体仍由单一 seed 完全复现。
    """
    rng = random.Random(f"{seed}:{energy:g}")
    targets = list(range(360))
    rng.shuffle(targets)
    out: dict[int, str] = {}
    i = 0
    for name, n in SPLIT_SIZES:
        for t in targets[i:i + n]:
            out[t] = name
        i += n
    assert len(out) == 360, out
    return out


def npy_header(path: Path) -> tuple[str, tuple]:
    """无 numpy 解析 .npy header，返回 (descr, shape)"""
    with open(path, "rb") as f:
        if f.read(6) != b"\x93NUMPY":
            raise ValueError(f"非 npy: {path}")
        major = f.read(2)[0]
        if major == 1:
            hlen = struct.unpack("<H", f.read(2))[0]
        else:
            hlen = struct.unpack("<I", f.read(4))[0]
        hdr = __import__("ast").literal_eval(f.read(hlen).decode("latin1"))
    return hdr["descr"], tuple(hdr["shape"])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--out", default=str(OUT))
    ap.add_argument("--n-sample", type=int, default=10)
    ap.add_argument("--seed", type=int, default=20261022,
                    help="【控制 split 划分】固定种子随机分层的 base seed")
    ap.add_argument("--sample-seed", type=int, default=20261022,
                    help="仅用于决定抽查哪 N 个 npy，与 split 无关")
    args = ap.parse_args()

    for p in (ANGLES_CSV, TRAIN_MANIFEST):
        if not p.is_file():
            print(f"[FATAL] 缺少依赖文件: {p}", file=sys.stderr)
            return 2

    # ---- 从 angles.csv 建立 (energy, scan_dir) -> {angle: file} 唯一真相映射
    ang_map: dict[str, dict[int, str]] = defaultdict(dict)
    ang_energy: dict[str, str] = {}
    with open(ANGLES_CSV, newline="", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            key = r["scan_dir"]
            ang_map[key][int(float(r["angle_deg"]))] = r["file"]
            ang_energy[key] = r["energy_MeV"]

    train_files = set()
    with open(TRAIN_MANIFEST, newline="", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            train_files.add(r["file"])

    rows: list[dict] = []
    for scan_dir in sorted(ang_map, key=lambda k: float(ang_energy[k])):
        e = float(ang_energy[scan_dir])
        if e not in FIRST_RUN_ENERGIES:
            continue
        e_str = f"{e:g}"
        amap = ang_map[scan_dir]
        if sorted(amap) != list(range(360)):
            print(f"[FATAL] {scan_dir} 密集角不是 0..359: {len(amap)} 个", file=sys.stderr)
            return 2
        smap = make_splits(args.seed, e)
        for t in range(360):
            pv, c0, nx = triple(t)
            rows.append({
                "energy_MeV": e_str,
                "scan_dir": scan_dir,
                "target_angle_deg": t,
                "target_file": amap[t],
                "phase": phase_of(t),
                "ch0_angle_deg": c0, "ch0_file": amap[c0],
                "ch_prev_angle_deg": pv, "ch_prev_file": amap[pv],
                "ch_next_angle_deg": nx, "ch_next_file": amap[nx],
                "split": smap[t],
            })

    # =========================================================== 自检
    fails: list[str] = []

    # 1. 行数
    per_e = Counter(r["energy_MeV"] for r in rows)
    for e in FIRST_RUN_ENERGIES:
        if per_e[f"{e:g}"] != 360:
            fails.append(f"[1] {e:g}MeV 行数 {per_e[f'{e:g}']} != 360")
    if len(rows) != 1440:
        fails.append(f"[1] 总行数 {len(rows)} != 1440")

    # 2. target 覆盖
    for e_str in per_e:
        ts = sorted(r["target_angle_deg"] for r in rows if r["energy_MeV"] == e_str)
        if ts != list(range(360)):
            fails.append(f"[2] {e_str}MeV target 未覆盖 0..359 或有重复/缺失")

    # 3. 固定例子
    probe = {r["target_angle_deg"]: r for r in rows if r["energy_MeV"] == "1.5"}
    for t, (ep, e0, en) in FIXED_CASES.items():
        r = probe.get(t)
        got = (r["ch_prev_angle_deg"], r["ch0_angle_deg"], r["ch_next_angle_deg"]) if r else None
        if got != (ep, e0, en):
            fails.append(f"[3] target={t} 期望 {(ep, e0, en)} 实得 {got}")

    # 4~7. 每条样本的环上关系
    for r in rows:
        t = r["target_angle_deg"]
        pv, c0, nx = r["ch_prev_angle_deg"], r["ch0_angle_deg"], r["ch_next_angle_deg"]
        if (t - c0) % 360 != 1:
            fails.append(f"[4] target={t} ch0={c0} 环上距离 != 1")
        if phase_of(c0) == phase_of(t):
            fails.append(f"[4] target={t} 与 ch0={c0} 同相位")
        if pv != (c0 - 2) % 360:
            fails.append(f"[5] target={t} ch_prev={pv} != (ch0-2)%360")
        if nx != (c0 + 2) % 360:
            fails.append(f"[5] target={t} ch_next={nx} != (ch0+2)%360")
        span = (nx - c0) % 360
        if not (0 <= (t - c0) % 360 < span):
            fails.append(f"[6] target={t} 不在 [ch0={c0}, ch_next={nx}) 顺时针区间")
        for name, a in (("ch_prev", pv), ("ch0", c0), ("ch_next", nx)):
            if phase_of(a) == phase_of(t):
                fails.append(f"[7] target={t} 与 {name}={a} 相位相同")
        for name, f in (("target", r["target_file"]), ("ch0", r["ch0_file"]),
                        ("ch_prev", r["ch_prev_file"]), ("ch_next", r["ch_next_file"])):
            if not f.startswith(f"3_3/{r['scan_dir']}/projections/"):
                fails.append(f"[8] target={t} {name}_file 不在同 scan_dir: {f}")

    # 8. 文件存在性
    missing = 0
    for r in rows:
        for k in ("target_file", "ch0_file", "ch_prev_file", "ch_next_file"):
            if not (ROOT / r[k]).is_file():
                missing += 1
                if missing <= 10:
                    fails.append(f"[8] 文件不存在: {r[k]}")
    if missing > 10:
        fails.append(f"[8] 共 {missing} 个文件不存在")

    # 9. 抽查 npy（sample_seed 与 split seed 分离）
    rng = random.Random(args.sample_seed)
    sample = rng.sample(rows, args.n_sample)
    sample_report = []
    for r in sample:
        for k in ("target_file", "ch0_file", "ch_prev_file", "ch_next_file"):
            descr, shape = npy_header(ROOT / r[k])
            if shape != (32, 1200) or descr != "<f4":
                fails.append(f"[9] {r[k]} shape={shape} dtype={descr} 不符 (32,1200)/<f4")
        sample_report.append((r, npy_header(ROOT / r["target_file"])))

    # 10. split 计数
    split_cnt: dict[tuple[str, str], int] = Counter((r["energy_MeV"], r["split"]) for r in rows)
    for e in FIRST_RUN_ENERGIES:
        for name, want in SPLIT_SIZES:
            got = split_cnt[(f"{e:g}", name)]
            if got != want:
                fails.append(f"[10] {e:g}MeV {name} {got} != {want}")

    # 12. split 集合完整性：三份互斥且并集 = 0..359
    for e in FIRST_RUN_ENERGIES:
        k = f"{e:g}"
        sets = {name: {r["target_angle_deg"] for r in rows
                       if r["energy_MeV"] == k and r["split"] == name}
                for name, _ in SPLIT_SIZES}
        if set().union(*sets.values()) != set(range(360)):
            fails.append(f"[12] {k}MeV 三 split 并集 != 0..359")
        if sum(len(v) for v in sets.values()) != 360:
            fails.append(f"[12] {k}MeV 三 split 存在重叠：{sum(len(v) for v in sets.values())}")

    # 13. 确定性：同一 seed 重算两次必须一致
    for e in FIRST_RUN_ENERGIES:
        if make_splits(args.seed, e) != make_splits(args.seed, e):
            fails.append(f"[13] {e:g}MeV make_splits 不可复现")
    # 14. seed 真的生效：与已废弃的固定区间方案必须不同
    legacy = {t: ("train" if t <= 269 else ("val" if t <= 314 else "test")) for t in range(360)}
    for e in FIRST_RUN_ENERGIES:
        if make_splits(args.seed, e) == legacy:
            fails.append(f"[14] {e:g}MeV split 与废弃的固定区间方案完全一致，seed 未生效")
    # 15. 四档排列互不相同（各自独立切分）
    sigs = {e: tuple(sorted(t for t, s in make_splits(args.seed, e).items() if s == "val"))
            for e in FIRST_RUN_ENERGIES}
    if len(set(sigs.values())) != len(FIRST_RUN_ENERGIES):
        fails.append(f"[15] 四档 val 集合存在重复，切分不独立: {sigs}")

    # 11. 交叉命中
    hit = sum(1 for r in rows if r["target_file"] in train_files)
    if hit != len(rows):
        fails.append(f"[11] 与 train_manifest 交叉命中 {hit}/{len(rows)}")
    all_ang = {f for v in ang_map.values() for f in v.values()}
    for r in rows:
        for k in ("target_file", "ch0_file", "ch_prev_file", "ch_next_file"):
            if r[k] not in all_ang:
                fails.append(f"[11] {r[k]} 不在 angles.csv")

    # =========================================================== 报告
    print("=" * 100)
    print("【固定例子核对】(ch_prev, ch0, ch_next)")
    print(f"{'target':>8}{'期望':>20}{'实得':>20}   结果")
    for t, exp in FIXED_CASES.items():
        r = probe.get(t)
        got = (r["ch_prev_angle_deg"], r["ch0_angle_deg"], r["ch_next_angle_deg"]) if r else None
        print(f"{t:>8}{str(exp):>20}{str(got):>20}   {'OK' if got == exp else 'FAIL'}")

    print("\n" + "=" * 100)
    print(f"【R3 每档 split 样本数】（划分方式：固定种子 {args.seed} 随机打乱后 270/45/45）")
    print(f"{'E(MeV)':>8}{'rows':>6}{'train':>7}{'val':>6}{'test':>6}{'phaseA':>8}{'phaseB':>8}")
    for e in FIRST_RUN_ENERGIES:
        k = f"{e:g}"
        sub = [r for r in rows if r["energy_MeV"] == k]
        print(f"{k:>8}{len(sub):>6}{split_cnt[(k,'train')]:>7}{split_cnt[(k,'val')]:>6}"
              f"{split_cnt[(k,'test')]:>6}{sum(1 for r in sub if r['phase']=='A'):>8}"
              f"{sum(1 for r in sub if r['phase']=='B'):>8}")
    print(f"{'TOTAL':>8}{len(rows):>6}{sum(v for (e,s),v in split_cnt.items() if s=='train'):>7}"
          f"{sum(v for (e,s),v in split_cnt.items() if s=='val'):>6}"
          f"{sum(v for (e,s),v in split_cnt.items() if s=='test'):>6}"
          f"{sum(1 for r in rows if r['phase']=='A'):>8}{sum(1 for r in rows if r['phase']=='B'):>8}")

    # ---- R1. 每档每 split 的角度分布
    def quant(sorted_vals: list[int], q: float) -> float:
        """线性插值分位数（与 numpy 默认 linear 一致）"""
        if not sorted_vals:
            return float("nan")
        pos = (len(sorted_vals) - 1) * q
        lo = int(pos)
        hi = min(lo + 1, len(sorted_vals) - 1)
        return sorted_vals[lo] + (sorted_vals[hi] - sorted_vals[lo]) * (pos - lo)

    print("\n" + "=" * 100)
    print(f"【R1 每档每 split 的角度分布】(seed={args.seed})")
    print(f"{'E(MeV)':>7}{'split':>7}{'n':>5}{'min':>6}{'max':>6}{'P25':>8}{'P50':>8}{'P75':>8}")
    for e in FIRST_RUN_ENERGIES:
        k = f"{e:g}"
        for name, _ in SPLIT_SIZES:
            vals = sorted(r["target_angle_deg"] for r in rows
                          if r["energy_MeV"] == k and r["split"] == name)
            print(f"{k:>7}{name:>7}{len(vals):>5}{vals[0]:>6}{vals[-1]:>6}"
                  f"{quant(vals,0.25):>8.1f}{quant(vals,0.50):>8.1f}{quant(vals,0.75):>8.1f}")

    # ---- R2. 跨 split 文件共享
    print("\n" + "=" * 100)
    print("【R2 跨 split 文件共享】train 的 target 文件有多少出现在 val/test 的输入通道中")
    print(f"{'E(MeV)':>7}{'train_target':>13}{'val_in':>8}{'test_in':>8}{'hit_in':>8}{'ratio':>9}")
    tot = [0, 0, 0, 0]
    for e in FIRST_RUN_ENERGIES:
        k = f"{e:g}"
        tr = {r["target_file"] for r in rows if r["energy_MeV"] == k and r["split"] == "train"}
        val_in = {r[c] for r in rows if r["energy_MeV"] == k and r["split"] == "val"
                  for c in ("ch_prev_file", "ch0_file", "ch_next_file")}
        tst_in = {r[c] for r in rows if r["energy_MeV"] == k and r["split"] == "test"
                  for c in ("ch_prev_file", "ch0_file", "ch_next_file")}
        hit = tr & (val_in | tst_in)
        print(f"{k:>7}{len(tr):>13}{len(val_in):>8}{len(tst_in):>8}{len(hit):>8}"
              f"{len(hit)/len(tr)*100:>8.1f}%")
        tot[0] += len(tr); tot[1] += len(val_in); tot[2] += len(tst_in); tot[3] += len(hit)
    print(f"{'TOTAL':>7}{tot[0]:>13}{tot[1]:>8}{tot[2]:>8}{tot[3]:>8}{tot[3]/tot[0]*100:>8.1f}%")

    # ---- R4. seed 用途
    print("\n" + "=" * 100)
    print(f"【R4 seed 用途】--seed {args.seed} 控制 split 随机打乱（子种子 f'{{seed}}:{{energy}}'）；"
          f"--sample-seed {args.sample_seed} 只控制抽查哪 {args.n_sample} 个 npy")

    print("\n" + "=" * 100)
    print(f"【前 5 行完整内容】")
    show = ["energy_MeV", "scan_dir", "target_angle_deg", "phase", "ch_prev_angle_deg",
            "ch0_angle_deg", "ch_next_angle_deg", "split"]
    for r in rows[:5]:
        print("  " + "  ".join(f"{k}={r[k]}" for k in show))
        print(f"      target_file   = {r['target_file']}")
        print(f"      ch_prev_file  = {r['ch_prev_file']}")
        print(f"      ch0_file      = {r['ch0_file']}")
        print(f"      ch_next_file  = {r['ch_next_file']}")

    print("\n" + "=" * 100)
    print(f"【抽查 {args.n_sample} 个 npy】(sample-seed={args.sample_seed})")
    for r, (descr, shape) in sample_report:
        print(f"  E={r['energy_MeV']:>4} target={r['target_angle_deg']:>3} "
              f"shape={shape} dtype={descr}  {r['target_file'].split('/')[-1]}")

    if fails:
        print(f"\n[BLOCKED] {len(fails)} 项自检失败，未写 csv：")
        for f in fails[:30]:
            print("  ! " + f)
        if len(fails) > 30:
            print(f"  ... 共 {len(fails)} 条")
        return 1

    print("\n[OK] 15 项自检全部通过")

    if args.dry_run:
        print("[DRY-RUN] 未写文件")
        return 0

    out = resolve_csv(args.out, for_write=True)   # 裸文件名一律落到 csv_data/
    with open(out, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)
    print(f"已写出 {out}（{len(rows)} 行）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
