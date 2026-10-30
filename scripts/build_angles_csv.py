#!/usr/bin/env python
# -*- coding: utf-8 -*-
r"""
build_angles_csv.py
===================
生成 angles.csv：投影文件 -> 角度 的唯一对照表。
并执行强制自检（角度 + macro 1:1 + macro 能量一致性）。fail-closed。

【唯一真相来源规则 — 必须遵守】
  角度只从投影文件名解析，能量只从目录名解析。
  !! 严禁读取 scan_parameters.json 的 angles 字段 !!
  !! 严禁读取 scan_parameters.json 的 energy_MeV 字段 !!（16 个目录全部硬编码 3.0，废弃字段）

【两套命名兼容】
  proj_a000_z+0.npy      -> 0.0   （9 个目录）
  proj_a0000.0_z+0.npy   -> 0.0   （7 个 _z0.0 目录）
  正则 ^proj_a([0-9.]+)_z[+-]?([0-9.]+)\.npy$ 同时命中，float() 归一为 0.0。

【macro 对应关系（实测确认）】
  proj_aNNN_z+0.npy  <->  macros/scan_aNNN_z+0.mac
  规则：前缀 proj_ 替换为 scan_，扩展名 .npy -> .mac，其余完全一致。
  !! 注意不是同名（不是 proj_*.mac），早期按同名匹配会 100% 落空 !!

【强制校验（默认开启，不得关闭）】
  每个目录逐项校验：
    1. 角度：解析集合排序后与预期序列逐元素比对（全量，不抽样）
    2. macro 存在性：每个 npy 有且仅有对应 .mac，匹配率必须 100%
    3. macro 能量：解析 .mac 中 /gps/ene/mono <value> <unit>，换算为 MeV，
       必须与该目录名解析出的能量完全一致
  !! 只解析 /gps/ene/mono 这一行；不解析 macro 的几何 / 材料 / 源位置；不降级为只查文件名 !!
  任一目录不通过 -> 退出码 1，不写 angles.csv。

用法：
  python scripts/build_angles_csv.py --selfcheck --check-macros     # 自检（默认行为）
  python scripts/build_angles_csv.py --dry-run                      # 只自检不写文件
  python scripts/build_angles_csv.py                                # 自检通过后写 angles.csv
  python scripts/build_angles_csv.py --first-run-only               # 只处理 4 档
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from data.csv_paths import resolve_csv  # noqa: E402  CSV 统一落点 csv_data/
DATA_ROOT = ROOT / "3_3"
OUT_CSV = resolve_csv("angles.csv", for_write=True)

FIRST_RUN_ENERGIES = (1.5, 3.0, 4.5, 6.0)

RE_DIR_ENERGY = re.compile(r"^energy_scan_([0-9.]+)MeV_")
RE_FILE_ANGLE = re.compile(r"^proj_a([0-9.]+)_z[+-]?([0-9.]+)\.npy$")
# macro 里只认这一行：/gps/ene/mono 1.5 MeV
RE_MACRO_ENERGY = re.compile(r"^\s*/gps/ene/mono\s+([0-9.]+)\s*([a-zA-Z]*)\s*$")

UNIT_TO_MEV = {"": 1.0, "MeV": 1.0, "mev": 1.0, "keV": 1e-3, "kev": 1e-3,
               "eV": 1e-6, "ev": 1e-6, "GeV": 1e3, "gev": 1e3}


def expected_angles(n: int) -> list[float]:
    """密集均匀扫描的预期角度序列：0..n-1，步长 1 度。

    局限：仅适用于当前数据集（每个能量目录 360 个 1 度步进的密集角）。
    若将来引入非均匀角度，必须改为从独立的角度定义文件读取，不得沿用本函数。
    因此本函数的比对只能证明"文件名角度自洽"，不能证明"角度物理正确"；
    后者由第 3 项 macro 能量校验和第 2 项 1:1 对应间接支撑。
    """
    return [float(i) for i in range(n)]


def npy_to_macro(npy_name: str) -> str:
    """proj_aNNN_z+0.npy -> scan_aNNN_z+0.mac（前缀替换，实测对应规则）"""
    if not npy_name.startswith("proj_"):
        raise ValueError(f"非预期前缀: {npy_name}")
    return "scan_" + npy_name[len("proj_"):-len(".npy")] + ".mac"


def macro_energies_mev(mac_path: Path) -> list[float]:
    """只解析 /gps/ene/mono 行，返回换算到 MeV 的全部取值。"""
    vals: list[float] = []
    with open(mac_path, "r", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            m = RE_MACRO_ENERGY.match(line)
            if not m:
                continue
            value = float(m.group(1))
            unit = m.group(2)
            if unit not in UNIT_TO_MEV:
                raise ValueError(f"未知能量单位 '{unit}' in {mac_path}")
            vals.append(value * UNIT_TO_MEV[unit])
    return vals


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--first-run-only", action="store_true",
                    help="只处理 1.5/3.0/4.5/6.0 MeV 这 4 档")
    ap.add_argument("--dry-run", action="store_true", help="只自检，不写 angles.csv")
    ap.add_argument("--selfcheck", action="store_true",
                    help="执行全量自检（默认已开启，此参数用于显式声明）")
    # macro 校验强制开启：默认 True，只提供关闭开关用于调试，且禁止与写文件同时使用
    ap.add_argument("--check-macros", dest="check_macros", action="store_true", default=True,
                    help="校验 macro 1:1 与能量一致（默认开启，强制）")
    ap.add_argument("--no-check-macros", dest="check_macros", action="store_false",
                    help="[禁止] 仅调试用；与写 angles.csv 互斥")
    ap.add_argument("--out", default=str(OUT_CSV))
    args = ap.parse_args()

    # C3：不开启 macro 校验不得生成最终 angles.csv
    if not args.check_macros and not args.dry_run:
        print("[ABORT] --no-check-macros 时禁止生成 angles.csv（macro 校验为强制项）。",
              file=sys.stderr)
        return 2

    if not DATA_ROOT.is_dir():
        print(f"[FATAL] 数据根不存在: {DATA_ROOT}", file=sys.stderr)
        return 2

    rows: list[dict] = []
    problems: list[str] = []
    detail_blocks: list[str] = []
    dirs = [d for d in sorted(DATA_ROOT.iterdir())
            if d.is_dir() and d.name.startswith("energy_scan_")]

    hdr = (f"{'scan_dir':<54}{'E':>5}{'npy':>5}{'ang':>5}{'A':>5}{'B':>5}"
           f"{'mac':>5}{'macE':>7}   angle  name  energy")
    print("=" * 118)
    print(hdr)
    print("=" * 118)

    tot = {"npy": 0, "ang": 0, "mac_ok": 0, "mac_missing": 0, "e_bad": 0}

    for d in dirs:
        m = RE_DIR_ENERGY.match(d.name)
        if not m:
            problems.append(f"目录名无法解析能量: {d.name}")
            print(f"{d.name:<54}{'?':>5}  <== 目录名无法解析能量")
            continue
        energy = float(m.group(1))
        if args.first_run_only and energy not in FIRST_RUN_ENERGIES:
            continue

        proj = d / "projections"
        macd = d / "macros"
        if not proj.is_dir():
            problems.append(f"缺少 projections/: {d.name}")
            print(f"{d.name:<54}{energy:>5g}  <== 缺少 projections/")
            continue
        if not macd.is_dir():
            problems.append(f"缺少 macros/: {d.name}")
            print(f"{d.name:<54}{energy:>5g}  <== 缺少 macros/")
            continue

        parsed: list[float] = []
        local_rows: list[dict] = []
        bad_name: list[str] = []
        missing_mac: list[str] = []
        mac_e_vals: set[float] = set()
        mac_ok = mac_bad_e = 0
        # 命名口径计数：按原始文件名里角度字段是否含小数点判断，不能用转换后的角度值
        # （A: proj_a000_z+0.npy  B: proj_a0000.0_z+0.npy）
        cnt_a = cnt_b = 0

        for f in sorted(proj.glob("*.npy")):
            fm = RE_FILE_ANGLE.match(f.name)
            if not fm:
                problems.append(f"文件名无法解析角度: {f}")
                bad_name.append(f.name)
                continue
            angle = float(fm.group(1))
            if "." in fm.group(1):
                cnt_b += 1
            else:
                cnt_a += 1
            parsed.append(angle)
            local_rows.append({
                "energy_MeV": f"{energy:g}",
                "angle_deg": f"{angle:g}",
                "scan_dir": d.name,
                "file": f.relative_to(DATA_ROOT.parent).as_posix(),
            })

            # ---- 2/3. macro 存在性 + 能量
            if args.check_macros:
                mac = macd / npy_to_macro(f.name)
                if not mac.is_file():
                    missing_mac.append(mac.name)
                    continue
                vals = macro_energies_mev(mac)
                if len(set(vals)) != 1:
                    problems.append(f"{mac} 含多个不同 /gps/ene/mono 取值: {sorted(set(vals))}")
                    mac_bad_e += 1
                    continue
                e_mac = vals[0]
                mac_e_vals.add(e_mac)
                if abs(e_mac - energy) > 1e-9:
                    mac_bad_e += 1
                else:
                    mac_ok += 1

        # ---- 1. 角度全量自检
        angle_bad = 0
        if parsed:
            uniq = sorted(set(parsed))
            dup = [a for a, c in Counter(parsed).items() if c > 1]
            exp = expected_angles(len(parsed))
            diff = [(i, got, want) for i, (got, want) in
                    enumerate(zip(uniq, exp)) if got != want] if len(uniq) == len(exp) else [("len", len(uniq), len(exp))]
            missing_a = [x for x in exp if x not in set(uniq)]
            if dup or diff or missing_a:
                angle_bad = len(dup) + len(diff) + len(missing_a)
                problems.append(f"{d.name}: 角度异常 dup={dup[:5]} diff={diff[:5]} missing={missing_a[:5]}")
        else:
            uniq, angle_bad = [], 1
            problems.append(f"{d.name}: 没有任何可解析的 npy")

        npy_n = len(parsed) + len(bad_name)
        mac_match = len(parsed) - len(missing_mac) - mac_bad_e if args.check_macros else -1
        e_str = ",".join(f"{v:g}" for v in sorted(mac_e_vals)) if mac_e_vals else "-"

        name_state = "OK" if not bad_name and not missing_mac else "FAIL"
        energy_state = "OK" if args.check_macros and mac_bad_e == 0 and mac_e_vals else "FAIL"
        angle_state = "OK" if angle_bad == 0 else "FAIL"

        if args.check_macros:
            if missing_mac:
                problems.append(f"{d.name}: 缺 macro {len(missing_mac)} 个，例: {missing_mac[:5]}")
            if mac_bad_e:
                problems.append(f"{d.name}: macro 能量与目录名不一致 {mac_bad_e} 个（目录={energy:g}MeV, macro={e_str}）")

        print(f"{d.name:<54}{energy:>5g}{npy_n:>5}{len(parsed):>5}"
              f"{cnt_a:>5}{cnt_b:>5}"
              f"{mac_match:>5}{e_str:>7}  {angle_state} {name_state} {energy_state}")

        # 明细（仅异常时展开）
        if missing_mac or mac_bad_e or bad_name:
            blk = [f"  --- {d.name} 明细 ---"]
            blk.append(f"      目录名能量 : {energy:g} MeV")
            blk.append(f"      macro 能量 : {e_str} MeV")
            blk.append(f"      npy 数     : {npy_n} / 解析成功 {len(parsed)}")
            blk.append(f"      macro 匹配 : {mac_match} / {len(parsed)}")
            blk.append(f"      缺失 macro : {len(missing_mac)} -> {missing_mac[:10]}")
            blk.append(f"      能量不一致 : {mac_bad_e}")
            blk.append(f"      命名异常   : {len(bad_name)} -> {bad_name[:10]}")
            detail_blocks.append("\n".join(blk))

        rows.extend(local_rows)
        tot["npy"] += npy_n
        tot["ang"] += len(parsed)
        tot["mac_ok"] += mac_ok
        tot["mac_missing"] += len(missing_mac)
        tot["e_bad"] += mac_bad_e

    print("=" * 118)
    print(f"目录={len({r['scan_dir'] for r in rows})}  npy={tot['npy']}  角度解析={tot['ang']}  "
          f"macro匹配={tot['mac_ok']}  缺macro={tot['mac_missing']}  能量不一致={tot['e_bad']}  "
          f"首轮4档行数={sum(1 for r in rows if float(r['energy_MeV']) in FIRST_RUN_ENERGIES)}")

    if detail_blocks:
        print()
        for b in detail_blocks:
            print(b)

    if problems:
        print(f"\n[BLOCKED] 发现 {len(problems)} 个阻断项，未写 angles.csv：")
        for p in problems[:40]:
            print("  ! " + p)
        if len(problems) > 40:
            print(f"  ... 共 {len(problems)} 条")
        return 1

    if args.dry_run:
        print("\n[DRY-RUN] 全部通过，未写文件。")
        return 0

    out = Path(args.out)
    with open(out, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["energy_MeV", "angle_deg", "scan_dir", "file"])
        w.writeheader()
        w.writerows(rows)
    print(f"\n[OK] 已写出 {out}（{len(rows)} 行）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
