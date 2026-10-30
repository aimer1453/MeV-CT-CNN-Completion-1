"""生成 3_3_flaw/ 缺陷数据的 sparse_targets CSV（每档 360 行）。

默认仍是 3.0 MeV 单档（`fuel_ct_scan_z0.0_20260804_195102`），保持与历史产物逐位一致。
多能量扩展（2026-10-26）：新增 `--scan-dir` / `--energy` / `--out`，为
`energy_scan_{1.5,3.0,4.5,6.0}MeV_*` 各生成一份 `sparse_targets_flaw_<E>.csv`，
口径完全不变（角度公式、相位、split 规则仍从 build_sparse_targets import）。

口径与根目录 sparse_targets.csv **完全一致**（直接 import build_sparse_targets 的
phase_of / triple / make_splits / FIELDS，不复制不重写）：
  - 角度公式：ch0=(t-1)%360, ch_next=(t+1)%360, ch_prev=(t-3)%360
  - 相位：A=偶数角 / B=奇数角，phase 列 = target 相位
  - split：seed=20261022，子种子 f"{seed}:3"（与 3_3/ 的 3.0 MeV 档划分一致），
    打乱 0..359 后 270/45/45；**不分组**（缺陷非对称，无 90° 等价重复）

与主数据的唯一差别：文件名是六位浮点（proj_a0000.0_z+0.npy），
file 列带 `3_3_flaw/` 前缀；scan_dir 列仍只写目录名（dataset.py 从 file 反推）。

用法：
  conda run -n ct_full python scripts/build_sparse_targets_flaw.py [--dry-run]
输出：sparse_targets_flaw.csv（仓库根）
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from data.csv_paths import resolve_csv  # noqa: E402  CSV 统一落点 csv_data/
sys.path.insert(0, str(ROOT / "scripts"))

import build_sparse_targets as bst  # noqa: E402  同一套口径，禁止复制公式

FLAW_ROOT_NAME = "3_3_flaw"
FLAW_ROOT = ROOT / FLAW_ROOT_NAME
SCAN_DIR = "fuel_ct_scan_z0.0_20260804_195102"
ENERGY = 3.0
OUT = resolve_csv("sparse_targets_flaw.csv", for_write=True)
SPLIT_SIZES = bst.SPLIT_SIZES


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--out", default=str(OUT))
    ap.add_argument("--seed", type=int, default=20261022)
    ap.add_argument("--scan-dir", default=SCAN_DIR,
                    help="3_3_flaw/ 下的扫描目录名；多能量时传 energy_scan_<E>MeV_*")
    ap.add_argument("--energy", type=float, default=ENERGY,
                    help="能量档（MeV），同时决定 split 子种子 f\"{seed}:{energy}\"")
    ap.add_argument("--flaw-root", default=FLAW_ROOT_NAME,
                    help="数据根目录名（默认 3_3_flaw；带包壳的新数据为 3_3_flaw_1_1）")
    args = ap.parse_args()

    flaw_root = ROOT / args.flaw_root
    if not flaw_root.is_dir():
        print(f"[FATAL] 缺数据根目录 {flaw_root}", file=sys.stderr)
        return 2
    scan_dir = args.scan_dir
    energy = args.energy
    proj_dir = flaw_root / scan_dir / "projections"
    if not proj_dir.is_dir():
        print(f"[FATAL] 缺目录 {proj_dir}", file=sys.stderr)
        return 2

    # 浮点命名 -> {angle:int: relpath}
    amap: dict[int, str] = {}
    for f in sorted(proj_dir.glob("proj_a*_z+0.npy")):
        a = int(round(float(f.name.split("proj_a")[1].split("_z")[0])))
        amap[a] = f"{args.flaw_root}/{scan_dir}/projections/{f.name}"
    if sorted(amap) != list(range(360)):
        print(f"[FATAL] 角度不是 0..359：{len(amap)} 个", file=sys.stderr)
        return 2

    smap = bst.make_splits(args.seed, energy)
    rows = []
    for t in range(360):
        pv, c0, nx = bst.triple(t)
        rows.append({
            "energy_MeV": f"{energy:g}",
            "scan_dir": scan_dir,
            "target_angle_deg": t,
            "target_file": amap[t],
            "phase": bst.phase_of(t),
            "ch0_angle_deg": c0, "ch0_file": amap[c0],
            "ch_prev_angle_deg": pv, "ch_prev_file": amap[pv],
            "ch_next_angle_deg": nx, "ch_next_file": amap[nx],
            "split": smap[t],
        })

    # -------------------------------------------------- 自检
    fails = []
    if len(rows) != 360:
        fails.append(f"行数 {len(rows)} != 360")
    if sorted(r["target_angle_deg"] for r in rows) != list(range(360)):
        fails.append("target 未覆盖 0..359")
    for t, (ep, e0, en) in bst.FIXED_CASES.items():
        r = rows[t]
        got = (r["ch_prev_angle_deg"], r["ch0_angle_deg"], r["ch_next_angle_deg"])
        if got != (ep, e0, en):
            fails.append(f"target={t} 期望 {(ep, e0, en)} 实得 {got}")
    for r in rows:
        t = r["target_angle_deg"]
        if (t - r["ch0_angle_deg"]) % 360 != 1:
            fails.append(f"target={t} ch0 距离 != 1")
        if bst.phase_of(r["ch0_angle_deg"]) == bst.phase_of(t):
            fails.append(f"target={t} 输入与 target 同相位")
        for k in ("target_file", "ch0_file", "ch_prev_file", "ch_next_file"):
            rel = r[k]
            if f"{scan_dir}/projections/" not in rel:
                fails.append(f"target={t} {k} 不在 scan_dir 下: {rel}")
            if not (ROOT / rel).is_file():
                fails.append(f"文件不存在: {rel}")
    from collections import Counter
    cnt = Counter(r["split"] for r in rows)
    for name, want in SPLIT_SIZES:
        if cnt[name] != want:
            fails.append(f"split {name} {cnt[name]} != {want}")

    print("=" * 76)
    print(f"flaw 数据表：{len(rows)} 行，scan_dir={scan_dir}，E={energy:g} MeV")
    print(f"split: train {cnt['train']} / val {cnt['val']} / test {cnt['test']}")
    print(f"phase: A {sum(1 for r in rows if r['phase']=='A')} / "
          f"B {sum(1 for r in rows if r['phase']=='B')}")
    print("前 3 行:")
    for r in rows[:3]:
        print(f"  t={r['target_angle_deg']:3d} {r['split']:5s} ph={r['phase']} "
              f"prev={r['ch_prev_angle_deg']:3d} ch0={r['ch0_angle_deg']:3d} "
              f"next={r['ch_next_angle_deg']:3d}")
        print(f"      target={r['target_file'].split('/')[-1]}")
    if fails:
        print(f"\n[BLOCKED] {len(fails)} 项失败:")
        for f in fails[:15]:
            print("  ! " + f)
        return 1
    print("\n[OK] 自检通过")

    if args.dry_run:
        print("[DRY-RUN] 未写文件")
        return 0
    # 写出路径也过解析器：命令行传裸文件名（如 `--out sparse_targets_flaw11_3.0.csv`）
    # 时落到 csv_data/，不再漏到仓库根（2026-10-27 的 flaw11 四档就是这样漏出去的）。
    out_path = resolve_csv(args.out, for_write=True)
    with open(out_path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=bst.FIELDS)
        w.writeheader()
        w.writerows(rows)
    print(f"已写出 {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
