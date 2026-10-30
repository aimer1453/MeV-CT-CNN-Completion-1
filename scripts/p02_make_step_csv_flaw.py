#!/usr/bin/env python
# -*- coding: utf-8 -*-
r"""
scripts/p02_make_step_csv_flaw.py —— 缺陷数据（边棒 3_3_flaw_1_2）的跨度表生成器

为什么需要
----------
主数据域的跨度扫描（`sparse_targets_s{2,4,6,8}.csv`）已经证明：CNN 在 2° 输、
在 4°/6°/8° 赢，即"工作点（难度）"决定胜负。审稿人会立刻追问——
**这是不是只对均匀数据域成立？** 缺陷数据（2° 下 CNN 0/360 全输）是更硬的靶子，
必须在同一几何协议下再扫一遍跨度，才能说"工作点效应与数据域无关"。

几何（与主域脚本 `p02_make_step_csv.py` **逐字同式**，只换数据源）
------------------------------------------------------------------
半跨 h = s/2：
    ch0     = θ − h      （bracketing 左端点）
    ch_next = θ + h      （bracketing 右端点）
    ch_prev = θ − 3h     （左侧再退一格的额外上下文，**不是**左端点）
线性插值基线恒为 ½·[L(θ−h) + L(θ+h)]。
输入必须全部落在"实测陪集" (c + h) mod s（c = θ mod s），脚本强制校验。

与主域的两个关键对齐（否则两批数据不可比）
------------------------------------------
1. **split 完全沿用** `sparse_targets_flaw12_3.0.csv` 的 270/45/45，不重新打乱。
   已核：该表的 45 个 test 角与主域 `sparse_targets.csv` 的 3.0 MeV test 集**逐角相同**
   （同一 seed 派生 f"{seed}:3"），故跨批次、跨跨度都是同一批 45 个 held-out 视角。
2. **全部 360 个角度都当目标**（按 coset c 分 s 个方向），训练样本数恒为 360，
   与 2° 表相同 → 每个 span 的训练集规模不受影响，比值差异只能归因于"补全难度"。

输出
----
    csv_data/sparse_targets_flaw12_3.0_s{4,6,8}.csv
    （s=2 只做自检，不落盘：必须与 sparse_targets_flaw12_3.0.csv 逐行一致）

用法
----
    python scripts/p02_make_step_csv_flaw.py --dry-run
    python scripts/p02_make_step_csv_flaw.py
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from data.csv_paths import resolve_csv  # noqa: E402

BASE_CSV = resolve_csv("sparse_targets_flaw12_3.0.csv")
ENERGY = "3"

FIELDS = ("energy_MeV", "scan_dir", "target_angle_deg", "target_file", "phase",
          "ch0_angle_deg", "ch0_file", "ch_prev_angle_deg", "ch_prev_file",
          "ch_next_angle_deg", "ch_next_file", "split")


def load_base() -> tuple[dict[int, dict[str, str]], dict[int, str]]:
    """-> (angle -> {file, scan_dir}, angle -> split)。要求恰好 360 行、覆盖 0..359。"""
    if not BASE_CSV.is_file():
        raise SystemExit(f"缺少基准表 {BASE_CSV}")
    tbl: dict[int, dict[str, str]] = {}
    spl: dict[int, str] = {}
    with open(BASE_CSV, newline="", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            if r["energy_MeV"] != ENERGY:
                raise SystemExit(
                    f"{BASE_CSV.name} 中出现非 {ENERGY} MeV 行（{r['energy_MeV']}），"
                    f"本脚本只处理单档缺陷表")
            a = int(round(float(r["target_angle_deg"])))
            tbl[a] = {"file": r["target_file"], "scan_dir": r["scan_dir"]}
            spl[a] = r["split"]
    if sorted(tbl) != list(range(360)):
        raise SystemExit(f"基准表角度未覆盖 0..359（{len(tbl)} 个）")
    if not all(str(v["file"]).startswith("3_3_flaw_1_2/") for v in tbl.values()):
        raise SystemExit("target_file 未指向 3_3_flaw_1_2/，数据源不对")
    return tbl, spl


def build(step: int, tbl: dict, spl: dict) -> list[dict]:
    if step % 2 != 0:
        raise SystemExit(f"step 必须为偶数，收到 {step}")
    h = step // 2
    rows: list[dict] = []
    for theta in range(360):
        c = theta % step
        rec: dict[str, str] = {
            "energy_MeV": ENERGY,
            "scan_dir": tbl[theta]["scan_dir"],
            "target_angle_deg": str(theta),
            "target_file": tbl[theta]["file"],
            "phase": f"C{c}",
            "split": spl[theta],
        }
        for key, off in (("ch0", -h), ("ch_prev", -3 * h), ("ch_next", +h)):
            a = (theta + off) % 360
            rec[f"{key}_angle_deg"] = str(a)
            rec[f"{key}_file"] = tbl[a]["file"]
            if a % step != (c + h) % step:
                raise SystemExit(
                    f"[s={step}] θ={theta} 的 {key}={a} 落在陪集 {a % step}，"
                    f"期望 {(c + h) % step}")
        rows.append(rec)
    rows.sort(key=lambda r: int(r["target_angle_deg"]))
    return rows


def verify_s2(rows: list[dict], tbl: dict, spl: dict) -> list[str]:
    """s=2 必须精确复现基准表（角度 + 文件 + split 全等）。"""
    bad: list[str] = []
    for r in rows:
        t = int(r["target_angle_deg"])
        if r["target_file"] != tbl[t]["file"]:
            bad.append(f"target {t} 文件不符")
        if r["split"] != spl[t]:
            bad.append(f"target {t} split 不符")
        for key, off in (("ch0", -1), ("ch_prev", -3), ("ch_next", +1)):
            if int(r[f"{key}_angle_deg"]) != (t + off) % 360:
                bad.append(f"target {t} {key} 角度不符")
            if r[f"{key}_file"] != tbl[(t + off) % 360]["file"]:
                bad.append(f"target {t} {key} 文件不符")
    return bad


def write(rows: list[dict], path: Path) -> None:
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(FIELDS))
        w.writeheader()
        w.writerows(rows)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="缺陷数据（边棒 3.0 MeV）跨度表生成")
    ap.add_argument("--steps", default="2,4,6,8")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args(argv)

    tbl, spl = load_base()
    n_test = sum(1 for a in spl if spl[a] == "test")
    print(f"基准表 {BASE_CSV.name}: 360 角, test={n_test} 角 "
          f"{sorted(a for a in spl if spl[a] == 'test')[:8]}…")

    fails: list[str] = []
    for s in (int(x) for x in args.steps.split(",")):
        rows = build(s, tbl, spl)
        n_tr = sum(1 for r in rows if r["split"] == "train")
        n_va = sum(1 for r in rows if r["split"] == "val")
        n_te = sum(1 for r in rows if r["split"] == "test")
        cosets = sorted({int(r["phase"][1:]) for r in rows})
        # 自检：split 计数、test 集合、coset 完整性、文件存在性抽样
        if (n_tr, n_va, n_te) != (270, 45, 45):
            fails.append(f"[s={s}] split 计数 {(n_tr, n_va, n_te)} != (270,45,45)")
        te_new = {int(r["target_angle_deg"]) for r in rows if r["split"] == "test"}
        te_old = {a for a in spl if spl[a] == "test"}
        if te_new != te_old:
            fails.append(f"[s={s}] test 角集合与基准表不一致")
        if cosets != list(range(s)):
            fails.append(f"[s={s}] coset 不完整 {cosets}")
        for r in rows[:40]:
            for k in ("target_file", "ch0_file", "ch_prev_file", "ch_next_file"):
                if not (ROOT / r[k]).is_file():
                    fails.append(f"[s={s}] 文件不存在 {r[k]}")
        if s == 2:
            bad = verify_s2(rows, tbl, spl)
            if bad:
                fails.append(f"[s=2] 未能复现基准表，{len(bad)} 处不符：{bad[:5]}")
            else:
                print("  [s=2] 自检：与基准表逐行一致 ✓（不落盘）")
        if s != 2:
            path = resolve_csv(f"sparse_targets_flaw12_3.0_s{s}.csv", for_write=True)
            if args.dry_run:
                print(f"  [s={s}] DRY-RUN 将写 {path.name}  h={s // 2}°  "
                      f"实测 {360 // s} 角 / 补全 {360 - 360 // s} 角  "
                      f"split={n_tr}/{n_va}/{n_te}")
            else:
                write(rows, path)
                print(f"  [s={s}] 已写 {path.name}  h={s // 2}°  "
                      f"实测 {360 // s} 角 / 补全 {360 - 360 // s} 角  "
                      f"split={n_tr}/{n_va}/{n_te}")

    if fails:
        print(f"\n[BLOCKED] {len(fails)} 项自检失败：")
        for f in fails:
            print("  ! " + f)
        return 1
    print("\n[OK] 全部自检通过")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
