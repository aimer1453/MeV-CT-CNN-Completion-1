#!/usr/bin/env python
"""Row-16-only 消融驱动：单能量 3.0 MeV，32 行对照 vs Row-16-only，各 3 种子。

为什么要对照组（A 组）
----------------------
论文正文的 32 行模型是**跨 4 个能量**（train 1080 样本）训练的，
而本消融在**单能量 3.0 MeV**（train 270 样本）上跑。
若直接拿「单能量 + 单行」去对比「跨能量 + 32 行」的 0.09516，会同时混入：
    (1) 训练数据规模/能量多样性  (2) 输入行数
两个变量，无法归因。因此必须补一组**同能量、同样本数、同 epoch** 的
32 行基线作为对照，使唯一差别退化为「另外 31 行有没有真实信息」。

组定义
------
    A 组 base32_3mev : 单能量 CSV + 正常 32 行输入（对照组）
    B 组 row16_3mev  : 单能量 CSV + --single-row（Row-16-only 消融组）

网络结构、参数量（1,950,017）、batch、lr、epoch、seed 两组完全一致。

用法
----
    python scripts/row16_ablation.py                # 跑全部 6 个 run
    python scripts/row16_ablation.py --group B      # 只跑消融组
"""
from __future__ import annotations

import argparse
import csv
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from data.csv_paths import resolve_csv  # noqa: E402  CSV 统一落点 csv_data/
RUNS = ROOT / "runs"
LOG = RUNS / "_row16_ablation.log"
INDEX = RUNS / "_row16_ablation_index.csv"
PY = sys.executable or r"<user-python>"

SINGLE_ENERGY_CSV = resolve_csv("sparse_targets_3MeV.csv")
SEEDS = [20261022, 20261023, 20261024]
EPOCHS = 100

#: (组名, train.py 附加参数)
GROUPS: list[tuple[str, list[str]]] = [
    ("base32_3mev", []),                  # A 组：32 行对照
    ("row16_3mev", ["--single-row"]),     # B 组：Row-16-only
]


def log(s: str) -> None:
    line = f"[{time.strftime('%m-%d %H:%M:%S')}] {s}"
    with open(LOG, "a", encoding="utf-8") as fh:
        fh.write(line + "\n")
    print(line, flush=True)


def already_done(group: str, seed: int) -> str | None:
    """返回已完成的 run_id；未完成返回 None。"""
    if not INDEX.is_file():
        return None
    for r in csv.DictReader(open(INDEX, newline="", encoding="utf-8")):
        if r["group"] == group and int(r["seed"]) == seed:
            return r["run_id"]
    return None


def main() -> int:
    ap = argparse.ArgumentParser(description="Row-16-only 消融")
    ap.add_argument("--group", choices=["A", "B", "ALL"], default="ALL")
    ap.add_argument("--epochs", type=int, default=EPOCHS)
    args = ap.parse_args()

    if not Path(SINGLE_ENERGY_CSV).is_file():
        log(f"[FATAL] 缺 {SINGLE_ENERGY_CSV}：先跑 "
            f"scripts/make_single_energy_csv.py --energy 3.0")
        return 2

    groups = GROUPS
    if args.group == "A":
        groups = [GROUPS[0]]
    elif args.group == "B":
        groups = [GROUPS[1]]

    INDEX.parent.mkdir(parents=True, exist_ok=True)
    if not INDEX.is_file():
        INDEX.write_text("group,seed,run_id,epochs,finished_at\n", encoding="utf-8")

    t_start = time.time()
    for gname, extra in groups:
        for seed in SEEDS:
            hit = already_done(gname, seed)
            if hit:
                log(f"[SKIP] {gname} seed={seed} 已完成 -> {hit}")
                continue
            cmd = [PY, str(ROOT / "train.py"),
                   "--csv", SINGLE_ENERGY_CSV,
                   "--epochs", str(args.epochs),
                   "--seed", str(seed),
                   "--tag", gname, *extra]
            log(f"[RUN] {gname} seed={seed}  cmd={' '.join(cmd[-6:])}")
            t0 = time.time()
            rc = subprocess.run(cmd, cwd=str(ROOT)).returncode
            dt = time.time() - t0
            if rc != 0:
                log(f"[FATAL] {gname} seed={seed} rc={rc}（耗时 {dt/60:.1f} min）")
                return 3
            log(f"[OK] {gname} seed={seed} 完成，耗时 {dt/60:.1f} min")

            # 从 test_metrics / 目录里找出本次 run_id（最新生成的 runs/<ts>_cnn_*_<tag>）
            cands = sorted(
                [p for p in RUNS.iterdir()
                 if p.is_dir() and p.name.endswith(gname) and (p / "best.pt").is_file()],
                key=lambda p: p.stat().st_mtime,
            )
            if not cands:
                log(f"[WARN] 找不到 {gname} seed={seed} 的 run 目录，无法登记索引")
                continue
            run_id = cands[-1].name
            with open(INDEX, "a", newline="", encoding="utf-8") as fh:
                csv.writer(fh).writerow([gname, seed, run_id, args.epochs,
                                         time.strftime("%Y-%m-%dT%H:%M:%S")])
            log(f"[IDX] {gname} seed={seed} -> {run_id}")

    log(f"全部完成，总耗时 {(time.time() - t_start)/60:.1f} min。"
        f"下一步：scripts/eval_row16_ablation.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
