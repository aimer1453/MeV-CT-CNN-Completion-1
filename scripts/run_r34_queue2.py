#!/usr/bin/env python
# -*- coding: utf-8 -*-
r"""
scripts/run_r34_queue2.py —— R34 剩余实验队列（可断点续跑）

相对 run_r34_queue.py 的两点改进（原队列被中断过两次）：
  1. **断点续跑**：按 (csv, seed, kind) 判定该 run 是否已产出 test_metrics.json，
     已完成的直接 skip，不会重复训练、不会污染 runs/。
  2. **分批选择**：--only sweep / contig / grouped，便于单独重跑某一批。

任务
----
  sweep  : ② span sweep 补种子 —— s=2/4/6/8 各补 seed 20261025/20261026
           （train.py + csv_data/sparse_targets_s<s>.csv，tag=step<s>_s<ss>）
  contig : ①-对照 等样本量连续划分 —— train 0-270 / val 270-315 / test 315-360
           （与随机划分 train270/val45/test45 样本数完全相同，唯一差异是"连续 vs 随机"）
  grouped: ① 主实验（已有结果，默认 skip；--force 才重跑）

用法
----
    python scripts/run_r34_queue2.py --only sweep --max-conc 4
    python scripts/run_r34_queue2.py --only contig --max-conc 3
    python scripts/run_r34_queue2.py --only sweep,contig --max-conc 4
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PY = str(ROOT / "venv" / "Scripts" / "python.exe")
RUNS = ROOT / "runs"

GROUPED_CSV = "csv_data/sparse_targets_3MeV_grouped.csv"
CONTIG_CSV = "csv_data/sparse_targets_3MeV_contig270.csv"
NEAR_CSV = "csv_data/sparse_targets_3MeV_near45.csv"
RAND90_CSV = "csv_data/sparse_targets_3MeV_rand90.csv"
EVEN90_CSV = "csv_data/sparse_targets_3MeV_even90.csv"
SEEDS_1 = [20261022, 20261023, 20261024]
SEEDS_2 = [20261025, 20261026]
SPANS = [2, 4, 6, 8]


def _existing_index() -> set[tuple[str, int]]:
    """扫描 runs/，返回已完成 run 的 (csv_basename, seed) 集合（有 test_metrics.json 才算完成）。"""
    done: set[tuple[str, int]] = set()
    if not RUNS.is_dir():
        return done
    for d in RUNS.iterdir():
        if not d.is_dir() or d.name.startswith("_"):
            continue
        cfg_p, met_p = d / "config.json", d / "test_metrics.json"
        if not (cfg_p.is_file() and met_p.is_file()):
            continue
        try:
            cfg = json.loads(cfg_p.read_text(encoding="utf-8"))
            csv_v = str(cfg.get("data", {}).get("csv", ""))
            seed_v = int(cfg.get("seed"))
        except Exception:
            continue
        done.add((Path(csv_v).name, seed_v))
    return done


def build_tasks(only: set[str], spans: list[int],
                contig_kinds: set[str], rand90_kinds: set[str]) -> list[dict]:
    tasks: list[dict] = []
    if "grouped" in only:
        for s in SEEDS_1:
            tasks.append(dict(name=f"grp1d_s{s}", kind="row1d", seed=s,
                              csv=GROUPED_CSV))
            tasks.append(dict(name=f"grp2d_s{s}", kind="full", seed=s,
                              csv=GROUPED_CSV, tag=f"grouped_s{str(s)[-2:]}"))
    if "sweep" in only:
        for sp in spans:
            for s in SEEDS_2:
                tasks.append(dict(name=f"sweep_s{sp}_seed{s}", kind="full", seed=s,
                                  csv=f"csv_data/sparse_targets_s{sp}.csv",
                                  tag=f"step{sp}_s{str(s)[-2:]}"))
    if "near" in only:
        # 单变量隔离：训练量与 grouped 相同（90 行），test 紧邻训练末端（90-135°）
        for s in SEEDS_1:
            tasks.append(dict(name=f"near45_1d_s{s}", kind="row1d", seed=s,
                              csv=NEAR_CSV))
    if "rand90" in only:
        # 任务 2 控制：训练量 90（同 near45），但训练角随机散布全圆（非同扇区）
        for s in SEEDS_1:
            tasks.append(dict(name=f"rand90_1d_s{s}", kind="row1d", seed=s,
                              csv=RAND90_CSV))
            if "2d" in rand90_kinds:
                tasks.append(dict(name=f"rand90_2d_s{s}", kind="full", seed=s,
                                  csv=RAND90_CSV, tag=f"rand90_s{str(s)[-2:]}"))
    if "even90" in only:
        # 任务 2 加强版：训练量 90，训练角**均匀铺满**全圆（排除 test 块）
        for s in SEEDS_1:
            tasks.append(dict(name=f"even90_1d_s{s}", kind="row1d", seed=s,
                              csv=EVEN90_CSV))
    if "contig" in only:
        for s in SEEDS_1:
            if "1d" in contig_kinds:
                tasks.append(dict(name=f"contig270_1d_s{s}", kind="row1d", seed=s,
                                  csv=CONTIG_CSV))
            if "2d" in contig_kinds:
                tasks.append(dict(name=f"contig270_2d_s{s}", kind="full", seed=s,
                                  csv=CONTIG_CSV, tag=f"contig270_s{str(s)[-2:]}"))
    return tasks


def cmd_of(t: dict) -> list[str]:
    if t["kind"] == "row1d":
        return [PY, "scripts/train_row1d.py", "--seed", str(t["seed"]),
                "--epochs", "100", "--csv", t["csv"]]
    c = [PY, "train.py", "--csv", t["csv"], "--seed", str(t["seed"]),
         "--epochs", "100"]
    if t.get("tag"):
        c += ["--tag", t["tag"]]
    return c


def run_one(t: dict) -> tuple[str, int, float]:
    log = RUNS / f"_r34b_{t['name']}.log"
    t0 = time.time()
    with open(log, "w", encoding="utf-8") as fh:
        p = subprocess.run(cmd_of(t), cwd=str(ROOT), stdout=fh,
                           stderr=subprocess.STDOUT)
    return t["name"], p.returncode, time.time() - t0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default="sweep,contig")
    ap.add_argument("--spans", default="2,4,6,8",
                    help="只跑指定跨度（如 6,8）。sweep 全部 8 个 run 约 2.2h/批，按需裁剪。")
    ap.add_argument("--contig-kinds", default="1d,2d",
                    help="contig 对照跑哪些模型：1d / 2d / 1d,2d")
    ap.add_argument("--rand90-kinds", default="1d",
                    help="rand90 控制跑哪些模型：1d / 2d / 1d,2d")
    ap.add_argument("--max-conc", type=int, default=4)
    ap.add_argument("--force", action="store_true", help="忽略断点续跑，全部重跑")
    args = ap.parse_args()

    only = {x.strip() for x in args.only.split(",") if x.strip()}
    spans = [int(x) for x in args.spans.split(",") if x.strip()]
    contig_kinds = {x.strip() for x in args.contig_kinds.split(",") if x.strip()}
    rand90_kinds = {x.strip() for x in args.rand90_kinds.split(",") if x.strip()}
    all_tasks = build_tasks(only, spans, contig_kinds, rand90_kinds)
    done_idx = set() if args.force else _existing_index()

    tasks, skipped = [], []
    for t in all_tasks:
        key = (Path(t["csv"]).name, t["seed"])
        (skipped if key in done_idx else tasks).append(t)

    print(f"[queue2] only={sorted(only)} total={len(all_tasks)} "
          f"skip={len(skipped)} run={len(tasks)} conc={args.max_conc}", flush=True)
    if skipped:
        print("[queue2] 已完成跳过: " + ", ".join(t["name"] for t in skipped), flush=True)
    if not tasks:
        print("[queue2] 无可执行任务", flush=True)
        return 0
    print("[queue2] 待跑: " + ", ".join(t["name"] for t in tasks), flush=True)

    ok = fail = 0
    with ThreadPoolExecutor(max_workers=args.max_conc) as ex:
        futs = {ex.submit(run_one, t): t["name"] for t in tasks}
        for i, fu in enumerate(as_completed(futs), 1):
            name, rc, dt = fu.result()
            if rc == 0:
                ok += 1
            else:
                fail += 1
            print(f"[{i}/{len(tasks)}] {'OK  ' if rc == 0 else 'FAIL'} {name} "
                  f"rc={rc}  {dt/60:.1f} min", flush=True)
    print(f"[queue2] DONE ok={ok} fail={fail}", flush=True)
    return 0 if fail == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
