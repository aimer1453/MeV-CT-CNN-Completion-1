#!/usr/bin/env python
# -*- coding: utf-8 -*-
r"""
scripts/run_flaw_span_sweep.py —— 缺陷数据跨度扫描的**并行**训练启动器

任务
----
在边棒缺陷（3_3_flaw_1_2，3.0 MeV）上扫 s = 4/6/8°，每档 3 个种子、100 epoch，
回答"2° 下 CNN 全输，是不是因为工作点太简单（线性插值近乎无偏）"。
s=2 的基准点**不需要重训**：现有 `runs/*_flaw12_3.0` 三个种子就是 s=2（同几何、
同 split），只需在 45-test 口径下重新评估（见 `p02_fault_span_eval.py`）。

为什么要并行
------------
单 run（单档 270 训练样本 × 100 epoch）约 12 min，9 个串行 ~110 min；
实测训练期 GPU 利用率仅 19–40%、功耗 20 W（显存 1.3/8 GB），瓶颈在 CPU 侧
逐样本 npy 载入，串行是纯浪费。这里用 k 个 worker 并发，每个 worker 限 2 个
OMP 线程（train.py 未调 set_num_threads，默认 16 线程并发会互相踩）。

run_id 登记（R34c 两次事故的教训）
----------------------------------
**绝不**用"训练结束后字典序最大的新目录"定位产出——并行时会被别的 run 抢走。
本脚本在子进程退出后，只认 `config.json` 里 (数据表名, seed) 与本任务精确匹配的
新目录；命中多个直接报错，一个不中回退并显式告警。

用法
----
    python scripts/run_flaw_span_sweep.py --dry-run
    python scripts/run_flaw_span_sweep.py --workers 3
    python scripts/run_flaw_span_sweep.py --workers 3 --only 4   # 只补某一跨度
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from data.csv_paths import resolve_csv  # noqa: E402

RUNS = ROOT / "runs"
PY = sys.executable
DEFAULT_STEPS = (4, 6, 8)
DEFAULT_SEEDS = (20261022, 20261023, 20261024)
CONFIG = "config_flaw12_3.0.json"
TAG_PREFIX = "flawspan"
FIELDS = ("step", "seed", "csv", "run_id", "minutes")


def csv_name(step: int) -> str:
    if step == 2:                       # s=2 就是基准表本身，没有 _s2 副本
        return "sparse_targets_flaw12_3.0.csv"
    return f"sparse_targets_flaw12_3.0_s{step}.csv"


def read_index(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    with open(path, encoding="utf-8") as fh:
        return [r for r in csv.DictReader(fh) if (r.get("run_id") or "").strip()]


def append_index(path: Path, step: int, seed: int, cname: str, run_id: str,
                 minutes: float) -> None:
    new = not path.is_file()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS)
        if new:
            w.writeheader()
        w.writerow({"step": step, "seed": seed, "csv": cname,
                    "run_id": run_id, "minutes": f"{minutes:.1f}"})


def run_dirs() -> set[str]:
    return {p.name for p in RUNS.iterdir()
            if p.is_dir() and (p / "config.json").is_file()}


def find_run_id(before: set[str], cname: str, seed: int, views: int = 3) -> str | None:
    """在新出现的 run 目录里按 (数据表, seed, 输入视角数) 精确匹配；多个/零个都报错。

    必须带 `views`：两帧消融与三帧正文用的是**同一张表、同一个种子**，只靠
    (csv, seed) 会命中两个目录（R34c 两次抢注事故就是定位键不够严）。
    """
    hits: list[str] = []
    for name in sorted(run_dirs() - before):
        cp = RUNS / name / "config.json"
        try:
            c = json.loads(cp.read_text(encoding="utf-8"))
        except Exception:
            continue
        tgt = Path(str(c.get("data", {}).get("csv")
                       or c.get("data", {}).get("targets") or "")).name
        try:
            sd = int(c.get("seed"))
            nv = int(c.get("input_views", 3))
        except (TypeError, ValueError):
            continue
        if tgt == cname and sd == seed and nv == views:
            hits.append(name)
    if len(hits) == 1:
        return hits[0]
    if len(hits) > 1:
        print(f"   [FATAL] ({cname}, seed={seed}, views={views}) 命中多个 run：{hits}",
              flush=True)
    else:
        print(f"   [FATAL] ({cname}, seed={seed}, views={views}) 未找到产出目录", flush=True)
    return None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", default=",".join(str(s) for s in DEFAULT_STEPS))
    ap.add_argument("--seeds", default=",".join(str(s) for s in DEFAULT_SEEDS))
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--threads", type=int, default=2,
                    help="每个 worker 的 OMP/MKL 线程数（默认 2，防 16 线程互踩）")
    ap.add_argument("--index", default="runs/_flaw12_span_index.csv")
    ap.add_argument("--only", default=None, help="只跑某一个跨度 s")
    ap.add_argument("--input-views", type=int, default=3, choices=(2, 3),
                    help="输入视角数：3=正文配置（θ−3h,θ−h,θ+h）；2=消融（去掉 θ−3h）")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    steps = [int(args.only)] if args.only else [int(x) for x in args.steps.split(",")]
    seeds = [int(x) for x in args.seeds.split(",")]
    views = int(args.input_views)
    index = Path(args.index)
    if not index.is_absolute():
        index = ROOT / index

    done = {(int(r["step"]), int(r["seed"])) for r in read_index(index)}
    plan = [(s, sd) for s in steps for sd in seeds if (s, sd) not in done]

    print(f"索引 {index.name}: 已登记 {len(done)} 个；待训练 {len(plan)} 个 "
          f"(workers={args.workers}, epochs={args.epochs})", flush=True)
    for s, sd in plan:
        p = resolve_csv(csv_name(s))
        flag = "OK" if p.is_file() else "MISSING"
        print(f"   s={s}° seed={sd}  {p.name}  [{flag}]", flush=True)
    if not plan:
        print("[DONE] 无待训练项", flush=True)
        return 0
    if args.dry_run:
        return 0

    # ---- worker：一个任务 → 一个子进程
    def launch(s: int, sd: int) -> tuple[subprocess.Popen, Path, set[str], float]:
        cname = csv_name(s)
        suffix = "" if views == 3 else f"_v{views}"
        log = RUNS / f"_span_{TAG_PREFIX}{suffix}_s{s}_{sd}.log"
        cmd = [PY, "-u", str(ROOT / "train.py"),
               "--config", CONFIG, "--csv", cname,
               "--seed", str(sd), "--epochs", str(args.epochs),
               "--batch", str(args.batch),
               "--tag", f"{TAG_PREFIX}{suffix}_s{s}_s{sd}"]
        if views != 3:
            cmd += ["--input-views", str(views)]
        env = {**os.environ, "PYTHONIOENCODING": "utf-8",
               "OMP_NUM_THREADS": str(args.threads),
               "MKL_NUM_THREADS": str(args.threads),
               "OPENBLAS_NUM_THREADS": str(args.threads)}
        before = run_dirs()
        fh = log.open("w", encoding="utf-8")
        return subprocess.Popen(cmd, cwd=str(ROOT), stdout=fh,
                                stderr=subprocess.STDOUT, env=env), log, before, time.time()
        # 注意：fh 不关闭 → 子进程退出时随进程结束回收；日志按行 flush（-u）

    running: list[tuple[int, int, subprocess.Popen, Path, set[str], float]] = []
    pending = list(plan)
    t_all = time.time()
    failed: list[str] = []

    while pending or running:
        while pending and len(running) < args.workers:
            s, sd = pending.pop(0)
            try:
                proc, log, before, t0 = launch(s, sd)
            except FileNotFoundError as exc:
                print(f"[FATAL] 启动失败 {exc}", flush=True)
                return 2
            running.append((s, sd, proc, log, before, t0))
            print(f"[start] s={s}° seed={sd}  -> {log.name}", flush=True)

        time.sleep(5)
        still = []
        for s, sd, proc, log, before, t0 in running:
            if proc.poll() is None:
                still.append((s, sd, proc, log, before, t0))
                continue
            dt = (time.time() - t0) / 60
            cname = csv_name(s)
            rid = find_run_id(before, cname, sd, views) if proc.returncode == 0 else None
            ok = proc.returncode == 0 and rid and (RUNS / rid / "test_metrics.json").is_file()
            if ok:
                print(f"[done ] s={s}° seed={sd}  {dt:.1f} min  {rid}", flush=True)
                append_index(index, s, sd, cname, rid, dt)
            else:
                tail = log.read_text(encoding="utf-8", errors="replace").splitlines()[-12:]
                print(f"[FAIL ] s={s}° seed={sd}  rc={proc.returncode}  {dt:.1f} min",
                      flush=True)
                print("   " + "\n   ".join(tail), flush=True)
                failed.append(f"s={s} seg={sd}")
        running = still

    print(f"\n[{'DONE' if not failed else 'PARTIAL'}] 耗时 {(time.time() - t_all) / 60:.1f} min，"
          f"失败 {len(failed)} 个 {failed}", flush=True)
    return 0 if not failed else 1


if __name__ == "__main__":
    raise SystemExit(main())
