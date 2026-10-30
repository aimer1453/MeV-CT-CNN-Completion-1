#!/usr/bin/env python
# -*- coding: utf-8 -*-
r"""
scripts/run_uniform_span_sweep.py —— **均匀域**跨度扫描的并行训练启动器（s = 10° / 12°）

为什么需要
----------
缺陷域（`3_3_flaw_1_2`）的跨度扫描已扩到 s = 2/4/6/8/10/12 六档；均匀域
（`3_3` 主域）只到 s = 8°，四档。论文侧要做 2 行（周期 / 非周期）× 6 列（跨度）
的制度格，均匀域必须补齐 10° / 12°，否则右栏两行不齐。

与缺陷域启动器的关系
--------------------
`scripts/run_flaw_span_sweep.py` 是它的缺陷域孪生版：逻辑（并行、run_id 按
(config.csv, seed, input_views) 三元组精确登记、失败不清索引）完全一致，
**只换三个常量**：config 文件、CSV 命名、tag 前缀与索引文件。
之所以复制而不加参数，是为了不动已验证的缺陷域队列脚本。

口径
----
* config：`config.json`（主域默认，4 档能量 1.5/3.0/4.5/6.0）
* 数据表：`sparse_targets_s<s>.csv`（由 `scripts/p02_make_step_csv.py --steps 2,10,12` 生成，
  自检要求 s=2 复算与 `sparse_targets.csv` 逐行一致）
* 种子：20261022 / 20261023 / 20261024，与既有 2°/4°/6°/8° 基准**同种子**，
  这样六档可以逐种子配对
* 100 epoch，batch 8，lr 1e-4，输入 3 视角（正文配置）

用法
----
    python scripts/run_uniform_span_sweep.py --dry-run
    python scripts/run_uniform_span_sweep.py --steps 10,12 --workers 3
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
DEFAULT_STEPS = (10, 12)
DEFAULT_SEEDS = (20261022, 20261023, 20261024)
CONFIG = "config.json"
TAG_PREFIX = "uspan"
FIELDS = ("step", "seed", "csv", "run_id", "minutes")


def csv_name(step: int) -> str:
    if step == 2:
        return "sparse_targets.csv"      # s=2 就是主域基准表本身
    return f"sparse_targets_s{step}.csv"


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
    """按 (数据表, seed, 输入视角数) 精确登记；命中多个或零个都报错。"""
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
    ap.add_argument("--index", default="runs/_uniform_span_index.csv")
    ap.add_argument("--only", default=None, help="只跑某一个跨度 s")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    steps = [int(args.only)] if args.only else [int(x) for x in args.steps.split(",")]
    seeds = [int(x) for x in args.seeds.split(",")]
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

    def launch(s: int, sd: int) -> tuple[subprocess.Popen, Path, set[str], float]:
        cname = csv_name(s)
        log = RUNS / f"_span_{TAG_PREFIX}_s{s}_{sd}.log"
        cmd = [PY, "-u", str(ROOT / "train.py"),
               "--config", CONFIG, "--csv", cname,
               "--seed", str(sd), "--epochs", str(args.epochs),
               "--batch", str(args.batch),
               "--tag", f"{TAG_PREFIX}_s{s}_s{sd}"]
        env = {**os.environ, "PYTHONIOENCODING": "utf-8",
               "OMP_NUM_THREADS": str(args.threads),
               "MKL_NUM_THREADS": str(args.threads),
               "OPENBLAS_NUM_THREADS": str(args.threads)}
        before = run_dirs()
        fh = log.open("w", encoding="utf-8")
        return subprocess.Popen(cmd, cwd=str(ROOT), stdout=fh,
                                stderr=subprocess.STDOUT, env=env), log, before, time.time()

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
            rid = find_run_id(before, cname, sd) if proc.returncode == 0 else None
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
