#!/usr/bin/env python
"""P0-1 看守流水线：串行跑 剂量 × 种子 的 100-epoch 训练，落索引。

每个 run 的命令等价于：

    python train.py --config config_dose_f<f>.json \
                    --csv sparse_targets_dose_f<f>.csv \
                    --epochs 100 --seed <seed> --tag dose<f>

串行（单卡 8 GB，避免显存争用）。run_id 由 train.py 生成，本脚本从 stdout 解析后
写入 runs/_dose_index.csv，供后续 `scripts/dose_eval.py` 汇总。

用法
----
    python scripts/dose_pipeline.py --factors 0.5 0.25 0.125 \
        --seeds 20261022 20261023 20261024
"""
from __future__ import annotations

import argparse
import csv
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from data.csv_paths import resolve_csv  # noqa: E402  CSV 统一落点 csv_data/
RUNS = ROOT / "runs"
INDEX = RUNS / "_dose_index.csv"
PY = sys.executable


def done_runs() -> dict[tuple[str, str], str]:
    """已有索引：(f, seed) -> run_id"""
    if not INDEX.is_file():
        return {}
    out = {}
    for r in csv.DictReader(open(INDEX, newline="", encoding="utf-8")):
        out[(r["factor"], r["seed"])] = r["run_id"]
    return out


def run_one(f: float, seed: int, epochs: int) -> str:
    cfg = ROOT / f"config_dose_f{f:g}.json"
    csvp = resolve_csv(f"sparse_targets_dose_f{f:g}.csv")
    for p in (cfg, csvp):
        if not p.is_file():
            raise SystemExit(f"[FATAL] 缺 {p}")
    tag = f"dose{f:g}"
    log = RUNS / f"_dose_f{f:g}_s{seed}.log"
    cmd = [PY, str(ROOT / "train.py"), "--config", str(cfg), "--csv", str(csvp),
           "--epochs", str(epochs), "--seed", str(seed), "--tag", tag]
    print(f"\n>>> f={f:g} seed={seed}  {datetime.now():%H:%M:%S}")
    print("    " + " ".join(cmd[1:]), flush=True)
    t0 = time.time()
    with open(log, "w", encoding="utf-8", errors="replace") as fh:
        proc = subprocess.Popen(cmd, cwd=str(ROOT), stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT, text=True,
                                encoding="utf-8", errors="replace", bufsize=1)
        run_id = None
        for line in proc.stdout:
            fh.write(line)
            fh.flush()
            if "run_id = " in line and run_id is None:
                run_id = line.split("run_id = ", 1)[1].strip()
                print(f"    run_id = {run_id}", flush=True)
            elif line.startswith(("[epoch", "       val", "  [test", "[done]")):
                print("    " + line.rstrip(), flush=True)
        rc = proc.wait()
    if rc != 0:
        raise SystemExit(f"[FATAL] f={f:g} seed={seed} 退出码 {rc}，见 {log}")
    if not run_id:
        raise SystemExit(f"[FATAL] 没解析到 run_id，见 {log}")
    print(f"    完成 {run_id}  用时 {(time.time() - t0) / 60:.1f} min", flush=True)
    return run_id


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--factors", type=float, nargs="+", default=[0.5, 0.25, 0.125])
    ap.add_argument("--seeds", type=int, nargs="+", default=[20261022, 20261023, 20261024])
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--skip-existing", action="store_true", default=True)
    args = ap.parse_args()

    have = done_runs()
    new = not INDEX.is_file()
    INDEX.parent.mkdir(parents=True, exist_ok=True)
    with open(INDEX, "a", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        if new:
            w.writerow(["factor", "seed", "run_id", "epochs", "finished_at"])

    total = len(args.factors) * len(args.seeds)
    k = 0
    t_all = time.time()
    for f in args.factors:
        for s in args.seeds:
            k += 1
            key = (f"{f:g}", str(s))
            if args.skip_existing and key in have:
                print(f"[{k}/{total}] f={f:g} seed={s} 已有 {have[key]}，跳过")
                continue
            rid = run_one(f, s, args.epochs)
            with open(INDEX, "a", newline="", encoding="utf-8") as fh:
                csv.writer(fh).writerow([f"{f:g}", s, rid, args.epochs,
                                         datetime.now().isoformat(timespec="seconds")])
    print(f"\n全部完成，总用时 {(time.time() - t_all) / 60:.1f} min")
    print(f"索引: {INDEX.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
