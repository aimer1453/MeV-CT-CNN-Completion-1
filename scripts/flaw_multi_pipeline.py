"""看守式流水线：多能量缺陷训练每完成一个 run，就自动跑 step3 评测，最后聚合出图。

为什么要看守而不是一次性批量
----------------------------
训练是串行的，12 个 run 要跑 2~3 小时。评测里 SART（20 iter × 2 相位）约 4.5 min/相位，
但**与模型权重无关**——同能量的 3 个种子结果逐位相同。所以：

  · 每个能量的**第一个**完成的种子：全量评测（含 dense/linear/FBP/SART），~10 min
  · 同能量其余种子：`--baseline-from <首种子>`，只算 CNN 补全，~1 min

省掉的不只是时间，还有与训练抢 CPU 造成的训练减速（实测争抢会把 epoch 从 5.7 s
拖到 13 s）。评测子进程用 OMP_NUM_THREADS 限制线程数，给训练的 dataloader 让路。

用法
----
  python scripts/flaw_multi_pipeline.py                  # 前台看守（Ctrl-C 停）
  python scripts/flaw_multi_pipeline.py --once           # 只处理当前已就绪的，不等待
  python scripts/flaw_multi_pipeline.py --poll 45 --threads 4
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
from data.csv_paths import resolve_csv  # noqa: E402  CSV 统一落点 csv_data/
RUNS = ROOT / "runs"
INDEX = RUNS / "_flaw_multi_index.csv"
EVAL_SCRIPT = ROOT / "scripts" / "p01_flaw_step3_eval.py"
AGG_SCRIPT = ROOT / "scripts" / "flaw_multi_aggregate.py"
EXPECTED_RUNS = 12
#: 数据表名模板（含 {e}）；由 --csv-template 覆盖，默认是上一轮无包壳数据的命名
CSV_TEMPLATE = "sparse_targets_flaw_{e}.csv"
PY = sys.executable


def read_index() -> list[dict]:
    if not INDEX.is_file():
        return []
    rows = []
    with INDEX.open(encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            rid = (r.get("run_id") or "").strip()
            e = (r.get("energy") or "").strip()
            s = (r.get("seed") or "").strip()
            if rid and e and s:
                rows.append({"energy": e, "seed": s, "run_id": rid})
    return rows


def eval_done(rid: str) -> bool:
    return (RUNS / f"{rid}_flaw_step3" / "image_metrics.json").is_file()


def first_done_of(energy: str, rows: list[dict]) -> str | None:
    """同能量里第一个已评测完的 run（作为基线源）。"""
    for r in rows:
        if r["energy"] == energy and eval_done(r["run_id"]):
            return r["run_id"]
    return None


def run_eval(row: dict, baseline_from: str | None, threads: int,
             defect_rod: str | None = None, traj_json: str | None = None,
             log_prefix: str = "_eval_flaw_") -> bool:
    rid, energy = row["run_id"], row["energy"]
    csv_name = CSV_TEMPLATE.format(e=energy)
    if not resolve_csv(csv_name).is_file():
        print(f"  [SKIP] 缺数据表 {csv_name}", flush=True)
        return False
    cmd = [PY, "-u", str(EVAL_SCRIPT), "--run-id", rid,
           "--energy", energy, "--flaw-csv", csv_name]
    if baseline_from:
        cmd += ["--baseline-from", baseline_from]
    if defect_rod:
        cmd += ["--defect-rod", defect_rod]
    if traj_json:
        cmd += ["--traj-json", traj_json]
    log = RUNS / f"{log_prefix}{energy}_{row['seed']}.log"
    env = dict(os.environ)
    env.update({"OMP_NUM_THREADS": str(threads), "MKL_NUM_THREADS": str(threads),
                "OPENBLAS_NUM_THREADS": str(threads)})
    t0 = time.time()
    tag = f"baseline_from={baseline_from}" if baseline_from else "全量"
    print(f"[EVAL] {energy} MeV seed={row['seed']} {rid} ({tag})", flush=True)
    with log.open("w", encoding="utf-8") as fh:
        p = subprocess.run(cmd, stdout=fh, stderr=subprocess.STDOUT, env=env)
    dt = time.time() - t0
    ok = p.returncode == 0 and eval_done(rid)
    print(f"       -> {'OK' if ok else 'FAIL'}  {dt / 60:.1f} min  log={log.name}",
          flush=True)
    if not ok:
        tail = log.read_text(encoding="utf-8", errors="replace").splitlines()[-15:]
        print("       " + "\n       ".join(tail), flush=True)
    return ok


def main() -> int:
    # global 声明必须在 argparse 之前：`default=CSV_TEMPLATE` 就是在读全局，
    # 放在后面会 SyntaxError: name 'CSV_TEMPLATE' is used prior to global declaration
    global INDEX, CSV_TEMPLATE, EXPECTED_RUNS

    ap = argparse.ArgumentParser()
    ap.add_argument("--poll", type=int, default=60, help="轮询间隔（秒）")
    ap.add_argument("--threads", type=int, default=4, help="评测进程的 BLAS 线程数")
    ap.add_argument("--once", action="store_true", help="只处理当前已就绪的，不等待")
    ap.add_argument("--max-wait", type=int, default=6 * 3600,
                    help="最长等待（秒），超时退出")
    ap.add_argument("--index", default=None,
                    help="run 索引 CSV（默认 runs/_flaw_multi_index.csv）")
    ap.add_argument("--csv-template", default=CSV_TEMPLATE, dest="csv_template",
                    help="数据表名模板，含 {e}（默认 sparse_targets_flaw_{e}.csv）")
    ap.add_argument("--defect-rod", default=None,
                    help="缺陷棒名，透传给 step3 评测（边棒批次传 edge2）；"
                         "不传则沿用角棒默认，保护 Table 8 可复现")
    ap.add_argument("--traj-json", default=None,
                    help="空腔轨迹定标 json（按能量匹配），透传给 step3 评测做 D4")
    ap.add_argument("--log-prefix", default="_eval_flaw_", dest="log_prefix",
                    help="评测日志名前缀。换数据批次时**必须换**（边棒传 _eval_flaw12_），"
                         "否则会静默覆盖上一批同名日志，破坏可追溯性")
    ap.add_argument("--expected", type=int, default=EXPECTED_RUNS,
                    help="期望的 run 数（默认 12）")
    ap.add_argument("--no-aggregate", action="store_true",
                    help="全部评完也不跑聚合（聚合改由人工带参数触发）")
    args = ap.parse_args()

    if args.index:
        INDEX = Path(args.index)
        if not INDEX.is_absolute():
            INDEX = ROOT / INDEX
    CSV_TEMPLATE = args.csv_template
    EXPECTED_RUNS = args.expected
    print(f"[CFG] index={INDEX.relative_to(ROOT)}  csv_template={CSV_TEMPLATE}"
          f"  expected={EXPECTED_RUNS}", flush=True)

    t_start = time.time()
    while True:
        rows = read_index()
        pending = [r for r in rows if not eval_done(r["run_id"])]
        for row in pending:
            base = first_done_of(row["energy"], rows)
            if base == row["run_id"]:
                base = None
            run_eval(row, base, args.threads, args.defect_rod, args.traj_json,
                     args.log_prefix)
            rows = read_index()          # 重新读，可能又多了新 run

        rows = read_index()
        done = [r for r in rows if eval_done(r["run_id"])]
        print(f"[STATUS] {time.strftime('%H:%M:%S')}  训练完成 {len(rows)}/{EXPECTED_RUNS}"
              f"  已评测 {len(done)}/{EXPECTED_RUNS}", flush=True)

        if len(done) >= EXPECTED_RUNS:
            print("[DONE] 全部评测完成", flush=True)
            if not args.no_aggregate:
                subprocess.run([PY, "-u", str(AGG_SCRIPT)], check=False)
            return 0
        if args.once:
            return 0
        if time.time() - t_start > args.max_wait:
            print("[TIMEOUT] 超过 --max-wait，退出", flush=True)
            return 3
        time.sleep(args.poll)


if __name__ == "__main__":
    raise SystemExit(main())
