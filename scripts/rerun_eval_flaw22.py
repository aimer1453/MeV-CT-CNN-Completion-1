"""中心棒批（3_3_flaw_2_2）12 个 run 的评测**重跑**，fail-fast。

背景
----
bubble_metrics 的 σ 基在 center 缺陷（R_DEFECT_PX=0）下恒为 NaN：r=0 的棒只有
center 自己，同半径族再剔除缺陷棒后必然为空（实测 12 个 run 全部命中）。
已在 p01_flaw_step3_eval.py 修复——center 分支改用 4 根完好角棒作噪声基，
边棒/角棒分支逻辑逐位不变（保护 Table 8）。
本脚本用修复后的评测脚本重跑 12 个 run，覆盖 sigma=nan 的旧产物。

与 flaw_multi_pipeline.py 的区别
--------------------------------
  · flaw_multi_pipeline 是**看守**（边训练边评测），且失败不中止（会继续下一个）；
  · 本脚本是**纯重跑**，任一 run 失败立即停止整个队列（用户要求 fail-fast）。
  · 同样沿用 --baseline-from 优化：每能量首种子全量（含 dense/linear/FBP/SART），
    其余种子只算 CNN 补全。

用法
----
  python scripts/rerun_eval_flaw22.py --dry-run   # 只打印命令，不执行（先核对）
  python scripts/rerun_eval_flaw22.py             # 重跑全部 12 个
"""
from __future__ import annotations

import argparse
import csv
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RUNS = ROOT / "runs"
INDEX = RUNS / "_flaw22_multi_index.csv"
EVAL_SCRIPT = ROOT / "scripts" / "p01_flaw_step3_eval.py"
TRAJ = "runs/_flaw22_cavity_localization.json"
CSV_TPL = "sparse_targets_flaw22_{e}.csv"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--threads", type=int, default=4,
                    help="评测进程的 BLAS 线程数")
    ap.add_argument("--dry-run", action="store_true",
                    help="只打印命令，不执行")
    ap.add_argument("--log-prefix", default="_eval_flaw22_",
                    help="评测日志名前缀（覆盖同名旧日志）")
    ap.add_argument("--reuse", action="store_true",
                    help="透传给评测脚本：从磁盘读已算重建与误差图，跳过 FBP/SART/"
                         "CNN 重算。用于只补算新增字段（如 row16 linear 基线）时，"
                         "分钟级完成，不触碰训练权重与既有重建")
    args = ap.parse_args()

    rows = [r for r in csv.DictReader(INDEX.open(encoding="utf-8"))
            if (r.get("run_id") or "").strip()]
    #: 每能量第一个 run 做全量，其余复用它的基线
    first: dict[str, str] = {}
    for r in rows:
        first.setdefault(r["energy"], r["run_id"])

    env = dict(os.environ)
    env.update({"OMP_NUM_THREADS": str(args.threads),
                "MKL_NUM_THREADS": str(args.threads),
                "OPENBLAS_NUM_THREADS": str(args.threads)})

    for r in rows:
        e, rid, seed = r["energy"], r["run_id"], r["seed"]
        base = None if rid == first[e] else first[e]
        cmd = [sys.executable, "-u", str(EVAL_SCRIPT),
               "--run-id", rid, "--energy", e,
               "--flaw-csv", CSV_TPL.format(e=e),
               "--defect-rod", "center", "--traj-json", TRAJ]
        if base:
            cmd += ["--baseline-from", base]
        if args.reuse:
            cmd += ["--reuse"]
        log = RUNS / f"{args.log_prefix}{e}_{seed}.log"
        tag = "全量" if base is None else f"baseline_from={base}"
        print(f"[EVAL] {e} MeV seed={seed} {rid} ({tag})", flush=True)
        if args.dry_run:
            print("       " + " ".join(cmd), flush=True)
            continue
        t0 = time.time()
        with log.open("w", encoding="utf-8") as fh:
            p = subprocess.run(cmd, stdout=fh, stderr=subprocess.STDOUT, env=env)
        dt = time.time() - t0
        ok = (p.returncode == 0
              and (RUNS / f"{rid}_flaw_step3" / "image_metrics.json").is_file())
        print(f"       -> {'OK' if ok else 'FAIL'}  {dt / 60:.1f} min"
              f"  log={log.name}", flush=True)
        if not ok:
            tail = log.read_text(encoding="utf-8",
                                 errors="replace").splitlines()[-15:]
            print("       " + "\n       ".join(tail), flush=True)
            print(f"[STOP] fail-fast：{rid} 失败，停止整个重跑队列", flush=True)
            return 1
    print("[DONE] 12/12 重跑完成", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
