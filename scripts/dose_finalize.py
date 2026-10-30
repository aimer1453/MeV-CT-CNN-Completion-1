#!/usr/bin/env python
"""P0-1 收尾驱动（后台）：等待剂量管线 9 档跑完 -> 跑 dose_eval (f=0.5/0.25/0.125)
-> 出 Fig 9 + improvement 表。

本脚本**只等待、不启动训练**：训练由既有管线进程（dose_pipeline.py）负责。
轮询 runs/_dose_index.csv 中 (0.5/0.25/0.125) 三档 × 3 种子的登记数，达 9 即视为训练完成。

用法：python scripts/dose_finalize.py   （建议 run_in_background）
"""
from __future__ import annotations

import csv
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from data.csv_paths import resolve_csv  # noqa: E402  CSV 统一落点 csv_data/
RUNS = ROOT / "runs"
INDEX = RUNS / "_dose_index.csv"
LOG = RUNS / "_dose_finalize.log"
PY = r"<user-python>"
DOSE_FACTORS = [0.5, 0.25, 0.125]
NEEDED = 9  # 3 剂量档 × 3 种子


def log(s: str) -> None:
    line = f"[{time.strftime('%m-%d %H:%M:%S')}] {s}"
    with open(LOG, "a", encoding="utf-8") as fh:
        fh.write(line + "\n")
    print(line, flush=True)


def count_index() -> int:
    if not INDEX.is_file():
        return 0
    n = 0
    for r in csv.DictReader(open(INDEX, newline="", encoding="utf-8")):
        if r["factor"] in ("0.5", "0.25", "0.125"):
            n += 1
    return n


def main() -> int:
    # ---- 1) 等待训练完成 ----
    t0 = time.time()
    timeout = 12 * 3600
    while True:
        n = count_index()
        log(f"剂量档登记 = {n}/{NEEDED}")
        if n >= NEEDED:
            break
        if time.time() - t0 > timeout:
            log("[FATAL] 超时：训练未达 9 档，请检查管线进程")
            return 2
        time.sleep(120)

    log("训练完成，开始评测")
    byf: dict[str, list[str]] = {}
    for r in csv.DictReader(open(INDEX, newline="", encoding="utf-8")):
        if r["factor"] in ("0.5", "0.25", "0.125"):
            byf.setdefault(r["factor"], []).append(r["run_id"])

    # ---- 2) dose_eval 三档 ----
    for f in DOSE_FACTORS:
        rids = byf[f"{f:g}"]
        if len(rids) != 3:
            log(f"[WARN] f={f:g} 只有 {len(rids)} 个 run_id：{rids}")
        cfg = ROOT / f"config_dose_f{f:g}.json"
        csvp = resolve_csv(f"sparse_targets_dose_f{f:g}.csv")
        out = RUNS / f"_dose_eval_f{f:g}.json"
        cmd = [PY, str(ROOT / "scripts" / "dose_eval.py"), "--factor", f"{f:g}",
               "--csv", str(csvp), "--config", str(cfg),
               "--run-ids", *rids, "--out", str(out)]
        log("RUN " + " ".join(cmd))
        rc = subprocess.run(cmd, cwd=str(ROOT)).returncode
        if rc != 0:
            log(f"[FATAL] dose_eval f={f:g} 失败 rc={rc}")
            return 3
        log(f"dose_eval f={f:g} OK")

    if not (RUNS / "_dose_eval_f1.json").is_file():
        log("[WARN] 缺 _dose_eval_f1.json（f=1.0 评测），Fig 9 将缺该档")

    # ---- 3) 出图 + 表 ----
    for script in ("dose_improvement_table.py", "plot_dose_rmse.py"):
        cmd = [PY, str(ROOT / "scripts" / script)]
        log("RUN " + " ".join(cmd))
        rc = subprocess.run(cmd, cwd=str(ROOT)).returncode
        if rc != 0:
            log(f"[WARN] {script} rc={rc}")
    log("全部完成：Fig 9 + improvement 表已生成（docs/figs/p01_dose_rmse/）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
