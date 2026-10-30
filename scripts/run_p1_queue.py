"""第 2 周 P1「现有数据多线补齐」队列（无歧义子集）。

只跑三类**口径明确、无需新数据、无需新归一化**的任务：

  A. grouped split 补到 5 seeds：已有 20261022/23/24，补 25/26（1D + 2D）→ 4 run
  B. 对称两视图消融：`--input-views 2`（θ−1°/θ+1° + sin/cos，4 通道）
     对照 = 主协议非对称三视图（θ−3°/θ−1°/θ+1°，5 通道）的 headline 三 seed
     `20261024_0400 / 1633 / 1805` → 3 run
  C. unmasked loss 消融：`--loss-mode unmasked`（全像素 MSE，评估仍 masked）
     对照 = 同三 seed 的 masked headline → 3 run

**不在本队列内的两项（口径待用户裁决，见 docs/handoff/status.md）**
  · span sweep「1.5/3.0/6.0 MeV 单档 × 2/6/8° × 3 seeds」
  · dose sweep「1.5/3.0 MeV 单档 × f=1.0/0.5/0.25/0.125 × 3 seeds」
  这两项在**四档池化训练 + 逐能量评估**口径下已有 3 seeds（R34b §6 已交付逐能量
  mean±SD）；若用户要的是「真·单档训练」，需新建单档 CSV/config 且会与
  「归一化锁全局 0.7202/0.9249/5.5」冲突，故先不擅自排 51 个 run。

Gate：默认等 P0 边棒训练队列 `runs/_flaw12_train_queue.log` 出现 `[DONE]` 才开跑，
避免与 Gate-1 核心验证抢 GPU/CPU。用 `--no-gate` 可跳过等待。

断点续跑：按 (csv, seed, method, input_views, loss_mode) 扫描已有 run 的
config.json + test_metrics.json；已完成的直接跳过。

用法
----
  python scripts/run_p1_queue.py                     # 等 P0 完成后依次跑 10 个 run
  python scripts/run_p1_queue.py --only grouped
  python scripts/run_p1_queue.py --only views,loss --no-gate
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RUNS = ROOT / "runs"
PY = sys.executable

GROUPED_CSV = "csv_data/sparse_targets_3MeV_grouped.csv"
MAIN_CSV = "csv_data/sparse_targets.csv"
P0_LOG = RUNS / "_flaw12_train_queue.log"

SEEDS_EXTRA = [20261025, 20261026]          # grouped 补到 5 seeds
SEEDS_MAIN = [20261022, 20261023, 20261024]  # 与主协议 headline 逐种子配对


def completed() -> set[tuple]:
    """已完成 run 的 (csv, seed, method, input_views, loss_mode) 集合。"""
    out: set[tuple] = set()
    for d in glob.glob(str(RUNS / "2026*")):
        cp, mt = Path(d) / "config.json", Path(d) / "test_metrics.json"
        if not (cp.is_file() and mt.is_file()):
            continue
        try:
            c = json.loads(cp.read_text(encoding="utf-8"))
        except Exception:
            continue
        csvn = Path(str(c.get("data", {}).get("csv")
                        or c.get("data", {}).get("targets") or "")).name
        out.add((csvn, int(c.get("seed", -1)), str(c.get("method", "")),
                 int(c.get("input_views", 3) or 3),
                 str(c.get("loss_mode", "masked") or "masked")))
    return out


def build(only: set[str]) -> list[dict]:
    tasks: list[dict] = []
    if "grouped" in only:
        for s in SEEDS_EXTRA:
            tasks.append(dict(
                name=f"grp1d_s{s}", key=(Path(GROUPED_CSV).name, s, "cnn1d", 3, "masked"),
                cmd=[PY, "scripts/train_row1d.py", "--seed", str(s),
                     "--epochs", "100", "--csv", GROUPED_CSV]))
            tasks.append(dict(
                name=f"grp2d_s{s}", key=(Path(GROUPED_CSV).name, s, "cnn", 3, "masked"),
                cmd=[PY, "train.py", "--csv", GROUPED_CSV, "--seed", str(s),
                     "--epochs", "100", "--tag", f"grouped_s{str(s)[-2:]}"]))
    if "views" in only:
        for s in SEEDS_MAIN:
            tasks.append(dict(
                name=f"view2_s{s}", key=(Path(MAIN_CSV).name, s, "cnn", 2, "masked"),
                cmd=[PY, "train.py", "--csv", MAIN_CSV, "--seed", str(s),
                     "--epochs", "100", "--input-views", "2",
                     "--tag", f"view2_s{str(s)[-2:]}"]))
    if "loss" in only:
        for s in SEEDS_MAIN:
            tasks.append(dict(
                name=f"unmask_s{s}", key=(Path(MAIN_CSV).name, s, "cnn", 3, "unmasked"),
                cmd=[PY, "train.py", "--csv", MAIN_CSV, "--seed", str(s),
                     "--epochs", "100", "--loss-mode", "unmasked",
                     "--tag", f"unmask_s{str(s)[-2:]}"]))
    return tasks


def wait_p0(max_wait_s: int, poll: int) -> bool:
    if not P0_LOG.is_file():
        print(f"[GATE] 未找到 {P0_LOG.name}，视为 P0 未启动；继续等待", flush=True)
    t0 = time.time()
    while True:
        if P0_LOG.is_file() and "[DONE]" in P0_LOG.read_text(
                encoding="utf-8", errors="replace"):
            print("[GATE] P0 边棒训练队列已完成，开始 P1", flush=True)
            return True
        if time.time() - t0 > max_wait_s:
            print(f"[GATE] 等待 {max_wait_s / 3600:.1f} h 超时，放弃启动 P1", flush=True)
            return False
        print(f"[GATE] {time.strftime('%H:%M:%S')} P0 训练仍在进行，{poll}s 后重试…",
              flush=True)
        time.sleep(poll)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default="grouped,views,loss",
                    help="跑哪些批：grouped / views / loss，逗号分隔")
    ap.add_argument("--no-gate", action="store_true", help="不等 P0，立刻开跑")
    ap.add_argument("--gate-max-wait", type=int, default=6 * 3600)
    ap.add_argument("--poll", type=int, default=120)
    ap.add_argument("--threads", type=int, default=6,
                    help="每个训练进程的 OMP/MKL 线程上限（16 核上默认超订会拖慢 2 倍）")
    args = ap.parse_args()

    only = {x.strip() for x in args.only.split(",") if x.strip()}
    tasks = build(only)
    if not tasks:
        print("[FATAL] --only 未匹配任何任务", file=sys.stderr)
        return 2
    print(f"[CFG] only={sorted(only)}  计划 {len(tasks)} 个 run  threads={args.threads}",
          flush=True)

    if not args.no_gate:
        if not wait_p0(args.gate_max_wait, args.poll):
            return 3

    env = dict(os.environ)
    env.update({"OMP_NUM_THREADS": str(args.threads),
                "MKL_NUM_THREADS": str(args.threads),
                "OPENBLAS_NUM_THREADS": str(args.threads),
                "PYTHONIOENCODING": "utf-8"})
    done = completed()
    t_all = time.time()
    n_run = n_skip = n_fail = 0
    for i, t in enumerate(tasks, 1):
        if t["key"] in done:
            print(f"[{i}/{len(tasks)}] {t['name']} 已完成，跳过", flush=True)
            n_skip += 1
            continue
        log = RUNS / f"_p1_{t['name']}.log"
        print(f"[{i}/{len(tasks)}] {t['name']}  -> {log.name}", flush=True)
        print("   " + " ".join(t["cmd"][1:]), flush=True)
        t0 = time.time()
        with log.open("w", encoding="utf-8") as fh:
            p = subprocess.run(t["cmd"], cwd=str(ROOT), stdout=fh,
                               stderr=subprocess.STDOUT, env=env)
        dt = (time.time() - t0) / 60
        ok = p.returncode == 0 and t["key"] in completed()
        print(f"   -> {'OK' if ok else 'FAIL'}  {dt:.1f} min", flush=True)
        if ok:
            n_run += 1
        else:
            n_fail += 1
            tail = log.read_text(encoding="utf-8", errors="replace").splitlines()[-18:]
            print("       " + "\n       ".join(tail), flush=True)

    print(f"\n[DONE] P1 队列结束：新完成 {n_run}、跳过 {n_skip}、失败 {n_fail}，"
          f"总耗时 {(time.time() - t_all) / 60:.1f} min", flush=True)
    return 0 if n_fail == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
