"""多能量缺陷域的训练启动器：串行跑完 4 档 × 3 种子，边跑边写 run 索引。

为什么要一个启动器
------------------
上一轮（无包壳数据）是手工一条条敲 `train.py`，索引 `_flaw_multi_index.csv` 也是手写的，
一旦漏记一个 run，评测看守流水线就永远不会去评它。这里把"训练 → 登记"绑在一起：
只有 train.py 退出码为 0 且 `runs/<run_id>/test_metrics.json` 存在，才写索引。

与 `flaw_multi_pipeline.py` 的分工
----------------------------------
本脚本只管训练与登记；评测由 `flaw_multi_pipeline.py --index ... --csv-template ...`
看守（它会在训练期间反复轮询索引，谁先完成先评谁）。

用法
----
  # 带包壳的新数据（2026-10-27）
  python scripts/run_flaw_multi.py --tag flaw11 \
      --csv-template "sparse_targets_flaw11_{e}.csv" \
      --config-template "config_flaw11_{e}.json" \
      --index runs/_flaw11_multi_index.csv

  # 只补跑某一档
  python scripts/run_flaw_multi.py --tag flaw11 ... --only 3.0
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
PY = sys.executable
ENERGIES = ("1.5", "3.0", "4.5", "6.0")
SEEDS = (20261022, 20261023, 20261024)
FIELDS = ("energy", "seed", "run_id")


def read_index(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    with path.open(encoding="utf-8") as fh:
        return [r for r in csv.DictReader(fh) if (r.get("run_id") or "").strip()]


def append_index(path: Path, energy: str, seed: int, run_id: str) -> None:
    new = not path.is_file()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS)
        if new:
            w.writeheader()
        w.writerow({"energy": energy, "seed": seed, "run_id": run_id})


def run_dirs() -> set[str]:
    """当前 runs/ 下已训练完（含 test_metrics.json）的 run 集合。"""
    return {p.name for p in RUNS.iterdir()
            if p.is_dir() and (p / "test_metrics.json").is_file()}


def latest_run_id(before: set[str]) -> str | None:
    """训练结束后找出新出现的 runs/<id>/（含 test_metrics.json 才算数）。

    ⚠ 历史实现取 `sorted(new)[-1]`，即**字典序最大**——这对并行 run 极不安全：
    2026-10-30 两次事故均由此引起（外部并行 run 目录名恰好字典序更大，被误登记为
    本队列的产出）。保留此函数只为兼容旧脚本调用，正式路径请走 `match_run_id()`。
    """
    diff = sorted(run_dirs() - before)
    return diff[-1] if diff else None


def match_run_id(before: set[str], csv_name: str, seed: int) -> str | None:
    """**安全**地产出定位：在新增 run 里找 config 的 (数据表, seed) 精确匹配者。

    只看本次训练前不存在的 run 目录，再按 `config.json` 的 `data.csv`/`data.targets`
    与 `seed` 精确比对。命中多个时报错（说明有重复训练），一个都不命中返回 None。
    """
    hits: list[str] = []
    for name in sorted(run_dirs() - before):
        cfg = RUNS / name / "config.json"
        if not cfg.is_file():
            continue
        try:
            c = json.loads(cfg.read_text(encoding="utf-8"))
        except Exception:
            continue
        tgt = Path(str(c.get("data", {}).get("csv")
                       or c.get("data", {}).get("targets") or "")).name
        try:
            sd = int(c.get("seed"))
        except (TypeError, ValueError):
            continue
        if tgt == Path(csv_name).name and sd == int(seed):
            hits.append(name)
    if len(hits) > 1:
        print(f"   [FATAL] 该 (数据表, seed) 命中多个 run：{hits}，索引登记中止",
              flush=True)
        return None
    if not hits:
        print(f"   [WARN] 未在新增 run 里找到 (数据表={Path(csv_name).name}, "
              f"seed={seed}) 的精确匹配，回退字典序定位", flush=True)
        return latest_run_id(before)
    return hits[0]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True, help="run_id 后缀标签，如 flaw11")
    ap.add_argument("--csv-template", required=True,
                    help="数据表名模板，含 {e}，如 sparse_targets_flaw11_{e}.csv")
    ap.add_argument("--config-template", required=True,
                    help="配置名模板，含 {e}，如 config_flaw11_{e}.json")
    ap.add_argument("--index", required=True, help="索引 CSV 路径")
    ap.add_argument("--energies", default=",".join(ENERGIES))
    ap.add_argument("--seeds", default=",".join(str(s) for s in SEEDS))
    ap.add_argument("--only", default=None, help="只跑某一档能量")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    energies = [args.only] if args.only else args.energies.split(",")
    seeds = [int(s) for s in args.seeds.split(",")]
    index = Path(args.index)
    if not index.is_absolute():
        index = ROOT / index

    done = read_index(index)
    have = {(r["energy"], int(r["seed"])) for r in done}
    print(f"索引 {index.name}：已登记 {len(done)} 个 run", flush=True)

    plan = [(e, s) for e in energies for s in seeds if (e, s) not in have]
    print(f"待训练 {len(plan)} 个：{plan}", flush=True)
    if args.dry_run:
        return 0

    t_all = time.time()
    for k, (e, s) in enumerate(plan, 1):
        csv_name = args.csv_template.format(e=e)
        cfg_name = args.config_template.format(e=e)
        csvp = resolve_csv(csv_name)
        cfgp = ROOT / cfg_name
        for p, lbl in ((csvp, "数据表"), (cfgp, "配置")):
            if not p.is_file():
                print(f"[FATAL] 缺{lbl} {p}", flush=True)
                return 2

        before = {p.name for p in RUNS.iterdir()
                  if p.is_dir() and (p / "test_metrics.json").is_file()}
        log = RUNS / f"_train_{args.tag}_{e}_{s}.log"
        cmd = [PY, "-u", str(ROOT / "train.py"),
               "--config", cfg_name, "--csv", csv_name,
               "--seed", str(s), "--epochs", str(args.epochs),
               "--batch", str(args.batch), "--tag", f"{args.tag}_{e}"]
        print(f"\n[{k}/{len(plan)}] {e} MeV seed={s}  -> {log.name}", flush=True)
        print("   " + " ".join(cmd[2:]), flush=True)
        t0 = time.time()
        with log.open("w", encoding="utf-8") as fh:
            p = subprocess.run(cmd, cwd=str(ROOT), stdout=fh, stderr=subprocess.STDOUT,
                               env={**os.environ, "PYTHONIOENCODING": "utf-8"})
        dt = (time.time() - t0) / 60
        rid = match_run_id(before, csv_name, s)
        ok = p.returncode == 0 and rid is not None
        print(f"   -> {'OK' if ok else 'FAIL'}  {dt:.1f} min  run_id={rid}", flush=True)
        if not ok:
            tail = log.read_text(encoding="utf-8", errors="replace").splitlines()[-20:]
            print("   " + "\n   ".join(tail), flush=True)
            return 1
        append_index(index, e, s, rid)

    print(f"\n[DONE] {len(plan)} 个 run 全部完成，总耗时 {(time.time()-t_all)/60:.1f} min",
          flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
