"""等 P1 队列跑完，然后对所有 P1 run 做 Row-16 投影域评估（`p02_proj_only.py`，step=2）。

为什么需要它
------------
R34c 任务 4 要回答「masked vs unmasked 对中心行精度的影响有多大」与「两视角 vs 三视角」。
这两个问题的主口径是 **Row16 投影域 masked RMSE**（不是 all32），只能由
`p02_proj_only.py` 出。它每次调用都会重载 4 档 ×360 投影，所以这里**一次性**把
所有待评 run 交给它（数据只载入一次）。

发现 run 的方式：**按 config 精确匹配**（`data.targets` + `seed`），
不用目录名/字典序——R34c 出过两次字典序抢登事故（`run_flaw_multi.py:match_run_id`）。

用法
----
  python scripts/p1_row16_eval_all.py                # 阻塞等队列结束后评估
  python scripts/p1_row16_eval_all.py --no-wait      # 不等，直接评估现有 run
  python scripts/p1_row16_eval_all.py --dry-run      # 只打印将要评估的 run
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
QUEUE_LOG = RUNS / "_p1_queue.log"
OUT_DIR = RUNS / "_reviewer_p0"
PY = sys.executable

# 主协议 masked headline 三种子：历史上同 (csv, seed) 的 run 有二十多个（大量短跑 +
# `_linear_interp` 派生目录），按 config 无法唯一命中，故**显式钉死**这三个 run_id。
# 它们的 Row16 早已评过（0.122508 / 0.126634 / 0.122513），这里只作对照，不重复评估。
HEADLINE = {
    20261022: "20261024_0400_cnn_bs8_lr1e-4",
    20261023: "20261024_1633_cnn_bs8_lr1e-4",
    20261024: "20261024_1805_cnn_bs8_lr1e-4",
}

# 需要**按 config 发现**的消融组：(label, csv, seed, input_views, loss_mode)
TARGETS: list[tuple[str, str, int, int, str]] = [
    ("two-view", "sparse_targets.csv", 20261022, 2, "masked"),
    ("two-view", "sparse_targets.csv", 20261023, 2, "masked"),
    ("two-view", "sparse_targets.csv", 20261024, 2, "masked"),
    ("unmasked-loss", "sparse_targets.csv", 20261022, 3, "unmasked"),
    ("unmasked-loss", "sparse_targets.csv", 20261023, 3, "unmasked"),
    ("unmasked-loss", "sparse_targets.csv", 20261024, 3, "unmasked"),
]

# 派生目录不是训练 run，发现时必须排除
DERIVED_SUFFIX = ("_linear_interp", "_cnn_recon", "_flaw_step3", "_cnn_recon_mask")


def run_cfg(rid: str) -> dict:
    return json.loads((RUNS / rid / "config.json").read_text(encoding="utf-8"))


def find_run(csv: str, seed: int, views: int, loss: str) -> str | None:
    hits: list[str] = []
    for d in sorted(glob.glob(str(RUNS / "2026*"))):
        name = Path(d).name
        if name.endswith(DERIVED_SUFFIX):
            continue
        cp = Path(d) / "config.json"
        if not cp.is_file():
            continue
        if not (Path(d) / "test_metrics.json").is_file():
            continue          # 未跑完的 run 不参与
        try:
            c = json.loads(cp.read_text(encoding="utf-8"))
        except Exception:
            continue
        tgt = Path(str(c.get("data", {}).get("csv")
                        or c.get("data", {}).get("targets") or "")).name
        if (tgt == Path(csv).name and int(c.get("seed", -1)) == seed
                and int(c.get("input_views", 3) or 3) == views
                and str(c.get("loss_mode", "masked") or "masked") == loss):
            hits.append(Path(d).name)
    if len(hits) > 1:
        print(f"[FATAL] (csv={csv}, seed={seed}, views={views}, loss={loss}) "
              f"命中多个 run: {hits}", flush=True)
        return None
    return hits[0] if hits else None


def wait_queue(poll: int = 60, timeout_s: int = 6 * 3600) -> bool:
    """等 `run_p1_queue.py` 跑完。

    判据是队列日志里的收尾行 ``[DONE] P1 队列结束：...``，**不是** ``count("-> OK")
    >= count("-> _p1_")``：队列允许 ``-> SKIP``（已有产出时跳过），SKIP 项永远
    不会变成 OK，旧判据在这种情况下永远不成立 —— R34c 实测空等到 6 h 超时，
    unmask 第三种子的 Row16 只能手动补评。改为认收尾行，并以"队列进程是否还在"
    作为兜底（防止收尾行还没写就被杀）。
    """
    t0 = time.time()
    while time.time() - t0 < timeout_s:
        txt = QUEUE_LOG.read_text(encoding="utf-8", errors="ignore") if QUEUE_LOG.is_file() else ""
        n_ok = txt.count("-> OK")
        n_skip = txt.count("已完成，跳过")          # 队列脚本的 SKIP 行是这句，不是 "-> SKIP"
        n_start = txt.count("-> _p1_")
        print(f"[wait] {time.strftime('%H:%M:%S')}  OK {n_ok} / SKIP {n_skip} / 启动 {n_start}",
              flush=True)
        if "[DONE]" in txt:
            print("[wait] 队列收尾行已出现，队列全部完成", flush=True)
            return True
        if n_start and (n_ok + n_skip) >= n_start and not _queue_proc_alive():
            print("[wait] 全部条目已终结且队列进程已退出", flush=True)
            return True
        time.sleep(poll)
    print("[wait] 超时", flush=True)
    return False


def _queue_proc_alive() -> bool:
    """队列启动器进程是否还在（Windows 用 tasklist，其它平台用 /proc 或 ps）。"""
    try:
        import subprocess
        if os.name == "nt":
            out = subprocess.run(["tasklist", "/FI", "IMAGENAME eq python.exe", "/FO", "CSV"],
                                 capture_output=True, text=True, timeout=30).stdout
            return "python" in out.lower()
        out = subprocess.run(["ps", "-eo", "args"], capture_output=True, text=True,
                             timeout=30).stdout
        return "run_p1_queue.py" in out
    except Exception:
        return True  # 判断不了就当作还活着，交给 [DONE] 判据


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-wait", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--poll", type=int, default=60)
    a = ap.parse_args()

    if not a.no_wait:
        wait_queue(a.poll)

    todo: list[str] = []
    for seed, rid in sorted(HEADLINE.items()):
        out = OUT_DIR / f"projonly_s2_{rid}.json"
        print(f"[headline] seed={seed} -> {rid}  "
              f"{'已有 Row16' if out.is_file() else '缺 Row16（异常）'}", flush=True)
    for label, csv, seed, views, loss in TARGETS:
        rid = find_run(csv, seed, views, loss)
        if rid is None:
            print(f"[skip] {label} seed={seed}：未找到（可能还没跑完）", flush=True)
            continue
        out = OUT_DIR / f"projonly_s2_{rid}.json"
        if out.is_file() and '"input_views"' in out.read_text(encoding="utf-8"):
            print(f"[have] {label} seed={seed} -> {rid}（已有 Row16）", flush=True)
            continue
        print(f"[todo] {label} seed={seed} -> {rid}", flush=True)
        todo.append(rid)

    if a.dry_run or not todo:
        print(f"[dry-run] 将评估 {len(todo)} 个 run", flush=True)
        return 0

    cmd = [PY, "scripts/p02_proj_only.py", "--step", "2"]
    for rid in todo:
        cmd += ["--cnn-run", rid]
    print("[eval] " + " ".join(cmd), flush=True)
    r = subprocess.run(cmd, cwd=str(ROOT), env={"PYTHONPATH": str(ROOT), **__import__("os").environ})
    return r.returncode


if __name__ == "__main__":
    sys.exit(main())
