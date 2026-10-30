#!/usr/bin/env python
# -*- coding: utf-8 -*-
r"""
scripts/p1_parallel_watchdog.py —— P1 并行期间的掉速看守（用户 2026-10-30 裁决：>2× 自动回退）

为什么需要
----------
为把 GPU 空闲算力用起来，P1 队列改 `--no-gate` 与 P0 边棒训练队列并行。风险是 CPU
（8 物理核 / 16 逻辑）被打满后**两个队列一起变慢**，历史教训：5 个 2D run 并发时
单 run 掉到 0.6 ep/min。因此需要一个独立看守：只看 P0 run 的 epoch 耗时，
一旦超过串行基线 2 倍，就杀掉 P1 相关进程恢复串行。

判定口径
--------
* 受保护的量：**当前正在训练的 P0 run** 的最近若干 epoch 耗时均值。
* P0 串行基线（实测，见 commit bcb65c7 前）：4.5/6.0 MeV 约 2.6–4.2 s/epoch；
  阈值取 **8.0 s**（约 2×）。
* **启动窗口不计**：新 run 的前若干 epoch 含一次性数据装载/建图开销，且若此时 P1
  也在启动重型任务会叠加一次性尖峰（2026-10-30 02:00 实测尖峰 8.83 s，而同一 run
  回退后立即回落到 3.1–3.6 s）。故指定 `--min-epochs`（默认 12）之前只采样不判定。
* **连续报警才回退**（`--consecutive-alerts`，默认 2），避免单次瞬时抖动误杀。

用法
----
    python scripts/p1_parallel_watchdog.py --poll 120 --max-epoch-s 8.0
    python scripts/p1_parallel_watchdog.py --dry-run        # 只报不杀
"""
from __future__ import annotations

import argparse
import re
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RUNS = ROOT / "runs"

EPOCH_RE = re.compile(r"^\[epoch\s+(\d+)\].*?\s([0-9.]+)s\s*(?:<- best)?\s*$")
P1_MARKERS = (
    "run_p1_queue.py",
    "--tag grouped_s",
    "--tag view2_s",
    "--tag unmask_s",
    "train_row1d.py",
)
P1_SEEDS = ("20261025", "20261026")


def p0_active_log() -> Path | None:
    """最新一个**未训练完**的 P0 训练日志。"""
    cands = [p for p in RUNS.glob("_train_flaw12_*.log")
             if "[OK] 训练" not in p.read_text(encoding="utf-8", errors="replace")]
    if not cands:
        return None
    return max(cands, key=lambda p: p.stat().st_mtime)


def count_epochs(log: Path) -> int:
    """该 run 已完成的 epoch 数（用于判定是否还在启动窗）。"""
    txt = log.read_text(encoding="utf-8", errors="replace")
    return sum(1 for line in txt.splitlines() if EPOCH_RE.match(line.strip()))


def recent_epoch_seconds(log: Path, window: int = 6) -> list[float]:
    txt = log.read_text(encoding="utf-8", errors="replace")
    out = []
    for line in txt.splitlines():
        m = EPOCH_RE.match(line.strip())
        if m:
            out.append(float(m.group(2)))
    return out[-window:]


def p0_pids_to_protect() -> set[int]:
    """当前 P0 队列/训练进程 PID（保护名单，绝不杀）。"""
    out: set[int] = set()
    ps = subprocess.run(
        ["powershell", "-NoProfile", "-Command",
         "Get-CimInstance Win32_Process -Filter \"Name like 'python%'\" | "
         "Select-Object ProcessId,CommandLine | ConvertTo-Json -Compress"],
        capture_output=True, text=True, cwd=str(ROOT))
    if ps.returncode != 0:
        return out
    import json
    try:
        data = json.loads(ps.stdout)
    except Exception:
        return out
    if isinstance(data, dict):
        data = [data]
    for item in data:
        cmd = (item or {}).get("CommandLine") or ""
        if "run_flaw_multi" in cmd or "_train_flaw12_" in cmd or "flaw_multi_pipeline" in cmd:
            out.add(int((item or {})["ProcessId"]))
    return out


def kill_p1(protect: set[int]) -> list[int]:
    ps = subprocess.run(
        ["powershell", "-NoProfile", "-Command",
         "Get-CimInstance Win32_Process -Filter \"Name like 'python%'\" | "
         "Select-Object ProcessId,CommandLine | ConvertTo-Json -Compress"],
        capture_output=True, text=True, cwd=str(ROOT))
    if ps.returncode != 0:
        return []
    import json
    try:
        data = json.loads(ps.stdout)
    except Exception:
        return []
    if isinstance(data, dict):
        data = [data]
    killed: list[int] = []
    for item in data:
        cmd = (item or {}).get("CommandLine") or ""
        pid = int((item or {})["ProcessId"])
        hit = any(mk in cmd for mk in P1_MARKERS)
        # train_row1d 只杀 P1 正在用的那两个补种子 seed，避免误伤 grouped 旧 run
        if "train_row1d" in cmd and not any(s in cmd for s in P1_SEEDS):
            hit = False
        if hit and pid not in protect:
            subprocess.run(["taskkill", "/F", "/PID", str(pid)],
                           capture_output=True, text=True, cwd=str(ROOT))
            killed.append(pid)
    return killed


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--poll", type=int, default=120, help="轮询间隔（秒）")
    ap.add_argument("--max-epoch-s", type=float, default=8.0,
                    help="P0 epoch 耗时上限，超过即回退（默认 8.0 s ≈ 2× 基线）")
    ap.add_argument("--window", type=int, default=6, help="取最近多少个 epoch 求均值")
    ap.add_argument("--dry-run", action="store_true", help="只报不杀")
    ap.add_argument("--min-epochs", type=int, default=12, dest="min_epochs",
                    help="新 run 的前 N 个 epoch 只采样不判定（启动/首次装载尖峰不计）")
    ap.add_argument("--consecutive-alerts", type=int, default=2,
                    dest="consecutive_alerts",
                    help="连续报警达到该次数才回退，避免单次瞬时抖动误杀")
    ap.add_argument("--max-wait", type=int, default=6 * 3600, help="最长看守时长（秒）")
    args = ap.parse_args()

    print(f"[WDG] 启动 poll={args.poll}s 阈值={args.max_epoch_s}s "
          f"window={args.window} min_epochs={args.min_epochs} "
          f"连续报警={args.consecutive_alerts} dry_run={args.dry_run}", flush=True)
    t0 = time.time()
    alert_n = 0
    while time.time() - t0 < args.max_wait:
        p0_log = p0_active_log()
        if p0_log is None:
            print(f"[WDG] {time.strftime('%H:%M:%S')} 无进行中的 P0 训练，退出看守",
                  flush=True)
            return 0
        secs = recent_epoch_seconds(p0_log, args.window)
        n_ep = count_epochs(p0_log)
        if n_ep < args.min_epochs:
            print(f"[WDG] {time.strftime('%H:%M:%S')} {p0_log.name} "
                  f"已 {n_ep} epoch < {args.min_epochs}（启动窗不判定）", flush=True)
        elif len(secs) >= 3:
            mean = sum(secs) / len(secs)
            if mean > args.max_epoch_s:
                alert_n += 1
            else:
                alert_n = 0
            tag = f"ALERT {alert_n}/{args.consecutive_alerts}" if alert_n else "ok"
            print(f"[WDG] {time.strftime('%H:%M:%S')} {p0_log.name} "
                  f"epoch {n_ep} 最近 {len(secs)} 均值 {mean:.2f}s  [{tag}]", flush=True)
            if alert_n >= args.consecutive_alerts:
                if args.dry_run:
                    print("[WDG] dry-run：不执行回退", flush=True)
                else:
                    protect = p0_pids_to_protect()
                    killed = kill_p1(protect)
                    print(f"[WDG] **回退触发**：已杀 P1 进程 {killed}"
                          f"（P0/评测保护名单未动）", flush=True)
                return 1
        else:
            print(f"[WDG] {time.strftime('%H:%M:%S')} {p0_log.name} 采样不足，等待",
                  flush=True)
        time.sleep(args.poll)
    print("[WDG] 超时退出", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
