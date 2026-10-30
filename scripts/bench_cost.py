"""成本表实测：参数量 / 单视角推理时间 / 显存占用。

为什么单独写脚本：审稿人要的是**可复核的实测值**，不是估算。三个数分别有明确的
测量口径，脚本把它们连同口径一起打印，避免日后"这个数怎么来的"说不清。

口径
----
参数量        `sum(p.numel() for p in model.parameters())`，含 BN 的 weight/bias。
推理时间      单样本 (1, 5, 32, 1200) 前向；warmup 30 次丢弃，计时 300 次取中位数
              （均值会被偶发调度抖动拉高）。**含 H2D 拷贝**的端到端口径与**纯 GPU
              计算**的口径分别报。单视角 = 一张 32×1200 的投影图。
显存占用      `torch.cuda.max_memory_allocated()`（PyTorch 分配器实际占用，不含
              CUDA context）。这是训练与推理的峰值；`reserved` 另报。
训练时长      从 run 目录 `config.json` 的 mtime（train.py 启动时写）到
              `test_metrics.json` 的 mtime（训练+test 结束），含数据加载与 test。

用法
----
  python scripts/bench_cost.py --run-id 20261024_0400_cnn_bs8_lr1e-4
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from models.unet import UNet                        # noqa: E402

RUNS = ROOT / "runs"
IN_SHAPE = (1, 5, 32, 1200)      # batch, channels, detector rows, detector cols


def training_wall_clock(run_id: str) -> dict:
    """run 的实际起止（按产物 mtime），用于"100 epoch 训练多久"。"""
    d = RUNS / run_id
    cfg = d / "config.json"
    end_candidates = [p for p in (d / "test_metrics.json", d / "best.pt",
                                  d / "train_log.csv") if p.is_file()]
    if not cfg.is_file() or not end_candidates:
        return {"available": False}
    t0 = cfg.stat().st_mtime
    t_end = max(p.stat().st_mtime for p in end_candidates)
    n_epoch = 0
    log = d / "train_log.csv"
    if log.is_file():
        with log.open(encoding="utf-8") as fh:
            next(fh, None)
            n_epoch = len({line.split(",")[0] for line in fh if line.strip()})
    return {"available": True,
            "seconds": t_end - t0,
            "minutes": (t_end - t0) / 60.0,
            "epochs_logged": n_epoch,
            "start": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(t0)),
            "end": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(t_end)),
            "note": "config.json 写入 → test_metrics.json/best.pt 最后写入，"
                    "含数据加载、100 epoch 训练与 test 评测"}


def bench(run_id: str, n_warmup: int = 30, n_iter: int = 300) -> dict:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UNet().to(device)
    ckpt_p = RUNS / run_id / "best.pt"
    if ckpt_p.is_file():
        ckpt = torch.load(ckpt_p, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model"])
        best_epoch = int(ckpt.get("epoch", -1))
    else:
        best_epoch = -1
    model.eval()

    n_params = sum(p.numel() for p in model.parameters())
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)

    x = torch.randn(*IN_SHAPE, device=device)
    times_gpu: list[float] = []
    with torch.no_grad():
        for i in range(n_warmup + n_iter):
            if device.type == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            model(x)
            if device.type == "cuda":
                torch.cuda.synchronize()
            dt = (time.perf_counter() - t0) * 1e3
            if i >= n_warmup:
                times_gpu.append(dt)

    # 端到端（含 H2D 拷贝，模拟逐视角流式推理）
    cpu_x = torch.randn(*IN_SHAPE)
    times_e2e: list[float] = []
    with torch.no_grad():
        for i in range(n_warmup + n_iter):
            t0 = time.perf_counter()
            y = cpu_x.to(device, non_blocking=False)
            model(y)
            if device.type == "cuda":
                torch.cuda.synchronize()
            dt = (time.perf_counter() - t0) * 1e3
            if i >= n_warmup:
                times_e2e.append(dt)

    # 批 8 吞吐（一次前向补 8 个视角）
    xb = torch.randn(8, *IN_SHAPE[1:], device=device)
    tb: list[float] = []
    with torch.no_grad():
        for i in range(10 + 50):
            if device.type == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            model(xb)
            if device.type == "cuda":
                torch.cuda.synchronize()
            if i >= 10:
                tb.append((time.perf_counter() - t0) * 1e3)

    peak_alloc = (torch.cuda.max_memory_allocated() / 1024 ** 3
                  if device.type == "cuda" else float("nan"))
    peak_res = (torch.cuda.max_memory_reserved() / 1024 ** 3
                if device.type == "cuda" else float("nan"))
    gpu_name = torch.cuda.get_device_name(0) if device.type == "cuda" else "CPU"

    return {
        "run_id": run_id, "device": gpu_name, "best_epoch_in_ckpt": best_epoch,
        "params_total": n_params, "params_trainable": n_train,
        "input_shape_per_view": list(IN_SHAPE),
        "inference_ms_per_view_gpu_only": {
            "median": statistics.median(times_gpu),
            "mean": statistics.fmean(times_gpu),
            "p95": statistics.quantiles(times_gpu, n=20)[18],
            "n_iter": n_iter},
        "inference_ms_per_view_end_to_end_h2d": {
            "median": statistics.median(times_e2e),
            "mean": statistics.fmean(times_e2e),
            "p95": statistics.quantiles(times_e2e, n=20)[18],
            "n_iter": n_iter},
        "inference_ms_per_batch8": {
            "median": statistics.median(tb),
            "ms_per_view": statistics.median(tb) / 8.0, "n_iter": 50},
        "peak_memory_allocated_GB": peak_alloc,
        "peak_memory_reserved_GB": peak_res,
        "memory_note": ("PyTorch 分配器峰值，不含 CUDA context(~0.3-0.5 GB)；"
                        "本进程只做前向，训练峰值另见 training_wall_clock 的 run"),
        "training_wall_clock": training_wall_clock(run_id),
        "measurement_note": ("warmup 30 次丢弃后计时 300 次取中位数；"
                             "只做前向（no_grad），不含数据加载与后处理"),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-id", default="20261024_0400_cnn_bs8_lr1e-4")
    ap.add_argument("--out", default=None)
    ap.add_argument("--n-iter", type=int, default=300)
    args = ap.parse_args()

    r = bench(args.run_id, n_iter=args.n_iter)
    out = Path(args.out) if args.out else ROOT / "runs" / "_cost_benchmark.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(r, indent=2, ensure_ascii=False), encoding="utf-8")

    print("=" * 72)
    print(f"成本实测  run={r['run_id']}  device={r['device']}")
    print("=" * 72)
    print(f"参数量            {r['params_total']:,}  (可训练 {r['params_trainable']:,})")
    print(f"推理 单视角       {r['inference_ms_per_view_gpu_only']['median']:.2f} ms"
          f"  (纯 GPU 计算, 中位数, n={r['inference_ms_per_view_gpu_only']['n_iter']})")
    print(f"推理 单视角端到端 {r['inference_ms_per_view_end_to_end_h2d']['median']:.2f} ms"
          f"  (含 H2D 拷贝)")
    print(f"推理 batch=8      {r['inference_ms_per_batch8']['median']:.2f} ms"
          f"  → {r['inference_ms_per_batch8']['ms_per_view']:.2f} ms/视角")
    print(f"峰值显存 已分配   {r['peak_memory_allocated_GB']:.2f} GB"
          f"  (reserved {r['peak_memory_reserved_GB']:.2f} GB)")
    tw = r["training_wall_clock"]
    if tw.get("available"):
        print(f"训练墙钟          {tw['minutes']:.1f} min"
              f"  ({tw['epochs_logged']} epoch, {tw['start']} → {tw['end']})")
    print(f"\n已写出 {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
