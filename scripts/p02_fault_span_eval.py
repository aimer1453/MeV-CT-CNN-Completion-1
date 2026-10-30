#!/usr/bin/env python
# -*- coding: utf-8 -*-
r"""
scripts/p02_fault_span_eval.py —— **缺陷数据**跨度扫描的投影域评估（45 test 视角）

它和 `p02_proj_only.py` 的关系
------------------------------
同口径、同 mask、同池化，只换数据源与 split 表：
  · 数据源 = 边棒缺陷 `3_3_flaw_1_2/`（不是 3_3/）→ empty_proj 在
    `<scan_dir>/results/empty_proj.npy`，由本脚本显式传入 `EnergyStack(empty_path=)`
  · split  = `sparse_targets_flaw12_3.0.csv` 的 270/45/45
    （已核：其 45 个 test 角与主域 3.0 MeV 的 test 集**逐角相同**）
  · 归一化 = 各 run 自己 config 的 norm（缺陷域是 0.7493/0.9335/5.8，
    **不是**主域的 0.7202/0.9249/5.5，绝不可拿主域常数断言）

口径（与论文 §5.6 跨度表一致，禁改）
-----------------------------------
  · 只在 **45 个 test 视角**上统计；每个 θ 只算一次（按 θ mod s 归属唯一方向）
  · row16 = 只算 RECON_ROW；all32 = 全 32 行；mask = (I_target>0) & (empty_proj>0)
  · 三种方法：cnn（best.pt）/ linear ½[L(θ−h)+L(θ+h)] / cubic 9/16·…−1/16·…
  · 池化 = sqrt(Σ误差² / Σmask 像素)

用法
----
    python scripts/p02_fault_span_eval.py --step 4 \
        --cnn-run 20261030_0???_cnn_bs8_lr1e-4_flawspan_s4_s20261022
    python scripts/p02_fault_span_eval.py --step 2 \
        --cnn-run 20261029_2351_cnn_bs8_lr1e-4_flaw12_3.0   # s=2 基准（已训练）
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
for p in (ROOT, ROOT / "scripts"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import baseline_recon as br              # noqa: E402
import p02_sweep_eval as swe             # noqa: E402
from data.csv_paths import resolve_csv   # noqa: E402

OUT_DIR = ROOT / "runs" / "_reviewer_p0"
BASE_CSV = "sparse_targets_flaw12_3.0.csv"
ENERGY = "3"


def load_files_and_split(csv_name: str = BASE_CSV) -> tuple[dict[int, str], dict[int, str], str]:
    """返回 (angle -> 相对路径, angle -> split, scan_dir 相对路径)。"""
    path = resolve_csv(csv_name)
    files: dict[int, str] = {}
    spl: dict[int, str] = {}
    scan = ""
    with open(path, newline="", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            a = int(round(float(r["target_angle_deg"])))
            files[a] = r["target_file"]
            spl[a] = r["split"]
            scan = r["scan_dir"]
    if sorted(files) != list(range(360)):
        raise SystemExit(f"{path.name} 角度未覆盖 0..359")
    return files, spl, scan


def eval_one(model, device, step: int, st: "swe.EnergyStack", files: dict[int, str],
             test_angles: list[int], mean: float, std: float, fill: float,
             n_views: int = 3) -> dict:
    h = step // 2
    acc = {m: {"row16": [0.0, 0], "all32": [0.0, 0]}
           for m in ("cnn", "linear", "cubic")}
    pv = {m: {"row16": [], "all32": []} for m in ("cnn", "linear", "cubic")}
    targ = sorted(test_angles)
    cnn32 = swe.predict(model, device, [(ENERGY, th) for th in targ], {ENERGY: st},
                        {ENERGY: files}, step, mean, std, fill,
                        n_views=n_views).astype(np.float64)
    for i, th in enumerate(targ):
        m16 = st.valid[th, br.RECON_ROW, :]
        m32 = st.valid[th]
        lin = 0.5 * (st.L[(th - h) % 360] + st.L[(th + h) % 360]).astype(np.float64)
        cub = ((9.0 / 16.0) * (st.L[(th - h) % 360] + st.L[(th + h) % 360])
               - (1.0 / 16.0) * (st.L[(th - 3 * h) % 360]
                                 + st.L[(th + 3 * h) % 360])).astype(np.float64)
        for meth, pred in (("cnn", cnn32[i]), ("linear", lin), ("cubic", cub)):
            for roi, (gt, pr, mk) in (
                    ("row16", (st.row16[th], pred[br.RECON_ROW], m16)),
                    ("all32", (st.L[th].astype(np.float64), pred, m32))):
                if mk.any():
                    sse_ = float(((pr - gt) ** 2)[mk].sum())
                    n_ = int(mk.sum())
                    acc[meth][roi][0] += sse_
                    acc[meth][roi][1] += n_
                    pv[meth][roi].append(math.sqrt(sse_ / n_) if n_ else float("nan"))
                else:
                    pv[meth][roi].append(float("nan"))
    out = {}
    for meth in ("cnn", "linear", "cubic"):
        out[meth] = {roi: {"rmse_valid": math.sqrt(s / n) if n else float("nan"),
                           "n_valid_pixels": int(n)}
                     for roi, (s, n) in acc[meth].items()}
    out["n_test_views"] = len(targ)
    out["per_view"] = pv            # 逐视角 RMSE，供配对检验使用（顺序 = test_angles）
    out["test_angles_order"] = targ
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--step", type=int, required=True)
    ap.add_argument("--cnn-run", action="append", required=True)
    ap.add_argument("--csv", default=BASE_CSV,
                    help="用于取 split 与文件表的表（默认 2° 基准表；各跨度 test 集相同）")
    args = ap.parse_args(argv)

    step = args.step
    files, spl, scan = load_files_and_split(args.csv)
    test_angles = sorted(a for a in spl if spl[a] == "test")
    print(f"[s={step}°] scan={scan}  test={len(test_angles)} 视角  runs={len(args.cnn_run)}",
          flush=True)

    import torch
    from models.unet import UNet

    # 归一化取各 run 自己 config 的（缺陷域常数），并断言本批 run 之间一致
    norms = set()
    for rid in args.cnn_run:
        cfg = json.loads((br.RUNS_DIR / rid / "config.json").read_text(encoding="utf-8"))
        n = cfg["norm"]
        norms.add((float(n["norm_mean"]), float(n["norm_std"]), float(n["zero_log_fill"])))
    if len(norms) != 1:
        raise SystemExit(f"[FATAL] 本批 run 的归一化不一致: {norms}")
    mean, std, fill = norms.pop()
    print(f"  norm mean={mean} std={std} fill={fill}", flush=True)

    empty_path = ROOT / Path(files[0]).parent.parent / "results" / "empty_proj.npy"
    if not empty_path.is_file():
        raise SystemExit(f"[FATAL] 缺 empty_proj: {empty_path}")
    t0 = time.time()
    st = swe.EnergyStack(ENERGY, files, fill, empty_path=empty_path)
    print(f"  [3.0 MeV] 载入 360×32×1200  {time.time() - t0:.1f}s", flush=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for rid in args.cnn_run:
        cfg = json.loads((br.RUNS_DIR / rid / "config.json").read_text(encoding="utf-8"))
        n_views = int(cfg.get("input_views") or 3)
        model = UNet(in_channels=n_views + 2).to(device)
        ck = torch.load(br.RUNS_DIR / rid / "best.pt", map_location=device,
                        weights_only=False)
        model.load_state_dict(ck["model"])
        model.eval()
        t1 = time.time()
        res = eval_one(model, device, step, st, files, test_angles, mean, std, fill,
                       n_views=n_views)
        r = res["cnn"]["row16"]["rmse_valid"]
        l = res["linear"]["row16"]["rmse_valid"]
        c = res["cubic"]["row16"]["rmse_valid"]
        a = res["cnn"]["all32"]["rmse_valid"]
        la = res["linear"]["all32"]["rmse_valid"]
        print(f"[{rid}] ep={ck['epoch']} {time.time() - t1:.0f}s  "
              f"row16 cnn={r:.6f} lin={l:.6f} ratio={r / l:.4f} | "
              f"cubic={c:.6f} ratio_c={r / c:.4f} | "
              f"all32 cnn={a:.6f} lin={la:.6f} ratio={a / la:.4f}", flush=True)
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        p = OUT_DIR / f"projonly_flaw_s{step}_{rid}.json"
        p.write_text(json.dumps({"step_deg": step, "half_step_deg": step // 2,
                                 "source": ("3_3_flaw_1_2" if "flaw" in scan else "3_3"),
                                 "energy": ENERGY,
                                 "csv_split": args.csv, "scan_dir": scan,
                                 "test_angles": test_angles,
                                 "cnn_run_id": rid, "input_views": n_views,
                                 "loss_mode": cfg.get("loss_mode", "masked"),
                                 "norm": {"mean": mean, "std": std, "fill": fill},
                                 "cnn_epoch": ck["epoch"], "projection": res},
                                ensure_ascii=False, indent=1), encoding="utf-8")
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
    return 0


if __name__ == "__main__":
    sys.exit(main())
