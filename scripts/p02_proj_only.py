#!/usr/bin/env python
# -*- coding: utf-8 -*-
r"""
scripts/p02_proj_only.py —— 跨度扫描的**投影域专用**评估（多 run 一次载入）

为什么另写一个
--------------
`p02_sweep_eval.py` 每次调用都要（a）重载 4 档 ×360 张投影，（b）对每个方向/能量做
4 次 FBP + SSIM。给 12 个 run（4 跨度 ×3 种子）逐一种子调用，光数据载入就重复 12 次，
实测 s=2 单次已 >2 min，s=8 更久。补种子这件事只需要**投影域 row16 / all32**，
图像域不在本表范围内，因此：

  · 数据只载入一次，然后在同一进程里遍历多个 run（`--cnn-run` 可重复给）；
  · 不做 FBP / SSIM / PSNR，只做 masked RMSE；
  · 其余口径与 `p02_sweep_eval.py` **逐字相同**：同 `load_angles`、同 `EnergyStack`、
    同 `predict`（同 5 通道、同归一化、同 best.pt）、同 mask（I>0 & E>0）、
    同 split（主表 sparse_targets.csv 的 test 角）、同池化（先逐能量再跨能量）。

输出
----
    runs/_reviewer_p0/projonly_s<step>_<run_id>.json
      {"step_deg","half_step_deg","cnn_run_id","projection":{cnn|linear|cubic:
       {row16|all32:{per_energy,overall}}}}

用法
----
    python scripts/p02_proj_only.py --step 8 \
        --cnn-run 20261024_0625_cnn_bs8_lr1e-4 \
        --cnn-run 20261029_0???_cnn_bs8_lr1e-4_step8_s25
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
for p in (ROOT, ROOT / "scripts"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import baseline_recon as br              # noqa: E402
import p02_sweep_eval as swe             # noqa: E402

OUT_DIR = ROOT / "runs" / "_reviewer_p0"


def eval_one(model, device, step: int, stacks, files, split, mean, std, fill,
             energies: list[str], n_views: int = 3) -> dict:
    h = step // 2
    acc = {m: {r: {e: [0.0, 0] for e in energies} for r in ("row16", "all32")}
           for m in ("cnn", "linear", "cubic")}
    for c in range(step):
        for e in energies:
            st = stacks[e]
            targ = [a for a in range(360) if a % step == (c + h) % step]
            cnn32 = swe.predict(model, device, [(e, th) for th in targ], stacks, files,
                                step, mean, std, fill, n_views=n_views).astype(np.float64)
            for i, th in enumerate(targ):
                if split.get((e, th)) != "test":
                    continue
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
                            acc[meth][roi][e][0] += float(((pr - gt) ** 2)[mk].sum())
                            acc[meth][roi][e][1] += int(mk.sum())

    out = {}
    for meth in ("cnn", "linear", "cubic"):
        out[meth] = {}
        for roi in ("row16", "all32"):
            per_e, se = {}, 0.0
            n_tot = 0
            for e in energies:
                s_, n_ = acc[meth][roi][e]
                per_e[e] = {"rmse_valid": math.sqrt(s_ / n_) if n_ else float("nan"),
                            "n_valid_pixels": int(n_)}
                se += s_
                n_tot += n_
            out[meth][roi] = {"per_energy": per_e,
                              "overall": {"rmse_valid": math.sqrt(se / n_tot) if n_tot else float("nan"),
                                          "n_valid_pixels": int(n_tot)}}
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--step", type=int, required=True, choices=(2, 4, 6, 8))
    ap.add_argument("--cnn-run", action="append", required=True,
                    help="可重复；同一进程内依次评估，数据只载入一次")
    ap.add_argument("--energies", default="1.5,3,4.5,6")
    ap.add_argument("--input-views", type=int, default=0, choices=(0, 2, 3),
                    help="0（默认）= 从每个 run 的 config 自动读；2/3 = 强制覆盖（慎用）")
    args = ap.parse_args(argv)

    step = args.step
    energies = [x.strip() for x in args.energies.split(",") if x.strip()]
    files = swe.load_angles()
    split = swe.load_split()
    print(f"[s={step}°] energies={energies}  runs={len(args.cnn_run)}", flush=True)

    # 归一化只从**各 run 自己的 config** 读，并在下面断言与锁定值一致
    import torch
    from models.unet import UNet

    fill0 = None
    stacks = {}
    for e in energies:
        cfg0 = json.loads((br.RUNS_DIR / args.cnn_run[0] / "config.json").read_text(encoding="utf-8"))
        fill0 = float(cfg0["norm"]["zero_log_fill"])
    for e in energies:
        t0 = time.time()
        stacks[e] = swe.EnergyStack(e, files[e], fill0)
        print(f"  [{e} MeV] 载入 360×32×1200  {time.time() - t0:.1f}s", flush=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for run_id in args.cnn_run:
        cfg = json.loads((br.RUNS_DIR / run_id / "config.json").read_text(encoding="utf-8"))
        norm = cfg["norm"]
        mean, std, fill = (float(norm["norm_mean"]), float(norm["norm_std"]),
                           float(norm["zero_log_fill"]))
        if (abs(mean - 0.7202) > 1e-6 or abs(std - 0.9249) > 1e-6 or abs(fill - 5.5) > 1e-6):
            raise SystemExit(f"[FATAL] {run_id} 归一化偏离锁定值: {mean}/{std}/{fill}")
        # 输入视图数 **只从 run 自己的 config 读**，避免手填 --input-views 把 4 通道权重
        # 塞进 5 通道装配（R34c 就踩过这个坑：size mismatch dec0.conv.weight）
        n_views = int(args.input_views or cfg.get("input_views") or 3)
        if n_views not in (2, 3):
            raise SystemExit(f"[FATAL] {run_id} input_views={n_views} 非 2/3")
        model = UNet(in_channels=n_views + 2).to(device)
        ck = torch.load(br.RUNS_DIR / run_id / "best.pt", map_location=device,
                        weights_only=False)
        model.load_state_dict(ck["model"])
        model.eval()
        t0 = time.time()
        proj = eval_one(model, device, step, stacks, files, split, mean, std, fill,
                        energies, n_views=n_views)
        r16 = proj["cnn"]["row16"]["overall"]["rmse_valid"]
        l16 = proj["linear"]["row16"]["overall"]["rmse_valid"]
        a32 = proj["cnn"]["all32"]["overall"]["rmse_valid"]
        la32 = proj["linear"]["all32"]["overall"]["rmse_valid"]
        print(f"[{run_id}] epoch={ck['epoch']}  {time.time() - t0:.0f}s  "
              f"row16 cnn={r16:.6f} lin={l16:.6f} ratio={r16/l16:.4f} | "
              f"all32 cnn={a32:.6f} lin={la32:.6f} ratio={a32/la32:.4f}", flush=True)
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        p = OUT_DIR / f"projonly_s{step}_{run_id}.json"
        p.write_text(json.dumps({"step_deg": step, "half_step_deg": step // 2,
                                 "cnn_run_id": run_id, "energies": energies,
                                 "input_views": n_views, "loss_mode": cfg.get("loss_mode", "masked"),
                                 "cnn_epoch": ck["epoch"], "projection": proj},
                                ensure_ascii=False, indent=1), encoding="utf-8")
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
    return 0


if __name__ == "__main__":
    sys.exit(main())
