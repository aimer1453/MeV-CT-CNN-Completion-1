#!/usr/bin/env python
# -*- coding: utf-8 -*-
r"""
scripts/r34_grouped_eval.py —— R34 ①grouped split 评估（3.0 MeV，s=2 几何）

与 scripts/p02_sweep_eval.py 完全同式（同 in =180 度参考、同 hamming FBP、
同 FOV/SSIM/psnr_common、同 per-pixel mask），只改两处：
  · 只评估 3.0 MeV 单档（grouped CSV 就是单能量）；
  · split 从**指定 CSV** 读（grouped = 扇区划分，random = 原随机划分），
    因此同一脚本可评估两种划分，口径严格一致。

三项
----
  投影域 row16 / all32（只统计 split=="test" 的目标视角，mask 加权）
  图像域 RMSE / PSNR / SSIM（对 360-view dense hamming FBP 参考）
  线性/三次插值基线同一脚本内重算，必然与 CNN 同坐标

用法
----
  python scripts/r34_grouped_eval.py --cnn-run <run_id> \
      --csv csv_data/sparse_targets_3MeV_grouped.csv --out runs/_r34_grp2d_sXX.json
  # 只算固定规则基线（不需要 CNN）：加 --baselines-only
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
sys.path.insert(0, str(ROOT))
for p in (ROOT, ROOT / "scripts"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from data.csv_paths import resolve_csv  # noqa: E402
import baseline_recon as br              # noqa: E402
import p02_sweep_eval as swe             # noqa: E402

ENERGY = "3"
FILTER = "hamming"


def load_split(csv_path: Path) -> dict[int, str]:
    """angle -> split，从**指定** CSV 读（不用 p02_sweep_eval 里写死的 sparse_targets.csv）。"""
    out: dict[int, str] = {}
    with open(csv_path, newline="", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            out[int(round(float(r["target_angle_deg"])))] = r["split"]
    return out


def run(args: argparse.Namespace) -> dict:
    step, h = 2, 1
    csv_path = resolve_csv(args.csv)
    split = load_split(csv_path)
    n_test = sum(1 for v in split.values() if v == "test")
    print(f"[split] {csv_path.name}  test={n_test} views", flush=True)

    # 归一化是全局锁定的常量；有 run 时从该 run 的 config 读（带自检），
    # baselines-only 时从根 config.json 读，绝不硬编码。
    cfg_path = (br.RUNS_DIR / args.cnn_run / "config.json") if not args.baselines_only \
        else (ROOT / "config.json")
    if not cfg_path.is_file():
        raise SystemExit(f"[FATAL] 缺少 config: {cfg_path}")
    norm = json.loads(cfg_path.read_text(encoding="utf-8"))["norm"]
    mean, std, fill = float(norm["norm_mean"]), float(norm["norm_std"]), float(norm["zero_log_fill"])

    files = swe.load_angles()
    energies = [ENERGY]

    stacks: dict[str, swe.EnergyStack] = {}
    for e in energies:
        t0 = time.time()
        stacks[e] = swe.EnergyStack(e, files[e], fill)
        print(f"[{e} MeV] 载入 360×32×1200 完成 {time.time() - t0:.1f}s", flush=True)

    dense = {}
    for e in energies:
        st = stacks[e]
        dense[e] = br.restrict_to_fov(br.recon_fbp(st.row16, st.theta, FILTER))
    dr_common = max(float(dense[e].max() - dense[e].min()) for e in energies)

    model = None
    device = None
    if not args.baselines_only:
        import torch
        from models.unet import UNet
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = UNet().to(device)
        ck = torch.load(br.RUNS_DIR / args.cnn_run / "best.pt", map_location=device,
                        weights_only=False)
        model.load_state_dict(ck["model"])
        model.eval()
        print(f"[CNN] {args.cnn_run}  epoch={ck['epoch']}  "
              f"val_rmse_valid_phys={ck['val_rmse_valid_phys']:.6f}", flush=True)

    mask_fov = br.fov_mask()
    per_sample: list[dict] = []
    acc: dict[str, dict[str, dict[str, list]]] = {
        m: {r: {e: [0.0, 0] for e in energies} for r in ("row16", "all32")}
        for m in ("cnn", "linear", "cubic")}

    for c in range(step):
        for e in energies:
            st = stacks[e]
            meas = [a for a in range(360) if a % step == c]
            targ = [a for a in range(360) if a % step == (c + h) % step]

            lin32 = np.empty((len(targ), 32, 1200), dtype=np.float64)
            cub32 = np.empty((len(targ), 32, 1200), dtype=np.float64)
            for i, th in enumerate(targ):
                lin32[i] = 0.5 * (st.L[(th - h) % 360] + st.L[(th + h) % 360])
                cub32[i] = (9.0 / 16.0) * (st.L[(th - h) % 360] + st.L[(th + h) % 360]) \
                    - (1.0 / 16.0) * (st.L[(th - 3 * h) % 360] + st.L[(th + 3 * h) % 360])
            if model is not None:
                cnn32 = swe.predict(model, device, [(e, th) for th in targ], stacks, files,
                                    step, mean, std, fill).astype(np.float64)
            else:
                cnn32 = None

            def compose(fill_rows: np.ndarray | None) -> tuple[np.ndarray, np.ndarray]:
                if fill_rows is None:
                    ang = np.array(meas, dtype=float)
                    return st.row16[meas], ang
                sino = st.row16.copy()
                ang = np.arange(360, dtype=float)
                for i, th in enumerate(targ):
                    sino[th] = fill_rows[i, br.RECON_ROW, :]
                sel = sorted(meas + targ)
                return sino[sel], ang[sel]

            for name, (fill_rows, meth) in (
                    ("fbp_sparse", (None, None)),
                    ("linear", (lin32, "linear")),
                    ("cubic", (cub32, "cubic")),
                    ("cnn", (cnn32, "cnn"))):
                if name == "cnn" and cnn32 is None:
                    continue
                s_, th_ = compose(fill_rows)
                img = br.restrict_to_fov(br.recon_fbp(s_, th_, FILTER))
                m = br.image_metrics(dense[e], img, mask_fov)
                per_sample.append({
                    "energy_MeV": e, "direction": c, "method": name,
                    "rmse_valid": m["rmse_valid"], "rmse_full": m["rmse_full"],
                    "psnr": m["psnr"], "ssim": m["ssim"],
                    "psnr_common": 10.0 * math.log10(dr_common * dr_common
                                                     / (m["rmse_valid"] ** 2)),
                    "n_views": int(s_.shape[0])})

            for i, th in enumerate(targ):
                if split.get(th) != "test":
                    continue
                m16 = st.valid[th, br.RECON_ROW, :]
                m32 = st.valid[th]
                preds = [("linear", lin32[i]), ("cubic", cub32[i])]
                if cnn32 is not None:
                    preds.append(("cnn", cnn32[i]))
                for meth, pred in preds:
                    for roi, (gt, pr, mk) in (
                            ("row16", (st.row16[th], pred[br.RECON_ROW], m16)),
                            ("all32", (st.L[th].astype(np.float64), pred, m32))):
                        if mk.any():
                            acc[meth][roi][e][0] += float(((pr - gt) ** 2)[mk].sum())
                            acc[meth][roi][e][1] += int(mk.sum())
        print(f"  direction {c + 1}/{step} done", flush=True)

    def agg(method: str) -> dict:
        sub = [s for s in per_sample if s["method"] == method]
        per_e, se_list = {}, []
        for e in energies:
            ss = [s for s in sub if s["energy_MeV"] == e]
            mse = float(np.mean([s["rmse_valid"] ** 2 for s in ss]))
            per_e[e] = {"n_directions": len(ss), "n_views_completed": ss[0]["n_views"],
                        "rmse_valid": math.sqrt(mse),
                        "rmse_full": float(np.mean([s["rmse_full"] for s in ss])),
                        "psnr": float(np.mean([s["psnr"] for s in ss])),
                        "psnr_common": float(np.mean([s["psnr_common"] for s in ss])),
                        "ssim": float(np.mean([s["ssim"] for s in ss]))}
            se_list.append(mse)
        return {"per_energy": per_e,
                "overall": {"n_samples": len(sub),
                            "n_views_completed": sub[0]["n_views"],
                            "rmse_valid": math.sqrt(float(np.mean(se_list))),
                            "rmse_full": float(np.mean([s["rmse_full"] for s in sub])),
                            "psnr": float(np.mean([s["psnr"] for s in sub])),
                            "psnr_common": float(np.mean([s["psnr_common"] for s in sub])),
                            "ssim": float(np.mean([s["ssim"] for s in sub]))}}

    image = {m: agg(m) for m in ("fbp_sparse", "linear", "cubic")}
    if not args.baselines_only:
        image["cnn"] = agg("cnn")

    proj = {}
    for meth in ("cnn", "linear", "cubic"):
        if args.baselines_only and meth == "cnn":
            continue
        proj[meth] = {}
        for roi in ("row16", "all32"):
            per_e = {}
            se = n = 0.0
            for e in energies:
                s_, n_ = acc[meth][roi][e]
                per_e[e] = {"rmse_valid": math.sqrt(s_ / n_) if n_ else float("nan"),
                            "n_valid_pixels": int(n_)}
                se += s_
                n += n_
            proj[meth][roi] = {"per_energy": per_e,
                               "overall": {"rmse_valid": math.sqrt(se / n) if n else float("nan"),
                                           "n_valid_pixels": int(n)}}

    return {"csv": str(args.csv), "n_test_views": n_test,
            "energy_MeV": float(ENERGY), "step_deg": step,
            "cnn_run_id": None if args.baselines_only else args.cnn_run,
            "filter": FILTER, "data_range_common": dr_common,
            "image": image, "projection": proj}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cnn-run", default=None)
    ap.add_argument("--csv", default="csv_data/sparse_targets_3MeV_grouped.csv")
    ap.add_argument("--out", default=None)
    ap.add_argument("--baselines-only", action="store_true")
    args = ap.parse_args(argv)
    if not args.cnn_run and not args.baselines_only:
        raise SystemExit("[FATAL] 需 --cnn-run 或 --baselines-only")
    out = run(args)
    p = Path(args.out) if args.out else br.RUNS_DIR / "_r34_grp_eval.json"
    p.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n[json] {p}")
    pr = out["projection"]
    print("proj row16 : " + "  ".join(
        f"{m}={pr[m]['row16']['overall']['rmse_valid']:.6f}" for m in pr))
    im = out["image"]
    print("image      : " + "  ".join(
        f"{m}={im[m]['overall']['rmse_valid']:.6f}" for m in im))
    return 0


if __name__ == "__main__":
    sys.exit(main())
