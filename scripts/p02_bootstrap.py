#!/usr/bin/env python
"""P0-2：用 bootstrap 置信区间替换 "resolved / unresolved" 二分表述。

统计单元
--------
  row16（投影域中心行）：test 划分的**逐样本** row16 RMSE（每能量 45 个角度）。
     配对：同一样本上 d_i = RMSE_CNN,i − RMSE_linear,i，因为有线插值无种子。
     CNN 有 3 个种子 → 主估计用"3 种子先平均再配对"，另报逐种子 CI 的极差。
  图像域：单元是 (能量, 种子)。4 档 × 3 种子 = 12 个配对差；
     **这一层的 n 很小，CI 只能当作"跨能量泛化"的粗界**，脚本里显式标注。
     另附一个**像素级** bootstrap 作次要参考（n≈1.13M 像素，但像素强相关，
     会给出过窄的区间，仅用于说明"区内误差"与"跨能量误差"是两个量级）。

输出
----
  mean diff、95% CI（百分位法）、Cohen's d（配对 d_z）、相对变化与比值、
  双单侧检验（TOST 等效界 ±2%）的 p 值。

用法
----
    python scripts/p02_bootstrap.py --eval-json runs/_dose_eval_f1.json
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
RUNS = ROOT / "runs"
NAN = float("nan")


def boot_mean_ci(d: np.ndarray, n_boot: int = 10000, seed: int = 20261026,
                 alpha: float = 0.05) -> dict:
    """配对差的均值 + 百分位 bootstrap CI + Cohen's d_z。"""
    d = np.asarray(d, dtype=np.float64)
    d = d[np.isfinite(d)]
    n = d.size
    if n < 3:
        return {"n": n}
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(n_boot, n))
    means = d[idx].mean(axis=1)
    lo, hi = np.percentile(means, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    sd = float(d.std(ddof=1))
    mean = float(d.mean())
    # 比值：成对求 CNN/linear 之外再取 log 均值（更稳），这里由调用方补充
    return {"n": n, "mean_diff": mean, "sd_diff": sd,
            "ci_lo": float(lo), "ci_hi": float(hi),
            "cohen_dz": mean / sd if sd > 0 else NAN,
            "cohen_dz_ci": [float(lo / sd), float(hi / sd)] if sd > 0 else [NAN, NAN],
            "frac_negative": float((d < 0).mean()),
            "boot_sd_of_mean": float(means.std(ddof=1))}


def tost(d: np.ndarray, margin: float = 0.02) -> dict:
    """双单侧检验（等效界 ±margin，绝对量）：H0: |mean| >= margin。"""
    d = np.asarray(d, dtype=np.float64)
    n = d.size
    m, s = float(d.mean()), float(d.std(ddof=1))
    se = s / math.sqrt(n)
    if se <= 0:
        return {"p_tost": NAN}
    from math import erf, sqrt

    def cdf(x):          # 标准正态 CDF
        return 0.5 * (1.0 + erf(x / sqrt(2.0)))

    # 下限侧: H1: mean > -margin  -> p1 = P(T > (m+margin)/se)
    p1 = 1.0 - cdf((m + margin) / se)
    # 上限侧: H1: mean < +margin  -> p2 = P(T < (m-margin)/se)
    p2 = cdf((m - margin) / se)
    return {"p_tost": max(p1, p2), "margin": margin, "se": se}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval-json", default=str(RUNS / "_dose_eval_f1.json"))
    ap.add_argument("--n-boot", type=int, default=10000)
    ap.add_argument("--out", default=str(RUNS / "_p02_bootstrap.json"))
    args = ap.parse_args()

    dat = json.loads(Path(args.eval_json).read_text(encoding="utf-8"))
    run_ids = [r for r in dat["run_ids"]]
    if not run_ids:
        raise SystemExit("[FATAL] eval JSON 里没有 CNN run")
    print(f"[f={dat['factor']:g}] CNN种子 {len(run_ids)} 个: {run_ids}")

    report: dict = {"source": str(Path(args.eval_json).name),
                    "factors": dat["factor"], "n_boot": args.n_boot,
                    "row16": {}, "image": {}, "run_ids": run_ids}

    # ---------------- row16：逐样本配对
    all_d_perseed: dict[str, list[float]] = {r: [] for r in run_ids}
    all_d_mean: list[float] = []
    by_energy: dict[str, dict] = {}
    for ekey, blk in dat["energies"].items():
        ps = blk["per_sample_test"]
        if not ps:
            continue
        lin = np.array([p["linear_interp"] for p in ps])
        cub = np.array([p["cubic_interp"] for p in ps])
        cnn = np.array([[p[r] for p in ps] for r in run_ids])          # (nseed, n)
        cnn_mean = cnn.mean(axis=0)
        d_mean = cnn_mean - lin
        all_d_mean.extend(d_mean.tolist())
        for k, r in enumerate(run_ids):
            all_d_perseed[r].extend((cnn[k] - lin).tolist())
        row = {"n_samples": len(ps),
               "cnn_mean": float(cnn_mean.mean()), "linear": float(lin.mean()),
               "cubic": float(cub.mean()),
               "cnn_vs_linear": boot_mean_ci(d_mean, args.n_boot),
               "cnn_vs_cubic": boot_mean_ci(cnn_mean - cub, args.n_boot),
               "rel_cnn_vs_linear_pct": float(100 * (cnn_mean.mean() / lin.mean() - 1)),
               "rel_cnn_vs_cubic_pct": float(100 * (cnn_mean.mean() / cub.mean() - 1)),
               "perseed": {r: boot_mean_ci(cnn[k] - lin, args.n_boot)
                           for k, r in enumerate(run_ids)}}
        by_energy[ekey] = row
    report["row16"]["by_energy"] = by_energy
    pooled = boot_mean_ci(np.array(all_d_mean), args.n_boot)
    # 池化相对变化：按能量样本数加权
    tot = sum(v["n_samples"] for v in by_energy.values())
    cnn_all = sum(v["cnn_mean"] * v["n_samples"] for v in by_energy.values()) / tot
    lin_all = sum(v["linear"] * v["n_samples"] for v in by_energy.values()) / tot
    cub_all = sum(v["cubic"] * v["n_samples"] for v in by_energy.values()) / tot
    report["row16"]["pooled"] = {
        **pooled, "cnn_mean": float(cnn_all), "linear": float(lin_all),
        "cubic": float(cub_all),
        "rel_cnn_vs_linear_pct": float(100 * (cnn_all / lin_all - 1)),
        "rel_cnn_vs_cubic_pct": float(100 * (cnn_all / cub_all - 1)),
        "tost": tost(np.array(all_d_mean), 0.02),
        "perseed_ci": {r: boot_mean_ci(np.array(v), args.n_boot)
                       for r, v in all_d_perseed.items()},
    }

    # ---------------- 图像域：单元 = (能量, 种子)
    e_keys = list(dat["energies"])
    d_img = []
    rows_img = []
    for ekey in e_keys:
        idm = dat["energies"][ekey]["image_domain"]
        if not idm:
            continue
        lin = float(idm["linear_interp"]["rmse_valid"])
        cub = float(idm["cubic_interp"]["rmse_valid"])
        for r in run_ids:
            if r not in idm:
                continue
            v = float(idm[r]["rmse_valid"])
            d_img.append(v - lin)
            rows_img.append({"energy": ekey, "run": r, "cnn": v, "linear": lin, "cubic": cub})
    d_img = np.array(d_img)
    rep_img = {"n_units": int(d_img.size),
               "unit_note": "单元 = (能量, 种子)；n 很小，CI 只是跨能量泛化的粗界",
               "cnn_vs_linear": boot_mean_ci(d_img, args.n_boot)}
    if d_img.size:
        lin_mean = float(np.mean([x["linear"] for x in rows_img]))
        cnn_mean = float(np.mean([x["cnn"] for x in rows_img]))
        cub_mean = float(np.mean([x["cubic"] for x in rows_img]))
        rep_img.update({"cnn_mean": cnn_mean, "linear": lin_mean, "cubic": cub_mean,
                        "rel_cnn_vs_linear_pct": 100 * (cnn_mean / lin_mean - 1),
                        "rel_cnn_vs_cubic_pct": 100 * (cnn_mean / cub_mean - 1),
                        "per_unit": rows_img})
    report["image"] = rep_img

    # ---------------- 打印
    p = report["row16"]["pooled"]
    print("\n================ row16（投影域中心行，test 逐样本配对）================")
    print(f"  样本数 n = {p['n']}（4 档 × 45 角度）")
    print(f"  CNN  {p['cnn_mean']:.6f}   linear {p['linear']:.6f}   cubic {p['cubic']:.6f}")
    print(f"  Δ(CNN−linear) = {p['mean_diff']:+.6f}   95% CI [{p['ci_lo']:+.6f}, {p['ci_hi']:+.6f}]")
    print(f"  相对变化 = {p['rel_cnn_vs_linear_pct']:+.2f} %   "
          f"Cohen's d_z = {p['cohen_dz']:+.3f} "
          f"CI [{p['cohen_dz_ci'][0]:+.3f}, {p['cohen_dz_ci'][1]:+.3f}]")
    print(f"  负差占比 = {p['frac_negative']*100:.1f} %   TOST(±2%) p = {p['tost']['p_tost']:.2e}")
    print("\n  逐能量：")
    for ekey, v in by_energy.items():
        c = v["cnn_vs_linear"]
        print(f"    {ekey:>4s} MeV  n={v['n_samples']:2d}  CNN={v['cnn_mean']:.6f}  "
              f"lin={v['linear']:.6f}  Δ={c['mean_diff']:+.6f} "
              f"[{c['ci_lo']:+.6f},{c['ci_hi']:+.6f}]  {v['rel_cnn_vs_linear_pct']:+.2f}%")
    print("\n  逐种子 CI（全 4 档池化）：")
    for r, v in p["perseed_ci"].items():
        print(f"    {r}  Δ={v['mean_diff']:+.6f}  [{v['ci_lo']:+.6f},{v['ci_hi']:+.6f}]")

    if rep_img.get("cnn_vs_linear"):
        c = rep_img["cnn_vs_linear"]
        print("\n================ 图像域（单元 = 能量×种子）================")
        print(f"  单元数 {rep_img['n_units']}（{len(e_keys)} 档 × {len(run_ids)} 种子）")
        print(f"  CNN {rep_img['cnn_mean']:.6f}  linear {rep_img['linear_mean'] if 'linear_mean' in rep_img else rep_img['linear']:.6f}")
        print(f"  Δ = {c['mean_diff']:+.6f}  95% CI [{c['ci_lo']:+.6f}, {c['ci_hi']:+.6f}]  "
              f"d_z={c['cohen_dz']:+.3f}  相对 {rep_img['rel_cnn_vs_linear_pct']:+.2f} %")

    Path(args.out).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n[json] {Path(args.out)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
