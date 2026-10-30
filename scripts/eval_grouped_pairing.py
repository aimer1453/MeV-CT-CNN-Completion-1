#!/usr/bin/env python
# -*- coding: utf-8 -*-
r"""grouped vs random 严格配对评估（3.0 MeV）。两个指标域：Row 16 / all-32-row。

目的：把 100-epoch 的 grouped split（0–179d 训练 / 180–359d 外推）与同 epoch 数的
random split 放在**同一批视角**上比较，避免「各自 held-out 不同 -> 不可比」。

三个配对域（一次性全部输出）：
  D1  grouped test 全域   : 180–359d 共 180 个视角（grouped 的 held-out）
  D2  random test 45 视角 : random 模型的 held-out（grouped 模型在此多为 in-domain）
  D3  双 held-out 交集     : D1 ∩ D2，两边都没见过 —— 唯一「严格配对」子集

两种聚合口径都报：
  view : 逐视角 RMSE 的算术平均（§5.6 等权口径）
  pool : 合并平方误差再开方（Table 5 二次方口径）

用法：
  python scripts/eval_grouped_pairing.py [--json X]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from data.dataset import SparseCTDataset, load_csv_rows  # noqa: E402
from models.unet import UNet  # noqa: E402

ROW = 16
LOG_EPS = 1e-10
NORM = (0.7202, 0.9249, 5.50)

RANDOM_RUNS = {
    20261022: "20261024_0400_cnn_bs8_lr1e-4",
    20261023: "20261024_1633_cnn_bs8_lr1e-4",
    20261024: "20261024_1805_cnn_bs8_lr1e-4",
}
GROUPED_RUNS = {
    20261022: "20261029_0217_cnn_bs8_lr1e-4_grouped_s22",
    20261023: "20261029_0217_cnn_bs8_lr1e-4_grouped_s23",
    20261024: "20261029_0225_cnn_bs8_lr1e-4_grouped_s24",
    20261025: "20261030_0155_cnn_bs8_lr1e-4_grouped_s25",
    20261026: "20261030_0211_cnn_bs8_lr1e-4_grouped_s26",
}
KEYS = ("r16", "a32")


def build_ds(csv: str, split: str) -> SparseCTDataset:
    mean, std, fill = NORM
    return SparseCTDataset(split=split, zero_log_fill=fill, norm_mean=mean,
                           norm_std=std, csv_path=csv, n_views=3, single_row=False)


def _pack(sq, m) -> dict:
    """(row16 rmse, row16 n, all32 rmse, all32 n)"""
    return {
        "r16": float(np.sqrt(sq[ROW][m[ROW]].sum() / m[ROW].sum())),
        "n16": int(m[ROW].sum()),
        "a32": float(np.sqrt(sq[m].sum() / m.sum())),
        "na32": int(m.sum()),
    }


@torch.no_grad()
def per_view(model, ds, device) -> dict[int, dict]:
    model.eval()
    out: dict[int, dict] = {}
    for i in range(len(ds)):
        s = ds[i]
        x = torch.from_numpy(s["x"]).unsqueeze(0).to(device)
        pred = np.squeeze(model(x).to(torch.float64).cpu().numpy())   # (32,1200)
        tgt = s["y"][0].astype(np.float64)
        m = s["mask"][0].astype(bool)
        sq = (pred * NORM[1] + NORM[0] - (tgt * NORM[1] + NORM[0])) ** 2
        if m[ROW].sum() == 0:
            continue
        out[int(s["meta"]["target_angle_deg"])] = _pack(sq, m)
    return out


def linear_view(csv: str, split: str) -> dict[int, dict]:
    """线性插值 0.5*(L[θ−1] + L[θ+1])，CSV 的 ch0/ch_next 列。"""
    out: dict[int, dict] = {}
    for r in load_csv_rows(csv, split):
        sd = str(Path(r["target_file"]).parent.parent)
        empty = np.load(ROOT / sd / "results" / "empty_proj.npy").astype(np.float64)
        raw = {k: np.load(ROOT / r[k]).astype(np.float64)
               for k in ("target_file", "ch0_file", "ch_next_file")}
        lg = lambda a: -np.log((a + LOG_EPS) / (empty + LOG_EPS))  # noqa: E731
        t = lg(raw["target_file"])
        lin = 0.5 * (lg(raw["ch0_file"]) + lg(raw["ch_next_file"]))
        m = (raw["target_file"] > 0) & (empty > 0)
        if m[ROW].sum() == 0:
            continue
        out[int(float(r["target_angle_deg"]))] = _pack((lin - t) ** 2, m)
    return out


def agg(d: dict[int, dict], angles, key: str, mode: str) -> float:
    v = np.array([d[a][key] for a in angles])
    if mode == "view":
        return float(v.mean())
    w = np.array([d[a]["n16" if key == "r16" else "na32"] for a in angles], dtype=np.float64)
    return float(np.sqrt((v ** 2 * w).sum() / w.sum()))


def load_model(run: str, device) -> UNet:
    p = ROOT / "runs" / run / "best.pt"
    if not p.is_file():
        raise SystemExit(f"缺权重 {p}")
    m = UNet(in_channels=5).to(device)
    ck = torch.load(p, map_location=device, weights_only=True)
    m.load_state_dict(ck["model"] if isinstance(ck, dict) and "model" in ck else ck, strict=True)
    m.eval()
    return m


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", default="")
    args = ap.parse_args()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dg = build_ds("sparse_targets_3MeV_grouped.csv", "test")
    dr = build_ds("sparse_targets_3MeV.csv", "test")
    ang_g = sorted(int(dg[i]["meta"]["target_angle_deg"]) for i in range(len(dg)))
    ang_r = sorted(int(dr[i]["meta"]["target_angle_deg"]) for i in range(len(dr)))
    inter = sorted(set(ang_g) & set(ang_r))
    dom = {"d1": ang_g, "d2": ang_r, "d3": inter}
    print(f"grouped test n={len(ang_g)}  random test n={len(ang_r)}  交集 n={len(inter)}")

    lin = {"d1": linear_view("sparse_targets_3MeV_grouped.csv", "test"),
           "d2": linear_view("sparse_targets_3MeV.csv", "test")}
    lin["d3"] = lin["d1"]

    res: dict = {"domains": {k: len(v) for k, v in dom.items()},
                 "linear": {k: {kk: agg(lin[k], dom[k], kk, "view") for kk in KEYS} for k in dom},
                 "random": {}, "grouped": {}}
    print("[linear] " + "  ".join(
        f"{k}: r16={res['linear'][k]['r16']:.6f} a32={res['linear'][k]['a32']:.6f}" for k in dom))

    for fam, runs in (("random", RANDOM_RUNS), ("grouped", GROUPED_RUNS)):
        for seed, run in runs.items():
            mdl = load_model(run, dev)
            pg, pr = per_view(mdl, dg, dev), per_view(mdl, dr, dev)
            # D3 严格配对：每个家族用自己的模型在交集视角上的预测
            d3src = pr if fam == "random" else pg
            res[fam][seed] = {
                "d1": {a: pg[a] for a in dom["d1"] if a in pg},
                "d2": {a: pr[a] for a in dom["d2"] if a in pr},
                "d3": {a: d3src[a] for a in inter if a in d3src},
            }
            print(f"[{fam} seed {seed}] " + "  ".join(
                f"{k}: r16={agg(res[fam][seed][k], dom[k], 'r16', 'view'):.6f} "
                f"a32={agg(res[fam][seed][k], dom[k], 'a32', 'view'):.6f}" for k in dom))

    print("\n" + "=" * 80)
    print("CNN / linear 比值   [view = 逐视角算术平均 · pool = 合并平方误差]")
    print("=" * 80)
    summary = {}
    for k in dom:
        keys = [a for a in dom[k] if a in lin[k]]
        summary[k] = {"n_views": len(keys), "by_metric": {}}
        for met in KEYS:
            block = {}
            L = {m: agg(lin[k], keys, met, m) for m in ("view", "pool")}
            for fam in ("random", "grouped"):
                seeds = [s for s in res[fam] if all(a in res[fam][s][k] for a in keys)]
                if not seeds:
                    continue
                rv = [agg(res[fam][s][k], keys, met, "view") / L["view"] for s in seeds]
                rp = [agg(res[fam][s][k], keys, met, "pool") / L["pool"] for s in seeds]
                block[fam] = {
                    "n_seed": len(seeds), "seeds": seeds,
                    "ratio_view": rv, "ratio_pool": rp,
                    "mean_view": float(np.mean(rv)), "sd_view": float(np.std(rv, ddof=1)) if len(rv) > 1 else 0.0,
                    "mean_pool": float(np.mean(rp)),
                    "abs_view_mean": float(np.mean([agg(res[fam][s][k], keys, met, "view") for s in seeds])),
                    "abs_pool": float(np.mean([agg(res[fam][s][k], keys, met, "pool") for s in seeds])),
                }
            # 严格配对（同种子、同视角、同 linear）
            common = [s for s in res["random"] if s in res["grouped"]
                      and all(a in res["random"][s][k] and a in res["grouped"][s][k] for a in keys)]
            if common:
                dv = [agg(res["random"][s][k], keys, met, "view") - agg(res["grouped"][s][k], keys, met, "view")
                      for s in common]
                block["paired"] = {"seeds": common, "diff_view": dv, "mean_diff_view": float(np.mean(dv))}
            summary[k]["by_metric"][met] = {"linear": L, **block}

            print(f"\n--- {k} ({met}, n={len(keys)})   linear view={L['view']:.6f} pool={L['pool']:.6f}")
            for fam in ("random", "grouped"):
                if fam in block:
                    b = block[fam]
                    print(f"  {fam:8s} ratio(view)={b['mean_view']:.4f}±{b['sd_view']:.4f}  "
                          f"ratio(pool)={b['mean_pool']:.4f}  "
                          f"abs(view)={b['abs_view_mean']:.6f}  "
                          f"逐种子 {', '.join(f'{x:.4f}' for x in b['ratio_view'])}")
            if "paired" in block:
                p = block["paired"]
                print(f"  配对差(random−grouped, view) {', '.join(f'{x:+.6f}' for x in p['diff_view'])}"
                      f"  → {p['mean_diff_view']:+.6f}")

    res["summary"] = summary
    if args.json:
        Path(args.json).write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"\n[json] {args.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
