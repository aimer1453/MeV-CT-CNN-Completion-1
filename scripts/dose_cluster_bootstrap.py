#!/usr/bin/env python
# -*- coding: utf-8 -*-
r"""§5.6 cluster-robust paired bootstrap（只算不改的正式落盘版）。

背景：§5.6 原以 view-level paired bootstrap 95% CI [+0.0005,+0.0038]（equal-weight）
支持"completion 显著差 2.05%"。外审质疑 held-out 视角非独立（同一物体相邻角度相关），
view-level 抽样高估有效样本量。本脚本按 30° sector 做 cluster bootstrap 复核。

数据：runs/_dose_eval_f1.json（4 能量 × 45 test 视角 = 180 个 (energy,view) 观测）。
每个观测 = (CNN RMSE, linear RMSE, n_valid)。CNN 两口径：
  3-seed mean（equal-weight 主口径）  /  headline run（pooled 方差诊断口径）。

两种统计量：
  eq   : mean(CNN_i - lin_i)                            §5.6 equal-weight 主结论
  pool : sqrt(Σ cnn²n/Σn) - sqrt(Σ lin²n/Σn)            Table 5 二次方池化（诊断）

三种抽样：
  view : 视角独立重抽（原 CI，应复现正文 [+0.0005,+0.0038] / [−0.0056,−0.0003]）
  sec12: 30° sector 跨能量聚块（12 块；假设同扇区 4 能量完全相关 → 过于乐观）
  secE : sector×energy 聚块（46 个非空块 / 48 名义；可辩护的保守选择）

用法：
  python scripts/dose_cluster_bootstrap.py            # 打印 + 写 runs/_dose_cluster_bootstrap.json
"""
from __future__ import annotations

import json
from math import sqrt
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
EN = ["1.5", "3", "4.5", "6"]
NB = 20000
SEED = 20261022


def main() -> int:
    d = json.load(open(ROOT / "runs" / "_dose_eval_f1.json"))
    cks = d["run_ids"]
    rng = np.random.default_rng(SEED)

    c, c0, l, n, th, en = [], [], [], [], [], []
    for e in EN:
        for p in d["energies"][e]["per_sample_test"]:
            c.append(float(np.mean([p[k] for k in cks])))
            c0.append(float(p[cks[0]]))
            l.append(float(p["linear_interp"]))
            n.append(float(p["n_valid"]))
            th.append(int(p["angle"]))
            en.append(e)
    c, c0, l, n = map(np.array, (c, c0, l, n))
    th, en = np.array(th), np.array(en)
    N = len(c)

    def pool(arr, idx):
        return sqrt((arr[idx] ** 2 * n[idx]).sum() / n[idx].sum())

    def ci(v):
        return float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))

    def stat_eq(idx):
        return float((c[idx] - l[idx]).mean())

    def stat_pool(idx):
        return float(pool(c0, idx) - pool(l, idx))

    # 聚块定义
    g12 = (th // 30).astype(int)
    gE = np.array([f"{int(t // 30)}|{e}" for t, e in zip(th, en)])
    blocks = {"view": None, "sec12": g12, "secE": gE}

    out = {"n_obs": N, "run_ids": cks, "n_boot": NB, "seed": SEED,
           "headline_run": cks[0], "results": {}}

    for tag, stat in (("eq", stat_eq), ("pool", stat_pool)):
        cv = c0 if tag == "pool" else c  # eq 用 3-seed mean；pool 用 headline run
        point = stat(np.arange(N))
        # view-level iid（复现正文 CI）
        idx = rng.integers(0, N, (NB, N))
        vals = np.array([stat(ix) for ix in idx])
        lo, hi = ci(vals)
        w0 = hi - lo
        out["results"][tag] = {"point": point, "view": {"lo": lo, "hi": hi, "width": w0}}
        # cluster 版
        for btag in ("sec12", "secE"):
            g = blocks[btag]
            uniq = np.unique(g)
            G = len(uniq)
            members = [np.where(g == u)[0] for u in uniq]
            vals = np.empty(NB)
            for i in range(NB):
                pick = rng.integers(0, G, G)
                vals[i] = stat(np.concatenate([members[p] for p in pick]))
            lo, hi = ci(vals)
            out["results"][tag][btag] = {
                "lo": lo, "hi": hi, "width": hi - lo,
                "G": int(G), "deff": (hi - lo) / w0, "cross_zero": bool(lo < 0 < hi),
            }
        out["results"][tag]["view"]["cross_zero"] = bool(
            out["results"][tag]["view"]["lo"] < 0 < out["results"][tag]["view"]["hi"])

    # 逐能量分解（解释为什么 12 块过于乐观：能量间相关被假设为 1）
    per_e = {}
    for e in EN:
        m = en == e
        per_e[e] = {"n": int(m.sum()), "delta_eq": float((c[m] - l[m]).mean())}
    out["per_energy_delta_eq"] = per_e

    r = out["results"]
    print(f"N={N}  NB={NB}  seed={SEED}")
    for tag, name in (("eq", "EQUAL-WEIGHT 3-seed mean（§5.6 主结论，+=completion 更差）"),
                      ("pool", "POOLED headline run（方差诊断，−=completion 更好）")):
        print(f"\n[{tag}] {name}")
        print(f"  point = {r[tag]['point']:+.5f}")
        v = r[tag]["view"]
        print(f"  view  CI=[{v['lo']:+.5f},{v['hi']:+.5f}]  width={v['width']:.5f}  "
              f"cross0={v['cross_zero']}")
        for b in ("sec12", "secE"):
            x = r[tag][b]
            print(f"  {b:5s} CI=[{x['lo']:+.5f},{x['hi']:+.5f}]  width={x['width']:.5f}  "
                  f"G={x['G']}  DEFF={x['deff']:.2f}  cross0={x['cross_zero']}")

    dst = ROOT / "runs" / "_dose_cluster_bootstrap.json"
    dst.write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(f"\n[json] {dst}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
