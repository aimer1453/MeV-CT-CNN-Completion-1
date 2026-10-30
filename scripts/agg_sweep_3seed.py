#!/usr/bin/env python
# -*- coding: utf-8 -*-
r"""
scripts/agg_sweep_3seed.py —— 把均匀域跨度扫描的单 run 评估聚合成 3 种子均值

为什么需要
----------
Fig 8 最初用的是 `sweep100_s<s>.json`，那是**单个 run（seed 20261022）**的评估。
论文侧 R48 指出：单 run 与 3 种子的结论不一致（交叉点在 5.98° vs 6.14°，
与 Fig 10 的 6.92° 相减得 0.92° vs 0.78°），要求 **Fig 8 全转 3 种子**。

因此需要把每档的 3 个 run 各自评估一次（`p02_sweep_eval.py`），再取均值，
产出一份与单 run 版**结构完全相同**的 `sweep100_3seed_s<s>.json`，
这样 `plot_p02_sweep.py --tag 100_3seed` 不必改任何绘图代码。

聚合规则
--------
* `float` 叶子 → 三种子算术平均（ddof=0 的 SD 另存于 `_sd` 键）。
* `int` 叶子（`n_valid_pixels` / `n_samples` / `n_views_completed` 等）→ 取第一个，
  因为它们**必须**在三者之间一致；不一致直接报错（口径被破坏的信号）。
* 其他（字符串、列表、字典键）→ 取第一个 run 的值。
* 额外写入 `_seeds` / `_runs` / `_n_seeds`，便于溯源。

输入（由 runs/_reviewer_p0/ 提供）
----------------------------------
    sweep100_s<s>.json             seed 20261022（既有主 run）
    sweep100_s2_s23.json / _s24     s=2 的另两个种子
    sweep100_s<s>_s25.json / _s26   s=4/6/8 的另两个种子

缺失的输入会被列为 `[MISS]` 并跳过该 span，**不补造**。

输出
----
    runs/_reviewer_p0/sweep100_3seed_s<s>.json

用法
----
    python scripts/agg_sweep_3seed.py
    python scripts/agg_sweep_3seed.py --dry-run
"""
from __future__ import annotations

import argparse
import json
import statistics as st
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "runs" / "_reviewer_p0"
MANIFEST = ROOT / "runs" / "_r34_span_sweep.json"

STEPS = (2, 4, 6, 8, 10, 12)
INT_KEYS = {"n_valid_pixels", "n_samples", "n_views_completed", "n_directions",
            "n_measured_views", "n_completed_views"}


def files_for(step: int) -> list[Path]:
    """该跨度下 3 个种子的评估文件（不存在的会被上层判 MISS）。

    命名规则不统一是历史造成的：
    * s=2 的主 run 文件名不带后缀，另两个种子写作 `_s23/_s24`；
    * s=4/6/8 的主 run 同样不带后缀，另两个写作 `_s25/_s26`；
    * s=10/12 是 2026-10-30 新训的（`uspan` 系列），三个种子都带 `_s22/_s23/_s24`。
    """
    if step == 2:
        return [SRC / "sweep100_s2.json",
                SRC / "sweep100_s2_s23.json",
                SRC / "sweep100_s2_s24.json"]
    if step >= 10:
        return [SRC / f"sweep100_s{step}_s22.json",
                SRC / f"sweep100_s{step}_s23.json",
                SRC / f"sweep100_s{step}_s24.json"]
    return [SRC / f"sweep100_s{step}.json",
            SRC / f"sweep100_s{step}_s25.json",
            SRC / f"sweep100_s{step}_s26.json"]


def merge(vals: list, key: str = ""):
    """递归聚合：float 取均值（SD 存 _sd），int 要求三者一致，其余取第一个。"""
    first = vals[0]
    if isinstance(first, dict):
        out = {}
        for k in first:
            sub = [v.get(k) for v in vals]
            if any(s is None for s in sub):
                out[k] = first[k]
                continue
            out[k] = merge(sub, k)
        return out
    if isinstance(first, list):
        return first                      # 列表（如 per_seed）不从单 run 汇总
    if isinstance(first, bool) or not isinstance(first, (int, float)):
        return first
    if key in INT_KEYS:
        if len(set(vals)) != 1:
            raise SystemExit(f"[FATAL] 整数字段 {key!r} 在三种子间不一致：{vals}")
        return first
    return st.mean(vals)


def merge_with_sd(vals: list, key: str = "") -> dict:
    """与 merge 相同，但 float 叶子返回 {mean, sd, values} 结构供核查。"""
    first = vals[0]
    if isinstance(first, dict):
        return {k: merge_with_sd([v[k] for v in vals], k) for k in first
                if all(k in v for v in vals)}
    if isinstance(first, list):
        return {"values": first}
    if isinstance(first, bool) or not isinstance(first, (int, float)):
        return {"value": first}
    if key in INT_KEYS:
        if len(set(vals)) != 1:
            raise SystemExit(f"[FATAL] 整数字段 {key!r} 在三种子间不一致：{vals}")
        return {"value": first}
    return {"mean": st.mean(vals),
            "sd": st.pstdev(vals) if len(vals) > 1 else 0.0,
            "values": [float(v) for v in vals]}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    runs_of = {}
    if MANIFEST.is_file():
        j = json.loads(MANIFEST.read_text(encoding="utf-8"))
        for k, rid in j.get("manifest", {}).items():
            runs_of.setdefault(int(k.split("_")[0].lstrip("s")), []).append(rid)

    ok = 0
    for s in STEPS:
        fs = files_for(s)
        missing = [f.name for f in fs if not f.is_file()]
        if missing:
            print(f"[MISS] s={s}° 缺 {missing}")
            continue
        ds = [json.loads(f.read_text(encoding="utf-8")) for f in fs]
        # 口径一致性：三份必须是同一个 step
        if len({d["step_deg"] for d in ds}) != 1:
            raise SystemExit(f"[FATAL] s={s}° 三份文件的 step_deg 不一致")
        out = merge(ds)
        out["_n_seeds"] = len(ds)
        out["_runs"] = [d.get("cnn_run_id") for d in ds]
        out["_source_files"] = [f.name for f in fs]
        # 带 SD 的副本，供核对（不参与绘图）
        out["_sd_block"] = merge_with_sd(ds)
        p = SRC / f"sweep100_3seed_s{s}.json"
        if not a.dry_run:
            p.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
        r16c = out["projection"]["cnn"]["row16"]["overall"]["rmse_valid"]
        r16l = out["projection"]["linear"]["row16"]["overall"]["rmse_valid"]
        img = out["image"]["cnn"]["overall"]["rmse_valid"]
        print(f"[OK] s={s}°  n_seeds={len(ds)}  row16 cnn={r16c:.6f} "
              f"linear={r16l:.6f}  ratio={r16c / r16l:.4f}  image cnn={img:.6f}"
              f"  -> {p.name}")
        ok += 1
    print(f"\n[DONE] 聚合 {ok} 个跨度")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
