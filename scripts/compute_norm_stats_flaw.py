"""P0-1 第三步：为 3_3_flaw/ 重算 norm 统计量（口径与 compute_norm_stats.py 一致）。

口径（与主数据同一套，直接 import 主脚本函数，禁止重写公式）：
  L = -log((I + 1e-10) / (E + 1e-10))，E = 同 scan_dir 的 results/empty_proj.npy
  valid = (I > 0) & (E > 0)
  zero_log_fill = ceil_to_0_1(valid p99.9)
  norm = valid 的 mean / std（norm_scope=global，但本数据只有 3.0 MeV 一档）

额外做「target 集 vs 输入文件并集」的同口径对比（主脚本的 BLOCK 判据：
|Δmean| > 0.05 或 |Δstd|/std > 5%），确保 train target 的代表性。

输出：docs/norm_stats_flaw.json（提案，供 config_flaw.json 使用）
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from data.csv_paths import resolve_csv  # noqa: E402  CSV 统一落点 csv_data/
sys.path.insert(0, str(ROOT / "scripts"))

import compute_norm_stats as cns  # noqa: E402  同一套口径

SPARSE = resolve_csv("sparse_targets_flaw.csv")
OUT_JSON = ROOT / "docs" / "norm_stats_flaw.json"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sparse", default=str(SPARSE),
                    help="缺陷稀疏角数据表；多能量时传 sparse_targets_flaw_<E>.csv")
    ap.add_argument("--out", default=str(OUT_JSON))
    args = ap.parse_args()

    rows = []
    with open(args.sparse, newline="", encoding="utf-8") as fh:
        import csv as _csv
        for r in _csv.DictReader(fh):
            rows.append(r)
    train = [r for r in rows if r["split"] == "train"]
    if len(train) != 270:
        print(f"[FATAL] train 样本 {len(train)} != 270", file=sys.stderr)
        return 2
    energy = float(train[0]["energy_MeV"])

    # scan_dir -> empty_proj
    scan_path: dict[str, Path] = {}
    for r in train:
        scan_path.setdefault(r["scan_dir"], (ROOT / r["target_file"]).parent.parent)
    empty_cache: dict[str, np.ndarray] = {}

    def empty_of(sd: str) -> np.ndarray:
        if sd not in empty_cache:
            empty_cache[sd] = np.load(scan_path[sd] / "results" / "empty_proj.npy").astype(np.float64)
        return empty_cache[sd]

    def accumulate(files: list[str], label: str) -> dict:
        s = s2 = 0.0
        n = n_invalid = n_cov = 0
        vals_chunks = []
        for i, rel in enumerate(files):
            img = np.load(ROOT / rel).astype(np.float32)
            if img.shape != (32, 1200):
                raise SystemExit(f"[FATAL] shape {img.shape}: {rel}")
            sd = rel.split("/projections/")[0].split("/")[-1]
            L, valid = cns.log_domain(img, empty_of(sd))
            v = L[valid]
            vals_chunks.append(v)
            s += float(v.sum())
            s2 += float((v.astype(np.float64) ** 2).sum())
            n += int(v.size)
            n_invalid += int((~valid).sum())
            n_cov += int(np.count_nonzero(img > 0))
            if (i + 1) % 90 == 0:
                print(f"  [{label}] {i+1}/{len(files)}", file=sys.stderr)
        vs = np.sort(np.concatenate(vals_chunks))
        mean = s / n
        std = math.sqrt(max(s2 / n - mean * mean, 0.0))
        p999 = float(np.percentile(vs, 99.9))
        return {
            "label": label,
            "n_files": len(files),
            "n_valid": int(n),
            "n_invalid": n_invalid,
            "n_total": int(n + n_invalid),
            "valid_coverage": n / (n + n_invalid),
            "coverage": n_cov / (n + n_invalid),
            "mean": mean,
            "std": std,
            "p50": float(np.percentile(vs, 50)),
            "p99": float(np.percentile(vs, 99)),
            "p99_9": p999,
            "zero_log_fill": cns.ceil_to_0_1(p999),
        }

    tgt = accumulate([r["target_file"] for r in train], "target")
    inp_files = sorted({r[k] for r in train for k in ("ch_prev_file", "ch0_file", "ch_next_file")})
    inp = accumulate(inp_files, "input_union")

    print("=" * 76)
    print(f"flaw 数据 norm 统计（{energy:g} MeV，split=train，源表 {args.sparse}）")
    print(f"{'口径':<14}{'n_files':>8}{'n_valid':>12}{'mean':>10}{'std':>9}"
          f"{'p99.9':>9}{'fill':>7}{'valid_cov':>11}")
    for d in (tgt, inp):
        print(f"{d['label']:<14}{d['n_files']:>8}{d['n_valid']:>12}"
              f"{d['mean']:>10.4f}{d['std']:>9.4f}{d['p99_9']:>9.4f}"
              f"{d['zero_log_fill']:>7.2f}{d['valid_coverage']:>11.4f}")

    d_mean = abs(inp["mean"] - tgt["mean"])
    d_std_rel = abs(inp["std"] - tgt["std"]) / tgt["std"]
    print(f"\n代表性检查（主脚本 BLOCK 判据）：|Δmean| {d_mean:.4f}（阈 0.05）；"
          f"|Δstd|/std {d_std_rel*100:.2f}%（阈 5%）→ "
          f"{'BLOCK' if (d_mean > 0.05 or d_std_rel > 0.05) else 'OK'}")

    fill = tgt["zero_log_fill"]
    scan_dirs = sorted({r["scan_dir"] for r in train})
    data_root = train[0]["target_file"].split("/")[0]
    out = {
        "energy_MeV": energy,
        "source": f"{data_root}/{','.join(scan_dirs)}，{args.sparse} "
                  "split=train 的 270 个 target 文件",
        "sparse_table": args.sparse,
        "norm_scope": f"global（本 run 只训练 {energy:g} MeV 一档，故等于该档统计量）",
        "norm_mean": round(tgt["mean"], 4),
        "norm_std": round(tgt["std"], 4),
        "zero_log_fill": fill,
        "zero_fill_normalized": (fill - tgt["mean"]) / tgt["std"],
        "valid": {k: tgt[k] for k in ("n_valid", "n_invalid", "n_total",
                                      "valid_coverage", "coverage", "p50", "p99")},
        "p99_9_raw": tgt["p99_9"],
        "representativeness": {"delta_mean": d_mean, "delta_std_rel": d_std_rel,
                               "blocked": bool(d_mean > 0.05 or d_std_rel > 0.05),
                               "input_union": inp},
        "compare_full_config": {"norm_mean": 0.7202, "norm_std": 0.9249,
                                "zero_log_fill": 5.5,
                                "note": "主 config 的全局值（四档 1080 文件）；本文件仅供 flaw 训练"},
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n已写出 {out_path}")
    print(f"norm_mean={out['norm_mean']}  norm_std={out['norm_std']}  "
          f"zero_log_fill={out['zero_log_fill']}  "
          f"zero_fill_normalized={out['zero_fill_normalized']:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
