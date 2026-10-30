#!/usr/bin/env python
"""生成 config_dose_f<f>.json：主 config 的副本，只替换 norm 段与数据表。

硬约束：norm 统计文件不存在或代表性检查 BLOCK 时必须报错，不得填默认值。
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CFG = ROOT / "config.json"

ap = argparse.ArgumentParser(description="剂量实验 config 生成")
ap.add_argument("--factor", type=float, required=True)
ap.add_argument("--norm-json", default=None)
ap.add_argument("--out", default=None)
ap.add_argument("--csv", default=None)
ap.add_argument("--epochs", type=int, default=100)
args = ap.parse_args()
f = args.factor

norm_p = Path(args.norm_json) if args.norm_json else ROOT / "docs" / f"norm_stats_dose_f{f:g}.json"
out_p = Path(args.out) if args.out else ROOT / f"config_dose_f{f:g}.json"
csv_name = args.csv or f"sparse_targets_dose_f{f:g}.csv"

if not norm_p.is_file():
    raise SystemExit(f"[FATAL] 缺 {norm_p}：先跑 compute_norm_stats.py --csv ...，禁止填默认值")
st = json.loads(norm_p.read_text(encoding="utf-8"))
if st.get("blocked"):
    raise SystemExit("[FATAL] norm 代表性检查 BLOCK，等裁决")

g = st["global"]
mean = float(g["valid"]["mean"])
std = float(g["valid"]["std"])
fill = float(g["zero_log_fill"])
if not (std > 0):
    raise SystemExit(f"[FATAL] norm_std={std} 非法")

cfg = json.loads(CFG.read_text(encoding="utf-8"))
cfg["epochs"] = args.epochs
cfg["run_id"] = ("PENDING: train.py 启动时按 runs/<YYYYMMDD_HHMM>_<method>_bs<batch>_lr<lr>"
                 f"_dose{f:g}/ 生成（P0-1 泊松降采样，f={f:g}）")
cfg["norm"] = {
    "norm_mean": mean,
    "norm_std": std,
    "norm_scope": "global",
    "zero_log_fill": fill,
    "zero_fill_normalized": round((fill - mean) / std, 4),
    "zero_fill_normalized_derivation": f"({fill} - {mean:.6f}) / {std:.6f}",
    "zero_fill_normalized_usage": "仅供人工核对；运行时由 norm_mean/norm_std/zero_log_fill 现算",
    "domain": "-log((I + 1e-10) / (E + 1e-10))",
    "fill_order": "先在 -log 域填 zero_log_fill，再归一化",
    "invalid_rule": "(I <= 0) | (empty_proj <= 0)",
    "scope_note": (f"P0-1 剂量实验专用：数据为 3_3/ 的泊松抽稀镜像（f={f:g}），"
                   "四档能量全局归一化；与主 config.json 的 norm 互不影响。"
                   f"来源 {st['meta']['source']}，coverage {g['coverage']:.4f} / "
                   f"valid_coverage {g['valid_coverage']:.4f}"),
    "source": st["meta"]["source"],
    "recomputed": True,
    "recompute_proposal": str(norm_p.relative_to(ROOT)).replace("\\", "/"),
    "coverage": round(g["coverage"], 4),
    "valid_coverage": round(g["valid_coverage"], 4),
}
cfg["data"]["targets"] = csv_name
cfg["dose"] = {
    "factor": f,
    "thinning": "I_f = Poisson(f * I) / f（无偏，Var = I / f）",
    "flat_field": "empty_proj 不抽稀（视为高剂量校准的 air scan）；理由见 build_dose_variants.py",
    "replication": "逐文件确定性 RNG：sha256(f|relpath)，同一文件在 train/val/test 与各通道里一致",
    "mirror_root": f"3_3_dose_f{f:g}/",
    "builder": "scripts/build_dose_variants.py",
    "note": "与 f=1.0 的 canonical run（sparse_targets.csv）共用同一 split 与同一 seed 策略",
}
out_p.write_text(json.dumps(cfg, indent=2, ensure_ascii=False), encoding="utf-8")
print(f"已写出 {out_p.name}")
print(f"  norm_mean={mean:.6f}  norm_std={std:.6f}  zero_log_fill={fill}")
print(f"  csv={csv_name}  epochs={args.epochs}")
print(f"  coverage={g['coverage']:.4f}  valid_coverage={g['valid_coverage']:.4f}")
