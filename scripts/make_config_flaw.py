"""生成 config_flaw.json：主 config 的副本，仅替换 norm 段为 flaw 数据重算值。

硬约束（记忆规约）：norm 未确认/未重算时必须报错，不得填默认值。
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CFG = ROOT / "config.json"
NORM = ROOT / "docs" / "norm_stats_flaw.json"
OUT = ROOT / "config_flaw.json"

# 多能量扩展（2026-10-26）：--norm-json / --out 指定逐能量产物，默认行为不变
_argv = sys.argv[1:]
def _opt(name: str, default: Path) -> Path:
    return Path(_argv[_argv.index(name) + 1]) if name in _argv else default
NORM = _opt("--norm-json", NORM)
OUT = _opt("--out", OUT)

if not NORM.is_file():
    raise SystemExit(f"[FATAL] 缺 {NORM}：先跑 compute_norm_stats_flaw.py，禁止填默认值")

cfg = json.loads(CFG.read_text(encoding="utf-8"))
st = json.loads(NORM.read_text(encoding="utf-8"))
ENERGY = float(st.get("energy_MeV", 3.0))

if st.get("representativeness", {}).get("blocked"):
    raise SystemExit("[FATAL] norm 代表性检查 BLOCK，等裁决")

fill = float(st["zero_log_fill"])
mean = float(st["norm_mean"])
std = float(st["norm_std"])
# 数据根目录名从 norm 的 source 首段取（3_3_flaw / 3_3_flaw_1_1），避免硬编码
DATA_ROOT = st.get("source", "3_3_flaw/").split("/")[0]
cfg["run_id"] = ("PENDING: train.py 启动时按 runs/<YYYYMMDD_HHMM>_<method>_bs<batch>_lr<lr>/ 生成"
                 f"（本次为 {DATA_ROOT} 缺陷数据域内训练，{ENERGY:g} MeV 单档）")
cfg["norm"] = {
    "norm_mean": mean,
    "norm_std": std,
    "norm_scope": "global",
    "zero_log_fill": fill,
    "zero_fill_normalized": round((fill - mean) / std, 4),
    "zero_fill_normalized_usage": "仅供人工核对；运行时代码由 norm_mean/norm_std/zero_log_fill 现算",
    "domain": "-log((I + 1e-10) / (E + 1e-10))",
    "fill_order": "先在 -log 域填 zero_log_fill，再归一化",
    "invalid_rule": "(I <= 0) | (empty_proj <= 0)",
    "scope_note": (f"norm_scope=global，本 run 只训练 {ENERGY:g} MeV 单档（故 global 即该档统计量）；"
                   f"缺陷数据（{DATA_ROOT}/）专用，与主 config.json 的全局 norm "
                   f"互不影响；来自 {st.get('sparse_table', 'sparse_targets_flaw.csv')} split=train 的 270 个 target，"
                   "input_union 代表性检查 |Δmean|=%.4f / |Δstd|/std=%.2f%%"
                   % (st["representativeness"]["delta_mean"],
                      st["representativeness"]["delta_std_rel"] * 100)),
    "source": st["source"],
    "recomputed": True,
    "compare_full_config": st["compare_full_config"],
    "valid_coverage": round(st["valid"]["valid_coverage"], 4),
    "coverage": round(st["valid"]["coverage"], 4),
}
OUT.write_text(json.dumps(cfg, indent=2, ensure_ascii=False), encoding="utf-8")
print(f"已写出 {OUT}")
print(f"  norm_mean={mean} norm_std={std} zero_log_fill={fill} "
      f"zero_fill_norm={cfg['norm']['zero_fill_normalized']}")
print(f"  其他配置不变: batch={cfg['batch']} lr={cfg['lr']} epochs={cfg['epochs']} "
      f"warmup={cfg['warmup']} model={cfg['model']['name']}")
