"""CSV 数据表的统一落点：`csv_data/`。

2026-10-27 起，仓库根不再散落 `sparse_targets*.csv`。所有数据表（训练三元组表、
剂量变体、角度跨度扫描、缺陷域各能量表）统一放在 `csv_data/` 下。

两个入口
--------
`resolve_csv(name)`          —— **读**。按下面的顺序找，找到哪个用哪个：
    1. 绝对路径 → 原样返回；
    2. `csv_data/<basename>` 存在 → 用它（新位置，优先）；
    3. `<basename>` 存在（仓库根，历史位置）→ 用它（向后兼容）；
    4. 都不存在 → 返回 `csv_data/<basename>`，让报错信息指向新位置。

`resolve_csv(name, for_write=True)` —— **写**。恒返回 `csv_data/<basename>` 并确保目录存在。

为什么读要留仓库根的回退
------------------------
`runs/*/config.json` 的 `data.targets` 记的是当时的**相对文件名**
（如 `sparse_targets_flaw_1.5.csv`）。这些 config 是论文数字的可追溯凭据，
不能因为搬家就失去复现能力。回退逻辑让历史 run 的 config 仍然可用，
而新代码默认写新目录 —— 两边都不妥协。

被 `data_manifest.csv` 那条禁读规则拦住的表（全量资产清单）不在本模块管辖范围内。
"""
from __future__ import annotations

from pathlib import Path

__all__ = ["CSV_DIR", "PROJECT_ROOT", "resolve_csv"]

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CSV_DIR = PROJECT_ROOT / "csv_data"

#: 历史位置（仓库根）——只用于读回退，不再写入
LEGACY_ROOT = PROJECT_ROOT


def resolve_csv(name: str | Path, *, for_write: bool = False) -> Path:
    """把数据表名解析成实际路径。语义见模块 docstring。"""
    p = Path(name)

    if p.is_absolute():
        if for_write:
            p.parent.mkdir(parents=True, exist_ok=True)
        return p

    # 只取 basename：传入 "csv_data/x.csv" 与 "x.csv" 等价，
    # 也避免拼成 csv_data/csv_data/x.csv
    base = p.name

    target = CSV_DIR / base
    if for_write:
        target.parent.mkdir(parents=True, exist_ok=True)
        return target

    if target.is_file():
        return target
    legacy = LEGACY_ROOT / base
    if legacy.is_file():
        return legacy
    return target
