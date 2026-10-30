#!/usr/bin/env python
# -*- coding: utf-8 -*-
r"""
data/dataset.py —— 稀疏角 CT 数据集（新电脑从零重建，不参考旧机器任何代码）

==================== 硬性约束（违反即视为 bug） ====================
1. 只读三份 CSV：sparse_targets.csv、train_manifest_first_run.csv、angles.csv。
   严禁读 data_manifest.csv（全量 16 档清单，会让 12 个非首轮能量静默渗入）。
2. 角度与文件路径**全部来自 CSV**，运行时**禁止**推导 prev/next、
   禁止出现 `% 360`、禁止 `- 1` / `+ 1` / `- 3` 这类角度运算。
3. 严禁读 scan_parameters.json 的 energy_MeV（16 档全部硬编码 3.0，废弃字段）
   或 angles（非均匀下不可靠）。
4. empty_proj 按**能量**匹配：用该行 scan_dir 下的
   `{scan_dir}/results/empty_proj.npy`，禁止用全局固定的一份。
5. 归一化统计量与零计数填充值**必须由调用方显式传入**；
   本模块不提供任何默认值，缺参直接抛异常（防止悄悄用错统计量）。
==================================================================

数据流（单个样本）
    raw  I  = 投影原始强度 (32, 1200) float32
    raw  E  = 同能量目录的 empty_proj (32, 1200) float32
    L       = -log((I + LOG_EPS) / (E + LOG_EPS))        # 只在 -log 域做比值
    invalid = (I <= 0) | (E <= 0)                        # 无效像素
    L[invalid] = zero_log_fill                           # 在 -log 域填充
    x_norm  = (L - norm_mean) / norm_std                 # 归一化在填充之后
    mask    = (I_target > 0) & (E > 0)                   # per-pixel，仅对 target

    输入通道（n_views=3，论文正文配置）：
        ch0 = ch0_file      -> θ_i     （三元组中心，实际为 θ−1°）
        ch1 = ch_prev_file  -> θ_{i-1}（实际为 θ−3°，左侧额外上下文）
        ch2 = ch_next_file  -> θ_{i+1}（实际为 θ+1°）
        ch3 = sin(θ_target)
        ch4 = cos(θ_target)
    消融配置（n_views=2）：去掉 ch1（θ−3°），输入 4 通道 = [θ−1°, θ+1°, sin, cos]。
    两个配置的监督 y 与 mask 完全一致，唯一差别是输入侧是否提供左侧第三帧。
    监督 y：
        target_file 的归一化 -log 值，shape (1, 32, 1200)

返回值用 numpy（float32 / bool），框架无关；训练侧自行 torch.from_numpy。
这样第一步的只读校验不需要安装 torch。

自检：
    python -m data.dataset --check --zero-fill 5.7 --norm-mean 0.0 --norm-std 1.0
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent

sys.path.insert(0, str(PROJECT_ROOT))
from data.csv_paths import resolve_csv  # noqa: E402  CSV 统一落点 csv_data/

LOG_EPS = 1e-10
IMAGE_SHAPE = (32, 1200)

#: 进入重建的那一行（探测器竖直剖面峰顶行）。
#: Row-16-only 消融专用，必须与 baseline_recon.RECON_ROW 保持一致。
RECON_ROW = 16

SPARSE_TARGETS = "sparse_targets.csv"
TRAIN_MANIFEST = "train_manifest_first_run.csv"
ANGLES_CSV = "angles.csv"
FORBIDDEN_CSV = "data_manifest.csv"

REQUIRED_COLUMNS = (
    "energy_MeV", "scan_dir",
    "target_angle_deg", "target_file", "phase",
    "ch0_angle_deg", "ch0_file",
    "ch_prev_angle_deg", "ch_prev_file",
    "ch_next_angle_deg", "ch_next_file",
    "split",
)
VALID_SPLITS = ("train", "val", "test")


class DatasetSpecError(RuntimeError):
    """数据集配置/数据不符合已裁决口径时抛出（fail fast，不做兜底）。"""


def _resolve(rel: str) -> Path:
    p = Path(rel)
    return p if p.is_absolute() else (PROJECT_ROOT / p)


def load_csv_rows(csv_path: str | Path, split: str) -> list[dict]:
    """读取 sparse_targets.csv 并筛选 split。禁止读 data_manifest.csv。"""
    # CSV 统一落点是 csv_data/（见 data/csv_paths.py）；传入的裸文件名也能解析，
    # 历史 run 的 config.json 里记的正是裸文件名，必须继续可复现。
    path = resolve_csv(csv_path)
    if path.name == FORBIDDEN_CSV:
        raise DatasetSpecError(
            f"禁止读取 {FORBIDDEN_CSV}：它是全量 16 档资产清单，"
            f"训练只允许读 {SPARSE_TARGETS} / {TRAIN_MANIFEST} / {ANGLES_CSV}"
        )
    if split not in VALID_SPLITS:
        raise DatasetSpecError(f"split 必须是 {VALID_SPLITS}，收到 {split!r}")
    if not path.is_file():
        raise DatasetSpecError(f"CSV 不存在: {path}")

    rows: list[dict] = []
    with open(path, newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        missing = [c for c in REQUIRED_COLUMNS if c not in (reader.fieldnames or [])]
        if missing:
            raise DatasetSpecError(f"{path.name} 缺列: {missing}")
        for r in reader:
            if r["split"] != split:
                continue
            rows.append(r)

    if not rows:
        raise DatasetSpecError(f"{path.name} 中 split={split} 没有样本")
    return rows


def log_ratio(img: np.ndarray, empty: np.ndarray) -> np.ndarray:
    """-log((I+eps)/(E+eps))，返回 float32；无效像素留 NaN 由调用方填充。"""
    a = img.astype(np.float64)
    b = empty.astype(np.float64)
    out = -np.log((a + LOG_EPS) / (b + LOG_EPS))
    invalid = (a <= 0.0) | (b <= 0.0)
    out[invalid] = np.nan
    return out.astype(np.float32)


def valid_mask(img: np.ndarray, empty: np.ndarray) -> np.ndarray:
    return (img > 0.0) & (empty > 0.0)


class SparseCTDataset:
    """按行从 CSV 取路径，绝不推导角度。

    参数
    ----
    split : "train" | "val" | "test"
    zero_log_fill : float
        -log 域的零计数填充值（由第二步从数据算出，此处必须显式传入）。
    norm_mean, norm_std : float
        归一化统计量（必须显式传入；本类不设默认值）。
    n_views : int
        输入实测视角数，仅允许 2 或 3（默认 3 = 论文正文配置）。
        3 = θ−3° / θ−1° / θ+1°（ch_prev / ch0 / ch_next，即"左侧多一帧上下文"的不对称三视角）；
        2 = θ−1° / θ+1°（ch0 / ch_next，即对称包围对，**消融实验专用**）。
        两个视角模式只丢弃 ch_prev 通道，sin/cos 角度编码通道保留，
        因此输入通道数 = n_views + 2（3→5，2→4）。
        csv_path : sparse_targets.csv 路径
        data_root : 项目根，用于拼接 CSV 中的相对路径
        single_row : bool
            Row-16-only 消融开关。默认 False = 论文正文口径（全 32 行训练，勿动默认）。
            True 时的两处改动：
              - 输入侧：前 n_views 个投影通道只保留第 RECON_ROW 行，
                其余 31 行填「归一化后的 zero_log_fill」（fill_norm），与真实零计数像素
                在数值上完全无法区分 —— 而不是简单置零，
                理由见 _collapse_projection_rows 的文档串（涉及 train.py 的既有不变量）。
                这样网络从输入拿不到任何跨行信息。
                sin/cos 是角度编码，对每一行取值相同，不构成跨行泄漏，
                故保留全图广播（否则网络连目标角度都不知道）。
              - 监督侧：mask 收缩到仅 RECON_ROW 行，loss 的梯度 100% 来自该行。
                y 本身仍是真实 32 行、不做置零：这样既不改变 (1,32,1200) 的形状，
                也保住了训练侧已裁决的「rmse_full >= rmse_valid」物理前提
                （该断言依赖填充像素误差更大，若把 y 也清零会被破坏）。
            附带好处：此模式下 rmse_valid 自动等于 Row 16 RMSE，可直接从 train_log 读取。
    """

    def __init__(
        self,
        split: str,
        zero_log_fill: float,
        norm_mean: float,
        norm_std: float,
        n_views: int = 3,
        csv_path: str | Path = SPARSE_TARGETS,
        data_root: str | Path = PROJECT_ROOT,
        single_row: bool = False,
    ) -> None:
        if zero_log_fill is None:
            raise DatasetSpecError("zero_log_fill 未提供：必须由第二步从数据算出后显式传入")
        if norm_mean is None or norm_std is None:
            raise DatasetSpecError("norm_mean / norm_std 未提供：禁止使用默认值")
        if not math.isfinite(norm_std) or norm_std <= 0:
            raise DatasetSpecError(f"norm_std 必须为正有限数，收到 {norm_std!r}")
        if n_views not in (2, 3):
            raise DatasetSpecError(
                f"n_views 仅允许 2 或 3，收到 {n_views!r}："
                f"3 = θ−3°/θ−1°/θ+1°（正文配置），2 = θ−1°/θ+1°（消融）"
            )

        self.split = split
        self.n_views = int(n_views)
        self.zero_log_fill = float(zero_log_fill)
        self.norm_mean = float(norm_mean)
        self.norm_std = float(norm_std)
        self.data_root = Path(data_root)
        self.single_row = bool(single_row)
        self.rows = load_csv_rows(csv_path, split)

        # empty_proj 按 scan_dir（即按能量）缓存；不同能量强度差异极大，禁止错配
        self._empty_cache: dict[str, np.ndarray] = {}
        # scan_dir 列只有目录名（不含 3_3/ 前缀），真实目录从 CSV 的文件路径反推：
        # <scan_path>/projections/<file>.npy  ->  <scan_path> = parent.parent
        self._scan_path: dict[str, Path] = {}
        # 只做「路径是否落在自己 scan_dir 下」的合法性校验，不推导任何角度
        for r in self.rows:
            for key in ("target_file", "ch0_file", "ch_prev_file", "ch_next_file"):
                if f"{r['scan_dir']}/projections/" not in r[key]:
                    raise DatasetSpecError(
                        f"target={r['target_angle_deg']} 的 {key} 不在 scan_dir 下: {r[key]}"
                    )
            self._scan_path.setdefault(
                r["scan_dir"], self._resolve_file(r["target_file"]).parent.parent
            )

    # ------------------------------------------------------------------ 基础
    def __len__(self) -> int:
        return len(self.rows)

    def _resolve_file(self, rel: str) -> Path:
        p = Path(rel)
        return p if p.is_absolute() else (self.data_root / p)

    def empty_proj(self, scan_dir: str) -> np.ndarray:
        if scan_dir not in self._empty_cache:
            p = self._scan_path[scan_dir] / "results" / "empty_proj.npy"
            if not p.is_file():
                raise DatasetSpecError(f"缺少 empty_proj: {p}")
            e = np.load(p).astype(np.float32)
            if e.shape != IMAGE_SHAPE:
                raise DatasetSpecError(f"empty_proj shape {e.shape} != {IMAGE_SHAPE}: {p}")
            self._empty_cache[scan_dir] = e
        return self._empty_cache[scan_dir]

    # ------------------------------------------------------------ coverage 两口径
    def _target_stats(self, idx: int) -> tuple[float, float]:
        """一次性读 target，返回 (coverage, valid_coverage)。

        coverage       = (I_target > 0) 像素数 / 总像素数            —— 资产口径
        valid_coverage = (I_target > 0) & (empty_proj > 0) / 总像素数 —— 评估口径
        恒有 coverage >= valid_coverage（后者条件更严），违反即抛错。
        """
        r = self.rows[idx]
        empty = self.empty_proj(r["scan_dir"])
        y_raw = self._load(r["target_file"])
        n = float(y_raw.size)
        cov = float(np.count_nonzero(y_raw > 0)) / n
        vcov = float(np.count_nonzero(valid_mask(y_raw, empty))) / n
        if cov < vcov:
            raise DatasetSpecError(
                f"coverage({cov}) < valid_coverage({vcov})，定义被破坏：{r['target_file']}"
            )
        return cov, vcov

    def coverage_and_valid_of(self, idx: int) -> tuple[float, float]:
        return self._target_stats(idx)

    def coverage_of(self, idx: int) -> float:
        """资产口径：I_target > 0 像素数 / 总像素数"""
        return self._target_stats(idx)[0]

    def valid_coverage_of(self, idx: int) -> float:
        """评估口径：(I_target > 0) & (empty_proj > 0) / 总像素数，与 mask 一致"""
        return self._target_stats(idx)[1]

    def _load(self, rel: str) -> np.ndarray:
        arr = np.load(self._resolve_file(rel), mmap_mode="r")
        if arr.shape != IMAGE_SHAPE:
            raise DatasetSpecError(f"shape {arr.shape} != {IMAGE_SHAPE}: {rel}")
        return np.asarray(arr, dtype=np.float32)

    def _channel(self, rel: str, empty: np.ndarray) -> np.ndarray:
        """单张投影 -> 归一化 -log 图（无效像素在 -log 域填充）"""
        img = self._load(rel)
        L = log_ratio(img, empty)
        invalid = ~np.isfinite(L)
        L[invalid] = self.zero_log_fill
        return ((L - self.norm_mean) / self.norm_std).astype(np.float32)

    # ------------------------------------------------- Row-16-only 消融（内部）
    def _collapse_projection_rows(self, x: np.ndarray) -> np.ndarray:
        """把输入里前 n_views 个真实投影通道的非 RECON_ROW 行**抹成零计数填充值**。

        x 通道数 = n_views + 2，最后两个是 sin/cos：保留全图广播，
        因为角度编码对每一行取值相同，不构成跨行泄漏，且网络需要它知道目标角度。

        为什么用 fill_norm 而不是 0 抹除
        -------------------------------
        train.py 有一条不变量检查：所有 -log 域的无效像素必须在归一化后恰好等于
        fill_norm = (zero_log_fill - norm_mean) / norm_std，反归一化后回到 zero_log_fill。
        若用 0 抹除，这些行会被判定为「填充值不一致」而 FAIL（smoke 实测偏差 12.87）。
        用 fill_norm 抹除则这些行在数值上与真实零计数像素完全无法区分 ——
        既是「无信息」，又不破坏既有不变量。
        """
        out = x.copy()
        keep = np.zeros(out.shape[-2:], dtype=bool)
        keep[RECON_ROW, :] = True
        fill_norm = (self.zero_log_fill - self.norm_mean) / self.norm_std
        for i in range(out.shape[0] - 2):          # 前 n_views 个投影通道
            out[i][~keep] = fill_norm
        return out

    @staticmethod
    def _row16_mask(mask: np.ndarray) -> np.ndarray:
        """per-pixel mask 收缩到仅 RECON_ROW 行 -> loss 梯度 100% 来自该行。"""
        out = np.zeros_like(mask)
        out[RECON_ROW, :] = mask[RECON_ROW, :]
        return out

    # ------------------------------------------------------------------ 取样本
    def __getitem__(self, idx: int) -> dict:
        r = self.rows[idx]
        scan_dir = r["scan_dir"]
        empty = self.empty_proj(scan_dir)

        ch0 = self._channel(r["ch0_file"], empty)          # θ_i  = θ−1°（三元组中心）
        ch1 = self._channel(r["ch_prev_file"], empty)      # θ_{i-1} = θ−3°（左侧额外上下文）
        ch2 = self._channel(r["ch_next_file"], empty)      # θ_{i+1} = θ+1°

        # 角度直接取 CSV 列，不做任何角度运算
        theta = math.radians(float(r["target_angle_deg"]))
        ch3 = np.full(IMAGE_SHAPE, math.sin(theta), dtype=np.float32)
        ch4 = np.full(IMAGE_SHAPE, math.cos(theta), dtype=np.float32)

        # n_views=3（正文）：[θ−3°, θ−1°, θ+1°] + sin/cos → 5 通道
        # n_views=2（消融）：丢掉 θ−3°（ch_prev），保留对称包围对 + sin/cos → 4 通道
        views = [ch0, ch1, ch2] if self.n_views == 3 else [ch0, ch2]
        x = np.stack(views + [ch3, ch4], axis=0)            # (n_views+2, 32, 1200)

        y_raw = self._load(r["target_file"])
        y_log = log_ratio(y_raw, empty)
        mask = valid_mask(y_raw, empty)                    # per-pixel
        y_log[~mask] = self.zero_log_fill
        y = ((y_log - self.norm_mean) / self.norm_std)[None, ...].astype(np.float32)

        if self.single_row:
            # Row-16-only 消融：输入侧抹掉其余 31 行，监督侧 mask 收缩到 row16。
            # y 保持真实 32 行不变（理由见 __init__ 文档串）。
            x = self._collapse_projection_rows(x)
            mask = self._row16_mask(mask)

        return {
            "x": x,
            "y": y,
            "mask": mask[None, ...].astype(bool),
            "meta": {
                "energy_MeV": float(r["energy_MeV"]),
                "scan_dir": scan_dir,
                "target_angle_deg": float(r["target_angle_deg"]),
                "phase": r["phase"],
                "split": r["split"],
                "target_file": r["target_file"],
                "n_input_views": self.n_views,
            },
        }

    # ------------------------------------------------------------------ 反归一化
    def denormalize(self, arr: np.ndarray) -> np.ndarray:
        """归一化域 -> 物理域（即未归一化的 -log 值），评估 RMSE 用。"""
        return arr * self.norm_std + self.norm_mean


# ====================================================================== 自检
def _check(args: argparse.Namespace) -> int:
    ds = SparseCTDataset(
        split=args.split,
        zero_log_fill=args.zero_fill,
        norm_mean=args.norm_mean,
        norm_std=args.norm_std,
        n_views=args.input_views,
    )
    print("=" * 92)
    print(f"split={ds.split}  样本数={len(ds)}  csv={SPARSE_TARGETS}")
    print(f"zero_log_fill={ds.zero_log_fill}  norm_mean={ds.norm_mean}  norm_std={ds.norm_std}")
    print(f"n_views={ds.n_views}（输入通道数应为 {ds.n_views + 2}）")
    if args.zero_fill == 5.7 or (args.norm_mean, args.norm_std) == (0.0, 1.0):
        print("[注意] 本次使用占位参数，仅用于 shape 校验，不代表任何已确认的统计量")
    print("=" * 92)

    items = [ds[i] for i in range(min(args.n, len(ds)))]
    x = np.stack([it["x"] for it in items])
    y = np.stack([it["y"] for it in items])
    m = np.stack([it["mask"] for it in items])

    if x.shape[1] != ds.n_views + 2:
        raise DatasetSpecError(
            f"输入通道数 {x.shape[1]} != n_views+2 = {ds.n_views + 2}，通道装配被破坏"
        )
    print(f"x    shape={x.shape} dtype={x.dtype}")
    print(f"y    shape={y.shape} dtype={y.dtype}")
    print(f"mask shape={m.shape} dtype={m.dtype}")
    print(f"x    min={np.nanmin(x):+.4f} max={np.nanmax(x):+.4f} mean={np.nanmean(x):+.4f} "
          f"NaN={int(np.isnan(x).sum())} Inf={int(np.isinf(x).sum())}")
    print(f"y    min={np.nanmin(y):+.4f} max={np.nanmax(y):+.4f} mean={np.nanmean(y):+.4f} "
          f"NaN={int(np.isnan(y).sum())} Inf={int(np.isinf(y).sum())}")
    print(f"mask 全 batch 覆盖率={m.mean():.4f}  (有效像素 {int(m.sum())} / {m.size})")
    print(f"y[~mask] 是否全为归一化填充值: ", end="")
    fill_norm = (ds.zero_log_fill - ds.norm_mean) / ds.norm_std
    print(bool(np.allclose(y[~m], fill_norm)) if (~m).any() else "该 batch 无无效像素")

    print("-" * 92)
    print(f"{'#':>2}{'E(MeV)':>8}{'target':>8}{'phase':>7}{'coverage':>11}"
          f"{'valid_cov':>11}{'x_min':>9}{'x_max':>9}{'y_min':>9}{'y_max':>9}")
    cov_list = []
    for i, it in enumerate(items):
        cov, vcov = ds.coverage_and_valid_of(i)
        cov_list.append((it["meta"]["energy_MeV"], cov, vcov))
        print(f"{i:>2}{it['meta']['energy_MeV']:>8g}{it['meta']['target_angle_deg']:>8g}"
              f"{it['meta']['phase']:>7}{cov:>11.4f}{vcov:>11.4f}"
              f"{x[i].min():>+9.3f}{x[i].max():>+9.3f}{y[i].min():>+9.3f}{y[i].max():>+9.3f}")

    print("-" * 92)
    print("coverage（资产口径 I>0）  vs  valid_coverage（评估口径 I>0 & E>0），按能量汇总本批：")
    by_e: dict[float, list[tuple[float, float]]] = {}
    for e, cov, vcov in cov_list:
        by_e.setdefault(e, []).append((cov, vcov))
    for e in sorted(by_e):
        v = by_e[e]
        c = sum(a for a, _ in v) / len(v)
        vc = sum(b for _, b in v) / len(v)
        print(f"  {e:>5g} MeV: coverage={c:.3f}, valid_coverage={vc:.3f}  (n={len(v)})")

    fails = []
    if x.shape != (len(items), ds.n_views + 2, 32, 1200):
        fails.append(f"x shape {x.shape}（期望 (n, {ds.n_views + 2}, 32, 1200)）")
    if y.shape != (len(items), 1, 32, 1200):
        fails.append(f"y shape {y.shape}")
    if m.shape != (len(items), 1, 32, 1200):
        fails.append(f"mask shape {m.shape}")
    if x.dtype != np.float32 or y.dtype != np.float32:
        fails.append(f"dtype x={x.dtype} y={y.dtype}")
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        fails.append("x/y 含 NaN 或 Inf")
    if not (0.0 < m.mean() < 1.0):
        fails.append(f"mask 覆盖率异常 {m.mean()}")
    for i, (_, cov, vcov) in enumerate(cov_list):
        if cov < vcov:
            fails.append(f"[{i}] coverage {cov:.4f} < valid_coverage {vcov:.4f}")
        if abs(vcov - float(m[i].mean())) > 1e-6:
            fails.append(f"[{i}] valid_coverage {vcov:.6f} != mask.mean() {float(m[i].mean()):.6f}")

    if fails:
        print("\n[FAIL] " + " | ".join(fails))
        return 1
    print("\n[OK] shape / dtype / 有限性 / mask 四项校验通过")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description="稀疏角 CT 数据集（只读校验）")
    ap.add_argument("--check", action="store_true", help="跑只读 shape 校验")
    ap.add_argument("--split", default="train", choices=list(VALID_SPLITS))
    ap.add_argument("--n", type=int, default=4, help="校验时取几个样本")
    ap.add_argument("--zero-fill", type=float, default=None,
                    help="-log 域零计数填充值（必填，占位可用 5.7）")
    ap.add_argument("--norm-mean", type=float, default=None)
    ap.add_argument("--norm-std", type=float, default=None)
    ap.add_argument("--input-views", type=int, default=3, choices=[2, 3],
                    help="输入实测视角数：3 = θ−3°/θ−1°/θ+1°（正文配置）；"
                         "2 = θ−1°/θ+1°（对称包围对，消融专用）")
    args = ap.parse_args()
    if not args.check:
        ap.print_help()
        return 2
    return _check(args)


if __name__ == "__main__":
    raise SystemExit(main())
