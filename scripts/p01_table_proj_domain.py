#!/usr/bin/env python
# -*- coding: utf-8 -*-
r"""
scripts/p01_table_proj_domain.py —— 投影域基线表（审稿意见 P0-2）

审稿人指出：§5.6 正文写了投影域 CNN 0.5570 vs 线性插值 0.8019，但 Table 4 只有图像域，
读者无法核对。本脚本生成一张**投影域**表，4 档能量 + Overall，两个口径并列。

口径（两列都必须是同一套，否则不可比）
--------------------------------------
  all32  —— 全 32 行、per-pixel mask 加权，与 `train.py` 的 evaluate 一致。
            这是 §5.6 与旧表用的口径，`linear_interp/test_metrics.json` 只有这一套。
  row16  —— 只算 `RECON_ROW=16`（真正进入正弦图、决定重建的那一行）。
            CNN 侧直接取 `cnn_recon/test_metrics.json` 的 `projection_domain_check`；
            线性插值侧本脚本现算（`0.5·(L[θ−1] + L[θ+1])` vs `L[θ]`，纯算术，无模型）。

数据来源（全部只读）
--------------------
  runs/<cnn_run_id>_cnn_recon/test_metrics.json → projection_domain_check
  runs/<cnn_run_id>_linear_interp/test_metrics.json → per_energy / overall
  正弦图（仅为算线性插值的 row16）：sparse_targets.csv + 3_3/

输出
----
  runs/_reviewer_p0/table_proj_domain.{json,md}

用法
----
  conda run -n ct_full --no-capture-output python scripts/p01_table_proj_domain.py \
      --cnn-run-id 20261022_2352_cnn_bs8_lr1e-4
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
for p in (ROOT, ROOT / "scripts"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import baseline_recon as br          # noqa: E402

OUT = ROOT / "runs" / "_reviewer_p0"


def load_json(p: Path) -> dict:
    if not p.is_file():
        raise SystemExit(f"[FAIL] 缺少 {p}")
    return json.loads(p.read_text(encoding="utf-8"))


def rows_of(energy: str) -> list[dict]:
    with open(br.SPARSE_TARGETS, newline="", encoding="utf-8") as fh:
        rows = [r for r in csv.DictReader(fh) if r["energy_MeV"] == energy]
    rows.sort(key=lambda r: float(r["target_angle_deg"]))
    if len(rows) != 360:
        raise SystemExit(f"[{energy}] 期望 360 行，实得 {len(rows)}")
    return rows


def sino_stack(rows: list[dict], fill: float) -> np.ndarray:
    """(360, 32, 1200) 的 -log 栈，零/非有限值统一填 fill（与 dataset 一致）。"""
    scan_dirs = {r["scan_dir"] for r in rows}
    empty = br.load_empty(ROOT / "3_3", scan_dirs.pop())
    out = np.empty((len(rows), 32, br.OUTPUT_SIZE), dtype=np.float64)
    for i, r in enumerate(rows):
        proj = np.load(ROOT / r["target_file"]).astype(np.float32)
        L = br.log_ratio(proj, empty).astype(np.float64)
        out[i] = np.where(np.isfinite(L), L, fill)
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cnn-run-id", default="20261022_2352_cnn_bs8_lr1e-4")
    ap.add_argument("--label", default="5-epoch pilot")
    ap.add_argument("--from-p0-json", default=None,
                    help="改用 runs/_reviewer_p0/reviewer_p0*.json 取数（CNN 与线性插值"
                         "在同一张 json 里、同为 random_test 45 视角，适合没有 "
                         "`_cnn_recon/` 目录的新 run）")
    ap.add_argument("--out-suffix", default="", help="输出文件名后缀，如 _30ep")
    args = ap.parse_args(argv)
    OUT.mkdir(parents=True, exist_ok=True)

    if args.from_p0_json:
        res = _res_from_p0(Path(args.from_p0_json), args.label)
        _write(res, args.out_suffix, args.label,
               source=f"reviewer_p0.json ← {args.from_p0_json}")
        return 0

    cfg = load_json(br.RUNS_DIR / args.cnn_run_id / "config.json")
    fill = float(cfg["norm"]["zero_log_fill"])

    tm_cnn = load_json(br.RUNS_DIR / f"{args.cnn_run_id}_cnn_recon" / "test_metrics.json")
    tm_lin = load_json(br.RUNS_DIR / f"{args.cnn_run_id}_linear_interp" / "test_metrics.json")
    pdc = tm_cnn["projection_domain_check"]

    res: dict = {"cnn_run_id": args.cnn_run_id, "label": args.label,
                 "per_energy": {}, "overall": {}}
    for e in br.ENERGIES:
        # ---- CNN：两个输入相位的 test split 按 n_valid_pixels 池化
        entry: dict = {}
        for roi in ("row16", "all32"):
            nv = se = 0
            n_views = 0
            for ph, v in pdc[e]["by_input_phase"].items():
                t = v[roi]["by_split"]["test"]
                nv += t["n_valid_pixels"]
                se += t["rmse_valid"] ** 2 * t["n_valid_pixels"]
                n_views += t["n_views"]
            entry[f"cnn_{roi}"] = {"rmse_valid": math.sqrt(se / nv),
                                   "n_valid_pixels": nv, "n_views": n_views}
        # ---- 线性插值：all32 的 rmse 直接取 JSON（口径即 train.py evaluate），
        #      但 n_valid_pixels 那份 JSON 没存 → 下面在同一个循环里现数一遍，
        #      并顺手把 all32 的 rmse 也重算一遍做交叉校验（tol 1e-9）。
        lin = tm_lin["per_energy"][e]

        # ---- 线性插值 row16 / all32：现算
        rows = rows_of(e)
        sino_all32 = sino_stack(rows, fill)
        sino = sino_all32[:, br.RECON_ROW, :]
        idx = {int(float(r["target_angle_deg"])): i for i, r in enumerate(rows)}
        test_angles = [int(float(r["target_angle_deg"])) for r in rows if r["split"] == "test"]
        sq, n = 0.0, 0
        sq32, n32 = 0.0, 0
        for a in test_angles:
            gt = sino[idx[a]]
            pr = 0.5 * (sino[idx[(a - 1) % 360]] + sino[idx[(a + 1) % 360]])
            # mask 与 train.py 一致：目标投影 > 0 且 empty > 0（行内逐像素）
            r0 = rows[idx[a]]
            proj = np.load(ROOT / r0["target_file"]).astype(np.float32)
            empty = br.load_empty(ROOT / "3_3", r0["scan_dir"])
            m = (proj[br.RECON_ROW] > 0) & (empty[br.RECON_ROW] > 0)
            if m.any():
                sq += float(((pr - gt) ** 2)[m].sum())
                n += int(m.sum())
            # all32：全 32 行，同一张 mask
            L_all = br.log_ratio(proj, empty).astype(np.float64)
            gt32 = np.where(np.isfinite(L_all), L_all, fill)
            pr32 = 0.5 * (sino_all32[idx[(a - 1) % 360]] + sino_all32[idx[(a + 1) % 360]])
            m32 = (proj > 0) & (empty > 0)
            if m32.any():
                sq32 += float(((pr32 - gt32) ** 2)[m32].sum())
                n32 += int(m32.sum())
        entry["linear_row16"] = {"rmse_valid": math.sqrt(sq / n), "n_valid_pixels": n,
                                 "n_views": len(test_angles)}
        rmse32 = math.sqrt(sq32 / n32)
        entry["linear_all32"] = {"rmse_valid": lin["rmse_valid"],
                                 "n_valid_pixels": n32,
                                 "n_views": len(test_angles),
                                 "recomputed_rmse": rmse32,
                                 "abs_diff_vs_json": abs(rmse32 - lin["rmse_valid"])}
        if abs(rmse32 - lin["rmse_valid"]) > 1e-9:
            raise SystemExit(
                f"[{e}] 线性插值 all32 现算 {rmse32:.9f} != JSON {lin['rmse_valid']:.9f}"
                f"（mask/视角集合不一致，表不可用）")
        res["per_energy"][e] = entry

    # ---- Overall：按 n_valid_pixels 池化
    for key in ("cnn_row16", "cnn_all32", "linear_row16", "linear_all32"):
        nv = 0
        se = 0.0
        cnt = 0
        for e in br.ENERGIES:
            c = res["per_energy"][e][key]
            nv += c["n_valid_pixels"]
            se += c["rmse_valid"] ** 2 * c["n_valid_pixels"]
            cnt += 1
        res["overall"][key] = {"rmse_valid": math.sqrt(se / nv),
                               "n_valid_pixels": nv, "n_energies": cnt}

    # 仅在未指定 --out-suffix 时才写无后缀的 table_proj_domain.json。
    # 该文件历史上固定存放 5-epoch pilot 表；带后缀运行时若仍覆写它，
    # 会把 5ep 基准表静默替换成别的代次（跨代混引的一种形式，2026-10-24 实际发生过）。
    if not args.out_suffix:
        (OUT / "table_proj_domain.json").write_text(
            json.dumps(res, ensure_ascii=False, indent=2, default=float) + "\n",
            encoding="utf-8")
    _write(res, args.out_suffix, args.label,
           source=f"{args.cnn_run_id}（cnn_recon / linear_interp 两份 test_metrics.json）")
    return 0


def _res_from_p0(path: Path, label: str) -> dict:
    """从 reviewer_p0 的输出取数：两边都是同一批 45 个 random_test 视角、同一张 mask。"""
    d = load_json(path)["uniform"]
    res: dict = {"cnn_run_id": d["run_id"], "label": label,
                 "source_json": str(path), "per_energy": {}, "overall": {}}
    for e in br.ENERGIES:
        m = d["per_energy"][e]["methods"]
        entry = {}
        for meth, key in (("cnn", "cnn"), ("linear", "linear_interp")):
            for roi in ("row16", "all32"):
                t = m[key][f"{roi}|random_test"]
                entry[f"{meth}_{roi}"] = {
                    "rmse_valid": t["rmse_valid"],
                    "n_valid_pixels": t["n_valid_pixels"],
                    "n_views": t["n_views"]}
        res["per_energy"][e] = entry
    for key in ("cnn_row16", "cnn_all32", "linear_row16", "linear_all32"):
        nv = sum(res["per_energy"][e][key]["n_valid_pixels"] for e in br.ENERGIES)
        se = sum(res["per_energy"][e][key]["rmse_valid"] ** 2 *
                 res["per_energy"][e][key]["n_valid_pixels"] for e in br.ENERGIES)
        res["overall"][key] = {"rmse_valid": math.sqrt(se / nv),
                               "n_valid_pixels": nv, "n_energies": len(br.ENERGIES)}
    return res


def _write(res: dict, suffix: str, label: str, source: str) -> None:
    def f(x: float) -> str:
        return f"{x:.4f}"

    md = [f"**Table X.** Projection-domain completion error ({label}). "
          "Per-pixel masked RMSE in the -log domain, 45 held-out test views per energy.",
          "",
          "| Energy (MeV) | Linear interp. (all 32 rows) | CNN (all 32 rows) | "
          "Linear interp. (row 16) | CNN (row 16) |",
          "|---|---|---|---|---|"]
    for e in br.ENERGIES:
        x = res["per_energy"][e]
        md.append(f"| {e} | {f(x['linear_all32']['rmse_valid'])} | "
                  f"{f(x['cnn_all32']['rmse_valid'])} | "
                  f"{f(x['linear_row16']['rmse_valid'])} | "
                  f"{f(x['cnn_row16']['rmse_valid'])} |")
    o = res["overall"]
    md.append(f"| **Overall** | **{f(o['linear_all32']['rmse_valid'])}** | "
              f"**{f(o['cnn_all32']['rmse_valid'])}** | "
              f"**{f(o['linear_row16']['rmse_valid'])}** | "
              f"**{f(o['cnn_row16']['rmse_valid'])}** |")
    md += ["",
           "> all 32 rows = mask-weighted over the full 1200 x 32 detector, identical to the "
           "training-time validation metric. Row 16 = the single detector row that enters the "
           "sinogram and therefore decides the reconstruction; the other 31 rows are recorded "
           "but never used for imaging. The two blocks differ by ~3.7x because the source "
           "profile peaks at the detector centre and the edge rows are near-zero-count.",
           ""
           f"> Source: {source}",
           ""]
    (OUT / f"table_proj_domain{suffix}.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    (OUT / f"table_proj_domain{suffix}.json").write_text(
        json.dumps(res, ensure_ascii=False, indent=2, default=float) + "\n", encoding="utf-8")
    print("\n".join(md))
    print(f"\n输出: {OUT / f'table_proj_domain{suffix}.md'}")


if __name__ == "__main__":
    raise SystemExit(main())
