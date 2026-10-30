#!/usr/bin/env python
# -*- coding: utf-8 -*-
r"""
scripts/p01_reviewer_p0.py —— 审稿意见 P0 两项 + 域外泛化

回答三件事
----------
  Q1（P0）**按角度扇区的独立划分**下 test_rmse_valid 是多少？
      扇区：0–89° 训练 / 90–179° 验证 / 180–359° 测试（180 个视角）。
      不重训，直接用现有 best.pt 在这 180 个视角上算。
      同时给出两个**只用实测数据**的基线，用来量化"90° 对称泄漏"到底有多大：
        shift90：pred(θ) = 实测(θ+90)   —— 如果物模严格 4 重对称，这个基线就是最优解
        linear ：pred(θ) = ½[实测(θ−1) + 实测(θ+1)]（部署合法，θ±1 属同一实测相位）
      若 CNN 打不过 shift90，说明模型在吃对称性红利；若打得过，说明它学的是插值。

  Q2（P0）**投影域基线表**：4 档能量 + Overall，线性插值 vs CNN 补全，
      两套口径（row16 / all32）并列，全部只取 test 视角、同 mask、同 gt。

  Q3 **域外泛化**：均匀数据训练的权重直接跑缺陷数据（norm 用均匀那套，冻结不改），
      报 rmse 与沿缺陷轨迹的匹配滤波 t 值（缺陷是否被当成分布外异常）。

只读、不改任何既有产物；结果写 runs/_reviewer_p0/。

用法
----
  conda run -n ct_full --no-capture-output python scripts/p01_reviewer_p0.py
  conda run -n ct_full --no-capture-output python scripts/p01_reviewer_p0.py --flaw-run-id <30ep>
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from data.csv_paths import resolve_csv  # noqa: E402  CSV 统一落点 csv_data/
for p in (ROOT, ROOT / "scripts"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import baseline_recon as br                            # noqa: E402
from cnn_recon import predict_views, write_subset_csv   # noqa: E402

UNIF_RUN_ID = "20261022_2352_cnn_bs8_lr1e-4"
FLAW_RUN_ID = "20261023_2239_cnn_bs8_lr1e-4"      # 可被 --flaw-run-id 覆盖
OUT = ROOT / "runs" / "_reviewer_p0"
TMP = ROOT / ".workbuddy" / "tmp" / "reviewer_p0"

#: 角度扇区划分（左闭右开）
SECTORS = {"train": (0, 90), "val": (90, 180), "test": (180, 360)}

TRAJ = {"amp": 367.1, "offset": 601.9, "phase_deg": 224.7}
NAN = float("nan")


# ------------------------------------------------------------------ 数据
def read_csv_rows(path: Path) -> list[dict]:
    with open(path, newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    rows.sort(key=lambda r: float(r["target_angle_deg"]))
    return rows


def build_stack(rows: list[dict], data_root: Path, fill: float
                ) -> tuple[np.ndarray, np.ndarray]:
    """返回 (L(360,32,1200) float32 的 -log 栈, mask(360,32,1200) bool)。"""
    scan_dirs = {r["scan_dir"] for r in rows}
    empty = br.load_empty(data_root, scan_dirs.pop())
    L = np.empty((len(rows), 32, br.OUTPUT_SIZE), dtype=np.float32)
    mk = np.empty((len(rows), 32, br.OUTPUT_SIZE), dtype=bool)
    for i, r in enumerate(rows):
        proj = np.load(br.ROOT / r["target_file"]).astype(np.float32)
        if proj.shape != (32, br.OUTPUT_SIZE):
            raise br.ReconSpecError(f"投影形状 {proj.shape}: {r['target_file']}")
        Lr = br.log_ratio(proj, empty).astype(np.float64)
        mk[i] = (proj > 0) & (empty > 0)
        L[i] = np.where(np.isfinite(Lr), Lr, fill).astype(np.float32)
    return L, mk


def predict_all(rows: list[dict], run_dir: Path, mean: float, std: float,
                fill: float, batch: int = 8) -> tuple[np.ndarray, np.ndarray]:
    """对全部 360 个角度各预测一次（不挑 split），返回 (pred(360,1,32,1200), mask)。"""
    import torch

    from models.unet import UNet

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = False
    model = UNet().to(device)
    ckpt = torch.load(run_dir / "best.pt", map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"])
    model.eval()

    TMP.mkdir(parents=True, exist_ok=True)
    header = list(rows[0].keys())
    # 全部标成 test：SparseCTDataset 仅按 split 过滤，不按 split 改输入构造
    p = TMP / "all360.csv"
    with open(p, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=header)
        w.writeheader()
        for r in rows:
            out = dict(r)
            out["split"] = "test"
            w.writerow(out)
    pred, mk = predict_views(model, p, mean, std, fill, batch, device,
                             expected_files=[r["target_file"] for r in rows])
    return pred, mk


# ------------------------------------------------------------------ 指标
def rmse_of(pred: np.ndarray, gt: np.ndarray, mk: np.ndarray,
            sel: np.ndarray, roi: str) -> dict:
    """roi="row16" 只算第 16 行；roi="all32" 全 32 行 mask 加权。"""
    if roi == "row16":
        pr, gt_, m_ = pred[:, 0, br.RECON_ROW, :], gt[:, br.RECON_ROW, :], \
            mk[:, br.RECON_ROW, :]
    elif roi == "all32":
        pr, gt_, m_ = pred[:, 0, :, :], gt, mk
    else:
        raise ValueError(roi)
    pr, gt_, m_ = pr[sel], gt_[sel], m_[sel]
    n = int(m_.sum())
    if n == 0:
        return {"n_views": int(sel.sum()), "n_valid_pixels": 0, "rmse_valid": NAN}
    se = float(((pr - gt_) ** 2)[m_].sum())
    return {"n_views": int(sel.sum()), "n_valid_pixels": n,
            "rmse_valid": math.sqrt(se / n)}


def angle_mask(rows: list[dict], lo: int, hi: int) -> np.ndarray:
    a = np.array([int(float(r["target_angle_deg"])) for r in rows])
    return (a >= lo) & (a < hi)


# ------------------------------------------------------------------ 匹配滤波
def matched_filter_err(pred_row16: np.ndarray, gt_row16: np.ndarray) -> dict:
    """在**预测误差**上做轨迹匹配滤波（域外泛化用）。"""
    cols = np.arange(br.OUTPUT_SIZE)
    angles = np.arange(360.0)
    e = pred_row16 - gt_row16
    out: dict = {}
    curves = {}
    for off in (0, 90, 180, 270):
        traj = TRAJ["offset"] + TRAJ["amp"] * np.cos(
            np.radians(angles + TRAJ["phase_deg"] + off))
        dl = np.empty(360)
        for a in range(360):
            m = np.abs(cols - traj[a]) <= 60.0
            dl[a] = e[a][m].mean() - e[a][~m].mean()
        curves[off] = dl
        mu, sd = float(dl.mean()), float(dl.std(ddof=1))
        sem = sd / math.sqrt(360)
        out[f"offset_{off}deg"] = {"mean": mu, "sem": sem,
                                   "t_stat": mu / sem if sem > 0 else NAN}
    base = [out[f"offset_{o}deg"]["mean"] for o in (90, 180, 270)]
    b0 = float(np.mean(base))
    b0_sem = float(np.sqrt(sum(out[f"offset_{o}deg"]["sem"] ** 2
                               for o in (90, 180, 270))) / 3)
    s0 = out["offset_0deg"]["sem"]
    c = out["offset_0deg"]["mean"] - b0
    sem = math.sqrt(s0 * s0 + b0_sem * b0_sem)
    out["d4_contrast"] = {"contrast": c, "sem": sem,
                          "t_stat": c / sem if sem > 0 else NAN}
    return out


# ------------------------------------------------------------------ main
def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--unif-run-id", default=UNIF_RUN_ID)
    ap.add_argument("--flaw-run-id", default=FLAW_RUN_ID)
    ap.add_argument("--skip-ood", action="store_true")
    args = ap.parse_args(argv)
    unif_run = args.unif_run_id
    flaw_run = args.flaw_run_id
    OUT.mkdir(parents=True, exist_ok=True)

    res: dict = {"sector_definition": {k: list(v) for k, v in SECTORS.items()},
                 "note": ("sector: [0,90) train / [90,180) val / [180,360) test；"
                          "不重训，直接用各数据集现有 best.pt 在 180 个 test 视角上算")}

    # ============================================================ 均匀数据
    print("=" * 60, flush=True)
    print("均匀数据（4 档）", flush=True)
    ucfg = json.loads((br.RUNS_DIR / unif_run / "config.json").read_text(encoding="utf-8"))
    unorm = ucfg["norm"]
    u_mean, u_std, u_fill = float(unorm["norm_mean"]), float(unorm["norm_std"]), \
        float(unorm["zero_log_fill"])
    urows = read_csv_rows(br.SPARSE_TARGETS)

    uni: dict = {"run_id": unif_run, "norm": [u_mean, u_std, u_fill],
                 "per_energy": {}, "overall": {}}
    for e in br.ENERGIES:
        t0 = time.time()
        rows = [r for r in urows if r["energy_MeV"] == e]
        rows.sort(key=lambda r: float(r["target_angle_deg"]))
        if len(rows) != 360:
            raise br.ReconSpecError(f"[{e}] 期望 360 行，实得 {len(rows)}")
        gt, gmk = build_stack(rows, ROOT / "3_3", u_fill)
        pred, pmk = predict_all(rows, br.RUNS_DIR / unif_run, u_mean, u_std, u_fill)
        if pred.shape[0] != 360:
            raise br.ReconSpecError(f"[{e}] 预测 {pred.shape[0]} 个视角")
        # 预测行序 = CSV 行序 = 角度 0..359，与 gt 对齐（已由 expected_files 断言）

        # 基线：90° 平移复制、相邻两点均值（都只用实测）
        n = 360
        idx = np.arange(n)
        gt_shift = gt[(idx + 90) % n]
        gt_lin = 0.5 * (gt[(idx - 1) % n] + gt[(idx + 1) % n])

        rnd = np.array([r["split"] == "test" for r in rows])
        sec = angle_mask(rows, *SECTORS["test"])
        trn = angle_mask(rows, *SECTORS["train"])

        entry = {"n_random_test": int(rnd.sum()), "n_sector_test": int(sec.sum()),
                 "n_sector_train": int(trn.sum()), "methods": {}}
        for name, pr in (("cnn", pred), ("shift90", gt_shift[:, None]),
                         ("linear_interp", gt_lin[:, None])):
            m_entry = {}
            for roi in ("row16", "all32"):
                for sub_name, sel in (("random_test", rnd), ("sector_test", sec),
                                      ("sector_train", trn)):
                    m_entry[f"{roi}|{sub_name}"] = rmse_of(pr, gt, gmk, sel, roi)
            entry["methods"][name] = m_entry
        uni["per_energy"][e] = entry
        print(f"[{e} MeV] {time.time() - t0:.1f}s  "
              f"cnn row16 random={entry['methods']['cnn']['row16|random_test']['rmse_valid']:.4f} "
              f"sector={entry['methods']['cnn']['row16|sector_test']['rmse_valid']:.4f} | "
              f"shift90 sector={entry['methods']['shift90']['row16|sector_test']['rmse_valid']:.4f} | "
              f"linear sector={entry['methods']['linear_interp']['row16|sector_test']['rmse_valid']:.4f}",
              flush=True)

    # 池化（按 n_valid_pixels 加权 MSE）
    for name in ("cnn", "shift90", "linear_interp"):
        for key in uni["per_energy"][br.ENERGIES[0]]["methods"][name]:
            nv = sum(uni["per_energy"][e]["methods"][name][key]["n_valid_pixels"]
                     for e in br.ENERGIES)
            se = sum(uni["per_energy"][e]["methods"][name][key]["rmse_valid"] ** 2 *
                     uni["per_energy"][e]["methods"][name][key]["n_valid_pixels"]
                     for e in br.ENERGIES)
            uni["overall"].setdefault(name, {})[key] = math.sqrt(se / nv)
    res["uniform"] = uni

    # ============================================================ 缺陷数据
    print("=" * 60, flush=True)
    print(f"缺陷数据（3.0 MeV，run={flaw_run}）", flush=True)
    fcfg = json.loads((br.RUNS_DIR / flaw_run / "config.json").read_text(encoding="utf-8"))
    fnorm = fcfg["norm"]
    f_mean, f_std, f_fill = float(fnorm["norm_mean"]), float(fnorm["norm_std"]), \
        float(fnorm["zero_log_fill"])
    frows = read_csv_rows(resolve_csv("sparse_targets_flaw.csv"))
    fgt, fmk = build_stack(frows, ROOT / "3_3_flaw", f_fill)

    flaw: dict = {"run_id": flaw_run, "norm": [f_mean, f_std, f_fill],
                  "per_energy": {}, "overall": {}}
    e = "3"
    fidx = np.arange(360)
    f_shift = fgt[(fidx + 90) % 360]
    f_lin = 0.5 * (fgt[(fidx - 1) % 360] + fgt[(fidx + 1) % 360])
    frnd = np.array([r["split"] == "test" for r in frows])
    fsec = angle_mask(frows, *SECTORS["test"])
    ftrn = angle_mask(frows, *SECTORS["train"])

    fpred, _ = predict_all(frows, br.RUNS_DIR / flaw_run, f_mean, f_std, f_fill)
    fentry = {"n_random_test": int(frnd.sum()), "n_sector_test": int(fsec.sum()),
              "n_sector_train": int(ftrn.sum()), "methods": {}}
    for name, pr in (("cnn", fpred), ("shift90", f_shift[:, None]),
                     ("linear_interp", f_lin[:, None])):
        m_entry = {}
        for roi in ("row16", "all32"):
            for sub_name, sel in (("random_test", frnd), ("sector_test", fsec),
                                  ("sector_train", ftrn)):
                m_entry[f"{roi}|{sub_name}"] = rmse_of(pr, fgt, fmk, sel, roi)
        fentry["methods"][name] = m_entry
    flaw["per_energy"][e] = fentry
    for name in ("cnn", "shift90", "linear_interp"):
        flaw["overall"][name] = dict(fentry["methods"][name])
    print(f"    cnn row16 random={fentry['methods']['cnn']['row16|random_test']['rmse_valid']:.4f} "
          f"sector={fentry['methods']['cnn']['row16|sector_test']['rmse_valid']:.4f} | "
          f"shift90 sector={fentry['methods']['shift90']['row16|sector_test']['rmse_valid']:.4f} | "
          f"linear sector={fentry['methods']['linear_interp']['row16|sector_test']['rmse_valid']:.4f}",
          flush=True)
    res["flaw"] = flaw

    # ============================================================ 域外泛化
    if not args.skip_ood:
        print("=" * 60, flush=True)
        print("域外泛化：均匀权重跑缺陷数据（norm 用均匀那套，冻结）", flush=True)
        ood_pred, ood_mk = predict_all(frows, br.RUNS_DIR / unif_run,
                                       u_mean, u_std, u_fill)
        # gt 也要用同一 fill 重建一次（口径必须与输入管线一致）
        o_gt, o_mk = build_stack(frows, ROOT / "3_3_flaw", u_fill)
        ood: dict = {"source_run_id": unif_run, "target": "3_3_flaw 3.0 MeV",
                     "norm": [u_mean, u_std, u_fill],
                     "note": "权重与 norm 全部沿用均匀侧，不做任何适配"}
        for roi in ("row16", "all32"):
            ood[f"{roi}|random_test"] = rmse_of(ood_pred, o_gt, o_mk, frnd, roi)
            ood[f"{roi}|sector_test"] = rmse_of(ood_pred, o_gt, o_mk, fsec, roi)
            ood[f"{roi}|all360"] = rmse_of(ood_pred, o_gt, o_mk,
                                           np.ones(360, bool), roi)
        # 同一 fill 下，域内训练模型的对照
        in_pred, _ = predict_all(frows, br.RUNS_DIR / flaw_run, f_mean, f_std, f_fill)
        o_gt2, o_mk2 = build_stack(frows, ROOT / "3_3_flaw", f_fill)
        ood["in_domain_reference"] = {
            f"{roi}|random_test": rmse_of(in_pred, o_gt2, o_mk2, frnd, roi)
            for roi in ("row16", "all32")}
        # 误差的轨迹匹配滤波（域外模型是否把缺陷当异常）
        ood["matched_filter_on_error"] = matched_filter_err(
            ood_pred[:, 0, br.RECON_ROW, :], o_gt[:, br.RECON_ROW, :])
        ood["matched_filter_on_error_in_domain"] = matched_filter_err(
            in_pred[:, 0, br.RECON_ROW, :], o_gt2[:, br.RECON_ROW, :])
        res["out_of_domain"] = ood
        print(f"    ood row16 random={ood['row16|random_test']['rmse_valid']:.4f} "
              f"vs 域内 {ood['in_domain_reference']['row16|random_test']['rmse_valid']:.4f} | "
              f"轨迹 t={ood['matched_filter_on_error']['d4_contrast']['t_stat']:+.2f} "
              f"vs 域内 {ood['matched_filter_on_error_in_domain']['d4_contrast']['t_stat']:+.2f}",
              flush=True)

    (OUT / "reviewer_p0.json").write_text(
        json.dumps(res, ensure_ascii=False, indent=2, default=float) + "\n",
        encoding="utf-8")
    print(f"\n输出: {OUT / 'reviewer_p0.json'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
