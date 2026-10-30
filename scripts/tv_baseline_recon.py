"""TV 正则化基线（180 视角）：SART-TV（POCS-TV 式交替）。

背景
----
审稿人要求补 TV 正则化基线。仓库自带的 TV 入口是
`3_3_geant4/quick_reconstruct.py --recon_algs tv`，它调用 **tomopy**
（`tomopy.recon(..., algorithm='tv')`）。本机 tomopy 的 Python 包存在（1.14.4），
但其编译共享库 `tomo-recon` 缺失（PyPI 包不含二进制），`import tomopy` 直接
ModuleNotFoundError；补装 conda-forge 二进制会改动 numpy/scipy，**训练正在进行，
不冒此险**。

因此改用环境内已有的 **scikit-image**：`denoise_tv_chambolle`（Chambolle 投影）
与 `iradon_sart`（支持 image= 热启动）交替，即 CT 文献里的 SART-TV / POCS-TV。
**λ 未调参**：只用下面 FIXED_* 一组固定设置；`--weight` 可用于做敏感性检查
（默认会额外跑两个邻近视重并一并报告，不做择优）。

    方法
----
  x0 = 0（零初值）
  重复 n_outer 轮：
      x ← SART(sino_sparse, theta_sparse, n_iter=n_iter_sart, relaxation, 热启动=x)
      x ← denoise_tv_chambolle(x, weight=w)          # TV 近端
  最后 restrict_to_fov（圆外置零，与 baseline_recon 同口径）

  **迭代预算与 SART 基线对齐**：n_outer=4 × n_iter_sart=5 = 20 次 SART，
  与 `baseline_recon.recon_sart` 的 20 次完全相同。否则"TV 更差"可能只是
  迭代更少，审稿人一眼就能反驳。

产出指标：
  - 图像域 RMSE / rmse_full / psnr_common / ssim（vs 同扫描的 dense 360 视角 hamming FBP）
  - 缺陷数据（--data flaw）额外给空腔 CNR（复用 p01_flaw_step3_eval 的主口径：
    空腔圆盘 vs 同棒外环，σ 取 3 根正常角棒）

用法
----
  python scripts/tv_baseline_recon.py --data clean --energy 3 --weight 2e-4
  python scripts/tv_baseline_recon.py --data flaw  --energy 3 --flaw-csv sparse_targets_flaw_3.0.csv
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from data.csv_paths import resolve_csv  # noqa: E402  CSV 统一落点 csv_data/
sys.path.insert(0, str(ROOT / "scripts"))

import baseline_recon as br                       # noqa: E402
import p01_flaw_step3_eval as fe                  # noqa: E402  （CNR 口径，import 安全）

from skimage.restoration import denoise_tv_chambolle   # noqa: E402
from skimage.transform import iradon_sart              # noqa: E402

FIXED_N_OUTER = 4
FIXED_N_ITER_SART = 5       # 4 × 5 = 20 次 SART，与 SART 基线（20 iter, relax 0.15）同预算
FIXED_RELAX = 0.15          # 与 baseline_recon 的 SART 基线完全一致
FIXED_FILTER = "hamming"
SENS_WEIGHTS = (0.5, 2.0)   # 主 weight 的倍率（敏感性，非择优）

OUT_DIR = ROOT / "runs" / "_tv_baseline"


# ------------------------------------------------------------------ 正弦图
def flaw_sinogram(csv_path: Path, fill: float) -> tuple[np.ndarray, np.ndarray]:
    """缺陷数据：直接从 sparse_targets_flaw_*.csv 的 360 行建 (360,1200) 正弦图。

    不读 angles.csv（那是均匀数据的表），与 p01_flaw_step3_eval 的口径一致。
    """
    rows = list(csv.DictReader(csv_path.open(encoding="utf-8")))
    if len(rows) != 360:
        raise SystemExit(f"[FATAL] {csv_path.name} 期望 360 行，实得 {len(rows)}")
    scan_dirs = {r["scan_dir"] for r in rows}
    if len(scan_dirs) != 1:
        raise SystemExit(f"[FATAL] 多个 scan_dir: {scan_dirs}")
    empty = br.load_empty(ROOT / "3_3_flaw", scan_dirs.pop())
    order = sorted(rows, key=lambda r: float(r["target_angle_deg"]))
    sino = np.empty((360, br.OUTPUT_SIZE), dtype=np.float64)
    theta = np.empty(360, dtype=np.float64)
    for i, r in enumerate(order):
        proj = np.load(ROOT / r["target_file"]).astype(np.float32)
        L = br.log_ratio(proj, empty).astype(np.float64)
        sino[i] = np.where(np.isfinite(L), L, float(fill))[br.RECON_ROW]
        theta[i] = float(r["target_angle_deg"])
    if not np.array_equal(np.round(theta).astype(np.int64), np.arange(360)):
        raise SystemExit("[FATAL] 缺陷角度不是 0..359")
    return sino, theta


# ------------------------------------------------------------------ TV 重建
def sart_tv(sino: np.ndarray, theta: np.ndarray, weight: float,
            n_outer: int = FIXED_N_OUTER, n_iter: int = FIXED_N_ITER_SART,
            relax: float = FIXED_RELAX, log_prefix: str = "") -> np.ndarray:
    """SART 与 Chambolle TV 近端交替；SART 全程热启动，零初值。"""
    img: np.ndarray | None = None
    t0 = time.time()
    for outer in range(n_outer):
        for _ in range(n_iter):
            img = iradon_sart(sino.T, theta=theta, image=img, clip=None,
                              relaxation=relax)
        img = denoise_tv_chambolle(img, weight=weight, channel_axis=None)
        print(f"    {log_prefix}outer {outer + 1}/{n_outer}  "
              f"{time.time() - t0:6.1f}s  (min {img.min():+.5f} max {img.max():+.5f})",
              flush=True)
    assert img is not None
    return img


def image_metrics(dense: np.ndarray, img: np.ndarray) -> dict:
    """与 baseline_recon.image_metrics 同口径（FOV 内 valid，psnr_common 统一 data_range）。"""
    mask = br.fov_mask()
    d = dense.astype(np.float64)[mask]
    c = img.astype(np.float64)[mask]
    err = c - d
    rmse = float(np.sqrt(np.mean(err ** 2)))
    full = float(np.sqrt(np.mean((img.astype(np.float64) - dense.astype(np.float64)) ** 2)))
    # psnr_common：四档统一 data_range（见 baseline_recon）
    data_range = 0.2737
    mse_c = float(np.mean(err ** 2))
    psnr_c = float(10 * np.log10(data_range ** 2 / mse_c)) if mse_c > 0 else float("inf")
    from skimage.metrics import structural_similarity
    sl = br.ssim_crop_slices()
    ssim = float(structural_similarity(dense[sl], img[sl], data_range=data_range))
    return {"rmse_valid": rmse, "rmse_full": full,
            "psnr_common": psnr_c, "ssim": ssim}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", choices=("clean", "flaw"), default="clean")
    ap.add_argument("--energy", default="3", help="能量字符串：1.5 / 3 / 4.5 / 6")
    ap.add_argument("--flaw-csv", default=None,
                    help="缺陷数据表（--data flaw 时必填，如 sparse_targets_flaw_3.0.csv）")
    ap.add_argument("--phase", choices=("A", "B"), default="A",
                    help="保留哪个相位作为 180 个实测视角")
    ap.add_argument("--weight", type=float, default=2e-4,
                    help="Chambolle TV 权重（固定设置；未调参）")
    ap.add_argument("--fill", type=float, default=None,
                    help="零计数填充（-log 域）；缺省取该档裁决值")
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "recon").mkdir(parents=True, exist_ok=True)
    tag = f"{args.data}_{args.energy}_{args.phase}"

    if args.data == "clean":
        fill = args.fill if args.fill is not None else 5.50
        sino, theta = br.build_sinogram(args.energy, ROOT / "3_3", fill)
        dense = br.get_dense_reference(args.energy, ROOT / "3_3", fill,
                                       FIXED_FILTER, OUT_DIR / "recon", reuse=True)
    else:
        if not args.flaw_csv:
            raise SystemExit("[FATAL] --data flaw 需要 --flaw-csv")
        fill = args.fill if args.fill is not None else \
            {"1.5": 5.80, "3": 5.30, "4.5": 4.50, "6": 3.80}[args.energy]
        sino, theta = flaw_sinogram(resolve_csv(args.flaw_csv), fill)
        dense = br.restrict_to_fov(br.recon_fbp(sino, theta, FIXED_FILTER))
        np.save(OUT_DIR / f"dense_{tag}.npy", dense.astype(np.float32))

    sel = br.select_phase(theta, args.phase)
    sino_s, theta_s = sino[sel], theta[sel]
    print(f"[{tag}] 稀疏视角 {sino_s.shape[0]}，fill={fill}，"
          f"weight={args.weight:g}", flush=True)

    results: dict = {
        "method": "sart_tv",
        "tv_implementation": ("SART (skimage iradon_sart, warm start, clip=None, "
                              "relaxation=0.15) 与 denoise_tv_chambolle 交替"),
        "tomopy_status": ("仓库自带 tomopy TV 不可用：tomopy 1.14.4 的 Python 包存在，"
                          "但编译库 tomo-recon 缺失，import 报 ModuleNotFoundError"),
        "tuned": False,
        "params": {"n_outer": FIXED_N_OUTER, "n_iter_sart": FIXED_N_ITER_SART,
                   "relaxation": FIXED_RELAX, "filter": FIXED_FILTER,
                   "weight": args.weight, "fill": fill,
                   "n_views": int(sino_s.shape[0]), "phase": args.phase},
        "data": {"kind": args.data, "energy_MeV": args.energy,
                 "flaw_csv": args.flaw_csv},
        "runs": {},
    }

    for mult in (1.0,) + SENS_WEIGHTS:
        w = args.weight * mult
        lab = "primary" if mult == 1.0 else f"sens_x{mult:g}"
        print(f"  -- {lab}: weight={w:g}", flush=True)
        img = sart_tv(sino_s, theta_s, w, log_prefix=f"{lab} ")
        img = br.restrict_to_fov(img)
        m = image_metrics(dense, img)
        entry: dict = {"weight": w, "image": m}
        if args.data == "flaw":
            bm = fe.bubble_metrics(img)
            entry["cnr"] = {"cnr": bm["cnr"], "contrast": bm["contrast"],
                            "noise_sd": bm["noise_sd"],
                            "mu_bubble": bm["mu_bubble"],
                            "mu_reference": bm["mu_reference"],
                            "bubble_radius_px": bm["bubble_radius_px"],
                            "definition": bm["definition"]}
            print(f"     CNR={bm['cnr']:.2f}  rmse={m['rmse_valid']:.6f}", flush=True)
        else:
            print(f"     rmse={m['rmse_valid']:.6f}  psnr_common={m['psnr_common']:.2f}"
                  f"  ssim={m['ssim']:.4f}", flush=True)
        results["runs"][lab] = entry
        np.save(OUT_DIR / f"tv_{tag}_{lab}.npy", img.astype(np.float32))

    out_p = OUT_DIR / f"tv_metrics_{tag}.json"
    out_p.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"已写出 {out_p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
