"""临时评估：图像域统一加 ringsort（排序法环校正）对三条线指标的影响。

**这是评估脚本，不改任何既有口径**：只读 sinogram 与 best.pt，把同一套滤镜
（Vo 2018 排序法，tomopy `remove_stripe_based_sorting` 的逐行等价复刻）分别
加到每条线各自的正弦图上，再各自 FBP，最后与 dense 参考比 RMSE / PSNR / SSIM。

为什么要"三条线一起过"
----------------------
只给一张图加滤镜是学术不端的一种形式：滤镜本身会改变误差，若只滤 CNN 那张，
读者无法区分"CNN 更好"与"CNN 那张被额外处理过"。所以 dense / sparse / linear / cnn
四条线的正弦图**全部**过同一个函数、同一组参数（`size=12`，即 tomopy 对 1200 列
探测器的默认值），参数不择优。

各条线的正弦图从哪来
--------------------
dense  : 360 视角全实测                       → FBP
sparse : 180 视角实测（输入相位）              → FBP
linear : 180 实测 + 180 两点算术平均插值（投影域）→ FBP
cnn    : 180 实测 + 180 CNN 补全                → FBP
每条线各自的正弦图各自过 ringsort（视角数不同，滤波器看到的数据自然不同）。

用法
----
  python scripts/eval_ringsort_postproc.py --energy 3 --limit 10
  python scripts/eval_ringsort_postproc.py --energy 3 --limit 10 --size 12 --sizes 5,12,21
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy.ndimage import median_filter

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from data.csv_paths import resolve_csv  # noqa: E402  CSV 统一落点 csv_data/
sys.path.insert(0, str(ROOT / "scripts"))

import baseline_recon as br          # noqa: E402
import cnn_recon as cr               # noqa: E402
from models.unet import UNet         # noqa: E402


# ------------------------------------------------------------------ 环校正
def ringsort(sino: np.ndarray, size: int = 12, dim: int = 1) -> np.ndarray:
    """Vo et al. (2018) 排序法条纹/环伪影去除，逐行等价于 tomopy 的 `_rs_sort`。

    本机 tomopy 的编译库 `tomo-recon` 缺失（`tomopy.misc.corr` 无法 import），
    因此按 `tomopy/prep/stripe.py::_rs_sort` 源码复刻，算法与默认参数完全一致：
      · 对每个探测器列，沿角度轴排序；
      · 在"排序名次"坐标上沿**探测器轴**做窗口中值滤波（size × 1）；
      · 再按原名次还原。
    排序保留单调结构，中值滤波只削掉单列/窄列的增益偏差（= 环的来源）。

    `dim=2` 时窗口为 size × size（tomopy 的另一档，适用宽带伪影）。
    """
    s = np.transpose(np.asarray(sino, dtype=np.float64))      # (det, angles)
    matindex = np.tile(np.arange(s.shape[1], dtype=np.float64), (s.shape[0], 1))
    matcomb = np.dstack((matindex, s))
    matsort = np.array([row[row[:, 1].argsort()] for row in matcomb])
    win = (size, 1) if dim == 1 else (size, size)
    matsort[:, :, 1] = median_filter(matsort[:, :, 1], win)
    matsortback = np.array([row[row[:, 0].argsort()] for row in matsort])
    return np.transpose(matsortback[:, :, 1])


# ------------------------------------------------------------------ 数据
def load_run(run_id: str, device):
    cfg_p = cr.RUNS_DIR / run_id / "config.json"
    ckpt_p = cr.RUNS_DIR / run_id / "best.pt"
    for p in (cfg_p, ckpt_p):
        if not p.is_file():
            raise SystemExit(f"[FATAL] 缺 {p}")
    cfg = json.loads(cfg_p.read_text(encoding="utf-8"))
    norm = cfg["norm"]
    mean, std = float(norm["norm_mean"]), float(norm["norm_std"])
    fill = float(norm["zero_log_fill"])
    n_views = int(cfg.get("input_views") or 3)
    model = UNet(in_channels=n_views + 2).to(device)
    ckpt = __import__("torch").load(ckpt_p, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"])
    model.eval()
    print(f"[run] {run_id}  epoch={ckpt['epoch']}  input_views={n_views}  "
          f"norm=({mean}, {std}, {fill})", flush=True)
    return model, mean, std, fill, n_views, int(ckpt["epoch"])


def two_point_linear(sino: np.ndarray, sel_measured: np.ndarray) -> np.ndarray:
    """投影域两点算术平均：每个未测视角 = 左右相邻两个已测视角的均值。

    与 `baseline_linear.py` 同口径（target = 两侧 ±1° 已测视角的中点）。
    """
    out = sino.copy()
    n = sino.shape[0]
    for a in range(n):
        if sel_measured[a]:
            continue
        out[a] = 0.5 * (sino[(a - 1) % n] + sino[(a + 1) % n])
    return out


# ------------------------------------------------------------------ 指标
def metrics(ref: np.ndarray, cmp_: np.ndarray) -> dict:
    m = br.image_metrics(ref, cmp_, br.fov_mask())
    return {"rmse_valid": m["rmse_valid"], "rmse_full": m["rmse_full"],
            "psnr": m["psnr"], "ssim": m["ssim"]}


def agg(rows: list[dict]) -> dict:
    out = {}
    for k in ("rmse_valid", "rmse_full", "psnr", "ssim"):
        v = np.array([r[k] for r in rows], dtype=np.float64)
        out[k] = float(v.mean())
        out[k + "_sd"] = float(v.std(ddof=1)) if len(v) > 1 else 0.0
    out["n"] = len(rows)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--energy", default="3")
    ap.add_argument("--limit", type=int, default=10, help="前 N 个 test 样本")
    ap.add_argument("--size", type=int, default=12,
                    help="主参数：中值窗口（tomopy 对 1200 列探测器的默认值 = 12）")
    ap.add_argument("--sizes", default="", help="附加敏感性档，逗号分隔，如 5,21")
    ap.add_argument("--run-id", default=cr.DEFAULT_CNN_RUN_ID)
    ap.add_argument("--filter-name", default="hamming")
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    import torch
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = False
    model, mean, std, fill, n_views, best_epoch = load_run(args.run_id, device)

    e = args.energy
    print(f"[{e} MeV] 建正弦图 / dense 参考 ...", flush=True)
    t0 = time.time()
    sino, theta = br.build_sinogram(e, ROOT / "3_3", fill)
    if not np.array_equal(np.round(theta).astype(np.int64), np.arange(360)):
        raise SystemExit("[FATAL] theta 不是 0..359 整数序列")
    dense_raw = br.restrict_to_fov(br.recon_fbp(sino, theta, args.filter_name))
    print(f"  完成 {time.time() - t0:.1f}s", flush=True)

    # CNN 补全：两相位各预测一次（与 cnn_recon.run 同路径）
    header, rows = cr.load_energy_rows(e)
    by_phase = {p: [r for r in rows if r["phase"] == p] for p in ("A", "B")}
    idx_of_angle = {int(a): i for i, a in enumerate(np.round(theta).astype(np.int64))}
    completed: dict[str, np.ndarray] = {}
    for in_phase in ("A", "B"):
        missing = br.opposite_phase(in_phase)
        csv_p = cr.write_subset_csv(e, missing, header, by_phase[missing])
        pred, _mask = cr.predict_views(model, csv_p, mean, std, fill, args.batch, device,
                                       n_views=n_views,
                                       expected_files=[r["target_file"] for r in by_phase[missing]])
        sc = sino.copy()
        for i, r in enumerate(by_phase[missing]):
            sc[idx_of_angle[int(float(r["target_angle_deg"]))]] = \
                pred[i, 0, br.RECON_ROW, :].astype(np.float64)
        completed[in_phase] = sc
        print(f"  [CNN] 输入相位 {in_phase} → 补全 {missing}（{len(by_phase[missing])} 视角）",
              flush=True)

    # 取前 N 个 test 样本
    test_rows = [r for r in rows if r["split"] == "test"]
    test_rows.sort(key=lambda r: int(float(r["target_angle_deg"])))
    sel_rows = test_rows[:args.limit]
    print(f"[样本] 前 {len(sel_rows)} 个 test 视角："
          f"{[int(float(r['target_angle_deg'])) for r in sel_rows]}", flush=True)

    sizes = [args.size] + [int(x) for x in args.sizes.split(",") if x.strip()]
    results: dict = {}
    for size in sizes:
        tag = f"size{size}"
        dense_rs = br.restrict_to_fov(
            br.recon_fbp(ringsort(sino, size), theta, args.filter_name)) if size else dense_raw
        rowset: dict[str, list[dict]] = {k: [] for k in
                                         ("sparse", "linear", "cnn", "sparse_rs", "linear_rs", "cnn_rs")}
        rowset_vs_rs: dict[str, list[dict]] = {k: [] for k in rowset}
        per_sample = []
        for r in sel_rows:
            in_phase = br.opposite_phase(r["phase"])
            sel = br.select_phase(theta, in_phase)                # 已测 180 视角
            th = theta[sel]
            # sparse：只用已测 180 视角
            img_sparse = br.restrict_to_fov(br.recon_fbp(sino[sel], th, args.filter_name))
            # linear / cnn：360 视角（已测 + 补齐）
            img_linear = br.restrict_to_fov(
                br.recon_fbp(two_point_linear(sino, sel), theta, args.filter_name))
            img_cnn = br.restrict_to_fov(
                br.recon_fbp(completed[in_phase], theta, args.filter_name))
            # 过 ringsort 的版本（各线各自的正弦图）
            img_sparse_rs = br.restrict_to_fov(
                br.recon_fbp(ringsort(sino, size)[sel], th, args.filter_name))
            img_linear_rs = br.restrict_to_fov(
                br.recon_fbp(ringsort(two_point_linear(sino, sel), size), theta, args.filter_name))
            img_cnn_rs = br.restrict_to_fov(
                br.recon_fbp(ringsort(completed[in_phase], size), theta, args.filter_name))

            for k, img in (("sparse", img_sparse), ("linear", img_linear), ("cnn", img_cnn),
                           ("sparse_rs", img_sparse_rs), ("linear_rs", img_linear_rs),
                           ("cnn_rs", img_cnn_rs)):
                rowset[k].append(metrics(dense_raw, img))
                rowset_vs_rs[k].append(metrics(dense_rs, img))
            per_sample.append({"angle": int(float(r["target_angle_deg"])),
                               "phase": r["phase"],
                               "cnn_rmse": rowset["cnn"][-1]["rmse_valid"],
                               "cnn_rs_rmse": rowset["cnn_rs"][-1]["rmse_valid"]})
        results[tag] = {"size": size,
                        "vs_dense_raw": {k: agg(v) for k, v in rowset.items()},
                        "vs_dense_ringsort": {k: agg(v) for k, v in rowset_vs_rs.items()},
                        "per_sample": per_sample}
        print(f"[size={size}] 完成", flush=True)

    # ------------------------------------------------------------- 报表
    def table(block: dict, title: str) -> None:
        print("\n" + "=" * 96)
        print(title)
        print("-" * 96)
        print(f"{'线':<12}{'RMSE_valid':>14}{'RMSE_full':>14}{'PSNR(dB)':>11}{'SSIM':>10}")
        for k in ("sparse", "linear", "cnn", "sparse_rs", "linear_rs", "cnn_rs"):
            d = block[k]
            print(f"{k:<12}{d['rmse_valid']:>10.6f}±{d['rmse_valid_sd']:<3.0e}"
                  f"{d['rmse_full']:>10.6f}   {d['psnr']:>8.2f} {d['ssim']:>9.4f}")

    main_tag = f"size{args.size}"
    table(results[main_tag]["vs_dense_raw"], f"对照 dense（未滤）参考 | ringsort size={args.size}")
    table(results[main_tag]["vs_dense_ringsort"], f"对照 dense（同 size 滤过）参考 | ringsort size={args.size}")

    print("\n" + "=" * 96)
    print("关键判读量（vs dense 未滤参考）")
    print("-" * 96)
    b = results[main_tag]["vs_dense_raw"]
    for k in ("sparse", "linear", "cnn"):
        a, c = b[k]["rmse_valid"], b[k + "_rs"]["rmse_valid"]
        print(f"{k:<8} RMSE {a:.6f} → {c:.6f}  ({100*(c-a)/a:+.2f}%)")
    gap_a = 100 * (b["cnn"]["rmse_valid"] - b["sparse"]["rmse_valid"]) / b["sparse"]["rmse_valid"]
    gap_b = 100 * (b["cnn_rs"]["rmse_valid"] - b["sparse_rs"]["rmse_valid"]) / b["sparse_rs"]["rmse_valid"]
    print(f"CNN 相对 sparse-FBP 的 RMSE 差：{gap_a:+.2f}% → 加 ringsort 后 {gap_b:+.2f}%")
    if "linear" in b:
        g0 = 100 * (b["cnn"]["rmse_valid"] - b["linear"]["rmse_valid"]) / b["linear"]["rmse_valid"]
        g1 = 100 * (b["cnn_rs"]["rmse_valid"] - b["linear_rs"]["rmse_valid"]) / b["linear_rs"]["rmse_valid"]
        print(f"CNN 相对 linear 的 RMSE 差：{g0:+.2f}% → 加 ringsort 后 {g1:+.2f}%")

    if len(sizes) > 1:
        print("\n[敏感性] RMSE_valid 随 size 变化（vs dense 未滤）")
        print(f"{'size':>6}" + "".join(f"{k:>14}" for k in
                                       ("sparse", "linear", "cnn", "cnn_rs")))
        for size in sizes:
            blk = results[f"size{size}"]["vs_dense_raw"]
            print(f"{size:>6}" + "".join(f"{blk[k]['rmse_valid']:>14.6f}"
                                         for k in ("sparse", "linear", "cnn", "cnn_rs")))

    out_p = Path(args.out) if args.out else (
        ROOT / "runs" / f"_ringsort_postproc_{e.replace('.', 'p')}.json")
    payload = {"energy": e, "limit": args.limit, "run_id": args.run_id,
               "filter_name": args.filter_name, "best_epoch": best_epoch,
               "ringsort_note": "Vo 2018 排序法，逐行等价 tomopy _rs_sort；"
                                "各线各自正弦图过同一函数同一参数",
               "results": results}
    out_p.parent.mkdir(parents=True, exist_ok=True)
    out_p.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n已写出 {out_p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
