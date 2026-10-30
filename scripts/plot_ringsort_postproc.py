"""临时图：ringsort 对三条线的视觉与指标影响（3.0 MeV，单个 test 视角）。

只用于决策，不是论文图。左上到右下：
  第一行 重建图（dense / sparse / sparse+rs / cnn / cnn+rs），共用一个灰度窗（dense 的 p1–p99）；
  第二行 与 dense 的绝对误差 |Δ|，共用一个色窗（各图误差的最大值）。

用法：python scripts/plot_ringsort_postproc.py [--angle 42] [--size 12]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

import baseline_recon as br      # noqa: E402
import cnn_recon as cr           # noqa: E402
from eval_ringsort_postproc import ringsort, two_point_linear, load_run, metrics  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--energy", default="3")
    ap.add_argument("--angle", type=int, default=42)
    ap.add_argument("--size", type=int, default=12)
    ap.add_argument("--run-id", default=cr.DEFAULT_CNN_RUN_ID)
    ap.add_argument("--filter-name", default="hamming")
    ap.add_argument("--out", default="runs/_ringsort_postproc/fig_ringsort_effect.png")
    args = ap.parse_args()

    import torch
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, mean, std, fill, n_views, _ep = load_run(args.run_id, device)

    e = args.energy
    sino, theta = br.build_sinogram(e, ROOT / "3_3", fill)
    dense = br.restrict_to_fov(br.recon_fbp(sino, theta, args.filter_name))

    header, rows = cr.load_energy_rows(e)
    by_phase = {p: [r for r in rows if r["phase"] == p] for p in ("A", "B")}
    idx_of_angle = {int(a): i for i, a in enumerate(np.round(theta).astype(np.int64))}

    tgt = [r for r in rows if r["split"] == "test"
           and int(float(r["target_angle_deg"])) == args.angle][0]
    phase = tgt["phase"]
    in_phase = br.opposite_phase(phase)
    missing = phase
    csv_p = cr.write_subset_csv(e, missing, header, by_phase[missing])
    pred, _m = cr.predict_views(model, csv_p, mean, std, fill, 8, device, n_views=n_views,
                                expected_files=[r["target_file"] for r in by_phase[missing]])
    sc = sino.copy()
    for i, r in enumerate(by_phase[missing]):
        sc[idx_of_angle[int(float(r["target_angle_deg"]))]] = \
            pred[i, 0, br.RECON_ROW, :].astype(np.float64)

    sel = br.select_phase(theta, in_phase)
    th = theta[sel]
    imgs = {
        "dense": dense,
        "sparse FBP": br.restrict_to_fov(br.recon_fbp(sino[sel], th, args.filter_name)),
        f"sparse FBP\n+ringsort({args.size})": br.restrict_to_fov(
            br.recon_fbp(ringsort(sino, args.size)[sel], th, args.filter_name)),
        "CNN completed": br.restrict_to_fov(br.recon_fbp(sc, theta, args.filter_name)),
        f"CNN completed\n+ringsort({args.size})": br.restrict_to_fov(
            br.recon_fbp(ringsort(sc, args.size), theta, args.filter_name)),
    }
    mets = {k: metrics(dense, v) for k, v in imgs.items()}

    vmin, vmax = np.percentile(dense, [1, 99])
    errs = {k: np.abs(v - dense) for k, v in imgs.items() if k != "dense"}
    emax = max(v.max() for v in errs.values())

    fig, axes = plt.subplots(2, 5, figsize=(15.0, 6.6))
    for j, (name, img) in enumerate(imgs.items()):
        ax = axes[0, j]
        ax.imshow(img, cmap="gray", vmin=vmin, vmax=vmax)
        ax.set_title(name, fontsize=9)
        ax.set_xticks([])
        ax.set_yticks([])
        if name == "dense":
            ax.text(0.5, -0.06, "reference", transform=ax.transAxes,
                    ha="center", fontsize=8, color="0.3")
        else:
            m = mets[name]
            ax.text(0.5, -0.09,
                    f"RMSE {m['rmse_valid']:.2e}   SSIM {m['ssim']:.4f}",
                    transform=ax.transAxes, ha="center", fontsize=8)
        if name != "dense":
            ax2 = axes[1, j]
            im = ax2.imshow(errs[name], cmap="inferno", vmin=0, vmax=emax)
            ax2.set_xticks([])
            ax2.set_yticks([])
    # 左下角空槽放色条（比一张空白 panel 好看，也避免色条压到最后一列）
    cax = axes[1, 0]
    cax.axis("off")
    inset = cax.inset_axes([0.10, 0.42, 0.80, 0.16])
    cb = fig.colorbar(im, cax=inset, orientation="horizontal")
    cb.set_label(f"|recon − dense|, shared 0–{emax:.1e}", fontsize=8)
    cb.ax.tick_params(labelsize=7)
    cax.text(0.5, 0.22, "absolute error", transform=cax.transAxes,
             ha="center", fontsize=8, color="0.3")
    for j in range(1, 5):
        axes[1, j].set_title("absolute error", fontsize=8, color="0.3")

    fig.suptitle(f"{e} MeV, target view {args.angle}$^\\circ$ — same ringsort applied to every line, "
                 f"window {args.size} (tomopy default for 1200 columns)", fontsize=10)
    fig.subplots_adjust(left=0.01, right=0.99, top=0.90, bottom=0.05, wspace=0.05, hspace=0.24)
    out = Path(args.out)
    if not out.is_absolute():
        out = ROOT / out
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=200)
    fig.savefig(out.with_suffix(".pdf"))
    print(f"已写出 {out}")
    for k, v in mets.items():
        print(f"  {k.replace(chr(10), ' '):<34} RMSE {v['rmse_valid']:.6f}  "
              f"PSNR {v['psnr']:.2f}  SSIM {v['ssim']:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
