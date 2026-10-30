"""边棒缺陷（flaw12 / `3_3_flaw_1_2`）跨能量汇总表（含 Row16 逐视角列）。

数据真相源（三处，全部只读）
--------------------------
1. `runs/_flaw12_multi_summary.json`      —— 图像域 RMSE / PSNR_c / SSIM / CNR / MF t
   （由 `scripts/flaw_multi_aggregate.py --index runs/_flaw12_multi_index.csv` 产出）
2. `docs/figs/p01_flaw_step3_indomain_100ep_flaw12/perview_<E>_s<seed>/fig12_per_view_rmse.csv`
   —— Row16 逐视角 CNN vs 线性插值（**masked** 口径，与已交付的 3.0 MeV 数字同源），
   含 `in_signal_lobe` 标记，用于「CNN 更优视角占比」。**本表的 Row16 只认这一处。**

口径红线
--------
* **Row16 有两套值，不要混用**：`fig12_per_view_rmse.csv` 是 **masked**
  （`proj>0 & empty_proj>0`，主口径）；`image_metrics.json → per_view_row16` 是
  **unmasked**（row16 全部 1200 列求 RMS）。3.0/4.5/6.0 MeV 两者逐位相同（mask 全 True），
  **1.5 MeV 差 +1.5%**（该档 row16 有零计数像素）。本表用 masked，与 Fig S8 一致。
* Row16 表是 **360 个待补全视角**（不是 45 个 test 视角）。45-test 口径只有 span sweep
  表用，两张表**禁止横比**（见 `r34_split_attribution.md` §3）。
* 边棒（edge2, r=260 px, az=180°）与角棒（corner1, 367.7 px / 135°）是**两批仿真**，
  产物目录/文件名默认值会互相覆盖，本脚本只认 flaw12 的显式路径。

用法
----
  python scripts/flaw12_crossenergy_table.py
  python scripts/flaw12_crossenergy_table.py --out-md runs/_flaw12_crossenergy_table.md \
      --out-tex docs/tables/flaw12_crossenergy.tex
"""
from __future__ import annotations

import argparse
import csv
import json
import statistics as st
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RUNS = ROOT / "runs"
INDEX = RUNS / "_flaw12_multi_index.csv"
SUMMARY = RUNS / "_flaw12_multi_summary.json"
PERVIEW = ROOT / "docs" / "figs" / "p01_flaw_step3_indomain_100ep_flaw12"

ENERGIES = ["1.5", "3.0", "4.5", "6.0"]
SEEDS = [20261022, 20261023, 20261024]


def _m(x):
    """summary 里的字段可能是 {'mean':..} 或裸 float。"""
    return x["mean"] if isinstance(x, dict) else x


def _sd(x):
    return x["std"] if isinstance(x, dict) else 0.0


def read_index() -> dict[tuple[str, int], str]:
    out: dict[tuple[str, int], str] = {}
    with open(INDEX, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            out[(r["energy"], int(r["seed"]))] = r["run_id"]
    return out


def perview_dir(energy: str, seed: int) -> Path:
    """3.0 MeV 的老目录名是 s<seed>，其余是 perview_<E>_s<seed>。"""
    cand = [PERVIEW / f"perview_{energy}_s{seed}", PERVIEW / f"s{seed}"]
    for c in cand:
        if (c / "fig12_per_view_rmse.csv").is_file():
            return c
    raise FileNotFoundError(f"缺 per-view CSV: {energy} MeV seed {seed} ({cand})")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", default=str(INDEX))
    ap.add_argument("--summary", default=str(SUMMARY))
    ap.add_argument("--out-md", default=str(RUNS / "_flaw12_crossenergy_table.md"))
    ap.add_argument("--out-tex", default="")
    a = ap.parse_args()

    idx = read_index()
    summ = json.load(open(a.summary, encoding="utf-8"))
    pe = summ["per_energy"]

    rows = []
    for E in ENERGIES:
        e = pe[E]
        im = e["image"]
        cnn = _m(im["cnn_completed"]["rmse_valid"])
        lin = _m(im["linear_interp"]["rmse_valid"])
        ratio = cnn / lin

        r16_cnn, r16_lin, better, ratios = [], [], [], []
        for s in SEEDS:
            rid = idx[(E, s)]
            # 主口径：fig12 的 masked Row16（与已交付的 3.0 MeV 数字同源）
            rr = list(csv.DictReader(open(perview_dir(E, s) / "fig12_per_view_rmse.csv",
                                          encoding="utf-8")))
            r16_cnn.append(st.mean(float(x["rmse_cnn"]) for x in rr))
            r16_lin.append(st.mean(float(x["rmse_linear"]) for x in rr))
            better.append(sum(1 for x in rr
                              if float(x["rmse_cnn"]) < float(x["rmse_linear"])) / len(rr))
            ratios += [float(x["rmse_cnn"]) / float(x["rmse_linear"]) for x in rr]

        r16c, r16c_sd = st.mean(r16_cnn), st.stdev(r16_cnn)
        r16l = st.mean(r16_lin)
        rows.append(dict(
            E=E, n=len(SEEDS),
            cnn=cnn, cnn_sd=_sd(im["cnn_completed"]["rmse_valid"]),
            lin=lin, ratio=ratio, ratio_sd=ratio * _sd(im["cnn_completed"]["rmse_valid"]) / cnn,
            r16c=r16c, r16c_sd=r16c_sd, r16l=r16l,
            r16ratio=st.mean(r16_cnn) / r16l, r16ratio_sd=r16c_sd / r16l,
            better=st.mean(better), better_min=min(better), better_max=max(better),
            med_ratio=st.median(ratios),
            psnr=_m(e["psnr_common"]["cnn_completed"]),
            psnr_sd=_sd(e["psnr_common"]["cnn_completed"]),
            psnr_lin=_m(e["psnr_common"]["linear_interp"]),
            ssim=_m(im["cnn_completed"]["ssim"]), ssim_sd=_sd(im["cnn_completed"]["ssim"]),
            cnr=_m(e["cnr"]["cnn_completed"]), cnr_sd=_sd(e["cnr"]["cnn_completed"]),
            cnr_lin=_m(e["cnr"]["linear_interp"]),
            mf_t=_m(e["mf_t_stat"]),
        ))

    # ---- Markdown ----
    L = []
    L.append("### 边棒缺陷（`3_3_flaw_1_2`，edge2 / r=260 px / az=180°）跨能量汇总，n=3 种子 × 100 ep")
    L.append("")
    L.append("**A. Row-16 投影域逐视角（360 个待补全视角；这是本表的主口径）**")
    L.append("")
    L.append("| 能量 | n | Row16 RMSE (CNN) | Row16 RMSE (linear) | CNN/linear | CNN 更优视角占比 | 逐视角比值中位数 |")
    L.append("|---|---|---|---|---|---|---|")
    for r in rows:
        L.append(f"| {r['E']} MeV | {r['n']} | {r['r16c']:.6f} ± {r['r16c_sd']:.6f} | "
                 f"{r['r16l']:.6f} | {r['r16ratio']:.4f} ± {r['r16ratio_sd']:.4f} | "
                 f"{r['better']*100:.1f}% (min {r['better_min']*100:.1f} / max {r['better_max']*100:.1f}) | "
                 f"{r['med_ratio']:.4f} |")
    L.append("")
    L.append("**B. 图像域（180 实测 + 180 补全，45 个 test 视角池化）**")
    L.append("")
    L.append("| 能量 | RMSE_valid (CNN) | RMSE_valid (linear) | CNN/linear | PSNR$_c$ (dB) | "
             "PSNR$_c$ (linear) | SSIM | 空腔 CNR | CNR (linear) | 匹配滤波 $t$ |")
    L.append("|---|---|---|---|---|---|---|---|---|---|")
    for r in rows:
        L.append(f"| {r['E']} MeV | {r['cnn']:.6f} ± {r['cnn_sd']:.6f} | {r['lin']:.6f} | "
                 f"{r['ratio']:.4f} | {r['psnr']:.2f} ± {r['psnr_sd']:.2f} | "
                 f"{r['psnr_lin']:.2f} | "
                 f"{r['ssim']:.4f} ± {r['ssim_sd']:.4f} | {r['cnr']:.3f} ± {r['cnr_sd']:.3f} | "
                 f"{r['cnr_lin']:.3f} | {r['mf_t']:.2f} |")
    L.append("")
    L.append("> 口径：Row16 表用 **360 个待补全视角**；span sweep 表用 **45 个 test 视角**。"
             "两者测量的是不同视角集合上的误差，**禁止横比**（`r34_split_attribution.md` §3）。")
    L.append("> Row16 取 **masked**（`fig12_per_view_rmse.csv`）；`image_metrics.json` 里的 "
             "`per_view_row16` 是 unmasked（row16 全 1200 列求 RMS）。已实测：3.0/4.5/6.0 MeV "
             "mask 覆盖率 = 100%，两者**逐位相同**；1.5 MeV 覆盖率 = 99.993%，即 360 视角合计"
             "约 60 个零计数像素被排除（平均 0.17 px/视角，单视角最多 2 个；`-log` 域 "
             "`zero_log_fill` = 6.3，逐能量 6.3/5.8/5.1/4.5，取自三种子 `config.json`），"
             "其 Σerr² = 0.65–0.78/视角，足以把 360 视角的 unmasked 均值抬高 **+1.5%**"
             "（0.1246 → 0.1265）。两者不可混用，本表只用 masked。")
    L.append("> PSNR$_c$ 用四档统一 data_range（跨能量比较唯一合法口径）；匹配滤波 $t$ 积分 "
             "`flaw − uniform` 两次仿真的差分正弦图，与 CNN 权重无关，故种子 std 恒为 0。")
    md = "\n".join(L) + "\n"
    Path(a.out_md).write_text(md, encoding="utf-8")
    print(f"[OK] {a.out_md}")
    print()
    print(md)

    # ---- LaTeX ----
    if a.out_tex:
        T = []
        T.append("% 边棒缺陷跨能量汇总（flaw12 / 3_3_flaw_1_2）n=3 seeds —— 自动生成，勿手改")
        T.append(r"\begin{table}[t]")
        T.append(r"\centering")
        T.append(r"\caption{Edge-rod cavity (rod \texttt{edge2}, $r=260$~px, azimuth $180^\circ$) "
                 r"across four energies, $n=3$ seeds $\times$ 100 epochs. "
                 r"Row~16 is evaluated on all 360 completed views; the image domain is evaluated "
                 r"on the 45 held-out test views. The learned completion never beats linear "
                 r"interpolation on Row~16 at 3.0/4.5/6.0~MeV and is at parity at 1.5~MeV.}")
        T.append(r"\label{tab:flawedge}")
        T.append(r"\footnotesize")
        T.append(r"\begin{tabular}{lccccc}")
        T.append(r"\hline")
        T.append(r"Energy & Row~16 CNN & Row~16 linear & CNN/linear & CNN-better views & "
                 r"Image RMSE$_\mathrm{valid}$ \\")
        T.append(r"\hline")
        for r in rows:
            T.append(f"{r['E']} MeV & ${r['r16c']:.6f}\\pm{r['r16c_sd']:.6f}$ & "
                     f"${r['r16l']:.6f}$ & ${r['r16ratio']:.3f}\\pm{r['r16ratio_sd']:.3f}$ & "
                     f"{r['better']*100:.0f}\\% & ${r['cnn']:.6f}\\pm{r['cnn_sd']:.6f}$ \\\\")
        T.append(r"\hline")
        T.append(r"\end{tabular}")
        T.append(r"\end{table}")
        p = Path(a.out_tex)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text("\n".join(T) + "\n", encoding="utf-8")
        print(f"[OK] {a.out_tex}")


if __name__ == "__main__":
    main()
