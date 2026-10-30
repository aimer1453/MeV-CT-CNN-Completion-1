"""两视角 vs 三视角消融对比表。

读取：
  runs/_ablation_3view_baseline.json                  （3 视角，runs/20261024_0400_cnn_bs8_lr1e-4_cnn_recon）
  runs/<2view_rid>_cnn_recon/test_metrics.json        （2 视角消融 run 的重建评测）
  runs/<2view_rid>/train_log.csv                      （训练曲线摘要：best epoch / pooled val RMSE）

输出（stdout，markdown 表格，可直接粘贴进 docs/handoff/algo_delivered.md）：
  1) 投影域 row16 / all32 RMSE，逐能量逐相位 A/B + 均值
  2) 图像域 overall（rmse_valid / psnr_common / ssim）
  3) 训练收敛摘要（best epoch、best/last 逐能量 val RMSE、像素池化）

口径警告（与论文侧一致，不得混用）：
  - row16 = 只算 RECON_ROW=16（进入正弦图的那一行）；all32 = 全 32 行 mask 加权。
    1.5 MeV 两口径相差约 3.5 倍，跨行比较必须同口径。
  - 消融结论以 **row16** 为主指标（审稿人明确要求的口径）。
  - 图像域 rmse_full <= rmse_valid（圆外两图皆 0），与投影域的不等式方向相反。

用法：
  python scripts/ablation_2view_vs_3view.py <2view_run_id>
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "runs"
BASELINE_JSON = RUNS / "_ablation_3view_baseline.json"
ENERGIES = ["1.5", "3", "4.5", "6"]


def load_train_summary(run_id: str) -> dict:
    rows = list(csv.DictReader(
        (RUNS / run_id / "train_log.csv").open(encoding="utf-8")))
    val = [r for r in rows if r["split"] == "val"]
    if not val:
        raise SystemExit(f"{run_id}/train_log.csv 无 val 行")
    last_ep = max(int(r["epoch"]) for r in val)
    # best epoch：按 pooled val rmse（像素加权再开方，与 train.py pooled() 同式）
    import math
    by_ep: dict[int, list[tuple[float, float]]] = {}
    for r in val:
        by_ep.setdefault(int(r["epoch"]), []).append(
            (float(r["rmse_valid"]) ** 2 * float(r["n_samples"]), float(r["n_samples"])))
    pooled = {ep: math.sqrt(sum(a for a, _ in v) / sum(b for _, b in v))
              for ep, v in by_ep.items()}
    best_ep = min(pooled, key=pooled.get)
    per_e_best, per_e_last = {}, {}
    for e in ENERGIES:
        be = [r for r in val if int(r["epoch"]) == best_ep and r["energy_MeV"] == e]
        le = [r for r in val if int(r["epoch"]) == last_ep and r["energy_MeV"] == e]
        if be:
            per_e_best[e] = float(be[0]["rmse_valid"])
        if le:
            per_e_last[e] = float(le[0]["rmse_valid"])
    return {"best_epoch": best_ep, "last_epoch": last_ep,
            "pooled_best": pooled[best_ep], "pooled_last": pooled[last_ep],
            "per_energy_best": per_e_best, "per_energy_last": per_e_last}


def proj_table(baseline: dict, two: dict) -> str:
    b, t = baseline["projection_domain"], two["projection_domain"]
    lines = [
        "| Energy (MeV) | Phase | row16 3-view | row16 2-view | Δ row16 | all32 3-view | all32 2-view | Δ all32 |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for e in ENERGIES:
        for ph in ("A", "B"):
            k = f"{e}_{ph}"
            r16b, r16t = b[k]["row16_all"], t[k]["row16_all"]
            a32b, a32t = b[k]["all32_all"], t[k]["all32_all"]
            d16 = 100.0 * (r16t - r16b) / r16b
            d32 = 100.0 * (a32t - a32b) / a32b
            lines.append(
                f"| {e} | {ph} | {r16b:.6f} | {r16t:.6f} | {d16:+.1f}% | "
                f"{a32b:.6f} | {a32t:.6f} | {d32:+.1f}% |")
    # 均值（逐相位算术平均，仅作摘要；主结论看逐档方向一致性）
    for tag, key in (("row16", "row16_all"), ("all32", "all32_all")):
        mb = sum(b[f"{e}_{p}"][key] for e in ENERGIES for p in "AB") / 8
        mt = sum(t[f"{e}_{p}"][key] for e in ENERGIES for p in "AB") / 8
        lines.append(f"| **mean** | A+B | **{mb:.6f}** | **{mt:.6f}** | "
                     f"**{100 * (mt - mb) / mb:+.1f}%** | | | |")
    return "\n".join(lines)


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit(__doc__)
    rid2 = sys.argv[1].removesuffix("_cnn_recon")
    baseline = json.loads(BASELINE_JSON.read_text(encoding="utf-8"))
    m2 = json.loads((RUNS / f"{rid2}_cnn_recon" / "test_metrics.json")
                    .read_text(encoding="utf-8"))
    two = {"projection_domain": {}, "image_domain": m2.get("overall")}
    for e, d in m2["projection_domain_check"].items():
        for ph in ("A", "B"):
            r16 = d["by_input_phase"][ph]["row16"]
            a32 = d["by_input_phase"][ph]["all32"]
            two["projection_domain"][f"{e}_{ph}"] = {
                "row16_all": r16["rmse_valid_all"],
                "all32_all": a32["rmse_valid_all"],
            }
    s2 = load_train_summary(rid2)
    s3 = load_train_summary("20261024_0400_cnn_bs8_lr1e-4")

    print("## 投影域 RMSE（物理域，valid 像素；主指标 = row16）\n")
    print(proj_table(baseline, two))
    print("\n## 图像域 overall（CNN-completed 360 视角 FBP vs dense FBP）\n")
    print("| run | input_views | rmse_valid | rmse_full | psnr_common | ssim |")
    print("|---|---|---|---|---|---|")
    for tag, d in (("3-view (20261024_0400)", baseline["image_domain"]),
                   ("2-view (" + rid2 + ")", two["image_domain"])):
        if not d:
            continue
        print(f"| {tag} | {3 if '3-view' in tag else 2} | {d['rmse_valid']:.6f} | "
              f"{d['rmse_full']:.6f} | {d['psnr_common']:.2f} | {d['ssim']:.4f} |")
    print("\n## 训练收敛摘要（像素池化 val RMSE，train.py pooled() 同式）\n")
    print("| run | best epoch | pooled best | pooled last |")
    print("|---|---|---|---|")
    for tag, s in (("3-view", s3), ("2-view", s2)):
        print(f"| {tag} | {s['best_epoch']}/{s['last_epoch']} | "
              f"{s['pooled_best']:.6f} | {s['pooled_last']:.6f} |")
    print("\n逐能量 val RMSE（best / last）：")
    for tag, s in (("3-view", s3), ("2-view", s2)):
        cells = "  ".join(
            f"{e}: {s['per_energy_best'].get(e, float('nan')):.4f}/"
            f"{s['per_energy_last'].get(e, float('nan')):.4f}"
            for e in ENERGIES)
        print(f"  {tag}: {cells}")


if __name__ == "__main__":
    main()
