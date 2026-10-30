#!/usr/bin/env python
"""P0-1 剂量档 · Row 16 逐视角配对 bootstrap。

口径严格对齐 §5.6（f=1.0 那套）：
  - 逐 view 等权算术平均，pooled 4 能量 × 45 test 视角 = 180 个配对样本；
  - CNN 的每个 view 取该 f 下 3 个种子的均值（ randomness 只来自视角重采样，不来自种子均值）；
  - Δ_view = linear_view − CNN_view，**正 = CNN 更好**；
  - 配对 bootstrap：对 180 个视角有放回重采样 N_BOOT 次（同一组 index 同时取 linear 与 CNN）；
  - 报 Δ 均值 + 95% percentile CI、improvement% + CI、Cohen's d_z + CI。

自检：f=1.0 必须复现已发布值 Δ=+0.002242、+2.05%、d_z=+0.200；对不上即报错退出。

用法：python scripts/dose_bootstrap.py
产出：runs/_dose_bootstrap.json 、 runs/_dose_bootstrap.md
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
RUNS = ROOT / "runs"
FACTORS = ["1", "0.5", "0.25", "0.125"]
ENERGIES = ["1.5", "3", "4.5", "6"]
N_BOOT = 10000
SEED = 20261022

# f=1.0 已发布值（§5.6），用于口径自检。
# 符号约定（两者天然反号，务必分清）：
#   Δ = CNN − linear，**正 = CNN 更差** → f=1.0 为 +0.002242、d_z +0.200
#   improvement% = (linear − CNN)/linear，**正 = CNN 更好** → f=1.0 为 −2.05（即 CNN 更差 2.05%）
REF_F1 = {"delta": 0.002242, "imp": -2.05, "dz": 0.200}
TOL = 5e-4  # Δ 允许 5e-4 以内的数值差异


def eval_path(f: str) -> Path:
    return RUNS / f"_dose_eval_f{f}.json"


def load(f: str) -> dict:
    p = eval_path(f)
    if not p.is_file():
        raise SystemExit(f"[FATAL] 缺 {p}")
    return json.loads(p.read_text(encoding="utf-8"))


def per_view(d: dict, e: str) -> tuple[np.ndarray, np.ndarray]:
    """某能量下 (cnn_mean_per_view, linear_per_view)，都是 test 视角且按 JSON 顺序配对。"""
    ps = d["energies"][e]["per_sample_test"]
    cks = d["run_ids"]
    if not cks:
        raise SystemExit("[FATAL] 该档无 CNN run_ids")
    lin = np.array([float(p["linear_interp"]) for p in ps])
    # 3 种子在该 view 上的均值
    cnn = np.array([np.mean([float(p[k]) for k in cks]) for p in ps])
    if lin.size != cnn.size:
        raise SystemExit("[FATAL] per-sample 数量不一致")
    return cnn, lin


def pooled(d: dict) -> tuple[np.ndarray, np.ndarray]:
    cs, ls = [], []
    for e in ENERGIES:
        if e not in d["energies"]:
            continue
        c, l = per_view(d, e)
        cs.append(c)
        ls.append(l)
    return np.concatenate(cs), np.concatenate(ls)


def bootstrap(cnn: np.ndarray, lin: np.ndarray, seed: int):
    """配对重采样。返回 dict(delta_mean, delta_lo, delta_hi, imp, imp_lo, imp_hi,
    dz, dz_lo, dz_hi, n, mean_cnn, mean_lin)。"""
    n = cnn.size
    # Δ = CNN − linear：**正 = CNN 更差**（对齐 §5.6 的符号约定）
    delta = cnn - lin
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(N_BOOT, n))
    cb, lb = cnn[idx], lin[idx]            # (N_BOOT, n) 配对
    db = cb - lb
    mb = db.mean(axis=1)
    imp_b = (lb.mean(axis=1) - cb.mean(axis=1)) / lb.mean(axis=1) * 100.0
    # Cohen's d_z：配对差分的均值 / 标准差（ddof=1），逐次重采样版本
    sdb = db.std(axis=1, ddof=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        dz_b = np.where(sdb > 0, mb / sdb, np.nan)

    def ci(v):
        v = v[~np.isnan(v)]
        return float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))

    dlo, dhi = ci(mb)
    ilo, ihi = ci(imp_b)
    zlo, zhi = ci(dz_b)
    return {
        "n": int(n),
        "mean_cnn": float(cnn.mean()),
        "mean_lin": float(lin.mean()),
        "delta": float(delta.mean()),
        "delta_ci": [dlo, dhi],
        "imp_pct": float((lin.mean() - cnn.mean()) / lin.mean() * 100.0),
        "imp_ci": [ilo, ihi],
        "dz": float(delta.mean() / delta.std(ddof=1)),
        "dz_ci": [zlo, zhi],
        "significant": bool(dlo > 0 or dhi < 0),   # CI 不跨 0
        # Δ 正 = CNN 更差 ⇒ dlo>0 表示 linear 显著更优；dhi<0 表示 CNN 显著更优
        "favours": ("linear" if dlo > 0 else ("CNN" if dhi < 0 else "n.s.")),
    }


def selfcheck_f1(res: dict) -> None:
    """口径自检：必须复现 §5.6 已发布值。"""
    ok = True
    if abs(res["delta"] - REF_F1["delta"]) > TOL:
        print(f"[FATAL] f=1.0 Δ 复现失败：{res['delta']:.6f} vs 期望 {REF_F1['delta']}")
        ok = False
    if abs(res["imp_pct"] - REF_F1["imp"]) > 0.05:
        print(f"[FATAL] f=1.0 improvement% 复现失败：{res['imp_pct']:.3f} vs {REF_F1['imp']}")
        ok = False
    if abs(res["dz"] - REF_F1["dz"]) > 0.01:
        print(f"[FATAL] f=1.0 d_z 复现失败：{res['dz']:.3f} vs {REF_F1['dz']}")
        ok = False
    if not ok:
        raise SystemExit("[FATAL] 自检未通过，口径与 §5.6 不一致，禁止产出。")
    print(f"[自检 OK] f=1.0 复现 §5.6：Δ={res['delta']:.6f} "
          f"imp={res['imp_pct']:.2f}% d_z={res['dz']:.3f}")


def main() -> int:
    out = {"n_boot": N_BOOT, "seed": SEED, "pool": {}, "per_energy": {}}

    for f in FACTORS:
        try:
            d = load(f)
        except SystemExit as exc:
            print(f"[SKIP] f={f}: {exc}")
            continue
        cnn, lin = pooled(d)
        res = bootstrap(cnn, lin, SEED)
        if f == "1":
            selfcheck_f1(res)
        out["pool"][f] = res

        pe: dict[str, dict] = {}
        for e in ENERGIES:
            if e not in d["energies"]:
                continue
            ce, le = per_view(d, e)
            pe[e] = bootstrap(ce, le, SEED)
        out["per_energy"][f] = pe

        flag = "显著" if res["significant"] else "不显著"
        print(f"f={f:5s} n={res['n']:3d} Δ={res['delta']:+.6f} "
              f"CI[{res['delta_ci'][0]:+.6f},{res['delta_ci'][1]:+.6f}] "
              f"imp={res['imp_pct']:+.2f}% CI[{res['imp_ci'][0]:+.2f},{res['imp_ci'][1]:+.2f}] "
              f"d_z={res['dz']:+.3f} → {flag}({res['favours']})")

    jp = RUNS / "_dose_bootstrap.json"
    jp.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n[json] {jp}")
    write_md(out, RUNS / "_dose_bootstrap.md")
    print(f"[md]   {RUNS / '_dose_bootstrap.md'}")
    return 0


def write_md(out: dict, path: Path) -> None:
    L: list[str] = []
    L.append("# P0-1 · Row 16 逐视角配对 bootstrap（剂量档）\n")
    L.append(f"> 口径同 §5.6：**逐 view 等权算术平均**，pooled 4 能量 × 45 test 视角 = **n=180**；"
             f"CNN 的每 view 取该剂量档 **3 种子的均值**；bootstrap **{out['n_boot']}** 次（seed {out['seed']}）。\n")
    L.append("> **符号约定（两者天然反号，务必分清）**：  \n")
    L.append("> - Δ = CNN − linear，**正 = CNN 更差**（对齐 §5.6）；  \n")
    L.append("> - improvement% = (linear − CNN)/linear，**正 = CNN 更好**（对齐 improvement 表）。  \n")
    L.append("> CI 均为 2.5/97.5 百分位。\n")
    L.append("\n## 池化结果（n=180）\n")
    L.append("| f | CNN row16 | linear row16 | Δ (CNN−lin，正=CNN更差) | Δ 95% CI | improvement % (正=CNN更好) | imp 95% CI | Cohen's d_z | d_z 95% CI | 显著性 |")
    L.append("|---|---|---|---|---|---|---|---|---|---|")
    for f in FACTORS:
        r = out["pool"].get(f)
        if not r:
            continue
        L.append(
            f"| {f} | {r['mean_cnn']:.5f} | {r['mean_lin']:.5f} | {r['delta']:+.6f} | "
            f"[{r['delta_ci'][0]:+.6f}, {r['delta_ci'][1]:+.6f}] | {r['imp_pct']:+.2f} | "
            f"[{r['imp_ci'][0]:+.2f}, {r['imp_ci'][1]:+.2f}] | {r['dz']:+.3f} | "
            f"[{r['dz_ci'][0]:+.3f}, {r['dz_ci'][1]:+.3f}] | "
            f"{'**显著**' if r['significant'] else '不显著'}（{r['favours']}） |")
    L.append("\n## 逐能量结果（n=45）\n")
    for f in FACTORS:
        pe = out["per_energy"].get(f)
        if not pe:
            continue
        L.append(f"\n### f = {f}\n")
        L.append("| energy | Δ | Δ 95% CI | improvement % | imp 95% CI | Cohen's d_z | 显著性 |")
        L.append("|---|---|---|---|---|---|---|")
        for e in ENERGIES:
            if e not in pe:
                continue
            r = pe[e]
            L.append(f"| {e} | {r['delta']:+.6f} | "
                     f"[{r['delta_ci'][0]:+.6f}, {r['delta_ci'][1]:+.6f}] | {r['imp_pct']:+.2f} | "
                     f"[{r['imp_ci'][0]:+.2f}, {r['imp_ci'][1]:+.2f}] | {r['dz']:+.3f} | "
                     f"{'显著' if r['significant'] else '不显著'}（{r['favours']}） |")

    p = out["pool"]
    L.append("\n## 结论（由数据自动生成）\n")
    for f in FACTORS:
        r = p.get(f)
        if not r:
            continue
        if not r["significant"]:
            L.append(f"- **f={f}**：Δ 的 95% CI 跨 0 → Row 16 上 CNN 与线性插值**不可区分（n.s.）**，"
                     f"点估计 improvement = {r['imp_pct']:+.2f}%")
        elif r["favours"] == "CNN":
            L.append(f"- **f={f}**：Δ 的 CI 完全位于 0 以下 → CNN **统计学显著优于**线性插值，"
                     f"improvement = {r['imp_pct']:+.2f}% "
                     f"（95% CI [{r['imp_ci'][0]:+.2f}, {r['imp_ci'][1]:+.2f}]），"
                     f"Cohen's d_z = {r['dz']:+.3f} [{r['dz_ci'][0]:+.3f}, {r['dz_ci'][1]:+.3f}]")
        else:
            L.append(f"- **f={f}**：Δ 的 CI 完全位于 0 以上 → 线性插值**统计学显著优于** CNN，"
                     f"improvement = {r['imp_pct']:+.2f}%（负号即 CNN 更差）")
    L.append("\n## 口径与陷阱\n")
    L.append("- 本表是 **Row 16**（探测器峰顶行）。**all-32-row / 图像域** 的结论更强；"
             "论文写「CNN 何时显著优于线性」时主语应为 all-32-row 或图像域。\n")
    L.append("- 本 bootstrap 只做**视角重采样**，未对训练种子再抽样（种子不确定性另由 3-种子 std 反映），"
             "故 CI 偏窄，属「给定这批 checkpoint 下」的条件显著性。\n")
    L.append("- **RMSE 聚合方式差异（非 bug）**：本表是 per-view RMSE 的算术平均，而 "
             "`improvement_table.md` 是角度池化 RMSE（先合并平方误差再开方）；二者因 sqrt 非线性而"
             "数值不同（例如 f=1.0：−2.05% vs −1.07%）。两套都对，引用时必须注明是哪一套。\n")
    path.write_text("\n".join(L) + "\n", encoding="utf-8")


if __name__ == "__main__":
    sys.exit(main())
