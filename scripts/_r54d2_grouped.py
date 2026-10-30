# -*- coding: utf-8 -*-
"""
R54d2: §5.3 grouped 100-epoch confirmed conclusion + cluster-robust 48->46 fix.

Basis: docs/handoff/algo_delivered.md R44 (2026-10-30 19:55), runs/_grp_pairing_100ep.json
and runs/_dose_cluster_bootstrap.json. Confirmed numbers:
  - cluster-robust defensible clustering = sector x energy, 46 NON-EMPTY blocks
    (48 nominal; empty: 0-29 deg @4.5 MeV, 60-89 deg @3.0 MeV) -> both eq and pool
    CIs cross zero. Handoff explicitly requests 48 -> 46 in abstract/§5.6/conclusion.
  - 100-epoch grouped vs random (3.0 MeV, CNN/linear view-aggregated):
      a32 pooled: random 0.384 vs grouped 0.617 (per-seed 0.591-0.654),
                  grouped/random = 1.61 (+60.7%)
      row16 (D1, 180 held-out views, 5 seeds): CNN/linear ratio 1.067 (random)
                  -> 2.156 (grouped); grouped CNN 2.16x WORSE than linear.
    => the 30-epoch 4.4% grouped reduction is an underfitting artefact;
       "random split does not significantly overestimate" does not survive convergence.

Synced beyond §5.3 (same claim, must not survive elsewhere):
  §5.2 trailing claim, Fig sectorsplit caption, and the now-false bridge clause
  "so removing the rotated duplicates from training does not remove information
   the network needs" (contradicted by +60.7%).

EN + ZH. Usage: python _r54d2_grouped.py --check | --apply
"""
import sys
import os

HERE = os.path.dirname(os.path.abspath(__file__))
LATEX = os.path.normpath(os.path.join(HERE, "..", "docs", "manuscript", "latex"))

FILES = {
    "en": os.path.join(LATEX, "main.tex"),
    "zh": os.path.join(LATEX, "main_zh.tex"),
}

# (old, new, expected_count)
OPS = {
    "en": [
        # op1: 48 -> 46 non-empty sector x energy blocks (abstract, §5.6, conclusion)
        (
            "($30^\\circ$ sectors, 48 blocks)",
            "($30^\\circ$ sectors, 46 non-empty sector×energy blocks)",
            3,
        ),
        # op2: tab:decision row 1 — add the block count
        (
            "cluster-robust interval does not exclude zero \\\\",
            "cluster-robust interval (46 non-empty blocks) does not exclude zero \\\\",
            1,
        ),
        # op3: §5.3 confirmed 100-epoch grouped result
        (
            "The grouped split therefore yields a slightly lower error, indicating the random split does not significantly overestimate performance at the 30-epoch checkpoint. The 100-epoch random-split value of 0.3804 reported above is a separate checkpoint and is not pooled with the 30-epoch grouped comparison.",
            "At the 100-epoch checkpoint this ordering reverses. On the all-32-row pool the grouped split reaches 0.617 against 0.384 for the random split (per-seed 0.591--0.654), a grouped-to-random ratio of 1.61 ($+60.7\\%$); on the imaging row the completion-to-interpolation ratio rises from 1.06 under the random split to 2.16 under the grouped split, so once the pixel-identical rotated views are withheld from training, the completion trails linear interpolation by a factor of 2.16 on the row that enters the reconstruction. The 4.4\\% reduction at the 30-epoch checkpoint is therefore an underfitting artefact rather than evidence that the random split is unbiased; after convergence the random split overstates absolute accuracy on this phantom, because a held-out view and its 90\\dgr{}-rotated counterpart are pixel-identical up to Monte-Carlo noise and land on opposite sides of the partition.",
            1,
        ),
        # op4: remove the now-false bridge clause (contradicted by +60.7%)
        (
            "This is not data leakage: the 90\\dgr{} symmetry is a constant that the network can learn from any sector, rather than a copy of a specific target view, so removing the rotated duplicates from training does not remove information the network needs.",
            "This redundancy is a property of the symmetric phantom rather than an implementation error: a random split places near-copies of the held-out views in the training set, and the grouped split therefore gives the conservative absolute errors on this object.",
            1,
        ),
        # op5: §5.2 trailing claim must not survive
        (
            "a sector-wise (grouped) split is reported in Section~\\ref{sec:testset} and Fig.~\\ref{fig:sectorsplit}, indicating that the random split does not significantly overestimate performance.",
            "a sector-wise (grouped) split is reported in Section~\\ref{sec:testset} and Fig.~\\ref{fig:sectorsplit}.",
            1,
        ),
        # op6: Fig sectorsplit caption
        (
            "the two differ by $-4.4\\%$ (0.4486 against 0.4288), so the random split does not significantly overestimate performance.",
            "at this checkpoint the grouped value is $4.4\\%$ lower (0.4288 against 0.4486); at the 100-epoch checkpoint the ordering reverses, the grouped split being $60.7\\%$ higher (Section~\\ref{sec:testset}), so the 30-epoch reduction is an underfitting artefact.",
            1,
        ),
    ],
    "zh": [
        # op1: 48 -> 46 (摘要、§5.6、结论)
        (
            "（$30^\\circ$ 扇区、48 个块）",
            "（$30^\\circ$ 扇区、46 个非空扇区×能量块）",
            3,
        ),
        # op2: tab:decision 行 1
        (
            "簇稳健区间不含零 \\\\",
            "簇稳健区间（46 个非空块）不含零 \\\\",
            1,
        ),
        # op3: §5.3 确认结论
        (
            "因此分块划分给出略低的误差，表明随机划分在 30-epoch checkpoint 下不会显著高估性能。上文报告的 100-epoch 随机划分值 0.3804 是另一个 checkpoint，不与 30-epoch 的分块比较合并。",
            "在 100-epoch checkpoint 下该排序反转：全 32 行池化上，分组划分达 0.617，对随机划分的 0.384（逐种子 0.591--0.654），分组/随机之比为 1.61（$+60.7\\%$）；成像行上，补全对线性插值之比从随机划分的 1.06 升到分组划分的 2.16，即一旦把逐像素相同的旋转视角从训练中排除，补全在进入重建的那一行上比线性插值差 2.16 倍。因此 30-epoch checkpoint 下的 4.4\\% 降低是欠拟合假象，而非随机划分无偏的证据；收敛后，随机划分高估了本模体上的绝对精度，因为留出视角与其 90\\dgr{} 旋转对应视角在蒙特卡罗噪声水平上逐像素相同、且分居划分两侧。",
            1,
        ),
        # op4: 删除失效的“不是数据泄漏”桥接句
        (
            "这不是数据泄漏：90\\dgr{} 对称性是网络可以从任意扇区学到的常量，而不是某个特定目标视角的副本，因此从训练中移除旋转重复项并未移除网络所依赖的信息。",
            "这种冗余是对称模体的性质而非实现错误：随机划分会把留出视角的近副本放进训练集，因此分组划分给出的才是本物体上保守的绝对误差。",
            1,
        ),
        # op5: §5.2 尾句同步
        (
            "报告了扇区分块（grouped）划分的结果，表明随机划分不会显著高估性能。",
            "报告了扇区分块（grouped）划分的结果。",
            1,
        ),
        # op6: 图 sectorsplit caption
        (
            "两者相差 $-4.4\\%$（0.4486 对 0.4288），因此随机划分不会显著高估性能。",
            "在该 checkpoint 下分组值低 $4.4\\%$（0.4288 对 0.4486）；100-epoch checkpoint 下排序反转，分组划分高 $60.7\\%$（§\\ref{sec:testset}），故 30-epoch 的降低是欠拟合假象。",
            1,
        ),
    ],
}


def normalize(text):
    return text.replace("\r\n", "\n").replace("\r", "\n")


def run(mode):
    for key, path in FILES.items():
        with open(path, "r", encoding="utf-8") as f:
            raw = f.read()
        text = normalize(raw)
        ops = OPS[key]
        for i, (old, new, cnt) in enumerate(ops, 1):
            n = text.count(old)
            if n != cnt:
                raise SystemExit(
                    f"[{key}] OP{i}: expected {cnt} match(es), found {n}.\n"
                    f"--- OLD ---\n{old}\n---"
                )
            if mode == "apply":
                text = text.replace(old, new)
        if mode == "apply":
            text = text.replace("\n", "\r\n")
            with open(path, "w", encoding="utf-8", newline="") as f:
                f.write(text)
            print(f"[{key}] applied {len(ops)} ops -> {path}")
        else:
            print(f"[{key}] check OK: {len(ops)} ops matched expected counts")


if __name__ == "__main__":
    mode = sys.argv[1].lstrip("-") if len(sys.argv) > 1 else "check"
    if mode not in ("check", "apply"):
        raise SystemExit("usage: python _r54d2_grouped.py --check | --apply")
    run(mode)
