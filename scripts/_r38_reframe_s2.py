#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""R38: close the six reframing gaps + 2 small fixes (text only; no numbers touched).

Usage:
  python scripts/_r38_reframe_s2.py --check
  python scripts/_r38_reframe_s2.py --apply

Gap 1  Section 1 paragraph 1 rewritten as high-Z MeV CT; inspection sentence moved to para 2
Gap 2  Section 4.1 testbed-rationale paragraph
Gap 3  New subsection "Applicability beyond nuclear fuel" between 6.4 and 6.5
Gap 4  Limitations item 1 rewritten
Gap 5  Conclusion final paragraph rewritten
Gap 6  Axis declaration at 5.2 (5.6 already carries the identical sentence from R36)
Fix 1  Abstract: two sentences, "primary view-level statistic", seed-level test clause
Fix 2  Not applicable: the manuscript already reads "thins"/"thinned" at both sites
"""
import io
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LATEX = os.path.join(ROOT, "docs", "manuscript", "latex")
EN = os.path.join(LATEX, "main.tex")
ZH = os.path.join(LATEX, "main_zh.tex")

EN_OLD_P1 = (
    r"Nuclear fuel assemblies are the core component of a light-water reactor. In-service inspection must "
    r"detect rod bowing, pellet-column gaps, cladding deformation and loose parts without dismantling the "
    r"assembly, leaving non-destructive volumetric imaging as the only practical option. X-ray computed "
    r"tomography (CT) reconstructs the cross-sectional distribution of the linear attenuation coefficient "
    r"from line-integral measurements, and has been applied to fuel-rod hardware, for instance Zircaloy "
    r"cladding weld zones~\cite{Fu2012ConeBeamCT}; optical machine vision, by contrast, remains confined to "
    r"surface features such as rod grooves and scratches~\cite{Guo2019DefectDetection}. A 3×3 array of "
    r"uranium-dioxide rods is, however, a dense high-Z object: photons in the conventional keV range are "
    r"strongly attenuated within the first rod, which places the array in the megavoltage "
    r"regime~\cite{Champley2022FusedCT,Firsching2012DualHighEnergy}, although full penetration is not "
    r"guaranteed even then~\cite{Champley2022FusedCT}."
)

EN_NEW_P1 = r"""High-energy photon computed tomography (MeV CT), employing 6--15~MeV bremsstrahlung or gamma
sources, has become a critical imaging modality for non-destructive evaluation of high-Z,
high-attenuation components in aerospace (single-crystal turbine blades), heavy manufacturing
(thick-walled pressure vessels, dense alloy castings), and nuclear engineering, where MeV CT has
already been applied to fuel-rod hardware such as Zircaloy cladding weld zones~\cite{Fu2012ConeBeamCT}
while optical machine vision remains confined to surface features such as rod grooves and
scratches~\cite{Guo2019DefectDetection}. Unlike diagnostic keV imaging, MeV photons penetrate thick
absorbers~\cite{Champley2022FusedCT,Firsching2012DualHighEnergy}, but the associated dose, beam-on
time, and detector dead-time impose a strong incentive toward sparse-view acquisition. A 3×3 array of
uranium-dioxide rods is an extreme testbed for this regime: with $Z = 92$ and attenuation coefficients
several times those of steel or titanium, it pushes photon starvation and angular undersampling to
their practical limits, while its four-fold symmetry provides a controlled baseline against which
learned priors can be measured. We use this testbed not because the application is nuclear-specific,
but because a boundary condition established under its extreme constraints is expected to hold, with
shifted numerical values, for less attenuating high-Z objects."""

EN_TESTBED = r"""The fuel-rod array is chosen as an extreme testbed rather than as an application target. Its linear
attenuation coefficient at 1.5~MeV is several times that of steel or titanium at the same energy, so
the photon-starvation regime that governs the operating-point boundary in other high-Z MeV CT
scenarios is reached at a lower dose and a wider angular span. The four-fold symmetry of the uniform
array provides a controlled baseline (Section~\ref{sec:testset}); the single-cavity defect case
(Section~\ref{sec:defect}) breaks that symmetry and tests whether the boundary survives. The numerical
boundary reported here is specific to this testbed; the sign and ordering of the effects are the
transferable claims."""

EN_APPLIC = r"""The operating-point map of Section~\ref{sec:axes} is defined by four measurable axes --- photon counts,
angular span, detector row, object symmetry --- and each axis maps onto a physical quantity in any
high-Z MeV CT scenario. The photon-count axis corresponds to the product of source flux, exposure time,
and object attenuation; the boundary at $f \approx 0.5$ in the dose sweep (Section~\ref{sec:dose})
translates to a wall-thickness threshold at a given source strength. The angular-span axis corresponds
to the mechanical or dose-driven limit on view count; the crossover at $s = 6$\dgr{} (Section~\ref{sec:span})
marks where a learned prior begins to outweigh the curvature bias of interpolation. The detector-row
axis corresponds to the transverse position of the region of interest; the Row-16 protocol is the
controlled case, and the zero-shot collapse on edge rows (Section~\ref{sec:row16}) identifies where a
learned prior should not be deployed. The symmetry axis corresponds to the periodicity of the object;
the four-fold floor of the uniform array (Section~\ref{sec:testset}) sets a lower bound that the defect
case (Section~\ref{sec:defect}) removes. We do not claim that the numerical boundary transfers unchanged;
we claim that the conditional law --- CNN completion pays off where counting noise dominates --- is the
transferable result, and that the axes along which it should be re-measured in a new scenario are these
four."""

EN_RULES = [
    # ---- Gap 1: paragraph 1 ----
    ("G1 intro paragraph 1", EN_OLD_P1, EN_NEW_P1),
    ("G1 intro paragraph 2 opener",
     r"Acquiring a complete data set is costly: either the assembly",
     r"Within this regime, in-service inspection of nuclear fuel assemblies illustrates the constraint at its "
     r"sharpest: rod bowing, pellet-column gaps, cladding deformation and loose parts must be detected "
     r"without dismantling the assembly. Acquiring a complete data set is costly: either the assembly"),

    # ---- Gap 2: 4.1 testbed rationale ----
    ("G2 sec:sim rationale",
     r"\label{sec:sim}" + "\n\n" + r"Data were generated with Geant4 11.0.3",
     r"\label{sec:sim}" + "\n\n" + EN_TESTBED + "\n\n" + r"Data were generated with Geant4 11.0.3"),

    # ---- Gap 3: new subsection before 6.5 ----
    ("G3 applicability subsection",
     r"\subsection{Defect detection as a negative control}",
     r"\subsection{Applicability beyond nuclear fuel}" + "\n" + r"\label{sec:applicability}" + "\n\n"
     + EN_APPLIC + "\n\n" + r"\subsection{Defect detection as a negative control}"),

    # ---- Gap 4: limitations item 1 ----
    ("G4 limits item 1",
     r"First, all data are simulated and the object is a single, strictly four-fold symmetric $3 \times 3$ array: scatter and detector response are those of the Geant4 model, and no measured projection has been used.",
     r"First, all data are simulated and the object is a single four-fold symmetric $3\times3$ array, chosen as an extreme testbed for high-Z MeV CT rather than as an application target; scatter and detector response are those of the Geant4 model, and no measured projection has been used. The transferability of the boundary to other high-Z objects is argued from measurable physical quantities (Section~\ref{sec:applicability}) but not experimentally verified here."),

    # ---- Gap 5: conclusion final paragraph ----
    ("G5 conclusion final",
     r"and the four-fold symmetry of the phantom sets a floor that prevents extrapolation to asymmetric objects. Extending the formulation to a continuous target angle, so that an arbitrary intermediate view can be synthesised under a Radon-consistency constraint, is the natural next step (Section~\ref{sec:future}).",
     r"and the four-fold symmetry of the testbed provides a controlled floor from which the boundary can be re-measured on less symmetric objects; the conditional law identified here is argued to transfer through the four axes of Section~\ref{sec:axes}, and that transfer is the natural next step. Extending the formulation to a continuous target angle, so that an arbitrary intermediate view can be synthesised under a Radon-consistency constraint, is a further extension (Section~\ref{sec:future})."),

    # ---- Gap 6: 5.2 axis declaration ----
    ("G6 5.2 axis",
     r"\label{sec:energy}" + "\n\n" + r"At the retained checkpoint of the 100-epoch schedule",
     r"\label{sec:energy}" + "\n\n"
     + r"This subsection sweeps the flux/energy axis; span is fixed at 2\dgr{} and the detector row is the full 32-row pool for the validation metric. "
     + r"At the retained checkpoint of the 100-epoch schedule"),

    # ---- Fix 1: abstract statistics ----
    ("F1 abstract statistics",
     r"resolved from linear interpolation across seven training seeds (5 wins, 2 losses, p=0.23); under the" + "\n"
     + r"equal-weight paired analysis, which is the primary statistic of this paper, the completion trails" + "\n"
     + r"interpolation by $2.05\%$ with a $95\%$ confidence interval that excludes zero. The completion's",
     r"resolved from linear interpolation across seven training seeds (5 wins, 2 losses, p=0.23). Under the" + "\n"
     + r"equal-weight paired analysis over held-out views, which is the primary view-level statistic, the" + "\n"
     + r"completion trails interpolation by $2.05\%$ with a $95\%$ CI that excludes zero; the seed-level test" + "\n"
     + r"treats the method-level difference as unresolved. The completion's"),
]

ZH_OLD_P1 = (
    "核燃料组件是轻水堆的核心部件。在役检查必须在不拆卸组件的情况下检出棒弯曲、芯块柱间隙、包壳变形与松动部件，"
    "这使得非破坏性体成像成为唯一可行的方案。X 射线计算机断层成像（CT）根据线积分测量重建线衰减系数的截面分布，"
    "并已应用于燃料棒硬件，例如锆合金包壳焊缝区~\\cite{Fu2012ConeBeamCT}；相比之下，光学机器视觉仍局限于棒沟槽与"
    "划痕等表面特征~\\cite{Guo2019DefectDetection}。然而，3×3 的二氧化铀棒阵是致密的高原子序数（high-Z）物体："
    "常规 keV 能段的光子在第一根棒内即被强烈衰减，从而将该阵列推入兆伏（MeV）区间~\\cite{Champley2022FusedCT,Firsching2012DualHighEnergy}，"
    "尽管即便如此也不能保证完全穿透~\\cite{Champley2022FusedCT}。"
)

ZH_NEW_P1 = (
    "高能光子计算机断层成像（MeV CT）以 6--15~MeV 轫致辐射或伽马源为射线源，已成为航空航天（单晶涡轮叶片）、"
    "重制造（厚壁压力容器、致密合金铸件）与核工程中高 Z、高衰减部件无损检测的关键成像手段；在核工程中，"
    "MeV CT 已应用于燃料棒硬件，例如锆合金包壳焊缝区~\\cite{Fu2012ConeBeamCT}，而光学机器视觉仍局限于棒沟槽与"
    "划痕等表面特征~\\cite{Guo2019DefectDetection}。与诊断级 keV 成像不同，MeV 光子能穿透厚吸收体~\\cite{Champley2022FusedCT,Firsching2012DualHighEnergy}，"
    "但随之而来的剂量、出束时间与探测器死时间，对稀疏视角采集构成强烈激励。3×3 的二氧化铀棒阵是这一区间的极端"
    "测试床：其 $Z = 92$，衰减系数为钢或钛的数倍，从而把光子饥饿与角度欠采样推到实际极限；同时其四重对称性提供了"
    "受控基准，可用以度量学习先验。我们采用这一测试床，并非因为应用本身属于核领域，而是因为在极端约束下确立的"
    "边界条件，预期会以平移后的数值适用于衰减更弱的高 Z 物体。"
)

ZH_TESTBED = (
    "把燃料棒阵选作极端测试床，而非应用目标。它在 1.5~MeV 的线性衰减系数是同能量下钢或钛的数倍，因此在更低的"
    "剂量与更宽的角跨度下即进入光子饥饿区间——正是该区间决定了其他高 Z MeV CT 场景中的工作点边界。均匀阵列的"
    "四重对称性提供了受控基准（§\\ref{sec:testset}）；单空腔缺陷情形（§\\ref{sec:defect}）打破该对称性，用以检验"
    "边界是否依然成立。本文报告的数值边界是本测试床特有的；可迁移的主张是各效应的符号与排序。"
)

ZH_APPLIC = (
    "§\\ref{sec:axes} 的工作点地图由四个可测量轴界定——光子计数、角跨度、探测器行、物体对称性——在任意高 Z MeV CT "
    "场景中，每个轴都对应一个物理量。光子计数轴对应源通量、曝光时间与物体衰减的乘积；剂量扫描中 $f \\approx 0.5$ "
    "处的边界（§\\ref{sec:dose}）在给定源强下转换为一个壁厚阈值。角跨度轴对应视角数在机械或剂量上的上限；"
    "$s = 6$\\dgr{} 处的交叉（§\\ref{sec:span}）标志着学习先验开始压过插值的曲率偏差。探测器行轴对应感兴趣区的"
    "横向位置；第 16 行协议是受控情形，而边缘行上的零样本崩塌（§\\ref{sec:row16}）指明了不应部署学习先验之处。"
    "对称性轴对应物体的周期性；均匀阵列的四重地板（§\\ref{sec:testset}）给出一个下界，而缺陷情形（§\\ref{sec:defect}）"
    "移除了该下界。我们不主张数值边界原样迁移；我们主张可迁移的结果是那条条件律——CNN 补全在计数噪声主导处获益——"
    "而在新场景中应当重新测量它的轴，就是这四条。"
)

ZH_RULES = [
    ("ZH G1 intro paragraph 1", ZH_OLD_P1, ZH_NEW_P1),
    ("ZH G1 intro paragraph 2 opener",
     "获取完整数据集成本高昂：组件或源–探测器扫描架必须旋转",
     "在这一框架下，核燃料组件的在役检查把该约束体现得最为尖锐：必须在不解体组件的前提下检出棒弯曲、芯块柱间隙、"
     "包壳变形与松动部件。获取完整数据集成本高昂：组件或源–探测器扫描架必须旋转"),

    ("ZH G2 sec:sim rationale",
     "\\label{sec:sim}" + "\n\n" + "数据由 Geant4 11.0.3 生成",
     "\\label{sec:sim}" + "\n\n" + ZH_TESTBED + "\n\n" + "数据由 Geant4 11.0.3 生成"),

    ("ZH G3 applicability subsection",
     "\\subsection{作为阴性对照的缺陷检测}",
     "\\subsection{核燃料之外的可迁移性}" + "\n" + "\\label{sec:applicability}" + "\n\n"
     + ZH_APPLIC + "\n\n" + "\\subsection{作为阴性对照的缺陷检测}"),

    ("ZH G4 limits item 1",
     "第一，全部数据为仿真，且物体只有一个、严格四重对称的 3×3 阵列：散射与探测器响应均来自 Geant4 模型，未使用任何实测投影。",
     "第一，全部数据为仿真，且物体只有一个四重对称的 3×3 阵列——它被选作高 Z MeV CT 的极端测试床，而非应用目标；"
     "散射与探测器响应均来自 Geant4 模型，未使用任何实测投影。边界向其他高 Z 物体的可迁移性由可测量物理量论证"
     "（§\\ref{sec:applicability}），但本文未作实验验证。"),

    ("ZH G5 conclusion final",
     "且模体的四重对称性设下了一个阻止向非对称物体外推的地板。把该表述推广到连续目标角度，使任意中间视角能在 Radon 一致性约束下被合成，是自然的下一步（§\\ref{sec:future}）。",
     "且测试床的四重对称性提供了一个受控地板，边界可据此在对称性更弱的物体上重新测量；本文论证的条件律经由 "
     "§\\ref{sec:axes} 的四条轴迁移，而这一迁移正是自然的下一步。把该表述推广到连续目标角度，使任意中间视角能在 "
     "Radon 一致性约束下被合成，是进一步的延伸（§\\ref{sec:future}）。"),

    ("ZH G6 5.2 axis",
     "\\label{sec:energy}" + "\n\n" + "在 100-epoch 计划所保留的 checkpoint 上",
     "\\label{sec:energy}" + "\n\n"
     + "本小节扫描通量／能量轴；跨度固定为 2\\dgr{}，探测器行为验证指标所用的全部 32 行池化。"
     + "在 100-epoch 计划所保留的 checkpoint 上"),

    ("ZH F1 abstract statistics",
     "（5 胜 2 负，p=0.23）；但在等权配对分析——本文的首要统计口径——下，补全落后插值 2.05\\%，其 95\\% 置信区间不含零。",
     "（5 胜 2 负，p=0.23）。但在留出视角上的等权配对分析——本文的首要视角层级统计口径——下，补全落后插值 2.05\\%，"
     "其 95\\% 置信区间不含零；种子层级检验把方法层级的差异视为未决。"),
]


def detect_eol(text):
    return "\r\n" if "\r\n" in text else "\n"


def run(path, rules, apply_changes):
    with io.open(path, "r", encoding="utf-8", newline="") as fh:
        text = fh.read()
    eol = detect_eol(text)
    ok = True
    new_text = text
    for name, old, new in rules:
        o = old.replace("\n", eol)
        n = new.replace("\n", eol)
        c = new_text.count(o)
        if c != 1:
            print("  [FAIL] %-36s count=%d" % (name, c))
            ok = False
            continue
        new_text = new_text.replace(o, n, 1)
        print("  [ok]   %-36s" % name)
    if apply_changes and ok:
        with io.open(path, "w", encoding="utf-8", newline="") as fh:
            fh.write(new_text)
        print("  -> written: %s" % os.path.basename(path))
    return ok


def main():
    apply_changes = "--apply" in sys.argv
    print("=== R38 %s : EN ===" % ("APPLY" if apply_changes else "CHECK"))
    a = run(EN, EN_RULES, apply_changes)
    print("=== R38 %s : ZH ===" % ("APPLY" if apply_changes else "CHECK"))
    b = run(ZH, ZH_RULES, apply_changes)
    if not (a and b):
        print("RESULT: FAILED (nothing written)")
        sys.exit(1)
    print("RESULT: OK")


if __name__ == "__main__":
    main()
