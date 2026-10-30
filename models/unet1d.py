#!/usr/bin/env python
# -*- coding: utf-8 -*-
r"""
models/unet1d.py —— 真正的单行（1D）U-Net：只在 Row 16 的 1200 个探测器像素上做角度插值

==================== 为什么需要它 ====================

C 组消融（1D 单行）与 B 组（伪单行）的区别是**本质性**的：
  B 组：网络仍是 32x1200 的 2D U-Net，只把非 Row 16 行的输入抹成零计数填充值、
        mask 收缩到 Row 16 → 其余 31 行既无输入信号也无监督，退化成常数输出（实测 0.708/1.378）。
        这不是「单行训练失败」，而是「用 2D 网络假装单行」的必然结果。
  C 组（本文件）：输入张量里**根本没有行轴**，每层卷积只在长度维上滑动，
        参数量约为 2D 的 1/3，因此「1 行」不是被抹出来的，而是架构本身只认 1 行。

==================== 结构（与 2D U-Net 逐级同构，单变量对照） ====================

    阶段         L     ch    说明
    --------  ----  ----  --------------------------------------------
    input     1200     5   3 张归一化 -log 投影（同一行）+ sin/cos
    enc0      1200    32   DoubleConv1d(5   -> 32)          -> skip0
    enc1       600    64   maxpool(2) + DoubleConv1d(32  -> 64)   -> skip1
    enc2       300   128   maxpool(2) + DoubleConv1d(64  -> 128)  -> skip2
    enc3       150   256   maxpool(2) + DoubleConv1d(128 -> 256)  -> skip3
    bottom      75   256   第 4 次 maxpool，本层不接卷积
    dec3       150   128   ConvT1d(256->128) + cat(skip3) + Conv3(384->128)
    dec2       300    64   ConvT1d(128->64)  + cat(skip2) + Conv3(192->64)
    dec1       600    32   ConvT1d(64->32)   + cat(skip1) + Conv3(96->32)
    dec0      1200    32   ConvT1d(32->32)   + cat(skip0) + Conv3(64->32)
    output    1200     1   1x1 卷积

对照口径（**刻意与 models/unet.py 保持一致，只减少维度**）：
1. 下采样次数同为 4；每级通道数同为 (32,64,128,256)；bias=False，偏置由 BatchNorm 承担。
2. decoder 每级同 2D：ConvTranspose + BN + ReLU + concat(skip) + **单层** Conv3 + BN + ReLU。
3. 末级 75 同样只 pool、不卷积（与 2D 的 2x75 层对应）。
4. 4 次下采样要求 L 能被 2^4 整除且末级 == 75；forward 内逐级校验，偏差即抛 ModelSpecError。
5. 因此唯一的自变量是「有无行轴」：2D 的 Conv3x3 有 9 个权重/通道对，1D 的 Conv3 只有 3 个，
   参数量约 1/3（实测 681,345 vs 1,950,017）。

参数量（实测）：681,345 ≈ 0.681 M，为 2D 的 34.9%。

自检：
    python -m models.unet1d --inspect
"""

from __future__ import annotations

import argparse
from typing import Sequence

import torch
import torch.nn as nn

# ------------------------------------------------------------------ 结构常量
IN_CHANNELS = 5
OUT_CHANNELS = 1
LENGTH = 1200
ENC_CHANNELS: tuple[int, int, int, int] = (32, 64, 128, 256)
N_DOWNSAMPLE = 4

#: 4 次下采样后各级期望长度。forward 内逐层比对，任何偏差立即抛错。
EXPECTED_L: tuple[int, ...] = (1200, 600, 300, 150, 75)


class ModelSpecError(RuntimeError):
    """结构或张量尺寸违反已裁决口径时抛出（fail fast，不做静默兜底）。"""


def _len(t: torch.Tensor) -> int:
    return int(t.shape[-1])


def _expect_len(t: torch.Tensor, expected: int, tag: str) -> torch.Tensor:
    got = _len(t)
    if got != expected:
        raise ModelSpecError(f"{tag} 长度 {got} != 期望 {expected}")
    return t


# ------------------------------------------------------------------ 基础块
class DoubleConv1d(nn.Module):
    """(Conv3 -> BN -> ReLU) x 2。bias=False，偏置由 BatchNorm 承担。"""

    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv1d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm1d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv1d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm1d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class DecoderBlock1d(nn.Module):
    """ConvTranspose1d 上采样 -> concat(skip) -> 单层 Conv3。与 2D 的 DecoderBlock 对应。"""

    def __init__(self, in_ch: int, out_ch: int, skip_ch: int) -> None:
        super().__init__()
        self.in_ch = in_ch
        self.out_ch = out_ch
        self.skip_ch = skip_ch
        self.up = nn.ConvTranspose1d(in_ch, out_ch, kernel_size=2, stride=2, bias=False)
        self.bn_up = nn.BatchNorm1d(out_ch)
        self.act_up = nn.ReLU(inplace=True)
        self.conv = nn.Conv1d(out_ch + skip_ch, out_ch, kernel_size=3, padding=1, bias=False)
        self.bn = nn.BatchNorm1d(out_ch)
        self.act = nn.ReLU(inplace=True)
        self._dbg: dict[str, int] = {}

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        u = self.act_up(self.bn_up(self.up(x)))
        if _len(u) != _len(skip):
            raise ModelSpecError(
                f"{type(self).__name__} 上采样长度 {_len(u)} 与跳跃连接 {_len(skip)} 不一致，无法 concat"
            )
        if skip.shape[1] != self.skip_ch:
            raise ModelSpecError(f"skip 通道 {skip.shape[1]} != 期望 {self.skip_ch}")
        if u.shape[1] != self.out_ch:
            raise ModelSpecError(f"上采样通道 {u.shape[1]} != 期望 {self.out_ch}")
        self._dbg = {"up_in": _len(x), "up_out": _len(u), "skip": _len(skip)}
        return self.act(self.bn(self.conv(torch.cat([u, skip], dim=1))))


# ------------------------------------------------------------------ 主模型
class UNet1D(nn.Module):
    """5 -> 1 通道，长度 1200 的单行 U-Net（无行轴）。"""

    def __init__(
        self,
        in_channels: int = IN_CHANNELS,
        out_channels: int = OUT_CHANNELS,
        channels: Sequence[int] = ENC_CHANNELS,
    ) -> None:
        super().__init__()
        if len(channels) != N_DOWNSAMPLE:
            raise ModelSpecError(
                f"enc 通道必须是 {N_DOWNSAMPLE} 段，收到 {len(channels)} 段: {tuple(channels)}"
            )
        c0, c1, c2, c3 = (int(c) for c in channels)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.channels = (c0, c1, c2, c3)

        self.enc0 = DoubleConv1d(in_channels, c0)
        self.enc1 = DoubleConv1d(c0, c1)
        self.enc2 = DoubleConv1d(c1, c2)
        self.enc3 = DoubleConv1d(c2, c3)
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)

        self.dec3 = DecoderBlock1d(c3, c2, c3)   # 75  -> 150, skip3=c3
        self.dec2 = DecoderBlock1d(c2, c1, c2)   # 150 -> 300, skip2=c2
        self.dec1 = DecoderBlock1d(c1, c0, c1)   # 300 -> 600, skip1=c1
        self.dec0 = DecoderBlock1d(c0, c0, c0)   # 600 -> 1200, skip0=c0

        self.head = nn.Conv1d(c0, out_channels, kernel_size=1)

        self._check_pool_divides()

    @staticmethod
    def _check_pool_divides() -> None:
        """4 次 2 倍 pool 必须整除，且逐级等于 EXPECTED_L。"""
        n = LENGTH
        for i in range(N_DOWNSAMPLE):
            if n % 2:
                raise ModelSpecError(f"第 {i + 1} 次 pool 除不尽：{n}")
            n //= 2
            if n != EXPECTED_L[i + 1]:
                raise ModelSpecError(
                    f"第 {i + 1} 次 pool 得到 {n}，与 EXPECTED_L[{i + 1}]={EXPECTED_L[i + 1]} 不符"
                )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 3:
            # 显式拒绝 4 维输入：这个模型不允许"用 2D 张量假装单行"
            raise ModelSpecError(
                f"输入必须是 3 维 (B,C,L)，收到 {tuple(x.shape)}；"
                f"若传入了 (B,C,H,W)，请先按行切片再喂入"
            )
        if x.shape[1] != self.in_channels:
            raise ModelSpecError(f"输入通道 {x.shape[1]} != 期望 {self.in_channels}")
        _expect_len(x, EXPECTED_L[0], "input")

        d0 = _expect_len(self.enc0(x), EXPECTED_L[0], "enc0")
        d1 = _expect_len(self.enc1(self.pool(d0)), EXPECTED_L[1], "enc1")
        d2 = _expect_len(self.enc2(self.pool(d1)), EXPECTED_L[2], "enc2")
        d3 = _expect_len(self.enc3(self.pool(d2)), EXPECTED_L[3], "enc3")
        b = _expect_len(self.pool(d3), EXPECTED_L[4], "bottom")

        u3 = _expect_len(self.dec3(b, d3), EXPECTED_L[3], "dec3")
        u2 = _expect_len(self.dec2(u3, d2), EXPECTED_L[2], "dec2")
        u1 = _expect_len(self.dec1(u2, d1), EXPECTED_L[1], "dec1")
        u0 = _expect_len(self.dec0(u1, d0), EXPECTED_L[0], "dec0")

        out = self.head(u0)
        _expect_len(out, EXPECTED_L[0], "head")
        if out.shape[1] != self.out_channels:
            raise ModelSpecError(f"输出通道 {out.shape[1]} != 期望 {self.out_channels}")
        return out

    def n_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())


# ================================================================== 自检
def inspect(device: str = "cpu", batch: int = 1) -> int:
    model = UNet1D().to(device).eval()
    x = torch.zeros(batch, IN_CHANNELS, LENGTH, device=device)
    print("=" * 90)
    print(f"UNet1D 结构自检   device={device}  batch={batch}  input={tuple(x.shape)}")
    print(f"enc channels={ENC_CHANNELS}  下采样次数={N_DOWNSAMPLE}  各级长度={EXPECTED_L}")
    print("=" * 90)

    with torch.no_grad():
        d0 = model.enc0(x)
        d1 = model.enc1(model.pool(d0))
        d2 = model.enc2(model.pool(d1))
        d3 = model.enc3(model.pool(d2))
        b = model.pool(d3)
        u3 = model.dec3(b, d3)
        u2 = model.dec2(u3, d2)
        u1 = model.dec1(u2, d1)
        u0 = model.dec0(u1, d0)
        y = model.head(u0)

    for tag, t in (("input", x), ("enc0", d0), ("enc1", d1), ("enc2", d2), ("enc3", d3),
                   ("bottom", b), ("dec3", u3), ("dec2", u2), ("dec1", u1), ("dec0", u0),
                   ("head", y)):
        print(f"  {tag:<8}{str(tuple(t.shape)):>22}")

    if _len(y) != _len(x):
        print(f"\n[FAIL] 输出长度 {_len(y)} != 输入 {_len(x)}")
        return 1

    total = model.n_parameters()
    print("\n[参数量明细]")
    for name, child in model.named_children():
        n = sum(p.numel() for p in child.parameters())
        bn = sum(p.numel() for m in child.modules() if isinstance(m, nn.BatchNorm1d)
                 for p in m.parameters())
        print(f"  {name:<8}{n:>10,}    (conv {n - bn:>9,} + bn {bn:>6,})")
    print(f"  {'-' * 40}\n  {'TOTAL':<8}{total:>10,}  = {total / 1e6:.3f} M")

    try:
        model.dec3(torch.zeros(1, 256, 75, device=device),
                   torch.zeros(1, 256, 149, device=device))
        print("\n[FAIL] 长度不符却没有报错，护栏失效")
        return 1
    except ModelSpecError as exc:
        print(f"\n[OK] 长度不符正确抛错: {str(exc)[:70]}")

    try:
        model(torch.zeros(1, 5, 32, 1200, device=device))
        print("[FAIL] 4 维输入没有被拒绝")
        return 1
    except ModelSpecError as exc:
        print(f"[OK] 4 维输入正确拒绝: {str(exc)[:70]}")

    print("\n[OK] 逐级长度校验通过，输出与输入等长，护栏生效")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description="UNet1D 结构自检（不训练）")
    ap.add_argument("--inspect", action="store_true")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--batch", type=int, default=1)
    args = ap.parse_args()
    if not args.inspect:
        ap.print_help()
        return 2
    return inspect(device=args.device, batch=args.batch)


if __name__ == "__main__":
    raise SystemExit(main())
