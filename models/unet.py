#!/usr/bin/env python
# -*- coding: utf-8 -*-
r"""
models/unet.py —— 稀疏角 CT 补全的 U-Net（新电脑从零重建，不参考旧机器任何代码）

==================== 已裁决结构（C1，硬约束） ====================

    阶段      H     W     ch    说明
    ----    ----  ----  ----  --------------------------------------------
    input     32  1200     5   3 张归一化 -log 投影 + sin(theta)/cos(theta)
    enc0      32  1200    32   DoubleConv(5   -> 32)          -> skip0
    enc1      16   600    64   maxpool(2) + DoubleConv(32  -> 64)   -> skip1
    enc2       8   300   128   maxpool(2) + DoubleConv(64  -> 128)  -> skip2
    enc3       4   150   256   maxpool(2) + DoubleConv(128 -> 256)  -> skip3
    bottom     2    75   256   第 4 次 maxpool(2)，本层不接卷积
    dec3       4   150   128   ConvT(256->128) + cat(skip3) + Conv3x3(384->128)
    dec2       8   300    64   ConvT(128->64)  + cat(skip2) + Conv3x3(192->64)
    dec1      16   600    32   ConvT(64->32)   + cat(skip1) + Conv3x3(96->32)
    dec0      32  1200    32   ConvT(32->32)   + cat(skip0) + Conv3x3(64->32)
    output    32  1200     1   1x1 卷积

设计口径（均来自裁决，不得擅改）：

1. 只做 4 次下采样。不做 5 次（高退到 1，丧失空间感受野），不做 3 次
   （高 4，要靠加通道凑参数量，结构偏离参考）。
2. 每个上采样层输出尺寸必须与对应跳跃连接**严格一致**，forward 内逐层校验。
   用显式 `raise ModelSpecError` 而不是 `assert`，避免 `python -O` 下断言被剥离。
3. 最后一层是 1x1 卷积，输出 1 通道。
4. decoder 每级 = ConvTranspose2d + BN + ReLU + concat(skip) + **单层** Conv3x3
   + BN + ReLU，不是 DoubleConv。这是把参数量压在 1.95M 附近的关键；
   若 decoder 也用 DoubleConv，参数量会升到约 3.8M。
5. 2x75 层只有 maxpool、没有卷积。若在此处加独立 bottleneck：
   双 3x3 会 +1.18M、单 3x3 会 +0.59M、1x1 会 +0.066M，三种都会改变参数量口径，
   需另行裁决；本文件不擅自加层。

参数量（实测，bias=False + BatchNorm 口径）：1,950,017 ≈ 1.950M
逐模块明细见 `python -m models.unet --inspect` 的输出。

==================================================================
"""

from __future__ import annotations

import argparse
from typing import Sequence

import torch
import torch.nn as nn

# ------------------------------------------------------------------ 结构常量
IN_CHANNELS = 5
OUT_CHANNELS = 1
BASE = 32
ENC_CHANNELS: tuple[int, int, int, int] = (32, 64, 128, 256)
N_DOWNSAMPLE = 4

IMAGE_H, IMAGE_W = 32, 1200

#: 4 次下采样后各级期望的 (H, W)。forward 内逐层比对，任何偏差立即抛错。
EXPECTED_HW: tuple[tuple[int, int], ...] = (
    (32, 1200),   # input / enc0
    (16, 600),    # enc1
    (8, 300),     # enc2
    (4, 150),     # enc3
    (2, 75),      # bottom（第 4 次 pool 的输出，无卷积）
)

PARAM_RANGE = (1_500_000, 2_500_000)


class ModelSpecError(RuntimeError):
    """结构或张量尺寸违反已裁决口径时抛出（fail fast，不做静默兜底）。"""


def _hw(t: torch.Tensor) -> tuple[int, int]:
    return int(t.shape[-2]), int(t.shape[-1])


def _expect_hw(t: torch.Tensor, expected: tuple[int, int], tag: str) -> torch.Tensor:
    got = _hw(t)
    if got != expected:
        raise ModelSpecError(f"{tag} 空间尺寸 {got} != 期望 {expected}")
    return t


def _expect_same(t: torch.Tensor, ref: torch.Tensor, tag: str) -> torch.Tensor:
    """上采样输出必须与跳跃连接逐像素对齐（H 和 W 都要相等）。"""
    if _hw(t) != _hw(ref):
        raise ModelSpecError(
            f"{tag} 上采样输出 {_hw(t)} 与跳跃连接 {_hw(ref)} 不一致，无法 concat"
        )
    return t


# ------------------------------------------------------------------ 基础块
class DoubleConv(nn.Module):
    """(Conv3x3 -> BN -> ReLU) x 2。bias=False，偏置由 BatchNorm 承担。"""

    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class DecoderBlock(nn.Module):
    """ConvTranspose2d 上采样 -> concat(skip) -> 单层 Conv3x3。

    上采样后先与 skip 做尺寸校验，再拼接；拼接通道数 = out_ch + skip_ch。
    """

    def __init__(self, in_ch: int, out_ch: int, skip_ch: int) -> None:
        super().__init__()
        self.in_ch = in_ch
        self.out_ch = out_ch
        self.skip_ch = skip_ch
        self.up = nn.ConvTranspose2d(
            in_ch, out_ch, kernel_size=2, stride=2, bias=False
        )
        self.bn_up = nn.BatchNorm2d(out_ch)
        self.act_up = nn.ReLU(inplace=True)
        self.conv = nn.Conv2d(out_ch + skip_ch, out_ch, kernel_size=3, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)
        self.act = nn.ReLU(inplace=True)
        # 供 --inspect 读取的实际张量尺寸，不参与计算图
        self._dbg: dict[str, tuple[int, int]] = {}

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        u = self.act_up(self.bn_up(self.up(x)))
        _expect_same(u, skip, f"{type(self).__name__}.up")

        if skip.shape[1] != self.skip_ch:
            raise ModelSpecError(
                f"skip 通道 {skip.shape[1]} != 期望 {self.skip_ch}"
            )
        if u.shape[1] != self.out_ch:
            raise ModelSpecError(f"上采样通道 {u.shape[1]} != 期望 {self.out_ch}")

        self._dbg = {
            "up_in": _hw(x),
            "up_out": _hw(u),
            "skip": _hw(skip),
            "cat": _hw(u),
        }
        c = torch.cat([u, skip], dim=1)
        return self.act(self.bn(self.conv(c)))


# ------------------------------------------------------------------ 主模型
class UNet(nn.Module):
    """5 -> 1 通道，32x1200 输入，4 次下采样的 U-Net。"""

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

        # ---- encoder：4 级，每级 DoubleConv；级间用 maxpool(2) 下采样
        self.enc0 = DoubleConv(in_channels, c0)
        self.enc1 = DoubleConv(c0, c1)
        self.enc2 = DoubleConv(c1, c2)
        self.enc3 = DoubleConv(c2, c3)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # ---- decoder：4 级，从 2x75 逐级上采样回 32x1200
        self.dec3 = DecoderBlock(c3, c2, c3)   # 2x75  -> 4x150,  skip3=c3
        self.dec2 = DecoderBlock(c2, c1, c2)   # 4x150 -> 8x300,  skip2=c2
        self.dec1 = DecoderBlock(c1, c0, c1)   # 8x300 -> 16x600, skip1=c1
        self.dec0 = DecoderBlock(c0, c0, c0)   # 16x600 -> 32x1200, skip0=c0

        # ---- 输出头：1x1 卷积 -> 1 通道
        self.head = nn.Conv2d(c0, out_channels, kernel_size=1)

        self._check_pool_divides()

    # -------------------------------------------------------------- 结构自检
    @staticmethod
    def _check_pool_divides() -> None:
        """4 次 2x2 maxpool 必须能整除，且结果等于 EXPECTED_HW 的末级。"""
        h, w = IMAGE_H, IMAGE_W
        for i in range(N_DOWNSAMPLE):
            if h % 2 or w % 2:
                raise ModelSpecError(f"第 {i + 1} 次 pool 除不尽：{h}x{w}")
            h //= 2
            w //= 2
            if (h, w) != EXPECTED_HW[i + 1]:
                raise ModelSpecError(
                    f"第 {i + 1} 次 pool 得到 {h}x{w}，与 EXPECTED_HW[{i + 1}]="
                    f"{EXPECTED_HW[i + 1]} 不符"
                )

    # ------------------------------------------------------------------ 前向
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 4:
            raise ModelSpecError(f"输入必须是 4 维 (B,C,H,W)，收到 {tuple(x.shape)}")
        if x.shape[1] != self.in_channels:
            raise ModelSpecError(
                f"输入通道 {x.shape[1]} != 期望 {self.in_channels}"
            )
        _expect_hw(x, EXPECTED_HW[0], "input")

        # ---------------- encoder ----------------
        d0 = _expect_hw(self.enc0(x), EXPECTED_HW[0], "enc0")
        d1 = _expect_hw(self.enc1(self.pool(d0)), EXPECTED_HW[1], "enc1")
        d2 = _expect_hw(self.enc2(self.pool(d1)), EXPECTED_HW[2], "enc2")
        d3 = _expect_hw(self.enc3(self.pool(d2)), EXPECTED_HW[3], "enc3")
        b = _expect_hw(self.pool(d3), EXPECTED_HW[4], "bottom")

        # ---------------- decoder ----------------
        u3 = _expect_hw(self.dec3(b, d3), EXPECTED_HW[3], "dec3")
        u2 = _expect_hw(self.dec2(u3, d2), EXPECTED_HW[2], "dec2")
        u1 = _expect_hw(self.dec1(u2, d1), EXPECTED_HW[1], "dec1")
        u0 = _expect_hw(self.dec0(u1, d0), EXPECTED_HW[0], "dec0")

        # ---------------- head ----------------
        out = self.head(u0)
        _expect_hw(out, EXPECTED_HW[0], "head")
        if out.shape[1] != self.out_channels:
            raise ModelSpecError(
                f"输出通道 {out.shape[1]} != 期望 {self.out_channels}"
            )
        return out

    # -------------------------------------------------------------- 参数量
    def n_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def n_trainable_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ================================================================== 自检
def _fmt(hw: tuple[int, int] | tuple[int, ...]) -> str:
    return "x".join(str(int(v)) for v in hw)


def inspect(device: str = "cpu", batch: int = 1) -> int:
    """打印逐层输入/输出 shape 与参数量明细（CPU 前向，不做训练）。

    这是模型自身的结构自检，不是训练 smoke：不加载数据、不算 loss、不反向。
    """
    model = UNet().to(device)
    model.eval()

    x = torch.zeros(batch, IN_CHANNELS, IMAGE_H, IMAGE_W, device=device)
    print("=" * 96)
    print(f"UNet 结构自检   device={device}  batch={batch}  input={_fmt(tuple(x.shape))}")
    print(f"enc channels={ENC_CHANNELS}  下采样次数={N_DOWNSAMPLE}")
    print("=" * 96)

    with torch.no_grad():
        enc_out = {}
        d0 = model.enc0(x)
        enc_out["enc0"] = d0
        d1 = model.enc1(model.pool(d0))
        enc_out["enc1"] = d1
        d2 = model.enc2(model.pool(d1))
        enc_out["enc2"] = d2
        d3 = model.enc3(model.pool(d2))
        enc_out["enc3"] = d3
        b = model.pool(d3)
        u3 = model.dec3(b, d3)
        u2 = model.dec2(u3, d2)
        u1 = model.dec1(u2, d1)
        u0 = model.dec0(u1, d0)
        y = model.head(u0)

    def line(tag: str, t: torch.Tensor, note: str = "") -> None:
        print(f"  {tag:<10}{_fmt(tuple(t.shape)):>26}   {note}")

    print("\n[编码路径]")
    line("input", x, "3 投影 + sin + cos")
    line("enc0", enc_out["enc0"], "DoubleConv(5->32)  -> skip0")
    line("enc1", enc_out["enc1"], "pool + DoubleConv(32->64)  -> skip1")
    line("enc2", enc_out["enc2"], "pool + DoubleConv(64->128) -> skip2")
    line("enc3", enc_out["enc3"], "pool + DoubleConv(128->256) -> skip3")
    line("bottom", b, "第 4 次 pool，无卷积")

    print("\n[解码路径]（逐级校验 up 输出 == skip 尺寸）")
    for name, blk, up_out, skip in (
        ("dec3", model.dec3, u3, d3),
        ("dec2", model.dec2, u2, d2),
        ("dec1", model.dec1, u1, d1),
        ("dec0", model.dec0, u0, d0),
    ):
        print(
            f"  {name:<10}up_in={_fmt(tuple(blk._dbg['up_in'])):<12}"
            f"up_out={_fmt(tuple(up_out.shape)):<14}"
            f"skip={_fmt(tuple(skip.shape)):<14}"
            f"cat_ch={up_out.shape[1] + skip.shape[1]:<5}"
            f"out_ch={up_out.shape[1]:<5}"
            f"{'OK' if _hw(up_out) == _hw(skip) else 'MISMATCH'}"
        )

    print("\n[输出]")
    line("head", y, "1x1 Conv -> 1 通道")
    if _hw(y) != _hw(x):
        print(f"\n[FAIL] 输出尺寸 {_hw(y)} != 输入尺寸 {_hw(x)}")
        return 1

    print("\n[参数量明细]")
    total = 0
    for name, child in model.named_children():
        n = sum(p.numel() for p in child.parameters())
        total += n
        bn = sum(p.numel() for m in child.modules() if isinstance(m, nn.BatchNorm2d)
                 for p in m.parameters())
        conv = n - bn
        print(f"  {name:<10}{n:>12,}    (conv/linear {conv:>11,} + bn {bn:>7,})")
    print(f"  {'-' * 62}")
    print(f"  {'TOTAL':<10}{total:>12,}    trainable={model.n_trainable_parameters():,}")

    lo, hi = PARAM_RANGE
    status = "在 1.5M-2.5M 范围内" if lo <= total <= hi else "超出 1.5M-2.5M 范围"
    print(f"\n  参考区间 {lo:,} - {hi:,}：{status}")
    print(f"  换算：{total / 1e6:.3f} M")

    # 反例自检：尺寸不符必须报错（证明护栏真的生效，不是摆设）
    print("\n[护栏反例自检]")
    try:
        model.dec3(torch.zeros(1, 256, 2, 74, device=device),
                   torch.zeros(1, 256, 4, 150, device=device))
        print("  [FAIL] 尺寸不符却没有报错，护栏失效")
        return 1
    except ModelSpecError as exc:
        print(f"  [OK] 尺寸不符正确抛错: {str(exc)[:72]}")

    print("\n[OK] 逐层 shape 校验通过，输出尺寸与输入一致，护栏生效")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description="U-Net 结构自检（不训练、不 smoke）")
    ap.add_argument("--inspect", action="store_true", help="打印逐层 shape 与参数量")
    ap.add_argument("--device", default="cpu", help="cpu（默认）或 cuda")
    ap.add_argument("--batch", type=int, default=1)
    args = ap.parse_args()
    if not args.inspect:
        ap.print_help()
        return 2
    return inspect(device=args.device, batch=args.batch)


if __name__ == "__main__":
    raise SystemExit(main())
