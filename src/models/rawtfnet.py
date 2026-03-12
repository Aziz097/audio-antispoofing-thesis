"""
src/models/rawtfnet.py
RawTFNet: Lightweight CNN with depthwise-separable SE-Res2Net frontend
         and Time-Frequency Separable Convolution classifier.

Reimplemented from:
    YangXiao-QMUL/RawTFNet
    "RawTFNet: Utilising Raw Time-Frequency Representations for
    Spoofed Speech Detection"

Two variants controlled by `tau` parameter:
    tau=32 (RawTFNet):       DWS_Frontend_SE (32/64 ch) + TfSepNet(depth=10, width=32)
    tau=16 (RawTFNet_small): DWS_Frontend_SE_small (16/32 ch) + TfSepNet(depth=18, width=16)

Input:  (batch, 64000) raw waveform at 16 kHz.
Output: (batch, 2) binary logits [spoof, bonafide].
"""

import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


# ============================================================
# SincConv — Shared sinc filterbank (same as other models)
# ============================================================

class SincConv(nn.Module):
    """Sinc-based bandpass filter convolution (mel-spaced, non-learnable)."""

    @staticmethod
    def to_mel(hz: float) -> float:
        return 2595.0 * np.log10(1.0 + hz / 700.0)

    @staticmethod
    def to_hz(mel: float) -> float:
        return 700.0 * (10.0 ** (mel / 2595.0) - 1.0)

    def __init__(
        self,
        out_channels: int = 70,
        kernel_size: int = 128,
        sample_rate: int = 16000,
    ) -> None:
        super().__init__()
        self.out_channels = out_channels
        self.kernel_size = kernel_size if kernel_size % 2 != 0 else kernel_size + 1
        self.sample_rate = sample_rate

        NFFT = 512
        f = int(sample_rate / 2) * np.linspace(0, 1, int(NFFT / 2) + 1)
        fmel = self.to_mel(f)
        filbandwidths = np.linspace(np.min(fmel), np.max(fmel), out_channels + 1)
        filbandwidthsf = self.to_hz(filbandwidths)

        self.mel = filbandwidthsf
        self.hsupp = torch.arange(
            -(self.kernel_size - 1) / 2, (self.kernel_size - 1) / 2 + 1
        )
        band_pass = torch.zeros(out_channels, self.kernel_size)
        for i in range(len(self.mel) - 1):
            fmin, fmax = self.mel[i], self.mel[i + 1]
            hHigh = (2 * fmax / sample_rate) * np.sinc(
                2 * fmax * self.hsupp.numpy() / sample_rate
            )
            hLow = (2 * fmin / sample_rate) * np.sinc(
                2 * fmin * self.hsupp.numpy() / sample_rate
            )
            band_pass[i, :] = Tensor(
                np.hamming(self.kernel_size)
            ) * Tensor(hHigh - hLow)
        self.register_buffer("band_pass", band_pass)

    def forward(self, x: Tensor) -> Tensor:
        filters = self.band_pass.view(self.out_channels, 1, self.kernel_size)
        return F.conv1d(x, filters, stride=1, padding=0, bias=None)


# ============================================================
# Separable Convolution 2D
# ============================================================

class SeparableConv2d(nn.Module):
    """Depthwise separable convolution: depthwise + optional pointwise."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size,
        stride: int = 1,
        padding=0,
        dilation: int = 1,
        bias: bool = False,
        pointwise: bool = False,
    ) -> None:
        super().__init__()
        self.conv2d = nn.Conv2d(
            in_channels, in_channels,
            kernel_size=kernel_size, stride=stride,
            groups=in_channels, padding=padding,
            dilation=dilation, bias=bias,
        )
        if pointwise:
            self.pointwise = nn.Conv2d(
                in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=bias,
            )
        else:
            self.pointwise = nn.Identity()

    def forward(self, x: Tensor) -> Tensor:
        x = self.conv2d(x)
        x = self.pointwise(x)
        return x


# ============================================================
# SE Layer
# ============================================================

class SELayer(nn.Module):
    """Channel-wise Squeeze-and-Excitation."""

    def __init__(self, channels: int, reduction: int = 8) -> None:
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels),
            nn.Sigmoid(),
        )

    def forward(self, x: Tensor) -> Tensor:
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y


# ============================================================
# Conv2DBlock_S — First frontend block (no SE)
# ============================================================

class Conv2DBlock_S(nn.Module):
    """Simple 2D residual block as first block in frontend."""

    def __init__(
        self, in_channels: int, out_channels: int, is_first_block: bool = False
    ) -> None:
        super().__init__()

        self.normalizer = None
        if not is_first_block:
            self.normalizer = nn.Sequential(
                nn.BatchNorm2d(in_channels),
                nn.SELU(inplace=True),
            )

        self.layers = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=(2, 5), padding=(1, 2)),
            nn.BatchNorm2d(out_channels),
            nn.SELU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=(2, 3), padding=(0, 1)),
        )

        self.downsampler = None
        if in_channels != out_channels:
            self.downsampler = nn.Conv2d(
                in_channels, out_channels, kernel_size=(1, 3), padding=(0, 1),
            )

        self.pooling = nn.MaxPool2d(kernel_size=(1, 6))

    def forward(self, x: Tensor) -> Tensor:
        identity = x
        if self.downsampler is not None:
            identity = self.downsampler(identity)
        if self.normalizer is not None:
            x = self.normalizer(x)
        x = self.layers(x)
        x = x + identity
        return self.pooling(x)


# ============================================================
# DWS_Conv2DBlock_SE — Depthwise-Separable SE-Res2Net block
# ============================================================

class DWSConv2DBlockSE(nn.Module):
    """Depthwise-separable SE-Res2Net block for RawTFNet frontend.

    Same as Conv2DBlock_SE but uses SeparableConv2d in the Res2Net
    branches instead of standard Conv2d → fewer parameters.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        scale: int = 8,
        se_reduction: int = 8,
    ) -> None:
        super().__init__()
        self.scale = scale
        self.sub_channels = out_channels // scale
        self.hidden_channels = self.sub_channels * scale
        relu = nn.ReLU(inplace=True)

        self.normalizer = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.SELU(inplace=True),
        )

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, self.hidden_channels, kernel_size=(1, 7), padding=(0, 3)),
            nn.BatchNorm2d(self.hidden_channels),
            relu,
        )

        # Cascaded depthwise-separable scale convolutions
        self.conv2 = nn.ModuleList()
        for _ in range(1, scale):
            self.conv2.append(nn.Sequential(
                SeparableConv2d(
                    self.sub_channels, self.sub_channels,
                    kernel_size=(3, 9), padding=(1, 4),
                ),
                nn.BatchNorm2d(self.sub_channels),
                relu,
            ))

        self.conv3 = nn.Sequential(
            nn.Conv2d(self.hidden_channels, out_channels, kernel_size=(1, 7), padding=(0, 3)),
            nn.BatchNorm2d(out_channels),
            relu,
        )

        self.se_module = SELayer(out_channels, reduction=se_reduction)

        self.downsampler = None
        if in_channels != out_channels:
            self.downsampler = nn.Conv2d(
                in_channels, out_channels, kernel_size=(1, 7), padding=(0, 3),
            )

        self.pooling = nn.MaxPool2d(kernel_size=(1, 6))

    def forward(self, x: Tensor) -> Tensor:
        identity = x
        if self.downsampler is not None:
            identity = self.downsampler(identity)

        x = self.normalizer(x)
        x = self.conv1(x)

        # Res2Net multi-scale with depthwise-separable convolutions
        x_sub = torch.split(x, self.sub_channels, dim=1)
        y_sub = [x_sub[0]]

        for i in range(1, self.scale):
            inp = x_sub[i] if i == 1 else x_sub[i] + y_sub[i - 1]
            y_sub.append(self.conv2[i - 1](inp))

        y = torch.cat(y_sub, dim=1)
        y = self.conv3(y)
        y = self.se_module(y)

        y = y + identity
        return self.pooling(y)


# ============================================================
# DWS Frontend SE — Standard (tau=32: channels 32→64)
# ============================================================

class DWSFrontendSE(nn.Module):
    """DWS-SE frontend: SincConv → abs → MaxPool → BN → SELU → 4 blocks.

    tau=32 config: Conv2DBlock_S(1→32) → DWS(32→32) → DWS(32→64) → DWS(64→64)
    Output: (batch, 64, 23, 16) for 4s input.

    Design note — block count:
        The block sequence is intentionally 1×Conv2DBlock_S + 3×DWSConv2DBlockSE,
        matching the original ``DWS_Frontend_SE`` in the RawTFNet reference
        implementation (external/RawTFNet/model_scripts/blocks/frontend.py).
        Conv2DBlock_S is a standard (non-depthwise-separable) strided conv that
        acts as the initial spatial downsampler; the three DWS blocks that follow
        are the lightweight feature extractors described in the paper.  Replacing
        Conv2DBlock_S with a fourth DWS block would diverge from the published
        architecture and is NOT intended.
    """

    def __init__(self, sinc_kernel_size: int = 128, sample_rate: int = 16000) -> None:
        super().__init__()
        self.sinc_layer = SincConv(
            out_channels=70, kernel_size=sinc_kernel_size, sample_rate=sample_rate,
        )
        self.bn = nn.BatchNorm2d(1)
        self.selu = nn.SELU(inplace=True)
        self.conv_blocks = nn.Sequential(
            Conv2DBlock_S(in_channels=1, out_channels=32, is_first_block=True),
            DWSConv2DBlockSE(in_channels=32, out_channels=32),
            DWSConv2DBlockSE(in_channels=32, out_channels=64),
            DWSConv2DBlockSE(in_channels=64, out_channels=64),
        )

    def forward(self, x: Tensor) -> Tensor:
        x = x.unsqueeze(1)  # (B, 1, T)
        x = self.sinc_layer(x)  # (B, 70, T')
        x = x.unsqueeze(1)  # (B, 1, 70, T')
        x = F.max_pool2d(torch.abs(x), (3, 3))
        x = self.bn(x)
        x = self.selu(x)
        return self.conv_blocks(x)


# ============================================================
# DWS Frontend SE Small — Halved channels (tau=16: 16→32)
# ============================================================

class DWSFrontendSESmall(nn.Module):
    """DWS-SE frontend (small): channels halved.

    tau=16 config: Conv2DBlock_S(1→16) → DWS(16→16) → DWS(16→32) → DWS(32→32)
    Output: (batch, 32, 23, 16) for 4s input.
    """

    def __init__(self, sinc_kernel_size: int = 128, sample_rate: int = 16000) -> None:
        super().__init__()
        self.sinc_layer = SincConv(
            out_channels=70, kernel_size=sinc_kernel_size, sample_rate=sample_rate,
        )
        self.bn = nn.BatchNorm2d(1)
        self.selu = nn.SELU(inplace=True)
        self.conv_blocks = nn.Sequential(
            Conv2DBlock_S(in_channels=1, out_channels=16, is_first_block=True),
            DWSConv2DBlockSE(in_channels=16, out_channels=16),
            DWSConv2DBlockSE(in_channels=16, out_channels=32),
            DWSConv2DBlockSE(in_channels=32, out_channels=32),
        )

    def forward(self, x: Tensor) -> Tensor:
        x = x.unsqueeze(1)
        x = self.sinc_layer(x)
        x = x.unsqueeze(1)
        x = F.max_pool2d(torch.abs(x), (3, 3))
        x = self.bn(x)
        x = self.selu(x)
        return self.conv_blocks(x)


# ============================================================
# TfSepNet Classifier Components
# ============================================================

class ShuffleLayer(nn.Module):
    """Channel shuffle for grouped convolutions."""

    def __init__(self, group: int = 10) -> None:
        super().__init__()
        self.group = group

    def forward(self, x: Tensor) -> Tensor:
        b, c, f, t = x.shape
        assert c % self.group == 0, f"Channels {c} not divisible by groups {self.group}"
        group_channels = c // self.group
        x = x.reshape(b, group_channels, self.group, f, t)
        x = x.permute(0, 2, 1, 3, 4)
        x = x.reshape(b, c, f, t)
        return x


class AdaResNorm(nn.Module):
    """Adaptive Residual Normalization.

    Learnable blend between identity and instance-frequency normalization.
    rho controls the mixing: output = rho * x + (1-rho) * IFN(x).
    """

    def __init__(self, c: int, grad: bool = False, eps: float = 1e-5) -> None:
        super().__init__()
        self.grad = grad
        self.eps_val = eps

        if self.grad:
            self.rho = nn.Parameter(torch.full((1, c, 1, 1), 0.5))
            self.gamma = nn.Parameter(torch.ones(1, c, 1, 1))
            self.beta = nn.Parameter(torch.zeros(1, c, 1, 1))
        else:
            self.register_buffer("rho", torch.full((1, c, 1, 1), 0.5))

    def forward(self, x: Tensor) -> Tensor:
        identity = x
        # Instance-frequency normalization: normalize across (channel, time) dims
        ifn_mean = x.mean((1, 3), keepdim=True)
        ifn_var = x.var((1, 3), keepdim=True)
        ifn = (x - ifn_mean) / (ifn_var + self.eps_val).sqrt()

        res_norm = self.rho * identity + (1 - self.rho) * ifn

        if self.grad:
            return self.gamma * res_norm + self.beta
        return res_norm


class ConvBlock(nn.Module):
    """Standard Conv2d + optional BN + optional ReLU."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size,
        stride: int = 1,
        padding=0,
        dilation: int = 1,
        groups: int = 1,
        bias: bool = False,
        use_bn: bool = False,
        use_relu: bool = False,
    ) -> None:
        super().__init__()
        self.use_bn = use_bn
        self.use_relu = use_relu

        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel_size=kernel_size,
            stride=stride, padding=padding, dilation=dilation,
            groups=groups, bias=bias,
        )
        if self.use_bn:
            self.bn = nn.BatchNorm2d(out_channels)
        if self.use_relu:
            self.activation = nn.ReLU(inplace=True)

    def forward(self, x: Tensor) -> Tensor:
        x = self.conv(x)
        if self.use_bn:
            x = self.bn(x)
        if self.use_relu:
            x = self.activation(x)
        return x


class TimeFreqSepConvs(nn.Module):
    """Time-Frequency Separable Convolution block.

    Split channels into two halves:
      - Half 1: Frequency-wise DW conv(3,1) → freq avg pool → PW conv → residual
      - Half 2: Time-wise DW conv(1,3) → time avg pool → PW conv → residual
    Concat → output.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        dropout_rate: float = 0.2,
        shuffle: bool = False,
        shuffle_groups: int = 10,
    ) -> None:
        super().__init__()
        self.transition = in_channels != out_channels
        self.shuffle = shuffle
        self.half_channels = out_channels // 2

        if self.transition:
            self.trans_conv = ConvBlock(
                in_channels, out_channels, kernel_size=1,
                use_bn=True, use_relu=True,
            )

        # Frequency branch
        self.freq_dw_conv = ConvBlock(
            self.half_channels, self.half_channels,
            kernel_size=(3, 1), padding=(1, 0),
            groups=self.half_channels, use_bn=True, use_relu=True,
        )
        self.freq_pw_conv = ConvBlock(
            self.half_channels, self.half_channels,
            kernel_size=1, use_bn=True, use_relu=True,
        )

        # Temporal branch
        self.temp_dw_conv = ConvBlock(
            self.half_channels, self.half_channels,
            kernel_size=(1, 3), padding=(0, 1),
            groups=self.half_channels, use_bn=True, use_relu=True,
        )
        self.temp_pw_conv = ConvBlock(
            self.half_channels, self.half_channels,
            kernel_size=1, use_bn=True, use_relu=True,
        )

        self.dropout = nn.Dropout(p=dropout_rate)
        self.shuffle_layer = ShuffleLayer(group=shuffle_groups)

    def forward(self, x: Tensor) -> Tensor:
        # Channel transition if needed
        if self.transition:
            x = self.trans_conv(x)

        # Split into freq and temp halves
        x1, x2 = torch.split(x, self.half_channels, dim=1)
        identity1, identity2 = x1, x2

        # Frequency branch: DW(3,1) → freq avg pool → PW → residual
        x1 = self.freq_dw_conv(x1)
        x1 = x1.mean(2, keepdim=True)  # Frequency average pooling
        x1 = self.freq_pw_conv(x1)
        x1 = self.dropout(x1)
        x1 = x1 + identity1  # broadcast: (B,C/2,1,T) + (B,C/2,F,T) → (B,C/2,F,T)

        # Temporal branch: DW(1,3) → time avg pool → PW → residual
        x2 = self.temp_dw_conv(x2)
        x2 = x2.mean(3, keepdim=True)  # Temporal average pooling
        x2 = self.temp_pw_conv(x2)
        x2 = self.dropout(x2)
        x2 = x2 + identity2  # broadcast: (B,C/2,F,1) + (B,C/2,F,T) → (B,C/2,F,T)

        # Channel shuffle AFTER concat so features from both branches are mixed.
        # Shuffling before the split would only permute within each branch's
        # input, not across the freq/temp outputs.
        out = torch.cat((x1, x2), dim=1)
        if self.shuffle:
            out = self.shuffle_layer(out)
        return out


# ============================================================
# TfSepNet architecture configs
# ============================================================

TFSEPNET_CONFIGS = {
    10: ["N", 1, 1, "N", "M", 1.5, 1.5, "N", "M", 2, 2, "N", 2.5, 2.5, 2.5, "N"],
    18: ["CONV", "N", 1, 1, "N", "M", 1.5, 1.5, "N", "M", 2, 2, "N", 2.5, 2.5, 2.5, "N"],
}


class TfSepNet(nn.Module):
    """Time-Frequency Separable Convolution Network classifier.

    Processes 2D feature maps from frontend through alternating
    TimeFreqSepConvs, AdaResNorm, and MaxPool layers.

    Args:
        depth: Config depth key (10 or 18).
        width: Base channel width (tau parameter).
        dropout_rate: Dropout probability.
        shuffle: Whether to use channel shuffle.
        shuffle_groups: Number of groups for shuffle.
    """

    def __init__(
        self,
        depth: int = 10,
        width: int = 32,
        dropout_rate: float = 0.2,
        shuffle: bool = True,
        shuffle_groups: int = 8,
    ) -> None:
        super().__init__()
        cfg = TFSEPNET_CONFIGS[depth]
        self.width = width
        self.dropout_rate = dropout_rate
        self.shuffle = shuffle
        self.shuffle_groups = shuffle_groups

        self.feature = self._make_layers(cfg)

        # Find last numeric entry in cfg for classifier input channels
        last_numeric = None
        for v in reversed(cfg):
            if not isinstance(v, str):
                last_numeric = v
                break
        self.classifier = nn.Conv2d(round(last_numeric * self.width), 2, 1, bias=True)

    def _make_layers(self, cfg: list) -> nn.Sequential:
        layers = []
        vt = 2  # Current channel multiplier (starts at 2 × width)

        for v in cfg:
            if v == "CONV":
                # Initial conv for depth=18 config
                layers.append(
                    nn.Conv2d(32, 2 * self.width, 5, stride=2, bias=False, padding=1)
                )
            elif v == "N":
                layers.append(AdaResNorm(c=round(vt * self.width), grad=True))
            elif v == "M":
                layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
            elif v != vt:
                # Channel transition
                layers.append(TimeFreqSepConvs(
                    in_channels=round(vt * self.width),
                    out_channels=round(v * self.width),
                    dropout_rate=self.dropout_rate,
                    shuffle=self.shuffle,
                    shuffle_groups=self.shuffle_groups,
                ))
                vt = v
            else:
                # Same-channel block
                layers.append(TimeFreqSepConvs(
                    in_channels=round(vt * self.width),
                    out_channels=round(vt * self.width),
                    dropout_rate=self.dropout_rate,
                    shuffle=self.shuffle,
                    shuffle_groups=self.shuffle_groups,
                ))

        return nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: Feature map (batch, C_frontend, f, t).
        Returns:
            Logits (batch, 2).
        """
        x = self.feature(x)
        y = self.classifier(x)
        y = y.mean((-1, -2))  # Global average pooling
        return y


# ============================================================
# RawTFNet Full Model
# ============================================================

class RawTFNet(nn.Module):
    """RawTFNet: DWS-SE Frontend + TfSepNet classifier.

    Unified model supporting both tau=32 and tau=16 variants.

    tau=32 (default):
        - DWS_Frontend_SE: channels [1→32→32→64→64]
        - TfSepNet: depth=10, width=32
        - ~0.41M params

    tau=16 (small):
        - DWS_Frontend_SE_small: channels [1→16→16→32→32]
        - TfSepNet: depth=18, width=16
        - ~0.08M params

    Args:
        tau: Width parameter (16 or 32).
        sample_rate: Audio sample rate.
        dropout_rate: Dropout in TfSepNet.
    """

    def __init__(
        self,
        tau: int = 32,
        sample_rate: int = 16000,
        dropout_rate: float = 0.2,
    ) -> None:
        super().__init__()
        self.tau = tau

        if tau == 32:
            self.front_end = DWSFrontendSE(
                sinc_kernel_size=128, sample_rate=sample_rate,
            )
            self.classifier = TfSepNet(
                depth=10, width=32,
                dropout_rate=dropout_rate, shuffle=True, shuffle_groups=8,
            )
        elif tau == 16:
            self.front_end = DWSFrontendSESmall(
                sinc_kernel_size=128, sample_rate=sample_rate,
            )
            self.classifier = TfSepNet(
                depth=18, width=16,
                dropout_rate=dropout_rate, shuffle=True, shuffle_groups=8,
            )
        else:
            raise ValueError(f"Unsupported tau={tau}. Choose 16 or 32.")

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass.

        Args:
            x: Raw waveform (batch, 64000).
        Returns:
            Logits (batch, 2).
        """
        x = self.front_end(x)   # (B, C, 23, t)
        x = self.classifier(x)  # (B, 2)
        return x


# ============================================================
# Smoke Test
# ============================================================

if __name__ == "__main__":
    print("=" * 60)
    print("RawTFNet — Smoke Test")
    print("=" * 60)

    for tau in [32, 16]:
        print(f"\n--- tau={tau} ---")
        model = RawTFNet(tau=tau)
        model.eval()

        n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Trainable parameters: {n_params:,}")

        x = torch.randn(2, 64000)
        with torch.no_grad():
            out = model(x)
        print(f"Input shape:  {x.shape}")
        print(f"Output shape: {out.shape}")
        assert out.shape == (2, 2), f"Expected (2, 2), got {out.shape}"

    print("\n✅ RawTFNet smoke test passed (both tau=32 and tau=16)!")
