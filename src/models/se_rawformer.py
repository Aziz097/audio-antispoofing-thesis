"""
src/models/se_rawformer.py
SE-Rawformer: CNN-Transformer hybrid with SE-Res2Net frontend and 1D positional encoding.

Reimplemented from:
    rst0070/Rawformer-implementation-anti-spoofing — SE variant
    "Leveraging Positional-Related Local-Global Dependency for
    Synthetic Speech Detection"

Uses 1D Positional Encoding variant (1D-PE ONLY).
Target: ~0.37M params, ~6.1G MACs.
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
# SincConv — Shared sinc filterbank
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
# Squeeze-and-Excitation Layer
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
# Conv2DBlock_S — First block (no SE, simpler residual)
# ============================================================

class Conv2DBlock_S(nn.Module):
    """Simple 2D residual block used as the first block in the frontend.

    Uses larger (2,5) kernel in first conv and MaxPool(1,6).
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        is_first_block: bool = False,
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
                in_channels, out_channels, kernel_size=(1, 3), padding=(0, 1)
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
# Conv2DBlock_SE — SE-Res2Net block
# ============================================================

class Conv2DBlock_SE(nn.Module):
    """SE-Res2Net block for SE-Rawformer frontend.

    Multi-scale feature extraction with squeeze-and-excitation:
    1×7 bottleneck → split into `scale` groups →
    cascaded 3×9 convolutions → concat → 1×7 → SE → residual + pool.
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

        # Cascaded scale convolutions
        self.conv2 = nn.ModuleList()
        for _ in range(1, scale):
            self.conv2.append(nn.Sequential(
                nn.Conv2d(self.sub_channels, self.sub_channels,
                          kernel_size=(3, 9), padding=(1, 4)),
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
                in_channels, out_channels, kernel_size=(1, 7), padding=(0, 3)
            )

        self.pooling = nn.MaxPool2d(kernel_size=(1, 6))

    def forward(self, x: Tensor) -> Tensor:
        identity = x
        if self.downsampler is not None:
            identity = self.downsampler(identity)

        x = self.normalizer(x)
        x = self.conv1(x)

        # Res2Net multi-scale
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
# Frontend_SE — SE-Rawformer frontend
# ============================================================

class FrontendSE(nn.Module):
    """SE-Rawformer frontend: SincConv → LFM → 4 × Conv2D blocks → HFM.

    Output shape for 4s input: (batch, 64, 23, 16).
    """

    def __init__(self, sinc_kernel_size: int = 128, sample_rate: int = 16000) -> None:
        super().__init__()
        self.sinc_layer = SincConv(
            out_channels=70, kernel_size=sinc_kernel_size, sample_rate=sample_rate
        )
        self.bn = nn.BatchNorm2d(1)
        self.selu = nn.SELU(inplace=True)
        self.conv_blocks = nn.Sequential(
            Conv2DBlock_S(in_channels=1, out_channels=32, is_first_block=True),
            Conv2DBlock_SE(in_channels=32, out_channels=32),
            Conv2DBlock_SE(in_channels=32, out_channels=64),
            Conv2DBlock_SE(in_channels=64, out_channels=64),
        )

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: (batch, T) raw waveform.
        Returns:
            HFM: (batch, C=64, f=23, t).
        """
        x = x.unsqueeze(1)  # (B, 1, T)
        x = self.sinc_layer(x)  # (B, 70, T')
        x = x.unsqueeze(1)  # (B, 1, 70, T')
        x = F.max_pool2d(torch.abs(x), (3, 3))  # (B, 1, 23, T'')
        x = self.bn(x)
        x = self.selu(x)
        return self.conv_blocks(x)  # (B, 64, 23, t)


# ============================================================
# 1D Positional Aggregator
# ============================================================

class PositionalAggregator1D(nn.Module):
    """Flatten 2D feature map and add 1D sinusoidal positional encoding.

    Input:  (batch, C, f, t) — HFM from frontend.
    Output: (batch, f*t, C) — sequence with positional encoding.
    """

    def __init__(self, max_C: int = 64, max_ft: int = 368) -> None:
        super().__init__()
        self.max_C = max_C
        self.max_ft = max_ft

        # Pre-compute sinusoidal encoding (not a parameter)
        encoding = torch.zeros(max_ft, max_C)
        pos = torch.arange(0, max_ft).float().unsqueeze(1)
        div_term = torch.arange(0, max_C, 2).float().unsqueeze(0)

        encoding[:, 0::2] = torch.sin(pos / (10000 ** (div_term / max_C)))
        encoding[:, 1::2] = torch.cos(pos / (10000 ** (div_term / max_C)))

        self.register_buffer("encoding", encoding)

    def forward(self, HFM: Tensor) -> Tensor:
        """
        Args:
            HFM: (batch, C, f, t) feature map.
        Returns:
            (batch, f*t, C) positionally-encoded sequence.
        """
        batch, C, f, t = HFM.shape
        # Flatten spatial dims and transpose
        out = HFM.flatten(start_dim=2).transpose(1, 2)  # (B, f*t, C)
        # Add positional encoding
        out = out + self.encoding[: f * t, :C].unsqueeze(0)
        return out


# ============================================================
# Transformer Encoder Layer
# ============================================================

class TransformerEncoderLayer(nn.Module):
    """Standard Transformer encoder layer using F.scaled_dot_product_attention.

    Uses PyTorch 2.x native SDPA for efficiency and Flash Attention support.
    """

    def __init__(
        self,
        d_model: int = 64,
        n_head: int = 8,
        ffn_hidden: int = 660,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.n_head = n_head
        self.d_model = d_model
        self.d_k = d_model // n_head

        # QKV projections
        self.W_Q = nn.Linear(d_model, d_model)
        self.W_K = nn.Linear(d_model, d_model)
        self.W_V = nn.Linear(d_model, d_model)
        self.W_O = nn.Linear(d_model, d_model)

        # Layer norms
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        # FFN
        self.ffn = nn.Sequential(
            nn.Linear(d_model, ffn_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_hidden, d_model),
        )

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: (batch, seq_len, d_model)
        Returns:
            (batch, seq_len, d_model)
        """
        batch, seq_len, _ = x.shape

        # Multi-head self-attention
        residual = x
        Q = self.W_Q(x).view(batch, seq_len, self.n_head, self.d_k).transpose(1, 2)
        K = self.W_K(x).view(batch, seq_len, self.n_head, self.d_k).transpose(1, 2)
        V = self.W_V(x).view(batch, seq_len, self.n_head, self.d_k).transpose(1, 2)

        # Use PyTorch 2.x native scaled_dot_product_attention
        attn_out = F.scaled_dot_product_attention(Q, K, V, dropout_p=0.0)

        # Concat heads
        attn_out = attn_out.transpose(1, 2).contiguous().view(batch, seq_len, self.d_model)
        attn_out = self.W_O(attn_out)
        x = self.norm1(residual + self.dropout1(attn_out))

        # FFN
        residual = x
        x = self.norm2(residual + self.dropout2(self.ffn(x)))

        return x


# ============================================================
# Sequence Pooling
# ============================================================

class SequencePooling(nn.Module):
    """Attention-based sequence pooling.

    Learns a weighted sum over sequence positions to produce
    a single vector representation.

    Reference: Li et al., "The Role of Long-Term Dependency in
    Synthetic Speech Detection", IEEE SPL 2022.
    """

    def __init__(self, d_model: int) -> None:
        super().__init__()
        self.linear = nn.Linear(d_model, 1)

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: (batch, seq_len, d_model)
        Returns:
            (batch, d_model)
        """
        w = self.linear(x)  # (B, L, 1)
        w = F.softmax(w.transpose(1, 2), dim=-1)  # (B, 1, L)
        x = torch.matmul(w, x).squeeze(1)  # (B, d_model)
        return x


# ============================================================
# SE-Rawformer Full Model
# ============================================================

class SERAWFormer(nn.Module):
    """SE-Rawformer with 1D Positional Encoding.

    Architecture:
        SincConv → SE-Res2Net Frontend (4 blocks) →
        1D Positional Aggregator →
        M=2 Transformer Encoders →
        Sequence Pooling → Linear(2)

    Target: ~0.37M params, ~6.1G MACs.

    Args:
        sinc_kernel_size: Sinc filter kernel size.
        sample_rate: Audio sample rate in Hz.
        transformer_hidden: FFN hidden dimension in Transformer.
        n_encoder: Number of Transformer encoder layers.
        C: Channel dimension / embedding dimension.
        n_head: Number of attention heads.
        dropout: Dropout rate.
    """

    def __init__(
        self,
        sinc_kernel_size: int = 128,
        sample_rate: int = 16000,
        transformer_hidden: int = 660,
        n_encoder: int = 2,
        C: int = 64,
        n_head: int = 8,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        # Frontend: SincConv → SE-Res2Net blocks
        self.frontend = FrontendSE(
            sinc_kernel_size=sinc_kernel_size,
            sample_rate=sample_rate,
        )

        # 1D Positional Aggregator
        # For 4s input at 16kHz: HFM → (B, 64, 23, 16) → f*t = 368
        self.pos_agg = PositionalAggregator1D(max_C=C, max_ft=1024)

        # Transformer encoders
        self.encoders = nn.Sequential(*[
            TransformerEncoderLayer(
                d_model=C, n_head=n_head,
                ffn_hidden=transformer_hidden, dropout=dropout,
            )
            for _ in range(n_encoder)
        ])

        # Sequence pooling → classification
        self.seq_pool = SequencePooling(d_model=C)
        self.classifier = nn.Linear(C, 2)

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass.

        Args:
            x: Raw waveform (batch, 64000).

        Returns:
            Logits (batch, 2).
        """
        # Frontend → HFM
        hfm = self.frontend(x)  # (B, 64, 23, t)

        # Positional aggregation
        _, _, f, t = hfm.shape
        assert f * t <= 1024, (
            f"Sequence length {f * t} exceeds positional encoding "
            f"buffer (1024). Reduce input length or increase max_ft."
        )
        seq = self.pos_agg(hfm)  # (B, f*t, 64)

        # Transformer encoders
        seq = self.encoders(seq)  # (B, f*t, 64)

        # Sequence pooling → classifier
        pooled = self.seq_pool(seq)  # (B, 64)
        logits = self.classifier(pooled)  # (B, 2)

        return logits


# ============================================================
# Smoke Test
# ============================================================

if __name__ == "__main__":
    print("=" * 60)
    print("SE-Rawformer — Smoke Test")
    print("=" * 60)

    model = SERAWFormer()
    model.eval()

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {n_params:,}")

    x = torch.randn(2, 64000)
    with torch.no_grad():
        out = model(x)
    print(f"Input shape:  {x.shape}")
    print(f"Output shape: {out.shape}")
    assert out.shape == (2, 2), f"Expected (2, 2), got {out.shape}"

    print("\n✅ SE-Rawformer smoke test passed!")
