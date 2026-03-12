"""
src/models/aasist.py
AASIST: Audio Anti-Spoofing using Integrated Spectro-Temporal Graph Attention Networks.

Reimplemented from:
    Jung et al., "AASIST: Audio Anti-Spoofing using Integrated
    Spectro-Temporal Graph Attention Networks", ICASSP 2022.

Original: https://github.com/clovaai/aasist (MIT License)

Target: ~0.30M params, ~8.9G MACs.
Input:  (batch, 64000) raw waveform at 16 kHz.
Output: (batch, 2) binary logits [spoof, bonafide].
"""

import random
from typing import Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


# ============================================================
# SincConv — Learnable sinc-based bandpass filterbank
# ============================================================

class SincConv(nn.Module):
    """Sinc-based bandpass filter convolution layer.

    Mel-spaced sinc filters applied to raw waveform.
    Not a learnable parameter — fixed filterbank.
    """

    @staticmethod
    def to_mel(hz: float) -> float:
        return 2595.0 * np.log10(1.0 + hz / 700.0)

    @staticmethod
    def to_hz(mel: float) -> float:
        return 700.0 * (10.0 ** (mel / 2595.0) - 1.0)

    def __init__(
        self,
        out_channels: int,
        kernel_size: int,
        sample_rate: int = 16000,
        in_channels: int = 1,
        stride: int = 1,
        padding: int = 0,
        dilation: int = 1,
    ) -> None:
        super().__init__()
        if in_channels != 1:
            raise ValueError("SincConv only supports in_channels=1")

        self.out_channels = out_channels
        self.kernel_size = kernel_size if kernel_size % 2 != 0 else kernel_size + 1
        self.sample_rate = sample_rate
        self.stride = stride
        self.padding = padding
        self.dilation = dilation

        # Compute mel-spaced filter band edges
        NFFT = 512
        f = int(self.sample_rate / 2) * np.linspace(0, 1, int(NFFT / 2) + 1)
        fmel = self.to_mel(f)
        fmelmax, fmelmin = np.max(fmel), np.min(fmel)
        filbandwidthsmel = np.linspace(fmelmin, fmelmax, self.out_channels + 1)
        filbandwidthsf = self.to_hz(filbandwidthsmel)
        self.mel = filbandwidthsf

        # Support vector for filter computation
        self.hsupp = torch.arange(
            -(self.kernel_size - 1) / 2, (self.kernel_size - 1) / 2 + 1
        )

        # Pre-compute bandpass filters
        band_pass = torch.zeros(self.out_channels, self.kernel_size)
        for i in range(len(self.mel) - 1):
            fmin = self.mel[i]
            fmax = self.mel[i + 1]
            hHigh = (2 * fmax / self.sample_rate) * np.sinc(
                2 * fmax * self.hsupp.numpy() / self.sample_rate
            )
            hLow = (2 * fmin / self.sample_rate) * np.sinc(
                2 * fmin * self.hsupp.numpy() / self.sample_rate
            )
            hideal = hHigh - hLow
            band_pass[i, :] = Tensor(
                np.hamming(self.kernel_size)
            ) * Tensor(hideal)
        self.register_buffer("band_pass", band_pass)

    def forward(self, x: Tensor, mask: bool = False) -> Tensor:
        """Apply sinc convolution.

        Args:
            x: Input tensor (batch, 1, samples).
            mask: If True, randomly zero out some filter bands (freq augment).

        Returns:
            Filtered output (batch, out_channels, T).
        """
        band_pass_filter = self.band_pass.clone()
        if mask:
            A = int(np.random.uniform(0, 20))
            A0 = random.randint(0, max(band_pass_filter.shape[0] - A, 1))
            band_pass_filter[A0 : A0 + A, :] = 0

        filters = band_pass_filter.view(self.out_channels, 1, self.kernel_size)
        return F.conv1d(
            x, filters,
            stride=self.stride, padding=self.padding,
            dilation=self.dilation, bias=None, groups=1,
        )


# ============================================================
# Residual Block (2D CNN Encoder)
# ============================================================

class ResidualBlock(nn.Module):
    """2D residual block for AASIST encoder."""

    def __init__(self, nb_filts: list[int], first: bool = False) -> None:
        super().__init__()
        self.first = first

        if not self.first:
            self.bn1 = nn.BatchNorm2d(num_features=nb_filts[0])
        self.conv1 = nn.Conv2d(
            in_channels=nb_filts[0], out_channels=nb_filts[1],
            kernel_size=(2, 3), padding=(1, 1), stride=1,
        )
        self.selu = nn.SELU(inplace=True)
        self.bn2 = nn.BatchNorm2d(num_features=nb_filts[1])
        self.conv2 = nn.Conv2d(
            in_channels=nb_filts[1], out_channels=nb_filts[1],
            kernel_size=(2, 3), padding=(0, 1), stride=1,
        )

        self.downsample = nb_filts[0] != nb_filts[1]
        if self.downsample:
            self.conv_downsample = nn.Conv2d(
                in_channels=nb_filts[0], out_channels=nb_filts[1],
                padding=(0, 1), kernel_size=(1, 3), stride=1,
            )
        self.mp = nn.MaxPool2d((1, 3))

    def forward(self, x: Tensor) -> Tensor:
        identity = x
        if not self.first:
            out = self.selu(self.bn1(x))
        else:
            out = x
        out = self.conv1(out)
        out = self.selu(self.bn2(out))
        out = self.conv2(out)
        if self.downsample:
            identity = self.conv_downsample(identity)
        out = out + identity
        return self.mp(out)


# ============================================================
# Graph Attention Layer
# ============================================================

class GraphAttentionLayer(nn.Module):
    """Single-type graph attention for spectral or temporal nodes."""

    def __init__(self, in_dim: int, out_dim: int, temperature: float = 1.0) -> None:
        super().__init__()
        self.att_proj = nn.Linear(in_dim, out_dim)
        self.att_weight = nn.Parameter(torch.empty(out_dim, 1))
        nn.init.xavier_normal_(self.att_weight)

        self.proj_with_att = nn.Linear(in_dim, out_dim)
        self.proj_without_att = nn.Linear(in_dim, out_dim)
        self.bn = nn.BatchNorm1d(out_dim)
        self.input_drop = nn.Dropout(p=0.2)
        self.act = nn.SELU(inplace=True)
        self.temp = temperature

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: (batch, n_nodes, in_dim)
        Returns:
            (batch, n_nodes, out_dim)
        """
        x = self.input_drop(x)

        # Pairwise multiplication → attention map
        nb_nodes = x.size(1)
        x_i = x.unsqueeze(2).expand(-1, -1, nb_nodes, -1)
        x_j = x_i.transpose(1, 2)
        att_map = torch.tanh(self.att_proj(x_i * x_j))
        att_map = torch.matmul(att_map, self.att_weight) / self.temp
        att_map = F.softmax(att_map, dim=-2)

        # Project
        x_out = self.proj_with_att(torch.matmul(att_map.squeeze(-1), x))
        x_out = x_out + self.proj_without_att(x)

        # BatchNorm
        orig_shape = x_out.size()
        x_out = self.bn(x_out.reshape(-1, orig_shape[-1])).reshape(orig_shape)
        return self.act(x_out)


# ============================================================
# Heterogeneous Graph Attention Layer
# ============================================================

class HtrgGraphAttentionLayer(nn.Module):
    """Heterogeneous graph attention for spectral+temporal nodes with master node."""

    def __init__(self, in_dim: int, out_dim: int, temperature: float = 1.0) -> None:
        super().__init__()
        self.proj_type1 = nn.Linear(in_dim, in_dim)
        self.proj_type2 = nn.Linear(in_dim, in_dim)

        self.att_proj = nn.Linear(in_dim, out_dim)
        self.att_projM = nn.Linear(in_dim, out_dim)

        self.att_weight11 = nn.Parameter(torch.empty(out_dim, 1))
        self.att_weight22 = nn.Parameter(torch.empty(out_dim, 1))
        self.att_weight12 = nn.Parameter(torch.empty(out_dim, 1))
        self.att_weightM = nn.Parameter(torch.empty(out_dim, 1))
        for w in [self.att_weight11, self.att_weight22, self.att_weight12, self.att_weightM]:
            nn.init.xavier_normal_(w)

        self.proj_with_att = nn.Linear(in_dim, out_dim)
        self.proj_without_att = nn.Linear(in_dim, out_dim)
        self.proj_with_attM = nn.Linear(in_dim, out_dim)
        self.proj_without_attM = nn.Linear(in_dim, out_dim)

        self.bn = nn.BatchNorm1d(out_dim)
        self.input_drop = nn.Dropout(p=0.2)
        self.act = nn.SELU(inplace=True)
        self.temp = temperature

    def forward(
        self, x1: Tensor, x2: Tensor, master: Tensor | None = None
    ) -> tuple[Tensor, Tensor, Tensor]:
        """
        Args:
            x1: Temporal nodes (batch, n_T, dim)
            x2: Spectral nodes (batch, n_S, dim)
            master: Master node (batch, 1, dim) or None

        Returns:
            Tuple of (x1_out, x2_out, master_out).
        """
        n1 = x1.size(1)
        n2 = x2.size(1)

        x1 = self.proj_type1(x1)
        x2 = self.proj_type2(x2)
        x = torch.cat([x1, x2], dim=1)

        if master is None:
            master = torch.mean(x, dim=1, keepdim=True)

        x = self.input_drop(x)

        # Pairwise attention
        nb_nodes = x.size(1)
        x_i = x.unsqueeze(2).expand(-1, -1, nb_nodes, -1)
        x_j = x_i.transpose(1, 2)
        att_map = torch.tanh(self.att_proj(x_i * x_j))

        # Type-specific attention weights
        att_board = torch.zeros_like(att_map[:, :, :, 0]).unsqueeze(-1)
        att_board[:, :n1, :n1, :] = torch.matmul(att_map[:, :n1, :n1, :], self.att_weight11)
        att_board[:, n1:, n1:, :] = torch.matmul(att_map[:, n1:, n1:, :], self.att_weight22)
        att_board[:, :n1, n1:, :] = torch.matmul(att_map[:, :n1, n1:, :], self.att_weight12)
        att_board[:, n1:, :n1, :] = torch.matmul(att_map[:, n1:, :n1, :], self.att_weight12)

        att_map = F.softmax(att_board / self.temp, dim=-2)

        # Master node update
        master_att = torch.tanh(self.att_projM(x * master))
        master_att = torch.matmul(master_att, self.att_weightM) / self.temp
        master_att = F.softmax(master_att, dim=-2)
        master = (
            self.proj_with_attM(torch.matmul(master_att.squeeze(-1).unsqueeze(1), x))
            + self.proj_without_attM(master)
        )

        # Node projection
        x = self.proj_with_att(torch.matmul(att_map.squeeze(-1), x)) + self.proj_without_att(x)

        # BatchNorm
        orig_shape = x.size()
        x = self.bn(x.reshape(-1, orig_shape[-1])).reshape(orig_shape)
        x = self.act(x)

        return x[:, :n1, :], x[:, n1:, :], master


# ============================================================
# Graph Pooling
# ============================================================

class GraphPool(nn.Module):
    """Top-k graph pooling with learned scoring."""

    def __init__(self, k: float, in_dim: int, p: float) -> None:
        super().__init__()
        self.k = k
        self.sigmoid = nn.Sigmoid()
        self.proj = nn.Linear(in_dim, 1)
        self.drop = nn.Dropout(p=p) if p > 0 else nn.Identity()

    def forward(self, h: Tensor) -> Tensor:
        """
        Args:
            h: (batch, n_nodes, dim)
        Returns:
            (batch, n_nodes', dim) where n_nodes' = floor(n_nodes * k)
        """
        Z = self.drop(h)
        scores = self.sigmoid(self.proj(Z))
        _, n_nodes, n_feat = h.size()
        n_keep = max(int(n_nodes * self.k), 1)
        _, idx = torch.topk(scores, n_keep, dim=1)
        idx = idx.expand(-1, -1, n_feat)
        h = h * scores
        return torch.gather(h, 1, idx)


# ============================================================
# AASIST Model
# ============================================================

class AASIST(nn.Module):
    """AASIST: Audio Anti-Spoofing using Integrated Spectro-Temporal
    Graph Attention Networks.

    Target: ~0.30M params, ~8.9G MACs.

    Args:
        sinc_channels: Number of sinc filter channels (default 70).
        sinc_kernel: Sinc filter kernel size (default 128).
        filts: Channel configurations for encoder blocks.
        gat_dims: Dimensions for GAT layers.
        pool_ratios: Pooling ratios for graph pooling.
        temperatures: Temperature values for attention scaling.
        sample_rate: Audio sample rate in Hz.
    """

    def __init__(
        self,
        sinc_channels: int = 70,
        sinc_kernel: int = 128,
        filts: list | None = None,
        gat_dims: list[int] | None = None,
        pool_ratios: list[float] | None = None,
        temperatures: list[float] | None = None,
        sample_rate: int = 16000,
    ) -> None:
        super().__init__()

        # Defaults from AASIST config
        if filts is None:
            filts = [sinc_channels, [1, 32], [32, 32], [32, 64], [64, 64]]
        if gat_dims is None:
            gat_dims = [64, 32]
        if pool_ratios is None:
            pool_ratios = [0.5, 0.7, 0.5, 0.5]
        if temperatures is None:
            temperatures = [2.0, 2.0, 100.0, 100.0]

        # ── SincConv frontend ──
        self.sinc_conv = SincConv(
            out_channels=filts[0],
            kernel_size=sinc_kernel,
            sample_rate=sample_rate,
        )
        self.first_bn = nn.BatchNorm2d(num_features=1)
        self.selu = nn.SELU(inplace=True)
        self.drop = nn.Dropout(0.5, inplace=True)
        self.drop_way = nn.Dropout(0.2, inplace=True)

        # ── 2D CNN encoder ──
        self.encoder = nn.Sequential(
            ResidualBlock(nb_filts=filts[1], first=True),
            ResidualBlock(nb_filts=filts[2]),
            ResidualBlock(nb_filts=filts[3]),
            ResidualBlock(nb_filts=filts[4]),
            ResidualBlock(nb_filts=filts[4]),
            ResidualBlock(nb_filts=filts[4]),
        )

        # ── Positional embedding for spectral nodes ──
        self.pos_S = nn.Parameter(torch.randn(1, 23, filts[-1][-1]))

        # ── Master nodes ──
        self.master1 = nn.Parameter(torch.randn(1, 1, gat_dims[0]))
        self.master2 = nn.Parameter(torch.randn(1, 1, gat_dims[0]))

        # ── GAT layers ──
        self.GAT_layer_S = GraphAttentionLayer(
            filts[-1][-1], gat_dims[0], temperature=temperatures[0]
        )
        self.GAT_layer_T = GraphAttentionLayer(
            filts[-1][-1], gat_dims[0], temperature=temperatures[1]
        )

        # ── Heterogeneous GAT (two paths) ──
        self.HtrgGAT_layer_ST11 = HtrgGraphAttentionLayer(
            gat_dims[0], gat_dims[1], temperature=temperatures[2]
        )
        self.HtrgGAT_layer_ST12 = HtrgGraphAttentionLayer(
            gat_dims[1], gat_dims[1], temperature=temperatures[2]
        )
        self.HtrgGAT_layer_ST21 = HtrgGraphAttentionLayer(
            gat_dims[0], gat_dims[1], temperature=temperatures[2]
        )
        self.HtrgGAT_layer_ST22 = HtrgGraphAttentionLayer(
            gat_dims[1], gat_dims[1], temperature=temperatures[2]
        )

        # ── Graph pooling ──
        self.pool_S = GraphPool(pool_ratios[0], gat_dims[0], 0.3)
        self.pool_T = GraphPool(pool_ratios[1], gat_dims[0], 0.3)
        self.pool_hS1 = GraphPool(pool_ratios[2], gat_dims[1], 0.3)
        self.pool_hT1 = GraphPool(pool_ratios[2], gat_dims[1], 0.3)
        self.pool_hS2 = GraphPool(pool_ratios[2], gat_dims[1], 0.3)
        self.pool_hT2 = GraphPool(pool_ratios[2], gat_dims[1], 0.3)

        # ── Output ──
        self.out_layer = nn.Linear(5 * gat_dims[1], 2)

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass.

        Args:
            x: Raw waveform (batch, 64000).

        Returns:
            Logits (batch, 2).
        """
        # SincConv frontend
        x = x.unsqueeze(1)  # (B, 1, T)
        x = self.sinc_conv(x)  # (B, 70, T')
        x = x.unsqueeze(1)  # (B, 1, 70, T')
        x = F.max_pool2d(torch.abs(x), (3, 3))  # (B, 1, 23, T'')
        x = self.first_bn(x)
        x = self.selu(x)

        # 2D CNN encoder → (B, C, F, T_enc)
        e = self.encoder(x)

        # Spectral GAT
        e_S, _ = torch.max(torch.abs(e), dim=3)  # max along time → (B, C, F)
        e_S = e_S.transpose(1, 2) + self.pos_S  # (B, F, C)
        gat_S = self.GAT_layer_S(e_S)
        out_S = self.pool_S(gat_S)

        # Temporal GAT
        e_T, _ = torch.max(torch.abs(e), dim=2)  # max along freq → (B, C, T)
        e_T = e_T.transpose(1, 2)  # (B, T, C)
        gat_T = self.GAT_layer_T(e_T)
        out_T = self.pool_T(gat_T)

        # Heterogeneous inference path 1
        out_T1, out_S1, master1 = self.HtrgGAT_layer_ST11(
            out_T, out_S, master=self.master1
        )
        out_S1 = self.pool_hS1(out_S1)
        out_T1 = self.pool_hT1(out_T1)
        out_T_aug, out_S_aug, master_aug = self.HtrgGAT_layer_ST12(
            out_T1, out_S1, master=master1
        )
        out_T1 = out_T1 + out_T_aug
        out_S1 = out_S1 + out_S_aug
        master1 = master1 + master_aug

        # Heterogeneous inference path 2
        out_T2, out_S2, master2 = self.HtrgGAT_layer_ST21(
            out_T, out_S, master=self.master2
        )
        out_S2 = self.pool_hS2(out_S2)
        out_T2 = self.pool_hT2(out_T2)
        out_T_aug, out_S_aug, master_aug = self.HtrgGAT_layer_ST22(
            out_T2, out_S2, master=master2
        )
        out_T2 = out_T2 + out_T_aug
        out_S2 = out_S2 + out_S_aug
        master2 = master2 + master_aug

        # Dropout on paths
        out_T1 = self.drop_way(out_T1)
        out_T2 = self.drop_way(out_T2)
        out_S1 = self.drop_way(out_S1)
        out_S2 = self.drop_way(out_S2)
        master1 = self.drop_way(master1)
        master2 = self.drop_way(master2)

        # Max fusion of two paths
        out_T = torch.max(out_T1, out_T2)
        out_S = torch.max(out_S1, out_S2)
        master = torch.max(master1, master2)

        # Aggregate features
        T_max, _ = torch.max(torch.abs(out_T), dim=1)
        T_avg = torch.mean(out_T, dim=1)
        S_max, _ = torch.max(torch.abs(out_S), dim=1)
        S_avg = torch.mean(out_S, dim=1)

        last_hidden = torch.cat(
            [T_max, T_avg, S_max, S_avg, master.squeeze(1)], dim=1
        )
        last_hidden = self.drop(last_hidden)
        output = self.out_layer(last_hidden)

        return output


# ============================================================
# Smoke Test
# ============================================================

if __name__ == "__main__":
    print("=" * 60)
    print("AASIST — Smoke Test")
    print("=" * 60)

    model = AASIST()
    model.eval()

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {n_params:,}")

    x = torch.randn(2, 64000)
    with torch.no_grad():
        out = model(x)
    print(f"Input shape:  {x.shape}")
    print(f"Output shape: {out.shape}")
    assert out.shape == (2, 2), f"Expected (2, 2), got {out.shape}"

    print("\n✅ AASIST smoke test passed!")
