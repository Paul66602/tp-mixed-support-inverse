#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PrecipBudgetNet: a controlled encoder-decoder backbone for the
support-explicit constrained inverse problem.

Input:
    (B, 4, H_f, W_f) with channels
        W_t0, W_t1, dW_3h, I_c(P_c_obs)

Output:
    P_hat:     (B, 1, H_f, W_f) on the fine grid
    R_app_hat: (B, 1, H_c, W_c) on the coarse grid when coarse_shape is given,
               otherwise on the bottleneck grid for patch-mode fallback.

The backbone is intentionally modest. The scientific novelty of the project
should come from the operator-aware inverse formulation and validation, not
from claiming that the network topology itself is new.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def _make_norm(norm_type: str, num_channels: int, n_groups: int = 8) -> nn.Module:
    """Factory for normalization layers: 'batch', 'group', or 'none'."""
    if norm_type == "batch":
        return nn.BatchNorm2d(num_channels)
    elif norm_type == "group":
        return nn.GroupNorm(min(n_groups, num_channels), num_channels)
    elif norm_type == "none":
        return nn.Identity()
    else:
        raise ValueError(f"Unknown norm_type: {norm_type!r}")


class ResBlock(nn.Module):
    """Residual block with two 3x3 convolutions and a skip path."""

    def __init__(self, in_ch: int, out_ch: int, dropout: float = 0.0,
                 norm_type: str = "batch", n_groups: int = 8):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False)
        self.bn1 = _make_norm(norm_type, out_ch, n_groups)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False)
        self.bn2 = _make_norm(norm_type, out_ch, n_groups)
        self.drop = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()
        self.skip = (
            nn.Conv2d(in_ch, out_ch, 1, bias=False)
            if in_ch != out_ch
            else nn.Identity()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.skip(x)
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.drop(out)
        return F.relu(out + residual)


class DownBlock(nn.Module):
    """Encoder block: residual block followed by stride-2 downsampling."""

    def __init__(self, in_ch: int, out_ch: int, dropout: float = 0.0,
                 norm_type: str = "batch", n_groups: int = 8):
        super().__init__()
        self.res = ResBlock(in_ch, out_ch, dropout=dropout,
                            norm_type=norm_type, n_groups=n_groups)
        self.down = nn.Conv2d(out_ch, out_ch, 3, stride=2, padding=1, bias=False)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        skip = self.res(x)
        down = self.down(skip)
        return down, skip


class UpBlock(nn.Module):
    """Decoder block: learned transposed-convolution upsample, concat skip,
    then apply the residual block."""

    def __init__(self, in_ch: int, skip_ch: int, out_ch: int,
                 dropout: float = 0.0, norm_type: str = "batch",
                 n_groups: int = 8):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, in_ch, 2, stride=2, bias=False)
        self.res = ResBlock(in_ch + skip_ch, out_ch, dropout=dropout,
                            norm_type=norm_type, n_groups=n_groups)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        up = self.up(x)
        if up.shape[2:] != skip.shape[2:]:
            up = F.interpolate(up, size=skip.shape[2:], mode="nearest")
        cat = torch.cat([up, skip], dim=1)
        return self.res(cat)


class PrecipBudgetNet(nn.Module):
    """
    Residual encoder-decoder for the support-explicit precipitation inversion.

    Parameters
    ----------
    in_channels:
        Number of input channels. Default is 4:
        `W_t0`, `W_t1`, `dW_3h`, `I_c(P_c_obs)`.
    base_channels:
        Channel count at the first encoder level.
    n_levels:
        Number of encoder / decoder levels.
    dropout:
        Dropout rate used inside residual blocks.
    coarse_shape:
        Output coarse-grid shape `(H_c, W_c)` for the `R_app` head. When `None`,
        `R_app` is returned at bottleneck resolution for patch-mode fallback.
    r_parameterization:
        `"coarse"` keeps the original coarse-support residual head used by the
        main formulation. `"fine"` predicts a fine-grid residual directly for
        the A6 pointwise-budget ablation.
    """

    def __init__(
        self,
        in_channels: int = 4,
        base_channels: int = 32,
        n_levels: int = 3,
        dropout: float = 0.1,
        coarse_shape: tuple[int, int] | None = None,
        norm_type: str = "batch",
        n_groups: int = 8,
        r_parameterization: str = "coarse",
    ):
        super().__init__()
        self.coarse_shape = coarse_shape
        if r_parameterization not in {"coarse", "fine"}:
            raise ValueError(
                f"Unknown r_parameterization: {r_parameterization!r}"
            )
        self.r_parameterization = r_parameterization

        self.encoders = nn.ModuleList()
        ch = in_channels
        enc_channels: list[int] = []
        for i in range(n_levels):
            out_ch = base_channels * (2**i)
            self.encoders.append(DownBlock(ch, out_ch, dropout=dropout,
                                           norm_type=norm_type,
                                           n_groups=n_groups))
            enc_channels.append(out_ch)
            ch = out_ch

        bottleneck_ch = base_channels * (2**n_levels)
        self.bottleneck = ResBlock(ch, bottleneck_ch, dropout=dropout,
                                   norm_type=norm_type, n_groups=n_groups)
        ch = bottleneck_ch

        self.decoders = nn.ModuleList()
        for i in range(n_levels - 1, -1, -1):
            skip_ch = enc_channels[i]
            out_ch = enc_channels[i]
            self.decoders.append(UpBlock(ch, skip_ch, out_ch, dropout=dropout,
                                         norm_type=norm_type,
                                         n_groups=n_groups))
            ch = out_ch

        self.p_head = nn.Sequential(
            nn.Conv2d(ch, max(ch // 2, 1), 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(max(ch // 2, 1), 1, 1),
        )

        if self.r_parameterization == "coarse":
            r_mid = max(bottleneck_ch // 2, 1)
            self.r_head = nn.Sequential(
                nn.Conv2d(bottleneck_ch, r_mid, 3, padding=1, bias=False),
                _make_norm(norm_type, r_mid, n_groups),
                nn.ReLU(inplace=True),
                nn.Conv2d(r_mid, 1, 1),
            )
        else:
            # A6 should test pointwise-vs-coarse support, not collapse because
            # the fine-R head is dramatically smaller than the coarse-R head.
            # These widths keep the fine head in the same parameter regime as
            # the default coarse head for the main n_levels=3 setting.
            r_fine_mid1 = max(ch * 4, 64)
            r_fine_mid2 = max(ch * 7, 112)
            self.r_head = nn.Sequential(
                nn.Conv2d(ch, r_fine_mid1, 3, padding=1, bias=False),
                _make_norm(norm_type, r_fine_mid1, n_groups),
                nn.ReLU(inplace=True),
                nn.Conv2d(r_fine_mid1, r_fine_mid2, 3, padding=1, bias=False),
                _make_norm(norm_type, r_fine_mid2, n_groups),
                nn.ReLU(inplace=True),
                nn.Conv2d(r_fine_mid2, 1, 1),
            )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return `(P_hat, R_app_hat)`."""
        skips = []
        h = x
        for enc in self.encoders:
            h, skip = enc(h)
            skips.append(skip)

        h = self.bottleneck(h)
        bottleneck_h = h

        for i, dec in enumerate(self.decoders):
            skip = skips[-(i + 1)]
            h = dec(h, skip)

        p_hat = F.softplus(self.p_head(h))
        if self.r_parameterization == "coarse":
            r_raw = self.r_head(bottleneck_h)
            if self.coarse_shape is not None:
                r_app = F.adaptive_avg_pool2d(r_raw, self.coarse_shape)
            else:
                r_app = r_raw
        else:
            r_app = self.r_head(h)
        return p_hat, r_app


# ---------------------------------------------------------------------------
# V2: Analytical Backbone + Constrained Correction
# ---------------------------------------------------------------------------

class GNResBlock(nn.Module):
    """Residual block with GroupNorm (safe for batch_size=1)."""

    def __init__(self, in_ch: int, out_ch: int, n_groups: int = 8,
                 dropout: float = 0.0):
        super().__init__()
        g1 = min(n_groups, out_ch)
        g2 = min(n_groups, out_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False)
        self.gn1 = nn.GroupNorm(g1, out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False)
        self.gn2 = nn.GroupNorm(g2, out_ch)
        self.drop = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()
        self.skip = (
            nn.Conv2d(in_ch, out_ch, 1, bias=False)
            if in_ch != out_ch
            else nn.Identity()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.skip(x)
        out = F.relu(self.gn1(self.conv1(x)))
        out = self.gn2(self.conv2(out))
        out = self.drop(out)
        return F.relu(out + residual)


class GNDownBlock(nn.Module):
    """Encoder block with GroupNorm: res block + stride-2 downsample."""

    def __init__(self, in_ch: int, out_ch: int, n_groups: int = 8,
                 dropout: float = 0.0):
        super().__init__()
        self.res = GNResBlock(in_ch, out_ch, n_groups, dropout)
        self.down = nn.Conv2d(out_ch, out_ch, 3, stride=2, padding=1,
                              bias=False)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        skip = self.res(x)
        down = self.down(skip)
        return down, skip


class GNUpBlock(nn.Module):
    """Decoder block with GroupNorm: transposed-convolution upsample + skip-cat
    + residual block."""

    def __init__(self, in_ch: int, skip_ch: int, out_ch: int,
                 n_groups: int = 8, dropout: float = 0.0):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, in_ch, 2, stride=2, bias=False)
        self.res = GNResBlock(in_ch + skip_ch, out_ch, n_groups, dropout)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        up = self.up(x)
        if up.shape[2:] != skip.shape[2:]:
            up = F.interpolate(up, size=skip.shape[2:], mode="nearest")
        return self.res(torch.cat([up, skip], dim=1))


class AnalyticalCorrectionNet(nn.Module):
    """
    V2 architecture: Analytical Backbone + Learned Constrained Correction.

    Architectural inspiration and attribution:
      - "Analytical step + learned CNN correction" pattern follows MoDL
        (Aggarwal, Mani & Jacob, IEEE TMI 2019).
      - Null-space projection as a structural constraint relates to the
        bilevel-vs-DEQ framework (Mukherjee et al., 2023) where constraining
        the learned regularizer improves generalization at the cost of
        expressivity.
      - GroupNorm choice follows Wu & He (ECCV 2018) for batch_size=1
        compatibility.
      - Learned Primal-Dual (Adler & Oktem, IEEE TMI 2018) provides
        the broader context of learned corrections to analytical iterates
        in inverse problems.

    Instead of learning P_hat from scratch, this model:
      1. Receives P_0 (analytical baseline P) as input channel 5
      2. Receives R_0 (analytical baseline R_app) as a separate tensor
      3. Learns corrections delta_P and delta_R
      4. Projects delta_P onto approximate ker(A_c) so it preserves
         coarse consistency
      5. Returns P_hat = softplus(P_0 + delta_P_proj),
                 R_app = R_0 + delta_R

    Key differences from PrecipBudgetNet (v1):
      - GroupNorm instead of BatchNorm (safe for batch_size=1)
      - Analytical backbone inherited; network only learns corrections
      - Null-space projection ensures corrections don't break coarse consistency
      - Lighter by default (base_channels=16, n_levels=2)

    Parameters
    ----------
    in_channels:
        Number of input channels. Default 5:
        W_t0, W_t1, dW, I_c(P_c_obs), P_0_analytical.
    base_channels:
        Channel count at first encoder level.
    n_levels:
        Number of encoder/decoder levels.
    n_groups:
        Number of groups for GroupNorm.
    dropout:
        Dropout rate inside residual blocks.
    coarse_shape:
        (H_c, W_c) for R_app output and null-space projection.
    fine_shape:
        (H_f, W_f) for null-space projection target size.
    """

    def __init__(
        self,
        in_channels: int = 5,
        base_channels: int = 16,
        n_levels: int = 2,
        n_groups: int = 8,
        dropout: float = 0.1,
        coarse_shape: tuple[int, int] | None = None,
        fine_shape: tuple[int, int] | None = None,
        Ac_scipy=None,
        Ic_scipy=None,
        freeze_delta_r: bool = False,
        disable_nullspace_proj: bool = False,
        disable_p0_input: bool = False,
    ):
        super().__init__()
        self.coarse_shape = coarse_shape
        self.fine_shape = fine_shape
        self.freeze_delta_r = freeze_delta_r
        self.disable_nullspace_proj = disable_nullspace_proj
        self.disable_p0_input = disable_p0_input

        # If P_0 input is disabled, reduce input channels by 1
        if disable_p0_input:
            in_channels = in_channels - 1

        # Exact sparse operators for null-space projection (optional)
        self.exact_projection = (Ac_scipy is not None and Ic_scipy is not None)
        if self.exact_projection:
            Ac_coo = Ac_scipy.tocoo().astype('float32')
            Ic_coo = Ic_scipy.tocoo().astype('float32')
            Ac_idx = torch.LongTensor(
                np.vstack([Ac_coo.row, Ac_coo.col]))
            Ac_val = torch.FloatTensor(Ac_coo.data)
            Ic_idx = torch.LongTensor(
                np.vstack([Ic_coo.row, Ic_coo.col]))
            Ic_val = torch.FloatTensor(Ic_coo.data)
            self.register_buffer(
                '_Ac', torch.sparse_coo_tensor(
                    Ac_idx, Ac_val, Ac_coo.shape).coalesce())
            self.register_buffer(
                '_Ic', torch.sparse_coo_tensor(
                    Ic_idx, Ic_val, Ic_coo.shape).coalesce())

        # Encoder
        self.encoders = nn.ModuleList()
        ch = in_channels
        enc_channels: list[int] = []
        for i in range(n_levels):
            out_ch = base_channels * (2 ** i)
            self.encoders.append(
                GNDownBlock(ch, out_ch, n_groups, dropout))
            enc_channels.append(out_ch)
            ch = out_ch

        bottleneck_ch = base_channels * (2 ** n_levels)
        self.bottleneck = GNResBlock(ch, bottleneck_ch, n_groups, dropout)
        ch = bottleneck_ch

        # Decoder
        self.decoders = nn.ModuleList()
        for i in range(n_levels - 1, -1, -1):
            skip_ch = enc_channels[i]
            out_ch = enc_channels[i]
            self.decoders.append(
                GNUpBlock(ch, skip_ch, out_ch, n_groups, dropout))
            ch = out_ch

        # delta_P head — raw correction, will be projected
        self.dp_head = nn.Sequential(
            nn.Conv2d(ch, max(ch // 2, 1), 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(max(ch // 2, 1), 1, 1),
        )

        # delta_R head — coarse correction from bottleneck
        g_r = min(n_groups, max(bottleneck_ch // 2, 1))
        self.dr_head = nn.Sequential(
            nn.Conv2d(bottleneck_ch, max(bottleneck_ch // 2, 1), 3,
                      padding=1, bias=False),
            nn.GroupNorm(g_r, max(bottleneck_ch // 2, 1)),
            nn.ReLU(inplace=True),
            nn.Conv2d(max(bottleneck_ch // 2, 1), 1, 1),
        )

        # Initialize correction heads with small weights (start near analytical)
        for head in [self.dp_head, self.dr_head]:
            for m in head.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.kaiming_normal_(m.weight, a=0.01)
                    m.weight.data.mul_(0.01)  # scale down
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)

    def _null_space_project(self, delta_P: torch.Tensor) -> torch.Tensor:
        """
        Projection onto ker(A_c):
        delta_P_proj = delta_P - I_c(A_c(delta_P))

        If exact sparse operators were provided at init, uses them.
        Otherwise falls back to adaptive_avg_pool2d + nearest interpolation.
        """
        if self.coarse_shape is None or self.fine_shape is None:
            return delta_P

        if self.exact_projection:
            # Exact sparse projection using registered A_c, I_c
            B = delta_P.shape[0]
            H_f, W_f = self.fine_shape
            needs_squeeze = (delta_P.dim() == 3)
            if needs_squeeze:
                dp_flat = delta_P.reshape(B, -1)  # (B, n_fine)
            else:
                dp_flat = delta_P.reshape(B, -1)  # (B, n_fine)
            # A_c @ dp^T : (n_coarse, n_fine) @ (n_fine, B) -> (n_coarse, B)
            coarse = torch.sparse.mm(self._Ac, dp_flat.T)
            # I_c @ coarse : (n_fine, n_coarse) @ (n_coarse, B) -> (n_fine, B)
            lifted = torch.sparse.mm(self._Ic, coarse)
            dp_proj = dp_flat - lifted.T  # (B, n_fine)
            return dp_proj.reshape(B, 1, H_f, W_f)
        else:
            # Approximate projection via pooling/interpolation
            needs_squeeze = (delta_P.dim() == 3)
            if needs_squeeze:
                delta_P = delta_P.unsqueeze(1)
            coarse_avg = F.adaptive_avg_pool2d(delta_P, self.coarse_shape)
            lifted_back = F.interpolate(
                coarse_avg, size=self.fine_shape, mode="nearest")
            out = delta_P - lifted_back
            return out.squeeze(1) if needs_squeeze else out

    def forward(
        self,
        x: torch.Tensor,
        R_0: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Parameters
        ----------
        x : (B, 5, H_f, W_f)
            Channels: W_t0, W_t1, dW, Pc_lifted, P_0_analytical.
            If disable_p0_input, x is (B, 4, H_f, W_f) but P_0 is still
            passed as the last channel of the ORIGINAL 5-ch tensor for
            assembly. The caller must pass the full 5-ch x; this method
            strips P_0 before the encoder when disable_p0_input is set.
        R_0 : (B, H_c, W_c) or (B, 1, H_c, W_c), optional
            Analytical coarse baseline. If None, delta_R is used directly.

        Returns
        -------
        P_hat : (B, 1, H_f, W_f)
        R_app : (B, 1, H_c, W_c) or (B, 1, H_bottleneck, W_bottleneck)
        """
        # Extract P_0 from the last input channel (always needed for assembly)
        P_0 = x[:, -1:, :, :]  # (B, 1, H_f, W_f)

        # Optionally strip P_0 from encoder input
        if self.disable_p0_input:
            h = x[:, :-1, :, :]  # (B, 4, H_f, W_f)
        else:
            h = x

        # Encoder
        skips = []
        for enc in self.encoders:
            h, skip = enc(h)
            skips.append(skip)

        h = self.bottleneck(h)

        # delta_R from bottleneck
        dr_raw = self.dr_head(h)
        if self.coarse_shape is not None:
            delta_R = F.adaptive_avg_pool2d(dr_raw, self.coarse_shape)
        else:
            delta_R = dr_raw

        # Freeze delta_R: zero out the correction (E5 ablation)
        if self.freeze_delta_r:
            delta_R = torch.zeros_like(delta_R)

        # Decoder
        for i, dec in enumerate(self.decoders):
            skip = skips[-(i + 1)]
            h = dec(h, skip)

        # delta_P with optional null-space projection
        dp_raw = self.dp_head(h)  # (B, 1, H_f, W_f)
        if self.disable_nullspace_proj:
            dp_proj = dp_raw
        else:
            dp_proj = self._null_space_project(dp_raw)

        # Final assembly: analytical + learned correction
        P_hat = F.softplus(P_0 + dp_proj)

        if R_0 is not None:
            R_0_4d = R_0.unsqueeze(1) if R_0.dim() == 3 else R_0
            R_app = R_0_4d + delta_R
        else:
            R_app = delta_R

        return P_hat, R_app


def count_parameters(model: nn.Module) -> int:
    """Count trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    model_full = PrecipBudgetNet(
        in_channels=4,
        base_channels=32,
        n_levels=3,
        dropout=0.1,
        coarse_shape=(150, 350),
    )
    print("=== Example: explicit coarse-shape mode ===")
    print(f"Parameters: {count_parameters(model_full):,}")
    x = torch.randn(1, 4, 128, 256)
    p_hat, r_app = model_full(x)
    print(f"Input:  {x.shape}")
    print(f"P_hat:  {p_hat.shape}")
    print(f"R_app:  {r_app.shape}")

    model_patch = PrecipBudgetNet(
        in_channels=4,
        base_channels=32,
        n_levels=3,
        dropout=0.1,
        coarse_shape=None,
    )
    print("\n=== Example: patch-mode fallback ===")
    print(f"Parameters: {count_parameters(model_patch):,}")
    x_patch = torch.randn(8, 4, 64, 64)
    p_patch, r_patch = model_patch(x_patch)
    print(f"Input:  {x_patch.shape}")
    print(f"P_hat:  {p_patch.shape}")
    print(f"R_app:  {r_patch.shape}  (bottleneck resolution)")

    # --- V2: AnalyticalCorrectionNet ---
    print("\n=== V2: AnalyticalCorrectionNet ===")
    fine_shape = (201, 501)
    coarse_shape = (40, 100)
    model_v2 = AnalyticalCorrectionNet(
        in_channels=5,
        base_channels=16,
        n_levels=2,
        coarse_shape=coarse_shape,
        fine_shape=fine_shape,
    )
    print(f"Parameters: {count_parameters(model_v2):,}")
    x_v2 = torch.randn(1, 5, fine_shape[0], fine_shape[1])
    R_0 = torch.randn(1, coarse_shape[0], coarse_shape[1]) * 0.1
    p_v2, r_v2 = model_v2(x_v2, R_0)
    print(f"Input:  {x_v2.shape}")
    print(f"R_0:    {R_0.shape}")
    print(f"P_hat:  {p_v2.shape}")
    print(f"R_app:  {r_v2.shape}")
    print(f"P_hat range: [{p_v2.min():.4f}, {p_v2.max():.4f}]")
    print(f"delta_R magnitude: {(r_v2.squeeze() - R_0).abs().mean():.6f}"
          f"  (should be ~0 at init)")
