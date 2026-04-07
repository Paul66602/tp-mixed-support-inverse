#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Budget-constrained loss functions.

Two implementations are provided:

  - BudgetLossExact:
      Uses sparse support operators A_c and I_c and should be the default for
      full-domain training / evaluation. This is the faithful implementation of
      the support-explicit formulation.

  - BudgetLossConv:
      Uses avg_pool2d + nearest-neighbor interpolation as an engineering
      approximation for patch-based training. This is convenient on GPU, but it
      is not operator-exact on the TPW longitude grid because the fine/coarse
      ratio is not perfectly nested.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def _masked_mean(x: torch.Tensor, mask: torch.Tensor | None) -> torch.Tensor:
    """Mean over valid entries only."""
    if mask is None:
        return x.mean()
    mask_f = mask.float()
    denom = mask_f.sum().clamp(min=1.0)
    return (x * mask_f).sum() / denom


def _smoothness_loss(
    field: torch.Tensor,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    Squared-gradient smoothness on a coarse 2D field.

    field: (B, H, W) or (H, W)
    mask:  same shape as field, boolean valid mask
    """
    if field.dim() == 2:
        field = field.unsqueeze(0)
        if mask is not None:
            mask = mask.unsqueeze(0)

    grad_lat = field[..., 1:, :] - field[..., :-1, :]
    grad_lon = field[..., :, 1:] - field[..., :, :-1]

    if mask is None:
        return (grad_lat ** 2).mean() + (grad_lon ** 2).mean()

    mask_lat = mask[..., 1:, :] & mask[..., :-1, :]
    mask_lon = mask[..., :, 1:] & mask[..., :, :-1]

    loss_lat = _masked_mean(grad_lat ** 2, mask_lat)
    loss_lon = _masked_mean(grad_lon ** 2, mask_lon)
    return loss_lat + loss_lon


class BudgetLossExact(nn.Module):
    """
    Exact operator loss using sparse operators A_c and I_c.

    This should be used for full-domain training/evaluation where the input
    shapes match the precomputed support operators exactly.
    """

    def __init__(
        self,
        Ac_csr,
        Ic_csr,
        fine_shape: tuple[int, int],
        coarse_shape: tuple[int, int],
        lambda_W: float = 1.0,
        lambda_Pc: float = 10.0,
        lambda_R: float = 0.01,
        alpha_smooth: float = 0.1,
    ):
        super().__init__()
        self.fine_shape = tuple(fine_shape)
        self.coarse_shape = tuple(coarse_shape)
        self.lambda_W = lambda_W
        self.lambda_Pc = lambda_Pc
        self.lambda_R = lambda_R
        self.alpha_smooth = alpha_smooth

        self._Ac_cache: dict[str, torch.Tensor] = {}
        self._Ic_cache: dict[str, torch.Tensor] = {}

        self._register_sparse("Ac", Ac_csr)
        self._register_sparse("Ic", Ic_csr)

    def _register_sparse(self, prefix: str, csr) -> None:
        coo = csr.tocoo()
        indices = np.vstack([coo.row, coo.col]).astype(np.int64)
        values = coo.data.astype(np.float32)
        self.register_buffer(
            f"{prefix}_indices", torch.from_numpy(indices), persistent=False
        )
        self.register_buffer(
            f"{prefix}_values", torch.from_numpy(values), persistent=False
        )
        setattr(self, f"{prefix}_shape", tuple(coo.shape))

    def _get_sparse(self, prefix: str, device: torch.device) -> torch.Tensor:
        cache = self._Ac_cache if prefix == "Ac" else self._Ic_cache
        key = str(device)
        if key not in cache:
            indices = getattr(self, f"{prefix}_indices").to(device)
            values = getattr(self, f"{prefix}_values").to(device)
            shape = getattr(self, f"{prefix}_shape")
            cache[key] = torch.sparse_coo_tensor(
                indices, values, size=shape, device=device
            ).coalesce()
        return cache[key]

    @staticmethod
    def _spmm_batch(mat: torch.Tensor, dense: torch.Tensor) -> torch.Tensor:
        """
        Batch sparse-dense multiplication.

        mat:   (M, N) sparse
        dense: (B, N) dense
        out:   (B, M)
        """
        return torch.sparse.mm(mat, dense.T).T

    def forward(
        self,
        P_hat: torch.Tensor,
        R_app_hat: torch.Tensor,
        dW_obs: torch.Tensor,
        P_c_obs: torch.Tensor,
        fine_mask: torch.Tensor | None = None,
        coarse_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        H_f, W_f = P_hat.shape[-2], P_hat.shape[-1]
        H_c, W_c = P_c_obs.shape[-2], P_c_obs.shape[-1]
        if (H_f, W_f) != self.fine_shape:
            raise ValueError(
                f"BudgetLossExact expects fine shape {self.fine_shape}, "
                f"got {(H_f, W_f)}. Use BudgetLossConv for patch mode."
            )
        if (H_c, W_c) != self.coarse_shape:
            raise ValueError(
                f"BudgetLossExact expects coarse shape {self.coarse_shape}, "
                f"got {(H_c, W_c)}."
            )

        device = P_hat.device
        Ac = self._get_sparse("Ac", device)
        Ic = self._get_sparse("Ic", device)

        P_sq = P_hat.squeeze(1) if P_hat.dim() == 4 else P_hat
        R_sq = R_app_hat.squeeze(1) if R_app_hat.dim() == 4 else R_app_hat

        batch = P_sq.shape[0]
        P_flat = P_sq.reshape(batch, -1)
        R_flat = R_sq.reshape(batch, -1)

        P_coarse = self._spmm_batch(Ac, P_flat).reshape(batch, H_c, W_c)
        R_fine = self._spmm_batch(Ic, R_flat).reshape(batch, H_f, W_f)
        dW_pred = R_fine - P_sq

        residual_W = (dW_obs - dW_pred) ** 2
        L_W = _masked_mean(residual_W, fine_mask)

        residual_Pc = (P_coarse - P_c_obs) ** 2
        L_Pc = _masked_mean(residual_Pc, coarse_mask)

        L_R_amp = _masked_mean(R_sq ** 2, coarse_mask)
        L_R_smooth = _smoothness_loss(R_sq, coarse_mask)
        L_R = L_R_amp + self.alpha_smooth * L_R_smooth

        loss = self.lambda_W * L_W + self.lambda_Pc * L_Pc + self.lambda_R * L_R
        return loss, {
            "L_W": float(L_W.item()),
            "L_Pc": float(L_Pc.item()),
            "L_R": float(L_R.item()),
            "total": float(loss.item()),
        }


class BudgetLossPointwise(nn.Module):
    """
    Pointwise-budget ablation loss.

    This is the A6 strong-form comparator to the main support-aware
    formulation. The model predicts a fine-grid residual field directly and
    enforces

        dW = R_fine - P

    pointwise on the fine grid, while still using the exact coarse operator
    A_c for the precipitation-support term.
    """

    def __init__(
        self,
        Ac_csr,
        fine_shape: tuple[int, int],
        coarse_shape: tuple[int, int],
        lambda_W: float = 1.0,
        lambda_Pc: float = 10.0,
        lambda_R: float = 0.01,
        alpha_smooth: float = 0.1,
    ):
        super().__init__()
        self.fine_shape = tuple(fine_shape)
        self.coarse_shape = tuple(coarse_shape)
        self.lambda_W = lambda_W
        self.lambda_Pc = lambda_Pc
        self.lambda_R = lambda_R
        self.alpha_smooth = alpha_smooth

        self._Ac_cache: dict[str, torch.Tensor] = {}
        coo = Ac_csr.tocoo()
        indices = np.vstack([coo.row, coo.col]).astype(np.int64)
        values = coo.data.astype(np.float32)
        self.register_buffer(
            "Ac_indices", torch.from_numpy(indices), persistent=False
        )
        self.register_buffer(
            "Ac_values", torch.from_numpy(values), persistent=False
        )
        self.Ac_shape = tuple(coo.shape)

    def _get_Ac(self, device: torch.device) -> torch.Tensor:
        key = str(device)
        if key not in self._Ac_cache:
            self._Ac_cache[key] = torch.sparse_coo_tensor(
                self.Ac_indices.to(device),
                self.Ac_values.to(device),
                size=self.Ac_shape,
                device=device,
            ).coalesce()
        return self._Ac_cache[key]

    @staticmethod
    def _spmm_batch(mat: torch.Tensor, dense: torch.Tensor) -> torch.Tensor:
        return torch.sparse.mm(mat, dense.T).T

    def forward(
        self,
        P_hat: torch.Tensor,
        R_fine_hat: torch.Tensor,
        dW_obs: torch.Tensor,
        P_c_obs: torch.Tensor,
        fine_mask: torch.Tensor | None = None,
        coarse_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        H_f, W_f = P_hat.shape[-2], P_hat.shape[-1]
        H_c, W_c = P_c_obs.shape[-2], P_c_obs.shape[-1]
        if (H_f, W_f) != self.fine_shape:
            raise ValueError(
                f"BudgetLossPointwise expects fine shape {self.fine_shape}, "
                f"got {(H_f, W_f)}."
            )
        if (H_c, W_c) != self.coarse_shape:
            raise ValueError(
                f"BudgetLossPointwise expects coarse shape {self.coarse_shape}, "
                f"got {(H_c, W_c)}."
            )

        Ac = self._get_Ac(P_hat.device)
        P_sq = P_hat.squeeze(1) if P_hat.dim() == 4 else P_hat
        R_sq = R_fine_hat.squeeze(1) if R_fine_hat.dim() == 4 else R_fine_hat

        dW_pred = R_sq - P_sq
        L_W = _masked_mean((dW_obs - dW_pred) ** 2, fine_mask)

        batch = P_sq.shape[0]
        P_flat = P_sq.reshape(batch, -1)
        P_coarse = self._spmm_batch(Ac, P_flat).reshape(batch, H_c, W_c)
        L_Pc = _masked_mean((P_coarse - P_c_obs) ** 2, coarse_mask)

        L_R_amp = _masked_mean(R_sq ** 2, fine_mask)
        L_R_smooth = _smoothness_loss(R_sq, fine_mask)
        L_R = L_R_amp + self.alpha_smooth * L_R_smooth

        loss = self.lambda_W * L_W + self.lambda_Pc * L_Pc + self.lambda_R * L_R
        return loss, {
            "L_W": float(L_W.item()),
            "L_Pc": float(L_Pc.item()),
            "L_R": float(L_R.item()),
            "total": float(loss.item()),
        }


class BudgetLossConv(nn.Module):
    """
    Approximate budget-constrained loss using PyTorch image ops.

    Intended for patch-based training, where exact sparse operators are not
    convenient to crop and batch. For full-domain training, prefer
    BudgetLossExact.
    """

    def __init__(
        self,
        pool_size: int = 5,
        lambda_W: float = 1.0,
        lambda_Pc: float = 10.0,
        lambda_R: float = 0.01,
        alpha_smooth: float = 0.1,
    ):
        super().__init__()
        self.pool_size = pool_size
        self.lambda_W = lambda_W
        self.lambda_Pc = lambda_Pc
        self.lambda_R = lambda_R
        self.alpha_smooth = alpha_smooth

    def _avg_pool_coarse(self, x_fine: torch.Tensor) -> torch.Tensor:
        """Approximate A_c via average pooling."""
        needs_squeeze = False
        if x_fine.dim() == 3:
            x_fine = x_fine.unsqueeze(1)
            needs_squeeze = True
        out = F.avg_pool2d(
            x_fine, kernel_size=self.pool_size, stride=self.pool_size
        )
        return out.squeeze(1) if needs_squeeze else out

    def _upsample_fine(
        self,
        x_coarse: torch.Tensor,
        target_size: tuple[int, int],
    ) -> torch.Tensor:
        """Approximate I_c via nearest-neighbor interpolation."""
        needs_squeeze = False
        if x_coarse.dim() == 3:
            x_coarse = x_coarse.unsqueeze(1)
            needs_squeeze = True
        out = F.interpolate(x_coarse, size=target_size, mode="nearest")
        return out.squeeze(1) if needs_squeeze else out

    def forward(
        self,
        P_hat: torch.Tensor,
        R_app_hat: torch.Tensor,
        dW_obs: torch.Tensor,
        P_c_obs: torch.Tensor,
        fine_mask: torch.Tensor | None = None,
        coarse_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        H_f, W_f = P_hat.shape[-2], P_hat.shape[-1]

        R_app_fine = self._upsample_fine(R_app_hat, (H_f, W_f))

        P_hat_sq = P_hat.squeeze(1) if P_hat.dim() == 4 else P_hat
        R_fine_sq = R_app_fine.squeeze(1) if R_app_fine.dim() == 4 else R_app_fine
        dW_pred = R_fine_sq - P_hat_sq

        residual_W = (dW_obs - dW_pred) ** 2
        L_W = _masked_mean(residual_W, fine_mask)

        P_hat_coarse = self._avg_pool_coarse(P_hat)
        P_hat_coarse_sq = (
            P_hat_coarse.squeeze(1) if P_hat_coarse.dim() == 4 else P_hat_coarse
        )

        if P_hat_coarse_sq.shape != P_c_obs.shape:
            min_h = min(P_hat_coarse_sq.shape[-2], P_c_obs.shape[-2])
            min_w = min(P_hat_coarse_sq.shape[-1], P_c_obs.shape[-1])
            P_hat_coarse_sq = P_hat_coarse_sq[..., :min_h, :min_w]
            P_c_obs = P_c_obs[..., :min_h, :min_w]
            if coarse_mask is not None:
                coarse_mask = coarse_mask[..., :min_h, :min_w]

        L_Pc = _masked_mean((P_hat_coarse_sq - P_c_obs) ** 2, coarse_mask)

        R_sq = R_app_hat.squeeze(1) if R_app_hat.dim() == 4 else R_app_hat
        L_R_amp = _masked_mean(R_sq ** 2, coarse_mask)
        L_R_smooth = _smoothness_loss(R_sq, coarse_mask)
        L_R = L_R_amp + self.alpha_smooth * L_R_smooth

        loss = self.lambda_W * L_W + self.lambda_Pc * L_Pc + self.lambda_R * L_R
        return loss, {
            "L_W": float(L_W.item()),
            "L_Pc": float(L_Pc.item()),
            "L_R": float(L_R.item()),
            "total": float(loss.item()),
        }


class SupervisedLoss(nn.Module):
    """
    Optional supervised loss for synthetic twin training where ground truth
    is available. Combines budget loss with direct P and R_app supervision.
    """

    def __init__(
        self,
        budget_loss: nn.Module,
        lambda_P_sup: float = 1.0,
        lambda_R_sup: float = 0.1,
    ):
        super().__init__()
        self.budget = budget_loss
        self.lambda_P_sup = lambda_P_sup
        self.lambda_R_sup = lambda_R_sup

    def forward(
        self,
        P_hat: torch.Tensor,
        R_app_hat: torch.Tensor,
        dW_obs: torch.Tensor,
        P_c_obs: torch.Tensor,
        P_true: torch.Tensor | None = None,
        R_app_true: torch.Tensor | None = None,
        fine_mask: torch.Tensor | None = None,
        coarse_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        loss, components = self.budget(
            P_hat, R_app_hat, dW_obs, P_c_obs, fine_mask, coarse_mask
        )

        if P_true is not None and self.lambda_P_sup > 0:
            P_sq = P_hat.squeeze(1) if P_hat.dim() == 4 else P_hat
            L_P_sup = ((P_sq - P_true) ** 2).mean()
            loss = loss + self.lambda_P_sup * L_P_sup
            components["L_P_sup"] = float(L_P_sup.item())

        if R_app_true is not None and self.lambda_R_sup > 0:
            R_sq = R_app_hat.squeeze(1) if R_app_hat.dim() == 4 else R_app_hat
            if R_sq.shape != R_app_true.shape:
                R_app_true_resized = F.interpolate(
                    R_app_true.unsqueeze(1) if R_app_true.dim() == 3 else R_app_true,
                    size=R_sq.shape[-2:],
                    mode="bilinear",
                    align_corners=False,
                ).squeeze(1)
            else:
                R_app_true_resized = R_app_true
            L_R_sup = ((R_sq - R_app_true_resized) ** 2).mean()
            loss = loss + self.lambda_R_sup * L_R_sup
            components["L_R_sup"] = float(L_R_sup.item())

        components["total"] = float(loss.item())
        return loss, components


if __name__ == "__main__":
    # Quick approximate-path verification
    batch = 2
    print("=== BudgetLossConv smoke test ===")
    P_hat = torch.rand(batch, 1, 64, 64) * 0.5
    R_app = torch.randn(batch, 1, 12, 12) * 0.1
    dW = torch.randn(batch, 64, 64) * 0.05
    Pc = torch.rand(batch, 12, 12) * 0.3
    fine_mask = torch.ones(batch, 64, 64, dtype=torch.bool)
    coarse_mask = torch.ones(batch, 12, 12, dtype=torch.bool)

    loss_fn = BudgetLossConv(pool_size=5)
    loss, comps = loss_fn(P_hat, R_app, dW, Pc, fine_mask, coarse_mask)
    print(f"loss={loss.item():.4f}")
    for k, v in comps.items():
        print(f"{k}={v:.4f}")
