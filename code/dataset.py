#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PyTorch Dataset for budget-constrained precipitation downscaling.

Supports full-tile and patch-based modes for both synthetic twin
and real data.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset
from scipy import sparse
from netCDF4 import Dataset as NC4Dataset

CODE_DIR = Path(__file__).resolve().parent
REPO_ROOT = CODE_DIR.parent
PREPROC = REPO_ROOT / "preprocessed"
DEFAULT_SYNTH_DIR = PREPROC / "synthetic"


def resolve_synthetic_dir(synthetic_dir: str | Path | None = None) -> Path:
    """Resolve the synthetic dataset directory used by training/evaluation."""
    if synthetic_dir is None:
        return DEFAULT_SYNTH_DIR
    path = Path(synthetic_dir)
    if not path.is_absolute():
        path = REPO_ROOT / path
    return path.resolve()


def load_real_cache(Ic: sparse.csr_matrix, precompute_lifted: bool = True) -> dict:
    """
    Load all real data into RAM once. Returns a dict that can be shared
    across multiple BudgetDataset instances (train/val/test) to avoid
    duplicating ~12 GB of arrays.

    Parameters
    ----------
    Ic : sparse.csr_matrix
        Coarse-to-fine lifting operator.
    precompute_lifted : bool
        If True, also precompute lifted coarse precipitation on the fine grid.
        This adds about 6 GB RAM, but removes one sparse matvec from each
        training step.

    Returns
    -------
    dict with keys: W0, W1, Pc, Pc_lifted, fine_mask, coarse_mask, n_t
    """
    import time
    t0 = time.time()
    print("  [cache] Loading real data into RAM...")

    domain_path = PREPROC / "common_domain.nc"
    with NC4Dataset(domain_path, "r") as ds:
        fine_mask = np.array(ds.variables["fine_mask"][:], dtype=bool)
        coarse_mask = np.array(ds.variables["coarse_mask"][:], dtype=bool)

    # Load TPW (two full-domain fields per timestep)
    tpw_path = PREPROC / "tpw_3h.nc"
    with NC4Dataset(tpw_path, "r") as ds:
        n_t = ds.dimensions["time"].size
        print(f"  [cache] Reading tpw_t0 ({n_t} steps)...")
        W0 = np.nan_to_num(
            np.array(ds.variables["tpw_t0"][:], dtype=np.float32),
            nan=0.0)
        print(f"  [cache] Reading tpw_t1 ({n_t} steps)...")
        W1 = np.nan_to_num(
            np.array(ds.variables["tpw_t1"][:], dtype=np.float32),
            nan=0.0)

    # Load IMERG coarse precipitation
    imerg_path = PREPROC / "imerg_3h.nc"
    with NC4Dataset(imerg_path, "r") as ds:
        print(f"  [cache] Reading precipitation_3h...")
        Pc = np.array(ds.variables["precipitation_3h"][:], dtype=np.float32)

    Pc_lifted = None
    if precompute_lifted:
        n_t, n_clat, n_clon = Pc.shape
        n_flat = fine_mask.shape[0]
        n_flon = fine_mask.shape[1]
        print(f"  [cache] Precomputing lifted precipitation ({n_t} steps)...")
        Pc_lifted = np.zeros((n_t, n_flat, n_flon), dtype=np.float32)
        for t in range(n_t):
            Pc_lifted[t] = np.array(
                Ic @ Pc[t].ravel(), dtype=np.float32
            ).reshape(n_flat, n_flon)

    mem_bytes = W0.nbytes + W1.nbytes + Pc.nbytes
    if Pc_lifted is not None:
        mem_bytes += Pc_lifted.nbytes
    mem_gb = mem_bytes / 1e9
    dt = time.time() - t0
    print(f"  [cache] Done: {mem_gb:.1f} GB in {dt:.0f}s")

    return {
        "W0": W0,
        "W1": W1,
        "Pc": Pc,
        "Pc_lifted": Pc_lifted,
        "fine_mask": fine_mask,
        "coarse_mask": coarse_mask,
        "n_t": n_t,
    }


class BudgetDataset(Dataset):
    """
    Dataset for budget-constrained precipitation training.

    Each sample is one timestep (t, t+1) pair providing:
      - input: (C, H, W) tensor
        v1 (4ch): [W_t0, W_t1, dW, P_c_lifted]
        v2 (5ch): [W_t0, W_t1, dW, P_c_lifted, P_0_analytical]
      - targets: P_true, R_app_true, P_c_obs, dW_obs, fine_mask, coarse_mask
      - v2 also returns R_0_analytical

    Parameters
    ----------
    tile : str
        Tile name (e.g., 'A_wet').
    steps : list[int]
        Timestep indices to include (temporal split).
    Ic : sparse matrix
        Lifting operator (coarse → fine).
    Ac : sparse matrix or None
        Averaging operator (fine → coarse). Required for v2.
    patch_size : int or None
        If set, randomly crop spatial patches of this size.
    augment : bool
        Enable random flips for data augmentation.
    synthetic : bool
        If True, load from synthetic_truth/obs files (includes P_true).
    v2 : bool
        If True, compute analytical backbone fields (P_0, R_0).
    """

    def __init__(
        self,
        tile: str,
        steps: list[int],
        Ic: sparse.csr_matrix,
        Ac: sparse.csr_matrix | None = None,
        patch_size: int | None = None,
        augment: bool = False,
        synthetic: bool = True,
        synthetic_dir: str | Path | None = None,
        real_cache: dict | None = None,
        v2: bool = False,
    ):
        self.steps = steps
        self.patch_size = patch_size
        self.augment = augment
        self.synthetic = synthetic
        self.v2 = v2
        self.Ac = Ac  # needed for v2 per-step analytical computation
        self._AcIc_lu = None  # precomputed factorization for exact R_0 solve

        # Load data
        if synthetic:
            synth_dir = resolve_synthetic_dir(synthetic_dir)
            truth_path = synth_dir / f"synthetic_truth_{tile}.nc"
            obs_path = synth_dir / f"synthetic_obs_{tile}.nc"

            with NC4Dataset(truth_path, "r") as ds:
                max_t = ds.dimensions["time"].size
                self.P_true = np.array(ds.variables["P_true"][:])
                self.R_app_true = np.array(ds.variables["R_app_true"][:])
                self.fine_mask = np.array(ds.variables["fine_mask"][:],
                                          dtype=bool)
            with NC4Dataset(obs_path, "r") as ds:
                self.W_obs = np.array(ds.variables["W_obs"][:])
                self.P_c_obs = np.array(ds.variables["P_c_obs"][:])
            # Synthetic tile: all coarse cells are valid
            self.coarse_mask = np.ones(
                self.P_c_obs.shape[1:], dtype=bool)
        else:
            self._cached = real_cache is not None
            if self._cached:
                # RAM-cached mode: no NC handles, safe for num_workers>0
                self.fine_mask = real_cache["fine_mask"]
                self.coarse_mask = real_cache["coarse_mask"]
                self._W0_all = real_cache["W0"]
                self._W1_all = real_cache["W1"]
                self._Pc_all = real_cache["Pc"]
                self._Pc_lifted_all = real_cache.get("Pc_lifted")
                self._tpw_ds = None
                self._imerg_ds = None
            else:
                # Lazy loading fallback (slow, num_workers must be 0)
                domain_path = PREPROC / "common_domain.nc"
                with NC4Dataset(domain_path, "r") as ds:
                    self.fine_mask = np.array(
                        ds.variables["fine_mask"][:], dtype=bool)
                    self.coarse_mask = np.array(
                        ds.variables["coarse_mask"][:], dtype=bool)
                self._tpw_ds = NC4Dataset(PREPROC / "tpw_3h.nc", "r")
                self._imerg_ds = NC4Dataset(PREPROC / "imerg_3h.nc", "r")
                self._tpw_t0_var = self._tpw_ds.variables["tpw_t0"]
                self._tpw_t1_var = self._tpw_ds.variables["tpw_t1"]
                self._pc_var = self._imerg_ds.variables["precipitation_3h"]
            self.P_c_obs = None  # loaded per-step
            self.P_true = None
            self.R_app_true = None

        # For synthetic: precompute lifted P_c and NaN-filled W
        if synthetic:
            self.W_filled = np.nan_to_num(self.W_obs, nan=0.0)
            n_flat, n_flon = self.fine_mask.shape
            n_t = self.P_c_obs.shape[0]
            self.P_c_lifted = np.zeros(
                (n_t, n_flat, n_flon), dtype=np.float32)
            for t in range(n_t):
                self.P_c_lifted[t] = np.array(
                    Ic @ self.P_c_obs[t].ravel(), dtype=np.float32
                ).reshape(n_flat, n_flon)

            # V2: precompute analytical backbone for all synthetic steps
            # Solve (A_c I_c) R_0 = P_c_obs + A_c(dW) exactly, since
            # A_c∘I_c ≠ Id (diag mean ~0.852 due to non-exact grid nesting).
            self.P_0_all = None
            self.R_0_all = None
            if v2 and Ac is not None:
                from scipy.sparse.linalg import splu
                print("  [dataset] Precomputing analytical backbone (v2, "
                      "exact solve)...")
                c_shape = self.P_c_obs.shape[1:]  # (n_clat, n_clon)
                n_clat_s, n_clon_s = c_shape
                # Precompute sparse LU of (A_c I_c) — same for all timesteps
                AcIc = (Ac @ Ic).tocsc()
                AcIc_lu = splu(AcIc)
                self._AcIc_lu = AcIc_lu
                print(f"  [dataset] AcIc factorized: "
                      f"{AcIc.shape}, nnz={AcIc.nnz}")
                self.R_0_all = np.zeros(
                    (n_t, n_clat_s, n_clon_s), dtype=np.float32)
                self.P_0_all = np.zeros(
                    (n_t, n_flat, n_flon), dtype=np.float32)
                for t in range(n_t):
                    if t + 1 >= self.W_filled.shape[0]:
                        continue
                    dw_t = self.W_filled[t + 1] - self.W_filled[t]
                    Ac_dw = np.array(
                        Ac @ dw_t.ravel(), dtype=np.float64)
                    rhs = self.P_c_obs[t].ravel().astype(np.float64) + Ac_dw
                    R_0_t = AcIc_lu.solve(rhs).astype(np.float32).reshape(
                        n_clat_s, n_clon_s)
                    R_0_lift = np.array(
                        Ic @ R_0_t.ravel(), dtype=np.float32
                    ).reshape(n_flat, n_flon)
                    P_0_t = R_0_lift - dw_t
                    self.R_0_all[t] = R_0_t
                    self.P_0_all[t] = P_0_t
                print(f"  [dataset] Analytical backbone done (exact): "
                      f"P_0 range [{self.P_0_all.min():.2f}, "
                      f"{self.P_0_all.max():.2f}]")
        else:
            if self._cached:
                n_t = self._W0_all.shape[0]
            else:
                n_t = self._tpw_ds.dimensions["time"].size
            self.P_c_lifted = None  # computed on-the-fly
            # For real-data v2: precompute AcIc factorization
            if v2 and Ac is not None and Ic is not None:
                from scipy.sparse.linalg import splu
                AcIc = (Ac @ Ic).tocsc()
                self._AcIc_lu = splu(AcIc)
                print(f"  [dataset] AcIc factorized (real): "
                      f"{AcIc.shape}, nnz={AcIc.nnz}")

        self.Ic = Ic  # keep for on-the-fly lifting

        # Filter to valid steps
        if synthetic:
            self.valid_steps = [t for t in steps
                                if t + 1 < self.W_filled.shape[0]]
        else:
            self.valid_steps = [t for t in steps if t < n_t]

    def __len__(self):
        return len(self.valid_steps)

    def __getitem__(self, idx):
        t = self.valid_steps[idx]

        if self.synthetic:
            w0 = self.W_filled[t]
            w1 = self.W_filled[t + 1]
            pc_obs = self.P_c_obs[t]
            pc_lift = self.P_c_lifted[t]
        elif self._cached:
            # RAM-cached mode: pure numpy indexing, no I/O
            w0 = self._W0_all[t]
            w1 = self._W1_all[t]
            pc_obs = self._Pc_all[t]
            if self._Pc_lifted_all is not None:
                pc_lift = self._Pc_lifted_all[t]
            else:
                n_flat, n_flon = self.fine_mask.shape
                pc_lift = np.array(
                    self.Ic @ pc_obs.ravel(), dtype=np.float32
                ).reshape(n_flat, n_flon)
        else:
            # Lazy load from NC variables (slow fallback)
            w0 = np.nan_to_num(
                np.array(self._tpw_t0_var[t], dtype=np.float32), nan=0.0)
            w1 = np.nan_to_num(
                np.array(self._tpw_t1_var[t], dtype=np.float32), nan=0.0)
            pc_obs = np.array(self._pc_var[t], dtype=np.float32)
            n_flat, n_flon = self.fine_mask.shape
            pc_lift = np.array(
                self.Ic @ pc_obs.ravel(), dtype=np.float32
            ).reshape(n_flat, n_flon)

        dw = w1 - w0

        # V2: compute or retrieve analytical backbone
        p_0 = None
        r_0 = None
        if self.v2:
            if self.synthetic and self.P_0_all is not None:
                p_0 = self.P_0_all[t]
                r_0 = self.R_0_all[t]
            elif self.Ac is not None:
                # Per-step analytical computation (cached real data)
                n_flat, n_flon = self.fine_mask.shape
                Ac_dw = np.array(
                    self.Ac @ dw.ravel(), dtype=np.float64)
                n_coarse = pc_obs.shape
                rhs = pc_obs.ravel().astype(np.float64) + Ac_dw
                if self._AcIc_lu is not None:
                    r_0 = self._AcIc_lu.solve(rhs).astype(
                        np.float32).reshape(n_coarse)
                else:
                    r_0 = rhs.astype(np.float32).reshape(n_coarse)
                R_0_lift = np.array(
                    self.Ic @ r_0.ravel(), dtype=np.float32
                ).reshape(n_flat, n_flon)
                p_0 = R_0_lift - dw

        # Stack input: (4, H, W) for v1, (5, H, W) for v2
        if self.v2 and p_0 is not None:
            x = np.stack([w0, w1, dw, pc_lift, p_0], axis=0)
        else:
            x = np.stack([w0, w1, dw, pc_lift], axis=0)

        # Targets (ground truth only available for synthetic)
        p_true = self.P_true[t] if self.P_true is not None else np.zeros_like(w0)
        r_true = (self.R_app_true[t] if self.R_app_true is not None
                  else np.zeros(pc_obs.shape))
        mask = self.fine_mask
        coarse_mask = self.coarse_mask

        # Patch crop (fine grid + corresponding coarse subregion)
        if self.patch_size is not None:
            H, W = x.shape[1], x.shape[2]
            ps = self.patch_size
            if H > ps and W > ps:
                # Align patch to coarse grid (multiple of pool_size=5)
                ps_aligned = (ps // 5) * 5
                i = np.random.randint(0, H - ps_aligned)
                j = np.random.randint(0, W - ps_aligned)
                # Snap to coarse-cell boundary
                i = (i // 5) * 5
                j = (j // 5) * 5
                x = x[:, i:i + ps_aligned, j:j + ps_aligned]
                p_true = p_true[i:i + ps_aligned, j:j + ps_aligned]
                dw = dw[i:i + ps_aligned, j:j + ps_aligned]
                mask = mask[i:i + ps_aligned, j:j + ps_aligned]
                # Extract corresponding coarse-grid region
                ci, cj = i // 5, j // 5
                ch, cw = ps_aligned // 5, ps_aligned // 5
                pc_obs = pc_obs[ci:ci + ch, cj:cj + cw]
                r_true = r_true[ci:ci + ch, cj:cj + cw]
                coarse_mask = coarse_mask[ci:ci + ch, cj:cj + cw]
                if r_0 is not None:
                    r_0 = r_0[ci:ci + ch, cj:cj + cw]

        # Augmentation (random flips — must flip coarse fields in sync)
        if self.augment:
            if np.random.rand() > 0.5:
                x = x[:, ::-1, :].copy()
                p_true = p_true[::-1, :].copy()
                dw = dw[::-1, :].copy()
                mask = mask[::-1, :].copy()
                pc_obs = pc_obs[::-1, :].copy()
                r_true = r_true[::-1, :].copy()
                coarse_mask = coarse_mask[::-1, :].copy()
                if r_0 is not None:
                    r_0 = r_0[::-1, :].copy()
            if np.random.rand() > 0.5:
                x = x[:, :, ::-1].copy()
                p_true = p_true[:, ::-1].copy()
                dw = dw[:, ::-1].copy()
                mask = mask[:, ::-1].copy()
                pc_obs = pc_obs[:, ::-1].copy()
                r_true = r_true[:, ::-1].copy()
                coarse_mask = coarse_mask[:, ::-1].copy()
                if r_0 is not None:
                    r_0 = r_0[:, ::-1].copy()

        out = {
            "x": torch.from_numpy(x.astype(np.float32)),
            "P_true": torch.from_numpy(p_true.astype(np.float32)),
            "R_app_true": torch.from_numpy(r_true.astype(np.float32)),
            "P_c_obs": torch.from_numpy(pc_obs.astype(np.float32)),
            "dW_obs": torch.from_numpy(dw.astype(np.float32)),
            "fine_mask": torch.from_numpy(mask),
            "coarse_mask": torch.from_numpy(coarse_mask),
            "t": t,
        }
        if self.v2 and r_0 is not None:
            out["R_0"] = torch.from_numpy(r_0.astype(np.float32))
        return out


def make_splits(n_t: int, train_frac=0.66, val_frac=0.17):
    """Temporal train/val/test split."""
    n_train = int(n_t * train_frac)
    n_val = int(n_t * val_frac)
    train = list(range(n_train))
    val = list(range(n_train, n_train + n_val))
    test = list(range(n_train + n_val, n_t))
    return train, val, test


if __name__ == "__main__":
    import time
    sys.path.insert(0, str(CODE_DIR))
    from build_support_operators import build_Ac, build_Ic

    # --- Synthetic tile test ---
    print("=== SYNTHETIC TILE TEST ===")
    synth_dir = resolve_synthetic_dir()
    with NC4Dataset(synth_dir / "synthetic_truth_A_wet.nc", "r") as ds:
        flat = np.array(ds.variables["fine_lat"][:])
        flon = np.array(ds.variables["fine_lon"][:])
        clat = np.array(ds.variables["coarse_lat"][:])
        clon = np.array(ds.variables["coarse_lon"][:])
        n_t = ds.dimensions["time"].size

    Ac = build_Ac(flat, flon, clat, clon)
    Ic = build_Ic(flat, flon, clat, clon)

    train_s, val_s, test_s = make_splits(n_t)
    print(f"Splits: train={len(train_s)}, val={len(val_s)}, test={len(test_s)}")

    ds_full = BudgetDataset("A_wet", train_s[:10], Ic)
    sample = ds_full[0]
    print(f"\nFull-tile sample:")
    for k, v in sample.items():
        if isinstance(v, torch.Tensor):
            print(f"  {k}: {v.shape} {v.dtype}")
        else:
            print(f"  {k}: {v}")

    # --- Real data test ---
    print("\n=== REAL DATA TEST (lazy loading) ===")
    # Use full-domain operators
    from scipy import sparse
    ops = np.load(PREPROC / "support_operators.npz", allow_pickle=True)
    Ic_full = sparse.csr_matrix(
        (ops["Ic_data"], ops["Ic_indices"], ops["Ic_indptr"]),
        shape=tuple(ops["Ic_shape"]))

    t0 = time.time()
    ds_real = BudgetDataset("full", [0, 1, 2], Ic_full, synthetic=False)
    print(f"  Dataset init: {time.time() - t0:.1f}s")
    print(f"  Valid steps: {len(ds_real)}")

    t0 = time.time()
    sample_r = ds_real[0]
    print(f"  Sample load: {time.time() - t0:.1f}s")
    print(f"\nReal data sample:")
    for k, v in sample_r.items():
        if isinstance(v, torch.Tensor):
            print(f"  {k}: {v.shape} {v.dtype}")
        else:
            print(f"  {k}: {v}")

    # Cleanup NC handles
    ds_real._tpw_ds.close()
    ds_real._imerg_ds.close()
