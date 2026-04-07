#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Build support operators A_c (fine to coarse) and I_c (coarse to fine)
for the water budget closure project.

Reads the common domain file produced by build_common_domain_and_masks.py and
computes sparse weight matrices stored as CSR in an .npz file plus metadata in
a companion NetCDF.

Outputs (saved to ``preprocessed/``):
    support_operators.npz
        - Ac_data, Ac_indices, Ac_indptr, Ac_shape   (CSR components of A_c)
        - Ic_data, Ic_indices, Ic_indptr, Ic_shape   (CSR components of I_c)
    support_operators_meta.nc
        - Consistency check results and diagnostics

Operators:
    A_c : G_f -> G_c   (overlap-based conservative remap, fine to coarse)
        Each row corresponds to one coarse cell (flattened j*n_clon + i).
        The row sums to 1 over all fine pixels that overlap that cell,
        with weights proportional to the latitude-longitude overlap area on the native grid.

    I_c : G_c -> G_f   (overlap-based piecewise-constant lifting, coarse to fine)
        Each row corresponds to one fine pixel (flattened j*n_flon + i).
        The row has a single non-zero entry: the coarse cell that contains
        the fine pixel's centre.  For boundary pixels whose centre falls
        between two coarse cells, the nearest coarse cell is chosen.

Consistency of `A_c @ I_c` is assessed numerically. For the present
non-nested grids, exact identity does not hold; the saved diagnostics quantify
diagonal retention and off-diagonal leakage.

Usage:
    python build_support_operators.py
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy import sparse
from netCDF4 import Dataset

CODE_DIR = Path(__file__).resolve().parent
REPO_ROOT = CODE_DIR.parent
PREPROC = REPO_ROOT / "preprocessed"


@dataclass(frozen=True)
class OperatorConfig:
    domain_file: Path
    output_dir: Path


def parse_args() -> OperatorConfig:
    parser = argparse.ArgumentParser(
        description="Build support operators A_c and I_c from a selected common-domain file."
    )
    parser.add_argument(
        "--preproc-dir",
        type=Path,
        default=PREPROC,
        help="Directory containing common_domain.nc and receiving the operator outputs.",
    )
    parser.add_argument(
        "--domain-file",
        type=Path,
        default=None,
        help="Explicit common-domain file. Defaults to <preproc-dir>/common_domain.nc.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory for support_operators.npz and support_operators_meta.nc. Defaults to <preproc-dir>.",
    )
    args = parser.parse_args()

    preproc_dir = args.preproc_dir.resolve()
    domain_file = args.domain_file.resolve() if args.domain_file else (preproc_dir / "common_domain.nc")
    output_dir = args.output_dir.resolve() if args.output_dir else preproc_dir
    return OperatorConfig(domain_file=domain_file, output_dir=output_dir)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def cell_edges(centers: np.ndarray) -> np.ndarray:
    """Cell edges from centres assuming locally uniform spacing."""
    n = len(centers)
    edges = np.zeros(n + 1, dtype=np.float64)
    if n == 1:
        edges[0] = centers[0] - 0.01
        edges[1] = centers[0] + 0.01
        return edges
    half = np.diff(centers) / 2.0
    edges[0] = centers[0] - half[0]
    edges[-1] = centers[-1] + half[-1]
    edges[1:-1] = centers[:-1] + half
    return edges

def build_Ac(
    fine_lat: np.ndarray,
    fine_lon: np.ndarray,
    coarse_lat: np.ndarray,
    coarse_lon: np.ndarray,
) -> sparse.csr_matrix:
    """
    Build the overlap-based averaging operator A_c (fine to coarse).

    Weights are proportional to the overlap area between each fine cell and its parent coarse cell on the native latitude-longitude grid.

    Returns a CSR matrix of shape (n_coarse, n_fine) where
        n_coarse = len(coarse_lat) * len(coarse_lon)
        n_fine   = len(fine_lat) * len(fine_lon)
    """
    n_clat, n_clon = len(coarse_lat), len(coarse_lon)
    n_flat, n_flon = len(fine_lat), len(fine_lon)
    n_coarse = n_clat * n_clon
    n_fine = n_flat * n_flon

    f_lat_edges = cell_edges(fine_lat)
    f_lon_edges = cell_edges(fine_lon)

    c_step = 0.1
    half_c = c_step / 2.0

    rows, cols, vals = [], [], []

    for cj in range(n_clat):
        c_lat_lo = coarse_lat[cj] - half_c
        c_lat_hi = coarse_lat[cj] + half_c

        fj_idx = np.where(
            (f_lat_edges[1:] > c_lat_lo) & (f_lat_edges[:-1] < c_lat_hi)
        )[0]
        if len(fj_idx) == 0:
            continue

        # Latitude bounds of each overlapping fine cell clipped to
        # the coarse cell boundary
        overlap_lat_lo = np.maximum(f_lat_edges[fj_idx], c_lat_lo)
        overlap_lat_hi = np.minimum(f_lat_edges[fj_idx + 1], c_lat_hi)

        for ci in range(n_clon):
            c_lon_lo = coarse_lon[ci] - half_c
            c_lon_hi = coarse_lon[ci] + half_c

            fi_idx = np.where(
                (f_lon_edges[1:] > c_lon_lo) & (f_lon_edges[:-1] < c_lon_hi)
            )[0]
            if len(fi_idx) == 0:
                continue

            overlap_lon_lo = np.maximum(f_lon_edges[fi_idx], c_lon_lo)
            overlap_lon_hi = np.minimum(f_lon_edges[fi_idx + 1], c_lon_hi)

            overlap_dlat = np.abs(overlap_lat_hi - overlap_lat_lo)
            overlap_dlon = np.abs(overlap_lon_hi - overlap_lon_lo)
            weights_2d = overlap_dlat[:, None] * overlap_dlon[None, :]
            total = np.sum(weights_2d)
            if total <= 0:
                continue

            c_flat = cj * n_clon + ci
            for local_j, fj in enumerate(fj_idx):
                for local_i, fi in enumerate(fi_idx):
                    w = weights_2d[local_j, local_i] / total
                    if w > 1e-12:
                        f_flat = fj * n_flon + fi
                        rows.append(c_flat)
                        cols.append(f_flat)
                        vals.append(w)

    Ac = sparse.csr_matrix(
        (np.array(vals, dtype=np.float64),
         (np.array(rows, dtype=np.int64), np.array(cols, dtype=np.int64))),
        shape=(n_coarse, n_fine),
    )
    return Ac


def build_Ic(
    fine_lat: np.ndarray,
    fine_lon: np.ndarray,
    coarse_lat: np.ndarray,
    coarse_lon: np.ndarray,
) -> sparse.csr_matrix:
    """
    Build the piecewise-constant lifting operator I_c (coarse to fine).

    Each fine pixel is assigned to the coarse cell whose centre is nearest
    to the fine pixel's centre.

    Returns CSR matrix of shape (n_fine, n_coarse).
    """
    n_clat, n_clon = len(coarse_lat), len(coarse_lon)
    n_flat, n_flon = len(fine_lat), len(fine_lon)
    n_coarse = n_clat * n_clon
    n_fine = n_flat * n_flon

    # For each fine lat, find nearest coarse lat index
    fj_to_cj = np.searchsorted(coarse_lat, fine_lat, side="left")
    fj_to_cj = np.clip(fj_to_cj, 0, n_clat - 1)
    # Refine: check if the previous coarse index is actually closer
    for k in range(len(fine_lat)):
        cj = fj_to_cj[k]
        if cj > 0 and abs(fine_lat[k] - coarse_lat[cj - 1]) < abs(fine_lat[k] - coarse_lat[cj]):
            fj_to_cj[k] = cj - 1

    fi_to_ci = np.searchsorted(coarse_lon, fine_lon, side="left")
    fi_to_ci = np.clip(fi_to_ci, 0, n_clon - 1)
    for k in range(len(fine_lon)):
        ci = fi_to_ci[k]
        if ci > 0 and abs(fine_lon[k] - coarse_lon[ci - 1]) < abs(fine_lon[k] - coarse_lon[ci]):
            fi_to_ci[k] = ci - 1

    rows = np.zeros(n_fine, dtype=np.int64)
    cols = np.zeros(n_fine, dtype=np.int64)

    idx = 0
    for fj in range(n_flat):
        cj = fj_to_cj[fj]
        for fi in range(n_flon):
            ci = fi_to_ci[fi]
            rows[idx] = fj * n_flon + fi
            cols[idx] = cj * n_clon + ci
            idx += 1

    vals = np.ones(n_fine, dtype=np.float64)
    Ic = sparse.csr_matrix((vals, (rows, cols)), shape=(n_fine, n_coarse))
    return Ic


def verify_consistency(
    Ac: sparse.csr_matrix,
    Ic: sparse.csr_matrix,
    n_coarse: int,
) -> dict:
    """
    Assess A_c @ I_c against Id_{G_c}.

    Returns dict with diagonal and off-diagonal error statistics.
    For non-nested grids, perfect consistency is impossible; the errors
    quantify the magnitude of grid-mismatch leakage.
    """
    AcIc = Ac @ Ic  # (n_coarse, n_coarse)

    # Diagonal: should be 1.0 for perfect consistency
    diag = np.array(AcIc.diagonal()).ravel()
    diag_err = diag - 1.0

    # Off-diagonal leakage: row sums minus diagonal
    row_sums = np.array(AcIc.sum(axis=1)).ravel()
    offdiag_sums = row_sums - diag  # should be 0 for perfect consistency

    stats = {
        "diag_min": float(np.min(diag)),
        "diag_max": float(np.max(diag)),
        "diag_mean": float(np.mean(diag)),
        "diag_err_max": float(np.max(np.abs(diag_err))),
        "diag_err_mean": float(np.mean(np.abs(diag_err))),
        "offdiag_sum_max": float(np.max(np.abs(offdiag_sums))),
        "offdiag_sum_mean": float(np.mean(np.abs(offdiag_sums))),
        "row_sum_min": float(np.min(row_sums)),
        "row_sum_max": float(np.max(row_sums)),
    }
    return stats


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    cfg = parse_args()

    print("=" * 60)
    print("  Build Support Operators (A_c, I_c)")
    print("=" * 60)
    print(f"  Domain file: {cfg.domain_file}")
    print(f"  Output dir:  {cfg.output_dir}")

    if not cfg.domain_file.exists():
        print(
            f"ERROR: {cfg.domain_file} not found. "
            "Run build_common_domain_and_masks.py first or pass --domain-file."
        )
        return 1

    # Load grids
    print("\n[1/5] Loading common domain grids...")
    ds = Dataset(cfg.domain_file, "r")
    coarse_lat = np.array(ds.variables["coarse_lat"][:], dtype=np.float64)
    coarse_lon = np.array(ds.variables["coarse_lon"][:], dtype=np.float64)
    fine_lat = np.array(ds.variables["fine_lat"][:], dtype=np.float64)
    fine_lon = np.array(ds.variables["fine_lon"][:], dtype=np.float64)
    ds.close()

    n_clat, n_clon = len(coarse_lat), len(coarse_lon)
    n_flat, n_flon = len(fine_lat), len(fine_lon)
    n_coarse = n_clat * n_clon
    n_fine = n_flat * n_flon

    print(f"  Coarse: {n_clat} x {n_clon} = {n_coarse} cells")
    print(f"  Fine:   {n_flat} x {n_flon} = {n_fine} pixels")

    # Build A_c
    print("\n[2/5] Building A_c (fine to coarse, overlap-based)...")
    Ac = build_Ac(fine_lat, fine_lon, coarse_lat, coarse_lon)
    print(f"  Shape: {Ac.shape}, nnz: {Ac.nnz}")
    print(f"  Density: {100 * Ac.nnz / (Ac.shape[0] * Ac.shape[1]):.4f}%")

    # Verify row sums ~= 1
    row_sums = np.array(Ac.sum(axis=1)).ravel()
    print(f"  Row sum range: [{np.min(row_sums):.6f}, {np.max(row_sums):.6f}]")
    print(f"  Row sum mean: {np.mean(row_sums):.6f}")

    # Build I_c
    print("\n[3/5] Building I_c (coarse to fine, piecewise-constant)...")
    Ic = build_Ic(fine_lat, fine_lon, coarse_lat, coarse_lon)
    print(f"  Shape: {Ic.shape}, nnz: {Ic.nnz}")

    # Verify each fine pixel maps to exactly one coarse cell
    row_sums_ic = np.array(Ic.sum(axis=1)).ravel()
    all_ones = np.allclose(row_sums_ic, 1.0)
    print(f"  Each fine pixel maps to exactly 1 coarse cell: {all_ones}")

    # Consistency check
    print("\n[4/5] Verifying A_c @ I_c against Id_{G_c}...")
    stats = verify_consistency(Ac, Ic, n_coarse)
    print(f"  Diagonal (should be 1.0):")
    print(f"    range: [{stats['diag_min']:.6f}, {stats['diag_max']:.6f}], "
          f"mean: {stats['diag_mean']:.6f}")
    print(f"    max |diag - 1|: {stats['diag_err_max']:.4e}, "
          f"mean |diag - 1|: {stats['diag_err_mean']:.4e}")
    print(f"  Off-diagonal leakage (should be 0):")
    print(f"    max |off-diag row sum|: {stats['offdiag_sum_max']:.4e}, "
          f"mean: {stats['offdiag_sum_mean']:.4e}")
    print(f"  Row sums of A_c @ I_c: [{stats['row_sum_min']:.6f}, "
          f"{stats['row_sum_max']:.6f}]")

    # For non-nested grids, exact consistency is impossible.
    # Accept if diagonal accuracy > 90% (error < 10%)
    diag_ok = stats["diag_err_max"] < 0.10
    print(f"  Diagonal accuracy {'OK' if diag_ok else 'POOR'} "
          f"(threshold: max error < 10%)")
    if not diag_ok:
        print("  NOTE: Non-exact longitude nesting (0.02002 deg != 0.02 deg)")
        print("        causes unavoidable grid-mismatch leakage at cell boundaries.")
        print("        This is documented in DATA_INTERFACE_SPEC.md Section 2.3.")

    # Save
    print("\n[5/5] Saving operators...")
    cfg.output_dir.mkdir(parents=True, exist_ok=True)

    # Save sparse matrices as .npz
    npz_path = cfg.output_dir / "support_operators.npz"
    np.savez_compressed(
        npz_path,
        Ac_data=Ac.data,
        Ac_indices=Ac.indices,
        Ac_indptr=Ac.indptr,
        Ac_shape=np.array(Ac.shape),
        Ic_data=Ic.data,
        Ic_indices=Ic.indices,
        Ic_indptr=Ic.indptr,
        Ic_shape=np.array(Ic.shape),
    )
    print(f"  Saved CSR data: {npz_path} ({npz_path.stat().st_size / 1024:.1f} KB)")

    # Save metadata NetCDF
    meta_path = cfg.output_dir / "support_operators_meta.nc"
    with Dataset(meta_path, "w", format="NETCDF4") as mds:
        mds.title = "Support operator metadata for TP water budget closure"
        mds.Ac_shape = f"{Ac.shape[0]} x {Ac.shape[1]}"
        mds.Ac_nnz = int(Ac.nnz)
        mds.Ic_shape = f"{Ic.shape[0]} x {Ic.shape[1]}"
        mds.Ic_nnz = int(Ic.nnz)
        mds.consistency_diag_err_max = stats["diag_err_max"]
        mds.consistency_diag_err_mean = stats["diag_err_mean"]
        mds.consistency_offdiag_max = stats["offdiag_sum_max"]
        mds.consistency_offdiag_mean = stats["offdiag_sum_mean"]
        mds.consistency_note = (
            "A_c @ I_c != Id because TPW lon step (0.02002 deg) does not "
            "nest exactly into 0.1 deg. Boundary pixels cause unavoidable "
            "leakage between adjacent coarse cells."
        )
        mds.Ac_description = (
            "Overlap-based conservative remap from fine (~0.02 deg) "
            "to coarse (0.1 deg) grid. Each row sums to 1. Weights are "
            "computed from latitude-longitude cell overlaps on the native "
            "grid, which handles non-exact longitude nesting through overlap "
            "areas."
        )
        mds.Ic_description = (
            "Piecewise-constant lifting from coarse (0.1 deg) to fine (~0.02 deg) "
            "grid. Each fine pixel maps to its nearest coarse cell centre."
        )
        mds.source_script = "build_support_operators.py"
        mds.domain_file = str(cfg.domain_file)
        mds.output_dir = str(cfg.output_dir)
    print(f"  Saved metadata: {meta_path}")

    print("\n" + "=" * 60)
    print("  DONE")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())


