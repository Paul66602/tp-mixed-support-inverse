#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generate a controlled misspecified synthetic twin for mechanism-stress tests.

This script is deliberately separate from ``synthetic_twin_generate.py`` so the
reference synthetic twin remains untouched. The misspecified twin
keeps the same observation supports and coarse precipitation anchor, but the
true fine-grid TPW tendency is perturbed by an unresolved fine-grid term that is
not represented by the reduced relation used for training:

    dW_true = I_c(R_app_true) - P_true + Q_unresolved

The default mismatch mode enforces A_c[Q_unresolved] = 0 exactly, so the added
term is invisible to the coarse-support observation geometry. This is the
highest-value first misspecification because it directly tests whether the A3,
A6, and objective-alignment conclusions survive when the true system contains a
fine-scale budget component outside the model class.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from netCDF4 import Dataset, date2num
from scipy.ndimage import gaussian_filter
from tqdm import tqdm

from synthetic_twin_generate import (
    PREPROC,
    TILES,
    add_observation_noise,
    coarse_to_fine_slices,
    generate_P_true_fast,
    generate_R_app_true,
    generate_orographic_proxy,
    get_tile_slices,
    load_domain,
    load_imerg_3h,
    load_tpw_3h,
)

CODE_DIR = Path(__file__).resolve().parent
DEFAULT_OUT_ROOT = PREPROC / "synthetic_misspecified"


def _compute_weighted_zero_mean_field(
    field: np.ndarray,
    Ac,
    fine_mask: np.ndarray,
    _gram_cache: dict = {},
) -> np.ndarray:
    """
    Project a fine-grid field onto the null space of the masked operator
    A_c[:, fine_mask].

    The unresolved term must satisfy two constraints simultaneously:
    it must vanish outside the valid fine-mask region and its coarse weighted
    average must be zero. Projecting with the full operator and then
    re-imposing the mask breaks the second constraint. We therefore restrict
    the operator to masked fine pixels first, then compute the standard
    pseudo-inverse projection inside that subspace:

        proj = x - A^T (A A^T)^{-1} (A x),  where A = A_c[:, fine_mask]

    Rows with no masked support are dropped before factorization.
    """
    from scipy.sparse.linalg import factorized

    mask_flat = fine_mask.reshape(-1).astype(bool)
    flat_masked = field.reshape(-1)[mask_flat].astype(np.float64, copy=True)

    # Cache the sparse LU factorization for the masked operator.
    cache_key = (id(Ac), mask_flat.shape, int(mask_flat.sum()))
    if cache_key not in _gram_cache:
        A_mask = Ac[:, mask_flat].tocsr()
        active_rows = np.array(A_mask.power(2).sum(axis=1)).ravel() > 0.0
        A_active = A_mask[active_rows]
        A_active_T = A_active.T.tocsc()
        G = (A_active @ A_active_T).tocsc()
        _gram_cache[cache_key] = (A_mask, active_rows, A_active_T, factorized(G))

    A_mask, active_rows, A_active_T, solve_gram = _gram_cache[cache_key]

    # Project inside the masked subspace.
    coarse_component = np.array(A_mask @ flat_masked, dtype=np.float64)
    coarse_component_active = coarse_component[active_rows]
    correction_coarse = solve_gram(coarse_component_active)
    correction_masked = A_active_T @ correction_coarse
    flat_masked -= correction_masked

    flat = np.zeros(field.size, dtype=np.float64)
    flat[mask_flat] = flat_masked

    return flat.reshape(field.shape).astype(np.float32)


def _generate_unresolved_term(
    n_t: int,
    n_flat: int,
    n_flon: int,
    Ac,
    fine_mask: np.ndarray,
    h_norm_fine: np.ndarray,
    base_dw: np.ndarray,
    epsilon: float,
    sigma: float,
    rng: np.random.Generator,
    mode: str,
) -> np.ndarray:
    """
    Generate Q_unresolved on the fine grid with controlled amplitude.

    Parameters
    ----------
    base_dw : np.ndarray
        Reference dW signal used only to calibrate mismatch amplitude.
    epsilon : float
        RMS scaling relative to mean absolute reference |dW|.
    sigma : float
        Gaussian smoothing scale for the synthetic unresolved field.
    mode : str
        Currently only ``fine_nullspace`` is supported.
    """
    if mode != "fine_nullspace":
        raise ValueError(f"Unsupported mismatch mode: {mode}")

    target_rms = float(epsilon * np.mean(np.abs(base_dw)))
    q_all = np.zeros((n_t, n_flat, n_flon), dtype=np.float32)

    for t in tqdm(range(n_t), desc="Q_unresolved generation"):
        raw = rng.standard_normal((n_flat, n_flon)).astype(np.float32)
        raw = gaussian_filter(raw, sigma=sigma)
        raw *= (0.5 + h_norm_fine)  # stronger structure over the same terrain proxy
        raw[~fine_mask] = 0.0
        raw = _compute_weighted_zero_mean_field(raw, Ac, fine_mask)
        raw[~fine_mask] = 0.0

        rms = float(np.sqrt(np.mean(raw[fine_mask] ** 2))) if np.any(fine_mask) else 0.0
        if rms > 0.0:
            raw *= target_rms / rms
        q_all[t] = raw

    return q_all


def _forward_integrate_w_misspecified(
    w_init: np.ndarray,
    p_true: np.ndarray,
    r_app_true: np.ndarray,
    q_unresolved: np.ndarray,
    Ic,
    n_flat: int,
    n_flon: int,
) -> np.ndarray:
    """Forward-integrate with an unresolved fine-grid mismatch term."""
    n_t = p_true.shape[0]
    w = np.zeros((n_t + 1, n_flat, n_flon), dtype=np.float32)
    w[0] = w_init.copy()

    for t in tqdm(range(n_t), desc="W forward integration (misspecified)"):
        r_fine = np.array(Ic @ r_app_true[t].ravel(), dtype=np.float32).reshape(
            n_flat, n_flon
        )
        w[t + 1] = w[t] + r_fine - p_true[t] + q_unresolved[t]

    return w


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Generate a misspecified synthetic twin"
    )
    parser.add_argument("--tile", choices=list(TILES.keys()), default="A_wet")
    parser.add_argument(
        "--mismatch-mode",
        choices=["fine_nullspace"],
        default="fine_nullspace",
        help="Misspecification family. 'fine_nullspace' adds a smooth fine-grid "
             "budget term whose coarse weighted average is exactly zero.",
    )
    parser.add_argument(
        "--epsilon",
        type=float,
        default=0.20,
        help="Mismatch RMS relative to mean absolute reference |dW|.",
    )
    parser.add_argument(
        "--beta",
        type=float,
        default=0.3,
        help="Sub-pixel variability amplitude passed through to P_true generation.",
    )
    parser.add_argument(
        "--sigma-w",
        type=float,
        default=1.0,
        help="TPW observation-noise std for W_obs.",
    )
    parser.add_argument(
        "--smooth-sigma",
        type=float,
        default=6.0,
        help="Gaussian smoothing scale for Q_unresolved.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--out-root",
        type=str,
        default=str(DEFAULT_OUT_ROOT),
        help="Root directory for the generated misspecified dataset.",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    out_root = Path(args.out_root)
    variant_name = (
        f"{args.tile}_{args.mismatch_mode}_eps{args.epsilon:.2f}_"
        f"beta{args.beta:.2f}_seed{args.seed}"
    )
    out_dir = out_root / variant_name

    rng = np.random.default_rng(args.seed)

    print("=" * 72)
    print("  Misspecified Synthetic Twin Generation")
    print("=" * 72)
    print(f"  Tile:           {args.tile}")
    print(f"  Mismatch mode:  {args.mismatch_mode}")
    print(f"  Epsilon:        {args.epsilon}")
    print(f"  Beta:           {args.beta}")
    print(f"  sigma_W:        {args.sigma_w}")
    print(f"  Smooth sigma:   {args.smooth_sigma}")
    print(f"  Seed:           {args.seed}")
    print(f"  Output root:    {out_dir}")

    domain = load_domain()
    coarse_lat = domain["coarse_lat"]
    coarse_lon = domain["coarse_lon"]
    fine_lat = domain["fine_lat"]
    fine_lon = domain["fine_lon"]
    fine_mask = domain["fine_mask"]

    n_clat_full, n_clon_full = len(coarse_lat), len(coarse_lon)
    clat_sl, clon_sl = get_tile_slices(args.tile, n_clat_full, n_clon_full)
    flat_sl, flon_sl = coarse_to_fine_slices(
        clat_sl, clon_sl, coarse_lat, coarse_lon, fine_lat, fine_lon
    )

    tile_clat = coarse_lat[clat_sl]
    tile_clon = coarse_lon[clon_sl]
    tile_flat = fine_lat[flat_sl]
    tile_flon = fine_lon[flon_sl]
    tile_fmask = fine_mask[flat_sl, flon_sl]
    n_clat = len(tile_clat)
    n_clon = len(tile_clon)
    n_flat = len(tile_flat)
    n_flon = len(tile_flon)

    from build_support_operators import build_Ac, build_Ic

    imerg_precip, imerg_error, canon_times = load_imerg_3h()
    tpw_t0 = load_tpw_3h()
    tile_imerg = imerg_precip[:, clat_sl, clon_sl]
    tile_error = imerg_error[:, clat_sl, clon_sl]
    tile_tpw = tpw_t0[:, flat_sl, flon_sl]

    tile_Ac = build_Ac(tile_flat, tile_flon, tile_clat, tile_clon)
    tile_Ic = build_Ic(tile_flat, tile_flon, tile_clat, tile_clon)
    n_t = tile_imerg.shape[0]

    print(f"  Coarse tile:    {n_clat} x {n_clon} = {n_clat * n_clon}")
    print(f"  Fine tile:      {n_flat} x {n_flon} = {n_flat * n_flon}")
    print(f"  Time steps:     {n_t}")

    if args.dry_run:
        return 0

    print("\n[1/6] Generating orographic proxy...")
    h_norm_fine = generate_orographic_proxy(tile_tpw, tile_fmask)
    h_coarse_flat = np.array(tile_Ac @ h_norm_fine.ravel())
    h_norm_coarse = h_coarse_flat.reshape(n_clat, n_clon).astype(np.float32)

    print("\n[2/6] Generating reference P_true...")
    p_true = generate_P_true_fast(
        tile_imerg,
        tile_Ic,
        tile_Ac,
        h_norm_fine,
        n_clat,
        n_clon,
        n_flat,
        n_flon,
        args.beta,
        rng,
    )

    print("\n[3/6] Generating coarse residual R_app_true...")
    r_app_true, delta = generate_R_app_true(
        p_true, tile_Ac, n_clat, n_clon, h_norm_coarse, canon_times
    )

    print("\n[4/6] Generating unresolved fine-grid mismatch Q...")
    exact_dw = np.zeros_like(p_true)
    for t in range(n_t):
        r_fine = np.array(tile_Ic @ r_app_true[t].ravel(), dtype=np.float32).reshape(
            n_flat, n_flon
        )
        exact_dw[t] = r_fine - p_true[t]

    q_unresolved = _generate_unresolved_term(
        n_t=n_t,
        n_flat=n_flat,
        n_flon=n_flon,
        Ac=tile_Ac,
        fine_mask=tile_fmask,
        h_norm_fine=h_norm_fine,
        base_dw=exact_dw,
        epsilon=args.epsilon,
        sigma=args.smooth_sigma,
        rng=rng,
        mode=args.mismatch_mode,
    )

    coarse_q = np.array(
        [tile_Ac @ q_unresolved[t].ravel() for t in range(n_t)],
        dtype=np.float32,
    ).reshape(n_t, n_clat, n_clon)
    target_q_rms = float(args.epsilon * np.mean(np.abs(exact_dw)))
    if np.any(tile_fmask):
        realized_q_rms = float(np.sqrt(np.mean(q_unresolved[:, tile_fmask] ** 2)))
    else:
        realized_q_rms = 0.0
    print(f"    target mismatch RMS: {target_q_rms:.4f} mm")
    print(f"    realized Q RMS:      {realized_q_rms:.4f} mm")
    print(f"    max|A_c Q|:          {np.max(np.abs(coarse_q)):.3e}")

    print("\n[5/6] Forward-integrating W_true with mismatch...")
    w_init = tile_tpw[0].copy()
    valid_w = ~np.isnan(w_init)
    if np.any(valid_w):
        w_init[~valid_w] = np.nanmean(w_init)
    else:
        w_init[:] = 15.0
    w_true = _forward_integrate_w_misspecified(
        w_init, p_true, r_app_true, q_unresolved, tile_Ic, n_flat, n_flon
    )

    print("\n[6/6] Adding observation noise...")
    w_obs, p_c_obs = add_observation_noise(
        w_true,
        p_true,
        tile_Ac,
        tile_error,
        tile_fmask,
        n_clat,
        n_clon,
        n_flat,
        n_flon,
        args.sigma_w,
        rng,
    )

    out_dir.mkdir(parents=True, exist_ok=True)

    truth_path = out_dir / f"synthetic_truth_{args.tile}.nc"
    with Dataset(truth_path, "w", format="NETCDF4") as ds:
        ds.createDimension("time", n_t)
        ds.createDimension("time_plus1", n_t + 1)
        ds.createDimension("coarse_lat", n_clat)
        ds.createDimension("coarse_lon", n_clon)
        ds.createDimension("fine_lat", n_flat)
        ds.createDimension("fine_lon", n_flon)

        tv = ds.createVariable("time", "f8", ("time",))
        tv.units = "hours since 2020-01-01 00:00:00"
        tv.calendar = "standard"
        tv[:] = date2num(canon_times, units=tv.units, calendar=tv.calendar)

        for name, arr, dim in [
            ("coarse_lat", tile_clat, "coarse_lat"),
            ("coarse_lon", tile_clon, "coarse_lon"),
            ("fine_lat", tile_flat, "fine_lat"),
            ("fine_lon", tile_flon, "fine_lon"),
        ]:
            v = ds.createVariable(name, "f8", (dim,))
            v[:] = arr

        v = ds.createVariable("P_true", "f4", ("time", "fine_lat", "fine_lon"), zlib=True, complevel=4)
        v[:] = p_true
        v.units = "mm/3h"
        v.long_name = "True fine-grid precipitation"

        v = ds.createVariable("R_app_true", "f4", ("time", "coarse_lat", "coarse_lon"), zlib=True, complevel=4)
        v[:] = r_app_true
        v.units = "mm/3h"
        v.long_name = "True coarse-grid closure residual"

        v = ds.createVariable("Q_unresolved_true", "f4", ("time", "fine_lat", "fine_lon"), zlib=True, complevel=4)
        v[:] = q_unresolved
        v.units = "mm/3h"
        v.long_name = "Unresolved fine-grid budget term omitted from the reduced training relation"

        v = ds.createVariable("W_true", "f4", ("time_plus1", "fine_lat", "fine_lon"), zlib=True, complevel=4)
        v[:] = w_true
        v.units = "kg/m2"
        v.long_name = "True fine-grid TPW"

        v = ds.createVariable("fine_mask", "i1", ("fine_lat", "fine_lon"), zlib=True)
        v[:] = tile_fmask.astype(np.int8)

        ds.title = f"Misspecified synthetic twin ground truth - tile {args.tile}"
        ds.tile = args.tile
        ds.seed = args.seed
        ds.beta = args.beta
        ds.sigma_w = args.sigma_w
        ds.mismatch_mode = args.mismatch_mode
        ds.mismatch_epsilon = args.epsilon
        ds.contract = "misspecified"
        ds.source_script = "synthetic_twin_generate_misspecified.py"
        ds.description = (
            "Truth violates dW = I_c(R_app) - P by adding Q_unresolved on the fine grid."
        )

    obs_path = out_dir / f"synthetic_obs_{args.tile}.nc"
    with Dataset(obs_path, "w", format="NETCDF4") as ds:
        ds.createDimension("time", n_t)
        ds.createDimension("coarse_lat", n_clat)
        ds.createDimension("coarse_lon", n_clon)
        ds.createDimension("fine_lat", n_flat)
        ds.createDimension("fine_lon", n_flon)

        tv = ds.createVariable("time", "f8", ("time",))
        tv.units = "hours since 2020-01-01 00:00:00"
        tv.calendar = "standard"
        tv[:] = date2num(canon_times, units=tv.units, calendar=tv.calendar)

        for name, arr, dim in [
            ("coarse_lat", tile_clat, "coarse_lat"),
            ("coarse_lon", tile_clon, "coarse_lon"),
            ("fine_lat", tile_flat, "fine_lat"),
            ("fine_lon", tile_flon, "fine_lon"),
        ]:
            v = ds.createVariable(name, "f8", (dim,))
            v[:] = arr

        v = ds.createVariable("W_obs", "f4", ("time", "fine_lat", "fine_lon"), zlib=True, complevel=4)
        v[:] = w_obs
        v.units = "kg/m2"

        v = ds.createVariable("P_c_obs", "f4", ("time", "coarse_lat", "coarse_lon"), zlib=True, complevel=4)
        v[:] = p_c_obs
        v.units = "mm/3h"

        ds.title = f"Misspecified synthetic twin observations - tile {args.tile}"
        ds.tile = args.tile
        ds.seed = args.seed
        ds.mismatch_mode = args.mismatch_mode
        ds.mismatch_epsilon = args.epsilon
        ds.contract = "misspecified"
        ds.source_script = "synthetic_twin_generate_misspecified.py"

    print("\nDiagnostics")
    print(f"  max|A_c[Q_unresolved]|: {np.max(np.abs(coarse_q)):.3e}")
    print(f"  mean|exact dW|:         {np.mean(np.abs(exact_dw)):.4f} mm")
    print(f"  mean|Q_unresolved|:     {np.mean(np.abs(q_unresolved)):.4f} mm")
    print(f"  W_true range:           [{w_true.min():.3f}, {w_true.max():.3f}]")
    print(f"  Truth file:             {truth_path}")
    print(f"  Obs file:               {obs_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
