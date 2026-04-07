#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generate synthetic twin data for the support-explicit inverse study.

Creates a synthetic truth and corresponding noisy observations for a selected
tile:

  1. ``P_true``: fine-grid precipitation with subgrid orographic structure
  2. ``R_app_true``: coarse-support apparent closure residual
  3. ``W_true``: fine-grid TPW proxy obtained by budget integration
  4. Synthetic observations with prescribed noise

Uses the simplified surrogate budget ``dW/dt = R_app - P``.

Outputs are written to the selected synthetic output directory:
    ``synthetic_truth_<tile>.nc``: ``P_true``, ``R_app_true``, ``W_true``
    ``synthetic_obs_<tile>.nc``: ``W_obs``, ``P_c_obs`` with noise

Usage:
    python synthetic_twin_generate.py [--tile full|A_wet|C_mid|B_dry] [--beta 0.3]
                                      [--sigma-w 1.0] [--seed 42]
"""

from __future__ import annotations

import argparse
import sys
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
from scipy import sparse
from scipy.ndimage import gaussian_filter
from netCDF4 import Dataset, date2num, num2date
from tqdm import tqdm

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
CODE_DIR = Path(__file__).resolve().parent
REPO_ROOT = CODE_DIR.parent
PREPROC = REPO_ROOT / "preprocessed"
OUT_DIR = PREPROC / "synthetic"

# ---------------------------------------------------------------------------
# Tile definitions (coarse grid indices, inclusive)
# ---------------------------------------------------------------------------
TILES = {
    "full": {
        "clat": (0, None),  # all
        "clon": (0, None),
    },
    "A_wet": {
        # 28-32 deg N, 85-95 deg E -> coarse indices (lat 25.05 + idx*0.1)
        # lat idx 30..70 ~= 28.05-32.05, lon idx 150..250 ~= 85.05-95.05
        # Southeast TP, high precipitation, monsoon-influenced
        "clat": (30, 70),
        "clon": (150, 250),
    },
    "B_dry": {
        # 33-37 deg N, 80-90 deg E
        # lat idx 80..120 ~= 33.05-37.05, lon idx 100..200 ~= 80.05-90.05
        # Northwest TP, arid, low signal regime
        "clat": (80, 120),
        "clon": (100, 200),
    },
    "C_mid": {
        # 30-34 deg N, 90-100 deg E
        # lat idx 50..90 ~= 30.05-34.05, lon idx 200..300 ~= 90.05-100.05
        # Central-eastern TP, intermediate-signal robustness tile chosen to
        # sit between the wettest and driest showcase tiles in the current
        # study domain.
        "clat": (50, 90),
        "clon": (200, 300),
    },
}

# ---------------------------------------------------------------------------
# Physical constants
# ---------------------------------------------------------------------------
DT_3H = 10800.0  # seconds
MM_DAY_TO_KG_M2_S = 1.0 / 86400.0  # mm/day -> kg m^-2 s^-1

# ---------------------------------------------------------------------------
# Load preprocessed data
# ---------------------------------------------------------------------------
def load_domain(preproc_dir: Path = PREPROC):
    ds = Dataset(preproc_dir / "common_domain.nc", "r")
    d = {
        "coarse_lat": np.array(ds.variables["coarse_lat"][:], dtype=np.float64),
        "coarse_lon": np.array(ds.variables["coarse_lon"][:], dtype=np.float64),
        "fine_lat": np.array(ds.variables["fine_lat"][:], dtype=np.float64),
        "fine_lon": np.array(ds.variables["fine_lon"][:], dtype=np.float64),
        "fine_mask": np.array(ds.variables["fine_mask"][:], dtype=bool),
        "coarse_mask": np.array(ds.variables["coarse_mask"][:], dtype=bool),
    }
    ds.close()
    return d


def load_operators(preproc_dir: Path = PREPROC):
    npz = np.load(preproc_dir / "support_operators.npz")
    Ac = sparse.csr_matrix((npz["Ac_data"], npz["Ac_indices"], npz["Ac_indptr"]),
                           shape=tuple(npz["Ac_shape"]))
    Ic = sparse.csr_matrix((npz["Ic_data"], npz["Ic_indices"], npz["Ic_indptr"]),
                           shape=tuple(npz["Ic_shape"]))
    return Ac, Ic


def load_imerg_3h(preproc_dir: Path = PREPROC):
    ds = Dataset(preproc_dir / "imerg_3h.nc", "r")
    precip = np.array(ds.variables["precipitation_3h"][:])  # (T, clat, clon) mm
    error = np.array(ds.variables["randomError_3h"][:])
    tvar = ds.variables["time"]
    times = num2date(tvar[:], units=tvar.units, calendar=tvar.calendar)
    times = [datetime(t.year, t.month, t.day, t.hour, t.minute, t.second)
             for t in times]
    ds.close()
    return precip, error, times


def load_tpw_3h(preproc_dir: Path = PREPROC):
    ds = Dataset(preproc_dir / "tpw_3h.nc", "r")
    tpw_t0 = np.array(ds.variables["tpw_t0"][:])  # (T, flat, flon)
    ds.close()
    return tpw_t0


# ---------------------------------------------------------------------------
# Tile slicing helpers
# ---------------------------------------------------------------------------
def get_tile_slices(tile_name: str, n_clat: int, n_clon: int):
    """Return (clat_slice, clon_slice) for the tile."""
    t = TILES[tile_name]
    clat_start, clat_end = t["clat"]
    clon_start, clon_end = t["clon"]
    if clat_end is None:
        clat_end = n_clat
    if clon_end is None:
        clon_end = n_clon
    return slice(clat_start, clat_end), slice(clon_start, clon_end)


def coarse_to_fine_slices(clat_sl, clon_sl, coarse_lat, coarse_lon,
                          fine_lat, fine_lon):
    """
    Given coarse grid slices, find corresponding fine grid slices
    that cover the same spatial extent (plus half a coarse cell margin).
    """
    c_lat_min = coarse_lat[clat_sl][0] - 0.05
    c_lat_max = coarse_lat[clat_sl][-1] + 0.05
    c_lon_min = coarse_lon[clon_sl][0] - 0.05
    c_lon_max = coarse_lon[clon_sl][-1] + 0.05

    flat_idx = np.where((fine_lat >= c_lat_min - 0.01) &
                        (fine_lat <= c_lat_max + 0.01))[0]
    flon_idx = np.where((fine_lon >= c_lon_min - 0.01) &
                        (fine_lon <= c_lon_max + 0.01))[0]
    return slice(flat_idx[0], flat_idx[-1] + 1), \
           slice(flon_idx[0], flon_idx[-1] + 1)


# ---------------------------------------------------------------------------
# Synthetic ground truth generation
# ---------------------------------------------------------------------------
def generate_orographic_proxy(tpw_t0: np.ndarray, fine_mask: np.ndarray) -> np.ndarray:
    """
    Create an orographic enhancement proxy from TPW climatology.
    Lower mean TPW corresponds to higher elevation and stronger enhancement.
    Returns h_norm in [0, 1] on the fine grid.
    """
    # Mean TPW over time (ignoring NaN)
    with np.errstate(invalid="ignore"):
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message="Mean of empty slice",
                category=RuntimeWarning,
            )
            tpw_mean = np.nanmean(tpw_t0, axis=0)

    # Fill NaN with spatial mean for smooth field
    valid = ~np.isnan(tpw_mean)
    if np.any(valid):
        fill_val = np.nanmean(tpw_mean)
        tpw_filled = np.where(valid, tpw_mean, fill_val)
    else:
        tpw_filled = np.full_like(tpw_mean, 15.0)

    # Smooth
    tpw_smooth = gaussian_filter(tpw_filled, sigma=10)

    # Invert and normalize: high elevation = low TPW
    h = 1.0 - (tpw_smooth - np.min(tpw_smooth)) / \
        (np.max(tpw_smooth) - np.min(tpw_smooth) + 1e-10)
    return h.astype(np.float32)


def generate_P_true(
    imerg_precip: np.ndarray,
    Ic: sparse.csr_matrix,
    Ac: sparse.csr_matrix,
    h_norm_fine: np.ndarray,
    fine_mask: np.ndarray,
    n_clat: int, n_clon: int,
    n_flat: int, n_flon: int,
    beta: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Generate P_true on the fine grid with sub-pixel orographic structure.

    Steps:
    1. Lift IMERG coarse precip to fine grid via I_c
    2. Add sub-pixel orographic modulation
    3. Enforce non-negativity
    4. Renormalize so A_c[P_true] = P_IMERG (mass conservation)
    """
    n_t = imerg_precip.shape[0]
    P_true = np.zeros((n_t, n_flat, n_flon), dtype=np.float32)

    for t in tqdm(range(n_t), desc="P_true generation"):
        # Coarse precip for this timestep
        p_coarse = imerg_precip[t]  # (clat, clon) in mm

        # Lift to fine grid
        p_coarse_flat = p_coarse.ravel()
        p_fine_flat = Ic @ p_coarse_flat  # (n_fine,)
        p_fine = p_fine_flat.reshape(n_flat, n_flon)

        # Generate smooth random field for sub-pixel variability
        noise = rng.standard_normal((n_flat, n_flon)).astype(np.float32)
        noise_smooth = gaussian_filter(noise, sigma=5)
        # Normalize to [-1, 1]
        noise_smooth = noise_smooth / (np.max(np.abs(noise_smooth)) + 1e-10)

        # Orographic modulation
        modulation = 1.0 + beta * h_norm_fine * noise_smooth

        # Apply modulation only where there's precipitation
        p_enriched = p_fine * modulation

        # Non-negativity
        p_enriched = np.maximum(p_enriched, 0.0)

        # Renormalize: ensure A_c[P_true] = P_IMERG (mass conservation)
        p_enriched_flat = p_enriched.ravel()
        p_avg = Ac @ p_enriched_flat  # (n_coarse,)
        p_avg = p_avg.reshape(n_clat, n_clon)

        # For each coarse cell, scale fine pixels to match
        for cj in range(n_clat):
            for ci in range(n_clon):
                if p_avg[cj, ci] > 1e-10:
                    # Find fine pixels in this coarse cell via I_c
                    # Scale uniformly
                    scale = p_coarse[cj, ci] / p_avg[cj, ci]
                else:
                    scale = 1.0

                # We need to know which fine pixels belong to this coarse cell
                # Use I_c: fine pixels that map to coarse cell (cj, ci)
                c_flat = cj * n_clon + ci
                # Get fine pixels mapping to this coarse cell from I_c columns
                fi_rows = Ic.getcol(c_flat).nonzero()[0]
                if len(fi_rows) > 0:
                    fj_arr = fi_rows // n_flon
                    fi_arr = fi_rows % n_flon
                    p_enriched[fj_arr, fi_arr] *= scale

        P_true[t] = p_enriched

    return P_true


def generate_P_true_fast(
    imerg_precip: np.ndarray,
    Ic: sparse.csr_matrix,
    Ac: sparse.csr_matrix,
    h_norm_fine: np.ndarray,
    n_clat: int, n_clon: int,
    n_flat: int, n_flon: int,
    beta: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Fast vectorized version of P_true generation.
    Uses precomputed coarse-to-fine mapping for renormalization.
    """
    n_t = imerg_precip.shape[0]
    n_fine = n_flat * n_flon
    n_coarse = n_clat * n_clon

    # Precompute: for each fine pixel, which coarse cell does it belong to?
    # From I_c: each row (fine pixel) has exactly one non-zero column (coarse cell)
    fine_to_coarse = np.array(Ic.argmax(axis=1)).ravel()  # (n_fine,)

    P_true = np.zeros((n_t, n_flat, n_flon), dtype=np.float32)

    for t in tqdm(range(n_t), desc="P_true (fast)"):
        p_coarse = imerg_precip[t]  # (clat, clon) mm
        p_coarse_flat = p_coarse.ravel()

        # Lift to fine
        p_fine_flat = np.array(Ic @ p_coarse_flat, dtype=np.float32)
        p_fine = p_fine_flat.reshape(n_flat, n_flon)

        # Smooth random field for sub-pixel variability
        noise = rng.standard_normal((n_flat, n_flon)).astype(np.float32)
        noise_smooth = gaussian_filter(noise, sigma=5)
        noise_smooth /= (np.max(np.abs(noise_smooth)) + 1e-10)

        # Modulate
        p_enriched = p_fine * (1.0 + beta * h_norm_fine * noise_smooth)
        p_enriched = np.maximum(p_enriched, 0.0)

        # Renormalize to preserve coarse-scale mass
        p_enriched_flat = p_enriched.ravel()
        p_avg_flat = np.array(Ac @ p_enriched_flat)  # (n_coarse,)

        # Compute per-coarse-cell scale factor
        scale = np.ones(n_coarse, dtype=np.float64)
        nonzero = p_avg_flat > 1e-10
        scale[nonzero] = p_coarse_flat[nonzero] / p_avg_flat[nonzero]

        # Apply scale to each fine pixel based on its coarse cell
        pixel_scale = scale[fine_to_coarse].astype(np.float32)
        p_enriched_flat *= pixel_scale

        P_true[t] = p_enriched_flat.reshape(n_flat, n_flon)

    return P_true


def generate_R_app_true(
    P_true: np.ndarray,
    Ac: sparse.csr_matrix,
    n_clat: int, n_clon: int,
    h_norm_coarse: np.ndarray,
    times: list[datetime],
    season_start: datetime | None = None,
) -> np.ndarray:
    """
    Generate R_app_true on coarse grid as A_c[P_true] plus a smooth perturbation.

    By setting R_app approximately equal to A_c[P_true] plus a perturbation,
    we ensure:
      - W_true doesn't drift to zero or blow up
      - the perturbation is a smooth, physically plausible closure residual

    The perturbation has:
      - Spatial pattern from elevation proxy (2-5 mm/day equivalent)
      - Diurnal cycle (peak afternoon)
      - 10-day monsoon modulation

    Units: mm per 3h.
    """
    n_t = P_true.shape[0]
    n_flat, n_flon = P_true.shape[1], P_true.shape[2]

    # Compute A_c[P_true] for each timestep
    P_coarse_mean = np.zeros((n_t, n_clat, n_clon), dtype=np.float32)
    for t in range(n_t):
        P_coarse_mean[t] = np.array(
            Ac @ P_true[t].ravel(), dtype=np.float32
        ).reshape(n_clat, n_clon)

    # Perturbation: zero-mean oscillation so cumulative W drift stays near 0.
    # Spatial amplitude: 0.1-0.6 mm/3h (= 0.8-4.8 mm/day peak-to-peak)
    delta_amp = 0.1 + 0.5 * (1.0 - h_norm_coarse)  # (clat, clon) mm/3h

    if season_start is None:
        season_start = datetime(times[0].year, 5, 1)

    delta = np.zeros((n_t, n_clat, n_clon), dtype=np.float32)
    for t_idx in range(n_t):
        ts = times[t_idx]
        hour = ts.hour + ts.minute / 60.0

        # Diurnal cycle (zero-mean): peak at 14:00 local (UTC+6 for TP)
        local_hour = (hour + 6) % 24
        diurnal = np.sin(2 * np.pi * local_hour / 24 - np.pi / 2)

        # Monsoon modulation (zero-mean): 10-day cycle
        day_of_season = (ts - season_start).total_seconds() / 86400.0
        monsoon = np.sin(2 * np.pi * day_of_season / 10)

        # Combine: sum of two zero-mean sinusoids times the spatial amplitude
        delta[t_idx] = (delta_amp * (0.7 * diurnal + 0.3 * monsoon)).astype(
            np.float32
        )

    R_app = P_coarse_mean + delta
    return R_app, delta


def forward_integrate_W(
    W_init: np.ndarray,
    P_true: np.ndarray,
    R_app_true: np.ndarray,
    Ic: sparse.csr_matrix,
    n_clat: int, n_clon: int,
    n_flat: int, n_flon: int,
) -> np.ndarray:
    """
    Forward-integrate the simplified budget: dW/dt = R_app - P.

    W(t+1) = W(t) + I_c(R_app(t)) - P(t)

    **No clamping, no nudging**: budget closes exactly by construction.
    W_init must be chosen large enough to prevent W going negative.

    All in mm units: P_true is mm/3h, R_app_true is mm/3h.
    dW is therefore in mm (= kg/m^2) per 3h step.
    """
    n_t = P_true.shape[0]
    W = np.zeros((n_t + 1, n_flat, n_flon), dtype=np.float32)
    W[0] = W_init.copy()

    for t in tqdm(range(n_t), desc="W forward integration"):
        # Lift R_app to fine grid
        r_coarse_flat = R_app_true[t].ravel()
        r_fine_flat = np.array(Ic @ r_coarse_flat, dtype=np.float32)
        r_fine = r_fine_flat.reshape(n_flat, n_flon)

        # Budget: dW = I_c(R_app) - P  (exact, no clamping)
        W[t + 1] = W[t] + r_fine - P_true[t]

    return W


def add_observation_noise(
    W_true: np.ndarray,
    P_true: np.ndarray,
    Ac: sparse.csr_matrix,
    imerg_error: np.ndarray,
    fine_mask: np.ndarray,
    n_clat: int, n_clon: int,
    n_flat: int, n_flon: int,
    sigma_w: float,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Create synthetic observations with realistic noise.

    W_obs = W_true + epsilon_W,  epsilon_W ~ N(0, sigma_W^2)
    P_c_obs = A_c[P_true] + epsilon_P,  epsilon_P ~ N(0, sigma_P^2)
    """
    n_t = W_true.shape[0] - 1  # W_true has n_t+1 snapshots

    # W observations: add noise, apply mask
    W_obs = np.zeros((n_t, n_flat, n_flon), dtype=np.float32)
    for t in range(n_t):
        noise = rng.normal(0, sigma_w, (n_flat, n_flon)).astype(np.float32)
        W_obs[t] = W_true[t] + noise
        W_obs[t][~fine_mask] = np.nan  # Apply TPW mask

    # P_c observations: A_c[P_true] + noise
    P_c_obs = np.zeros((n_t, n_clat, n_clon), dtype=np.float32)
    for t in range(n_t):
        p_fine_flat = P_true[t].ravel()
        p_coarse_flat = np.array(Ac @ p_fine_flat)
        p_coarse = p_coarse_flat.reshape(n_clat, n_clon)

        # Noise from IMERG randomError (scaled, already in mm/3h)
        sigma_p = np.maximum(imerg_error[t], 0.1)  # floor at 0.1 mm
        noise_p = rng.normal(0, 1, (n_clat, n_clon)).astype(np.float32) * sigma_p
        P_c_obs[t] = np.maximum(p_coarse + noise_p, 0.0)

    return W_obs, P_c_obs


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Generate synthetic twin experiment data"
    )
    parser.add_argument("--tile", choices=list(TILES.keys()), default="A_wet",
                        help="Spatial tile (default: A_wet)")
    parser.add_argument("--beta", type=float, default=0.3,
                        help="Sub-pixel variability amplitude (default: 0.3)")
    parser.add_argument("--sigma-w", type=float, default=1.0,
                        help="TPW observation noise std (kg/m2, default: 1.0)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed (default: 42)")
    parser.add_argument(
        "--preproc-dir",
        type=Path,
        default=PREPROC,
        help="Directory containing common_domain.nc, support_operators.npz, imerg_3h.nc, and tpw_3h.nc.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Directory for synthetic outputs. Defaults to <preproc-dir>/synthetic.",
    )
    parser.add_argument("--dry-run", action="store_true",
                        help="Print configuration without generating")
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    preproc_dir = args.preproc_dir
    out_dir = args.out_dir or (preproc_dir / "synthetic")

    print("=" * 60)
    print("  Synthetic Twin - Data Generation")
    print("=" * 60)
    print(f"  Tile: {args.tile}, beta={args.beta}, sigma_W={args.sigma_w}, seed={args.seed}")

    # Load data
    print("\n[1/7] Loading preprocessed data...")
    domain = load_domain(preproc_dir)
    Ac_full, Ic_full = load_operators(preproc_dir)
    imerg_precip, imerg_error, canon_times = load_imerg_3h(preproc_dir)
    tpw_t0 = load_tpw_3h(preproc_dir)

    coarse_lat = domain["coarse_lat"]
    coarse_lon = domain["coarse_lon"]
    fine_lat = domain["fine_lat"]
    fine_lon = domain["fine_lon"]
    fine_mask = domain["fine_mask"]

    n_clat_full, n_clon_full = len(coarse_lat), len(coarse_lon)
    n_flat_full, n_flon_full = len(fine_lat), len(fine_lon)
    n_t = len(canon_times)

    # Tile slicing
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

    print(f"  Coarse tile: {n_clat} x {n_clon} = {n_clat * n_clon} cells")
    print(f"  Fine tile:   {n_flat} x {n_flon} = {n_flat * n_flon} pixels")
    print(f"  Time steps:  {n_t}")

    # Slice IMERG data to tile
    tile_imerg = imerg_precip[:, clat_sl, clon_sl]
    tile_error = imerg_error[:, clat_sl, clon_sl]

    # Slice TPW data to tile
    tile_tpw = tpw_t0[:, flat_sl, flon_sl]

    if args.dry_run:
        print("\n  [DRY RUN] Configuration verified. Exiting.")
        return 0

    time_units = f"hours since {canon_times[0].year}-01-01 00:00:00"
    season_start = datetime(canon_times[0].year, 5, 1)

    # Build tile-local operators
    print("\n[2/7] Building tile-local operators...")
    from build_support_operators import build_Ac, build_Ic
    tile_Ac = build_Ac(tile_flat, tile_flon, tile_clat, tile_clon)
    tile_Ic = build_Ic(tile_flat, tile_flon, tile_clat, tile_clon)
    print(f"  A_c: {tile_Ac.shape}, nnz={tile_Ac.nnz}")
    print(f"  I_c: {tile_Ic.shape}, nnz={tile_Ic.nnz}")

    # Orographic proxy
    print("\n[3/7] Generating orographic proxy...")
    h_norm_fine = generate_orographic_proxy(tile_tpw, tile_fmask)
    # Coarse version for R_app
    h_fine_flat = h_norm_fine.ravel()
    h_coarse_flat = np.array(tile_Ac @ h_fine_flat)
    h_norm_coarse = h_coarse_flat.reshape(n_clat, n_clon).astype(np.float32)
    print(f"  h_norm fine range:   [{h_norm_fine.min():.3f}, {h_norm_fine.max():.3f}]")
    print(f"  h_norm coarse range: [{h_norm_coarse.min():.3f}, {h_norm_coarse.max():.3f}]")

    # Generate P_true first (needed for R_app derivation)
    print("\n[4/7] Generating P_true (with sub-pixel structure)...")
    P_true = generate_P_true_fast(
        tile_imerg, tile_Ic, tile_Ac, h_norm_fine,
        n_clat, n_clon, n_flat, n_flon, args.beta, rng
    )
    print(f"  P_true range: [{P_true.min():.3f}, {P_true.max():.3f}] mm/3h")
    print(f"  P_true mean:  {P_true.mean():.4f} mm/3h")

    # Verify coarse consistency
    p_verify = np.array(tile_Ac @ P_true[0].ravel()).reshape(n_clat, n_clon)
    max_diff = np.max(np.abs(p_verify - tile_imerg[0]))
    print(f"  Coarse consistency (t=0): max|A_c[P_true] - P_IMERG| = {max_diff:.6f} mm")

    # Generate R_app_true = A_c[P_true] + delta (ensures bounded W drift)
    print("\n[5/7] Generating R_app_true = A_c[P_true] + perturbation...")
    R_app_true, delta = generate_R_app_true(
        P_true, tile_Ac, n_clat, n_clon, h_norm_coarse, canon_times, season_start
    )
    print(f"  R_app range: [{R_app_true.min():.3f}, {R_app_true.max():.3f}] mm/3h")
    print(f"  R_app mean:  {R_app_true.mean():.3f} mm/3h "
          f"(= {R_app_true.mean() * 8:.1f} mm/day)")
    print(f"  Perturbation range: [{delta.min():.3f}, {delta.max():.3f}] mm/3h")
    print(f"  Perturbation mean:  {delta.mean():.3f} mm/3h "
          f"(= {delta.mean() * 8:.1f} mm/day)")

    # Forward-integrate W (NO clamping, NO nudging - exact budget closure)
    print("\n[6/7] Forward-integrating W_true (exact budget, no clamp/nudge)...")
    # Initial condition: real TPW at t=0 (fill NaN with spatial mean)
    W_init = tile_tpw[0].copy()
    valid_w = ~np.isnan(W_init)
    if np.any(valid_w):
        W_init[~valid_w] = np.nanmean(W_init)
    else:
        W_init[:] = 15.0

    W_true = forward_integrate_W(
        W_init, P_true, R_app_true, tile_Ic,
        n_clat, n_clon, n_flat, n_flon,
    )
    print(f"  W_true range: [{W_true.min():.2f}, {W_true.max():.2f}] kg/m2")
    print(f"  W_true mean:  {W_true.mean():.2f} kg/m2")
    n_negative = np.sum(W_true < 0)
    if n_negative > 0:
        print(f"  WARNING: {n_negative} negative W values "
              f"({100*n_negative/W_true.size:.2f}%)")

    # Generate observations
    print("\n[7/7] Adding observation noise...")
    W_obs, P_c_obs = add_observation_noise(
        W_true, P_true, tile_Ac, tile_error, tile_fmask,
        n_clat, n_clon, n_flat, n_flon,
        args.sigma_w, rng,
    )
    print(f"  W_obs valid: {100 * np.sum(~np.isnan(W_obs[0])) / W_obs[0].size:.1f}%")
    print(f"  P_c_obs range: [{P_c_obs.min():.3f}, {P_c_obs.max():.3f}] mm/3h")

    # Save
    print("\nSaving outputs...")
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- synthetic_truth.nc ---
    truth_path = out_dir / f"synthetic_truth_{args.tile}.nc"
    with Dataset(truth_path, "w", format="NETCDF4") as ds:
        ds.createDimension("time", n_t)
        ds.createDimension("time_plus1", n_t + 1)
        ds.createDimension("coarse_lat", n_clat)
        ds.createDimension("coarse_lon", n_clon)
        ds.createDimension("fine_lat", n_flat)
        ds.createDimension("fine_lon", n_flon)

        tv = ds.createVariable("time", "f8", ("time",))
        tv.units = time_units
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

        v = ds.createVariable("P_true", "f4",
                              ("time", "fine_lat", "fine_lon"),
                              zlib=True, complevel=4,
                              chunksizes=(1, n_flat, n_flon))
        v.units = "mm/3h"
        v.long_name = "True fine-grid precipitation (3h accumulation)"
        v[:] = P_true

        v = ds.createVariable("R_app_true", "f4",
                              ("time", "coarse_lat", "coarse_lon"),
                              zlib=True, complevel=4)
        v.units = "mm/3h"
        v.long_name = "True coarse-grid closure residual (3h)"
        v[:] = R_app_true

        v = ds.createVariable("W_true", "f4",
                              ("time_plus1", "fine_lat", "fine_lon"),
                              zlib=True, complevel=4,
                              chunksizes=(1, n_flat, n_flon))
        v.units = "kg/m2"
        v.long_name = "True fine-grid TPW (budget-integrated)"
        v[:] = W_true

        v = ds.createVariable("fine_mask", "i1", ("fine_lat", "fine_lon"),
                              zlib=True)
        v[:] = tile_fmask.astype(np.int8)

        v = ds.createVariable("h_norm_fine", "f4", ("fine_lat", "fine_lon"),
                              zlib=True)
        v.long_name = "Orographic enhancement proxy (fine grid)"
        v[:] = h_norm_fine

        ds.title = f"Synthetic twin ground truth - tile {args.tile}"
        ds.beta = args.beta
        ds.seed = args.seed
        ds.tile = args.tile
        ds.budget = "dW/dt = R_app - P (v1 simplified, no Q)"
        ds.source_script = "synthetic_twin_generate.py"

    print(f"  Truth: {truth_path} ({truth_path.stat().st_size / 1024 / 1024:.1f} MB)")

    # --- synthetic_obs.nc ---
    obs_path = out_dir / f"synthetic_obs_{args.tile}.nc"
    with Dataset(obs_path, "w", format="NETCDF4") as ds:
        ds.createDimension("time", n_t)
        ds.createDimension("coarse_lat", n_clat)
        ds.createDimension("coarse_lon", n_clon)
        ds.createDimension("fine_lat", n_flat)
        ds.createDimension("fine_lon", n_flon)

        tv = ds.createVariable("time", "f8", ("time",))
        tv.units = time_units
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

        v = ds.createVariable("W_obs", "f4",
                              ("time", "fine_lat", "fine_lon"),
                              zlib=True, complevel=4,
                              chunksizes=(1, n_flat, n_flon))
        v.units = "kg/m2"
        v.long_name = "Synthetic TPW observation (with noise and mask)"
        v.sigma_w = args.sigma_w
        v[:] = W_obs

        v = ds.createVariable("P_c_obs", "f4",
                              ("time", "coarse_lat", "coarse_lon"),
                              zlib=True, complevel=4)
        v.units = "mm/3h"
        v.long_name = "Synthetic coarse precipitation observation (with noise)"
        v[:] = P_c_obs

        ds.title = f"Synthetic twin observations - tile {args.tile}"
        ds.sigma_w = args.sigma_w
        ds.seed = args.seed
        ds.tile = args.tile
        ds.source_script = "synthetic_twin_generate.py"

    print(f"  Obs: {obs_path} ({obs_path.stat().st_size / 1024 / 1024:.1f} MB)")

    # Summary diagnostics
    print("\n" + "=" * 60)
    print("  DIAGNOSTICS")
    print("=" * 60)
    # Budget check: dW_true should equal R_app - P
    dW_true = W_true[1:] - W_true[:-1]  # (n_t, flat, flon)
    # Compute budget-implied dW
    dW_budget = np.zeros_like(dW_true)
    for t in range(n_t):
        r_coarse_flat = R_app_true[t].ravel()
        r_fine_flat = np.array(tile_Ic @ r_coarse_flat, dtype=np.float32)
        r_fine = r_fine_flat.reshape(n_flat, n_flon)
        dW_budget[t] = r_fine - P_true[t]

    budget_error = dW_true - dW_budget
    print(f"  Budget closure error (should be ~0 = float32 precision):")
    print(f"    max|dW - (I_c(R)-P)|:  {np.max(np.abs(budget_error)):.2e}")
    print(f"    mean|dW - (I_c(R)-P)|: {np.mean(np.abs(budget_error)):.2e}")

    print(f"\n  dW_true range: [{dW_true.min():.3f}, {dW_true.max():.3f}] mm")
    print(f"  Sigma_W noise:   {args.sigma_w} kg/m2")
    print(f"  Signal-to-noise: mean|dW|/sigma_W = "
          f"{np.mean(np.abs(dW_true)):.3f}/{args.sigma_w:.1f} = "
          f"{np.mean(np.abs(dW_true)) / args.sigma_w:.2f}")

    print("\n" + "=" * 60)
    print("  DONE")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())
