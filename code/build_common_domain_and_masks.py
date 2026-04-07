#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Build the common spatial domain and masks for the water budget closure project.

This script preserves the 2020 warm-season geometry chain as the default, but
it can also rebuild a second-year common domain. It does not
silently reuse the 2020 geometry; if fixed-geometry reuse is desired, that
should be an explicit downstream choice.

Outputs (saved to the selected output directory):
    common_domain.nc
        coarse_lat, coarse_lon
        fine_lat, fine_lon
        fine_mask
        coarse_mask
        coarse_valid_frac

Usage:
    python build_common_domain_and_masks.py
    python build_common_domain_and_masks.py --year 2021
    python build_common_domain_and_masks.py --year 2021 --output-dir "/path/to/preprocessed_2021"
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path

import numpy as np
from netCDF4 import Dataset


CODE_DIR = Path(__file__).resolve().parent
REPO_ROOT = CODE_DIR.parent

COMMON_LON_MIN = 70.05
COMMON_LON_MAX = 104.95
COMMON_LAT_MIN = 25.05
COMMON_LAT_MAX = 39.95

LEGACY_2020_START = date(2020, 5, 1)
LEGACY_2020_END = date(2020, 9, 30)

COARSE_VALID_THRESHOLD_DEFAULT = 0.6


@dataclass(frozen=True)
class GeometryConfig:
    year: int
    imerg_dir: Path
    tpw_dir: Path
    cmfd_dir: Path
    output_dir: Path
    threshold: float


def default_imerg_dir(year: int) -> Path:
    if year == 2020:
        return REPO_ROOT / "IMERG_V07B_TP_daily"
    return REPO_ROOT / f"IMERG_V07B_TP_{year}_May_Sep_daily"


def default_tpw_dir(year: int) -> Path:
    if year == 2020:
        legacy = REPO_ROOT / "Sun_TPW_2020_May_Sep"
        if legacy.exists():
            return legacy
    year_dir = REPO_ROOT / "Sun_TPW" / f"{year}"
    if year_dir.exists():
        return year_dir
    return REPO_ROOT / "Sun_TPW"


def default_cmfd_dir(year: int) -> Path:
    if year == 2020:
        legacy = REPO_ROOT / "CMFD_2020_May_Sep_All"
        if legacy.exists():
            return legacy
    return REPO_ROOT / "CMFD_v2.0"


def default_output_dir(year: int) -> Path:
    if year == 2020:
        return REPO_ROOT / "preprocessed"
    return REPO_ROOT / f"preprocessed_{year}"


def resolve_cmfd_file(cmfd_dir: Path, var: str, year: int, month: int) -> Path:
    month_token = f"{year}{month:02d}"
    fname = f"{var}_CMFD_V0200_B-01_03hr_010deg_{month_token}.nc"
    flat = cmfd_dir / fname
    if flat.exists():
        return flat
    return cmfd_dir / var / fname


def parse_args() -> GeometryConfig:
    parser = argparse.ArgumentParser(
        description="Build the common domain and masks for a selected warm-season year."
    )
    parser.add_argument(
        "--year",
        type=int,
        default=2020,
        help="Warm-season year to audit and rebuild. Default: 2020.",
    )
    parser.add_argument("--imerg-dir", type=Path, default=None)
    parser.add_argument("--tpw-dir", type=Path, default=None)
    parser.add_argument("--cmfd-dir", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument(
        "--threshold",
        type=float,
        default=COARSE_VALID_THRESHOLD_DEFAULT,
        help="Minimum fraction of valid fine pixels for a coarse cell to remain valid.",
    )
    args = parser.parse_args()

    return GeometryConfig(
        year=args.year,
        imerg_dir=args.imerg_dir or default_imerg_dir(args.year),
        tpw_dir=args.tpw_dir or default_tpw_dir(args.year),
        cmfd_dir=args.cmfd_dir or default_cmfd_dir(args.year),
        output_dir=args.output_dir or default_output_dir(args.year),
        threshold=args.threshold,
    )


def first_existing(path_iterable: list[Path], label: str) -> Path:
    for path in path_iterable:
        if path.exists():
            return path
    raise RuntimeError(f"No {label} file found in the configured source directory.")


def load_coarse_grid(cfg: GeometryConfig) -> tuple[np.ndarray, np.ndarray]:
    fpath = first_existing(sorted(cfg.imerg_dir.glob("IMERG_V07B_TP_*.nc")), "IMERG")
    ds = Dataset(fpath, "r")
    lat = np.array(ds.variables["lat"][:], dtype=np.float64)
    lon = np.array(ds.variables["lon"][:], dtype=np.float64)
    ds.close()

    lat_mask = (lat >= COMMON_LAT_MIN - 0.001) & (lat <= COMMON_LAT_MAX + 0.001)
    lon_mask = (lon >= COMMON_LON_MIN - 0.001) & (lon <= COMMON_LON_MAX + 0.001)
    return lat[lat_mask], lon[lon_mask]


def load_fine_grid(cfg: GeometryConfig) -> tuple[np.ndarray, np.ndarray]:
    fpath = first_existing(sorted(cfg.tpw_dir.glob("tp_fuse_tpw_*.nc")), "TPW")
    ds = Dataset(fpath, "r")
    lat = np.array(ds.variables["lat"][:], dtype=np.float64)
    lon = np.array(ds.variables["lon"][:], dtype=np.float64)
    ds.close()

    if lat[0] > lat[-1]:
        lat = lat[::-1]

    margin = 0.05
    lat_mask = (lat >= COMMON_LAT_MIN - margin) & (lat <= COMMON_LAT_MAX + margin)
    lon_mask = (lon >= COMMON_LON_MIN - margin) & (lon <= COMMON_LON_MAX + margin)
    return lat[lat_mask], lon[lon_mask]


def load_tpw_static_mask(
    cfg: GeometryConfig,
    fine_lat: np.ndarray,
    fine_lon: np.ndarray,
) -> np.ndarray:
    fpath = first_existing(sorted(cfg.tpw_dir.glob("tp_fuse_tpw_*.nc")), "TPW")
    ds = Dataset(fpath, "r")
    lat_raw = np.array(ds.variables["lat"][:], dtype=np.float64)
    lon_raw = np.array(ds.variables["lon"][:], dtype=np.float64)
    data = np.array(ds.variables["tpw"][:])
    ds.close()

    if lat_raw[0] > lat_raw[-1]:
        lat_raw = lat_raw[::-1]
        data = data[::-1, :]

    margin = 0.05
    lat_idx = np.where(
        (lat_raw >= COMMON_LAT_MIN - margin) & (lat_raw <= COMMON_LAT_MAX + margin)
    )[0]
    lon_idx = np.where(
        (lon_raw >= COMMON_LON_MIN - margin) & (lon_raw <= COMMON_LON_MAX + margin)
    )[0]
    data_sub = data[np.ix_(lat_idx, lon_idx)]

    if data_sub.shape != (len(fine_lat), len(fine_lon)):
        raise RuntimeError(
            f"TPW mask shape mismatch: source {data_sub.shape}, grid ({len(fine_lat)}, {len(fine_lon)})"
        )

    return ~np.isnan(data_sub)


def load_cmfd_land_mask(
    cfg: GeometryConfig,
    coarse_lat: np.ndarray,
    coarse_lon: np.ndarray,
) -> np.ndarray:
    fpath = resolve_cmfd_file(cfg.cmfd_dir, "prec", cfg.year, 5)
    if not fpath.exists():
        raise RuntimeError(f"CMFD reference file not found: {fpath}")

    ds = Dataset(fpath, "r")
    lat = np.array(ds.variables["lat"][:], dtype=np.float64)
    lon = np.array(ds.variables["lon"][:], dtype=np.float64)

    lat_idx = np.where(
        (lat >= COMMON_LAT_MIN - 0.001) & (lat <= COMMON_LAT_MAX + 0.001)
    )[0]
    lon_idx = np.where(
        (lon >= COMMON_LON_MIN - 0.001) & (lon <= COMMON_LON_MAX + 0.001)
    )[0]

    prec = np.array(ds.variables["prec"][0, :, :])
    ds.close()

    prec_sub = prec[np.ix_(lat_idx, lon_idx)]
    if prec_sub.shape != (len(coarse_lat), len(coarse_lon)):
        raise RuntimeError(
            f"CMFD mask shape mismatch: source {prec_sub.shape}, grid ({len(coarse_lat)}, {len(coarse_lon)})"
        )

    return np.abs(prec_sub) < 1e19


def _cell_edges(centers: np.ndarray) -> np.ndarray:
    n = len(centers)
    edges = np.zeros(n + 1, dtype=np.float64)
    if n == 1:
        edges[0] = centers[0] - 0.01
        edges[1] = centers[0] + 0.01
        return edges

    half_steps = np.diff(centers) / 2.0
    edges[0] = centers[0] - half_steps[0]
    edges[-1] = centers[-1] + half_steps[-1]
    edges[1:-1] = centers[:-1] + half_steps
    return edges


def compute_coarse_valid_fraction(
    fine_mask: np.ndarray,
    fine_lat: np.ndarray,
    fine_lon: np.ndarray,
    coarse_lat: np.ndarray,
    coarse_lon: np.ndarray,
) -> np.ndarray:
    n_clat = len(coarse_lat)
    n_clon = len(coarse_lon)
    coarse_step = 0.1
    half_c = coarse_step / 2.0

    fine_lat_edges = _cell_edges(fine_lat)
    fine_lon_edges = _cell_edges(fine_lon)

    valid_frac = np.zeros((n_clat, n_clon), dtype=np.float64)

    for j in range(n_clat):
        c_lat_lo = coarse_lat[j] - half_c
        c_lat_hi = coarse_lat[j] + half_c
        f_lat_idx = np.where(
            (fine_lat_edges[1:] > c_lat_lo) & (fine_lat_edges[:-1] < c_lat_hi)
        )[0]
        if len(f_lat_idx) == 0:
            continue

        lat_overlaps = (
            np.minimum(fine_lat_edges[f_lat_idx + 1], c_lat_hi)
            - np.maximum(fine_lat_edges[f_lat_idx], c_lat_lo)
        )

        for i in range(n_clon):
            c_lon_lo = coarse_lon[i] - half_c
            c_lon_hi = coarse_lon[i] + half_c
            f_lon_idx = np.where(
                (fine_lon_edges[1:] > c_lon_lo) & (fine_lon_edges[:-1] < c_lon_hi)
            )[0]
            if len(f_lon_idx) == 0:
                continue

            lon_overlaps = (
                np.minimum(fine_lon_edges[f_lon_idx + 1], c_lon_hi)
                - np.maximum(fine_lon_edges[f_lon_idx], c_lon_lo)
            )
            weights = lat_overlaps[:, None] * lon_overlaps[None, :]
            mask_sub = fine_mask[np.ix_(f_lat_idx, f_lon_idx)]

            total_weight = np.sum(weights)
            if total_weight > 0:
                valid_frac[j, i] = np.sum(weights * mask_sub) / total_weight

    return valid_frac


def save_output(
    out_path: Path,
    coarse_lat: np.ndarray,
    coarse_lon: np.ndarray,
    fine_lat: np.ndarray,
    fine_lon: np.ndarray,
    fine_mask: np.ndarray,
    coarse_mask: np.ndarray,
    coarse_valid_frac: np.ndarray,
    threshold: float,
    year: int,
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with Dataset(out_path, "w", format="NETCDF4") as ds:
        ds.createDimension("coarse_lat", len(coarse_lat))
        ds.createDimension("coarse_lon", len(coarse_lon))
        ds.createDimension("fine_lat", len(fine_lat))
        ds.createDimension("fine_lon", len(fine_lon))

        v = ds.createVariable("coarse_lat", "f8", ("coarse_lat",))
        v.units = "degrees_north"
        v.long_name = "Coarse grid latitude (0.1 deg)"
        v[:] = coarse_lat

        v = ds.createVariable("coarse_lon", "f8", ("coarse_lon",))
        v.units = "degrees_east"
        v.long_name = "Coarse grid longitude (0.1 deg)"
        v[:] = coarse_lon

        v = ds.createVariable("fine_lat", "f8", ("fine_lat",))
        v.units = "degrees_north"
        v.long_name = "Fine grid latitude (~0.02 deg, ascending)"
        v[:] = fine_lat

        v = ds.createVariable("fine_lon", "f8", ("fine_lon",))
        v.units = "degrees_east"
        v.long_name = "Fine grid longitude (~0.02002 deg)"
        v[:] = fine_lon

        v = ds.createVariable("fine_mask", "i1", ("fine_lat", "fine_lon"), zlib=True, complevel=4)
        v.long_name = "TPW static valid-pixel mask (1=valid, 0=NaN)"
        v.comment = "Derived from the TPW NaN mask; treated as time-invariant."
        v[:] = fine_mask.astype(np.int8)

        v = ds.createVariable("coarse_mask", "i1", ("coarse_lat", "coarse_lon"), zlib=True, complevel=4)
        v.long_name = "Coarse valid-cell mask (1=valid, 0=masked)"
        v.comment = (
            f"Intersection of TPW valid fraction >= {threshold} and CMFD non-fill support."
        )
        v[:] = coarse_mask.astype(np.int8)

        v = ds.createVariable("coarse_valid_frac", "f4", ("coarse_lat", "coarse_lon"), zlib=True, complevel=4)
        v.long_name = "Fraction of valid fine pixels per coarse cell"
        v.comment = "Area-weighted overlap fraction."
        v[:] = coarse_valid_frac.astype(np.float32)

        ds.title = "Common domain and masks for TP water budget closure"
        ds.reference_year = year
        ds.common_domain = (
            f"lon [{COMMON_LON_MIN}, {COMMON_LON_MAX}], "
            f"lat [{COMMON_LAT_MIN}, {COMMON_LAT_MAX}]"
        )
        ds.coarse_grid = f"{len(coarse_lat)} lat x {len(coarse_lon)} lon at 0.1 deg"
        ds.fine_grid = f"{len(fine_lat)} lat x {len(fine_lon)} lon at ~0.02 deg"
        ds.coarse_valid_threshold = threshold
        ds.geometry_policy = "conservative_rebuild"
        ds.source_script = "build_common_domain_and_masks.py"


def main() -> int:
    cfg = parse_args()

    print("=" * 60)
    print("  Build Common Domain and Masks")
    print("=" * 60)
    print(f"  Year:       {cfg.year}")
    print(f"  IMERG dir:  {cfg.imerg_dir}")
    print(f"  TPW dir:    {cfg.tpw_dir}")
    print(f"  CMFD dir:   {cfg.cmfd_dir}")
    print(f"  Output dir: {cfg.output_dir}")

    print("\n[1/5] Loading coarse grid (IMERG)...")
    coarse_lat, coarse_lon = load_coarse_grid(cfg)
    print(f"  Coarse: {len(coarse_lat)} lat x {len(coarse_lon)} lon")
    print(f"  Range: lat [{coarse_lat[0]:.2f}, {coarse_lat[-1]:.2f}], lon [{coarse_lon[0]:.2f}, {coarse_lon[-1]:.2f}]")

    print("\n[2/5] Loading fine grid (TPW)...")
    fine_lat, fine_lon = load_fine_grid(cfg)
    print(f"  Fine: {len(fine_lat)} lat x {len(fine_lon)} lon")
    print(f"  Range: lat [{fine_lat[0]:.4f}, {fine_lat[-1]:.4f}], lon [{fine_lon[0]:.4f}, {fine_lon[-1]:.4f}]")

    print("\n[3/5] Loading TPW static mask...")
    fine_mask = load_tpw_static_mask(cfg, fine_lat, fine_lon)
    valid_pct = 100.0 * np.sum(fine_mask) / fine_mask.size
    print(f"  Fine mask: {valid_pct:.1f}% valid ({np.sum(fine_mask)}/{fine_mask.size} pixels)")

    print("\n[4/5] Computing coarse valid fractions (overlap-based)...")
    coarse_valid_frac = compute_coarse_valid_fraction(
        fine_mask,
        fine_lat,
        fine_lon,
        coarse_lat,
        coarse_lon,
    )
    cmfd_mask = load_cmfd_land_mask(cfg, coarse_lat, coarse_lon)
    cmfd_land_pct = 100.0 * np.sum(cmfd_mask) / cmfd_mask.size
    print(f"  CMFD land mask: {cmfd_land_pct:.1f}% land")

    coarse_mask = (coarse_valid_frac >= cfg.threshold) & cmfd_mask
    coarse_valid_pct = 100.0 * np.sum(coarse_mask) / coarse_mask.size
    print(
        f"  Coarse mask (threshold={cfg.threshold}): {coarse_valid_pct:.1f}% valid "
        f"({np.sum(coarse_mask)}/{coarse_mask.size} cells)"
    )

    print("\n  Coarse valid fraction stats (over valid cells):")
    valid_fracs = coarse_valid_frac[coarse_mask]
    if len(valid_fracs) > 0:
        print(
            f"    min={np.min(valid_fracs):.3f}, "
            f"max={np.max(valid_fracs):.3f}, "
            f"mean={np.mean(valid_fracs):.3f}"
        )

    print("\n[5/5] Saving...")
    out_path = cfg.output_dir / "common_domain.nc"
    save_output(
        out_path,
        coarse_lat,
        coarse_lon,
        fine_lat,
        fine_lon,
        fine_mask,
        coarse_mask,
        coarse_valid_frac,
        cfg.threshold,
        cfg.year,
    )
    print(f"  Saved to: {out_path}")
    print(f"  File size: {out_path.stat().st_size / 1024:.1f} KB")

    print("\n" + "=" * 60)
    print("  DONE")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())
