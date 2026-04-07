#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Produce aligned 3-hour canonical tensors for IMERG, TPW, and CMFD inputs.

This script preserves the 2020 warm-season preprocessing chain as the default,
but it can also preprocess a second warm-season year without
silently overwriting the 2020 outputs.

Outputs (written to the selected output directory):
    imerg_3h.nc
        precipitation_3h          (time, coarse_lat, coarse_lon)  [mm]
        precipitationQualityIndex (time, coarse_lat, coarse_lon)  [0-1]
        randomError_3h            (time, coarse_lat, coarse_lon)  [mm]

    tpw_3h.nc
        tpw_t0                    (time, fine_lat, fine_lon)      [kg m-2]
        tpw_t1                    (time, fine_lat, fine_lon)      [kg m-2]
        dW_3h                     (time, fine_lat, fine_lon)      [kg m-2]

    cmfd_3h.nc
        prec, pres, shum, temp, wind
            (time, coarse_lat, coarse_lon)

Examples:
    python preprocess_canonical_3h.py
    python preprocess_canonical_3h.py --year 2021 --dry-run
    python preprocess_canonical_3h.py --year 2021 ^
        --domain-file "/path/to/preprocessed/common_domain.nc" ^
        --output-dir "/path/to/preprocessed_2021"
"""

from __future__ import annotations

import os
os.environ.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")
os.environ.setdefault("HDF5_DISABLE_VERSION_CHECK", "1")

import argparse
import json
import sys
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path

import numpy as np
from netCDF4 import Dataset, date2num, num2date
from tqdm import tqdm


CODE_DIR = Path(__file__).resolve().parent
REPO_ROOT = CODE_DIR.parent

COMMON_LON_MIN = 70.05
COMMON_LON_MAX = 104.95
COMMON_LAT_MIN = 25.05
COMMON_LAT_MAX = 39.95

LEGACY_2020_START = datetime(2020, 5, 1, 0, 0)
LEGACY_2020_END = datetime(2020, 9, 30, 21, 0)  # last 3-h window starts at 21:00
DT_3H = timedelta(hours=3)

CMFD_VARS = ["prec", "pres", "shum", "temp", "wind"]


@dataclass(frozen=True)
class PreprocessConfig:
    start: datetime
    end: datetime
    imerg_dir: Path
    tpw_dir: Path
    cmfd_dir: Path
    output_dir: Path
    domain_file: Path
    dry_run: bool

    @property
    def time_units(self) -> str:
        return f"hours since {self.start.year}-01-01 00:00:00"


def parse_date_arg(value: str) -> date:
    return datetime.strptime(value, "%Y-%m-%d").date()


def default_imerg_dir(start_date: date, end_date: date) -> Path:
    if (
        start_date == LEGACY_2020_START.date()
        and end_date == LEGACY_2020_END.date()
    ):
        return REPO_ROOT / "IMERG_V07B_TP_daily"
    if (
        start_date.year == end_date.year
        and start_date.month == 5 and start_date.day == 1
        and end_date.month == 9 and end_date.day == 30
    ):
        return REPO_ROOT / f"IMERG_V07B_TP_{start_date.year}_May_Sep_daily"
    return REPO_ROOT / f"IMERG_V07B_TP_{start_date:%Y%m%d}_{end_date:%Y%m%d}_daily"


def default_tpw_dir(start_date: date, end_date: date) -> Path:
    if (
        start_date == LEGACY_2020_START.date()
        and end_date == LEGACY_2020_END.date()
    ):
        legacy = REPO_ROOT / "Sun_TPW_2020_May_Sep"
        if legacy.exists():
            return legacy
    year_dir = REPO_ROOT / "Sun_TPW" / f"{start_date.year}"
    if year_dir.exists():
        return year_dir
    return REPO_ROOT / "Sun_TPW"


def default_cmfd_dir(start_date: date, end_date: date) -> Path:
    if (
        start_date == LEGACY_2020_START.date()
        and end_date == LEGACY_2020_END.date()
    ):
        legacy = REPO_ROOT / "CMFD_2020_May_Sep_All"
        if legacy.exists():
            return legacy
    return REPO_ROOT / "CMFD_v2.0"


def default_output_dir(start_date: date, end_date: date) -> Path:
    if (
        start_date == LEGACY_2020_START.date()
        and end_date == LEGACY_2020_END.date()
    ):
        return REPO_ROOT / "preprocessed"
    if (
        start_date.year == end_date.year
        and start_date.month == 5 and start_date.day == 1
        and end_date.month == 9 and end_date.day == 30
    ):
        return REPO_ROOT / f"preprocessed_{start_date.year}"
    return REPO_ROOT / f"preprocessed_{start_date:%Y%m%d}_{end_date:%Y%m%d}"


def parse_args() -> PreprocessConfig:
    parser = argparse.ArgumentParser(
        description="Preprocess IMERG, TPW, and CMFD to the canonical 3-hour grid."
    )
    parser.add_argument(
        "--year",
        type=int,
        default=None,
        help="Convenience option: preprocess YYYY-05-01 through YYYY-09-30.",
    )
    parser.add_argument("--start-date", type=parse_date_arg, default=LEGACY_2020_START.date())
    parser.add_argument("--end-date", type=parse_date_arg, default=LEGACY_2020_END.date())
    parser.add_argument("--imerg-dir", type=Path, default=None)
    parser.add_argument("--tpw-dir", type=Path, default=None)
    parser.add_argument("--cmfd-dir", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--domain-file", type=Path, default=None)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    explicit_date_args = any(
        arg == "--start-date"
        or arg.startswith("--start-date=")
        or arg == "--end-date"
        or arg.startswith("--end-date=")
        for arg in sys.argv[1:]
    )
    if args.year is not None:
        if explicit_date_args:
            raise RuntimeError("--year cannot be combined with --start-date/--end-date")
        start_date = date(args.year, 5, 1)
        end_date = date(args.year, 9, 30)
    else:
        start_date = args.start_date
        end_date = args.end_date

    if end_date < start_date:
        raise RuntimeError("end-date must be greater than or equal to start-date")
    if start_date.year != end_date.year:
        raise RuntimeError("Only single-year preprocessing windows are supported.")

    start = datetime(start_date.year, start_date.month, start_date.day, 0, 0)
    end = datetime(end_date.year, end_date.month, end_date.day, 21, 0)

    output_dir = args.output_dir or default_output_dir(start_date, end_date)
    domain_file = args.domain_file or (output_dir / "common_domain.nc")

    return PreprocessConfig(
        start=start,
        end=end,
        imerg_dir=args.imerg_dir or default_imerg_dir(start_date, end_date),
        tpw_dir=args.tpw_dir or default_tpw_dir(start_date, end_date),
        cmfd_dir=args.cmfd_dir or default_cmfd_dir(start_date, end_date),
        output_dir=output_dir,
        domain_file=domain_file,
        dry_run=args.dry_run,
    )


def build_time_axis(cfg: PreprocessConfig) -> list[datetime]:
    times: list[datetime] = []
    t = cfg.start
    while t <= cfg.end:
        times.append(t)
        t += DT_3H
    return times


def month_tokens(cfg: PreprocessConfig) -> list[str]:
    return [f"{cfg.start.year}{month:02d}" for month in range(cfg.start.month, cfg.end.month + 1)]


def resolve_cmfd_file(cmfd_dir: Path, var: str, month_token: str) -> Path:
    fname = f"{var}_CMFD_V0200_B-01_03hr_010deg_{month_token}.nc"
    flat = cmfd_dir / fname
    if flat.exists():
        return flat
    return cmfd_dir / var / fname


def first_existing(path_iterable: list[Path], label: str) -> Path:
    for path in path_iterable:
        if path.exists():
            return path
    raise RuntimeError(f"No {label} file found in the configured source directory.")


def process_imerg(
    cfg: PreprocessConfig,
    canon_times: list[datetime],
    coarse_lat: np.ndarray,
    coarse_lon: np.ndarray,
) -> Path | None:
    print("\n" + "=" * 60)
    print("  IMERG: 30-min to 3-h accumulation")
    print("=" * 60)

    n_t = len(canon_times)
    n_lat = len(coarse_lat)
    n_lon = len(coarse_lon)

    sample = first_existing(
        sorted(cfg.imerg_dir.glob("IMERG_V07B_TP_*.nc")),
        "IMERG",
    )
    ds = Dataset(sample, "r")
    full_lat = np.array(ds.variables["lat"][:])
    full_lon = np.array(ds.variables["lon"][:])
    ds.close()

    lat_idx = np.where(
        (full_lat >= COMMON_LAT_MIN - 0.001)
        & (full_lat <= COMMON_LAT_MAX + 0.001)
    )[0]
    lon_idx = np.where(
        (full_lon >= COMMON_LON_MIN - 0.001)
        & (full_lon <= COMMON_LON_MAX + 0.001)
    )[0]
    lat_sl = slice(lat_idx[0], lat_idx[-1] + 1)
    lon_sl = slice(lon_idx[0], lon_idx[-1] + 1)

    print(f"  Source directory: {cfg.imerg_dir}")
    print(f"  Output shape: ({n_t}, {n_lat}, {n_lon})")
    print(f"  Slicing: lat[{lat_sl.start}:{lat_sl.stop}], lon[{lon_sl.start}:{lon_sl.stop}]")

    if cfg.dry_run:
        print("  [DRY RUN] Skipping actual processing.")
        return None

    precip_3h = np.full((n_t, n_lat, n_lon), np.nan, dtype=np.float32)
    qi_3h = np.full((n_t, n_lat, n_lon), np.nan, dtype=np.float32)
    error_3h = np.full((n_t, n_lat, n_lon), np.nan, dtype=np.float32)

    daily_cache: dict[date, Dataset | None] = {}

    def get_daily(d: date) -> Dataset | None:
        if d not in daily_cache:
            fpath = cfg.imerg_dir / f"IMERG_V07B_TP_{d:%Y%m%d}.nc"
            daily_cache[d] = Dataset(fpath, "r") if fpath.exists() else None
        return daily_cache[d]

    for ti, t0 in enumerate(tqdm(canon_times, desc="IMERG 3h")):
        accum_p = np.zeros((n_lat, n_lon), dtype=np.float64)
        accum_e = np.zeros((n_lat, n_lon), dtype=np.float64)
        accum_q = np.zeros((n_lat, n_lon), dtype=np.float64)
        n_valid = 0

        for slot_offset in range(6):
            ts = t0 + timedelta(minutes=30 * slot_offset)
            slot_idx = ts.hour * 2 + (1 if ts.minute >= 30 else 0)
            ds_day = get_daily(ts.date())
            if ds_day is None:
                continue

            p = np.array(ds_day.variables["precipitation"][slot_idx, lat_sl, lon_sl], dtype=np.float64)
            q = np.array(
                ds_day.variables["precipitationQualityIndex"][slot_idx, lat_sl, lon_sl],
                dtype=np.float64,
            )
            e = np.array(ds_day.variables["randomError"][slot_idx, lat_sl, lon_sl], dtype=np.float64)

            accum_p += p * 0.5
            accum_e += e * 0.5
            accum_q += q
            n_valid += 1

        if n_valid == 6:
            precip_3h[ti] = accum_p.astype(np.float32)
            error_3h[ti] = accum_e.astype(np.float32)
            qi_3h[ti] = (accum_q / 6.0).astype(np.float32)

    for ds_day in daily_cache.values():
        if ds_day is not None:
            ds_day.close()

    out_path = cfg.output_dir / "imerg_3h.nc"
    with Dataset(out_path, "w", format="NETCDF4") as ds:
        ds.createDimension("time", n_t)
        ds.createDimension("coarse_lat", n_lat)
        ds.createDimension("coarse_lon", n_lon)

        tv = ds.createVariable("time", "f8", ("time",))
        tv.units = cfg.time_units
        tv.calendar = "standard"
        tv.long_name = "3-h window start time"
        tv[:] = date2num(canon_times, units=tv.units, calendar=tv.calendar)

        latv = ds.createVariable("coarse_lat", "f8", ("coarse_lat",))
        latv.units = "degrees_north"
        latv[:] = coarse_lat

        lonv = ds.createVariable("coarse_lon", "f8", ("coarse_lon",))
        lonv.units = "degrees_east"
        lonv[:] = coarse_lon

        v = ds.createVariable(
            "precipitation_3h",
            "f4",
            ("time", "coarse_lat", "coarse_lon"),
            zlib=True,
            complevel=4,
        )
        v.units = "mm"
        v.long_name = "3-h precipitation accumulation"
        v.comment = "Sum of 6 half-hour slots converted from mm hr-1 to mm."
        v[:] = precip_3h

        v = ds.createVariable(
            "precipitationQualityIndex",
            "f4",
            ("time", "coarse_lat", "coarse_lon"),
            zlib=True,
            complevel=4,
        )
        v.units = "1"
        v.long_name = "Mean quality index over the 3-h window"
        v[:] = qi_3h

        v = ds.createVariable(
            "randomError_3h",
            "f4",
            ("time", "coarse_lat", "coarse_lon"),
            zlib=True,
            complevel=4,
        )
        v.units = "mm"
        v.long_name = "3-h random-error accumulation"
        v.comment = "Sum of 6 half-hour error slots converted from mm hr-1 to mm."
        v[:] = error_3h

        ds.title = "IMERG V07B 3-h aggregation on the common coarse domain"
        ds.source_script = "preprocess_canonical_3h.py"

    valid_pct = 100.0 * float(np.sum(~np.isnan(precip_3h))) / precip_3h.size
    print(f"  Output: {out_path}")
    print(f"  Valid: {valid_pct:.1f}%")
    print(f"  Precip 3h range: [{np.nanmin(precip_3h):.3f}, {np.nanmax(precip_3h):.3f}] mm")
    return out_path


def process_tpw(
    cfg: PreprocessConfig,
    canon_times: list[datetime],
    fine_lat: np.ndarray,
    fine_lon: np.ndarray,
) -> Path | None:
    print("\n" + "=" * 60)
    print("  TPW: hourly to 3-h increments")
    print("=" * 60)

    n_t = len(canon_times)
    n_lat = len(fine_lat)
    n_lon = len(fine_lon)
    print(f"  Source directory: {cfg.tpw_dir}")
    print(f"  Output shape: ({n_t}, {n_lat}, {n_lon})")

    if cfg.dry_run:
        print("  [DRY RUN] Skipping actual processing.")
        return None

    sample = first_existing(
        sorted(cfg.tpw_dir.glob("tp_fuse_tpw_*.nc")),
        "TPW",
    )
    with Dataset(sample, "r") as ds:
        raw_lat = np.array(ds.variables["lat"][:], dtype=np.float64)
        raw_lon = np.array(ds.variables["lon"][:], dtype=np.float64)

    lat_flipped = raw_lat[0] > raw_lat[-1]
    raw_lat_asc = raw_lat[::-1] if lat_flipped else raw_lat

    margin = 0.05
    lat_idx = np.where(
        (raw_lat_asc >= COMMON_LAT_MIN - margin)
        & (raw_lat_asc <= COMMON_LAT_MAX + margin)
    )[0]
    lon_idx = np.where(
        (raw_lon >= COMMON_LON_MIN - margin)
        & (raw_lon <= COMMON_LON_MAX + margin)
    )[0]

    if len(lat_idx) != n_lat or len(lon_idx) != n_lon:
        raise RuntimeError(
            f"TPW slice mismatch: source gives ({len(lat_idx)}, {len(lon_idx)}), "
            f"domain expects ({n_lat}, {n_lon})"
        )

    if lat_flipped:
        n_raw = len(raw_lat)
        orig_lat_idx = n_raw - 1 - lat_idx[::-1]
    else:
        orig_lat_idx = lat_idx

    lat_sl = slice(orig_lat_idx[0], orig_lat_idx[-1] + 1)
    lon_sl = slice(lon_idx[0], lon_idx[-1] + 1)

    def load_tpw_snapshot(ts: datetime) -> np.ndarray | None:
        fpath = cfg.tpw_dir / f"tp_fuse_tpw_{ts:%Y%m%d%H}.nc"
        if not fpath.exists():
            return None
        try:
            with Dataset(fpath, "r") as ds_local:
                data = np.array(
                    ds_local.variables["tpw"][lat_sl, lon_sl],
                    dtype=np.float32,
                )
        except Exception as exc:
            raise RuntimeError(f"Failed to read TPW snapshot: {fpath}") from exc
        if lat_flipped:
            data = data[::-1, :]
        return data

    tpw_t0 = np.full((n_t, n_lat, n_lon), np.nan, dtype=np.float32)
    tpw_t1 = np.full((n_t, n_lat, n_lon), np.nan, dtype=np.float32)
    dW_3h = np.full((n_t, n_lat, n_lon), np.nan, dtype=np.float32)
    missing_snapshots: list[str] = []
    unreadable_snapshots: list[str] = []

    for ti, t0 in enumerate(tqdm(canon_times, desc="TPW 3h")):
        t1 = t0 + DT_3H
        try:
            snap0 = load_tpw_snapshot(t0)
        except RuntimeError as exc:
            unreadable_snapshots.append(t0.strftime("%Y-%m-%d %H:%M"))
            print(f"  WARN: {exc}")
            snap0 = None
        else:
            if snap0 is None:
                missing_snapshots.append(t0.strftime("%Y-%m-%d %H:%M"))

        try:
            snap1 = load_tpw_snapshot(t1)
        except RuntimeError as exc:
            unreadable_snapshots.append(t1.strftime("%Y-%m-%d %H:%M"))
            print(f"  WARN: {exc}")
            snap1 = None
        else:
            if snap1 is None:
                missing_snapshots.append(t1.strftime("%Y-%m-%d %H:%M"))

        if snap0 is not None:
            tpw_t0[ti] = snap0
        if snap1 is not None:
            tpw_t1[ti] = snap1
        if snap0 is not None and snap1 is not None:
            dW_3h[ti] = snap1 - snap0

    out_path = cfg.output_dir / "tpw_3h.nc"
    with Dataset(out_path, "w", format="NETCDF4") as ds:
        ds.createDimension("time", n_t)
        ds.createDimension("fine_lat", n_lat)
        ds.createDimension("fine_lon", n_lon)

        tv = ds.createVariable("time", "f8", ("time",))
        tv.units = cfg.time_units
        tv.calendar = "standard"
        tv.long_name = "3-h window start time"
        tv[:] = date2num(canon_times, units=tv.units, calendar=tv.calendar)

        latv = ds.createVariable("fine_lat", "f8", ("fine_lat",))
        latv.units = "degrees_north"
        latv[:] = fine_lat

        lonv = ds.createVariable("fine_lon", "f8", ("fine_lon",))
        lonv.units = "degrees_east"
        lonv[:] = fine_lon

        v = ds.createVariable(
            "tpw_t0",
            "f4",
            ("time", "fine_lat", "fine_lon"),
            zlib=True,
            complevel=4,
            chunksizes=(1, n_lat, n_lon),
        )
        v.units = "kg m-2"
        v.long_name = "TPW snapshot at the 3-h window start"
        v[:] = tpw_t0

        v = ds.createVariable(
            "tpw_t1",
            "f4",
            ("time", "fine_lat", "fine_lon"),
            zlib=True,
            complevel=4,
            chunksizes=(1, n_lat, n_lon),
        )
        v.units = "kg m-2"
        v.long_name = "TPW snapshot at the 3-h window end"
        v[:] = tpw_t1

        v = ds.createVariable(
            "dW_3h",
            "f4",
            ("time", "fine_lat", "fine_lon"),
            zlib=True,
            complevel=4,
            chunksizes=(1, n_lat, n_lon),
        )
        v.units = "kg m-2"
        v.long_name = "TPW 3-h increment (tpw_t1 - tpw_t0)"
        v[:] = dW_3h

        ds.title = "TPW 3-h snapshots and increments on the common fine domain"
        ds.source_script = "preprocess_canonical_3h.py"
        ds.tpw_units_note = "Assumed kg m-2 because the source files do not provide a units attribute."
        ds.tpw_missing_snapshot_count = len(sorted(set(missing_snapshots)))
        ds.tpw_unreadable_snapshot_count = len(sorted(set(unreadable_snapshots)))

    valid_pct = 100.0 * float(np.sum(~np.isnan(dW_3h))) / dW_3h.size
    print(f"  Output: {out_path}")
    print(f"  dW_3h valid: {valid_pct:.1f}%")
    print(f"  dW_3h range: [{np.nanmin(dW_3h):.3f}, {np.nanmax(dW_3h):.3f}] kg m-2")

    audit_payload = {
        "source_dir": str(cfg.tpw_dir),
        "window_start": canon_times[0].strftime("%Y-%m-%d %H:%M:%S") if canon_times else None,
        "window_end": canon_times[-1].strftime("%Y-%m-%d %H:%M:%S") if canon_times else None,
        "missing_snapshot_count": len(sorted(set(missing_snapshots))),
        "unreadable_snapshot_count": len(sorted(set(unreadable_snapshots))),
        "missing_snapshots": sorted(set(missing_snapshots)),
        "unreadable_snapshots": sorted(set(unreadable_snapshots)),
    }
    audit_path = cfg.output_dir / "tpw_snapshot_audit.json"
    audit_path.write_text(json.dumps(audit_payload, indent=2), encoding="utf-8")
    print(f"  Snapshot audit: {audit_path}")
    return out_path


def process_cmfd(
    cfg: PreprocessConfig,
    canon_times: list[datetime],
    coarse_lat: np.ndarray,
    coarse_lon: np.ndarray,
) -> Path | None:
    print("\n" + "=" * 60)
    print("  CMFD: clip to common domain")
    print("=" * 60)

    n_t = len(canon_times)
    n_lat = len(coarse_lat)
    n_lon = len(coarse_lon)
    print(f"  Source directory: {cfg.cmfd_dir}")
    print(f"  Output shape: ({n_t}, {n_lat}, {n_lon})")

    if cfg.dry_run:
        print("  [DRY RUN] Skipping actual processing.")
        return None

    month_list = month_tokens(cfg)
    sample = resolve_cmfd_file(cfg.cmfd_dir, "prec", month_list[0])
    if not sample.exists():
        raise RuntimeError(f"CMFD sample file not found: {sample}")

    ds = Dataset(sample, "r")
    cmfd_lat = np.array(ds.variables["lat"][:])
    cmfd_lon = np.array(ds.variables["lon"][:])
    ds.close()

    lat_idx = np.where(
        (cmfd_lat >= COMMON_LAT_MIN - 0.001)
        & (cmfd_lat <= COMMON_LAT_MAX + 0.001)
    )[0]
    lon_idx = np.where(
        (cmfd_lon >= COMMON_LON_MIN - 0.001)
        & (cmfd_lon <= COMMON_LON_MAX + 0.001)
    )[0]
    lat_sl = slice(lat_idx[0], lat_idx[-1] + 1)
    lon_sl = slice(lon_idx[0], lon_idx[-1] + 1)

    if lat_idx[-1] - lat_idx[0] + 1 != n_lat or lon_idx[-1] - lon_idx[0] + 1 != n_lon:
        raise RuntimeError(
            f"CMFD slice mismatch: source gives ({lat_idx[-1] - lat_idx[0] + 1}, "
            f"{lon_idx[-1] - lon_idx[0] + 1}), domain expects ({n_lat}, {n_lon})"
        )

    cmfd_time_map: dict[str, tuple[list[datetime], Path]] = {}
    for month_token in month_list:
        fpath = resolve_cmfd_file(cfg.cmfd_dir, "prec", month_token)
        if not fpath.exists():
            raise RuntimeError(f"Required CMFD file not found: {fpath}")
        ds_month = Dataset(fpath, "r")
        tvar = ds_month.variables["time"]
        times = num2date(tvar[:], units=tvar.units, calendar=tvar.calendar)
        times = [
            datetime(t.year, t.month, t.day, t.hour, t.minute, t.second)
            for t in times
        ]
        cmfd_time_map[month_token] = (times, fpath)
        ds_month.close()

    time_lookup: list[tuple[str, int] | None] = []
    for ct in canon_times:
        found = False
        month_key = f"{ct.year}{ct.month:02d}"
        if month_key in cmfd_time_map:
            times_list, _ = cmfd_time_map[month_key]
            for ti, ts in enumerate(times_list):
                if ts == ct:
                    time_lookup.append((month_key, ti))
                    found = True
                    break
        if not found:
            time_lookup.append(None)

    n_found = sum(1 for item in time_lookup if item is not None)
    print(f"  Time alignment: {n_found}/{n_t} canonical times matched")

    data = {
        var: np.full((n_t, n_lat, n_lon), np.nan, dtype=np.float32)
        for var in CMFD_VARS
    }

    for var in tqdm(CMFD_VARS, desc="CMFD vars"):
        open_files: dict[str, Dataset] = {}
        for ti, entry in enumerate(time_lookup):
            if entry is None:
                continue
            month_key, t_idx = entry
            if month_key not in open_files:
                fpath = resolve_cmfd_file(cfg.cmfd_dir, var, month_key)
                if not fpath.exists():
                    raise RuntimeError(f"Required CMFD file not found: {fpath}")
                open_files[month_key] = Dataset(fpath, "r")
            ds_month = open_files[month_key]
            arr = np.array(ds_month.variables[var][t_idx, lat_sl, lon_sl], dtype=np.float32)
            arr[np.abs(arr) > 1e19] = np.nan
            data[var][ti] = arr
        for ds_month in open_files.values():
            ds_month.close()

    out_path = cfg.output_dir / "cmfd_3h.nc"
    with Dataset(out_path, "w", format="NETCDF4") as ds:
        ds.createDimension("time", n_t)
        ds.createDimension("coarse_lat", n_lat)
        ds.createDimension("coarse_lon", n_lon)

        tv = ds.createVariable("time", "f8", ("time",))
        tv.units = cfg.time_units
        tv.calendar = "standard"
        tv.long_name = "3-h window start time"
        tv[:] = date2num(canon_times, units=tv.units, calendar=tv.calendar)

        latv = ds.createVariable("coarse_lat", "f8", ("coarse_lat",))
        latv.units = "degrees_north"
        latv[:] = coarse_lat

        lonv = ds.createVariable("coarse_lon", "f8", ("coarse_lon",))
        lonv.units = "degrees_east"
        lonv[:] = coarse_lon

        units_map = {
            "prec": ("kg m-2 s-1", "Precipitation rate"),
            "pres": ("Pa", "Surface air pressure"),
            "shum": ("kg kg-1", "Near-surface specific humidity"),
            "temp": ("K", "Near-surface air temperature"),
            "wind": ("m s-1", "Near-surface wind speed (scalar)"),
        }

        for var in CMFD_VARS:
            units, long_name = units_map[var]
            v = ds.createVariable(
                var,
                "f4",
                ("time", "coarse_lat", "coarse_lon"),
                zlib=True,
                complevel=4,
            )
            v.units = units
            v.long_name = long_name
            v[:] = data[var]

        ds.title = "CMFD v2.0 on the common coarse domain at native 3-h cadence"
        ds.source_script = "preprocess_canonical_3h.py"
        ds.wind_note = "Scalar wind speed only; no u/v components are stored."

    for var in CMFD_VARS:
        arr = data[var]
        valid_pct = 100.0 * float(np.sum(~np.isnan(arr))) / arr.size
        vmin = np.nanmin(arr) if np.any(~np.isnan(arr)) else float("nan")
        vmax = np.nanmax(arr) if np.any(~np.isnan(arr)) else float("nan")
        print(f"  {var}: valid={valid_pct:.1f}%, range=[{vmin:.4g}, {vmax:.4g}]")

    print(f"  Output: {out_path}")
    return out_path


def main() -> int:
    cfg = parse_args()

    print("=" * 60)
    print("  Preprocess Canonical 3-hour Tensors")
    print("=" * 60)
    print(f"  Time window: {cfg.start} to {cfg.end}")
    print(f"  IMERG dir:   {cfg.imerg_dir}")
    print(f"  TPW dir:     {cfg.tpw_dir}")
    print(f"  CMFD dir:    {cfg.cmfd_dir}")
    print(f"  Output dir:  {cfg.output_dir}")
    print(f"  Domain file: {cfg.domain_file}")

    if not cfg.domain_file.exists():
        print(f"ERROR: {cfg.domain_file} not found.")
        print("Run build_common_domain_and_masks.py first, or pass --domain-file explicitly.")
        return 1

    ds = Dataset(cfg.domain_file, "r")
    coarse_lat = np.array(ds.variables["coarse_lat"][:])
    coarse_lon = np.array(ds.variables["coarse_lon"][:])
    fine_lat = np.array(ds.variables["fine_lat"][:])
    fine_lon = np.array(ds.variables["fine_lon"][:])
    ds.close()

    canon_times = build_time_axis(cfg)
    print(f"\n  Canonical time axis: {len(canon_times)} steps")
    print(f"  From {canon_times[0]} to {canon_times[-1]}")
    print(f"  Coarse grid: {len(coarse_lat)} x {len(coarse_lon)}")
    print(f"  Fine grid:   {len(fine_lat)} x {len(fine_lon)}")

    cfg.output_dir.mkdir(parents=True, exist_ok=True)

    process_imerg(cfg, canon_times, coarse_lat, coarse_lon)
    process_tpw(cfg, canon_times, fine_lat, fine_lon)
    process_cmfd(cfg, canon_times, coarse_lat, coarse_lon)

    print("\n" + "=" * 60)
    print("  SUMMARY")
    print("=" * 60)
    for name in ["imerg_3h.nc", "tpw_3h.nc", "cmfd_3h.nc"]:
        path = cfg.output_dir / name
        if path.exists():
            print(f"  {name}: {path.stat().st_size / 1024 / 1024:.1f} MB")
        else:
            print(f"  {name}: not created (dry run?)")

    print("\n  DONE")
    return 0


if __name__ == "__main__":
    sys.exit(main())
