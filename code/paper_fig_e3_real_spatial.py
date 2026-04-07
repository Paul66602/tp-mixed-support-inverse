#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Publication-quality geographic figure for the E3 real-data diagnostic.

Produces a 3-panel row:
  (a) IMERG coarse lift on the fine grid
  (b) Learned fine-grid precipitation (diagnostic)
  (c) Departure from lifted IMERG

All panels include a Tibetan Plateau boundary overlay when a shapefile is
available, plus latitude/longitude gridlines and styled colorbars.

When multiple cases are rendered without ``--shared-scale``, each case uses
its own precipitation and difference limits. This matches the released
supplementary real-data diagnostic workflow.

Usage:
  python paper_fig_e3_real_spatial.py --case-dir paper_figs/case_data_real
  python paper_fig_e3_real_spatial.py --case-file paper_figs/case_data_real/case_real_t0800.npz

Implementation note:
This script follows the export conventions of `case_export_real.py` and uses
standard cartographic plotting components (matplotlib, cartopy, geopandas, and
cmocean where available) rather than reproducing third-party figure code.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

try:
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature

    HAS_CARTOPY = True
except ImportError:
    HAS_CARTOPY = False
    print("WARNING: cartopy not installed. Falling back to plain imshow.")

try:
    import cmocean

    HAS_CMOCEAN = True
except ImportError:
    HAS_CMOCEAN = False

try:
    import geopandas as gpd

    HAS_GEOPANDAS = True
except ImportError:
    HAS_GEOPANDAS = False

CODE_DIR = Path(__file__).resolve().parent
REPO_ROOT = CODE_DIR.parent
TP_SHAPEFILE_DIR = REPO_ROOT / "supportings" / "shapefiles"
DBATP2_DIR = REPO_ROOT / "DBATP2_shp"
CHINA_ADMIN_DIR = REPO_ROOT / "ChinaAdminDivisonSHP"
OUT_DIR = REPO_ROOT / "paper_figs"
_GDF_CACHE: dict[Path, object] = {}
_ADMIN_CONTEXT_MSG_SHOWN = False
_NO_OVERLAY_MSG_SHOWN = False


def setup_style():
    """Set publication-quality matplotlib defaults."""
    mpl.rcParams.update(
        {
            "font.family": "serif",
            "font.size": 9,
            "axes.labelsize": 10,
            "axes.titlesize": 10,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 8,
            "figure.dpi": 300,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.05,
        }
    )


def get_precip_cmap():
    """Return a perceptually uniform precipitation colormap."""
    if HAS_CMOCEAN:
        return cmocean.cm.rain
    return plt.cm.Blues


def get_diff_cmap():
    """Return a diverging colormap for difference fields."""
    if HAS_CMOCEAN:
        return cmocean.cm.balance
    return plt.cm.RdBu_r


def load_case(npz_path: Path):
    """Load a single exported .npz case."""
    data = np.load(npz_path, allow_pickle=True)
    return {k: data[k] for k in data.files}


def _read_gdf(path: Path, *, to_epsg4326: bool = False):
    """Read a shapefile once and cache it for repeated panel overlays."""
    cache_key = Path(f"{path}::epsg4326") if to_epsg4326 else path
    if cache_key not in _GDF_CACHE:
        gdf = gpd.read_file(path)
        if to_epsg4326 and gdf.crs is not None and not gdf.crs.is_geographic:
            gdf = gdf.to_crs(epsg=4326)
        _GDF_CACHE[cache_key] = gdf
    return _GDF_CACHE[cache_key]


def _plot_gdf(ax, gdf, edgecolor, linewidth, linestyle="-", alpha=1.0):
    """Plot a GeoDataFrame on either cartopy or plain matplotlib axes."""
    if HAS_CARTOPY:
        gdf.plot(
            ax=ax,
            facecolor="none",
            edgecolor=edgecolor,
            linewidth=linewidth,
            linestyle=linestyle,
            alpha=alpha,
            transform=ccrs.PlateCarree(),
        )
    else:
        gdf.plot(
            ax=ax,
            facecolor="none",
            edgecolor=edgecolor,
            linewidth=linewidth,
            linestyle=linestyle,
            alpha=alpha,
        )


def add_geographic_context(ax):
    """
    Add the strongest available geographic overlay.

    Preference order:
      1. Dedicated Tibetan Plateau boundary polygon
      2. China country + province administrative outlines for context

    Administrative boundaries are useful geographic context, but they are not a
    physical Tibetan Plateau mask and must never be described as one. When the
    DBATP2 boundary dataset is present, it is preferred as the study-domain
    overlay because it provides a physically motivated Tibetan Plateau polygon.
    """
    global _ADMIN_CONTEXT_MSG_SHOWN, _NO_OVERLAY_MSG_SHOWN

    if not HAS_GEOPANDAS:
        return False

    used_overlay = False

    country_shp = CHINA_ADMIN_DIR / "1. Country" / "country.shp"
    province_shp = CHINA_ADMIN_DIR / "2. Province" / "province.shp"

    if province_shp.exists():
        _plot_gdf(
            ax,
            _read_gdf(province_shp),
            edgecolor="0.35",
            linewidth=0.35,
            alpha=0.55,
        )
        used_overlay = True
    if country_shp.exists():
        _plot_gdf(ax, _read_gdf(country_shp), edgecolor="0.10", linewidth=0.85)
        used_overlay = True

    candidates = [
        DBATP2_DIR / "TPboundaryPolygon.shp",
        TP_SHAPEFILE_DIR / "tibetan_plateau_boundary.shp",
        TP_SHAPEFILE_DIR / "TP_boundary.shp",
        TP_SHAPEFILE_DIR / "tp_boundary.shp",
    ]
    for shp in candidates:
        if shp.exists():
            _plot_gdf(
                ax,
                _read_gdf(shp, to_epsg4326=True),
                edgecolor="k",
                linewidth=1.1,
            )
            return True

    if not used_overlay:
        if not _NO_OVERLAY_MSG_SHOWN:
            print(
                "  INFO: No dedicated TP boundary or China admin shapefile found. "
                "Proceeding without geographic overlay."
            )
            _NO_OVERLAY_MSG_SHOWN = True
    else:
        if not _ADMIN_CONTEXT_MSG_SHOWN:
            print(
                "  INFO: Using China administrative boundaries for geographic context; "
                "no dedicated Tibetan Plateau boundary polygon was found."
            )
            _ADMIN_CONTEXT_MSG_SHOWN = True
    return False


def compute_shared_scales(case_files: list[Path]):
    """Compute shared precipitation and difference scales across multiple cases."""
    precip_vals = []
    diff_vals = []
    for case_file in case_files:
        case = load_case(case_file)
        mask = case["fine_mask"].astype(bool)
        precip_vals.append(case["P_imerg_lift"][mask])
        diff_vals.append(np.abs(case["P_pred"][mask] - case["P_imerg_lift"][mask]))
    precip_cat = np.concatenate(precip_vals)
    diff_cat = np.concatenate(diff_vals)
    vmax_precip = max(float(np.percentile(precip_cat, 98)), 0.05)
    vmax_diff = max(float(np.percentile(diff_cat, 95)), 0.01)
    return {"vmax_precip": vmax_precip, "vmax_diff": vmax_diff}


def make_figure_cartopy(case, outpath: Path, scales: dict | None = None):
    """Generate the 3-panel geographic figure using cartopy."""
    setup_style()

    fine_lat = case["fine_lat"]
    fine_lon = case["fine_lon"]
    fine_mask = case["fine_mask"].astype(bool)

    p_pred = np.ma.masked_where(~fine_mask, case["P_pred"])
    p_lift = np.ma.masked_where(~fine_mask, case["P_imerg_lift"])
    p_diff = np.ma.masked_where(~fine_mask, case["P_pred"] - case["P_imerg_lift"])

    if scales is None:
        vmax_precip = float(np.percentile(case["P_imerg_lift"][fine_mask], 98))
        vmax_precip = max(vmax_precip, 0.05)
        vmax_diff = float(np.percentile(np.abs(p_diff.compressed()), 95))
        vmax_diff = max(vmax_diff, 0.01)
    else:
        vmax_precip = float(scales["vmax_precip"])
        vmax_diff = float(scales["vmax_diff"])

    extent = [fine_lon.min(), fine_lon.max(), fine_lat.min(), fine_lat.max()]
    proj = ccrs.PlateCarree()

    fig, axes = plt.subplots(
        1,
        3,
        figsize=(14, 4.2),
        subplot_kw={"projection": proj},
    )

    panels = [
        ("(a) IMERG coarse lift", p_lift, get_precip_cmap(), 0, vmax_precip),
        ("(b) Learned fine-grid precipitation (diagnostic)", p_pred, get_precip_cmap(), 0, vmax_precip),
        ("(c) Departure from lifted IMERG", p_diff, get_diff_cmap(), -vmax_diff, vmax_diff),
    ]

    for ax, (title, field, cmap, vmin, vmax) in zip(axes, panels):
        ax.set_extent(extent, crs=proj)

        im = ax.pcolormesh(
            fine_lon,
            fine_lat,
            field,
            transform=proj,
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            shading="nearest",
            rasterized=True,
        )

        ax.add_feature(cfeature.COASTLINE, linewidth=0.5, color="0.3")
        ax.add_feature(cfeature.BORDERS, linewidth=0.3, color="0.5", linestyle="--")

        gl = ax.gridlines(draw_labels=True, linewidth=0.3, color="0.6", alpha=0.5, linestyle=":")
        gl.top_labels = False
        gl.right_labels = False
        gl.xlocator = mticker.FixedLocator(np.arange(70, 110, 5))
        gl.ylocator = mticker.FixedLocator(np.arange(25, 42, 5))
        gl.xlabel_style = {"size": 7}
        gl.ylabel_style = {"size": 7}

        add_geographic_context(ax)
        ax.set_title(title, fontsize=10, fontweight="bold", pad=6)

        cbar = fig.colorbar(im, ax=ax, orientation="horizontal", fraction=0.06, pad=0.08, aspect=25)
        if title.startswith("(c)"):
            cbar.set_label("Difference (mm / 3 h)", fontsize=8)
        else:
            cbar.set_label("Precipitation (mm / 3 h)", fontsize=8)
        cbar.ax.tick_params(labelsize=7)

    plt.tight_layout()
    fig.savefig(outpath, bbox_inches="tight")
    print(f"  Saved: {outpath}")
    plt.close(fig)


def make_figure_plain(case, outpath: Path, scales: dict | None = None):
    """Fallback figure without cartopy."""
    setup_style()

    fine_mask = case["fine_mask"].astype(bool)
    p_pred = np.ma.masked_where(~fine_mask, case["P_pred"])
    p_lift = np.ma.masked_where(~fine_mask, case["P_imerg_lift"])
    p_diff = np.ma.masked_where(~fine_mask, case["P_pred"] - case["P_imerg_lift"])

    if scales is None:
        vmax = float(np.percentile(case["P_imerg_lift"][fine_mask], 98))
        vmax = max(vmax, 0.05)
        vmax_diff = float(np.percentile(np.abs(p_diff.compressed()), 95))
        vmax_diff = max(vmax_diff, 0.01)
    else:
        vmax = float(scales["vmax_precip"])
        vmax_diff = float(scales["vmax_diff"])

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))

    panels = [
        ("(a) IMERG coarse lift", p_lift, get_precip_cmap(), 0, vmax),
        ("(b) Learned fine-grid precipitation (diagnostic)", p_pred, get_precip_cmap(), 0, vmax),
        ("(c) Departure from lifted IMERG", p_diff, get_diff_cmap(), -vmax_diff, vmax_diff),
    ]
    for ax, (title, field, cmap, vmin, vmax_here) in zip(axes, panels):
        im = ax.imshow(field, origin="lower", cmap=cmap, vmin=vmin, vmax=vmax_here, aspect="auto")
        ax.set_title(title, fontsize=10, fontweight="bold")
        fig.colorbar(im, ax=ax, orientation="horizontal", fraction=0.06, pad=0.12)

    plt.tight_layout()
    fig.savefig(outpath, bbox_inches="tight")
    print(f"  Saved (plain): {outpath}")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="E3 real-data spatial diagnostic figure")
    parser.add_argument(
        "--case-dir",
        type=str,
        default=str(OUT_DIR / "case_data_real"),
        help="Directory containing exported .npz case files",
    )
    parser.add_argument(
        "--case-file",
        type=str,
        default=None,
        help="Single .npz file to plot (overrides --case-dir)",
    )
    parser.add_argument("--outdir", type=str, default=str(OUT_DIR))
    parser.add_argument("--format", choices=["pdf", "png", "both"], default="both")
    parser.add_argument(
        "--shared-scale",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Use one shared precipitation/difference scale when plotting multiple cases",
    )
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    if args.case_file:
        case_files = [Path(args.case_file)]
    else:
        case_dir = Path(args.case_dir)
        case_files = sorted(case_dir.glob("case_real_t*.npz"))

    if not case_files:
        print("No case files found. Run case_export_real.py first.")
        return

    print(f"Found {len(case_files)} case file(s)")
    shared_scales = None
    if args.shared_scale and len(case_files) > 1:
        shared_scales = compute_shared_scales(case_files)
        print(
            "Using shared scales across cases: "
            f"vmax_precip={shared_scales['vmax_precip']:.3f}, "
            f"vmax_diff={shared_scales['vmax_diff']:.3f}"
        )
    elif len(case_files) > 1:
        print("Using separate precipitation/difference scales for each case.")

    for case_file in case_files:
        print(f"\nProcessing {case_file.name}...")
        case = load_case(case_file)
        
        # Determine output formats
        formats = ["pdf", "png"] if args.format == "both" else [args.format]
        
        for fmt in formats:
            outpath = outdir / f"fig_e3_{case_file.stem}.{fmt}"
            if HAS_CARTOPY:
                make_figure_cartopy(case, outpath, scales=shared_scales)
            else:
                make_figure_plain(case, outpath, scales=shared_scales)

    print("\nAll figures generated.")


if __name__ == "__main__":
    main()
