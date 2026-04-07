#!/usr/bin/env python3
"""
Paper figure export script.
Generates all figures for the paper from local result files.
Unified Python stack: matplotlib + numpy. Cartopy is a future option for
full-domain geospatial maps, but is not required for the current synthetic
tile figures.

Usage:
    python paper_figures.py              # generate all figures
    python paper_figures.py --fig 5      # generate only Figure 5
    python paper_figures.py --fig 7      # generate only Figure 7
    python paper_figures.py --outdir figs # output to figs/ directory

Output: PDF files in the specified output directory (default: paper_figs/).
"""

import argparse
import json
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
CODE_DIR = Path(__file__).resolve().parent
PROJECT = CODE_DIR.parent
RESULTS = PROJECT / "results"
BASELINES = RESULTS / "baselines"
PREPROCESSED = PROJECT / "preprocessed"

# Case export directory (spatial fields from case_export.py)
CASE_DIR = PROJECT / "paper_figs" / "case_data"

# Key run directories
RUN_CTRL_VALLOSS = RESULTS / "synth_ctrl_W10Pc5R0_valLoss_Awet_20260323_052916"
RUN_CTRL_ORACLE = RESULTS / "synth_align_W10Pc5_Awet_20260323_044111"
RUN_A5_NAIVE = RESULTS / "synth_A5_naive_W10Pc5R0_Awet_20260323_052924"
RUN_A6_MATCHED = RESULTS / "synth_A6_pointwiseMatched_W10Pc5R0_Awet_20260323_053012"
RUN_BDRY = RESULTS / "synth_Bdry_exact_W10Pc5R0_valLoss_20260323_085403"
RUN_REAL = RESULTS / "real_exact_W10Pc5R0_2020_20260323_085347"

# Alignment sweep runs
SWEEP_RUNS = {
    "W1/Pc10": RESULTS / "synth_align_W1Pc10_Awet_20260323_042716",
    "W5/Pc5":  RESULTS / "synth_align_W5Pc5_Awet_20260323_043420",
    "W10/Pc5": RESULTS / "synth_align_W10Pc5_Awet_20260323_044111",
    "W5/Pc1":  RESULTS / "synth_align_W5Pc1_Awet_20260323_044525",
    "W10/Pc1": RESULTS / "synth_align_W10Pc1_Awet_20260323_044936",
}

# Baseline eval files (test split)
EVAL_CTRL_VALLOSS = BASELINES / "baseline_eval_A_wet_test_synth_ctrl_W10Pc5R0_valLoss_Awet_20260323_052916.json"
EVAL_CTRL_ORACLE = BASELINES / "baseline_eval_A_wet_test_synth_align_W10Pc5_Awet_20260323_044111.json"
EVAL_A3 = BASELINES / "baseline_eval_A_wet_test_synth_anchorRemoved_deploy_Awet.json"
EVAL_A5_NAIVE = BASELINES / "baseline_eval_A_wet_test_synth_A5_naive_W10Pc5R0_Awet_20260323_052924.json"
EVAL_A6_MATCHED = BASELINES / "baseline_eval_A_wet_test_synth_A6_pointwiseMatched_W10Pc5R0_Awet_20260323_053012.json"
EVAL_BDRY_TEST = BASELINES / "baseline_eval_B_dry_test_synth_Bdry_exact_W10Pc5R0_valLoss_20260323_085403.json"
EVAL_BDRY_VAL = BASELINES / "baseline_eval_B_dry_val_synth_Bdry_exact_W10Pc5R0_valLoss_20260323_085403.json"
EVAL_AWET_TEST = BASELINES / "baseline_eval_A_wet_test_baselines_n1224.json"
EVAL_SWEEP_DEPLOY = {
    "W1/Pc10": BASELINES / "baseline_eval_A_wet_test_synth_deploy_W1Pc10_Awet.json",
    "W5/Pc5": BASELINES / "baseline_eval_A_wet_test_synth_deploy_W5Pc5_Awet.json",
    "W10/Pc5": EVAL_CTRL_VALLOSS,
    "W5/Pc1": BASELINES / "baseline_eval_A_wet_test_synth_deploy_W5Pc1_Awet.json",
    "W10/Pc1": BASELINES / "baseline_eval_A_wet_test_synth_deploy_W10Pc1_Awet.json",
}

# Real-data diagnostics
REAL_DIAGNOSTICS = RUN_REAL / "real_checkpoint_diagnostics.json"

# Paper style
PAPER_RC = {
    "font.size": 10,
    "axes.labelsize": 11,
    "axes.titlesize": 11,
    "legend.fontsize": 9,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.05,
    "axes.grid": True,
    "grid.alpha": 0.3,
}

AXIS_LABEL_FONTSIZE = 8
PANEL_TITLE_FONTSIZE = 9
SUPTITLE_FONTSIZE = 11
SUPTITLE_LARGE_FONTSIZE = 12
CBAR_LABEL_FONTSIZE = 8
ANNOTATION_FONTSIZE = 8.5
DRY_PRECIP_DISPLAY_FLOOR = 0.05

# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def load_json(path):
    """Load a JSON file and return as dict."""
    with open(path, "r") as f:
        return json.load(f)


def load_history(run_dir):
    """Load history.json from a run directory."""
    return load_json(run_dir / "history.json")


def load_awet_imerg_lift():
    """Load the A_wet IMERG-lift median from the reference baseline."""
    ev = load_awet_base_eval()
    sec = ev.get("IMERG lift", {})
    val = sec.get("P_fine_nrmse", {}).get("median")
    if val is not None:
        return float(val)
    raise RuntimeError(
        f"Missing IMERG lift median in reference baseline {EVAL_AWET_TEST}"
    )


def load_awet_base_eval():
    """Load the reference A_wet baseline and validate provenance."""
    if not EVAL_AWET_TEST.exists():
        raise FileNotFoundError(f"Missing reference A_wet baseline file: {EVAL_AWET_TEST}")
    ev = load_json(EVAL_AWET_TEST)
    synth_dir = str(ev.get("_meta", {}).get("synthetic_dir", ""))
    synth_norm = synth_dir.replace("\\", "/").rstrip("/")
    if synth_dir and not synth_norm.endswith("preprocessed/synthetic"):
        raise RuntimeError(
            f"A_wet baseline provenance mismatch in {EVAL_AWET_TEST}: "
            f"expected preprocessed/synthetic, got {synth_dir}"
        )
    return ev


def ensure_outdir(outdir):
    """Create output directory if needed."""
    os.makedirs(outdir, exist_ok=True)


def save_fig(fig, outdir, name):
    """Save figure as PDF and PNG."""
    ensure_outdir(outdir)
    fig.savefig(os.path.join(outdir, f"{name}.pdf"))
    fig.savefig(os.path.join(outdir, f"{name}.png"))
    print(f"  Saved {name}.pdf and {name}.png")
    plt.close(fig)


def _eval_epochs(total_epochs, eval_every):
    """Return actual evaluation epochs used by train_real.py."""
    epochs = []
    for epoch in range(total_epochs):
        do_eval = ((epoch + 1) % eval_every == 0
                   or epoch == 0 or epoch == total_epochs - 1)
        if do_eval:
            epochs.append(epoch + 1)
    return epochs


def _checkpoint_label(metric_name):
    """Map checkpoint metric names to concise provenance labels."""
    if metric_name == "val_P_nrmse":
        return "oracle"
    if metric_name == "val_loss":
        return "validation-loss-selected"
    if metric_name == "supervised_p":
        return "supervised"
    return metric_name or "NA"


def _history_provenance(run_dir):
    """Return checkpoint provenance from history.json, inferring when needed."""
    hist_path = run_dir / "history.json"
    if not hist_path.exists():
        return "NA", "NA"

    hist = load_json(hist_path)
    cli = hist.get("cli_config", {})
    metric_name = (hist.get("best_metric_name")
                   or cli.get("checkpoint_metric_resolved")
                   or cli.get("checkpoint_metric"))
    if not metric_name and cli.get("train_objective") == "supervised_p":
        metric_name = "supervised_p"
    metric_name = metric_name or "NA"
    label = _checkpoint_label(metric_name)

    best_epoch = hist.get("best_epoch")
    if best_epoch is not None:
        return label, str(best_epoch)

    total_epochs = len(hist.get("train_loss", [])) or int(cli.get("epochs", 0) or 0)
    eval_every = int(cli.get("eval_every", 1) or 1)
    eval_epochs = _eval_epochs(total_epochs, eval_every) if total_epochs > 0 else []

    if metric_name == "val_P_nrmse":
        vals = hist.get("val_P_nrmse", [])
    else:
        vals = hist.get("val_loss", [])
    if vals and eval_epochs and len(vals) == len(eval_epochs):
        return label, str(eval_epochs[int(np.argmin(vals))])

    return label, "NA"


# ---------------------------------------------------------------------------
# Spatial map helpers
# ---------------------------------------------------------------------------

def _load_case(tile, tag, label="median"):
    """Load exported case .npz from CASE_DIR."""
    pattern = f"case_{tile}_{tag}_*_{label}.npz"
    matches = sorted(CASE_DIR.glob(pattern))
    if not matches:
        return None
    return dict(np.load(matches[0], allow_pickle=True))


def _load_geometry(tile):
    """Load exported geometry .npz."""
    path = CASE_DIR / f"geometry_{tile}.npz"
    if not path.exists():
        return None
    return dict(np.load(path, allow_pickle=True))


def _precip_cmap():
    """Return a precipitation-appropriate colormap."""
    from matplotlib.colors import LinearSegmentedColormap
    colors = ["#f7fbff", "#deebf7", "#9ecae1", "#3182bd",
              "#31a354", "#fee08b", "#fc8d59", "#d73027", "#7f0000"]
    return LinearSegmentedColormap.from_list("precip", colors, N=256)


def _plot_field(ax, field, lat, lon, mask=None, vmin=0, vmax=None,
                cmap=None, title="", cbar_label=""):
    """Plot a 2D spatial field on lat/lon axes with masking."""
    if cmap is None:
        cmap = _precip_cmap()
    ax.grid(False)
    plot_data = field.copy().astype(float)
    if mask is not None:
        plot_data[~mask] = np.nan
    im = ax.pcolormesh(lon, lat, plot_data, cmap=cmap, vmin=vmin, vmax=vmax,
                        shading="auto", rasterized=True)
    ax.set_xlabel("Longitude (\u00b0E)", fontsize=AXIS_LABEL_FONTSIZE)
    ax.set_ylabel("Latitude (\u00b0N)", fontsize=AXIS_LABEL_FONTSIZE)
    ax.set_title(title, fontsize=PANEL_TITLE_FONTSIZE)
    ax.set_aspect("auto")
    return im


# ---------------------------------------------------------------------------
# Figure 1: Problem geometry
# ---------------------------------------------------------------------------

def fig_problem_geometry(outdir):
    """
    Figure 1: Problem geometry schematic showing the domain,
    fine/coarse grids, TPW mask, and A_c @ I_c support mismatch.
    """
    print("Generating Figure 1: Problem geometry...")

    geom = _load_geometry("A_wet")
    if geom is None:
        print("  WARNING: geometry_A_wet.npz not found. Run case_export.py first.")
        return

    fine_lat = geom["fine_lat"]
    fine_lon = geom["fine_lon"]
    coarse_lat = geom["coarse_lat"]
    coarse_lon = geom["coarse_lon"]
    fine_mask = geom["fine_mask"].astype(bool)
    AcIc_diag = geom["AcIc_diag"]

    with plt.rc_context(PAPER_RC):
        fig, axes = plt.subplots(1, 3, figsize=(14, 4))
        fig.subplots_adjust(left=0.05, right=0.88, bottom=0.14, top=0.82, wspace=0.18)
        for ax in axes:
            ax.grid(False)

        # Panel (a): Fine-grid mask (TPW coverage)
        mask_float = fine_mask.astype(float)
        mask_float[~fine_mask] = np.nan
        im0 = axes[0].pcolormesh(fine_lon, fine_lat, mask_float,
                                  cmap="Blues", vmin=0, vmax=1,
                                  shading="auto", rasterized=True)
        x_edges = np.concatenate([
            [coarse_lon[0] - 0.5 * (coarse_lon[1] - coarse_lon[0])],
            0.5 * (coarse_lon[:-1] + coarse_lon[1:]),
            [coarse_lon[-1] + 0.5 * (coarse_lon[-1] - coarse_lon[-2])],
        ])
        y_edges = np.concatenate([
            [coarse_lat[0] - 0.5 * (coarse_lat[1] - coarse_lat[0])],
            0.5 * (coarse_lat[:-1] + coarse_lat[1:]),
            [coarse_lat[-1] + 0.5 * (coarse_lat[-1] - coarse_lat[-2])],
        ])
        for x in x_edges:
            axes[0].axvline(x, color="white", linewidth=0.12, alpha=0.35, zorder=3)
        for y in y_edges:
            axes[0].axhline(y, color="white", linewidth=0.12, alpha=0.35, zorder=3)
        axes[0].set_title(f"(a) Fine grid $G_f$ ({len(fine_lat)}\u00d7{len(fine_lon)})\n"
                          f"TPW valid mask ({100*fine_mask.mean():.0f}% valid) + coarse overlay",
                          fontsize=PANEL_TITLE_FONTSIZE)
        axes[0].set_xlabel("Longitude (\u00b0E)", fontsize=AXIS_LABEL_FONTSIZE)
        axes[0].set_ylabel("Latitude (\u00b0N)", fontsize=AXIS_LABEL_FONTSIZE)

        # Panel (b): Coarse grid with grid lines
        # Show coarse P_c coverage as uniform (all valid for synthetic)
        coarse_ones = np.ones((len(coarse_lat), len(coarse_lon)))
        im1 = axes[1].pcolormesh(coarse_lon, coarse_lat, coarse_ones,
                                  cmap="Oranges", vmin=0, vmax=1,
                                  shading="auto", rasterized=True,
                                  edgecolors="gray", linewidth=0.1)
        axes[1].set_title(f"(b) Coarse grid $G_c$ ({len(coarse_lat)}\u00d7{len(coarse_lon)})\n"
                          f"IMERG grid with valid-cell mask",
                          fontsize=PANEL_TITLE_FONTSIZE)
        axes[1].set_xlabel("Longitude (\u00b0E)", fontsize=AXIS_LABEL_FONTSIZE)
        axes[1].set_ylabel("Latitude (\u00b0N)", fontsize=AXIS_LABEL_FONTSIZE)

        # Panel (c): A_c @ I_c diagonal - support mismatch
        im2 = axes[2].pcolormesh(coarse_lon, coarse_lat, AcIc_diag,
                                  cmap="RdYlBu", vmin=0.5, vmax=1.05,
                                  shading="auto", rasterized=True)
        cbar2_ax = fig.add_axes([0.90, 0.22, 0.015, 0.46])
        cbar2_ax.grid(False)
        cbar2 = fig.colorbar(im2, cax=cbar2_ax)
        cbar2.set_label("diag($A_c I_c$)", fontsize=CBAR_LABEL_FONTSIZE)
        axes[2].set_title(f"(c) Support mismatch: diag($A_c I_c$)\n"
                          f"mean={AcIc_diag.mean():.3f}, "
                          f"$A_c I_c \\neq I$ (non-nested grids)",
                          fontsize=PANEL_TITLE_FONTSIZE)
        axes[2].set_xlabel("Longitude (\u00b0E)", fontsize=AXIS_LABEL_FONTSIZE)
        axes[2].set_ylabel("Latitude (\u00b0N)", fontsize=AXIS_LABEL_FONTSIZE)

        save_fig(fig, outdir, "fig1_problem_geometry")


# ---------------------------------------------------------------------------
# Figure 2: A3 degradation (no P_c anchor)
# ---------------------------------------------------------------------------

def fig_a3_degradation(outdir):
    """
    Figure 2: Spatial maps showing A3 degradation.
    3 columns: truth / ctrl (with P_c) / A3 (no P_c)
    Row: wettest representative test case from A_wet.
    """
    print("Generating Figure 2: A3 degradation maps...")

    ctrl = _load_case("A_wet", "ctrl_deploy", "wettest")
    a3 = _load_case("A_wet", "A3_noPc", "wettest")
    if ctrl is None or a3 is None:
        print("  WARNING: Case files not found. Run case_export.py --export-all.")
        return

    P_true = ctrl["P_true"]
    P_ctrl = ctrl["P_pred"]
    P_a3 = a3["P_pred"]
    fine_lat = ctrl["fine_lat"]
    fine_lon = ctrl["fine_lon"]
    mask = ctrl["fine_mask"].astype(bool)

    vmax = float(np.nanpercentile(P_true[mask], 98))
    cmap = _precip_cmap()
    with plt.rc_context(PAPER_RC):
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        fig.subplots_adjust(left=0.04, right=0.88, wspace=0.15, top=0.85)

        _plot_field(axes[0], P_true, fine_lat, fine_lon, mask,
                    vmax=vmax, cmap=cmap,
                    title="(a) $P_{true}$ (synthetic truth)")
        _plot_field(axes[1], P_ctrl, fine_lat, fine_lon, mask,
                    vmax=vmax, cmap=cmap,
                    title="(b) With $P_c$ anchor\n(validation-loss-selected model, test median nRMSE=12.39%)")
        im = _plot_field(axes[2], P_a3, fine_lat, fine_lon, mask,
                         vmax=vmax, cmap=cmap,
                         title="(c) No-anchor structural comparison\n(validation-loss-selected model, test median nRMSE=12.16%)")

        cbar_ax = fig.add_axes([0.90, 0.15, 0.015, 0.65])
        cbar_ax.grid(False)
        cbar = fig.colorbar(im, cax=cbar_ax)
        cbar.set_label("Precipitation (mm/3h)", fontsize=CBAR_LABEL_FONTSIZE)

        save_fig(fig, outdir, "fig2_a3_degradation")


# ---------------------------------------------------------------------------
# Figure 3: A6 coarse-R vs fine-R comparison
# ---------------------------------------------------------------------------

def fig_a6_comparison(outdir):
    """
    Figure 3: Spatial maps comparing coarse-support R (ctrl) vs
    fine-support R (A6 matched).
    3 columns: truth / coarse-R ctrl / fine-R A6 matched
    """
    print("Generating Figure 3: A6 comparison maps...")

    ctrl = _load_case("A_wet", "ctrl_oracle", "wettest")
    if ctrl is None:
        ctrl = _load_case("A_wet", "ctrl_deploy", "wettest")
    a6 = _load_case("A_wet", "A6_matched", "wettest")
    if ctrl is None or a6 is None:
        print("  WARNING: Case files not found. Run case_export.py --export-all.")
        return

    P_true = ctrl["P_true"]
    P_ctrl = ctrl["P_pred"]
    P_a6 = a6["P_pred"]
    fine_lat = ctrl["fine_lat"]
    fine_lon = ctrl["fine_lon"]
    mask = ctrl["fine_mask"].astype(bool)

    vmax = float(np.nanpercentile(P_true[mask], 98))
    cmap = _precip_cmap()
    with plt.rc_context(PAPER_RC):
        fig, axes = plt.subplots(2, 3, figsize=(15, 8))
        fig.subplots_adjust(left=0.04, right=0.88, wspace=0.15, hspace=0.30, top=0.95)

        # Top row: Precipitation fields
        _plot_field(axes[0, 0], P_true, fine_lat, fine_lon, mask,
                    vmax=vmax, cmap=cmap,
                    title="(a) $P_{true}$")
        _plot_field(axes[0, 1], P_ctrl, fine_lat, fine_lon, mask,
                    vmax=vmax, cmap=cmap,
                    title="(b) Coarse-$R$\n(oracle-selected, test median nRMSE=8.23%)")
        im_p = _plot_field(axes[0, 2], P_a6, fine_lat, fine_lon, mask,
                           vmax=vmax, cmap=cmap,
                           title="(c) Capacity-matched fine-support residual\n(test median nRMSE=14.15%)")
        cbar_ax_p = fig.add_axes([0.90, 0.53, 0.015, 0.33])
        cbar_ax_p.grid(False)
        cbar_p = fig.colorbar(im_p, cax=cbar_ax_p)
        cbar_p.set_label("$P$ (mm/3h)", fontsize=CBAR_LABEL_FONTSIZE)

        # Bottom row: Error maps |P_pred - P_true|
        err_ctrl = np.abs(P_ctrl - P_true)
        err_a6 = np.abs(P_a6 - P_true)
        err_vmax = float(np.nanpercentile(
            np.maximum(err_ctrl[mask], err_a6[mask]), 95))

        _plot_field(axes[1, 0], P_true, fine_lat, fine_lon, mask,
                    vmax=vmax, cmap=cmap,
                    title="(d) $P_{true}$ (repeated)")
        _plot_field(axes[1, 1], err_ctrl, fine_lat, fine_lon, mask,
                    vmax=err_vmax, cmap="Reds",
                    title="(e) Coarse-$R$ absolute error")
        im_e = _plot_field(axes[1, 2], err_a6, fine_lat, fine_lon, mask,
                           vmax=err_vmax, cmap="Reds",
                           title="(f) Fine-$R$ absolute error")
        cbar_ax_e = fig.add_axes([0.90, 0.10, 0.015, 0.33])
        cbar_ax_e.grid(False)
        cbar_e = fig.colorbar(im_e, cax=cbar_ax_e)
        cbar_e.set_label("$|\\Delta P|$ (mm/3h)", fontsize=CBAR_LABEL_FONTSIZE)

        save_fig(fig, outdir, "fig3_a6_comparison")


# ---------------------------------------------------------------------------
# Figure 4: Alignment sweep heatmap
# ---------------------------------------------------------------------------

def fig_alignment_sweep(outdir):
    """
    Figure 4: Two side-by-side heatmaps - oracle (val_P_nrmse checkpoint)
    vs validation-loss-selected (val_loss checkpoint) - so the W10/Pc5 cell is
    not
    silently overwritten.
    """
    print("Generating Figure 4: Alignment sweep heatmap...")

    oracle_data = []
    for label, run_dir in SWEEP_RUNS.items():
        lw, lpc = label.replace("W", "").replace("Pc", "").split("/")
        lw, lpc = int(lw), int(lpc)
        run_name = run_dir.name
        eval_path = BASELINES / f"baseline_eval_A_wet_test_{run_name}.json"
        if eval_path.exists():
            ev = load_json(eval_path)
            sec = ev.get("Trained model", {})
            val = sec.get("P_fine_nrmse", {}).get("median")
            if val is not None:
                oracle_data.append((lw, lpc, float(val)))

    deploy_data = []
    for label, eval_path in EVAL_SWEEP_DEPLOY.items():
        lw, lpc = label.replace("W", "").replace("Pc", "").split("/")
        lw, lpc = int(lw), int(lpc)
        if eval_path.exists():
            ev = load_json(eval_path)
            sec = ev.get("Trained model", {})
            val = sec.get("P_fine_nrmse", {}).get("median")
            if val is not None:
                deploy_data.append((lw, lpc, float(val)))

    if not oracle_data:
        print("  WARNING: No sweep data found. Skipping Figure 4.")
        return

    def _build_grid(data_pts, ws, pcs):
        g = np.full((len(ws), len(pcs)), np.nan)
        for lw, lpc, v in data_pts:
            if lw in ws and lpc in pcs:
                g[ws.index(lw), pcs.index(lpc)] = v
        return g

    all_ws = sorted(set(d[0] for d in oracle_data + deploy_data))
    all_pcs = sorted(set(d[1] for d in oracle_data + deploy_data))
    grid_oracle = _build_grid(oracle_data, all_ws, all_pcs)
    grid_deploy = _build_grid(deploy_data, all_ws, all_pcs)

    imerg_lift = load_awet_imerg_lift()
    deploy_min = np.nanmin(grid_deploy) if np.isfinite(grid_deploy).any() else np.nanmin(grid_oracle)
    deploy_max = np.nanmax(grid_deploy) if np.isfinite(grid_deploy).any() else np.nanmax(grid_oracle)
    vmin = min(np.nanmin(grid_oracle), deploy_min)
    vmax = max(np.nanmax(grid_oracle), deploy_max, imerg_lift + 1)

    with plt.rc_context(PAPER_RC):
        fig, axes = plt.subplots(1, 2, figsize=(12.6, 4.8), sharey=True)
        fig.subplots_adjust(left=0.07, right=0.85, bottom=0.16, top=0.90, wspace=0.12)

        for ax, grid, title_str in [
            (axes[0], grid_oracle, "(a) Oracle (best validation $P$ nRMSE epoch)"),
            (axes[1], grid_deploy, "(b) Validation-loss-selected run"),
        ]:
            ax.set_facecolor("#f0f0f0")
            im = ax.imshow(grid, cmap="RdYlGn_r", aspect="auto",
                           origin="lower", interpolation="nearest",
                           vmin=vmin, vmax=vmax)
            ax.set_xticks(range(len(all_pcs)))
            ax.set_xticklabels([str(p) for p in all_pcs])
            ax.set_yticks(range(len(all_ws)))
            ax.set_yticklabels([str(w) for w in all_ws])
            ax.set_xlabel(r"$\lambda_{P_c}$")
            ax.set_title(title_str, fontsize=PANEL_TITLE_FONTSIZE)
            ax.grid(False)

            for i in range(len(all_ws)):
                for j in range(len(all_pcs)):
                    v = grid[i, j]
                    if not np.isnan(v):
                        ax.text(j, i, f"{v:.1f}", ha="center", va="center",
                                fontsize=PANEL_TITLE_FONTSIZE + 1, fontweight="bold",
                                color="white" if v > 12 else "black")

        axes[0].set_ylabel(r"$\lambda_W$")

        cbar_ax = fig.add_axes([0.87, 0.24, 0.018, 0.47])
        cbar_ax.grid(False)
        cbar = fig.colorbar(im, cax=cbar_ax)
        cbar.set_label(r"Test $P$ nRMSE (%)", fontsize=CBAR_LABEL_FONTSIZE)
        fig.text(0.879, 0.18, f"IMERG lift: {imerg_lift:.2f}",
                 color="blue", fontsize=ANNOTATION_FONTSIZE, ha="left", va="top")

        save_fig(fig, outdir, "fig4_alignment_sweep")


# ---------------------------------------------------------------------------
# Figure 5: Semiconvergence plot
# ---------------------------------------------------------------------------

def fig_semiconvergence(outdir):
    """
    Figure 5: val_loss and val_P_nrmse divergence over training epochs.
    Shows the checkpoint-selection gap (best val_loss epoch != best val_P_nrmse epoch).
    """
    print("Generating Figure 5: Semiconvergence plot...")

    run = SWEEP_RUNS.get("W10/Pc5")
    if run is None or not (run / "history.json").exists():
        print("  WARNING: W10/Pc5 history not found. Skipping Figure 5.")
        return

    hist = load_history(run)
    eval_every = hist.get("cli_config", {}).get("eval_every", 5)
    val_loss = hist.get("val_loss", [])
    val_p_nrmse = hist.get("val_P_nrmse", [])

    if not val_loss or not val_p_nrmse:
        print("  WARNING: Missing metrics in history. Skipping Figure 5.")
        return

    n = min(len(val_loss), len(val_p_nrmse))
    val_loss = val_loss[:n]
    val_p_nrmse = val_p_nrmse[:n]
    all_eval_epochs = _eval_epochs(int(hist.get("cli_config", {}).get("epochs", 0) or 0),
                                   int(eval_every))
    epochs = all_eval_epochs[:n] if len(all_eval_epochs) >= n else [eval_every * (i + 1) for i in range(n)]

    with plt.rc_context(PAPER_RC):
        fig, ax1 = plt.subplots(figsize=(8.4, 4.8))
        fig.subplots_adjust(left=0.09, right=0.91, bottom=0.27, top=0.88)

        color1 = "#1f77b4"
        color2 = "#d62728"

        ax1.plot(epochs, val_loss, "o-", color=color1, markersize=4,
                 label="Validation loss")
        ax1.set_xlabel("Epoch")
        ax1.set_ylabel("Validation loss", color=color1)
        ax1.tick_params(axis="y", labelcolor=color1)

        ax2 = ax1.twinx()
        ax2.plot(epochs, val_p_nrmse, "s-", color=color2, markersize=4,
                 label=r"Validation $P$ nRMSE")
        ax2.set_ylabel(r"$P$ nRMSE (%)", color=color2)
        ax2.tick_params(axis="y", labelcolor=color2)

        best_loss_idx = int(np.argmin(val_loss))
        best_p_idx = int(np.argmin(val_p_nrmse))
        ax1.axvline(epochs[best_loss_idx], color=color1, linestyle="--",
                    alpha=0.5, label=f"best validation loss (ep {epochs[best_loss_idx]})")
        ax2.axvline(epochs[best_p_idx], color=color2, linestyle="--",
                    alpha=0.5, label=f"best $P$ nRMSE (ep {epochs[best_p_idx]})")
        ax2.axhline(load_awet_imerg_lift(), color="#2ca02c", linestyle=":", alpha=0.8,
                    label="IMERG lift")

        ax1.grid(True, alpha=0.25)
        ax2.grid(False)

        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2,
                   loc="upper center", bbox_to_anchor=(0.5, -0.20),
                   ncol=3, borderaxespad=0.0, fontsize=8, frameon=True,
                   columnspacing=1.0, handlelength=2.0)

        save_fig(fig, outdir, "fig5_semiconvergence")


# ---------------------------------------------------------------------------
# Figure 6: B_dry failure
# ---------------------------------------------------------------------------

def fig_bdry_failure(outdir):
    """
    Figure 6: B_dry failure - spatial maps (truth/predicted/IMERG) in top row,
    training dynamics (val_loss, val_P_nrmse, val_R_corr) in bottom row.
    """
    print("Generating Figure 6: B_dry failure...")

    if not (RUN_BDRY / "history.json").exists():
        print("  WARNING: B_dry history not found. Skipping Figure 6.")
        return

    hist = load_history(RUN_BDRY)
    eval_every = hist.get("cli_config", {}).get("eval_every", 5)
    val_loss = hist.get("val_loss", [])
    val_p_nrmse = hist.get("val_P_nrmse", [])
    val_r_corr = hist.get("val_R_corr", [])

    n = min(len(val_loss), len(val_p_nrmse))
    if n == 0:
        print("  WARNING: Empty B_dry history. Skipping Figure 6.")
        return

    epochs = [eval_every * (i + 1) for i in range(n)]
    val_loss = val_loss[:n]
    val_p_nrmse = val_p_nrmse[:n]
    val_r_corr = val_r_corr[:min(n, len(val_r_corr))]

    # --- Try to load spatial case data for top row ---
    bdry_case = _load_case("B_dry", "Bdry_deploy", "median")
    has_spatial = bdry_case is not None

    if has_spatial:
        n_rows = 2
        fig_h = 7.5
    else:
        n_rows = 1
        fig_h = 3.5

    with plt.rc_context(PAPER_RC):
        fig, axes = plt.subplots(n_rows, 3, figsize=(15, fig_h))
        if n_rows == 1:
            curve_axes = axes
            fig.subplots_adjust(left=0.06, right=0.96, wspace=0.30, top=0.85)
        else:
            curve_axes = axes[1]
            fig.subplots_adjust(left=0.04, right=0.88, wspace=0.15,
                                hspace=0.35, top=0.90)

        # --- Top row: spatial maps (if available) ---
        if has_spatial:
            P_true = bdry_case["P_true"]
            P_pred = bdry_case["P_pred"]
            P_lift = bdry_case["P_imerg_lift"]
            flat = bdry_case["fine_lat"]
            flon = bdry_case["fine_lon"]
            mask = bdry_case["fine_mask"].astype(bool)

            vmax_b = max(float(np.nanpercentile(P_true[mask], 98)),
                         DRY_PRECIP_DISPLAY_FLOOR)
            cmap = _precip_cmap()

            _plot_field(axes[0, 0], P_true, flat, flon, mask,
                        vmax=vmax_b, cmap=cmap,
                        title="(a) $P_{true}$ (B$_{dry}$, median timestep)")
            _plot_field(axes[0, 1], P_lift, flat, flon, mask,
                        vmax=vmax_b, cmap=cmap,
                        title="(b) IMERG lift")
            im_sp = _plot_field(axes[0, 2], P_pred, flat, flon, mask,
                                vmax=vmax_b, cmap=cmap,
                                title="(c) Trained model (false-rain artifacts)")
            cbar_ax_sp = fig.add_axes([0.90, 0.53, 0.015, 0.33])
            cbar_ax_sp.grid(False)
            cbar_sp = fig.colorbar(im_sp, cax=cbar_ax_sp)
            cbar_sp.set_label("$P$ (mm/3h)", fontsize=CBAR_LABEL_FONTSIZE)

        # --- Bottom row: training dynamics ---
        curve_axes[0].plot(epochs, val_loss, "o-", color="#1f77b4", markersize=4)
        curve_axes[0].set_xlabel("Epoch")
        curve_axes[0].set_ylabel("Validation loss")
        curve_axes[0].set_title("(d) Validation loss" if has_spatial else "(a) Validation loss",
                                fontsize=PANEL_TITLE_FONTSIZE)

        curve_axes[1].plot(epochs, val_p_nrmse, "s-", color="#d62728", markersize=4)
        curve_axes[1].set_xlabel("Epoch")
        curve_axes[1].set_ylabel(r"$P$ nRMSE (%)")
        curve_axes[1].set_title(r"(e) Validation $P$ nRMSE" if has_spatial
                                else r"(b) Validation $P$ nRMSE",
                                fontsize=PANEL_TITLE_FONTSIZE)

        # Reference lines for IMERG lift
        if EVAL_BDRY_TEST.exists():
            bdry_eval = load_json(EVAL_BDRY_TEST)
            il_sec = bdry_eval.get("IMERG lift", {})
            il_val = il_sec.get("P_fine_nrmse", {}).get("median")
            if il_val is not None:
                curve_axes[1].axhline(y=float(il_val), color="green",
                                      linestyle="--", alpha=0.7,
                                      label=f"IMERG lift (test {float(il_val):.0f})")
                curve_axes[1].legend(fontsize=7)

        if val_r_corr:
            ep_r = epochs[:len(val_r_corr)]
            curve_axes[2].plot(ep_r, val_r_corr, "^-", color="#2ca02c", markersize=4)
            curve_axes[2].set_xlabel("Epoch")
            curve_axes[2].set_ylabel(r"$R$ correlation")
            curve_axes[2].set_title(r"(f) Validation $R_{corr}$" if has_spatial
                                    else r"(c) Validation $R_{corr}$",
                                    fontsize=PANEL_TITLE_FONTSIZE)

        save_fig(fig, outdir, "fig6_bdry_failure")


# ---------------------------------------------------------------------------
# Figure 7: Real-data convergence
# ---------------------------------------------------------------------------

def fig_real_convergence(outdir):
    """
    Figure 7: Real-data training convergence showing val_loss, L_W, L_Pc
    over epochs.
    """
    print("Generating Figure 7: Real-data convergence...")

    rows = None
    if REAL_DIAGNOSTICS.exists():
        diag = load_json(REAL_DIAGNOSTICS)
        rows = diag.get("rows", [])

    if rows:
        # Sort by epoch, deduplicate (best_model and checkpoint_epoch_050 both epoch=50)
        seen = {}
        for r in rows:
            ep = r["epoch"]
            if ep not in seen:
                seen[ep] = r
        rows_sorted = sorted(seen.values(), key=lambda r: r["epoch"])
        epochs = [r["epoch"] for r in rows_sorted]
        val_loss = [r["val_loss"] for r in rows_sorted]
        val_lw = [r["val_L_W"] for r in rows_sorted]
        val_lpc = [r["val_L_Pc"] for r in rows_sorted]
    else:
        if not (RUN_REAL / "history.json").exists():
            print("  WARNING: Real-data history not found. Skipping Figure 7.")
            return
        hist = load_history(RUN_REAL)
        eval_every = hist.get("cli_config", {}).get("eval_every", 5)
        val_loss = hist.get("val_loss", [])
        val_lw = hist.get("val_L_W", [])
        val_lpc = hist.get("val_L_Pc", [])
        if not val_loss:
            print("  WARNING: No epoch data. Skipping Figure 7.")
            return
        epochs = [eval_every * (i + 1) for i in range(len(val_loss))]

    if not epochs:
        print("  WARNING: No epoch data. Skipping Figure 7.")
        return

    with plt.rc_context(PAPER_RC):
        fig, axes = plt.subplots(1, 3, figsize=(12, 3.5))

        axes[0].plot(epochs, val_loss, "o-", color="#1f77b4", markersize=5)
        axes[0].set_xlabel("Epoch")
        axes[0].set_ylabel("Validation loss")
        axes[0].set_title("(a) Total validation loss", fontsize=PANEL_TITLE_FONTSIZE)

        # Mark best
        best_idx = int(np.argmin(val_loss))
        axes[0].axvline(epochs[best_idx], color="red", linestyle="--",
                         alpha=0.5)
        axes[0].annotate(f"best: {val_loss[best_idx]:.3f}\n(ep {epochs[best_idx]})",
                          xy=(epochs[best_idx], val_loss[best_idx]),
                          xytext=(10, 10), textcoords="offset points",
                          fontsize=8, color="red",
                          arrowprops=dict(arrowstyle="->", color="red"))

        axes[1].plot(epochs, val_lw, "s-", color="#2ca02c", markersize=5)
        axes[1].set_xlabel("Epoch")
        axes[1].set_ylabel(r"$L_W$")
        axes[1].set_title(r"(b) Tendency-consistency loss $L_W$", fontsize=PANEL_TITLE_FONTSIZE)

        axes[2].plot(epochs, val_lpc, "^-", color="#d62728", markersize=5)
        axes[2].set_xlabel("Epoch")
        axes[2].set_ylabel(r"$L_{P_c}$")
        axes[2].set_title(r"(c) Coarse-precipitation loss $L_{P_c}$", fontsize=PANEL_TITLE_FONTSIZE)

        axes[0].text(0.02, 0.95, "No external truth\n(surrogate diagnostics only)",
                     transform=axes[0].transAxes,
                     va="top", ha="left",
                     fontsize=CBAR_LABEL_FONTSIZE, color="red",
                     bbox=dict(facecolor="white", alpha=0.8,
                               edgecolor="red"))

        fig.tight_layout()
        save_fig(fig, outdir, "fig7_real_convergence")


# ---------------------------------------------------------------------------
# Table 1: Full A_wet ablation results
# ---------------------------------------------------------------------------

def table_awet_ablation(outdir):
    """
    Support export for the wider A_wet audit surface.
    This markdown table is intended for reproducibility and audit.
    """
    print("Generating support export: A_wet audit table...")

    run_specs = [
        ("IMERG lift", None, "IMERG lift"),
        ("Analytical (noisy)", None, "Analytical (noisy)"),
        ("Supervised reference", "baseline_eval_A_wet_test_synth_A2_supervised_Awet_20260323_022128.json", "Trained model"),
        ("Anchor removed", "baseline_eval_A_wet_test_synth_anchorRemoved_deploy_Awet.json", "Trained model"),
        ("No residual regularization", "baseline_eval_A_wet_test_synth_A4_noRreg_Awet_20260323_031300.json", "Trained model"),
        ("Naive operators (default)", "baseline_eval_A_wet_test_synth_A5_naiveOps_Awet_20260323_023846.json", "Trained model"),
        ("Naive operators (W10/Pc5/R0)", "baseline_eval_A_wet_test_synth_A5_naive_W10Pc5R0_Awet_20260323_052924.json", "Trained model"),
        ("Fine-support residual (original)", "baseline_eval_A_wet_test_synth_A6_pointwise_Awet_20260323_024250.json", "Trained model"),
        ("Fine-support residual (matched)", "baseline_eval_A_wet_test_synth_A6_pointwiseMatched_W10Pc5R0_Awet_20260323_053012.json", "Trained model"),
        ("Val-loss exact W10/Pc5/R0", "baseline_eval_A_wet_test_synth_ctrl_W10Pc5R0_valLoss_Awet_20260323_052916.json", "Trained model"),
    ]

    for label, run_dir in SWEEP_RUNS.items():
        fname = f"baseline_eval_A_wet_test_{run_dir.name}.json"
        run_specs.append((f"Objective-weight config {label}", fname, "Trained model"))

    base_eval = load_awet_base_eval()

    def _fmt(v):
        if isinstance(v, (int, float)):
            return f"{v:.2f}"
        return "NA"

    header = [
        "# Support export: A_wet audit table",
        "",
        "This markdown export is wider than the compact publication table and is intended for reproducibility/audit use.",
        "Use the typeset paper table for the publication layout and this export for audit/provenance.",
        "",
    ]

    lines = [
        "| Run | Checkpoint | Epoch | P_nRMSE | R_corr | P_spCorr |",
        "|-----|------------|-------|---------|--------|----------|",
    ]

    for label, fname, section in run_specs:
        checkpoint_label = "NA"
        epoch_label = "NA"
        if fname is None:
            if base_eval is None:
                continue
            ev = base_eval
            checkpoint_label = "baseline"
        else:
            eval_path = BASELINES / fname
            if not eval_path.exists():
                continue
            ev = load_json(eval_path)
            run_name = fname.replace("baseline_eval_A_wet_test_", "").replace(".json", "")
            run_dir = RESULTS / run_name
            checkpoint_label, epoch_label = _history_provenance(run_dir)
            if checkpoint_label == "NA" and epoch_label != "NA":
                checkpoint_label = "legacy"

        sec = ev.get(section, {})
        p_nrmse = sec.get("P_fine_nrmse", {}).get("median", "NA")
        if label == "Supervised reference":
            r_corr = "NA"
        else:
            r_corr = sec.get("R_corr", {}).get("median", "NA")
        p_spcorr = sec.get("P_spatial_corr", {}).get("median", "NA")
        lines.append(
            f"| {label} | {checkpoint_label} | {epoch_label} | "
            f"{_fmt(p_nrmse)} | {_fmt(r_corr)} | {_fmt(p_spcorr)} |"
        )

    notes = [
        "",
        "Notes:",
        "- `baseline`: non-learned reference.",
        "- `supervised`: direct `P_true` supervision, not comparable to self-supervised training; only precipitation metrics are interpreted.",
        "- `oracle`: synthetic checkpoint selected by validation `P_nRMSE`.",
        "- `validation-loss-selected`: checkpoint selected by validation loss.",
        "- `legacy`: run whose stored history lacks checkpoint-metric metadata; epoch inferred from recorded validation history.",
    ]

    table_text = "\n".join(header + lines + notes)
    print(table_text)

    ensure_outdir(outdir)
    with open(os.path.join(outdir, "table1_awet_ablation.md"), "w", encoding="utf-8", newline="\n") as f:
        f.write(table_text)
    print("  Saved table1_awet_ablation.md")


# ---------------------------------------------------------------------------
# Table 2: B_dry results
# ---------------------------------------------------------------------------

def table_bdry_results(outdir):
    """
    Support export for the compact B_dry comparison surface.
    This markdown table is intended for reproducibility and audit.
    """
    print("Generating support export: B_dry comparison table...")

    header = [
        "# Support export: B_dry comparison table",
        "",
        "This markdown export is a compact audit table and omits the bootstrap confidence intervals shown in the publication table.",
        "Use the typeset paper table for the publication layout and this export for audit/provenance.",
        "",
    ]

    lines = [
        "| Method | Split | P_nRMSE | R_corr | P_spCorr | CSI@0.1 |",
        "|--------|-------|---------|--------|----------|---------|",
    ]

    def _fmt(v):
        if isinstance(v, (int, float)):
            return f"{v:.2f}" if abs(v) > 1 else f"{v:.3f}"
        return "NA"

    for split_label, eval_path in [("Val", EVAL_BDRY_VAL), ("Test", EVAL_BDRY_TEST)]:
        if not eval_path.exists():
            continue
        ev = load_json(eval_path)

        for section, method in [
            ("IMERG lift", "IMERG lift"),
            ("Analytical (noisy)", "Analytical (noisy)"),
            ("Trained model", "Val-loss exact"),
        ]:
            sec = ev.get(section, {})
            p_nrmse = sec.get("P_fine_nrmse", {}).get("median", "NA")
            r_corr = sec.get("R_corr", {}).get("median", "NA")
            p_spcorr = sec.get("P_spatial_corr", {}).get("median", "NA")
            csi = sec.get("CSI_0.1", sec.get("CSI_01", {}))
            if isinstance(csi, dict):
                csi = csi.get("median", "NA")
            elif not isinstance(csi, (int, float)):
                csi = "NA"

            lines.append(
                f"| {method} | {split_label} | {_fmt(p_nrmse)} | {_fmt(r_corr)} | {_fmt(p_spcorr)} | {_fmt(csi)} |"
            )

    table_text = "\n".join(header + lines)
    print(table_text)

    ensure_outdir(outdir)
    with open(os.path.join(outdir, "table2_bdry_results.md"), "w", encoding="utf-8", newline="\n") as f:
        f.write(table_text)
    print("  Saved table2_bdry_results.md")


def table_bdry_aux_metrics(outdir):
    """
    Support export for the compact dry-regime auxiliary metrics.
    This markdown table is intended for reproducibility and audit.
    """
    print("Generating support export: supplementary dry-regime diagnostics table...")

    header = [
        "# Support export: dry-regime auxiliary metrics",
        "",
        "This markdown export is a compact audit table aligned to the SI Table S6 metric set.",
        "Use the typeset supplementary table for the publication layout and this export for audit/provenance.",
        "",
    ]

    lines = [
        "| Method | P_RMSE | P_MAE | P_condRMSE@0.1 | POD@0.1 | FAR@0.1 | CSI@0.1 | P_spCorr | R_corr |",
        "|--------|--------|-------|----------------|---------|---------|---------|----------|--------|",
    ]

    def _fmt(v):
        if isinstance(v, (int, float)):
            return f"{v:.2f}" if abs(v) > 1 else f"{v:.3f}"
        return "NA"

    ev = load_json(EVAL_BDRY_TEST)
    for section, method in [
        ("IMERG lift", "IMERG lift"),
        ("Analytical (noisy)", "Analytical (noisy)"),
        ("Trained model", "Val-loss exact"),
    ]:
        sec = ev.get(section, {})
        p_rmse = sec.get("P_fine_rmse", {}).get("median", "NA")
        p_mae = sec.get("P_fine_mae", {}).get("median", "NA")
        p_cond = sec.get("P_cond_rmse_0.1", {}).get("median", "NA")
        pod = sec.get("POD_0.1", sec.get("POD_01", {}))
        if isinstance(pod, dict):
            pod = pod.get("median", "NA")
        elif not isinstance(pod, (int, float)):
            pod = "NA"
        far = sec.get("FAR_0.1", sec.get("FAR_01", {}))
        if isinstance(far, dict):
            far = far.get("median", "NA")
        elif not isinstance(far, (int, float)):
            far = "NA"
        csi = sec.get("CSI_0.1", sec.get("CSI_01", {}))
        if isinstance(csi, dict):
            csi = csi.get("median", "NA")
        elif not isinstance(csi, (int, float)):
            csi = "NA"
        p_spcorr = sec.get("P_spatial_corr", {}).get("median", "NA")
        r_corr = sec.get("R_corr", {}).get("median", "NA")

        lines.append(
            f"| {method} | {_fmt(p_rmse)} | {_fmt(p_mae)} | {_fmt(p_cond)} | "
            f"{_fmt(pod)} | {_fmt(far)} | {_fmt(csi)} | {_fmt(p_spcorr)} | {_fmt(r_corr)} |"
        )

    table_text = "\n".join(header + lines)
    print(table_text)

    ensure_outdir(outdir)
    with open(os.path.join(outdir, "table_s6_bdry_aux.md"), "w", encoding="utf-8", newline="\n") as f:
        f.write(table_text)
    print("  Saved table_s6_bdry_aux.md")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

FIGURE_DISPATCH = {
    1: fig_problem_geometry,
    2: fig_a3_degradation,
    3: fig_a6_comparison,
    4: fig_alignment_sweep,
    5: fig_semiconvergence,
    6: fig_bdry_failure,
    7: fig_real_convergence,
}

TABLE_DISPATCH = {
    1: table_awet_ablation,
    2: table_bdry_results,
    3: table_bdry_aux_metrics,
}


def main():
    parser = argparse.ArgumentParser(description="Paper figure export")
    parser.add_argument("--fig", type=int, default=None,
                        help="Generate only this figure number (4-7)")
    parser.add_argument("--table", type=int, default=None,
                        help="Generate only this table number (1-3)")
    parser.add_argument("--outdir", type=str, default="paper_figs",
                        help="Output directory")
    args = parser.parse_args()

    outdir = os.path.join(PROJECT, args.outdir)
    print(f"Output directory: {outdir}")

    if args.fig is not None:
        fn = FIGURE_DISPATCH.get(args.fig)
        if fn is None:
            print(f"Unknown figure number: {args.fig}. Available: {list(FIGURE_DISPATCH.keys())}")
            sys.exit(1)
        fn(outdir)
    elif args.table is not None:
        fn = TABLE_DISPATCH.get(args.table)
        if fn is None:
            print(f"Unknown table number: {args.table}. Available: {list(TABLE_DISPATCH.keys())}")
            sys.exit(1)
        fn(outdir)
    else:
        # Generate all
        for num, fn in sorted(FIGURE_DISPATCH.items()):
            fn(outdir)
        for num, fn in sorted(TABLE_DISPATCH.items()):
            fn(outdir)

    print("\nDone.")


if __name__ == "__main__":
    main()
