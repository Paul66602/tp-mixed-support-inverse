#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Case-export pipeline for paper-grade spatial figures.

Runs model inference on representative timesteps and exports spatial fields
as .npz files that the figure script can load for geographic maps.

Exported fields per case:
  - P_true       (H_f, W_f)  fine-grid truth precipitation
  - P_pred       (H_f, W_f)  model-predicted fine precipitation
  - P_imerg_lift (H_f, W_f)  IMERG coarse lift baseline
  - R_true       (H_c, W_c)  truth closure residual
  - R_pred       (H_c, W_c)  model-predicted residual
  - P_c_obs      (H_c, W_c)  coarse observed precipitation
  - dW           (H_f, W_f)  TPW tendency
  - fine_mask    (H_f, W_f)  valid-pixel mask (fine grid)
  - fine_lat     (H_f,)      latitude coordinates
  - fine_lon     (W_f,)      longitude coordinates
  - coarse_lat   (H_c,)      coarse latitude
  - coarse_lon   (W_c,)      coarse longitude

Usage:
  python case_export.py --tile A_wet --model-dir results/synth_ctrl_W10Pc5R0_valLoss_Awet_20260323_052916 --tag ctrl_deploy
  python case_export.py --tile A_wet --model-dir results/synth_align_W10Pc5_Awet_20260323_044111 --tag ctrl_oracle
  python case_export.py --tile A_wet --model-dir results/synth_anchorRemoved_deploy_Awet --tag A3_noPc
  python case_export.py --tile A_wet --model-dir results/synth_A6_pointwiseMatched_W10Pc5R0_Awet_20260323_053012 --tag A6_matched
  python case_export.py --tile B_dry --model-dir results/synth_Bdry_exact_W10Pc5R0_valLoss_20260323_085403 --tag Bdry_deploy
  python case_export.py --tile A_wet --synthetic-dir preprocessed/synthetic_misspecified/A_wet_fine_nullspace_eps0.20_beta0.30_seed42 --model-dir results/synth_misspec_ctrl --tag misspec_ctrl
  python case_export.py --export-all   # run all 5 key exports
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from scipy import sparse
from netCDF4 import Dataset

CODE_DIR = Path(__file__).resolve().parent
REPO_ROOT = CODE_DIR.parent
PREPROC = REPO_ROOT / "preprocessed"
DEFAULT_SYNTH_DIR = PREPROC / "synthetic"
EXPORT_DIR = REPO_ROOT / "paper_figs" / "case_data"


# ---------------------------------------------------------------------------
# Data loading (reuses baseline_evaluation patterns)
# ---------------------------------------------------------------------------

def resolve_synthetic_dir(synthetic_dir: str | Path | None = None) -> Path:
    """Resolve the synthetic dataset directory used for case export."""
    if synthetic_dir is None:
        return DEFAULT_SYNTH_DIR
    path = Path(synthetic_dir)
    if not path.is_absolute():
        path = REPO_ROOT / path
    return path.resolve()


def load_synthetic(
    tile: str,
    n_steps: int | None = None,
    synthetic_dir: str | Path | None = None,
):
    """Load synthetic truth and observations."""
    synth_dir = resolve_synthetic_dir(synthetic_dir)
    with Dataset(synth_dir / f"synthetic_truth_{tile}.nc", "r") as ds:
        ns = n_steps if n_steps else ds.dimensions["time"].size
        data = {
            "P_true": np.array(ds.variables["P_true"][:ns]),
            "R_app_true": np.array(ds.variables["R_app_true"][:ns]),
            "W_true": np.array(ds.variables["W_true"][:ns + 1]),
            "fine_mask": np.array(ds.variables["fine_mask"][:], dtype=bool),
            "fine_lat": np.array(ds.variables["fine_lat"][:]),
            "fine_lon": np.array(ds.variables["fine_lon"][:]),
            "coarse_lat": np.array(ds.variables["coarse_lat"][:]),
            "coarse_lon": np.array(ds.variables["coarse_lon"][:]),
        }
    with Dataset(synth_dir / f"synthetic_obs_{tile}.nc", "r") as ds:
        data["W_obs"] = np.array(ds.variables["W_obs"][:ns])
        data["P_c_obs"] = np.array(ds.variables["P_c_obs"][:ns])
    return data


def build_tile_operators(fine_lat, fine_lon, coarse_lat, coarse_lon):
    """Build tile-local support operators."""
    sys.path.insert(0, str(CODE_DIR))
    from build_support_operators import build_Ac, build_Ic
    return build_Ac(fine_lat, fine_lon, coarse_lat, coarse_lon), \
           build_Ic(fine_lat, fine_lon, coarse_lat, coarse_lon)


# ---------------------------------------------------------------------------
# Case selection: pick representative timesteps
# ---------------------------------------------------------------------------

def select_cases(data, n_cases=3, split_steps=None):
    """
    Select representative timesteps from the test split:
      - wettest  (max mean P_true)
      - median   (median mean P_true)
      - driest   (min mean P_true among non-trivial steps)

    Returns list of (timestep_index, label) tuples.
    """
    if split_steps is None:
        from dataset import make_splits
        n_t = data["P_true"].shape[0]
        _, _, test_s = make_splits(n_t)
        split_steps = test_s

    # Filter to valid steps (t+1 must exist for dW)
    valid = [t for t in split_steps if t + 1 < data["W_obs"].shape[0]]
    if not valid:
        print("  WARNING: No valid test steps.")
        return []

    # Compute mean P_true per step
    mean_p = [(t, float(np.mean(data["P_true"][t]))) for t in valid]
    mean_p.sort(key=lambda x: x[1])

    # Pick driest (non-trivial), median, wettest
    # Filter out steps where mean P < 0.001 (effectively zero)
    nontrivial = [(t, mp) for t, mp in mean_p if mp > 0.001]
    if len(nontrivial) < 3:
        nontrivial = mean_p  # fallback

    cases = []
    cases.append((nontrivial[-1][0], "wettest"))
    mid = len(nontrivial) // 2
    cases.append((nontrivial[mid][0], "median"))
    cases.append((nontrivial[0][0], "driest"))

    return cases


# ---------------------------------------------------------------------------
# Model inference
# ---------------------------------------------------------------------------

def run_inference(model_dir, data, Ac, Ic, timesteps, device="cpu"):
    """
    Load a trained model and run inference on selected timesteps.
    Returns list of dicts with P_pred, R_pred per timestep.
    """
    sys.path.insert(0, str(BASE))
    from model import PrecipBudgetNet

    model_path = Path(model_dir) / "best_model.pt"
    hist_path = Path(model_dir) / "history.json"

    if not model_path.exists():
        raise FileNotFoundError(f"No best_model.pt in {model_dir}")

    with open(hist_path) as f:
        hist = json.load(f)

    cli_config = hist.get("cli_config", {})
    r_param = str(cli_config.get("r_parameterization", "coarse"))
    n_clat = len(data["coarse_lat"])
    n_clon = len(data["coarse_lon"])
    n_flat = len(data["fine_lat"])
    n_flon = len(data["fine_lon"])

    model = PrecipBudgetNet(
        in_channels=4,
        base_channels=hist.get("base_ch", 32),
        n_levels=hist.get("n_levels", 3),
        coarse_shape=(n_clat, n_clon),
        norm_type=cli_config.get("norm_type", "batch"),
        r_parameterization=r_param,
    ).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    W_filled = np.nan_to_num(data["W_obs"], nan=0.0)
    results = []

    with torch.no_grad():
        for t, label in timesteps:
            if t + 1 >= W_filled.shape[0]:
                continue
            w0 = W_filled[t]
            w1 = W_filled[t + 1]
            dw = w1 - w0
            pc_obs = data["P_c_obs"][t]
            pc_lift = np.array(
                Ic @ pc_obs.ravel(), dtype=np.float32
            ).reshape(n_flat, n_flon)

            x = np.stack([w0, w1, dw, pc_lift], axis=0)[None]
            x_t = torch.from_numpy(x.astype(np.float32)).to(device)
            P_hat_t, R_app_t = model(x_t)
            P_pred = P_hat_t.squeeze().cpu().numpy()

            if r_param == "fine":
                R_fine = R_app_t.squeeze().cpu().numpy()
                R_pred = np.array(Ac @ R_fine.ravel()).reshape(n_clat, n_clon)
            else:
                R_pred = R_app_t.squeeze().cpu().numpy().reshape(n_clat, n_clon)

            results.append({
                "t": t,
                "label": label,
                "P_pred": P_pred,
                "R_pred": R_pred,
                "P_imerg_lift": pc_lift,
                "dW": dw,
                "P_c_obs": pc_obs,
            })

    return results


# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------

def export_case(data, inference_result, tag, tile, outdir):
    """Save one case as .npz."""
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    r = inference_result
    t = r["t"]
    fname = f"case_{tile}_{tag}_t{t:04d}_{r['label']}.npz"

    np.savez_compressed(
        outdir / fname,
        # Truth
        P_true=data["P_true"][t].astype(np.float32),
        R_true=data["R_app_true"][t].astype(np.float32),
        # Predictions
        P_pred=r["P_pred"].astype(np.float32),
        R_pred=r["R_pred"].astype(np.float32),
        # Baselines
        P_imerg_lift=r["P_imerg_lift"].astype(np.float32),
        P_c_obs=r["P_c_obs"].astype(np.float32),
        dW=r["dW"].astype(np.float32),
        # Coordinates and mask
        fine_mask=data["fine_mask"],
        fine_lat=data["fine_lat"].astype(np.float32),
        fine_lon=data["fine_lon"].astype(np.float32),
        coarse_lat=data["coarse_lat"].astype(np.float32),
        coarse_lon=data["coarse_lon"].astype(np.float32),
        # Metadata
        tile=tile,
        tag=tag,
        timestep=t,
        label=r["label"],
    )
    print(f"  Exported: {fname}")
    return outdir / fname


def export_geometry(data, Ac, Ic, tile, outdir):
    """Export geometry-only data for F1 (problem geometry figure)."""
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # Compute A_c @ I_c diagonal for support mismatch visualization
    AcIc = Ac @ Ic
    diag = np.array(AcIc.diagonal(), dtype=np.float32)
    n_clat = len(data["coarse_lat"])
    n_clon = len(data["coarse_lon"])
    diag_2d = diag.reshape(n_clat, n_clon)

    fname = f"geometry_{tile}.npz"
    np.savez_compressed(
        outdir / fname,
        fine_mask=data["fine_mask"],
        fine_lat=data["fine_lat"].astype(np.float32),
        fine_lon=data["fine_lon"].astype(np.float32),
        coarse_lat=data["coarse_lat"].astype(np.float32),
        coarse_lon=data["coarse_lon"].astype(np.float32),
        AcIc_diag=diag_2d,
        tile=tile,
    )
    print(f"  Exported: {fname}")


# ---------------------------------------------------------------------------
# Predefined export configurations
# ---------------------------------------------------------------------------

ALL_EXPORTS = [
    {
        "tile": "A_wet",
        "model_dir": "results/synth_ctrl_W10Pc5R0_valLoss_Awet_20260323_052916",
        "tag": "ctrl_deploy",
    },
    {
        "tile": "A_wet",
        "model_dir": "results/synth_align_W10Pc5_Awet_20260323_044111",
        "tag": "ctrl_oracle",
    },
    {
        "tile": "A_wet",
        "model_dir": "results/synth_anchorRemoved_deploy_Awet",
        "tag": "A3_noPc",
    },
    {
        "tile": "A_wet",
        "model_dir": "results/synth_A6_pointwiseMatched_W10Pc5R0_Awet_20260323_053012",
        "tag": "A6_matched",
    },
    {
        "tile": "B_dry",
        "model_dir": "results/synth_Bdry_exact_W10Pc5R0_valLoss_20260323_085403",
        "tag": "Bdry_deploy",
    },
]


def main():
    parser = argparse.ArgumentParser(description="Case export for paper figures")
    parser.add_argument("--tile", type=str, default=None)
    parser.add_argument("--model-dir", type=str, default=None)
    parser.add_argument("--tag", type=str, default=None)
    parser.add_argument("--synthetic-dir", default=None,
                        help="Synthetic dataset directory. Defaults to "
                             "preprocessed/synthetic.")
    parser.add_argument("--n-steps", type=int, default=1224)
    parser.add_argument("--n-cases", type=int, default=3,
                        help="Number of representative timesteps per run")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--outdir", default=str(EXPORT_DIR))
    parser.add_argument("--export-all", action="store_true",
                        help="Export all predefined configurations")
    parser.add_argument("--geometry-only", action="store_true",
                        help="Export only geometry data (no model inference)")
    args = parser.parse_args()

    if args.export_all:
        configs = ALL_EXPORTS
    elif args.tile and args.model_dir and args.tag:
        configs = [{"tile": args.tile, "model_dir": args.model_dir,
                     "tag": args.tag}]
    elif args.geometry_only and args.tile:
        configs = [{"tile": args.tile, "model_dir": None, "tag": None}]
    else:
        parser.error("Provide --export-all, or --tile + --model-dir + --tag, "
                      "or --geometry-only + --tile")

    # Group by tile to avoid reloading data
    from collections import defaultdict
    by_tile = defaultdict(list)
    for cfg in configs:
        by_tile[cfg["tile"]].append(cfg)

    for tile, tile_configs in by_tile.items():
        print(f"\n{'='*60}")
        print(f"  Tile: {tile}")
        print(f"{'='*60}")

        print("  Loading data...")
        synth_dir = resolve_synthetic_dir(args.synthetic_dir)
        print(f"  Synthetic source: {synth_dir}")
        data = load_synthetic(tile, args.n_steps, synth_dir)
        n_clat = len(data["coarse_lat"])
        n_clon = len(data["coarse_lon"])
        n_flat = len(data["fine_lat"])
        n_flon = len(data["fine_lon"])
        print(f"  Coarse: {n_clat}x{n_clon}, Fine: {n_flat}x{n_flon}")

        print("  Building operators...")
        Ac, Ic = build_tile_operators(
            data["fine_lat"], data["fine_lon"],
            data["coarse_lat"], data["coarse_lon"])

        # Always export geometry
        export_geometry(data, Ac, Ic, tile, args.outdir)

        if args.geometry_only:
            continue

        # Select representative timesteps
        cases = select_cases(data, n_cases=args.n_cases)
        print(f"  Selected cases: {[(t, l) for t, l in cases]}")

        for cfg in tile_configs:
            if cfg["model_dir"] is None:
                continue
            model_dir = REPO_ROOT / cfg["model_dir"]
            tag = cfg["tag"]
            print(f"\n  --- {tag} ({model_dir.name}) ---")

            try:
                results = run_inference(
                    model_dir, data, Ac, Ic, cases, args.device)
                for r in results:
                    export_case(data, r, tag, tile, args.outdir)
            except Exception as e:
                print(f"  ERROR: {e}")
                import traceback
                traceback.print_exc()

    print(f"\nDone. Exports in: {args.outdir}")


if __name__ == "__main__":
    main()
