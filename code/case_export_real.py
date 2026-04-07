#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Real-data inference export for the E3 spatial diagnostic figure.

Runs model inference on selected timesteps from the full Tibetan Plateau
domain and exports spatial fields as .npz files for geographic plotting.
There is NO ground truth for real data; all outputs are diagnostic only.

Implementation note:
This script mirrors the synthetic export workflow in `case_export.py` but
adapts it to full-domain real-data inference and truth-free diagnostics.

Exported fields per case:
  - P_pred       (H_f, W_f)  model-predicted fine precipitation
  - R_pred       (H_c, W_c)  model-predicted closure residual
  - P_imerg_lift (H_f, W_f)  IMERG coarse lift baseline
  - P_c_obs      (H_c, W_c)  coarse observed precipitation
  - dW           (H_f, W_f)  TPW tendency
  - fine_mask    (H_f, W_f)  valid-pixel mask (fine grid)
  - coarse_mask  (H_c, W_c)  valid-pixel mask (coarse grid)
  - fine_lat     (H_f,)      latitude coordinates
  - fine_lon     (W_f,)      longitude coordinates
  - coarse_lat   (H_c,)      coarse latitude
  - coarse_lon   (W_c,)      coarse longitude

Usage:
  python case_export_real.py \\
    --model-dir results/real_exact_W10Pc5R0_2020_20260323_085347 \\
    --timesteps 600 800 1000 \\
    --device cpu
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from scipy import sparse
from netCDF4 import Dataset as NC4Dataset

CODE_DIR = Path(__file__).resolve().parent
REPO_ROOT = CODE_DIR.parent
PREPROC = REPO_ROOT / "preprocessed"
EXPORT_DIR = REPO_ROOT / "paper_figs" / "case_data_real"


# ---------------------------------------------------------------------------
# Data loading for full-domain real data
# ---------------------------------------------------------------------------

def load_real_data(timesteps: list[int]):
    """
    Load real-data fields for specific timesteps only (memory-efficient).

    Returns dict with coordinates, masks, and per-timestep observation arrays.
    """
    domain_path = PREPROC / "common_domain.nc"
    with NC4Dataset(domain_path, "r") as ds:
        fine_mask = np.array(ds.variables["fine_mask"][:], dtype=bool)
        coarse_mask = np.array(ds.variables["coarse_mask"][:], dtype=bool)
        fine_lat = np.array(ds.variables["fine_lat"][:], dtype=np.float64)
        fine_lon = np.array(ds.variables["fine_lon"][:], dtype=np.float64)
        coarse_lat = np.array(ds.variables["coarse_lat"][:], dtype=np.float64)
        coarse_lon = np.array(ds.variables["coarse_lon"][:], dtype=np.float64)

    tpw_path = PREPROC / "tpw_3h.nc"
    imerg_path = PREPROC / "imerg_3h.nc"

    data = {
        "fine_mask": fine_mask,
        "coarse_mask": coarse_mask,
        "fine_lat": fine_lat,
        "fine_lon": fine_lon,
        "coarse_lat": coarse_lat,
        "coarse_lon": coarse_lon,
        "timesteps": {},
    }

    with NC4Dataset(tpw_path, "r") as tpw_ds, \
         NC4Dataset(imerg_path, "r") as imerg_ds:

        n_t = tpw_ds.dimensions["time"].size
        for t in timesteps:
            if t < 0 or t >= n_t:
                print(f"  WARNING: timestep {t} out of range [0, {n_t}), skipping")
                continue

            w0 = np.nan_to_num(
                np.array(tpw_ds.variables["tpw_t0"][t], dtype=np.float32),
                nan=0.0)
            w1 = np.nan_to_num(
                np.array(tpw_ds.variables["tpw_t1"][t], dtype=np.float32),
                nan=0.0)
            dw = w1 - w0
            pc = np.array(imerg_ds.variables["precipitation_3h"][t],
                          dtype=np.float32)

            data["timesteps"][t] = {
                "W0": w0, "W1": w1, "dW": dw, "P_c_obs": pc,
            }

    print(f"  Loaded {len(data['timesteps'])} timesteps from real data")
    return data


def load_operators():
    """Load full-domain support operators."""
    ops_path = PREPROC / "support_operators.npz"
    ops = np.load(ops_path, allow_pickle=True)
    Ac = sparse.csr_matrix(
        (ops["Ac_data"], ops["Ac_indices"], ops["Ac_indptr"]),
        shape=tuple(ops["Ac_shape"]))
    Ic = sparse.csr_matrix(
        (ops["Ic_data"], ops["Ic_indices"], ops["Ic_indptr"]),
        shape=tuple(ops["Ic_shape"]))
    return Ac, Ic


# ---------------------------------------------------------------------------
# Model inference
# ---------------------------------------------------------------------------

def run_inference(model_dir, data, Ic, timesteps, device="cpu"):
    """
    Load a trained model and run inference on selected timesteps.
    Returns list of dicts with P_pred, R_pred per timestep.
    """
    sys.path.insert(0, str(CODE_DIR))
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

    if r_param != "coarse":
        raise ValueError(
            "case_export_real.py currently supports only coarse-support residual "
            "models. Update the export path before using a fine-support model."
        )

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

    results = []
    with torch.no_grad():
        for t in timesteps:
            if t not in data["timesteps"]:
                continue
            td = data["timesteps"][t]
            pc_lift = np.array(
                Ic @ td["P_c_obs"].ravel(), dtype=np.float32
            ).reshape(n_flat, n_flon)

            x = np.stack([td["W0"], td["W1"], td["dW"], pc_lift], axis=0)[None]
            x_t = torch.from_numpy(x).to(device)

            P_hat_t, R_app_t = model(x_t)
            P_pred = P_hat_t.squeeze().cpu().numpy()
            R_pred = R_app_t.squeeze().cpu().numpy().reshape(n_clat, n_clon)

            results.append({
                "t": t,
                "P_pred": P_pred,
                "R_pred": R_pred,
                "P_imerg_lift": pc_lift,
                "dW": td["dW"],
                "P_c_obs": td["P_c_obs"],
            })
            mean_p = float(np.mean(P_pred[data["fine_mask"]]))
            mean_pc = float(np.mean(td["P_c_obs"][data["coarse_mask"]]))
            print(f"  t={t:4d}  mean P_pred={mean_p:.4f}  mean P_c={mean_pc:.4f}")

    return results


# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------

def export_case(data, inference_result, outdir):
    """Save one real-data case as .npz."""
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    r = inference_result
    t = r["t"]
    fname = f"case_real_t{t:04d}.npz"

    np.savez_compressed(
        outdir / fname,
        # Predictions (no truth available for real data)
        P_pred=r["P_pred"].astype(np.float32),
        R_pred=r["R_pred"].astype(np.float32),
        # Baselines and observations
        P_imerg_lift=r["P_imerg_lift"].astype(np.float32),
        P_c_obs=r["P_c_obs"].astype(np.float32),
        dW=r["dW"].astype(np.float32),
        # Coordinates and masks
        fine_mask=data["fine_mask"],
        coarse_mask=data["coarse_mask"],
        fine_lat=data["fine_lat"].astype(np.float64),
        fine_lon=data["fine_lon"].astype(np.float64),
        coarse_lat=data["coarse_lat"].astype(np.float64),
        coarse_lon=data["coarse_lon"].astype(np.float64),
        # Metadata
        timestep=t,
    )
    print(f"  Exported: {fname}")
    return outdir / fname


# ---------------------------------------------------------------------------
# Timestep selection helpers
# ---------------------------------------------------------------------------

def suggest_timesteps(n_suggestions=5):
    """
    Suggest representative timesteps based on IMERG coarse precipitation
    intensity. Useful when no specific timesteps are provided.
    """
    imerg_path = PREPROC / "imerg_3h.nc"
    domain_path = PREPROC / "common_domain.nc"

    with NC4Dataset(domain_path, "r") as ds:
        coarse_mask = np.array(ds.variables["coarse_mask"][:], dtype=bool)

    with NC4Dataset(imerg_path, "r") as ds:
        n_t = ds.dimensions["time"].size
        # Compute domain-mean precip per timestep (masked)
        mean_pc = np.zeros(n_t, dtype=np.float32)
        for t in range(n_t):
            pc = np.array(ds.variables["precipitation_3h"][t],
                          dtype=np.float32)
            mean_pc[t] = float(np.mean(pc[coarse_mask]))

    # Pick timesteps spanning the intensity range
    # Focus on the test split (last 17% = indices 1017-1223)
    from dataset import make_splits
    _, _, test_steps = make_splits(n_t)

    test_mean = mean_pc[test_steps]
    sorted_idx = np.argsort(test_mean)

    picks = []
    # Wettest
    picks.append(test_steps[sorted_idx[-1]])
    # 75th percentile
    picks.append(test_steps[sorted_idx[int(0.75 * len(sorted_idx))]])
    # Median
    picks.append(test_steps[sorted_idx[len(sorted_idx) // 2]])
    # 25th percentile
    picks.append(test_steps[sorted_idx[int(0.25 * len(sorted_idx))]])
    # Driest non-zero
    nz = [i for i in sorted_idx if test_mean[i] > 0.001]
    if nz:
        picks.append(test_steps[nz[0]])

    picks = picks[:n_suggestions]
    for t in picks:
        print(f"  Suggested t={t:4d}  mean_Pc={mean_pc[t]:.4f}")
    return picks


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Real-data inference export for E3 figure")
    parser.add_argument("--model-dir", type=str, required=True,
                        help="Path to real-data training results directory")
    parser.add_argument("--timesteps", type=int, nargs="*", default=None,
                        help="Specific timestep indices to export. "
                             "If omitted, suggests representative timesteps.")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--outdir", default=str(EXPORT_DIR))
    args = parser.parse_args()

    print("=" * 60)
    print("  Real-data case export for E3 diagnostic figure")
    print("=" * 60)

    # Determine timesteps
    if args.timesteps:
        timesteps = args.timesteps
    else:
        print("\nNo timesteps specified. Suggesting representative ones...")
        timesteps = suggest_timesteps(n_suggestions=3)

    print(f"\nTimesteps to export: {timesteps}")

    # Load operators
    print("\nLoading support operators...")
    Ac, Ic = load_operators()
    print(f"  A_c: {Ac.shape}, I_c: {Ic.shape}")

    # Load data for selected timesteps only
    print("\nLoading real data for selected timesteps...")
    data = load_real_data(timesteps)

    # Run inference
    print(f"\nRunning inference from {args.model_dir}...")
    results = run_inference(args.model_dir, data, Ic, timesteps, args.device)

    # Export
    print(f"\nExporting to {args.outdir}...")
    for r in results:
        export_case(data, r, args.outdir)

    print(f"\nDone. {len(results)} cases exported to {args.outdir}")


if __name__ == "__main__":
    main()
