#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Comprehensive baseline evaluation for the support-explicit inverse study.

Computes all baselines and metrics needed for the paper's evidence chain:
  1. IMERG lift baseline (P_hat = I_c(P_c_obs))
  2. Analytical inversion baseline (closed-form per-timestep)
  3. Trained model evaluation (load best_model.pt from a results dir)

Metrics include:
  - P fine nRMSE (standard)
  - P fine RMSE / MAE (whole-field absolute error)
  - P fine conditional RMSE (wet pixels only, P_true > threshold)
  - P coarse nRMSE (A_c consistency)
  - R_app spatial correlation
  - Categorical: POD, FAR, CSI at multiple thresholds
  - Within-cell anomaly norm (measures fine-scale structure beyond lifting)

Usage:
    python baseline_evaluation.py --tile A_wet --n-steps 1224
    python baseline_evaluation.py --tile B_dry --n-steps 1224
    python baseline_evaluation.py --tile A_wet --model-dir results/synth_exact_Awet_v1_20260321_153438
    python baseline_evaluation.py --tile A_wet --model-dir results/synth_exact_Awet_v1_20260321_153438 --split val
    python baseline_evaluation.py --tile A_wet --synthetic-dir preprocessed/synthetic_misspecified/A_wet_fine_nullspace_eps0.20_beta0.30_seed42
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Iterable

import numpy as np
from scipy import sparse
from netCDF4 import Dataset

CODE_DIR = Path(__file__).resolve().parent
REPO_ROOT = CODE_DIR.parent
PREPROC = REPO_ROOT / "preprocessed"
DEFAULT_SYNTH_DIR = PREPROC / "synthetic"


def resolve_synthetic_dir(synthetic_dir: str | Path | None = None) -> Path:
    """Resolve the synthetic dataset directory used for evaluation."""
    if synthetic_dir is None:
        return DEFAULT_SYNTH_DIR
    path = Path(synthetic_dir)
    if not path.is_absolute():
        path = REPO_ROOT / path
    return path.resolve()

# ---------------------------------------------------------------------------
# Operator construction
# ---------------------------------------------------------------------------
def build_tile_operators(fine_lat, fine_lon, coarse_lat, coarse_lon):
    sys.path.insert(0, str(CODE_DIR))
    from build_support_operators import build_Ac, build_Ic
    return build_Ac(fine_lat, fine_lon, coarse_lat, coarse_lon), \
           build_Ic(fine_lat, fine_lon, coarse_lat, coarse_lon)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def load_synthetic(
    tile: str,
    n_steps: int | None = None,
    synthetic_dir: str | Path | None = None,
):
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


# ---------------------------------------------------------------------------
# Metric functions
# ---------------------------------------------------------------------------
def nrmse(pred, true):
    """Normalized RMSE: RMSE / mean(|true|)."""
    denom = max(np.mean(np.abs(true)), 1e-8)
    return float(np.sqrt(np.mean((pred - true) ** 2)) / denom)


def rmse(pred, true):
    """Whole-field RMSE."""
    return float(np.sqrt(np.mean((pred - true) ** 2)))


def mae(pred, true):
    """Whole-field MAE."""
    return float(np.mean(np.abs(pred - true)))


def conditional_rmse(pred, true, threshold=0.1):
    """RMSE computed only where true > threshold (wet pixels)."""
    wet = true > threshold
    if wet.sum() < 10:
        return float("nan")
    return float(np.sqrt(np.mean((pred[wet] - true[wet]) ** 2)))


def categorical_scores(pred, true, threshold=0.1):
    """POD, FAR, CSI for precipitation detection."""
    pred_yes = pred > threshold
    true_yes = true > threshold

    hits = (pred_yes & true_yes).sum()
    misses = (~pred_yes & true_yes).sum()
    false_alarms = (pred_yes & ~true_yes).sum()

    pod = hits / max(hits + misses, 1)
    far = false_alarms / max(hits + false_alarms, 1)
    csi = hits / max(hits + misses + false_alarms, 1)
    return {"POD": float(pod), "FAR": float(far), "CSI": float(csi)}


def within_cell_anomaly_norm(P_hat, Ac, Ic, n_flat, n_flon):
    """
    Measures the norm of within-coarse-cell fine-scale anomaly.
    delta_P = P_hat - I_c(A_c(P_hat))
    This is the component of P_hat that is invisible to coarse constraints.
    """
    P_flat = P_hat.ravel()
    P_coarse = np.array(Ac @ P_flat)
    P_lifted_back = np.array(Ic @ P_coarse).reshape(n_flat, n_flon)
    delta = P_hat - P_lifted_back
    return float(np.sqrt(np.mean(delta ** 2)))


def spatial_correlation(pred, true):
    """Pearson correlation over flattened spatial fields."""
    r = np.corrcoef(pred.ravel(), true.ravel())[0, 1]
    return float(r) if not np.isnan(r) else 0.0


def compute_all_metrics(P_hat, P_true, R_hat, R_true,
                        Ac, Ic, n_clat, n_clon, n_flat, n_flon,
                        thresholds=(0.01, 0.1, 0.5)):
    """Compute full metric suite for one timestep."""
    m = {}

    # P fine nRMSE (standard)
    m["P_fine_nrmse"] = nrmse(P_hat, P_true)
    m["P_fine_rmse"] = rmse(P_hat, P_true)
    m["P_fine_mae"] = mae(P_hat, P_true)

    # P fine conditional RMSE at thresholds
    for th in thresholds:
        m[f"P_cond_rmse_{th}"] = conditional_rmse(P_hat, P_true, th)

    # P coarse nRMSE
    Pc_hat = np.array(Ac @ P_hat.ravel()).reshape(n_clat, n_clon)
    Pc_true = np.array(Ac @ P_true.ravel()).reshape(n_clat, n_clon)
    m["Pc_nrmse"] = nrmse(Pc_hat, Pc_true)

    # R_app correlation
    if R_hat is not None and R_true is not None:
        m["R_corr"] = spatial_correlation(R_hat, R_true)

    # Categorical at thresholds
    for th in thresholds:
        cat = categorical_scores(P_hat, P_true, th)
        for k, v in cat.items():
            m[f"{k}_{th}"] = v

    # Within-cell anomaly
    m["within_cell_anomaly"] = within_cell_anomaly_norm(
        P_hat, Ac, Ic, n_flat, n_flon)

    # P spatial correlation
    m["P_spatial_corr"] = spatial_correlation(P_hat, P_true)

    return m


# ---------------------------------------------------------------------------
# Baseline methods
# ---------------------------------------------------------------------------
def imerg_lift_baseline(data, Ic, Ac, n_clat, n_clon, n_flat, n_flon,
                        eval_steps: Iterable[int]):
    """Baseline: P_hat = I_c(P_c_obs). No learning, no budget."""
    all_metrics = []
    for t in eval_steps:
        P_hat = np.array(
            Ic @ data["P_c_obs"][t].ravel()
        ).reshape(n_flat, n_flon)
        m = compute_all_metrics(
            P_hat, data["P_true"][t], None, None,
            Ac, Ic, n_clat, n_clon, n_flat, n_flon)
        all_metrics.append(m)
    return all_metrics


def analytical_baseline(data, Ac, Ic, n_clat, n_clon, n_flat, n_flon,
                        eval_steps: Iterable[int], noisy=True):
    """
    Exact operator-consistent analytical inversion baseline.

    This solves

        A_c P = P_c
        P = I_c R - dW

    by the exact coarse solve

        (A_c I_c) R = P_c + A_c dW

    rather than the older approximation R = P_c + A_c dW, which is only valid
    if A_c I_c = I.

    When `noisy=True`, the method requires a paired `(t, t+1)` water-storage
    observation to form `dW_obs`. If a split contains the final time index with
    no successor observation, that index is skipped by construction.
    """
    from scipy.sparse.linalg import splu
    all_metrics = []
    AcIc_lu = splu((Ac @ Ic).tocsc())
    for t in eval_steps:
        if noisy:
            dW = np.nan_to_num(
                data["W_obs"][t + 1] - data["W_obs"][t], nan=0.0) \
                if t + 1 < data["W_obs"].shape[0] else None
            rhs = (
                data["P_c_obs"][t].ravel().astype(np.float64)
                + np.array(Ac @ dW.ravel(), dtype=np.float64)
            ) if dW is not None else None
        else:
            dW = data["W_true"][t + 1] - data["W_true"][t]
            rhs = (
                np.array(Ac @ data["P_true"][t].ravel(), dtype=np.float64)
                + np.array(Ac @ dW.ravel(), dtype=np.float64)
            )

        if dW is None or rhs is None:
            continue

        R_hat = AcIc_lu.solve(rhs)
        P_hat = np.array(Ic @ R_hat) - dW.ravel()
        P_hat_2d = P_hat.reshape(n_flat, n_flon)
        R_hat_2d = R_hat.reshape(n_clat, n_clon)

        m = compute_all_metrics(
            P_hat_2d, data["P_true"][t],
            R_hat_2d, data["R_app_true"][t],
            Ac, Ic, n_clat, n_clon, n_flat, n_flon)
        all_metrics.append(m)
    return all_metrics


def trained_model_baseline(data, model_dir, Ac, Ic,
                           n_clat, n_clon, n_flat, n_flon,
                           n_steps, device="cpu", split="test"):
    """Evaluate a trained v1/v2 model from a results directory.

    All learned models consume `(W_t, W_{t+1}, dW_t)`, so the final split index
    is skipped when no successor observation exists.
    """
    import torch
    from scipy.sparse.linalg import splu
    sys.path.insert(0, str(BASE))
    from model import PrecipBudgetNet, AnalyticalCorrectionNet
    from dataset import make_splits

    # Load model
    model_path = Path(model_dir) / "best_model.pt"
    hist_path = Path(model_dir) / "history.json"
    if not model_path.exists():
        print(f"  WARNING: {model_path} not found, skipping.")
        return None

    with open(hist_path) as f:
        hist = json.load(f)

    model_version = hist.get("model_version", "v1")
    cli_config = hist.get("cli_config", {})
    resolved_norm_type = cli_config.get("norm_type")
    if resolved_norm_type is None:
        resolved_norm_type = "group" if model_version == "v2" else "batch"
    r_parameterization = str(cli_config.get("r_parameterization", "coarse"))
    if model_version == "v2":
        model = AnalyticalCorrectionNet(
            in_channels=5,
            base_channels=hist.get("base_ch", 16),
            n_levels=hist.get("n_levels", 2),
            coarse_shape=(n_clat, n_clon),
            fine_shape=(n_flat, n_flon),
            Ac_scipy=Ac,
            Ic_scipy=Ic,
            freeze_delta_r=bool(cli_config.get("freeze_delta_r", False)),
            disable_nullspace_proj=bool(
                cli_config.get("disable_nullspace_proj", False)
            ),
            disable_p0_input=bool(cli_config.get("disable_p0_input", False)),
        ).to(device)
    elif model_version == "v1":
        model = PrecipBudgetNet(
            in_channels=4,
            base_channels=hist.get("base_ch", 32),
            n_levels=hist.get("n_levels", 3),
            coarse_shape=(n_clat, n_clon),
            norm_type=resolved_norm_type,
            r_parameterization=r_parameterization,
        ).to(device)
    else:
        print(
            f"  WARNING: unsupported model_version={model_version!r} in this "
            "release package. Skipping."
        )
        return None
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # Prepare data
    W_filled = np.nan_to_num(data["W_obs"], nan=0.0)
    fine_mask = data["fine_mask"]

    _, val_s, test_s = make_splits(n_steps)
    if split == "val":
        eval_steps = val_s
    elif split == "all":
        eval_steps = list(range(n_steps))
    else:
        eval_steps = test_s

    all_metrics = []
    AcIc_lu = splu((Ac @ Ic).tocsc()) if model_version == "v2" else None
    with torch.no_grad():
        for t in eval_steps:
            if t + 1 >= W_filled.shape[0]:
                continue
            w0 = W_filled[t]
            w1 = W_filled[t + 1]
            dw = w1 - w0
            pc_obs = data["P_c_obs"][t]
            pc_lift = np.array(
                Ic @ pc_obs.ravel(), dtype=np.float32
            ).reshape(n_flat, n_flon)

            if model_version == "v2":
                Ac_dw = np.array(Ac @ dw.ravel(), dtype=np.float64)
                rhs = pc_obs.ravel().astype(np.float64) + Ac_dw
                r0 = AcIc_lu.solve(rhs).astype(np.float32).reshape(
                    n_clat, n_clon
                )
                p0 = (
                    np.array(Ic @ r0.ravel(), dtype=np.float32)
                    .reshape(n_flat, n_flon)
                    - dw
                )
                x = np.stack([w0, w1, dw, pc_lift, p0], axis=0)[None]
                x_t = torch.from_numpy(x.astype(np.float32)).to(device)
                r0_t = torch.from_numpy(r0[None].astype(np.float32)).to(device)
                P_hat_t, R_app_t = model(x_t, r0_t)
            else:
                x = np.stack([w0, w1, dw, pc_lift], axis=0)[None]
                x_t = torch.from_numpy(x.astype(np.float32)).to(device)
                P_hat_t, R_app_t = model(x_t)
            P_hat = P_hat_t.squeeze().cpu().numpy()
            if model_version == "v1" and r_parameterization == "fine":
                R_fine = R_app_t.squeeze().cpu().numpy()
                R_hat = np.array(Ac @ R_fine.ravel()).reshape(n_clat, n_clon)
            else:
                R_hat = R_app_t.squeeze().cpu().numpy().reshape(n_clat, n_clon)

            m = compute_all_metrics(
                P_hat, data["P_true"][t],
                R_hat, data["R_app_true"][t],
                Ac, Ic, n_clat, n_clon, n_flat, n_flon)
            m["t"] = int(t)
            all_metrics.append(m)

    return all_metrics


# ---------------------------------------------------------------------------
# Summary and aggregation
# ---------------------------------------------------------------------------
def bootstrap_ci(vals, n_boot=10000, ci=95, seed=42):
    """Bootstrap 95% CI on the median of *vals*.

    Returns (ci_lo, ci_hi), the lower and upper bounds of the
    confidence interval for the population median.
    """
    rng = np.random.RandomState(seed)
    arr = np.asarray(vals, dtype=np.float64)
    n = len(arr)
    if n < 3:
        return float(np.min(arr)), float(np.max(arr))
    boot_medians = np.empty(n_boot)
    for i in range(n_boot):
        boot_medians[i] = np.median(rng.choice(arr, size=n, replace=True))
    alpha = (100 - ci) / 2
    return float(np.percentile(boot_medians, alpha)), \
           float(np.percentile(boot_medians, 100 - alpha))


def aggregate_metrics(metrics_list):
    """Compute median, IQR, and bootstrap 95% CI for each metric across timesteps."""
    if not metrics_list:
        return {}
    keys = [k for k in metrics_list[0].keys() if k != "t"]
    agg = {}
    for k in keys:
        vals = [m[k] for m in metrics_list if not np.isnan(m.get(k, np.nan))]
        if vals:
            ci_lo, ci_hi = bootstrap_ci(vals)
            agg[k] = {
                "median": float(np.median(vals)),
                "mean": float(np.mean(vals)),
                "q25": float(np.percentile(vals, 25)),
                "q75": float(np.percentile(vals, 75)),
                "ci_lo": ci_lo,
                "ci_hi": ci_hi,
                "n": len(vals),
            }
    return agg


def print_comparison_table(results_dict):
    """Print a formatted comparison table."""
    # Header
    cols = ["P_fine_nrmse", "Pc_nrmse", "R_corr",
            "CSI_0.1", "P_spatial_corr", "within_cell_anomaly"]
    col_labels = ["P_nRMSE", "Pc_nRMSE", "R_corr",
                  "CSI@0.1", "P_spCorr", "dP_norm"]

    header = f"  {'Method':30s}"
    for lbl in col_labels:
        header += f"  {lbl:>10s}"
    print(header)
    print("  " + "-" * (30 + 12 * len(col_labels)))

    for method_name, agg in results_dict.items():
        row = f"  {method_name:30s}"
        for col in cols:
            if col in agg:
                row += f"  {agg[col]['median']:10.4f}"
            else:
                row += f"  {'-':>10s}"
        print(row)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    sys.path.insert(0, str(BASE))
    from dataset import make_splits

    parser = argparse.ArgumentParser(
        description="Comprehensive baseline evaluation")
    parser.add_argument("--tile", default="A_wet")
    parser.add_argument("--synthetic-dir", default=None,
                        help="Synthetic dataset directory. Defaults to "
                             "preprocessed/synthetic.")
    parser.add_argument("--n-steps", type=int, default=1224)
    parser.add_argument("--model-dir", default=None,
                        help="Path to trained model results dir")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--split", choices=["test", "val", "all"],
                        default="test",
                        help="Evaluation split for trained model metrics")
    args = parser.parse_args()

    print("=" * 80)
    print("  COMPREHENSIVE BASELINE EVALUATION")
    print("=" * 80)
    print(f"  Tile: {args.tile}, Steps: {args.n_steps}")
    synth_dir = resolve_synthetic_dir(args.synthetic_dir)
    print(f"  Synthetic source: {synth_dir}")

    # Load data
    print("\n  Loading synthetic data...")
    data = load_synthetic(args.tile, args.n_steps, synth_dir)
    n_clat, n_clon = len(data["coarse_lat"]), len(data["coarse_lon"])
    n_flat, n_flon = len(data["fine_lat"]), len(data["fine_lon"])
    print(f"  Coarse: {n_clat}x{n_clon}, Fine: {n_flat}x{n_flon}")

    # Signal characterization
    print(f"\n  Signal characterization:")
    print(f"    mean|P_true|:   {np.mean(np.abs(data['P_true'])):.4f} mm/3h")
    print(f"    std(P_true):    {np.std(data['P_true']):.4f} mm/3h")
    p_nonzero = data["P_true"][data["P_true"] > 0.001]
    print(f"    mean|P_true|>0: {np.mean(p_nonzero):.4f} mm/3h "
          f"({100*len(p_nonzero)/data['P_true'].size:.1f}% of pixels)")
    dW = data["W_true"][1:] - data["W_true"][:-1]
    print(f"    mean|dW_true|:  {np.mean(np.abs(dW)):.4f} mm")
    print(f"    mean|R_app|:    {np.mean(np.abs(data['R_app_true'])):.4f} mm/3h")

    # Build operators
    print("\n  Building operators...")
    Ac, Ic = build_tile_operators(
        data["fine_lat"], data["fine_lon"],
        data["coarse_lat"], data["coarse_lon"])
    AcIc = Ac @ Ic
    print(f"  AcIc diagonal mean: {AcIc.diagonal().mean():.4f}")

    _, val_s, test_s = make_splits(args.n_steps)
    if args.split == "val":
        eval_steps = val_s
    elif args.split == "all":
        eval_steps = list(range(args.n_steps))
    else:
        eval_steps = test_s
    print(f"  Eval split: {args.split} ({len(eval_steps)} steps)")

    results = {}

    # 1. IMERG lift baseline
    print("\n  [1] IMERG lift baseline...")
    imerg_m = imerg_lift_baseline(
        data, Ic, Ac, n_clat, n_clon, n_flat, n_flon, eval_steps)
    results["IMERG lift"] = aggregate_metrics(imerg_m)
    print(f"    P_nRMSE median: {results['IMERG lift']['P_fine_nrmse']['median']:.4f}")

    # 2. Analytical baseline (noiseless)
    print("\n  [2] Analytical baseline (noiseless)...")
    anal_noiseless = analytical_baseline(
        data, Ac, Ic, n_clat, n_clon, n_flat, n_flon,
        eval_steps, noisy=False)
    results["Analytical (noiseless)"] = aggregate_metrics(anal_noiseless)
    print(f"    P_nRMSE median: {results['Analytical (noiseless)']['P_fine_nrmse']['median']:.4f}")

    # 3. Analytical baseline (noisy)
    print("\n  [3] Analytical baseline (noisy sigma_W=1.0)...")
    anal_noisy = analytical_baseline(
        data, Ac, Ic, n_clat, n_clon, n_flat, n_flon,
        eval_steps, noisy=True)
    results["Analytical (noisy)"] = aggregate_metrics(anal_noisy)
    print(f"    P_nRMSE median: {results['Analytical (noisy)']['P_fine_nrmse']['median']:.4f}")

    # 4. Trained model (if provided)
    if args.model_dir:
        print(f"\n  [4] Trained model: {args.model_dir}")
        model_m = trained_model_baseline(
            data, args.model_dir, Ac, Ic,
            n_clat, n_clon, n_flat, n_flon,
            args.n_steps, args.device, args.split)
        if model_m:
            results["Trained model"] = aggregate_metrics(model_m)
            print(f"    P_nRMSE median: {results['Trained model']['P_fine_nrmse']['median']:.4f}")

    # Summary table
    print("\n" + "=" * 80)
    print("  COMPARISON TABLE (medians)")
    print("=" * 80)
    print_comparison_table(results)

    # Save
    out_dir = BASE / "results" / "baselines"
    out_dir.mkdir(parents=True, exist_ok=True)
    if args.model_dir:
        model_tag = Path(args.model_dir).name
    else:
        if synth_dir == DEFAULT_SYNTH_DIR:
            model_tag = f"baselines_n{args.n_steps}"
        else:
            model_tag = f"baselines_{synth_dir.name}_n{args.n_steps}"
    out_path = out_dir / (
        f"baseline_eval_{args.tile}_{args.split}_{model_tag}.json"
    )
    # Convert numpy types for JSON
    def sanitize(obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, dict):
            return {k: sanitize(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [sanitize(v) for v in obj]
        return obj

    train_objective = None
    loss_operator = None
    r_parameterization = None
    best_epoch = None
    best_eval_index = None
    best_metric_name = None
    best_metric_value = None
    best_snapshot = None
    if args.model_dir:
        hist_path = Path(args.model_dir) / "history.json"
        if hist_path.exists():
            hist_obj = json.loads(hist_path.read_text(encoding="utf-8"))
            cli_cfg = hist_obj.get("cli_config", {})
            train_objective = cli_cfg.get("train_objective")
            loss_operator = cli_cfg.get("loss_operator")
            r_parameterization = cli_cfg.get("r_parameterization")
            best_epoch = hist_obj.get("best_epoch")
            best_eval_index = hist_obj.get("best_eval_index")
            best_metric_name = hist_obj.get("best_metric_name")
            best_metric_value = hist_obj.get("best_metric_value")
            best_snapshot = hist_obj.get("best_snapshot")

    payload = {
        "_meta": {
            "tile": args.tile,
            "n_steps": args.n_steps,
            "synthetic_dir": str(synth_dir),
            "split": args.split,
            "eval_step_count": len(eval_steps),
            "pairwise_eval_step_count": max(len(eval_steps) - 1, 0),
            "pairwise_eval_note": (
                "Noisy analytical and trained-model evaluations require "
                "(t, t+1) observations and skip the final split index when "
                "no successor field exists."
            ),
            "trained_model_dir": args.model_dir,
            "train_objective": train_objective,
            "loss_operator": loss_operator,
            "r_parameterization": r_parameterization,
            "best_epoch": best_epoch,
            "best_eval_index": best_eval_index,
            "best_metric_name": best_metric_name,
            "best_metric_value": best_metric_value,
            "best_snapshot": best_snapshot,
            "acic_diag_mean": float(AcIc.diagonal().mean()),
            "analytical_solver": "exact_acic_solve",
        },
        **results,
    }
    with open(out_path, "w") as f:
        json.dump(sanitize(payload), f, indent=2)
    print(f"\n  Results saved: {out_path}")

    # Key findings
    print("\n" + "=" * 80)
    print("  KEY FINDINGS")
    print("=" * 80)

    imerg_p = results["IMERG lift"]["P_fine_nrmse"]["median"]
    anal_noiseless_p = results["Analytical (noiseless)"]["P_fine_nrmse"]["median"]
    print(f"  IMERG lift P_nRMSE:           {imerg_p:.4f}")
    print(f"  Analytical noiseless P_nRMSE: {anal_noiseless_p:.4f}")
    print("    -> Under the exact synthetic twin and exact operators, this should")
    print("       be near zero; any residual reflects only numerical precision.")

    if "IMERG lift" in results and "within_cell_anomaly" in results["IMERG lift"]:
        imerg_da = results["IMERG lift"]["within_cell_anomaly"]["median"]
        print(f"  IMERG lift within-cell anomaly: {imerg_da:.6f}")
        print("    -> Not zero when A_c I_c != I; this is mainly a support-mismatch artifact")

    if "Trained model" in results and "within_cell_anomaly" in results["Trained model"]:
        model_da = results["Trained model"]["within_cell_anomaly"]["median"]
        print(f"  Trained model within-cell anomaly: {model_da:.6f}")
        print("    -> Includes support-mismatch artifact plus any learned fine-scale structure")

    print("\n" + "=" * 80)
    print("  DONE")
    print("=" * 80)


if __name__ == "__main__":
    sys.exit(main() or 0)
