#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Unified training script for PrecipBudgetNet.

Supports both synthetic twin (with ground truth metrics) and real data
(self-supervised, budget loss only). Uses BudgetLossConv for GPU-friendly
patch-compatible training.

Usage:
    # Synthetic twin (full tile, with ground truth evaluation)
    python train_real.py --mode synthetic --tile A_wet --epochs 50 --n-steps 1224 \
        --lambda-W 10.0 --lambda-Pc 5.0 --lambda-R 0.0 \
        --checkpoint-metric val_loss

    # Synthetic twin from an alternate directory
    python train_real.py --mode synthetic --tile A_wet \
        --synthetic-dir preprocessed/synthetic_misspecified/A_wet_fine_nullspace_eps0.20_beta0.30_seed42 \
        --epochs 50 --n-steps 1224 --lambda-W 10.0 --lambda-Pc 5.0 --lambda-R 0.0

    # Real data (self-supervised, patch-based)
    python train_real.py --mode real --epochs 50 --patch-size 64 --batch-size 4

    # Real data with GPU
    python train_real.py --mode real --epochs 50 --device cuda
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

CODE_DIR = Path(__file__).resolve().parent
REPO_ROOT = CODE_DIR.parent
RESULTS_DIR = REPO_ROOT / "results"
sys.path.insert(0, str(CODE_DIR))

from model import PrecipBudgetNet, AnalyticalCorrectionNet, count_parameters
from dataset import (
    BudgetDataset,
    make_splits,
    load_real_cache,
    resolve_synthetic_dir,
)
from losses import BudgetLossConv, BudgetLossExact, BudgetLossPointwise
from reproducibility import set_global_seed, worker_init_fn, log_seed_to_history


def _masked_mse(pred: torch.Tensor, target: torch.Tensor,
                mask: torch.Tensor | None) -> torch.Tensor:
    """Masked mean-squared error on the fine grid."""
    if pred.dim() == 4:
        pred = pred.squeeze(1)
    if target.dim() == 4:
        target = target.squeeze(1)
    sq = (pred - target) ** 2
    if mask is None:
        return sq.mean()
    mask_f = mask.float()
    denom = mask_f.sum().clamp(min=1.0)
    return (sq * mask_f).sum() / denom


def get_operators(
    mode: str,
    tile: str = "A_wet",
    synthetic_dir: str | Path | None = None,
):
    """Load support operators and grid shapes."""
    from scipy import sparse
    if mode == "synthetic":
        from netCDF4 import Dataset
        from build_support_operators import build_Ac, build_Ic
        synth_dir = resolve_synthetic_dir(synthetic_dir)
        with Dataset(synth_dir / f"synthetic_truth_{tile}.nc", "r") as ds:
            flat = np.array(ds.variables["fine_lat"][:])
            flon = np.array(ds.variables["fine_lon"][:])
            clat = np.array(ds.variables["coarse_lat"][:])
            clon = np.array(ds.variables["coarse_lon"][:])
            n_t = ds.dimensions["time"].size
        Ac = build_Ac(flat, flon, clat, clon)
        Ic = build_Ic(flat, flon, clat, clon)
        return {
            "Ac": Ac,
            "Ic": Ic,
            "n_t": n_t,
            "fine_shape": (len(flat), len(flon)),
            "coarse_shape": (len(clat), len(clon)),
        }
    else:
        ops = np.load(REPO_ROOT / "preprocessed" / "support_operators.npz",
                      allow_pickle=True)
        Ac = sparse.csr_matrix(
            (ops["Ac_data"], ops["Ac_indices"], ops["Ac_indptr"]),
            shape=tuple(ops["Ac_shape"]))
        Ic = sparse.csr_matrix(
            (ops["Ic_data"], ops["Ic_indices"], ops["Ic_indptr"]),
            shape=tuple(ops["Ic_shape"]))
        from netCDF4 import Dataset
        with Dataset(REPO_ROOT / "preprocessed" / "common_domain.nc", "r") as ds:
            fine_shape = (
                ds.dimensions["fine_lat"].size,
                ds.dimensions["fine_lon"].size,
            )
            coarse_shape = (
                ds.dimensions["coarse_lat"].size,
                ds.dimensions["coarse_lon"].size,
            )
        return {
            "Ac": Ac,
            "Ic": Ic,
            "n_t": 1224,
            "fine_shape": fine_shape,
            "coarse_shape": coarse_shape,
        }


def load_truth(
    tile: str,
    n_steps: int,
    synthetic_dir: str | Path | None = None,
):
    """Load ground truth for synthetic evaluation."""
    from netCDF4 import Dataset
    synth_dir = resolve_synthetic_dir(synthetic_dir)
    with Dataset(synth_dir / f"synthetic_truth_{tile}.nc", "r") as ds:
        P_true = np.array(ds.variables["P_true"][:n_steps])
        R_true = np.array(ds.variables["R_app_true"][:n_steps])
    return P_true, R_true


def evaluate_synthetic(model, dataloader, loss_fn, device, P_true, R_true,
                       coarse_shape, is_v2=False,
                       train_objective: str = "constrained",
                       Ac=None,
                       r_parameterization: str = "coarse"):
    """Evaluate on synthetic data with ground truth metrics."""
    model.eval()
    losses, p_nrmses, r_corrs = [], [], []
    n_clat, n_clon = coarse_shape

    with torch.no_grad():
        for batch in dataloader:
            x = batch["x"].to(device)
            dw = batch["dW_obs"].to(device)
            p_true_batch = (
                batch["P_true"].to(device)
                if train_objective == "supervised_p"
                else None
            )
            pc = batch["P_c_obs"].to(device)
            mask = batch["fine_mask"].to(device)
            coarse_mask = batch["coarse_mask"].to(device)
            t_idx = batch["t"].item()

            if is_v2 and "R_0" in batch:
                R_0 = batch["R_0"].to(device)
                P_hat, R_app = model(x, R_0)
            else:
                P_hat, R_app = model(x)
            if train_objective == "supervised_p":
                loss = _masked_mse(P_hat, p_true_batch, mask)
            else:
                loss, _ = loss_fn(P_hat, R_app, dw, pc, mask, coarse_mask)
            losses.append(float(loss.item()))

            P_np = P_hat.squeeze().cpu().numpy()
            if r_parameterization == "fine":
                R_fine = R_app.squeeze().cpu().numpy()
                R_np = np.array(Ac @ R_fine.ravel()).reshape(n_clat, n_clon)
            else:
                R_np = R_app.squeeze().cpu().numpy().reshape(n_clat, n_clon)

            if t_idx < len(P_true):
                P_t = P_true[t_idx]
                R_t = R_true[t_idx]
                pm = max(np.mean(np.abs(P_t)), 1e-6)
                p_nrmses.append(np.sqrt(np.mean((P_np - P_t) ** 2)) / pm)
                rc = np.corrcoef(R_np.ravel(), R_t.ravel())[0, 1]
                r_corrs.append(rc if not np.isnan(rc) else 0.0)

    out = {
        "loss": float(np.mean(losses)),
        "P_nrmse": float(np.mean(p_nrmses)) if p_nrmses else float("nan"),
        "R_corr": float(np.mean(r_corrs)) if r_corrs else float("nan"),
    }
    return out


def evaluate_real(model, dataloader, loss_fn, device, is_v2=False):
    """Evaluate on real data (no ground truth — monitor loss components)."""
    model.eval()
    all_lw, all_lpc, all_lr, all_total = [], [], [], []

    with torch.no_grad():
        for batch in dataloader:
            x = batch["x"].to(device)
            dw = batch["dW_obs"].to(device)
            pc = batch["P_c_obs"].to(device)
            mask = batch["fine_mask"].to(device)
            coarse_mask = batch["coarse_mask"].to(device)

            if is_v2 and "R_0" in batch:
                R_0 = batch["R_0"].to(device)
                P_hat, R_app = model(x, R_0)
            else:
                P_hat, R_app = model(x)
            _, comps = loss_fn(P_hat, R_app, dw, pc, mask, coarse_mask)
            all_lw.append(comps["L_W"])
            all_lpc.append(comps["L_Pc"])
            all_lr.append(comps["L_R"])
            all_total.append(comps["total"])

    return {
        "loss": float(np.mean(all_total)),
        "L_W": float(np.mean(all_lw)),
        "L_Pc": float(np.mean(all_lpc)),
        "L_R": float(np.mean(all_lr)),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Unified PrecipBudgetNet training")
    parser.add_argument("--mode", choices=["synthetic", "real"],
                        default="synthetic")
    parser.add_argument("--tile", default="A_wet",
                        help="Tile name for synthetic mode")
    parser.add_argument("--synthetic-dir", default=None,
                        help="Synthetic dataset directory. Defaults to "
                             "preprocessed/synthetic. Useful for "
                             "misspecified-twin stress tests.")
    parser.add_argument("--n-steps", type=int, default=200,
                        help="Timesteps for synthetic (default: 200)")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--batch-size", type=int, default=1,
                        help="Batch size (1 for full-tile, >1 for patches)")
    parser.add_argument("--patch-size", type=int, default=None,
                        help="Spatial patch size (None for full tile)")
    parser.add_argument("--base-ch", type=int, default=32)
    parser.add_argument("--n-levels", type=int, default=3)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--loss-operator", choices=["exact", "naive"],
                        default="exact",
                        help="Operator family used inside the training loss. "
                             "'exact' uses sparse A_c/I_c support operators. "
                             "'naive' uses avg-pooling + nearest "
                             "interpolation on the full tile or patches. "
                             "Useful for the support-operator ablation.")
    parser.add_argument("--r-parameterization", choices=["coarse", "fine"],
                        default="coarse",
                        help="Residual support used by v1. 'coarse' is the "
                             "main formulation. 'fine' is the A6 pointwise-"
                             "budget ablation.")
    parser.add_argument("--train-objective",
                        choices=["constrained", "supervised_p"],
                        default="constrained",
                        help="'constrained' is the main inverse-learning "
                             "objective. 'supervised_p' is a synthetic-only "
                             "unconstrained baseline that regresses P_true "
                             "directly.")
    parser.add_argument("--lambda-W", type=float, default=1.0)
    parser.add_argument("--lambda-Pc", type=float, default=10.0)
    parser.add_argument("--lambda-R", type=float, default=0.01)
    parser.add_argument("--checkpoint-metric",
                        choices=[
                            "val_loss",
                            "val_P_nrmse",
                        ],
                        default="val_loss",
                        help="Metric for checkpoint selection and early stopping. "
                             "'val_loss' uses the weighted budget loss (default). "
                             "'val_P_nrmse' uses fine-grid P accuracy directly "
                             "(synthetic only, addresses surrogate alignment).")
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--eval-every", type=int, default=5)
    parser.add_argument("--save-eval-checkpoints", action="store_true",
                        default=False,
                        help="Save a checkpoint at every validation event. "
                             "Useful when semiconvergence or checkpoint "
                             "selection is scientifically important.")
    parser.add_argument("--device", default="cpu",
                        help="Device (cpu or cuda)")
    parser.add_argument("--cache-real", action="store_true", default=False,
                        help="Load all real data into RAM (~12 GB). "
                             "Eliminates I/O bottleneck, enables num_workers>0.")
    parser.add_argument("--cache-lifted-pc", action="store_true", default=False,
                        help="When used with --cache-real, also precompute "
                             "lifted coarse precipitation on the fine grid "
                             "(adds ~6 GB RAM, removes per-step sparse matvec).")
    parser.add_argument("--num-workers", type=int, default=None,
                        help="Override DataLoader num_workers "
                             "(default: auto-detect)")
    parser.add_argument("--model-version", choices=["v1", "v2"],
                        default="v1",
                        help="v1: PrecipBudgetNet (4ch, BN). "
                             "v2: AnalyticalCorrectionNet (5ch, GN, "
                             "analytical backbone + learned correction).")
    parser.add_argument("--norm-type", choices=["batch", "group", "none"],
                        default=None,
                        help="Normalization type for v1. Default: batch for "
                             "v1, group for v2. Overrides the default.")
    parser.add_argument("--freeze-delta-r", action="store_true", default=False,
                        help="(v2 only) Zero out delta_R correction, forcing "
                             "all learning through delta_P. Tests H2.")
    parser.add_argument("--disable-nullspace-proj", action="store_true",
                        default=False,
                        help="(v2 only) Skip null-space projection on "
                             "delta_P. Tests H3.")
    parser.add_argument("--disable-p0-input", action="store_true",
                        default=False,
                        help="(v2 only) Remove P_0 from encoder input "
                             "channels (still used in output assembly). "
                             "Tests H4.")
    # Reproducibility
    parser.add_argument("--seed", type=int, default=None,
                        help="Random seed for reproducibility. If not provided, "
                             "a seed is auto-generated from system entropy and "
                             "logged to history.json for audit.")
    parser.add_argument("--deterministic", action="store_true", default=False,
                        help="Enable strict deterministic mode (slower but fully "
                             "reproducible). Sets cudnn.deterministic=True and "
                             "cudnn.benchmark=False.")
    parser.add_argument("--run-name", default=None,
                        help="Custom run name for output files")
    args = parser.parse_args()

    # =========================================================================
    # SEED CONTROL - Set seeds BEFORE any other torch/numpy operations
    # =========================================================================
    seed_info = set_global_seed(args.seed)
    deterministic_info = None
    if args.deterministic:
        from reproducibility import set_deterministic_mode
        deterministic_info = set_deterministic_mode(True)
    print(f"  Seed: {seed_info['seed']} (source: {seed_info['seed_source']})")
    if args.deterministic:
        print(f"  Deterministic mode: enabled")

    device = torch.device(args.device)
    is_synthetic = (args.mode == "synthetic")
    run_name = args.run_name or f"{args.mode}_{args.tile if is_synthetic else 'full'}"
    out_dir = RESULTS_DIR / run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    if device.type == "cuda" and not args.deterministic:
        torch.backends.cudnn.benchmark = True

    if args.train_objective == "supervised_p":
        if not is_synthetic:
            print("ERROR: --train-objective supervised_p is synthetic-only.")
            return
        if args.model_version != "v1":
            print("ERROR: --train-objective supervised_p currently supports "
                  "only --model-version v1.")
            return
    if args.r_parameterization == "fine":
        if args.model_version != "v1":
            print("ERROR: --r-parameterization fine currently supports only "
                  "--model-version v1.")
            return
        if args.patch_size is not None:
            print("ERROR: --r-parameterization fine currently requires "
                  "full-tile mode.")
            return
        if args.loss_operator != "exact":
            print("ERROR: --r-parameterization fine currently requires "
                  "--loss-operator exact so A6 isolates pointwise budget "
                  "rather than naive support operators.")
            return

    print("=" * 70)
    print(f"  PrecipBudgetNet Training - {args.mode.upper()}")
    print("=" * 70)
    print(f"  Device: {device}")
    synthetic_dir = (
        resolve_synthetic_dir(args.synthetic_dir) if is_synthetic else None
    )
    if synthetic_dir is not None:
        print(f"  Synthetic source: {synthetic_dir}")

    # Operators and data info
    ops = get_operators(args.mode, args.tile, synthetic_dir)
    n_steps = min(args.n_steps, ops["n_t"]) if is_synthetic else ops["n_t"]
    n_clat, n_clon = ops["coarse_shape"]

    # Splits
    train_s, val_s, test_s = make_splits(n_steps)
    print(f"  Steps: {n_steps}, Splits: train={len(train_s)}, "
          f"val={len(val_s)}, test={len(test_s)}")

    # Real-data RAM cache (load once, share between train/val)
    real_cache = None
    if not is_synthetic and args.cache_real:
        real_cache = load_real_cache(
            ops["Ic"], precompute_lifted=args.cache_lifted_pc)

    # Datasets
    is_v2 = (args.model_version == "v2")
    Ac_for_ds = ops["Ac"] if is_v2 else None
    print(f"  Building datasets... (model={args.model_version})")
    train_ds = BudgetDataset(
        args.tile if is_synthetic else "full", train_s, ops["Ic"],
        Ac=Ac_for_ds,
        patch_size=args.patch_size, augment=True, synthetic=is_synthetic,
        synthetic_dir=synthetic_dir,
        real_cache=real_cache, v2=is_v2,
    )
    val_ds = BudgetDataset(
        args.tile if is_synthetic else "full", val_s, ops["Ic"],
        Ac=Ac_for_ds,
        patch_size=None, augment=False, synthetic=is_synthetic,
        synthetic_dir=synthetic_dir,
        real_cache=real_cache, v2=is_v2,
    )

    # DataLoader: use workers + pin_memory when data is cached (no NC handles)
    use_workers = (is_synthetic or real_cache is not None)
    if args.num_workers is not None:
        nw_train = args.num_workers
        nw_val = min(1, args.num_workers)
    else:
        nw_train = min(4, (os.cpu_count() or 1)) if use_workers else 0
        nw_val = 1 if use_workers else 0
    pin = (device.type == "cuda")

    # Create generator for reproducible shuffling
    g = torch.Generator()
    g.manual_seed(seed_info["seed"])

    train_loader_kwargs = {
        "batch_size": args.batch_size,
        "shuffle": True,
        "num_workers": nw_train,
        "pin_memory": pin,
        "persistent_workers": (nw_train > 0),
        "generator": g,
        "worker_init_fn": worker_init_fn,
    }
    val_loader_kwargs = {
        "batch_size": 1,
        "shuffle": False,
        "num_workers": nw_val,
        "pin_memory": pin,
        "persistent_workers": (nw_val > 0),
        "worker_init_fn": worker_init_fn,
    }
    if nw_train > 0:
        train_loader_kwargs["prefetch_factor"] = 2
    if nw_val > 0:
        val_loader_kwargs["prefetch_factor"] = 2

    train_loader = DataLoader(train_ds, **train_loader_kwargs)
    val_loader = DataLoader(val_ds, **val_loader_kwargs)
    print(f"  DataLoader: train_workers={nw_train}, val_workers={nw_val}, "
          f"pin_memory={pin}, cached={'yes' if real_cache else 'no'}, "
          f"lifted_pc_cached={'yes' if (real_cache and real_cache.get('Pc_lifted') is not None) else 'no'}")

    # Ground truth for synthetic evaluation
    P_true_all, R_true_all = None, None
    if is_synthetic:
        P_true_all, R_true_all = load_truth(args.tile, n_steps, synthetic_dir)

    # Model
    if args.train_objective == "supervised_p":
        print("  Objective: supervised_p (synthetic unconstrained baseline)")
    model_coarse = ops["coarse_shape"] if args.patch_size is None else None
    # Resolve norm_type: default is 'batch' for v1, 'group' for v2
    norm_type = args.norm_type
    if norm_type is None:
        norm_type = "group" if is_v2 else "batch"
    if is_v2:
        model_fine = ops["fine_shape"] if args.patch_size is None else None
        # Pass exact sparse operators for null-space projection
        # (only in full-tile mode; patch mode uses approximate projection)
        Ac_for_model = ops["Ac"] if args.patch_size is None else None
        Ic_for_model = ops["Ic"] if args.patch_size is None else None
        model = AnalyticalCorrectionNet(
            in_channels=5, base_channels=args.base_ch,
            n_levels=args.n_levels, dropout=args.dropout,
            coarse_shape=model_coarse,
            fine_shape=model_fine,
            Ac_scipy=Ac_for_model,
            Ic_scipy=Ic_for_model,
            freeze_delta_r=args.freeze_delta_r,
            disable_nullspace_proj=args.disable_nullspace_proj,
            disable_p0_input=args.disable_p0_input,
        ).to(device)
    else:
        model = PrecipBudgetNet(
            in_channels=4, base_channels=args.base_ch,
            n_levels=args.n_levels, dropout=args.dropout,
            coarse_shape=model_coarse,
            norm_type=norm_type,
            r_parameterization=args.r_parameterization,
        ).to(device)
    n_params = count_parameters(model)
    print(f"  Model: {args.model_version} - {n_params:,} params, "
          f"base_ch={args.base_ch}, levels={args.n_levels}, "
          f"dropout={args.dropout}, norm={norm_type}")
    if args.model_version == "v1":
        print(f"  Residual parameterization: {args.r_parameterization}")
    if is_v2:
        ablation_flags = []
        if args.freeze_delta_r:
            ablation_flags.append("freeze_delta_r")
        if args.disable_nullspace_proj:
            ablation_flags.append("disable_nullspace_proj")
        if args.disable_p0_input:
            ablation_flags.append("disable_p0_input")
        if ablation_flags:
            print(f"  Ablation flags: {', '.join(ablation_flags)}")
    if args.model_version == "v1" and args.r_parameterization == "fine":
        print("  Residual output: fine grid (A6 pointwise-budget ablation)")
    else:
        print(f"  Coarse output: {model_coarse or 'bottleneck (patch mode)'}")

    # Loss
    use_exact_loss = (args.loss_operator == "exact" and args.patch_size is None)
    if args.train_objective == "supervised_p":
        loss_fn = None
        print("  Loss: supervised fine-grid P regression "
              "(constraint-free synthetic baseline)")
    elif args.r_parameterization == "fine":
        loss_fn = BudgetLossPointwise(
            ops["Ac"],
            fine_shape=ops["fine_shape"],
            coarse_shape=ops["coarse_shape"],
            lambda_W=args.lambda_W,
            lambda_Pc=args.lambda_Pc,
            lambda_R=args.lambda_R,
        )
        print("  Loss: exact pointwise budget + exact coarse Pc term "
              "(A6 pointwise-budget ablation)")
    elif use_exact_loss:
        loss_fn = BudgetLossExact(
            ops["Ac"],
            ops["Ic"],
            fine_shape=ops["fine_shape"],
            coarse_shape=ops["coarse_shape"],
            lambda_W=args.lambda_W,
            lambda_Pc=args.lambda_Pc,
            lambda_R=args.lambda_R,
        )
        print("  Loss: exact sparse operators (full-domain mode)")
    else:
        loss_fn = BudgetLossConv(
            pool_size=5,
            lambda_W=args.lambda_W,
            lambda_Pc=args.lambda_Pc,
            lambda_R=args.lambda_R,
        )
        if args.patch_size is None:
            print("  Loss: naive pooling/interpolation operators "
                  "(full-domain ablation mode)")
        else:
            print("  Loss: approximate pooling/interpolation operators "
                  "(patch mode)")

    # Optimizer
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr,
                             weight_decay=args.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)

    # Training loop
    history = {"train_loss": [], "val_loss": [], "lr": []}
    if is_synthetic:
        history["val_P_nrmse"] = []
        history["val_R_corr"] = []
    else:
        history["val_L_W"] = []
        history["val_L_Pc"] = []

    best_val = float("inf")
    best_epoch = None
    best_eval_index = None
    best_snapshot = {}
    patience_ctr = 0
    ckpt_metric = args.checkpoint_metric
    if ckpt_metric == "val_P_nrmse" and not is_synthetic:
        print("  WARNING: --checkpoint-metric val_P_nrmse requires synthetic "
              "mode (ground truth). Falling back to val_loss.")
        ckpt_metric = "val_loss"
    if ckpt_metric == "val_P_nrmse":
        print(f"  Checkpoint metric: val_P_nrmse (direct P accuracy)")
    else:
        print(f"  Checkpoint metric: val_loss (weighted budget loss)")

    if is_synthetic:
        header = (f"  {'Ep':>4s}  {'Train':>8s}  {'Val':>8s}  "
                  f"{'P_nRMSE':>8s}  {'R_corr':>8s}  {'LR':>10s}  {'T':>5s}")
    else:
        header = (f"  {'Ep':>4s}  {'Train':>8s}  {'Val':>8s}  "
                  f"{'L_W':>8s}  {'L_Pc':>8s}  {'LR':>10s}  {'T':>5s}")
    print(f"\n{header}")
    print("  " + "-" * (len(header) - 2))

    for epoch in range(args.epochs):
        t0 = time.time()
        model.train()
        epoch_loss = 0.0
        n_batch = 0

        for batch in train_loader:
            x = batch["x"].to(device, non_blocking=pin)
            dw = batch["dW_obs"].to(device, non_blocking=pin)
            pc = batch["P_c_obs"].to(device, non_blocking=pin)
            mask = batch["fine_mask"].to(device, non_blocking=pin)
            coarse_mask = batch["coarse_mask"].to(device, non_blocking=pin)
            p_true_batch = None
            if args.train_objective == "supervised_p":
                p_true_batch = batch["P_true"].to(device, non_blocking=pin)

            if is_v2 and "R_0" in batch:
                R_0 = batch["R_0"].to(device, non_blocking=pin)
                P_hat, R_app = model(x, R_0)
            else:
                P_hat, R_app = model(x)
            if args.train_objective == "supervised_p":
                loss = _masked_mse(P_hat, p_true_batch, mask)
            else:
                loss, _ = loss_fn(P_hat, R_app, dw, pc, mask, coarse_mask)

            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

            epoch_loss += loss.item()
            n_batch += 1

        sched.step()
        avg_train = epoch_loss / max(n_batch, 1)
        history["train_loss"].append(avg_train)
        history["lr"].append(sched.get_last_lr()[0])
        dt = time.time() - t0

        # Validation
        do_eval = ((epoch + 1) % args.eval_every == 0
                   or epoch == 0 or epoch == args.epochs - 1)
        if do_eval:
            if is_synthetic:
                val_m = evaluate_synthetic(
                    model, val_loader, loss_fn, device,
                    P_true_all, R_true_all, ops["coarse_shape"],
                    is_v2=is_v2,
                    train_objective=args.train_objective,
                    Ac=ops["Ac"],
                    r_parameterization=args.r_parameterization)
                history["val_loss"].append(val_m["loss"])
                history["val_P_nrmse"].append(val_m["P_nrmse"])
                history["val_R_corr"].append(val_m["R_corr"])
                print(f"  {epoch+1:4d}  {avg_train:8.3f}  "
                      f"{val_m['loss']:8.3f}  "
                      f"{val_m['P_nrmse']:8.3f}  "
                      f"{val_m['R_corr']:8.3f}  "
                      f"{sched.get_last_lr()[0]:10.2e}  {dt:4.0f}s")
            else:
                val_m = evaluate_real(
                    model, val_loader, loss_fn, device, is_v2=is_v2)
                history["val_loss"].append(val_m["loss"])
                history["val_L_W"].append(val_m["L_W"])
                history["val_L_Pc"].append(val_m["L_Pc"])
                print(f"  {epoch+1:4d}  {avg_train:8.3f}  "
                      f"{val_m['loss']:8.3f}  "
                      f"{val_m['L_W']:8.3f}  "
                      f"{val_m['L_Pc']:8.3f}  "
                      f"{sched.get_last_lr()[0]:10.2e}  {dt:4.0f}s")

            # Early stopping
            if ckpt_metric == "val_P_nrmse" and is_synthetic:
                ckpt_val = val_m["P_nrmse"]
            else:
                ckpt_val = val_m["loss"]
            if ckpt_val < best_val:
                best_val = ckpt_val
                best_epoch = epoch + 1
                best_eval_index = len(history["val_loss"]) - 1
                best_snapshot = {
                    "val_loss": float(val_m["loss"]),
                }
                if is_synthetic:
                    best_snapshot["val_P_nrmse"] = float(val_m["P_nrmse"])
                    best_snapshot["val_R_corr"] = float(val_m["R_corr"])
                else:
                    best_snapshot["val_L_W"] = float(val_m["L_W"])
                    best_snapshot["val_L_Pc"] = float(val_m["L_Pc"])
                patience_ctr = 0
                torch.save(model.state_dict(), out_dir / "best_model.pt")
            else:
                patience_ctr += args.eval_every
                if patience_ctr >= args.patience:
                    print(f"\n  Early stop at epoch {epoch+1}")
                    break
            if args.save_eval_checkpoints:
                torch.save(
                    model.state_dict(),
                    out_dir / f"checkpoint_epoch_{epoch+1:03d}.pt"
                )

    # Save
    hist_path = out_dir / "history.json"
    # Log full CLI configuration for reproducibility
    cli_config = {k: v for k, v in vars(args).items()
                  if not k.startswith("_")}
    # Persist the resolved normalization mode, not the raw CLI default (None),
    # so evaluation can reconstruct the trained model faithfully.
    cli_config["norm_type"] = norm_type
    cli_config["checkpoint_metric_resolved"] = ckpt_metric
    cli_config["tile"] = args.tile if is_synthetic else "full"
    cli_config["n_steps"] = n_steps

    # Build history dict with seed information for audit trail
    history_out = {
        "mode": args.mode, "tile": args.tile if is_synthetic else "full",
        "model_version": args.model_version,
        "n_steps": n_steps, "n_params": n_params,
        "best_epoch": best_epoch,
        "best_eval_index": best_eval_index,
        "best_metric_name": ckpt_metric,
        "best_metric_value": float(best_val),
        "best_snapshot": best_snapshot,
        "base_ch": args.base_ch, "n_levels": args.n_levels,
        "cli_config": cli_config,
        **{k: [float(v) for v in vs] for k, vs in history.items()},
    }
    # Add seed info for reproducibility audit
    log_seed_to_history(history_out, seed_info, deterministic_info)

    with open(hist_path, "w") as f:
        json.dump(history_out, f, indent=2)

    print(f"\n{'=' * 70}")
    print(f"  DONE - best {ckpt_metric}: {best_val:.4f} (seed={seed_info['seed']})")
    print(f"  Model: {out_dir / 'best_model.pt'}")
    print(f"  History: {hist_path}")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
