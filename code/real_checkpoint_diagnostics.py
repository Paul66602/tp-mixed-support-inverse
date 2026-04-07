#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Evaluate saved checkpoints from a real-data training run on the validation set.

This is meant for semiconvergence diagnostics on self-supervised runs where
fine-scale precipitation truth is unavailable. It recomputes validation loss
components (`val_loss`, `L_W`, `L_Pc`, `L_R`) for every saved checkpoint in a
results directory and writes a compact JSON summary.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from dataset import BudgetDataset, make_splits, load_real_cache
from losses import BudgetLossConv, BudgetLossExact, BudgetLossPointwise
from model import AnalyticalCorrectionNet, PrecipBudgetNet
from train_real import evaluate_real, get_operators


def _load_history(model_dir: Path) -> dict:
    hist_path = model_dir / "history.json"
    if not hist_path.exists():
        raise FileNotFoundError(f"Missing history.json: {hist_path}")
    return json.loads(hist_path.read_text(encoding="utf-8"))


def _resolved_norm_type(hist: dict) -> str:
    cli = hist.get("cli_config", {})
    norm_type = cli.get("norm_type")
    if norm_type is not None:
        return str(norm_type)
    return "group" if hist.get("model_version") == "v2" else "batch"


def _build_model(hist: dict, ops: dict, device: torch.device) -> torch.nn.Module:
    cli = hist.get("cli_config", {})
    model_version = hist.get("model_version", "v1")
    if model_version == "v2":
        model = AnalyticalCorrectionNet(
            in_channels=5,
            base_channels=int(hist.get("base_ch", 16)),
            n_levels=int(hist.get("n_levels", 3)),
            dropout=float(cli.get("dropout", 0.1)),
            coarse_shape=ops["coarse_shape"],
            fine_shape=ops["fine_shape"],
            Ac_scipy=ops["Ac"],
            Ic_scipy=ops["Ic"],
            freeze_delta_r=bool(cli.get("freeze_delta_r", False)),
            disable_nullspace_proj=bool(cli.get("disable_nullspace_proj", False)),
            disable_p0_input=bool(cli.get("disable_p0_input", False)),
        )
    elif model_version == "v1":
        model = PrecipBudgetNet(
            in_channels=4,
            base_channels=int(hist.get("base_ch", 32)),
            n_levels=int(hist.get("n_levels", 3)),
            dropout=float(cli.get("dropout", 0.1)),
            coarse_shape=ops["coarse_shape"],
            norm_type=_resolved_norm_type(hist),
            r_parameterization=str(cli.get("r_parameterization", "coarse")),
        )
    else:
        raise RuntimeError(
            f"Unsupported model_version={model_version!r} in this release package."
        )
    return model.to(device)


def _build_loss(hist: dict, ops: dict):
    cli = hist.get("cli_config", {})
    loss_operator = str(cli.get("loss_operator", "exact"))
    r_parameterization = str(cli.get("r_parameterization", "coarse"))
    lambda_w = float(cli.get("lambda_W", 1.0))
    lambda_pc = float(cli.get("lambda_Pc", 10.0))
    lambda_r = float(cli.get("lambda_R", 0.01))
    if r_parameterization == "fine":
        return BudgetLossPointwise(
            ops["Ac"],
            fine_shape=ops["fine_shape"],
            coarse_shape=ops["coarse_shape"],
            lambda_W=lambda_w,
            lambda_Pc=lambda_pc,
            lambda_R=lambda_r,
        )
    if loss_operator == "exact":
        return BudgetLossExact(
            ops["Ac"],
            ops["Ic"],
            fine_shape=ops["fine_shape"],
            coarse_shape=ops["coarse_shape"],
            lambda_W=lambda_w,
            lambda_Pc=lambda_pc,
            lambda_R=lambda_r,
        )
    return BudgetLossConv(
        pool_size=5,
        lambda_W=lambda_w,
        lambda_Pc=lambda_pc,
        lambda_R=lambda_r,
    )


def _collect_checkpoints(model_dir: Path, hist: dict) -> list[tuple[str, Path, int | None]]:
    ckpts: list[tuple[str, Path, int | None]] = []
    best_path = model_dir / "best_model.pt"
    if best_path.exists():
        ckpts.append(("best_model", best_path, hist.get("best_epoch")))
    pattern = re.compile(r"checkpoint_epoch_(\d+)\.pt$")
    for p in sorted(model_dir.glob("checkpoint_epoch_*.pt")):
        m = pattern.search(p.name)
        epoch = int(m.group(1)) if m else None
        ckpts.append((p.stem, p, epoch))
    return ckpts


def main() -> int:
    parser = argparse.ArgumentParser(description="Real-data checkpoint diagnostics")
    parser.add_argument("--model-dir", required=True,
                        help="Results directory containing history.json and checkpoints")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--num-workers", type=int, default=0)
    args = parser.parse_args()

    model_dir = Path(args.model_dir)
    hist = _load_history(model_dir)
    if hist.get("mode") != "real":
        raise ValueError("This diagnostic is only intended for real-data runs.")

    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    ops = get_operators("real")
    _, val_steps, _ = make_splits(int(ops["n_t"]))
    cli = hist.get("cli_config", {})
    is_v2 = hist.get("model_version") == "v2"
    real_cache = None
    if bool(cli.get("cache_real", False)):
        real_cache = load_real_cache(
            ops["Ic"],
            precompute_lifted=bool(cli.get("cache_lifted_pc", False)),
        )

    val_ds = BudgetDataset(
        "full",
        val_steps,
        ops["Ic"],
        Ac=ops["Ac"] if is_v2 else None,
        patch_size=None,
        augment=False,
        synthetic=False,
        real_cache=real_cache,
        v2=is_v2,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=1,
        shuffle=False,
        num_workers=max(0, int(args.num_workers)),
        pin_memory=(device.type == "cuda"),
        persistent_workers=(int(args.num_workers) > 0),
    )

    model = _build_model(hist, ops, device)
    loss_fn = _build_loss(hist, ops)
    ckpts = _collect_checkpoints(model_dir, hist)
    rows = []

    print("=" * 92)
    print(f"  Real Checkpoint Diagnostics: {model_dir.name}")
    print("=" * 92)
    print(f"  {'Checkpoint':28s}  {'Epoch':>5s}  {'Val':>9s}  {'L_W':>9s}  {'L_Pc':>9s}  {'L_R':>9s}")
    print("  " + "-" * 78)

    for label, ckpt_path, epoch in ckpts:
        state = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(state)
        model.eval()
        metrics = evaluate_real(model, val_loader, loss_fn, device, is_v2=is_v2)
        row = {
            "checkpoint": label,
            "path": str(ckpt_path),
            "epoch": epoch,
            "val_loss": float(metrics["loss"]),
            "val_L_W": float(metrics["L_W"]),
            "val_L_Pc": float(metrics["L_Pc"]),
            "val_L_R": float(metrics["L_R"]),
        }
        rows.append(row)
        ep = "-" if epoch is None else str(epoch)
        print(
            f"  {label:28s}  {ep:>5s}  "
            f"{row['val_loss']:9.4f}  {row['val_L_W']:9.4f}  "
            f"{row['val_L_Pc']:9.4f}  {row['val_L_R']:9.4f}"
        )

    out_path = model_dir / "real_checkpoint_diagnostics.json"
    payload = {
        "model_dir": str(model_dir),
        "mode": hist.get("mode"),
        "model_version": hist.get("model_version"),
        "checkpoint_metric": hist.get("best_metric_name"),
        "best_epoch": hist.get("best_epoch"),
        "best_snapshot": hist.get("best_snapshot"),
        "rows": rows,
    }
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print("=" * 92)
    print(f"Saved: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
