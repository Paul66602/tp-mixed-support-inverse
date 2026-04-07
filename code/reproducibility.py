#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Reproducibility utilities for auditable seed control.

This module provides a centralized interface for setting random seeds
across all sources of randomness (Python, NumPy, PyTorch CPU/CUDA).
It also logs seed information to JSON for audit trails.

Usage:
    from reproducibility import set_global_seed, get_seed_info

    # At the start of any training script:
    seed_info = set_global_seed(42)  # Returns dict with seed info

    # Or auto-generate a seed if none provided:
    seed_info = set_global_seed(None)  # Uses entropy-based seed

    # Get current seed state for logging:
    info = get_seed_info()
"""

from __future__ import annotations

import hashlib
import os
import random
import sys
import time
from datetime import datetime, timezone
from typing import Any


def set_global_seed(seed: int | None = None) -> dict[str, Any]:
    """
    Set random seeds for Python, NumPy, and PyTorch.

    Parameters
    ----------
    seed : int or None
        If int, use this seed. If None, generate a seed from system entropy
        combined with timestamp and PID for uniqueness.

    Returns
    -------
    dict
        Seed information for audit logging:
        - seed: the actual seed used
        - seed_source: 'user' or 'auto'
        - timestamp: ISO 8601 timestamp when seed was set
        - torch_cuda_available: whether CUDA was available
        - torch_cudnn_deterministic: cudnn.deterministic setting
        - torch_cudnn_benchmark: cudnn.benchmark setting
    """
    import numpy as np
    import torch

    # Determine seed
    if seed is None:
        # Generate reproducible-by-inspection seed from entropy sources
        entropy_bytes = os.urandom(4)
        entropy_int = int.from_bytes(entropy_bytes, "big")
        # Mix with timestamp and PID for additional uniqueness
        timestamp_ns = time.time_ns()
        pid = os.getpid()
        # Combine deterministically
        combined = f"{entropy_int}:{timestamp_ns}:{pid}"
        hash_digest = hashlib.sha256(combined.encode()).digest()[:4]
        seed = int.from_bytes(hash_digest, "big")
        seed_source = "auto"
    else:
        seed = int(seed)
        seed_source = "user"

    # Ensure seed is in valid range for PyTorch (uint32)
    seed = seed % (2**32)

    # Set Python random
    random.seed(seed)

    # Set NumPy
    np.random.seed(seed)

    # Set PyTorch CPU
    torch.manual_seed(seed)

    # Set PyTorch CUDA if available
    cuda_available = torch.cuda.is_available()
    if cuda_available:
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    # Configure cuDNN for reproducibility (trades some speed for determinism)
    # Note: Full determinism also requires CUBLAS_WORKSPACE_CONFIG=:4096:8
    # on some CUDA versions, but we don't force that here.
    cudnn_deterministic = False
    cudnn_benchmark = True  # Default - fast but non-deterministic
    if hasattr(torch.backends, "cudnn"):
        # For strict reproducibility, users can set these after calling
        # set_global_seed() if needed:
        # torch.backends.cudnn.deterministic = True
        # torch.backends.cudnn.benchmark = False
        cudnn_deterministic = torch.backends.cudnn.deterministic
        cudnn_benchmark = torch.backends.cudnn.benchmark

    # Build info dict
    timestamp = datetime.now(timezone.utc).isoformat()
    info = {
        "seed": seed,
        "seed_source": seed_source,
        "timestamp": timestamp,
        "python_version": sys.version.split()[0],
        "numpy_version": np.__version__,
        "torch_version": torch.__version__,
        "torch_cuda_available": cuda_available,
        "torch_cuda_version": torch.version.cuda if cuda_available else None,
        "torch_cudnn_deterministic": cudnn_deterministic,
        "torch_cudnn_benchmark": cudnn_benchmark,
    }

    return info


def set_deterministic_mode(enabled: bool = True) -> dict[str, Any]:
    """
    Enable or disable strict deterministic mode for PyTorch.

    This is more aggressive than just setting seeds - it also configures
    cuDNN and PyTorch's internal algorithms for full reproducibility.

    WARNING: This can significantly reduce performance (up to 2-3x slower
    on some workloads). Only use when exact reproducibility is critical.

    Parameters
    ----------
    enabled : bool
        If True, enable strict deterministic mode. If False, revert to
        non-deterministic (faster) mode.

    Returns
    -------
    dict
        Configuration state after change.
    """
    import torch

    info = {}

    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = enabled
        torch.backends.cudnn.benchmark = not enabled
        info["cudnn_deterministic"] = torch.backends.cudnn.deterministic
        info["cudnn_benchmark"] = torch.backends.cudnn.benchmark

    # PyTorch >= 1.8 has use_deterministic_algorithms
    if hasattr(torch, "use_deterministic_algorithms"):
        try:
            torch.use_deterministic_algorithms(enabled)
            info["use_deterministic_algorithms"] = enabled
        except RuntimeError as e:
            # Some operations don't have deterministic implementations
            info["use_deterministic_algorithms"] = f"failed: {e}"

    return info


def get_seed_info() -> dict[str, Any]:
    """
    Get current state of random generators for logging.

    Note: This doesn't return the original seed (that's not recoverable
    from NumPy/PyTorch state). It returns the current RNG states which
    can be used to verify reproducibility.

    Returns
    -------
    dict
        Current RNG state information.
    """
    import numpy as np
    import torch

    info = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "python_random_state_hash": hash(tuple(random.getstate()[1][:10])),
        "numpy_random_state_hash": hash(tuple(np.random.get_state()[1][:10])),
        "torch_initial_seed": torch.initial_seed(),
    }

    if torch.cuda.is_available():
        info["torch_cuda_initial_seed"] = torch.cuda.initial_seed()

    return info


def worker_init_fn(worker_id: int) -> None:
    """
    DataLoader worker initialization function for reproducibility.

    Use this with torch.utils.data.DataLoader(worker_init_fn=worker_init_fn)
    to ensure each worker has a unique but reproducible seed.

    The worker seed is derived from the main process seed plus worker_id,
    ensuring reproducibility across runs while avoiding correlation between
    workers.

    Parameters
    ----------
    worker_id : int
        Worker ID assigned by DataLoader.
    """
    import numpy as np
    import torch

    # Get the base seed from the main process
    base_seed = torch.initial_seed() % (2**32)

    # Derive worker-specific seed deterministically
    worker_seed = (base_seed + worker_id) % (2**32)

    # Set all RNGs for this worker
    random.seed(worker_seed)
    np.random.seed(worker_seed)
    # Note: torch.manual_seed is already called per-worker by DataLoader
    # when a generator is provided, but we set it explicitly for clarity
    torch.manual_seed(worker_seed)


def log_seed_to_history(
    history: dict,
    seed_info: dict,
    deterministic_info: dict | None = None
) -> None:
    """
    Add seed information to a training history dict.

    Parameters
    ----------
    history : dict
        The training history dictionary (modified in place).
    seed_info : dict
        Output from set_global_seed().
    deterministic_info : dict or None
        Output from set_deterministic_mode(), if used.
    """
    history["seed_info"] = seed_info
    if deterministic_info is not None:
        history["deterministic_mode"] = deterministic_info


if __name__ == "__main__":
    # Quick test
    print("Testing reproducibility module...")

    # Test with explicit seed
    info1 = set_global_seed(42)
    print(f"Explicit seed: {info1}")

    import numpy as np
    import torch

    val1_np = np.random.rand()
    val1_torch = torch.rand(1).item()

    # Reset and verify
    info2 = set_global_seed(42)
    val2_np = np.random.rand()
    val2_torch = torch.rand(1).item()

    assert val1_np == val2_np, "NumPy reproducibility failed"
    assert val1_torch == val2_torch, "PyTorch reproducibility failed"
    print("✓ Reproducibility test passed")

    # Test auto seed
    info3 = set_global_seed(None)
    print(f"Auto seed: {info3['seed']} (source: {info3['seed_source']})")

    # Test deterministic mode
    det_info = set_deterministic_mode(True)
    print(f"Deterministic mode: {det_info}")

    print("\nAll tests passed!")
