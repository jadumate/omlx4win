# SPDX-License-Identifier: Apache-2.0
"""
Unified hardware detection for Windows (torch backend).

Single source of truth for:
- CPU identification
- Memory detection (total, available, max working set)
- Torch/CUDA availability checks
"""

from __future__ import annotations

import hashlib
import logging
import platform
import re
import uuid
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger(__name__)

# Default fallback value for all memory functions (conservative)
DEFAULT_MEMORY_BYTES = 8 * 1024 * 1024 * 1024  # 8GB


@dataclass
class HardwareInfo:
    """Hardware information."""

    chip_name: str
    total_memory_gb: float
    max_working_set_bytes: int
    mlx_device_name: Optional[str] = None


# =============================================================================
# Core Detection Functions
# =============================================================================


def get_chip_name() -> str:
    """
    Get CPU name via platform.processor() on Windows or fallback.

    Returns:
        CPU name (e.g., "Intel64 Family 6 Model 85") or "Unknown CPU" as fallback.
    """
    try:
        name = platform.processor()
        if name:
            return name
    except Exception:
        pass
    return "Unknown CPU"


def get_total_memory_bytes() -> int:
    """
    Get total system memory in bytes via psutil.

    Returns:
        Total memory in bytes.
    """
    try:
        import psutil

        return psutil.virtual_memory().total
    except ImportError:
        pass

    # Last resort: default
    logger.warning(f"Using default memory size: {DEFAULT_MEMORY_BYTES // (1024**3)} GB")
    return DEFAULT_MEMORY_BYTES


def get_total_memory_gb() -> float:
    """Get total system memory in GB."""
    return get_total_memory_bytes() / (1024**3)


def get_max_working_set_bytes() -> int:
    """
    Get maximum recommended working set size (75% of total RAM).

    Returns:
        Maximum working set size in bytes.
    """
    try:
        import psutil

        total_ram = psutil.virtual_memory().total
        return int(total_ram * 0.75)
    except ImportError:
        pass

    # Last resort: default
    logger.warning(
        f"Using default max working set: {DEFAULT_MEMORY_BYTES // (1024**3)} GB"
    )
    return DEFAULT_MEMORY_BYTES


def get_mlx_device_name() -> Optional[str]:
    """Get device name — returns None on Windows (no MLX)."""
    return None


def detect_hardware() -> HardwareInfo:
    """
    Detect hardware and return complete info.

    Returns:
        HardwareInfo with all hardware specifications.
    """
    return HardwareInfo(
        chip_name=get_chip_name(),
        total_memory_gb=get_total_memory_gb(),
        max_working_set_bytes=get_max_working_set_bytes(),
        mlx_device_name=get_mlx_device_name(),
    )


# =============================================================================
# MLX Availability Checks (always False on Windows)
# =============================================================================


def is_apple_silicon() -> bool:
    """Check if running on Apple Silicon — always False on Windows."""
    return False


def is_mlx_available() -> bool:
    """Check if MLX is available — always False on Windows."""
    return False


# =============================================================================
# Version Information
# =============================================================================


def get_mlx_version() -> str:
    """Get MLX version string — N/A on Windows."""
    return "N/A"


def get_mlx_lm_version() -> str:
    """Get mlx-lm version string — N/A on Windows."""
    return "N/A"


def get_mlx_vlm_version() -> str:
    """Get mlx-vlm version string — N/A on Windows."""
    return "N/A"


# =============================================================================
# Benchmark / omlx.ai Integration
# =============================================================================

_OWNER_HASH_ALPHABET = "0123456789abcdefghijklmnopqrstuvwxyz"


def get_gpu_core_count() -> Optional[int]:
    """Get CUDA device count via torch."""
    try:
        import torch

        count = torch.cuda.device_count()
        return count if count > 0 else None
    except Exception:
        pass
    return None


def get_io_platform_uuid() -> Optional[str]:
    """Get a unique machine identifier via uuid.getnode()."""
    try:
        node = uuid.getnode()
        # Format as UUID-like hex string
        return format(node, "012x")
    except Exception:
        pass
    return None


def parse_chip_info(chip_string: str) -> tuple[str, str]:
    """Parse chip name and variant from brand string.

    Args:
        chip_string: e.g. "Apple M4 Pro", "Intel Core i9", "AMD Ryzen 9"

    Returns:
        (chip_name, chip_variant) e.g. ("M4", "Pro"), ("M3", "Max"), ("M2", "")
    """
    match = re.search(r"M(\d+)\s*(Pro|Max|Ultra)?", chip_string)
    if not match:
        return ("M1", "")
    chip_name = f"M{match.group(1)}"
    chip_variant = match.group(2) or ""
    return (chip_name, chip_variant)


def compute_owner_hash(
    uuid: str, chip_name: str, gpu_cores: Optional[int], memory_gb: int
) -> str:
    """Compute owner_hash for omlx.ai benchmark submissions.

    Format: SHA-256(uuid + chip_name + gpu_cores + memory_gb) + verify_char
    The verify_char is ALPHABET[sum(charCodes of hash) % 36].

    Returns:
        Full owner_hash including verify character.
    """
    raw = f"{uuid}{chip_name}{gpu_cores}{memory_gb}"
    hash_hex = hashlib.sha256(raw.encode()).hexdigest()
    verify_sum = sum(ord(c) for c in hash_hex)
    verify_char = _OWNER_HASH_ALPHABET[verify_sum % 36]
    return hash_hex + verify_char


def get_os_version() -> str:
    """Get OS version string."""
    try:
        ver = platform.version()
        if ver:
            return ver
    except Exception:
        pass
    return "Windows"


# =============================================================================
# Utility Functions
# =============================================================================


def format_bytes(bytes_value: int) -> str:
    """Format bytes as human-readable string (e.g., '16.00 GB')."""
    if bytes_value >= 1024**3:
        return f"{bytes_value / 1024**3:.2f} GB"
    elif bytes_value >= 1024**2:
        return f"{bytes_value / 1024**2:.2f} MB"
    elif bytes_value >= 1024:
        return f"{bytes_value / 1024:.2f} KB"
    else:
        return f"{bytes_value} B"
