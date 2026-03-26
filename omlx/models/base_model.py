# SPDX-License-Identifier: Apache-2.0
"""
Base model utilities for omlx custom model implementations.

This module provides common utilities for implementing custom models
using torch+transformers on Windows.
"""

from dataclasses import dataclass
from typing import Optional

import torch


@dataclass
class BaseModelArgs:
    """Base class for model configuration arguments."""

    pass


@dataclass
class BaseModelOutput:
    """Base output class for model forward pass."""

    last_hidden_state: torch.Tensor
    """Hidden states from the last layer."""

    text_embeds: Optional[torch.Tensor] = None
    """Normalized text embeddings."""

    pooler_output: Optional[torch.Tensor] = None
    """Pooled output (e.g., CLS token or mean pooling)."""

    hidden_states: Optional[tuple] = None
    """All hidden states if output_hidden_states=True."""


def mean_pooling(hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    """
    Perform mean pooling over sequence with attention mask.

    Args:
        hidden_states: Shape (batch_size, seq_len, hidden_size)
        attention_mask: Shape (batch_size, seq_len)

    Returns:
        Pooled output of shape (batch_size, hidden_size)
    """
    # Expand mask to match hidden states shape
    mask_expanded = attention_mask[:, :, None].float()

    # Sum embeddings weighted by mask
    sum_embeddings = torch.sum(hidden_states * mask_expanded, dim=1)

    # Sum mask values (clamp to avoid division by zero)
    sum_mask = torch.clamp(torch.sum(mask_expanded, dim=1), min=1e-9)

    return sum_embeddings / sum_mask


def normalize_embeddings(embeddings: torch.Tensor) -> torch.Tensor:
    """
    L2 normalize embeddings.

    Args:
        embeddings: Shape (..., hidden_size)

    Returns:
        Normalized embeddings with same shape
    """
    return embeddings / torch.linalg.norm(embeddings, dim=-1, keepdim=True)
