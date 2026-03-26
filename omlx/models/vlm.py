# SPDX-License-Identifier: Apache-2.0
"""Torch VLM (Vision-Language Model) adapter."""

import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class VLMModelAdapter:
    """
    Adapter wrapping a torch VLM model.
    Provides interface compatibility with the VLM engine.
    """

    def __init__(self, vlm_model: Any):
        self._vlm_model = vlm_model
        self._pending_embeds = None
        self._pending_kwargs: Dict[str, Any] = {}

    @property
    def model_type(self) -> str:
        if hasattr(self._vlm_model, "config") and hasattr(self._vlm_model.config, "model_type"):
            return self._vlm_model.config.model_type
        return "vlm"

    @property
    def config(self):
        return getattr(self._vlm_model, "config", None)

    def set_pending_embeddings(self, inputs_embeds: Any, extra_kwargs: Optional[Dict[str, Any]] = None, start_offset: int = 0) -> None:
        self._pending_embeds = inputs_embeds
        self._pending_kwargs = extra_kwargs or {}

    def clear_pending_embeddings(self) -> None:
        self._pending_embeds = None
        self._pending_kwargs = {}

    @property
    def has_pending_embeddings(self) -> bool:
        return self._pending_embeds is not None

    def get_input_embeddings(self, input_ids: Any, pixel_values: Optional[Any] = None, **kwargs) -> Any:
        if hasattr(self._vlm_model, "get_input_embeddings"):
            return self._vlm_model.get_input_embeddings(input_ids, pixel_values, **kwargs)
        return None
