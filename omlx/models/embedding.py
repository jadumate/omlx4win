# SPDX-License-Identifier: Apache-2.0
"""Torch Embedding Model wrapper using transformers."""

import logging
from dataclasses import dataclass
from typing import List, Optional

logger = logging.getLogger(__name__)


@dataclass
class EmbeddingOutput:
    embeddings: List[List[float]]
    total_tokens: int
    dimensions: int = 0


class TorchEmbeddingModel:
    """Wrapper around transformers for embedding generation."""

    def __init__(self, model_name: str):
        self.model_name = model_name
        self.model = None
        self.processor = None
        self._loaded = False
        self._hidden_size: Optional[int] = None
        self._device = None

    def load(self) -> None:
        if self._loaded:
            return
        import torch
        from transformers import AutoModel, AutoTokenizer
        logger.info(f"Loading embedding model: {self.model_name}")
        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        self.processor = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModel.from_pretrained(
            self.model_name,
            torch_dtype=torch.float16 if self._device == "cuda" else torch.float32,
        ).to(self._device)
        self.model.eval()
        if hasattr(self.model.config, "hidden_size"):
            self._hidden_size = self.model.config.hidden_size
        self._loaded = True
        logger.info(f"Embedding model loaded: {self.model_name}")

    def embed(self, texts: List[str], max_length: int = 512, padding: bool = True, truncation: bool = True) -> EmbeddingOutput:
        if not self._loaded:
            self.load()
        if isinstance(texts, str):
            texts = [texts]
        import torch
        encoded = self.processor(texts, max_length=max_length, padding=padding,
                                   truncation=truncation, return_tensors="pt")
        input_ids = encoded["input_ids"].to(self._device)
        attention_mask = encoded["attention_mask"].to(self._device)
        with torch.no_grad():
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        # Mean pooling
        hidden = outputs.last_hidden_state  # (batch, seq, hidden)
        mask = attention_mask.unsqueeze(-1).float()
        pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)
        # L2 normalize
        norms = pooled.norm(dim=-1, keepdim=True).clamp(min=1e-9)
        normalized = (pooled / norms).cpu().float().tolist()
        total_tokens = int(attention_mask.sum().item())
        dims = len(normalized[0]) if normalized else 0
        return EmbeddingOutput(embeddings=normalized, total_tokens=total_tokens, dimensions=dims)

    def _count_tokens(self, texts: List[str]) -> int:
        total = 0
        for text in texts:
            tokens = self.processor.encode(text, add_special_tokens=True)
            total += len(tokens) if isinstance(tokens, list) else len(list(tokens))
        return total

    @property
    def hidden_size(self) -> Optional[int]:
        return self._hidden_size

    def get_model_info(self) -> dict:
        if not self._loaded:
            return {"loaded": False, "model_name": self.model_name}
        return {"loaded": True, "model_name": self.model_name, "hidden_size": self._hidden_size}

    def __repr__(self):
        return f"<TorchEmbeddingModel model={self.model_name} loaded={self._loaded}>"


# Backward compatibility alias
MLXEmbeddingModel = TorchEmbeddingModel
