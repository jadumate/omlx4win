# SPDX-License-Identifier: Apache-2.0
"""XLMRoberta stub — uses transformers natively on Windows."""

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class ModelArgs:
    hidden_size: int = 768
    num_hidden_layers: int = 12
    num_attention_heads: int = 12
    intermediate_size: int = 3072
    vocab_size: int = 250002
    max_position_embeddings: int = 514
    type_vocab_size: int = 1
    pad_token_id: int = 1
    num_labels: int = 1
    architectures: List[str] = field(default_factory=list)
    model_type: str = "xlm-roberta"
    hidden_dropout_prob: float = 0.1
    attention_probs_dropout_prob: float = 0.1
    layer_norm_eps: float = 1e-5

    # Allow extra fields from config.json
    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)


class Model:
    """Stub - actual loading done via transformers AutoModel."""
    def __init__(self, config: ModelArgs):
        self.config = config

    def sanitize(self, weights: dict) -> dict:
        return weights

    def load_weights(self, weights):
        pass

    def parameters(self):
        return []
