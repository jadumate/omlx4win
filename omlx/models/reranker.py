# SPDX-License-Identifier: Apache-2.0
"""Torch Reranker Model wrapper using transformers."""

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class RerankOutput:
    scores: list[float]
    indices: list[int]
    total_tokens: int


class TorchRerankerModel:
    """Wrapper for document reranking using transformers."""

    _CAUSAL_LM_SYSTEM_PROMPT = (
        "Judge whether the Document meets the requirements based on the "
        'Query and the Instruct provided. Note that the answer can only be "yes" or "no".'
    )
    _CAUSAL_LM_DEFAULT_INSTRUCTION = "Given a web search query, retrieve relevant passages that answer the query"
    _DEFAULT_MAX_LENGTH_SEQ = 512
    _DEFAULT_MAX_LENGTH_CAUSAL = 8192

    def __init__(self, model_name: str):
        self.model_name = model_name
        self.model = None
        self.processor = None
        self._loaded = False
        self._num_labels: int | None = None
        self._is_causal_lm = False
        self._device = None
        self._token_true_id: int | None = None
        self._token_false_id: int | None = None
        self._prefix_tokens: list[int] | None = None
        self._suffix_tokens: list[int] | None = None

    def _get_architecture(self) -> str | None:
        config_path = Path(self.model_name) / "config.json"
        if not config_path.exists():
            return None
        try:
            with open(config_path) as f:
                cfg = json.load(f)
            arches = cfg.get("architectures", [])
            return arches[0] if arches else None
        except Exception:
            return None

    def load(self) -> None:
        if self._loaded:
            return
        import torch
        from transformers import AutoTokenizer
        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        arch = self._get_architecture()
        logger.info(f"Loading reranker: {self.model_name} (arch={arch})")

        # Try SequenceClassification first
        try:
            from transformers import AutoModelForSequenceClassification
            self.model = AutoModelForSequenceClassification.from_pretrained(
                self.model_name,
                torch_dtype=torch.float16 if self._device == "cuda" else torch.float32,
            ).to(self._device)
            self.processor = AutoTokenizer.from_pretrained(self.model_name)
            self._num_labels = self.model.config.num_labels if hasattr(self.model.config, "num_labels") else 1
            self.model.eval()
            self._loaded = True
            logger.info(f"Reranker loaded as SequenceClassification: {self.model_name}")
            return
        except Exception as e:
            logger.debug(f"SequenceClassification load failed: {e}, trying CausalLM")

        # Fallback: CausalLM-based reranker
        from transformers import AutoModelForCausalLM
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=torch.float16 if self._device == "cuda" else torch.float32,
            device_map="auto" if self._device == "cuda" else None,
        )
        if self._device == "cpu":
            self.model = self.model.to(self._device)
        self.processor = AutoTokenizer.from_pretrained(self.model_name)
        self._is_causal_lm = True
        self._num_labels = 2
        self.model.eval()
        # Resolve yes/no tokens
        self._token_true_id = self.processor.convert_tokens_to_ids("yes")
        self._token_false_id = self.processor.convert_tokens_to_ids("no")
        # Build prompt template
        _SENTINEL = "<<__CONTENT_SENTINEL__>>"
        msgs = [{"role": "system", "content": self._CAUSAL_LM_SYSTEM_PROMPT},
                {"role": "user", "content": _SENTINEL}]
        try:
            tmpl = self.processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
            parts = tmpl.split(_SENTINEL)
            prefix = parts[0]
            suffix = parts[1] + "<think>\n\n</think>\n\n" if len(parts) > 1 else ""
            self._prefix_tokens = self.processor.encode(prefix, add_special_tokens=False)
            self._suffix_tokens = self.processor.encode(suffix, add_special_tokens=False)
        except Exception:
            self._prefix_tokens = []
            self._suffix_tokens = []
        self._loaded = True
        logger.info(f"Reranker loaded as CausalLM: {self.model_name}")

    def rerank(self, query: str, documents: list[str], max_length: int | None = None) -> RerankOutput:
        if not self._loaded:
            self.load()
        if not documents:
            return RerankOutput(scores=[], indices=[], total_tokens=0)
        if self._is_causal_lm:
            return self._rerank_causal_lm(query, documents, max_length or self._DEFAULT_MAX_LENGTH_CAUSAL)
        return self._rerank_seq_classification(query, documents, max_length or self._DEFAULT_MAX_LENGTH_SEQ)

    def _rerank_seq_classification(self, query: str, documents: list[str], max_length: int) -> RerankOutput:
        import torch
        pairs = [(query, doc) for doc in documents]
        encoded = self.processor([p[0] for p in pairs], [p[1] for p in pairs],
                                   max_length=max_length, padding=True, truncation=True, return_tensors="pt")
        input_ids = encoded["input_ids"].to(self._device)
        attention_mask = encoded["attention_mask"].to(self._device)
        with torch.no_grad():
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits if hasattr(outputs, "logits") else outputs[0]
        if logits.shape[-1] == 1:
            scores = torch.sigmoid(logits.squeeze(-1)).cpu().float().tolist()
        else:
            scores = torch.softmax(logits, dim=-1)[:, -1].cpu().float().tolist()
        indexed = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)
        indices = [i for i, _ in indexed]
        total_tokens = int(attention_mask.sum().item())
        return RerankOutput(scores=scores, indices=indices, total_tokens=total_tokens)

    def _rerank_causal_lm(self, query: str, documents: list[str], max_length: int) -> RerankOutput:
        import torch
        scores = []
        total_tokens = 0
        for doc in documents:
            content = f"<Instruct>: {self._CAUSAL_LM_DEFAULT_INSTRUCTION}\n<Query>: {query}\n<Document>: {doc}"
            content_ids = self.processor.encode(content, add_special_tokens=False,
                                                 max_length=max_length - len(self._prefix_tokens or []) - len(self._suffix_tokens or []),
                                                 truncation=True)
            ids = (self._prefix_tokens or []) + content_ids + (self._suffix_tokens or [])
            input_ids = torch.tensor([ids]).to(self._device)
            with torch.no_grad():
                out = self.model(input_ids=input_ids)
            last_logits = out.logits[0, -1, :]
            if self._token_true_id and self._token_false_id:
                paired = torch.tensor([last_logits[self._token_false_id], last_logits[self._token_true_id]])
                prob = torch.softmax(paired, dim=0)[1].item()
            else:
                prob = torch.sigmoid(last_logits.max()).item()
            scores.append(prob)
            total_tokens += len(ids)
        indexed = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)
        indices = [i for i, _ in indexed]
        return RerankOutput(scores=scores, indices=indices, total_tokens=total_tokens)

    @property
    def num_labels(self) -> int | None:
        return self._num_labels

    def get_model_info(self) -> dict:
        if not self._loaded:
            return {"loaded": False, "model_name": self.model_name}
        return {"loaded": True, "model_name": self.model_name, "num_labels": self._num_labels}

    def __repr__(self):
        return f"<TorchRerankerModel model={self.model_name} loaded={self._loaded}>"


# Backward compatibility alias
MLXRerankerModel = TorchRerankerModel
