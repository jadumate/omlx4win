# SPDX-License-Identifier: Apache-2.0
"""Torch Language Model wrapper using transformers."""

import logging
import threading
from dataclasses import dataclass
from typing import Iterator, Optional

logger = logging.getLogger(__name__)


@dataclass
class GenerationOutput:
    text: str
    tokens: list[int]
    finish_reason: str | None = None


@dataclass
class StreamingOutput:
    text: str
    token: int
    finished: bool = False
    finish_reason: str | None = None


class TorchLanguageModel:
    """Wrapper around transformers for LLM inference."""

    def __init__(self, model_name: str, tokenizer_name: str | None = None, trust_remote_code: bool = False):
        self.model_name = model_name
        self.tokenizer_name = tokenizer_name or model_name
        self.trust_remote_code = trust_remote_code
        self.model = None
        self.tokenizer = None
        self._loaded = False
        self._device = None

    def load(self) -> None:
        if self._loaded:
            return
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        logger.info(f"Loading model: {self.model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.tokenizer_name, trust_remote_code=self.trust_remote_code
        )
        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            trust_remote_code=self.trust_remote_code,
            torch_dtype=torch.float16 if self._device == "cuda" else torch.float32,
            device_map="auto" if self._device == "cuda" else None,
        )
        if self._device == "cpu":
            self.model = self.model.to(self._device)
        self.model.eval()
        self._loaded = True
        logger.info(f"Model loaded: {self.model_name}")

    def _sample_token(self, logits, temperature: float, top_p: float, top_k: int):
        import torch
        if temperature == 0 or temperature < 1e-7:
            return logits.argmax(dim=-1).item()
        logits = logits / temperature
        if top_k > 0:
            top_k_vals, _ = torch.topk(logits, top_k)
            logits[logits < top_k_vals[..., -1:]] = float("-inf")
        if top_p < 1.0:
            sorted_logits, sorted_idx = torch.sort(logits, descending=True)
            cumprobs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
            sorted_logits[cumprobs - torch.softmax(sorted_logits, dim=-1) > top_p] = float("-inf")
            logits = sorted_logits.scatter(-1, sorted_idx, sorted_logits)
        probs = torch.softmax(logits, dim=-1)
        return torch.multinomial(probs, num_samples=1).item()

    def generate(self, prompt: str, max_tokens: int = 256, temperature: float = 0.7,
                  top_p: float = 0.9, top_k: int = 0, stop: list[str] | None = None, **kwargs) -> GenerationOutput:
        if not self._loaded:
            self.load()
        import torch
        input_ids = self.tokenizer.encode(prompt, return_tensors="pt").to(self._device)
        generated_ids = []
        past_key_values = None
        accumulated_text = ""
        finish_reason = "length"
        eos_id = self.tokenizer.eos_token_id
        with torch.no_grad():
            attn_mask = torch.ones_like(input_ids)
            for _ in range(max_tokens):
                out = self.model(input_ids=input_ids if past_key_values is None else input_ids[:, -1:],
                                  attention_mask=attn_mask, past_key_values=past_key_values, use_cache=True)
                past_key_values = out.past_key_values
                token_id = self._sample_token(out.logits[0, -1], temperature, top_p, top_k)
                if past_key_values is not None:
                    attn_mask = torch.cat([attn_mask, torch.ones((1, 1), device=self._device)], dim=1)
                input_ids = torch.tensor([[token_id]], device=self._device)
                generated_ids.append(token_id)
                new_text = self.tokenizer.decode([token_id], skip_special_tokens=True)
                accumulated_text += new_text
                if token_id == eos_id:
                    finish_reason = "stop"
                    break
                if stop:
                    for seq in stop:
                        if seq in accumulated_text:
                            finish_reason = "stop"
                            break
                    if finish_reason == "stop":
                        break
        return GenerationOutput(text=accumulated_text, tokens=generated_ids, finish_reason=finish_reason)

    def stream_generate(self, prompt: str, max_tokens: int = 256, temperature: float = 0.7,
                         top_p: float = 0.9, top_k: int = 0, stop: list[str] | None = None, **kwargs) -> Iterator[StreamingOutput]:
        if not self._loaded:
            self.load()
        import torch
        input_ids = self.tokenizer.encode(prompt, return_tensors="pt").to(self._device)
        past_key_values = None
        accumulated_text = ""
        eos_id = self.tokenizer.eos_token_id
        attn_mask = torch.ones_like(input_ids)
        with torch.no_grad():
            for step in range(max_tokens):
                out = self.model(input_ids=input_ids if past_key_values is None else input_ids[:, -1:],
                                  attention_mask=attn_mask, past_key_values=past_key_values, use_cache=True)
                past_key_values = out.past_key_values
                if past_key_values is not None:
                    attn_mask = torch.cat([attn_mask, torch.ones((1, 1), device=self._device)], dim=1)
                token_id = self._sample_token(out.logits[0, -1], temperature, top_p, top_k)
                input_ids = torch.tensor([[token_id]], device=self._device)
                new_text = self.tokenizer.decode([token_id], skip_special_tokens=False)
                accumulated_text += new_text
                is_eos = token_id == eos_id
                should_stop = is_eos
                if stop and not should_stop:
                    for seq in stop:
                        if seq in accumulated_text:
                            should_stop = True
                            break
                finished = should_stop or (step + 1 >= max_tokens)
                finish_reason = None
                if finished:
                    finish_reason = "stop" if should_stop else "length"
                yield StreamingOutput(text=new_text, token=token_id, finished=finished, finish_reason=finish_reason)
                if finished:
                    break

    def chat(self, messages: list[dict], max_tokens: int = 256, temperature: float = 0.7,
              top_p: float = 0.9, tools: list | None = None, enable_thinking: bool | None = None, **kwargs) -> GenerationOutput:
        if not self._loaded:
            self.load()
        from ..api.utils import detect_and_strip_partial
        is_partial = detect_and_strip_partial(messages)
        template_kwargs = {"tokenize": False, "add_generation_prompt": not is_partial}
        if is_partial:
            template_kwargs["continue_final_message"] = True
        if tools:
            template_kwargs["tools"] = tools
        if enable_thinking is not None:
            template_kwargs["enable_thinking"] = enable_thinking
        try:
            prompt = self.tokenizer.apply_chat_template(messages, **template_kwargs)
        except TypeError:
            template_kwargs.pop("tools", None)
            template_kwargs.pop("enable_thinking", None)
            try:
                prompt = self.tokenizer.apply_chat_template(messages, **template_kwargs)
            except Exception:
                prompt = "\n".join(f"{m['role']}: {m['content']}" for m in messages) + "\nassistant:"
        return self.generate(prompt=prompt, max_tokens=max_tokens, temperature=temperature, top_p=top_p, **kwargs)

    def get_model_info(self) -> dict:
        if not self._loaded:
            return {"loaded": False, "model_name": self.model_name}
        info = {"loaded": True, "model_name": self.model_name}
        if hasattr(self.model, "config"):
            cfg = self.model.config
            info.update({"vocab_size": getattr(cfg, "vocab_size", None),
                          "hidden_size": getattr(cfg, "hidden_size", None),
                          "num_layers": getattr(cfg, "num_hidden_layers", None)})
        return info

    def __repr__(self):
        return f"<TorchLanguageModel model={self.model_name} loaded={self._loaded}>"


# Backward compatibility alias
MLXLanguageModel = TorchLanguageModel
