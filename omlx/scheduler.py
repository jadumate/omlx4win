# SPDX-License-Identifier: Apache-2.0
"""
Torch-based scheduler for oMLX on Windows.

Replaces the MLX/BatchGenerator-based scheduler with a torch+transformers
implementation. Uses per-request KV cache (past_key_values) for streaming.
"""

import gc
import logging
import time
import threading
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)

# Import streaming detokenizer if available
try:
    from transformers import AutoTokenizer
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False

# These imports are stubs for cache/tiered cache compatibility
try:
    from .cache.paged_cache import PagedCacheManager
    HAS_PAGED_CACHE = True
except ImportError:
    PagedCacheManager = None
    HAS_PAGED_CACHE = False

# Import Harmony adapter
try:
    from .adapter.harmony import HarmonyStreamingParser, parse_tool_calls_from_tokens
    from .utils.tokenizer import is_harmony_model
    HAS_HARMONY_ADAPTER = True
except ImportError:
    HarmonyStreamingParser = None
    parse_tool_calls_from_tokens = None
    is_harmony_model = None
    HAS_HARMONY_ADAPTER = False

from .request import Request, RequestOutput, RequestStatus, SamplingParams


class SchedulingPolicy(Enum):
    FCFS = "fcfs"
    PRIORITY = "priority"


@dataclass
class SchedulerConfig:
    max_num_seqs: int = 256
    max_num_batched_tokens: int = 8192
    policy: SchedulingPolicy = SchedulingPolicy.FCFS
    completion_batch_size: int = 32
    prefill_step_size: int = 2048
    paged_cache_block_size: int = 256
    max_cache_blocks: Optional[int] = None
    paged_ssd_cache_dir: Optional[str] = None
    paged_ssd_cache_max_size: int = 100 * 1024 * 1024 * 1024
    hot_cache_max_size: int = 0
    model_name: str = ""
    gc_cleanup_interval: int = 0
    mlx_cache_cleanup_interval: int = 512


@dataclass
class SchedulerOutput:
    scheduled_request_ids: List[str] = field(default_factory=list)
    num_scheduled_tokens: int = 0
    finished_request_ids: Set[str] = field(default_factory=set)
    outputs: List[RequestOutput] = field(default_factory=list)
    did_work: bool = False
    num_running: int = 0
    num_waiting: int = 0
    num_completed: int = 0


class _RequestState:
    """State for an in-progress generation request."""

    def __init__(self, request: Request, model: Any, tokenizer: Any, device: str):
        self.request = request
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.past_key_values = None
        self.attention_mask = None
        self.last_token_id: Optional[int] = None
        self.is_prefilled = False
        self._step_count = 0

    def prefill(self) -> int:
        """Run the initial prefill and return the first generated token."""
        import torch
        token_ids = self.request.prompt_token_ids
        if not token_ids:
            token_ids = self.tokenizer.encode(
                self.request.prompt if isinstance(self.request.prompt, str) else "",
                add_special_tokens=True
            )
        input_ids = torch.tensor([token_ids]).to(self.device)
        self.attention_mask = torch.ones_like(input_ids)
        with torch.no_grad():
            out = self.model(input_ids=input_ids, attention_mask=self.attention_mask, use_cache=True)
        self.past_key_values = out.past_key_values
        sp = self.request.sampling_params
        token_id = self._sample(out.logits[0, -1], sp)
        self.last_token_id = token_id
        self.is_prefilled = True
        return token_id

    def decode_step(self) -> int:
        """Run one decode step and return the next token."""
        import torch
        assert self.last_token_id is not None
        input_ids = torch.tensor([[self.last_token_id]]).to(self.device)
        self.attention_mask = torch.cat(
            [self.attention_mask, torch.ones((1, 1), device=self.device)], dim=1
        )
        with torch.no_grad():
            out = self.model(
                input_ids=input_ids,
                attention_mask=self.attention_mask,
                past_key_values=self.past_key_values,
                use_cache=True,
            )
        self.past_key_values = out.past_key_values
        sp = self.request.sampling_params
        token_id = self._sample(out.logits[0, -1], sp)
        self.last_token_id = token_id
        self._step_count += 1
        return token_id

    def _sample(self, logits: Any, sp: SamplingParams) -> int:
        import torch
        if sp.temperature == 0 or sp.temperature < 1e-7:
            return int(logits.argmax(dim=-1).item())
        logits = logits / max(sp.temperature, 1e-7)
        if sp.top_k > 0:
            top_k_vals, _ = torch.topk(logits, min(sp.top_k, logits.size(-1)))
            logits = logits.masked_fill(logits < top_k_vals[..., -1:], float("-inf"))
        if sp.top_p < 1.0:
            sorted_logits, sorted_idx = torch.sort(logits, descending=True)
            cumprobs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
            mask = cumprobs - torch.softmax(sorted_logits, dim=-1) > sp.top_p
            sorted_logits = sorted_logits.masked_fill(mask, float("-inf"))
            logits = torch.zeros_like(logits).scatter(-1, sorted_idx, sorted_logits)
        probs = torch.softmax(logits, dim=-1)
        return int(torch.multinomial(probs, num_samples=1).item())

    def free(self):
        """Release KV cache memory."""
        self.past_key_values = None
        self.attention_mask = None


class Scheduler:
    """
    Torch-based scheduler for continuous batching.

    Processes requests sequentially using per-request KV caches (past_key_values).
    Maintains the same interface as the original MLX-based scheduler.
    """

    def __init__(self, model: Any, tokenizer: Any, config: Optional[SchedulerConfig] = None):
        self.model = model
        self.tokenizer = tokenizer
        self.config = config or SchedulerConfig()
        self._device = self._get_device()

        self._waiting: deque = deque()
        self._running: Dict[str, _RequestState] = {}
        self._uid_counter = 0
        self._uid_to_rid: Dict[int, str] = {}
        self._lock = threading.Lock()

        self._steps = 0
        self._total_completion_tokens = 0
        self._start_time = time.time()

        # Compatibility attributes used by engine_core and process_memory_enforcer
        self._specprefill_draft_model = None
        self._memory_limit_bytes: int = 0
        self._memory_hard_limit_bytes: int = 0
        self._prefill_memory_guard: bool = False

    def _get_device(self) -> str:
        try:
            import torch
            if torch.cuda.is_available():
                return "cuda"
        except ImportError:
            pass
        return "cpu"

    def add_request(self, request: Request) -> None:
        with self._lock:
            uid = self._uid_counter
            self._uid_counter += 1
            request.batch_uid = uid
            self._uid_to_rid[uid] = request.request_id
            self._waiting.append(request)

    def abort_request(self, uid_or_rid) -> bool:
        """Abort a request by UID (int) or request_id (str)."""
        if isinstance(uid_or_rid, int):
            rid = self._uid_to_rid.get(uid_or_rid)
        else:
            rid = uid_or_rid
        if rid and rid in self._running:
            state = self._running.pop(rid)
            state.free()
            return True
        return False

    def has_requests(self) -> bool:
        return bool(self._waiting) or bool(self._running)

    def step(self) -> SchedulerOutput:
        """Run one generation step for all active requests."""
        # Start new requests
        with self._lock:
            while self._waiting and len(self._running) < self.config.max_num_seqs:
                req = self._waiting.popleft()
                req.status = RequestStatus.RUNNING
                state = _RequestState(req, self.model, self.tokenizer, self._device)
                self._running[req.request_id] = state

        if not self._running:
            return SchedulerOutput(did_work=False)

        outputs = []
        finished_ids: Set[str] = set()
        eos_id = getattr(self.tokenizer, "eos_token_id", None)

        for rid, state in list(self._running.items()):
            request = state.request

            try:
                # Prefill or decode
                if not state.is_prefilled:
                    token_id = state.prefill()
                else:
                    token_id = state.decode_step()
            except Exception as e:
                logger.error(f"Generation error for {rid}: {e}")
                output = RequestOutput(
                    request_id=rid, finished=True, finish_reason="error",
                    error=str(e), prompt_tokens=request.num_prompt_tokens
                )
                outputs.append(output)
                finished_ids.add(rid)
                state.free()
                continue

            # Decode token to text
            try:
                new_text = self.tokenizer.decode([token_id], skip_special_tokens=False)
            except Exception:
                new_text = ""

            request.append_output_token(token_id)
            request.output_text = request.output_text + new_text
            self._total_completion_tokens += 1

            # Check stop conditions
            sp = request.sampling_params
            finished = False
            finish_reason = None

            if token_id == eos_id:
                finished = True
                finish_reason = "stop"
            elif sp.stop_token_ids and token_id in sp.stop_token_ids:
                finished = True
                finish_reason = "stop"
            elif sp.stop:
                for seq in sp.stop:
                    if seq in request.output_text:
                        finished = True
                        finish_reason = "stop"
                        break
            if not finished and request.num_output_tokens >= request.max_tokens:
                finished = True
                finish_reason = "length"

            if finished:
                request.set_finished(
                    RequestStatus.FINISHED_STOPPED if finish_reason == "stop"
                    else RequestStatus.FINISHED_LENGTH_CAPPED,
                    finish_reason,
                )
                finished_ids.add(rid)
                state.free()

            output = RequestOutput(
                request_id=rid,
                new_token_ids=[token_id],
                new_text=new_text,
                output_token_ids=list(request.output_token_ids),
                output_text=request.output_text,
                finished=finished,
                finish_reason=finish_reason,
                prompt_tokens=request.num_prompt_tokens,
                completion_tokens=request.num_output_tokens,
            )
            outputs.append(output)

        # Clean up finished
        for rid in finished_ids:
            self._running.pop(rid, None)

        # GC cleanup
        if self.config.gc_cleanup_interval > 0 and self._steps % self.config.gc_cleanup_interval == 0:
            gc.collect()

        self._steps += 1

        return SchedulerOutput(
            outputs=outputs,
            finished_request_ids=finished_ids,
            did_work=bool(outputs),
            num_running=len(self._running),
            num_waiting=len(self._waiting),
        )

    def fail_all_requests(self) -> List[str]:
        failed = []
        with self._lock:
            for rid, state in list(self._running.items()):
                state.free()
                failed.append(rid)
            self._running.clear()
            while self._waiting:
                req = self._waiting.popleft()
                failed.append(req.request_id)
        return failed

    def remove_finished_request(self, request_id: str) -> None:
        """Remove a finished request from tracking (no-op if already removed)."""
        self._running.pop(request_id, None)

    def get_cache_stats(self) -> Optional[Dict[str, Any]]:
        """Get cache statistics — returns None (no cache on torch backend)."""
        return None

    def set_specprefill_draft_model(self, draft_model: Any, draft_model_name: str = "") -> None:
        """SpecPrefill — no-op on torch backend."""
        pass

    def shutdown(self) -> None:
        """Shutdown the scheduler."""
        self.fail_all_requests()

    def deep_reset(self) -> None:
        """Deep reset — clears all state."""
        with self._lock:
            for state in self._running.values():
                state.free()
            self._running.clear()
            self._waiting.clear()
        gc.collect()

    def get_stats(self) -> Dict[str, Any]:
        elapsed = time.time() - self._start_time
        return {
            "num_waiting": len(self._waiting),
            "num_running": len(self._running),
            "total_steps": self._steps,
            "total_completion_tokens": self._total_completion_tokens,
            "tokens_per_second": self._total_completion_tokens / max(elapsed, 1),
            "uptime_seconds": elapsed,
        }
