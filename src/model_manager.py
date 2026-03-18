from __future__ import annotations
"""
Model Manager: Handles Ollama calls with automatic model swapping
and tracks token usage + estimated energy costs.

Ollama swaps models automatically when you call a different one,
but we track it explicitly for logging and VRAM awareness.
"""

import time
import ollama
from src.config import VISION_MODEL, TEXT_MODEL, GPU_TDP_WATTS, ENERGY_COST_PER_KWH


class UsageStats:
    """Tracks token counts and estimates energy cost per call."""

    def __init__(self, model: str, prompt_tokens: int, completion_tokens: int,
                 duration_seconds: float):
        self.model = model
        self.prompt_tokens = prompt_tokens
        self.completion_tokens = completion_tokens
        self.total_tokens = prompt_tokens + completion_tokens
        self.duration_seconds = duration_seconds

        # Energy estimate: GPU TDP × duration
        # Real power draw is ~60-80% of TDP during inference, we use 70%
        gpu_power_kw = (GPU_TDP_WATTS * 0.7) / 1000
        self.energy_kwh = gpu_power_kw * (duration_seconds / 3600)
        self.energy_cost_eur = self.energy_kwh * ENERGY_COST_PER_KWH

        # Estimated API cost comparison (rough OpenAI GPT-4o-mini prices)
        # Input: $0.15/1M tokens, Output: $0.60/1M tokens
        self.estimated_api_cost_usd = (
            (prompt_tokens * 0.15 / 1_000_000)
            + (completion_tokens * 0.60 / 1_000_000)
        )

    def __repr__(self):
        return (
            f"[{self.model}] {self.total_tokens} tokens "
            f"({self.prompt_tokens} in, {self.completion_tokens} out) | "
            f"{self.duration_seconds:.1f}s | "
            f"~{self.energy_kwh * 1000:.2f} Wh / {self.energy_cost_eur:.4f} EUR | "
            f"API equiv: ~${self.estimated_api_cost_usd:.4f}"
        )


class ModelManager:
    """Manages Ollama model calls with tracking."""

    def __init__(self):
        self.current_model = None
        self.total_stats = {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_energy_kwh": 0,
            "total_cost_eur": 0,
            "total_api_equiv_usd": 0,
            "call_count": 0,
        }

    def _extract_stats(self, response: dict, model: str,
                       start_time: float) -> UsageStats:
        """Extract token counts from Ollama response."""
        duration = time.time() - start_time

        # Ollama returns token counts in different fields depending on endpoint
        prompt_tokens = response.get("prompt_eval_count", 0)
        completion_tokens = response.get("eval_count", 0)

        stats = UsageStats(model, prompt_tokens, completion_tokens, duration)

        # Accumulate totals
        self.total_stats["prompt_tokens"] += prompt_tokens
        self.total_stats["completion_tokens"] += completion_tokens
        self.total_stats["total_energy_kwh"] += stats.energy_kwh
        self.total_stats["total_cost_eur"] += stats.energy_cost_eur
        self.total_stats["total_api_equiv_usd"] += stats.estimated_api_cost_usd
        self.total_stats["call_count"] += 1

        return stats

    def chat(self, model: str, messages: list[dict],
             images: list[bytes] | None = None) -> tuple[str, UsageStats]:
        """
        Send a chat request to Ollama.

        Args:
            model: Model name (e.g. "llava:7b" or "llama3.1:8b")
            messages: Chat messages in OpenAI format
            images: Optional list of image bytes (for vision models)

        Returns:
            (response_text, usage_stats)
        """
        # If images provided, attach to the last user message
        if images and messages:
            messages[-1]["images"] = images

        self.current_model = model
        start_time = time.time()

        response = ollama.chat(model=model, messages=messages)

        stats = self._extract_stats(response, model, start_time)
        text = response["message"]["content"]

        print(f"  📊 {stats}")
        return text, stats

    def vision(self, image_bytes: bytes, prompt: str) -> tuple[str, UsageStats]:
        """Analyze an image with LLaVA."""
        messages = [{"role": "user", "content": prompt}]
        return self.chat(VISION_MODEL, messages, images=[image_bytes])

    def reason(self, messages: list[dict]) -> tuple[str, UsageStats]:
        """Text reasoning with Llama."""
        return self.chat(TEXT_MODEL, messages)

    def get_total_stats(self) -> dict:
        """Return accumulated stats for reporting."""
        return {
            **self.total_stats,
            "total_tokens": (self.total_stats["prompt_tokens"]
                             + self.total_stats["completion_tokens"]),
        }
