"""Latency and cost trackers.

``ChatResponse`` already carries ``latency_s`` and token counts, so the trackers
never *re-measure* anything — they only **aggregate**.  Latency feeds Variant-B
savings (time lost on a prune is bounded by the LLM latency); tokens feed the
per-experiment cost figure.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

from backend import ChatResponse


@dataclass
class LatencyTracker:
    """Aggregates per-call LLM latencies."""

    latencies: List[float] = field(default_factory=list)

    def record(self, response: ChatResponse) -> float:
        self.latencies.append(float(response.latency_s))
        return response.latency_s

    @property
    def n_calls(self) -> int:
        return len(self.latencies)

    @property
    def total_s(self) -> float:
        return float(sum(self.latencies))

    @property
    def mean_s(self) -> float:
        return self.total_s / self.n_calls if self.latencies else 0.0

    def reset(self) -> None:
        self.latencies.clear()


@dataclass
class CostTracker:
    """Aggregates token usage (and #calls) across an experiment."""

    input_tokens: int = 0
    output_tokens: int = 0
    n_calls: int = 0

    def record(self, response: ChatResponse) -> None:
        self.input_tokens += int(response.input_tokens or 0)
        self.output_tokens += int(response.output_tokens or 0)
        self.n_calls += 1

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens

    def as_dict(self) -> dict:
        return {
            "n_calls": self.n_calls,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "total_tokens": self.total_tokens,
        }

    def reset(self) -> None:
        self.input_tokens = self.output_tokens = self.n_calls = 0
