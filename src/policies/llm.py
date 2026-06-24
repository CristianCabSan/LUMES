"""The LLM judge policy."""

from __future__ import annotations

from collections import Counter
from typing import Dict, Optional

from backend import Backend
from config import ContextLevel, JudgeConfig
from prompting import CONTINUE, PromptBuilder, get_system_prompt, parse_decision
from scenario import DecisionScenario
from tracking import CostTracker, LatencyTracker
from .base import Decision, Policy


class LLMPolicy(Policy):
    """Queries an LLM backend to decide CONTINUE/PRUNE per epoch.

    * builds ``(system, user)`` at the configured :class:`ContextLevel`;
    * samples the backend ``n_samples`` times and takes a majority vote;
    * ``confidence`` = fraction of samples agreeing with the winning action;
    * caches by scenario hash (so identical queries cost nothing and the
      determinism study can be run separately and on purpose);
    * aggregates latency and tokens into the injected trackers.
    """

    name = "llm"

    def __init__(
        self,
        backend: Backend,
        config: Optional[JudgeConfig] = None,
        prompt_builder: Optional[PromptBuilder] = None,
        latency_tracker: Optional[LatencyTracker] = None,
        cost_tracker: Optional[CostTracker] = None,
    ):
        self.backend = backend
        self.cfg = config or JudgeConfig()
        self.prompts = prompt_builder or PromptBuilder(
            system_prompt=get_system_prompt(self.cfg.prompt_variant))
        self.latency = latency_tracker or LatencyTracker()
        self.cost = cost_tracker or CostTracker()
        self._cache: Dict[str, Decision] = {}
        self.name = (f"llm:{self.cfg.prompt_variant}:{getattr(backend, 'model', '?')}"
                     f":L{int(self.cfg.context_level)}")

    def _cache_key(self, scenario: DecisionScenario) -> str:
        return f"{getattr(self.backend, 'model', '?')}|L{int(self.cfg.context_level)}|{scenario.cache_key()}"

    def decide(self, scenario: DecisionScenario) -> Decision:
        key = self._cache_key(scenario)
        if self.cfg.use_cache and key in self._cache:
            return self._cache[key]

        system, user = self.prompts.build(scenario, self.cfg.context_level)

        votes = []
        latency_s = 0.0
        tokens = 0
        raw_last = None
        for _ in range(max(1, self.cfg.n_samples)):
            resp = self.backend.chat(system, user)
            self.latency.record(resp)
            self.cost.record(resp)
            latency_s += float(resp.latency_s)
            tokens += int(resp.input_tokens or 0) + int(resp.output_tokens or 0)
            raw_last = resp.text
            votes.append(parse_decision(resp.text, default=self.cfg.default_on_parse_error))

        counts = Counter(votes)
        action, n_winner = counts.most_common(1)[0]
        confidence = n_winner / len(votes)

        decision = Decision(
            action=action,
            confidence=confidence,
            latency_s=latency_s,
            tokens=tokens,
            votes=votes,
            raw=raw_last,
        )
        if self.cfg.use_cache:
            self._cache[key] = decision
        return decision

    def clear_cache(self) -> None:
        self._cache.clear()
