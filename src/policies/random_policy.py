"""Random baseline — mirror of phmlc's ``random`` rule (train.py:1993).

phmlc flips a single coin per run: continue with probability ``random_pct``,
otherwise prune the whole run.  We reproduce the *one decision per run* semantics
inside the per-epoch harness by drawing once per unit and caching it (so the
patience mechanism simply confirms the prune).
"""

from __future__ import annotations

import random
from typing import Dict

from prompting import CONTINUE, PRUNE
from scenario import DecisionScenario
from .base import Decision, Policy


class RandomPolicy(Policy):
    name = "random"

    def __init__(self, random_pct: float = 0.5, seed: int = 0):
        self.random_pct = random_pct
        self._base_seed = seed
        self._rng = random.Random(seed)
        self._cache: Dict[str, str] = {}

    def new_run(self, seed: int) -> None:
        # vary the draws across Monte-Carlo repeats, deterministically
        self._rng = random.Random(self._base_seed ^ (seed * 2654435761 & 0xFFFFFFFF))
        self._cache.clear()

    def decide(self, scenario: DecisionScenario) -> Decision:
        action = self._cache.get(scenario.unit_id)
        if action is None:
            action = CONTINUE if self._rng.uniform(0, 1) < self.random_pct else PRUNE
            self._cache[scenario.unit_id] = action
        return Decision(action=action)
