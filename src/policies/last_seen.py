"""Last-seen baseline — mirror of phmlc's ``last-seen`` rule (train.py:1809-1817).

phmlc predicts the final loss as the last partial value and prunes when that
prediction is at least twice the running best (``pred >= filter_best * 2``).  Here
``filter_best`` is the BO incumbent (``best_so_far_final_val``); during warmup it is
unknown, so we never prune.
"""

from __future__ import annotations

import numpy as np

from prompting import CONTINUE, PRUNE
from scenario import DecisionScenario
from .base import Decision, Policy


class LastSeenPolicy(Policy):
    name = "last-seen"

    def __init__(self, factor: float = 2.0):
        self.factor = factor

    def decide(self, scenario: DecisionScenario) -> Decision:
        best = scenario.bo_state.get("best_so_far_final_val")
        if best is None or not np.isfinite(best):
            return Decision(action=CONTINUE)
        pred = scenario.partial_val[-1]          # last seen value
        action = PRUNE if pred >= best * self.factor else CONTINUE
        return Decision(action=action)
