"""Oracle policy — the upper bound (regret 0, maximum savings).

The oracle is the one deliberate exception to the no-leak rule: it is given the
full curves via :meth:`set_experiment` and reads each run's *true final loss*.

Rule (per epoch, after warmup): a run is worth keeping iff its true final loss
would improve the current BO incumbent; otherwise it can be pruned with zero
regret, because it would never have changed ``filter_best``.

* Runs that would improve the incumbent always CONTINUE -> they complete, so
  ``filter_best`` equals ``real_best`` (regret 0).
* Every non-improving run PRUNEs as early as the harness allows (checkpoint +
  patience) -> maximum achievable savings within this harness.
"""

from __future__ import annotations

import numpy as np

import phmlc_bridge
from prompting import CONTINUE, PRUNE
from scenario import DecisionScenario
from .base import Decision, Policy


class OraclePolicy(Policy):
    name = "oracle"

    def __init__(self):
        self._final_val = {}   # unit_id -> true final val loss

    def set_experiment(self, experiment_data) -> None:
        self._final_val = {
            uid: float(curve[-1, phmlc_bridge.VAL_COL])
            for uid, curve in experiment_data.curves.items()
        }

    def decide(self, scenario: DecisionScenario) -> Decision:
        best = scenario.bo_state.get("best_so_far_final_val")
        if best is None or not np.isfinite(best):
            return Decision(action=CONTINUE)   # warmup: never prune

        final = self._final_val.get(scenario.unit_id)
        if final is None:
            return Decision(action=CONTINUE)
        action = CONTINUE if final < best else PRUNE
        return Decision(action=action)
