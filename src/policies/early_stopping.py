"""Classical early-stopping baseline (intra-run plateau detection).

Unlike the other baselines (which prune *relative to the BO incumbent*), this is
textbook early stopping: stop a run when its **own** validation loss has not improved
for ``es_patience`` epochs — no reference to other configurations at all.

It uses the past-only feature ``epochs_since_best`` (already computed in
``phmlc_bridge.build_past_features``), so it is strictly causal. It runs inside the
same harness as everything else; note the simulator's own ``patience`` (consecutive
PRUNE votes) applies on top, so the effective stop is a couple of epochs after the
plateau is first detected — fair, since every method shares that harness.
"""

from __future__ import annotations

from prompting import CONTINUE, PRUNE
from scenario import DecisionScenario
from .base import Decision, Policy


class EarlyStoppingPolicy(Policy):
    name = "early-stopping"

    def __init__(self, es_patience: int = 5):
        self.es_patience = es_patience

    def decide(self, scenario: DecisionScenario) -> Decision:
        # epochs since this run's best (lowest) validation value so far
        since_best = scenario.features.get("epochs_since_best")
        if since_best is None:
            # fall back to computing it from the partial curve
            vals = scenario.partial_val
            best_idx = min(range(len(vals)), key=lambda i: vals[i])
            since_best = (len(vals) - 1) - best_idx
        action = PRUNE if since_best >= self.es_patience else CONTINUE
        return Decision(action=action)
