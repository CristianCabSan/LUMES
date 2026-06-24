"""``DecisionScenario`` — the no-leak information bundle every policy sees.

The whole fairness/validity argument of the TFM rests on one invariant: **every
policy (LLM, baselines, oracle alike) receives exactly the information in a
``DecisionScenario``, and that information is strictly past/present.**  Nothing in
here may depend on ``val[e+1:]``, the final loss, or the real ``num_epochs``.

``ScenarioBuilder`` is the only sanctioned way to build one: it slices the curve to
``[:epoch+1]`` *internally* and never stores the tail, so leakage is impossible by
construction.  (The oracle is the single, deliberate exception and does not go
through this builder — it reads full curves directly inside its policy.)
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple

import phmlc_bridge


@dataclass(frozen=True)
class DecisionScenario:
    """Immutable, strictly past/present view of one BO trial at one epoch."""

    unit_id: str
    experiment_id: str
    epoch: int                       # 0-based index of the current epoch
    partial_train: Tuple[float, ...] # train_loss[:epoch+1]
    partial_val: Tuple[float, ...]   # val_loss[:epoch+1]
    features: Dict[str, float] = field(default_factory=dict)
    bo_state: Dict[str, object] = field(default_factory=dict)
    meta: Dict[str, object] = field(default_factory=dict)

    def cache_key(self) -> str:
        """Stable content hash (rounded) for caching identical queries."""
        payload = {
            "val": [round(v, 6) for v in self.partial_val],
            "epoch": self.epoch,
            "bo": {
                k: (round(v, 6) if isinstance(v, float) else v)
                for k, v in sorted(self.bo_state.items())
            },
            "meta": {k: self.meta[k] for k in sorted(self.meta)},
        }
        blob = json.dumps(payload, sort_keys=True, default=str)
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()


class ScenarioBuilder:
    """Builds leakage-free ``DecisionScenario`` objects from full curves.

    The builder receives the *full* curve (because the simulator owns it) but only
    ever exposes ``[:epoch+1]``.  This keeps the slicing logic in exactly one place
    and lets the no-leak test assert on a single seam.
    """

    def build(
        self,
        unit_id: str,
        experiment_id: str,
        full_curve,            # ndarray(n, 2): col0=train, col1=val
        epoch: int,
        bo_state: Optional[dict] = None,
        meta: Optional[dict] = None,
    ) -> DecisionScenario:
        bo_state = dict(bo_state or {})
        meta = dict(meta or {})

        # ---- the one and only place the curve is truncated ----
        train_partial = [float(x) for x in full_curve[: epoch + 1, phmlc_bridge.TRAIN_COL]]
        val_partial = [float(x) for x in full_curve[: epoch + 1, phmlc_bridge.VAL_COL]]

        features = phmlc_bridge.build_past_features(
            val_partial=val_partial,
            train_partial=train_partial,
            best_performance=bo_state.get("best_so_far_final_val"),
            best_val_at_epoch=bo_state.get("best_val_at_epoch"),
        )

        return DecisionScenario(
            unit_id=str(unit_id),
            experiment_id=str(experiment_id),
            epoch=int(epoch),
            partial_train=tuple(train_partial),
            partial_val=tuple(val_partial),
            features=features,
            bo_state=bo_state,
            meta=meta,
        )
