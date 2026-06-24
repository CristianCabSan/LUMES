"""ARIMA baseline — mirror of phmlc's ``arima`` rule (train.py:1599-1632).

phmlc fits an ARIMA to the partial validation curve, forecasts to the (assumed
100-epoch) horizon, and prunes when the forecast final loss is at least twice the
running best (``forecast >= filter_best * 2``).

phmlc's literal grid (``d, q`` up to ``ts_len-1``) is intractable when evaluated
*every epoch*; we keep the same idea but with a small bounded ``(p, d, q)`` grid
selected by AIC, and we select the order **once per run** (first decision) and then
reuse it on later epochs (one fit/epoch).  This is documented as a deliberate
adaptation in the thesis limitations.
"""

from __future__ import annotations

import warnings

import numpy as np

from config import SimConfig
from prompting import CONTINUE, PRUNE
from scenario import DecisionScenario
from .base import Decision, Policy

try:
    from statsmodels.tsa.arima.model import ARIMA
    from statsmodels.tools.sm_exceptions import ConvergenceWarning
    _HAS_STATSMODELS = True
except Exception:  # pragma: no cover - statsmodels missing
    _HAS_STATSMODELS = False
    ConvergenceWarning = Warning


class ArimaPolicy(Policy):
    name = "arima"

    def __init__(self, sim: SimConfig = None, factor: float = None):
        sim = sim or SimConfig()
        self.factor = factor if factor is not None else sim.arima_factor
        self.max_p = sim.arima_max_p
        self.max_d = sim.arima_max_d
        self.max_q = sim.arima_max_q
        self.horizon = sim.horizon
        self._order_cache = {}   # unit_id -> (p, d, q)

    def new_run(self, seed: int) -> None:
        self._order_cache.clear()

    def _select_order(self, series: np.ndarray):
        best_aic = np.inf
        best_order = None
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            for p in range(1, self.max_p + 1):
                for d in range(0, self.max_d + 1):
                    for q in range(1, self.max_q + 1):
                        try:
                            fit = ARIMA(series, order=(p, d, q)).fit()
                            if fit.aic < best_aic:
                                best_aic = fit.aic
                                best_order = (p, d, q)
                        except Exception:
                            continue
        return best_order

    def _forecast_final(self, unit_id: str, series: np.ndarray, steps: int):
        order = self._order_cache.get(unit_id)
        if order is None:
            order = self._select_order(series)
            if order is None:
                return None
            self._order_cache[unit_id] = order
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                fit = ARIMA(series, order=order).fit()
                return float(fit.forecast(steps)[-1])
            except Exception:
                return None

    def decide(self, scenario: DecisionScenario) -> Decision:
        best = scenario.bo_state.get("best_so_far_final_val")
        if best is None or not np.isfinite(best):
            return Decision(action=CONTINUE)

        series = np.asarray(scenario.partial_val, dtype=float)
        steps = max(1, self.horizon - len(series))
        if not _HAS_STATSMODELS or len(series) < 4:
            # not enough signal to fit -> fall back to last-seen prediction
            pred = float(series[-1])
        else:
            pred = self._forecast_final(scenario.unit_id, series, steps)
            if pred is None:
                pred = float(series[-1])

        action = PRUNE if pred >= best * self.factor else CONTINUE
        return Decision(action=action)
