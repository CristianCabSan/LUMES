"""Metric collection and aggregation.

One :class:`ExperimentResult` is produced per (policy, experiment, seed).  The
:class:`MetricsCollector` gathers them, emits a clean per-run CSV and computes the
grouped summary the thesis reports:

* **Regret** = ``filter_best_loss - real_best_loss`` (+ ``rank_pct``).
* **Savings Variant A** (no latency): ``epoch_time * epochs_avoided``.
* **Savings Variant B** (with latency): ``A - Σ min(llm_latency, epoch_time)`` over
  prunes.
* **False Continues / False Prunes** vs the per-epoch ground truth.
* **Confidence** (determinism) and **tokens/cost**.
"""

from __future__ import annotations

import math
import os
from dataclasses import asdict, dataclass, field
from typing import Dict, List, Optional

import numpy as np
import pandas as pd


@dataclass
class ExperimentResult:
    policy: str
    experiment_id: str
    seed: int

    # quality
    filter_best_loss: float = math.inf
    real_best_loss: float = math.inf
    regret: float = 0.0
    rank_pct: float = 0.0

    # savings
    num_runs: int = 0
    num_prunings: int = 0
    total_epochs: int = 0
    epochs_avoided: int = 0
    total_train_time: float = 0.0
    saved_time_A: float = 0.0
    saved_time_B: float = 0.0

    # correctness vs per-epoch ground truth
    n_decisions: int = 0
    false_continues: int = 0
    false_prunes: int = 0

    # cost / confidence
    llm_calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    total_latency_s: float = 0.0
    mean_confidence: float = 1.0

    @property
    def saved_pct_A(self) -> float:
        return self.saved_time_A / self.total_train_time if self.total_train_time else 0.0

    @property
    def saved_pct_B(self) -> float:
        return self.saved_time_B / self.total_train_time if self.total_train_time else 0.0

    @property
    def found_best(self) -> bool:
        return abs(self.filter_best_loss - self.real_best_loss) < 1e-6


def save_decisions(rows: list, path: str) -> str:
    """Write a per-decision dump (list of dicts) to CSV for offline reuse."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def _ci95(values: np.ndarray) -> float:
    n = len(values)
    if n < 2:
        return 0.0
    return float(1.96 * np.std(values, ddof=1) / math.sqrt(n))


class MetricsCollector:
    """Accumulates :class:`ExperimentResult` rows and summarises them."""

    def __init__(self):
        self.results: List[ExperimentResult] = []

    def add(self, result: ExperimentResult) -> None:
        self.results.append(result)

    def extend(self, results: List[ExperimentResult]) -> None:
        self.results.extend(results)

    # ---- export ---- #
    def to_dataframe(self) -> pd.DataFrame:
        rows = []
        for r in self.results:
            d = asdict(r)
            d["saved_pct_A"] = r.saved_pct_A
            d["saved_pct_B"] = r.saved_pct_B
            d["found_best"] = r.found_best
            rows.append(d)
        return pd.DataFrame(rows)

    def save_csv(self, path: str) -> str:
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        self.to_dataframe().to_csv(path, index=False)
        return path

    # ---- aggregation ---- #
    def summary(self) -> pd.DataFrame:
        """Per-policy means with 95% CIs and rates."""
        df = self.to_dataframe()
        if df.empty:
            return df
        out = []
        for policy, g in df.groupby("policy"):
            out.append(
                {
                    "policy": policy,
                    "n": len(g),
                    "regret_mean": g["regret"].mean(),
                    "regret_median": g["regret"].median(),
                    "regret_ci95": _ci95(g["regret"].to_numpy()),
                    "rank_pct_mean": g["rank_pct"].mean(),
                    "found_best_rate": g["found_best"].mean(),
                    "saved_pct_A_mean": g["saved_pct_A"].mean(),
                    "saved_pct_A_ci95": _ci95(g["saved_pct_A"].to_numpy()),
                    "saved_pct_B_mean": g["saved_pct_B"].mean(),
                    "saved_pct_B_ci95": _ci95(g["saved_pct_B"].to_numpy()),
                    "epochs_avoided_mean": g["epochs_avoided"].mean(),
                    "false_continue_rate": (
                        g["false_continues"].sum() / max(1, g["n_decisions"].sum())
                    ),
                    "false_prune_rate": (
                        g["false_prunes"].sum() / max(1, g["n_decisions"].sum())
                    ),
                    "prune_rate": g["num_prunings"].sum() / max(1, g["num_runs"].sum()),
                    "mean_confidence": g["mean_confidence"].mean(),
                    "total_tokens": (g["input_tokens"] + g["output_tokens"]).sum(),
                    "total_latency_s": g["total_latency_s"].sum(),
                }
            )
        return pd.DataFrame(out).sort_values("policy").reset_index(drop=True)

    def save_summary_csv(self, path: str) -> str:
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        self.summary().to_csv(path, index=False)
        return path
