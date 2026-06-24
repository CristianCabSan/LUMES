"""Policy ABC + the ``Decision`` value object shared by all methods."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Optional

from prompting import CONTINUE, PRUNE
from scenario import DecisionScenario


@dataclass
class Decision:
    """One per-epoch decision plus the bookkeeping the metrics need."""

    action: str                       # CONTINUE | PRUNE
    confidence: float = 1.0           # fraction of agreeing samples (1.0 for non-LLM)
    latency_s: float = 0.0            # wall-clock of the call (0.0 for non-LLM)
    tokens: int = 0                   # total tokens consumed by this decision
    votes: Optional[List[str]] = None # individual samples (LLM, n_samples>1)
    raw: Optional[str] = None         # raw model text (debugging)

    def is_prune(self) -> bool:
        return self.action == PRUNE

    def is_continue(self) -> bool:
        return self.action == CONTINUE


class Policy(ABC):
    """Common interface for every early-stopping method."""

    name: str = "policy"

    @abstractmethod
    def decide(self, scenario: DecisionScenario) -> Decision:
        """Return a :class:`Decision` for the given (past/present) scenario."""

    # ---- optional lifecycle hooks (default: no-op) ---- #
    def set_experiment(self, experiment_data) -> None:
        """Called once per experiment before its trials run.

        The oracle uses this to gain full-curve knowledge; everyone else ignores it.
        """

    def new_run(self, seed: int) -> None:
        """Called at the start of every ``run_once(seed)`` (Monte-Carlo repeat)."""
