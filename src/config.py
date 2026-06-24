"""Configuration dataclasses for the whole project.

Everything that an experiment can tune lives here so the experiment scripts stay
thin and every knob is overridable from the CLI. Nothing in this module imports
``phm_framework`` (or anything heavy) so it is safe to import anywhere.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum
from typing import List, Optional


class ContextLevel(IntEnum):
    """How much context the prompt exposes to the LLM judge.

    L0 — only the partial validation curve.
    L1 — + derived past features (velocity, EMA, improvement vs best).
    L2 — + Bayesian-optimization state (best-so-far, warmup flag, trial count).
    L3 — + task / architecture / hyperparameters (full metadata).
    """

    L0 = 0
    L1 = 1
    L2 = 2
    L3 = 3

    @classmethod
    def parse(cls, value) -> "ContextLevel":
        if isinstance(value, ContextLevel):
            return value
        if isinstance(value, int):
            return cls(value)
        s = str(value).strip().upper()
        if s.startswith("L"):
            s = s[1:]
        return cls(int(s))


@dataclass
class LLMConfig:
    """Transport-level configuration for the LLM backend."""

    backend: str = "ollama"
    model: Optional[str] = None          # None -> taken from LLM_MODEL env
    temperature: float = 0.3
    timeout: int = 180
    max_retries: int = 2


@dataclass
class JudgeConfig:
    """Behaviour of the LLM judge policy on top of the transport."""

    context_level: ContextLevel = ContextLevel.L2
    n_samples: int = 1                   # >1 -> majority vote + confidence
    use_cache: bool = True               # cache by scenario hash
    default_on_parse_error: str = "CONTINUE"   # safe default
    prompt_variant: str = "neutral"      # conservative | neutral | aggressive


@dataclass
class SimConfig:
    """Bayesian-optimization replay + early-stopping harness parameters."""

    checkpoint_epoch: int = 10           # first epoch a policy is queried (0-based)
    checkpoint_frac: float = None        # if set, checkpoint = max(min_checkpoint,
                                         # int(frac * n_epochs)) PER curve, so short and
                                         # long curves both get enough decision points
    min_checkpoint: int = 2              # floor for the adaptive checkpoint
    patience: int = 3                    # consecutive PRUNE votes needed to prune
    warmup_trials: int = 3               # first N trials never pruned (set incumbent)
    n_trials_cap: int = 100              # min(cap, n_units) trials per experiment
    minimize: bool = True                # val loss -> minimize

    # baseline knobs (mirror phmlc defaults)
    random_pct: float = 0.5              # random: P(continue) = random_pct
    es_patience: int = 5                 # classical early-stopping: prune if val has not
                                         # improved (within THIS run) for es_patience epochs
    last_seen_factor: float = 2.0        # prune if pred  >= best_so_far * factor
    arima_factor: float = 2.0            # prune if forecast >= best_so_far * factor
    arima_max_p: int = 3                 # small ARIMA grid (phmlc grid is intractable
    arima_max_d: int = 2                 # per-epoch; we use a bounded search instead)
    arima_max_q: int = 3
    horizon: int = 100                   # forecast horizon used by arima (max epochs)


@dataclass
class ExperimentConfig:
    """Top-level experiment configuration (data selection + run control)."""

    # data selection (passed to the phmlc bridge)
    filters: dict = field(default_factory=lambda: {"data": "curves"})
    test_dataset_names: List[str] = field(default_factory=list)
    random_state: int = 666
    num_folds: int = 5
    fold: int = 0

    # run control
    n_experiments: Optional[int] = None  # None -> all experiments in the split
    spread_datasets: bool = False        # select round-robin across datasets (diversity)
    min_units_per_experiment: int = 2    # skip groups too small to optimize
    seeds: List[int] = field(default_factory=lambda: [0, 1, 2, 3, 4])
    split: str = "test"                  # which load_curves split to evaluate on
    debug: bool = False                  # use a synthetic, offline dataset
    results_dir: str = "results"

    # sub-configs
    llm: LLMConfig = field(default_factory=LLMConfig)
    judge: JudgeConfig = field(default_factory=JudgeConfig)
    sim: SimConfig = field(default_factory=SimConfig)
