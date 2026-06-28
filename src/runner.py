"""Glue helpers shared by the experiment scripts.

Keeps the ``experiments/*.py`` scripts thin: data loading (real vs synthetic),
backend/policy factories, and a single run loop that drives a set of policies
through the simulator and returns a populated :class:`MetricsCollector`.
"""

from __future__ import annotations

from typing import List, Optional

from backend import Backend, MockBackend, build_backend_from_env
from config import ExperimentConfig, JudgeConfig
from dataset import build_experiments_from_phmlc, make_synthetic_experiments
from metrics import MetricsCollector
from policies import (
    ArimaPolicy, EarlyStoppingPolicy, LastSeenPolicy, LLMPolicy, OraclePolicy,
    Policy, RandomPolicy,
)
from simulator import BOSimulator, ExperimentData

try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:  # pragma: no cover
    pass


# --------------------------------------------------------------------------- #
def load_experiments(cfg: ExperimentConfig) -> List[ExperimentData]:
    """Real CURVES experiments, or synthetic ones when ``cfg.debug``."""
    if cfg.debug:
        n = cfg.n_experiments or 1
        return make_synthetic_experiments(n_experiments=n, seed=cfg.random_state)

    import phmlc_bridge  # lazy: pulls TensorFlow
    sets = phmlc_bridge.load_experiment_curves(
        fold=cfg.fold, num_folds=cfg.num_folds, filters=cfg.filters,
        test_dataset_names=cfg.test_dataset_names, random_state=cfg.random_state,
    )
    curves_df = sets[cfg.split]
    results_df = phmlc_bridge.load_results(random_state=cfg.random_state)
    return build_experiments_from_phmlc(
        curves_df, results_df,
        min_units=cfg.min_units_per_experiment, n_experiments=cfg.n_experiments,
        spread_datasets=cfg.spread_datasets,
    )


def make_backend(cfg: ExperimentConfig, model: Optional[str] = None) -> Backend:
    if cfg.debug or cfg.llm.backend == "mock":
        return MockBackend(model=model or cfg.llm.model or "mock")
    return build_backend_from_env(
        backend=cfg.llm.backend, model=model or cfg.llm.model,
        temperature=cfg.llm.temperature, timeout=cfg.llm.timeout,
    )


def make_policy(
    name: str, cfg: ExperimentConfig, backend: Optional[Backend] = None,
    judge: Optional[JudgeConfig] = None,
) -> Policy:
    name = name.lower()
    if name in ("llm", "judge"):
        return LLMPolicy(backend=backend or make_backend(cfg), config=judge or cfg.judge)
    if name == "random":
        return RandomPolicy(random_pct=cfg.sim.random_pct, seed=cfg.random_state)
    if name in ("last-seen", "last_seen", "lastseen"):
        return LastSeenPolicy(factor=cfg.sim.last_seen_factor)
    if name == "arima":
        return ArimaPolicy(sim=cfg.sim)
    if name in ("early-stopping", "early_stopping", "es", "classical-es"):
        return EarlyStoppingPolicy(es_patience=cfg.sim.es_patience)
    if name == "oracle":
        return OraclePolicy()
    raise ValueError(f"Unknown policy '{name}'")


def run_policy(
    policy: Policy, experiments: List[ExperimentData], cfg: ExperimentConfig,
    collector: Optional[MetricsCollector] = None,
    decision_sink: Optional[list] = None,
    full_trajectory: bool = False,
    verbose: bool = False,
) -> MetricsCollector:
    collector = collector or MetricsCollector()
    sim = BOSimulator(policy=policy, sim=cfg.sim,
                      dump_decisions=decision_sink is not None,
                      full_trajectory=full_trajectory)
    if verbose:
        # heartbeat: print after each (experiment, seed) so long silent LLM phases
        # show liveness/progress.
        n = len(experiments) * len(cfg.seeds)
        i = 0
        for exp in experiments:
            for seed in cfg.seeds:
                r = sim.run_once(exp, seed)
                collector.add(r)
                i += 1
                print(f"     [{policy.name}] {i}/{n} {exp.experiment_id} seed{seed}: "
                      f"regret={r.regret:.3f} savedB={r.saved_pct_B:.3f} "
                      f"prune={r.num_prunings}/{r.num_runs}", flush=True)
    else:
        collector.extend(sim.run_all_experiments(experiments, cfg.seeds))
    if decision_sink is not None:
        decision_sink.extend(sim.decision_log)
    return collector


def run_policies(
    names: List[str], experiments: List[ExperimentData], cfg: ExperimentConfig,
    backend: Optional[Backend] = None, judge: Optional[JudgeConfig] = None,
) -> MetricsCollector:
    collector = MetricsCollector()
    for name in names:
        policy = make_policy(name, cfg, backend=backend, judge=judge)
        run_policy(policy, experiments, cfg, collector)
    return collector
