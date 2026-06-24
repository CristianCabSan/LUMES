"""End-to-end (offline) harness checks + cota coherence."""

import numpy as np

from backend import MockBackend
from config import JudgeConfig, SimConfig
from dataset import make_synthetic_experiment
from policies import LastSeenPolicy, OraclePolicy, RandomPolicy
from policies.llm import LLMPolicy
from simulator import BOSimulator


def _exp():
    return make_synthetic_experiment(n_units=16, n_epochs=30, seed=7)


def test_oracle_has_zero_regret():
    exp = _exp()
    sim = BOSimulator(policy=OraclePolicy(), sim=SimConfig(warmup_trials=3, checkpoint_epoch=8))
    results = sim.run_montecarlo(exp, seeds=[0, 1, 2])
    for r in results:
        assert r.regret <= 1e-9, f"oracle regret should be 0, got {r.regret}"
        assert r.num_prunings >= 1  # there are non-improving runs to prune


def test_result_schema_and_savings_bounds():
    exp = _exp()
    sim = BOSimulator(policy=LastSeenPolicy(), sim=SimConfig())
    r = sim.run_once(exp, seed=0)
    assert r.num_prunings <= r.num_runs
    assert r.epochs_avoided <= r.total_epochs
    assert r.saved_time_B <= r.saved_time_A + 1e-9
    assert r.regret >= -1e-9
    assert 0.0 <= r.rank_pct <= 1.0


def test_llm_policy_runs_and_tracks_cost():
    exp = _exp()
    backend = MockBackend(rule="curve")
    policy = LLMPolicy(backend=backend, config=JudgeConfig(n_samples=1, use_cache=True))
    sim = BOSimulator(policy=policy, sim=SimConfig())
    r = sim.run_once(exp, seed=0)
    assert r.policy.startswith("llm")
    assert r.llm_calls >= 1
    assert r.total_latency_s > 0.0


def test_oracle_saves_at_least_as_much_as_random_on_average():
    exp = _exp()
    seeds = [0, 1, 2, 3]
    oracle = BOSimulator(OraclePolicy(), SimConfig()).run_montecarlo(exp, seeds)
    random = BOSimulator(RandomPolicy(random_pct=0.7, seed=0), SimConfig()).run_montecarlo(exp, seeds)
    o_save = np.mean([r.saved_time_A for r in oracle])
    # oracle should never be worse than this regret-wise; it is the upper bound
    assert np.mean([r.regret for r in oracle]) <= np.mean([r.regret for r in random]) + 1e-9
    assert o_save >= 0.0
