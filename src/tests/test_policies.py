"""Baselines reproduce phmlc's rules; oracle behaves as the upper bound."""

import numpy as np

from config import SimConfig
from policies import ArimaPolicy, LastSeenPolicy, OraclePolicy, RandomPolicy
from policies.llm import LLMPolicy
from backend import MockBackend
from config import ContextLevel, JudgeConfig
from scenario import ScenarioBuilder


def _scenario(vals, best, builder=None):
    builder = builder or ScenarioBuilder()
    vals = np.asarray(vals, dtype=float)
    curve = np.stack([np.clip(vals - 0.05, 1e-3, None), vals], axis=1)
    bo = {"best_so_far_final_val": best, "best_val_at_epoch": best}
    return builder.build("u", "exp", curve, len(vals) - 1, bo, {})


def test_last_seen_rule_matches_phmlc():
    p = LastSeenPolicy(factor=2.0)
    # pred = last value = 1.0 ; best 0.4 -> 1.0 >= 0.8 -> PRUNE
    assert p.decide(_scenario([0.9, 1.0], 0.4)).action == "PRUNE"
    # best 0.6 -> 1.0 >= 1.2 is False -> CONTINUE
    assert p.decide(_scenario([0.9, 1.0], 0.6)).action == "CONTINUE"
    # warmup (best unknown) -> CONTINUE
    assert p.decide(_scenario([0.9, 1.0], None)).action == "CONTINUE"


def test_random_is_one_decision_per_unit():
    p = RandomPolicy(random_pct=0.5, seed=1)
    p.new_run(0)
    sc = _scenario([0.9, 0.8, 0.7], 0.5)
    first = p.decide(sc).action
    # querying the same unit again at later epochs returns the same decision
    for _ in range(5):
        assert p.decide(sc).action == first


def test_random_respects_probability_bounds():
    always_continue = RandomPolicy(random_pct=1.0, seed=3)
    always_continue.new_run(0)
    assert always_continue.decide(_scenario([1.0, 1.0], 0.1)).action == "CONTINUE"
    always_prune = RandomPolicy(random_pct=0.0, seed=3)
    always_prune.new_run(0)
    assert always_prune.decide(_scenario([1.0, 1.0], 0.1)).action == "PRUNE"


def test_arima_short_series_falls_back_to_last_seen():
    p = ArimaPolicy(sim=SimConfig())
    # len < 4 -> uses last value as prediction
    assert p.decide(_scenario([0.9, 1.0], 0.4)).action == "PRUNE"
    assert p.decide(_scenario([0.9, 1.0], 0.6)).action == "CONTINUE"


def test_oracle_prunes_only_non_improving():
    from dataset import make_synthetic_experiment
    exp = make_synthetic_experiment(n_units=6, n_epochs=20, seed=0)
    oracle = OraclePolicy()
    oracle.set_experiment(exp)
    uid = exp.unit_ids[0]
    final = float(exp.curves[uid][-1, 1])
    # build scenarios with the real unit id so the oracle can look the curve up
    b = ScenarioBuilder()
    # incumbent strictly better than this run's final -> prune
    bo = {"best_so_far_final_val": final - 0.1, "best_val_at_epoch": final}
    sc = b.build(uid, exp.experiment_id, exp.curves[uid], 12, bo, {})
    assert oracle.decide(sc).action == "PRUNE"
    # incumbent worse than this run's final -> keep
    bo2 = {"best_so_far_final_val": final + 0.1, "best_val_at_epoch": final}
    sc2 = b.build(uid, exp.experiment_id, exp.curves[uid], 12, bo2, {})
    assert oracle.decide(sc2).action == "CONTINUE"


def test_llm_policy_with_mock_majority_and_confidence():
    backend = MockBackend(scripted=["DECISION: PRUNE", "DECISION: PRUNE", "DECISION: CONTINUE"])
    judge = LLMPolicy(backend=backend, config=JudgeConfig(
        context_level=ContextLevel.L1, n_samples=3, use_cache=False))
    d = judge.decide(_scenario([0.9, 0.8, 0.7, 0.7, 0.7], 0.5))
    assert d.action == "PRUNE"
    assert abs(d.confidence - 2 / 3) < 1e-9
    assert d.tokens > 0
