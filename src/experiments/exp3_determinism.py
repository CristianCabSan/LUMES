"""Experiment 3 — determinism / confidence of a single LLM (Fase 5.3).

Picks a set of decision points (from synthetic or real curves), queries the LLM
``--repeats`` times per point with caching OFF, and reports the agreement
(confidence) distribution and the fraction of perfectly deterministic points.

Examples
--------
    python experiments/exp3_determinism.py --debug --repeats 5
    python experiments/exp3_determinism.py --model llama3.1:8b \
        --context-level L2 --repeats 10 --n-scenarios 50
"""

from __future__ import annotations

import argparse
import os
from collections import Counter

import numpy as np
import pandas as pd

from _cli import add_common_args, config_from_args  # noqa: E402

from config import JudgeConfig
from policies.llm import LLMPolicy
from prompting import parse_decision
from runner import load_experiments, make_backend
from scenario import ScenarioBuilder


def sample_scenarios(experiments, cfg, n_scenarios: int, rng):
    """Build a set of (past/present) scenarios from mid-training epochs."""
    builder = ScenarioBuilder()
    scenarios = []
    for exp in experiments:
        for uid, curve in exp.curves.items():
            n_ep = curve.shape[0]
            if cfg.sim.checkpoint_frac:
                cp = max(cfg.sim.min_checkpoint, int(cfg.sim.checkpoint_frac * n_ep))
            else:
                cp = cfg.sim.checkpoint_epoch
            if n_ep <= cp + 1:
                continue
            e = int(rng.integers(cp, n_ep))
            best = float(min(c[-1, 1] for c in exp.curves.values()))
            bo_state = {"best_so_far_final_val": best, "best_val_at_epoch": best,
                        "n_trials_done": 5, "in_warmup": False}
            scenarios.append(builder.build(uid, exp.experiment_id, curve, e, bo_state,
                                           exp.meta.get(uid, {})))
    rng.shuffle(scenarios)
    return scenarios[:n_scenarios]


def main():
    ap = argparse.ArgumentParser(description="exp3: determinism / confidence")
    add_common_args(ap)
    ap.add_argument("--repeats", type=int, default=5, help="queries per scenario")
    ap.add_argument("--n-scenarios", type=int, default=30)
    args = ap.parse_args()

    cfg = config_from_args(args)
    rng = np.random.default_rng(cfg.random_state)
    experiments = load_experiments(cfg)
    scenarios = sample_scenarios(experiments, cfg, args.n_scenarios, rng)
    print(f"[exp3] {len(scenarios)} scenarios x {args.repeats} repeats, "
          f"model={cfg.llm.model}, level=L{int(cfg.judge.context_level)}")

    backend = make_backend(cfg)
    judge = LLMPolicy(backend=backend, config=JudgeConfig(
        context_level=cfg.judge.context_level, n_samples=1, use_cache=False))

    rows = []
    for i, sc in enumerate(scenarios):
        system, user = judge.prompts.build(sc, cfg.judge.context_level)
        votes = [parse_decision(backend.chat(system, user).text) for _ in range(args.repeats)]
        counts = Counter(votes)
        action, n_win = counts.most_common(1)[0]
        rows.append({
            "scenario": i, "unit_id": sc.unit_id, "epoch": sc.epoch,
            "majority": action, "confidence": n_win / len(votes),
            "deterministic": len(counts) == 1,
            "n_prune": counts.get("PRUNE", 0), "n_continue": counts.get("CONTINUE", 0),
        })

    df = pd.DataFrame(rows)
    os.makedirs(cfg.results_dir, exist_ok=True)
    df.to_csv(os.path.join(cfg.results_dir, "exp3_determinism.csv"), index=False)

    print("\n=== exp3 determinism ===")
    if not df.empty:
        print(f"mean confidence      : {df['confidence'].mean():.3f}")
        print(f"deterministic points : {df['deterministic'].mean():.1%}")
        print(f"min confidence       : {df['confidence'].min():.3f}")
    print(f"rows -> {os.path.join(cfg.results_dir, 'exp3_determinism.csv')}")


if __name__ == "__main__":
    main()
