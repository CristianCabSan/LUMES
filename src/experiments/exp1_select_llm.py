"""Experiment 1 — model + context-level selection (Fase 5.1).

Sweeps several LLMs (``--models``) and context levels (``--levels``) over a *small*
subset of experiments and ranks (model, level) configs by a combined score
(high Variant-B savings, low regret).  Smoke runs use a single experiment.

Examples
--------
    # offline smoke (1 experiment, MockBackend):
    python experiments/exp1_select_llm.py --debug

    # real sweep over a reduced subset (configure the model list to Ollamus):
    python experiments/exp1_select_llm.py \
        --models llama3.1:8b,mistral:7b,qwen2.5:7b \
        --levels 0,1,2,3 --n-experiments 5 --seeds 0,1,2
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np
import pandas as pd

from _cli import add_common_args, config_from_args  # noqa: E402

from config import ContextLevel, JudgeConfig
from metrics import MetricsCollector
from policies.llm import LLMPolicy
from runner import load_experiments, make_backend, run_policy


def _minmax(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    rng = x.max() - x.min()
    return np.zeros_like(x) if rng == 0 else (x - x.min()) / rng


def score_summary(summary: pd.DataFrame, w_save: float, w_regret: float) -> pd.DataFrame:
    """Rank configs by time saved, penalised by regret — in natural units.

    ``saved_pct_B`` is a 0-1 fraction of train time; ``regret`` is the val-loss the
    search gave up. Using the raw values avoids the min-max pitfall where the
    *smallest* non-zero regret receives the *full* penalty, which (in v5) let a
    slow, lower-saving regret-0 model beat much higher-saving configs that had only
    a negligible 0.01 regret. Increase ``--w-regret`` to be stricter about regret.
    """
    s = summary.copy()
    s["score"] = w_save * s["saved_pct_B_mean"] - w_regret * s["regret_mean"]
    return s.sort_values("score", ascending=False).reset_index(drop=True)


def probe_backend(backend) -> tuple:
    """Cheap availability check: one tiny call. Returns (ok, detail)."""
    try:
        r = backend.chat("You are a judge.", "Reply with exactly: DECISION: CONTINUE")
        return True, f"{r.latency_s:.1f}s"
    except Exception as e:  # model absent / auth / network
        return False, f"{type(e).__name__}: {str(e)[:80]}"


def run_sweep(models, levels, experiments, cfg, decision_sink=None):
    """Resilient sweep over models x levels.

    Probes each model first; **skips any model that errors** (absent/unavailable)
    and continues with the next.  A model that fails mid-run has its partial rows
    dropped.  When ``decision_sink`` is a list, every per-epoch decision is appended
    to it for offline reuse.  Returns (collector, name_to_config, working, failed).
    """
    collector = MetricsCollector()
    name_to_cfg = {}
    working, failed = [], []
    for model in models:
        backend = make_backend(cfg, model=model)
        ok, detail = probe_backend(backend)
        if not ok:
            print(f"  [skip] {model}: {detail}")
            failed.append(model)
            continue
        print(f"  [ok]   {model} (probe {detail})")
        model_rows = []
        model_decisions = []
        try:
            for level in levels:
                judge = JudgeConfig(context_level=level, n_samples=cfg.judge.n_samples,
                                    use_cache=cfg.judge.use_cache,
                                    prompt_variant=cfg.judge.prompt_variant)
                policy = LLMPolicy(backend=backend, config=judge)
                mc = MetricsCollector()
                sink = [] if decision_sink is not None else None
                run_policy(policy, experiments, cfg, mc, decision_sink=sink)
                model_rows.extend(mc.results)
                if sink is not None:
                    model_decisions.extend(sink)
                for r in mc.results:
                    name_to_cfg[r.policy] = (model, int(level))
                print(f"     done {policy.name}")
        except Exception as e:
            print(f"  [skip] {model} failed mid-run ({type(e).__name__}: "
                  f"{str(e)[:80]}); dropping partial results")
            failed.append(model)
            continue
        collector.extend(model_rows)
        if decision_sink is not None:
            decision_sink.extend(model_decisions)
        working.append(model)
    return collector, name_to_cfg, working, failed


def main():
    ap = argparse.ArgumentParser(description="exp1: select LLM + context level")
    add_common_args(ap)
    ap.add_argument("--models", type=str, default=None,
                    help="comma-separated model ids to sweep (default: $LLM_MODEL)")
    ap.add_argument("--levels", type=str, default="0,1,2,3",
                    help="comma-separated context levels to sweep")
    ap.add_argument("--w-save", type=float, default=1.0)
    ap.add_argument("--w-regret", type=float, default=1.0)
    args = ap.parse_args()

    cfg = config_from_args(args)
    models = [m.strip() for m in args.models.split(",")] if args.models else [cfg.llm.model or "mock"]
    levels = [ContextLevel.parse(x) for x in args.levels.split(",")]

    experiments = load_experiments(cfg)
    print(f"[exp1] {len(experiments)} experiment(s), models={models}, "
          f"levels={[int(l) for l in levels]}, seeds={cfg.seeds}")

    collector, name_to_cfg, working, failed = run_sweep(models, levels, experiments, cfg)
    print(f"[exp1] working models: {working} | skipped: {failed}")
    if not collector.results:
        print("[exp1] no working models produced results; aborting.")
        return

    os.makedirs(cfg.results_dir, exist_ok=True)
    collector.save_csv(os.path.join(cfg.results_dir, "exp1_runs.csv"))
    summary = collector.summary()
    ranked = score_summary(summary, args.w_save, args.w_regret)
    ranked.to_csv(os.path.join(cfg.results_dir, "exp1_summary.csv"), index=False)

    cols = ["policy", "score", "regret_mean", "saved_pct_B_mean",
            "false_prune_rate", "false_continue_rate", "mean_confidence"]
    print("\n=== exp1 ranking ===")
    print(ranked[cols].to_string(index=False))
    best = ranked.iloc[0]["policy"]
    best_model, best_level = name_to_cfg.get(best, (None, None))
    with open(os.path.join(cfg.results_dir, "exp1_best.json"), "w") as f:
        json.dump({"policy": best, "model": best_model, "context_level": best_level,
                   "working": working, "failed": failed}, f, indent=2)
    print(f"\nBest config: {best}  (model={best_model}, level=L{best_level})")


if __name__ == "__main__":
    main()
