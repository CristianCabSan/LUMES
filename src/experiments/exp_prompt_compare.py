"""Prompt-variant comparison (the v6 study).

Runs the LLM judge under 3 system-prompt variants -- conservative / neutral /
aggressive -- on a FIXED model+context-level (so the prompt is isolated), against
the baselines (random, last-seen, arima, classical early-stopping) + oracle, over a
bigger, more diverse set of experiments.  Everything is saved for offline analysis:
per-run CSVs, per-decision dumps, paired significance tests, the exact prompt texts,
and the run config.

    python experiments/exp_prompt_compare.py --debug            # offline smoke
    python experiments/exp_prompt_compare.py --model gemma3:12b --context-level L2 \
        --n-experiments 10 --seeds 0,1,2 --es-patience 5
"""

from __future__ import annotations

import argparse
import json
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_SRC = os.path.dirname(_HERE)
for p in (_SRC, _HERE):
    if p not in sys.path:
        sys.path.insert(0, p)

import pandas as pd  # noqa: E402

from config import (  # noqa: E402
    ContextLevel, ExperimentConfig, JudgeConfig, LLMConfig, SimConfig,
)
from metrics import MetricsCollector, save_decisions  # noqa: E402
from policies.llm import LLMPolicy  # noqa: E402
from prompting import PROMPTS  # noqa: E402
from runner import load_experiments, make_backend, make_policy, run_policy  # noqa: E402

from exp1_select_llm import probe_backend  # noqa: E402
from exp2_vs_baselines import paired_tests  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description="3-prompt-variant comparison vs baselines")
    ap.add_argument("--debug", action="store_true")
    ap.add_argument("--model", type=str, default=None)
    ap.add_argument("--context-level", type=str, default="L2")
    ap.add_argument("--variants", type=str, default="conservative,neutral,aggressive")
    ap.add_argument("--baselines", type=str,
                    default="random,last-seen,arima,early-stopping,oracle")
    ap.add_argument("--n-experiments", type=int, default=10)
    ap.add_argument("--spread-datasets", action="store_true",
                    help="select experiments round-robin across datasets (diversity)")
    ap.add_argument("--seeds", type=str, default="0,1,2")
    ap.add_argument("--split", type=str, default="test")
    ap.add_argument("--min-units", type=int, default=2)
    ap.add_argument("--random-state", type=int, default=666)
    ap.add_argument("--trials", type=int, default=12)
    ap.add_argument("--checkpoint", type=int, default=3)
    ap.add_argument("--patience", type=int, default=2)
    ap.add_argument("--warmup-trials", type=int, default=3)
    ap.add_argument("--es-patience", type=int, default=5)
    ap.add_argument("--temperature", type=float, default=0.3)
    ap.add_argument("--timeout", type=int, default=120)
    ap.add_argument("--run-name", type=str, default=None,
                    help="bundle ALL artifacts under results/runs/<name>/ (recommended)")
    ap.add_argument("--results-dir", type=str, default="results")
    ap.add_argument("--full-trajectory", action="store_true",
                    help="dump every epoch past the effective stop (Nx more LLM calls)")
    args = ap.parse_args()

    seeds = [int(x) for x in args.seeds.split(",") if x.strip() != ""]
    level = ContextLevel.parse(args.context_level)
    variants = [v.strip() for v in args.variants.split(",") if v.strip()]
    baselines = [b.strip() for b in args.baselines.split(",") if b.strip()]
    from _cli import tee_to  # noqa: E402
    if args.run_name:
        args.results_dir = os.path.join("results", "runs", args.run_name)
    os.makedirs(args.results_dir, exist_ok=True)
    tee_to(os.path.join(args.results_dir, "run.log"))

    cfg = ExperimentConfig(
        n_experiments=args.n_experiments, spread_datasets=args.spread_datasets,
        seeds=seeds, split=args.split, debug=args.debug,
        min_units_per_experiment=args.min_units, random_state=args.random_state,
        results_dir=args.results_dir,
        llm=LLMConfig(backend=("mock" if args.debug else "ollama"),
                      model=args.model, temperature=args.temperature, timeout=args.timeout),
        sim=SimConfig(checkpoint_epoch=args.checkpoint, patience=args.patience,
                      warmup_trials=args.warmup_trials, n_trials_cap=args.trials,
                      es_patience=args.es_patience),
    )

    # --- save the exact prompts + config (for the writeup / reproducibility) ---
    pdir = os.path.join(args.results_dir, "prompts")
    os.makedirs(pdir, exist_ok=True)
    for v in variants:
        with open(os.path.join(pdir, f"{v}.txt"), "w", encoding="utf-8") as f:
            f.write(PROMPTS[v])
    with open(os.path.join(args.results_dir, "compare_config.json"), "w") as f:
        json.dump({"model": args.model, "context_level": int(level), "variants": variants,
                   "baselines": baselines, "n_experiments": args.n_experiments,
                   "seeds": seeds, "trials": args.trials, "checkpoint": args.checkpoint,
                   "patience": args.patience, "warmup_trials": args.warmup_trials,
                   "es_patience": args.es_patience, "full_trajectory": args.full_trajectory},
                  f, indent=2)

    experiments = load_experiments(cfg)
    print(f"[compare] {len(experiments)} experiments: {[e.experiment_id for e in experiments]}")
    print(f"[compare] model={args.model} level=L{int(level)} variants={variants} "
          f"baselines={baselines} seeds={seeds}")
    if not experiments:
        print("no experiments; aborting."); return

    collector = MetricsCollector()
    decisions = []

    # --- the 3 LLM prompt variants (fixed model+level) ---
    backend = make_backend(cfg, model=args.model)
    if not args.debug:
        ok, detail = probe_backend(backend)
        if not ok:
            print(f"[compare] model {args.model} unavailable: {detail}; aborting."); return
        print(f"[compare] model probe ok ({detail})")
    for v in variants:
        judge = JudgeConfig(context_level=level, n_samples=1, use_cache=True, prompt_variant=v)
        policy = LLMPolicy(backend=backend, config=judge)
        run_policy(policy, experiments, cfg, collector, decision_sink=decisions,
                   full_trajectory=args.full_trajectory, verbose=True)
        print(f"  done {policy.name}")

    # --- baselines + oracle ---
    for name in baselines:
        policy = make_policy(name, cfg)
        run_policy(policy, experiments, cfg, collector, decision_sink=decisions,
                   full_trajectory=args.full_trajectory, verbose=True)
        print(f"  done {policy.name}")

    # --- save everything ---
    collector.save_csv(os.path.join(args.results_dir, "compare_runs.csv"))
    summary = collector.summary()
    summary.to_csv(os.path.join(args.results_dir, "compare_summary.csv"), index=False)
    if decisions:
        save_decisions(decisions, os.path.join(args.results_dir, "raw", "compare_decisions.csv"))

    df = collector.to_dataframe()
    tests = []
    llm_names = [n for n in df.policy.unique() if n.startswith("llm:")]
    for ln in llm_names:
        for metric in ("regret", "saved_pct_B"):
            tests.append(paired_tests(df, ln, metric))
    if tests:
        pd.concat(tests, ignore_index=True).to_csv(
            os.path.join(args.results_dir, "compare_significance.csv"), index=False)

    cols = ["policy", "n", "regret_mean", "regret_ci95", "saved_pct_A_mean",
            "saved_pct_B_mean", "false_prune_rate", "false_continue_rate",
            "found_best_rate", "prune_rate"]
    print("\n=== prompt-variant comparison ===")
    print(summary[cols].to_string(index=False))
    print(f"\nresults in {args.results_dir}/ (compare_*.csv, prompts/, raw/compare_decisions.csv)")


if __name__ == "__main__":
    main()
