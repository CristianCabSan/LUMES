"""Experiment 2 — best LLM vs baselines + oracle (Fase 5.2).

Runs the chosen LLM (``--model`` / ``--context-level``) against
random / last-seen / arima / oracle over a *large* set of experiments with N
Monte-Carlo seeds, then writes per-run + summary CSVs and paired significance
tests (LLM vs each baseline, by experiment+seed).

Examples
--------
    python experiments/exp2_vs_baselines.py --debug                 # offline e2e
    python experiments/exp2_vs_baselines.py --model llama3.1:8b \
        --context-level L2 --seeds 0,1,2,3,4         # all test experiments
"""

from __future__ import annotations

import argparse
import os

import pandas as pd

from _cli import add_common_args, config_from_args  # noqa: E402

from metrics import MetricsCollector
from runner import load_experiments, make_backend, make_policy, run_policy

try:
    from scipy.stats import wilcoxon
    _HAS_SCIPY = True
except Exception:
    _HAS_SCIPY = False


def paired_tests(df: pd.DataFrame, llm_name: str, metric: str) -> pd.DataFrame:
    """Paired Wilcoxon of the LLM vs every other policy on ``metric``."""
    if not _HAS_SCIPY:
        return pd.DataFrame()
    key = ["experiment_id", "seed"]
    wide = df.pivot_table(index=key, columns="policy", values=metric)
    rows = []
    if llm_name not in wide.columns:
        return pd.DataFrame()
    for other in wide.columns:
        if other == llm_name:
            continue
        pair = wide[[llm_name, other]].dropna()
        if len(pair) < 3 or (pair[llm_name] - pair[other]).abs().sum() == 0:
            stat, p = float("nan"), float("nan")
        else:
            stat, p = wilcoxon(pair[llm_name], pair[other])
        rows.append({
            "metric": metric, "llm": llm_name, "vs": other, "n_pairs": len(pair),
            "llm_mean": pair[llm_name].mean(), "other_mean": pair[other].mean(),
            "wilcoxon_stat": stat, "p_value": p,
        })
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser(description="exp2: LLM vs baselines + oracle")
    add_common_args(ap)
    ap.add_argument("--baselines", type=str, default="random,last-seen,arima,oracle",
                    help="comma-separated baseline policies to compare against")
    args = ap.parse_args()

    cfg = config_from_args(args)
    experiments = load_experiments(cfg)
    print(f"[exp2] {len(experiments)} experiment(s), seeds={cfg.seeds}, "
          f"model={cfg.llm.model}, level=L{int(cfg.judge.context_level)}")

    collector = MetricsCollector()

    # the LLM judge
    backend = make_backend(cfg)
    llm = make_policy("llm", cfg, backend=backend)
    run_policy(llm, experiments, cfg, collector)
    print(f"  done {llm.name}")

    # baselines + oracle
    for name in [b.strip() for b in args.baselines.split(",") if b.strip()]:
        policy = make_policy(name, cfg)
        run_policy(policy, experiments, cfg, collector)
        print(f"  done {policy.name}")

    os.makedirs(cfg.results_dir, exist_ok=True)
    runs_path = collector.save_csv(os.path.join(cfg.results_dir, "exp2_runs.csv"))
    summary = collector.summary()
    summary.to_csv(os.path.join(cfg.results_dir, "exp2_summary.csv"), index=False)

    df = collector.to_dataframe()
    tests = pd.concat(
        [paired_tests(df, llm.name, "regret"),
         paired_tests(df, llm.name, "saved_pct_B")],
        ignore_index=True,
    )
    if not tests.empty:
        tests.to_csv(os.path.join(cfg.results_dir, "exp2_significance.csv"), index=False)

    print("\n=== exp2 summary ===")
    cols = ["policy", "regret_mean", "regret_ci95", "saved_pct_A_mean",
            "saved_pct_B_mean", "false_prune_rate", "false_continue_rate",
            "found_best_rate", "prune_rate"]
    print(summary[cols].to_string(index=False))
    if not tests.empty:
        print("\n=== paired significance (LLM vs baseline) ===")
        print(tests.to_string(index=False))
    print(f"\nrows -> {runs_path}")


if __name__ == "__main__":
    main()
