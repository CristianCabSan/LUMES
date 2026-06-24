"""Run the full study end-to-end: exp1 (select) -> exp2 (vs baselines) -> exp3.

Loads the real CURVES data **once** and reuses it across the three studies.  The
model sweep is error-proof: unavailable models are probed and skipped.  Every phase
is independently sized via CLI flags; defaults are tuned to the measured Ollamus
latencies (≈7–27 s/call) so the whole pipeline finishes in a bounded time.  Scale
any phase up with its flags.

    python experiments/run_all.py            # real data, default model list
    python experiments/run_all.py --debug    # offline smoke of the whole pipeline
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import Counter

# path bootstrap (also makes sibling experiment modules importable)
_HERE = os.path.dirname(os.path.abspath(__file__))
_SRC = os.path.dirname(_HERE)
for p in (_SRC, _HERE):
    if p not in sys.path:
        sys.path.insert(0, p)

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from config import (  # noqa: E402
    ContextLevel, ExperimentConfig, JudgeConfig, LLMConfig, SimConfig,
)
from metrics import MetricsCollector, save_decisions  # noqa: E402
from policies.llm import LLMPolicy  # noqa: E402
from prompting import parse_decision  # noqa: E402
from runner import load_experiments, make_backend, make_policy, run_policy  # noqa: E402

from exp1_select_llm import run_sweep, score_summary  # noqa: E402
from exp2_vs_baselines import paired_tests  # noqa: E402
from exp3_determinism import sample_scenarios  # noqa: E402

DEFAULT_MODELS = [
    "llama3.1:8b", "gemma3:12b", "gemma3:27b", "gemma4:31b",
    "gpt-oss:20b", "gpt-oss:120b", "llama3.3:70b", "qwen3.6:latest",
]


def _now():
    return time.strftime("%H:%M:%S")


def base_cfg(args, model, level, seeds, sim, n_experiments) -> ExperimentConfig:
    return ExperimentConfig(
        n_experiments=n_experiments, seeds=seeds, split=args.split, debug=args.debug,
        min_units_per_experiment=args.min_units, results_dir=args.results_dir,
        llm=LLMConfig(backend=("mock" if args.debug else "ollama"),
                      model=model, temperature=args.temperature, timeout=args.timeout),
        spread_datasets=getattr(args, "spread_datasets", False),
        judge=JudgeConfig(context_level=ContextLevel.parse(level), n_samples=1,
                          use_cache=True, prompt_variant=args.prompt_variant),
        sim=sim,
    )


# --------------------------------------------------------------------------- #
def phase_exp1(args, experiments):
    print(f"\n===== [{_now()}] EXP1: model + context-level selection =====")
    levels = [ContextLevel.parse(x) for x in args.levels.split(",")]
    sim = SimConfig(checkpoint_epoch=args.exp1_checkpoint, checkpoint_frac=args.checkpoint_frac,
                    patience=args.patience, warmup_trials=args.warmup_trials,
                    n_trials_cap=args.exp1_trials)
    cfg = base_cfg(args, None, levels[0], _seeds(args.exp1_seeds), sim, args.exp1_experiments)
    exps = experiments[: args.exp1_experiments]
    t0 = time.time()
    sink = [] if args.dump_decisions else None
    collector, name_to_cfg, working, failed = run_sweep(args.models, levels, exps, cfg,
                                                        decision_sink=sink)
    print(f"[exp1] {_now()} working={working} skipped={failed} ({time.time()-t0:.0f}s)")
    if not collector.results:
        return None

    collector.save_csv(os.path.join(args.results_dir, "exp1_runs.csv"))
    if sink:
        p = save_decisions(sink, os.path.join(args.results_dir, "raw", "exp1_decisions.csv"))
        print(f"[exp1] dumped {len(sink)} decisions -> {p}")
    ranked = score_summary(collector.summary(), args.w_save, args.w_regret)
    ranked.to_csv(os.path.join(args.results_dir, "exp1_summary.csv"), index=False)
    cols = ["policy", "score", "regret_mean", "saved_pct_B_mean",
            "false_prune_rate", "false_continue_rate", "mean_confidence"]
    print(ranked[cols].to_string(index=False))

    best = ranked.iloc[0]["policy"]
    best_model, best_level = name_to_cfg.get(best, (None, None))
    out = {"policy": best, "model": best_model, "context_level": best_level,
           "working": working, "failed": failed}
    with open(os.path.join(args.results_dir, "exp1_best.json"), "w") as f:
        json.dump(out, f, indent=2)
    print(f"[exp1] BEST -> {best} (model={best_model}, level=L{best_level})")
    return out


def phase_exp2(args, experiments, best_model, best_level):
    print(f"\n===== [{_now()}] EXP2: best LLM vs baselines + oracle =====")
    sim = SimConfig(checkpoint_epoch=args.exp2_checkpoint, checkpoint_frac=args.checkpoint_frac,
                    patience=args.patience, warmup_trials=args.warmup_trials,
                    n_trials_cap=args.exp2_trials)
    cfg = base_cfg(args, best_model, best_level, _seeds(args.exp2_seeds), sim,
                   args.exp2_experiments)
    exps = experiments[: args.exp2_experiments]
    collector = MetricsCollector()
    sink = [] if args.dump_decisions else None

    # LLM judge (resilient: keep baselines even if the model dies mid-run)
    try:
        backend = make_backend(cfg, model=best_model)
        llm = LLMPolicy(backend=backend, config=cfg.judge)
        t0 = time.time()
        run_policy(llm, exps, cfg, collector, decision_sink=sink)
        llm_name = llm.name
        print(f"  done {llm_name} ({time.time()-t0:.0f}s)")
    except Exception as e:
        llm_name = None
        print(f"  [warn] LLM judge failed in exp2 ({type(e).__name__}: {str(e)[:80]})")

    for name in [b.strip() for b in args.baselines.split(",") if b.strip()]:
        policy = make_policy(name, cfg)
        run_policy(policy, exps, cfg, collector, decision_sink=sink)
        print(f"  done {policy.name}")

    if sink:
        p = save_decisions(sink, os.path.join(args.results_dir, "raw", "exp2_decisions.csv"))
        print(f"  dumped {len(sink)} decisions -> {p}")

    collector.save_csv(os.path.join(args.results_dir, "exp2_runs.csv"))
    summary = collector.summary()
    summary.to_csv(os.path.join(args.results_dir, "exp2_summary.csv"), index=False)
    df = collector.to_dataframe()
    if llm_name:
        tests = pd.concat([paired_tests(df, llm_name, "regret"),
                           paired_tests(df, llm_name, "saved_pct_B")], ignore_index=True)
        if not tests.empty:
            tests.to_csv(os.path.join(args.results_dir, "exp2_significance.csv"), index=False)
    cols = ["policy", "regret_mean", "saved_pct_A_mean", "saved_pct_B_mean",
            "false_prune_rate", "false_continue_rate", "found_best_rate", "prune_rate"]
    print(summary[cols].to_string(index=False))
    return df


def phase_exp3(args, experiments, best_model, best_level):
    print(f"\n===== [{_now()}] EXP3: determinism of the best LLM =====")
    sim = SimConfig(checkpoint_epoch=args.exp2_checkpoint, checkpoint_frac=args.checkpoint_frac)
    cfg = base_cfg(args, best_model, best_level, [0], sim, len(experiments))
    rng = np.random.default_rng(args.random_state)
    scenarios = sample_scenarios(experiments, cfg, args.exp3_scenarios, rng)
    backend = make_backend(cfg, model=best_model)
    judge = LLMPolicy(backend=backend, config=JudgeConfig(
        context_level=ContextLevel.parse(best_level), n_samples=1, use_cache=False,
        prompt_variant=args.prompt_variant))

    rows = []
    vote_rows = []
    for i, sc in enumerate(scenarios):
        system, user = judge.prompts.build(sc, ContextLevel.parse(best_level))
        votes = []
        for rep in range(args.exp3_repeats):
            try:
                resp = backend.chat(system, user)
                v = parse_decision(resp.text)
                votes.append(v)
                if args.dump_decisions:
                    vote_rows.append({"scenario": i, "unit_id": sc.unit_id, "epoch": sc.epoch,
                                      "repeat": rep, "vote": v, "latency_s": resp.latency_s,
                                      "tokens": (resp.input_tokens or 0) + (resp.output_tokens or 0),
                                      "raw": (resp.text or "")[:2000]})
            except Exception:
                continue
        if not votes:
            continue
        counts = Counter(votes)
        action, n = counts.most_common(1)[0]
        rows.append({"scenario": i, "unit_id": sc.unit_id, "epoch": sc.epoch,
                     "majority": action, "confidence": n / len(votes),
                     "deterministic": len(counts) == 1,
                     "n_prune": counts.get("PRUNE", 0), "n_continue": counts.get("CONTINUE", 0)})
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(args.results_dir, "exp3_determinism.csv"), index=False)
    if vote_rows:
        p = save_decisions(vote_rows, os.path.join(args.results_dir, "raw", "exp3_votes.csv"))
        print(f"  dumped {len(vote_rows)} votes -> {p}")
    if not df.empty:
        print(f"  mean confidence={df['confidence'].mean():.3f} "
              f"deterministic={df['deterministic'].mean():.1%} (n={len(df)})")
    return df


def _make_plots(args, exp2_df, exp3_df):
    try:
        from plots import plot_runs, plot_determinism
        outdir = os.path.join(args.results_dir, "figs")
        if exp2_df is not None and not exp2_df.empty:
            plot_runs(exp2_df, outdir)
        if exp3_df is not None and not exp3_df.empty:
            plot_determinism(exp3_df, outdir)
    except Exception as e:
        print(f"  [warn] plotting failed: {type(e).__name__}: {e}")


def _seeds(s):
    return [int(x) for x in str(s).split(",") if x.strip() != ""]


def main():
    ap = argparse.ArgumentParser(description="run exp1 -> exp2 -> exp3 end to end")
    ap.add_argument("--debug", action="store_true")
    ap.add_argument("--models", type=str, default=",".join(DEFAULT_MODELS))
    ap.add_argument("--levels", type=str, default="0,1,2,3")
    ap.add_argument("--baselines", type=str, default="random,last-seen,arima,oracle")
    ap.add_argument("--split", type=str, default="test")
    ap.add_argument("--min-units", type=int, default=2)
    ap.add_argument("--random-state", type=int, default=666)
    ap.add_argument("--run-name", type=str, default=None,
                    help="bundle ALL artifacts under results/runs/<name>/ (recommended)")
    ap.add_argument("--results-dir", type=str, default="results")
    ap.add_argument("--temperature", type=float, default=0.3)
    ap.add_argument("--timeout", type=int, default=120)
    ap.add_argument("--patience", type=int, default=3)
    ap.add_argument("--warmup-trials", type=int, default=3)
    ap.add_argument("--checkpoint-frac", type=float, default=None,
                    help="if set, checkpoint = max(2, frac*n_epochs) per curve "
                         "(robust to varying curve lengths; overrides --*-checkpoint)")
    ap.add_argument("--prompt-variant", type=str, default="neutral",
                    choices=["conservative", "neutral", "aggressive"],
                    help="system-prompt variant used by the LLM judge in all phases")
    ap.add_argument("--spread-datasets", action="store_true",
                    help="select experiments round-robin across datasets (diversity)")
    ap.add_argument("--dump-decisions", action="store_true",
                    help="write per-decision records (full trajectory) to results/raw/ "
                         "so any metric/plot can be recomputed offline without rerunning")
    ap.add_argument("--skip-exp1", action="store_true",
                    help="skip the exp1 sweep and use --best-model/--best-level "
                         "(reuse an already-computed exp1 selection)")
    ap.add_argument("--best-model", type=str, default=None)
    ap.add_argument("--best-level", type=int, default=None)
    ap.add_argument("--w-save", type=float, default=1.0)
    ap.add_argument("--w-regret", type=float, default=1.0)
    # exp1 (reduced subset for selection)
    ap.add_argument("--exp1-experiments", type=int, default=1)
    ap.add_argument("--exp1-trials", type=int, default=5)
    ap.add_argument("--exp1-checkpoint", type=int, default=32)
    ap.add_argument("--exp1-seeds", type=str, default="0")
    # exp2 (best vs baselines)
    ap.add_argument("--exp2-experiments", type=int, default=1)
    ap.add_argument("--exp2-trials", type=int, default=8)
    ap.add_argument("--exp2-checkpoint", type=int, default=22)
    ap.add_argument("--exp2-seeds", type=str, default="0,1")
    # exp3 (determinism)
    ap.add_argument("--exp3-scenarios", type=int, default=20)
    ap.add_argument("--exp3-repeats", type=int, default=5)
    args = ap.parse_args()
    args.models = [m.strip() for m in args.models.split(",") if m.strip()]
    from _cli import tee_to  # noqa: E402
    if args.run_name:
        args.results_dir = os.path.join("results", "runs", args.run_name)
    os.makedirs(args.results_dir, exist_ok=True)
    tee_to(os.path.join(args.results_dir, "run.log"))
    # save the exact prompt used (by design)
    try:
        from prompting import get_system_prompt
        pdir = os.path.join(args.results_dir, "prompts")
        os.makedirs(pdir, exist_ok=True)
        with open(os.path.join(pdir, f"{args.prompt_variant}.txt"), "w", encoding="utf-8") as f:
            f.write(get_system_prompt(args.prompt_variant))
    except Exception as e:
        print(f"[run_all] could not save prompt text: {e}")

    print(f"[{_now()}] loading data (debug={args.debug})...")
    n_needed = max(args.exp1_experiments, args.exp2_experiments)
    load_cfg = base_cfg(args, None, 0, [0], SimConfig(), n_needed)
    experiments = load_experiments(load_cfg)
    print(f"[{_now()}] loaded {len(experiments)} experiment(s): "
          f"{[e.experiment_id for e in experiments]}")
    if not experiments:
        print("no experiments available; aborting.")
        return

    if args.skip_exp1:
        if not args.best_model or args.best_level is None:
            print("--skip-exp1 requires --best-model and --best-level; aborting.")
            return
        bm, bl = args.best_model, args.best_level
        print(f"[exp1] skipped; using provided best = {bm} / L{bl}")
    else:
        best = phase_exp1(args, experiments)
        if not best or not best.get("model"):
            print("exp1 produced no working model; skipping exp2/exp3.")
            return
        bm, bl = best["model"], best["context_level"]

    exp2_df = phase_exp2(args, experiments, bm, bl)
    exp3_df = phase_exp3(args, experiments, bm, bl)
    _make_plots(args, exp2_df, exp3_df)
    print(f"\n[{_now()}] ALL DONE. CSVs + figures in {args.results_dir}/")


if __name__ == "__main__":
    main()
