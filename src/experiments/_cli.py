"""Shared CLI argument wiring for the experiment scripts.

Every experiment is independently and easily configurable from the command line;
this module centralises the common flags so the three studies stay consistent.
"""

from __future__ import annotations

import argparse
import os
import sys

# make the flat modules under src/ importable whether or not the package was pip-installed
_SRC = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from config import (  # noqa: E402
    ContextLevel, ExperimentConfig, JudgeConfig, LLMConfig, SimConfig,
)


def add_common_args(parser: argparse.ArgumentParser) -> None:
    g = parser.add_argument_group("data / run control")
    g.add_argument("--debug", action="store_true",
                   help="use synthetic offline data + MockBackend (no network/TF)")
    g.add_argument("--n-experiments", type=int, default=None,
                   help="limit number of experiments (default: all; debug default: 1)")
    g.add_argument("--seeds", type=str, default="0,1,2,3,4",
                   help="comma-separated Monte-Carlo seeds (e.g. '0,1,2')")
    g.add_argument("--split", type=str, default="test", choices=["train", "val", "test"])
    g.add_argument("--fold", type=int, default=0)
    g.add_argument("--num-folds", type=int, default=5)
    g.add_argument("--random-state", type=int, default=666)
    g.add_argument("--min-units", type=int, default=2)
    g.add_argument("--results-dir", type=str, default="results")

    s = parser.add_argument_group("simulator")
    s.add_argument("--checkpoint-epoch", type=int, default=10)
    s.add_argument("--patience", type=int, default=3)
    s.add_argument("--warmup-trials", type=int, default=3)
    s.add_argument("--n-trials-cap", type=int, default=100)

    l = parser.add_argument_group("LLM / judge")
    l.add_argument("--backend", type=str, default="ollama", choices=["ollama", "mock"])
    l.add_argument("--model", type=str, default=None, help="LLM model id (default: $LLM_MODEL)")
    l.add_argument("--temperature", type=float, default=0.3)
    l.add_argument("--timeout", type=int, default=180)
    l.add_argument("--context-level", type=str, default="L2", help="L0|L1|L2|L3")
    l.add_argument("--n-samples", type=int, default=1)
    l.add_argument("--no-cache", action="store_true")


def parse_seeds(s: str):
    return [int(x) for x in str(s).split(",") if x.strip() != ""]


class _Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, s):
        for st in self.streams:
            st.write(s); st.flush()

    def flush(self):
        for st in self.streams:
            st.flush()


def tee_to(path: str):
    """Mirror stdout/stderr into ``path`` (so each run keeps its own log by design)."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    f = open(path, "a", encoding="utf-8", buffering=1)
    sys.stdout = _Tee(sys.__stdout__, f)
    sys.stderr = _Tee(sys.__stderr__, f)
    return f


def resolve_run_dir(run_name, results_dir):
    """If a run name is given, bundle all artifacts under results/runs/<name>/."""
    return os.path.join("results", "runs", run_name) if run_name else results_dir


def config_from_args(args) -> ExperimentConfig:
    n_exp = args.n_experiments
    if n_exp is None and args.debug:
        n_exp = 1
    return ExperimentConfig(
        test_dataset_names=[],
        random_state=args.random_state,
        num_folds=args.num_folds,
        fold=args.fold,
        n_experiments=n_exp,
        min_units_per_experiment=args.min_units,
        seeds=parse_seeds(args.seeds),
        split=args.split,
        debug=args.debug,
        results_dir=args.results_dir,
        llm=LLMConfig(
            backend=("mock" if args.debug else args.backend),
            model=args.model, temperature=args.temperature, timeout=args.timeout,
        ),
        judge=JudgeConfig(
            context_level=ContextLevel.parse(args.context_level),
            n_samples=args.n_samples, use_cache=not args.no_cache,
        ),
        sim=SimConfig(
            checkpoint_epoch=args.checkpoint_epoch, patience=args.patience,
            warmup_trials=args.warmup_trials, n_trials_cap=args.n_trials_cap,
        ),
    )
