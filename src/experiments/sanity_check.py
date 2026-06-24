"""Sanity check (Fase 3): scripted curves -> judge -> expected decision.

Runs three textbook curves (improving / plateau-above-best / diverging) through
the LLM judge and prints decision + confidence next to the expected answer.  Uses
the offline MockBackend by default; pass ``--real`` to hit the configured LLM.

    python experiments/sanity_check.py            # offline MockBackend
    python experiments/sanity_check.py --real --model llama3.1:8b --context-level L2
"""

from __future__ import annotations

import argparse

import numpy as np

from _cli import _SRC  # noqa: F401  (ensures src on path)

from backend import MockBackend, build_backend_from_env
from config import ContextLevel, JudgeConfig
from policies.llm import LLMPolicy
from scenario import ScenarioBuilder

CASES = [
    ("improving", 0.20, "CONTINUE"),     # steadily decreasing toward a good loss
    ("plateau_above_best", 0.20, "PRUNE"),
    ("diverging", 0.20, "PRUNE"),
]


def make_curve(kind: str, n: int = 25) -> np.ndarray:
    e = np.arange(n)
    if kind == "improving":
        val = 0.15 + 0.85 * np.exp(-e / 6.0)
    elif kind == "plateau_above_best":
        val = 0.6 + 0.05 * np.exp(-e / 5.0)         # flat ~0.6, best is 0.2
    elif kind == "diverging":
        val = 0.4 + 0.03 * e                          # climbing
    else:
        raise ValueError(kind)
    train = np.clip(val - 0.05, 1e-3, None)
    return np.stack([train, val], axis=1)


def main():
    ap = argparse.ArgumentParser(description="LLM judge sanity check")
    ap.add_argument("--real", action="store_true", help="use the real LLM backend")
    ap.add_argument("--model", type=str, default=None)
    ap.add_argument("--context-level", type=str, default="L2")
    ap.add_argument("--n-samples", type=int, default=1)
    ap.add_argument("--temperature", type=float, default=0.3)
    ap.add_argument("--timeout", type=int, default=180)
    args = ap.parse_args()

    if args.real:
        backend = build_backend_from_env("ollama", args.model, args.temperature, args.timeout)
    else:
        backend = MockBackend(model=args.model or "mock")

    judge = LLMPolicy(
        backend=backend,
        config=JudgeConfig(
            context_level=ContextLevel.parse(args.context_level),
            n_samples=args.n_samples, use_cache=False,
        ),
    )
    builder = ScenarioBuilder()

    print(f"backend={backend.name} model={getattr(backend,'model','?')} "
          f"level={args.context_level} n_samples={args.n_samples}\n")
    n_ok = 0
    for kind, best_so_far, expected in CASES:
        curve = make_curve(kind)
        epoch = curve.shape[0] - 1
        bo_state = {
            "best_so_far_final_val": best_so_far,
            "best_val_at_epoch": best_so_far,
            "n_trials_done": 5, "in_warmup": False,
        }
        scenario = builder.build("sanity_u", "sanity_exp", curve, epoch, bo_state,
                                 {"dataset": "demo", "task": "final_loss", "net": "demo"})
        d = judge.decide(scenario)
        ok = d.action == expected
        n_ok += ok
        print(f"[{'OK ' if ok else 'XX '}] {kind:20s} -> {d.action:8s} "
              f"(conf={d.confidence:.2f}, expected={expected})")
    print(f"\n{n_ok}/{len(CASES)} as expected")


if __name__ == "__main__":
    main()
