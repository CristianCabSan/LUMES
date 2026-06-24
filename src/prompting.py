"""Prompt construction (context levels L0..L3) and decision parsing.

``PromptBuilder.build(scenario, level)`` returns a ``(system, user)`` pair.  The
contract with the model is a single strict line ``DECISION: CONTINUE`` or
``DECISION: PRUNE``; ``parse_decision`` recovers it robustly and falls back to the
*safe* default (CONTINUE) when nothing parses, so a chatty/garbled model never
causes an unwanted prune.
"""

from __future__ import annotations

import re
from typing import Tuple

from config import ContextLevel
from scenario import DecisionScenario

CONTINUE = "CONTINUE"
PRUNE = "PRUNE"

# --- system-prompt variants (the only thing that differs is the decision-bias) ---
_HEAD = (
    "You are an expert judge for early stopping during hyperparameter optimization "
    "of neural networks. A model is being trained and you are called once per epoch. "
    "Given ONLY the past/present training signal (you never see the future), decide "
    "whether this run is worth continuing or should be pruned to save compute.\n"
)
_TAIL = (
    "Answer with EXACTLY one final line and nothing after it:\n"
    "DECISION: CONTINUE\n"
    "or\n"
    "DECISION: PRUNE"
)
_GUIDANCE = {
    # default toward CONTINUE
    "conservative": (
        "PRUNE only when the run is clearly not going to beat the best configuration "
        "found so far (diverging, plateaued well above the best, or improving far too "
        "slowly). When in doubt, CONTINUE.\n"
    ),
    # no stated default — let the model decide purely from the signal
    "neutral": "",
    # default toward PRUNE
    "aggressive": (
        "Be decisive and lean towards pruning: training compute is expensive, so PRUNE "
        "any run that is unlikely to end up beating the best configuration found so far "
        "-- e.g. it is diverging, has plateaued, is already worse than the best-so-far "
        "and not clearly closing the gap, or is improving too slowly to realistically "
        "catch up before training ends. Only CONTINUE when the run is still competitive "
        "with the best-so-far, or is improving fast enough that it could plausibly "
        "overtake it. A run that has been mediocre for several epochs should be PRUNED.\n"
    ),
}
PROMPTS = {name: _HEAD + g + _TAIL for name, g in _GUIDANCE.items()}


def get_system_prompt(variant: str) -> str:
    """Return the system prompt for 'conservative' | 'neutral' | 'aggressive'."""
    if variant not in PROMPTS:
        raise ValueError(f"unknown prompt variant '{variant}'; choose from {list(PROMPTS)}")
    return PROMPTS[variant]


# default used when no variant is specified
SYSTEM_PROMPT = PROMPTS["neutral"]


def _fmt_list(xs, k: int = 6) -> str:
    return "[" + ", ".join(f"{x:.4f}" for x in xs[-k:]) + "]"


class PromptBuilder:
    """Builds (system, user) prompts at a given :class:`ContextLevel`."""

    def __init__(self, system_prompt: str = SYSTEM_PROMPT):
        self.system_prompt = system_prompt

    def build(self, scenario: DecisionScenario, level: ContextLevel) -> Tuple[str, str]:
        level = ContextLevel.parse(level)
        lines = []

        # ---- L0: the partial validation curve (always present) ----
        lines.append(f"Epoch: {scenario.epoch} (0-based).")
        lines.append(
            f"Partial validation loss (last values, lower is better): "
            f"{_fmt_list(scenario.partial_val)}"
        )

        # ---- L1: derived past features ----
        if level >= ContextLevel.L1:
            f = scenario.features
            lines.append("Derived features (past only):")
            lines.append(f"  current_val = {f.get('current_val'):.4f}")
            lines.append(f"  val_velocity (last delta) = {f.get('val_velocity'):.4f}")
            lines.append(f"  val_ema (span 3) = {f.get('val_ema'):.4f}")
            lines.append(f"  val_min_so_far = {f.get('val_min_so_far'):.4f}")
            lines.append(f"  epochs_since_best = {f.get('epochs_since_best')}")
            if f.get("val_improvement") is not None:
                lines.append(f"  val_improvement_vs_best = {f['val_improvement']:.4f}")
            if f.get("expected_improvement") is not None:
                lines.append(f"  expected_improvement (heuristic) = {f['expected_improvement']:.4f}")

        # ---- L2: Bayesian-optimization state ----
        if level >= ContextLevel.L2:
            b = scenario.bo_state
            lines.append("Bayesian-optimization state:")
            best = b.get("best_so_far_final_val")
            lines.append(
                "  best_final_val_so_far = "
                + ("unknown (warmup)" if best is None else f"{best:.4f}")
            )
            bve = b.get("best_val_at_epoch")
            if bve is not None:
                lines.append(f"  best_run_val_at_this_epoch = {bve:.4f}")
            lines.append(f"  trials_completed = {b.get('n_trials_done', 0)}")
            lines.append(f"  in_warmup = {bool(b.get('in_warmup', False))}")

        # ---- L3: task / architecture / hyperparameters ----
        if level >= ContextLevel.L3:
            m = scenario.meta
            lines.append("Task / architecture / hyperparameters:")
            lines.append(f"  dataset = {m.get('dataset')}")
            lines.append(f"  task = {m.get('task')}")
            lines.append(f"  net = {m.get('net')}")
            hp = m.get("hyperparameters") or {}
            if hp:
                hp_str = ", ".join(f"{k}={v}" for k, v in sorted(hp.items()))
                lines.append(f"  hyperparameters: {hp_str}")

        lines.append("")
        lines.append("Decide now. Reply with the strict final line only.")
        return self.system_prompt, "\n".join(lines)


_DECISION_RE = re.compile(r"DECISION\s*[:\-]?\s*(CONTINUE|PRUNE)", re.IGNORECASE)


def parse_decision(text: str, default: str = CONTINUE) -> str:
    """Extract CONTINUE/PRUNE from a model reply; safe default on failure.

    Strategy: prefer the explicit ``DECISION:`` marker (last occurrence wins, since
    models sometimes restate the format before answering); else fall back to a bare
    keyword anywhere in the text; else the provided safe default.
    """
    if not text:
        return default
    matches = _DECISION_RE.findall(text)
    if matches:
        return matches[-1].upper()

    upper = text.upper()
    has_prune = PRUNE in upper
    has_continue = CONTINUE in upper
    if has_prune and not has_continue:
        return PRUNE
    if has_continue and not has_prune:
        return CONTINUE
    return default
