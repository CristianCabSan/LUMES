# v6 — prompt-variant comparison — babysitting log

Controlled study: **fixed model `gemma3:12b` / L2**, 3 system-prompt variants
(conservative / neutral / aggressive) vs baselines (random, last-seen, arima,
**classical early-stopping**) + oracle. Bigger, diverse set.

Config: 10 experiments × 3 seeds (30 runs/method), trials=12, checkpoint=3,
patience=2, warmup=3, es_patience=5, timeout=120, dumps on (up-to-stop).
Task: bkndhg5n6 → `results/runs/v6_prompt_compare/run.log`. Babysit ~every 60 min.
Goal: isolate the prompt effect (model held fixed) on the savings–regret frontier,
with a classical-ES baseline and proper CIs across datasets.

| time | progress | status | action |
|------|----------|--------|--------|
| 08:30 | launched (loading data) | ⚠️ first 10 experiments were ALL HSF15 (no cross-dataset diversity) | **fixed:** added `spread_datasets` (round-robin across datasets) to dataset/config/runner + `--spread-datasets`; stopped before any LLM cost |
| 08:3x | relaunched with --spread-datasets (task b0psh3boh; 26 tests pass) | ✅ running | monitor armed |
| done | 10 datasets confirmed (HSF15/MOSFET11/MPM20/OBDD17/PHM18/PHMAP23/PHME20/PRONOSTIA/RRB23/UPM20), all 8 methods complete, 24924 decisions dumped | ✅✅ completed, no failures | **loop ending** |

## v6 outcome — clean controlled prompt frontier (n=30, 10 datasets)
Prompt monotonically trades savings↔regret: conservative (regret .008 / saved .165 /
found_best .767) → neutral (.051 / .289 / .433) → aggressive (.148 / .609 / .233).
Conservative LLM = best non-oracle QUALITY (regret≈last-seen, found_best highest).
Aggressive = most savings (>oracle) but high regret. **Classical early-stopping is the
WORST pruner** (regret .124, saved only .10) → validates incumbent-aware pruning.
last-seen remains the strong all-rounder (saved .41 / regret .012 / found .70).
All artifacts in this folder; babysitting complete.
