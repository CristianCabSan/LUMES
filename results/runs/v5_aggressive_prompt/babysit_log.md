# v5 run — babysitting log (aggressive-pruning prompt)

Task: check ~every 30 min; if something goes wrong, document it here, fix it, relaunch.
Repeat until exp1 + exp2 + exp3 all complete correctly.

Change vs v4: `SYSTEM_PROMPT` made more aggressive on pruning (lean towards PRUNE for
mediocre/plateaued runs). v4 fully archived at `results/runs/v4_baseline_prompt/`.

Run: `run_all_v5` (task bdhobgq2u) → log `results/run_all_v5.log`
Config: 7 working models × L0–L3, checkpoint=3, patience=2, warmup=3,
exp1=1 exp/1 seed/trials6, exp2=4 exps/2 seeds/trials12, exp3=80×8, `--dump-decisions`.
Goal: more pruning / more time saved than v4 (v4 winner gemma3:27b/L2: saved ~0.22 in
exp1, ~0.48 in exp2 with regret 0.087).

| time | phase | progress | status | action |
|------|-------|----------|--------|--------|
| 07:49 | start | launched v5 (aggressive prompt) | ✅ launched | monitor armed; 30-min checks |
| 10:51 | EXP1 done → FIX | exp1 done (3h). Aggressive prompt worked (mean prune 0.16→0.48, saved 0.08→0.29). **BUT scoring bug**: min-max regret penalty made the *smallest* non-zero regret = full penalty, so it picked slow `gpt-oss:120b/L3` (saved 0.22) over fast high-savers (saved 0.38). exp2 was heading for ~15h on the 120b. | ⚠️ flawed selection → fixed | **Fix:** (1) `score_summary` now uses raw `saved − regret` (interpretable units). (2) added `run_all --skip-exp1 --best-model/--best-level` to reuse exp1. Re-scored exp1 → new winner `gemma3:12b/L2` (saved 0.38, regret 0.0103, ~7s/call). Stopped v5; relaunching exp2+exp3 on the corrected winner (`run_all_v5b.log`). exp1 CSVs/dumps preserved + re-scored. |
| 07:56 | EXP2 (v5b) | relaunch: first dispatch died (nested `&` orphaned it); relaunched cleanly as task `b16hxafh7`, data loaded, exp2 starting on `gemma3:12b/L2` | ✅ running | continue 30-min checks |
| 12:08 | DONE | exp2 (~57m) + exp3 done; ALL DONE; CSVs+figs+raw written (1848/9702/640 rows) | ✅✅ **v5 complete, all experiments passed** | **loop ending** (no reschedule) |

## v5 Outcome
Aggressive prompt achieved the goal: LLM (`gemma3:12b/L2`) exp2 saved **0.594** of train
time (v4 was 0.481), false-continue 0→0, prune-rate 0.67→0.75 — and now saves
*significantly more than oracle/random/arima*. Trade-off as expected: regret 0.087→0.133
(sig. worse than last-seen/oracle/random), found-best 0.50→0.25 (prunes some winners).
One fix during the run (exp1 scoring min-max → raw saved−regret; re-used exp1 via
--skip-exp1). v4 preserved at results/runs/v4_baseline_prompt/. Babysitting complete.
