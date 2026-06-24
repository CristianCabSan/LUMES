# v4 run — babysitting log

Task: check ~hourly; if something goes wrong, document it here, fix it, relaunch.
Repeat until exp1 + exp2 + exp3 all complete correctly.

Run: `run_all_v4` (task b88i026uz) → log `results/run_all_v4.log`
Config: 7 working models × L0–L3, checkpoint=3, patience=2, warmup=3,
exp1=1 exp/1 seed/trials6, exp2=4 exps/2 seeds/trials12, exp3=80×8, `--dump-decisions`.

| time | phase | progress | status | action |
|------|-------|----------|--------|--------|
| 23:47 | EXP1 | 18/28 cells (through `gpt-oss:20b:L1`) | ✅ healthy, progressing (13→18 since 22:19) | none; slow models (120b, 70b) still queued |
| 00:27 | EXP1→EXP2 | EXP1 done (10744s≈3h), 1848 decisions dumped, BEST=`gemma3:27b:L2`; EXP2 started | ✅ exp1 completed correctly, all 7 models swept, qwen skipped | none; exp2 running (winner ~9s/call) |
| 01:33 | EXP2→EXP3 | EXP2 done (~66m): LLM + 4 baselines over 4 exps × 2 seeds, 9659 decisions dumped; EXP3 started (80×8) | ✅ exp2 completed correctly (all policies ran, CSVs+significance written) | none; exp3 final phase running |
| 01:44 | DONE | EXP3 done (conf 0.998, 98.8% determ, 640 votes); ALL DONE, all CSVs+figs+raw written | ✅✅ **all experiments passed correctly** — nothing went wrong, no fixes needed | **loop ending** (no reschedule) |

## Outcome
Run completed end-to-end with **zero failures**. No crashes, hangs, or degenerate
phases. Only expected event: `qwen3.6:latest` skipped (absent on Ollamus, HTTP 400).
Total wall-clock ~4h17m. Results: exp1 BEST=`gemma3:27b/L2`; exp2 LLM saves ~48%
train time (sig > random) but higher regret than last-seen (more aggressive); exp3
98.8% deterministic. Raw dumps: 1848 + 9659 + 640 decision rows.
Babysitting task complete.
