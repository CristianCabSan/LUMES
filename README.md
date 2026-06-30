# LUMES: An LLM-Underpinned Method for Early Stopping to Accelerate Bayesian Optimization in Predictive Maintenance

TFM project. An **LLM acts as an early-stopping judge** for Bayesian optimization
(BO) over **pre-computed learning curves** from the `CURVES` meta-dataset (consumed
through the read-only `phmlc` framework). Epoch by epoch the judge decides
`CONTINUE` / `PRUNE` using **only past/present** information, and is compared
against three baselines (`random`, `last-seen`, `arima`) and an `oracle`.

## 1. Install

```bash
git clone <repo-url> LUMES && cd LUMES
```

`phmd` pins `pandas==2.0.3 / scipy==1.10.1 / scikit-learn==1.3.2`, which resolve
cleanly on **Python 3.10–3.12**. A 3.10 virtualenv is used here.

```bash
py -3.10 -m venv .venv
.venv/Scripts/python -m pip install -U pip wheel setuptools
.venv/Scripts/python -m pip install -e ".[dev]"     # runtime + pytest, from pyproject
```

This is **standalone**: `phmd` is a hard dependency (installed above) and supplies
the CURVES meta-dataset (auto-downloaded + auto-unzipped on first real, non-`--debug`
run); `phmlc`/`phm_framework` and TensorFlow are **not required** — see below.

**TensorFlow is optional.** The canonical loader is
`phm_framework.optimization.curves.load_curves`, but importing `phm_framework` runs
its package `__init__`, which eagerly pulls the whole training stack (TensorFlow,
`einops`, a bare `import utils`) that `load_curves` does not need and that does not
import cleanly here. The bridge therefore **falls back automatically** to a
phmd-only replica of `load_curves` (identical CURVES normalisation + split) — so the
real data path works without TensorFlow. If you do want the canonical path, install
the extras and fix phmlc's environment: `pip install -e ".[phmlc]"`.

Then create your `.env` (copy from [.env.example](.env.example)):

```
LLM_BASE_URL=https://<ollamus-host>     # Ollamus proxy root (Ollama /api/chat)
LLM_API_KEY=<bearer token>
LLM_MODEL=llama3.1:8b                    # default model id
PHMLC_SRC=../phmlc/src                   # path to phmlc sources
```

## 2. Verify the install

```bash
.venv/Scripts/python -m pytest -q                     # 26 unit tests (offline)
.venv/Scripts/python src/experiments/sanity_check.py   # scripted curves -> judge
.venv/Scripts/python src/experiments/smoke_bridge.py   # REAL data path (TF + CURVES)
```

## 3. The three experiments — and how to configure each one

All scripts share the flags in [src/experiments/_cli.py](src/experiments/_cli.py) and each
is configured **independently** from the command line. Common knobs:

| flag | meaning | default |
|---|---|---|
| `--debug` | synthetic offline data + MockBackend (no TF/network) | off |
| `--n-experiments N` | limit number of `dataset_task_net` groups | all (debug: 1) |
| `--seeds a,b,c` | Monte-Carlo seeds | `0,1,2,3,4` |
| `--split` | `train`/`val`/`test` split to evaluate | `test` |
| `--model` | LLM id (else `$LLM_MODEL`) | env |
| `--context-level` | `L0`/`L1`/`L2`/`L3` | `L2` |
| `--n-samples` | LLM samples per decision (majority vote) | 1 |
| `--checkpoint-epoch` / `--patience` / `--warmup-trials` | harness | 10 / 3 / 3 |
| `--results-dir` | output folder | `results` |

**Smoke runs use a single experiment**: just add `--debug` (sets `--n-experiments 1`
and the MockBackend), or pass `--n-experiments 1` on the real path.

### Exp 1 — select LLM + context level (reduced subset)
Sweeps models × context levels and ranks `(model, level)` by a combined score
(high Variant-B savings, low regret).
```bash
# offline smoke (1 experiment):
python src/experiments/exp1_select_llm.py --debug
# real, reduced subset:
python src/experiments/exp1_select_llm.py \
    --models llama3.1:8b,mistral:7b,qwen2.5:7b \
    --levels 0,1,2,3 --n-experiments 5 --seeds 0,1,2
```
Outputs: `results/exp1_runs.csv`, `results/exp1_summary.csv` (ranked, `score`).
Tune the score weights with `--w-save` / `--w-regret`.

### Exp 2 — best LLM vs baselines + oracle (large set)
```bash
python src/experiments/exp2_vs_baselines.py --debug                      # offline e2e
python src/experiments/exp2_vs_baselines.py \
    --model llama3.1:8b --context-level L2 --seeds 0,1,2,3,4              # all test exps
```
Outputs: `results/exp2_runs.csv`, `results/exp2_summary.csv`,
`results/exp2_significance.csv` (paired Wilcoxon, LLM vs each baseline).
Pick the comparison set with `--baselines random,last-seen,arima,oracle`.

### Exp 3 — determinism / confidence (same LLM)
```bash
python src/experiments/exp3_determinism.py --debug --repeats 5
python src/experiments/exp3_determinism.py \
    --model llama3.1:8b --context-level L2 --repeats 10 --n-scenarios 50
```
Outputs: `results/exp3_determinism.csv` (per-scenario agreement / confidence).

### Figures
```bash
python src/experiments/plots.py --input results/exp2_runs.csv --outdir results/figs
python src/experiments/plots.py --input results/exp3_determinism.csv --outdir results/figs
```

## 4. Metrics
* **Regret** = `filter_best_loss − real_best_loss` (+ `rank_pct`).
* **Savings A** (no latency) = `epoch_time × epochs_avoided`;
  **Savings B** = `A − Σ min(llm_latency, epoch_time)` over prunes.
* **False Continues / False Prunes** vs the per-epoch ground truth
  (`current_val ≤ best_val_at_epoch × 1.05`, phmlc's rule).
* **Confidence** (agreement across samples) and **tokens / latency**.

## 5. Architecture
Flat `src/`-layout (no package wrapper, only
`policies/` is a real subpackage): `phmlc_bridge` (only phmlc boundary) · `config` ·
`backend` (+`MockBackend`) · `tracking` · `scenario` (no-leak bundle + builder) ·
`prompting` (L0–L3 + parser) · `policies/` (`llm`, `random`, `last_seen`, `arima`,
`oracle`) · `simulator` (TPE replay + per-epoch decision + patience) · `metrics` ·
`dataset` (real + synthetic) · `runner` (glue). `src/experiments/` holds the
runnable scripts above; `src/tests/` the unit tests.

## 6. Limitations (thesis)
Models limited to those served by Ollamus; times are relative to hardware + network
latency; not all combinations are run (time budget); the per-epoch pruning ground
truth uses phmlc's `×1.05` heuristic; the ARIMA grid is bounded (phmlc's full grid
is intractable per-epoch) and its order is selected once per run then reused.


## Disclaimer
This README may not reflect the latest implementation details or be fully up to date. 
Its purpose is to provide a high-level overview of the repository along with a brief guide to help get started.
