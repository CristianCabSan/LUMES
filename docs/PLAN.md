# Plan TFM — `LUMES`: Juez LLM de Early Stopping para Optimización Bayesiana sobre curvas

> **Documento extraíble.** Pensado para ejecutarse en otro entorno. Cópialo a `LUMES/docs/PLAN.md`.
> **Ubicación de trabajo**: `C:\Users\Cristian\Documents\Obsidian Vault\Master\TFM\` con dos repos hermanos:
> `TFM/phmlc/` (framework existente, **READ-ONLY**) y `TFM/LUMES/` (**NUEVO**, todo nuestro código).
> Todos los paths de este plan son **relativos a `TFM/LUMES/`**; a `phmlc` se le referencia como `../phmlc`.

## Reglas duras (acordadas)
1. **No se modifica NADA de `phmlc`.** Se consume como librería: solo se importan/llaman sus funciones.
2. Todo el código nuevo vive en el repo independiente **`LUMES`** (proyecto Python propio).
3. **Baselines: únicamente `random`, `last-seen`, `arima` + `oracle`.** Nada de hyperband/bohb/FSL/NN.
4. El plan **no se ejecuta aquí**; se entrega como documento para otro entorno.

## Contexto

`phmlc` no entrena redes en tiempo de experimento: **replica curvas de aprendizaje pre-computadas** (62k curvas, 59 datasets) de la meta-dataset `CURVES` (vía `phmd`) y simula optimización bayesiana con poda (early stopping). El TFM introduce un **juez LLM** (API Ollamus de la US) que decide CONTINUE/PRUNE **época a época** con información **solo del pasado/presente**, y lo compara contra los 3 baselines + un Oracle. Simulación de tiempo: en CONTINUE coste ≈ 0 (la época N+1 corre en paralelo a la llamada); en PRUNE el tiempo perdido ≤ latencia del LLM.

### Hechos de `phmlc` que reutilizamos (solo lectura)
- **Carga de datos**: `from phm_framework.optimization.curves import load_curves` ([../phmlc/src/phm_framework/optimization/curves/__init__.py](../phmlc/src/phm_framework/optimization/curves/__init__.py)).
  - `load_curves(fold, num_folds, filters={"data":"curves"}, test_dataset_names, random_state)` → `{'train','val','test'}` de DataFrames con una fila por época: `unit, dataset, task, net, train_loss, val_loss, num_epochs`. Split **por dataset**.
  - `load_curves(None, num_folds=0, filters={"data":"results"})` → una fila por `unit`: hiperparámetros `model__*` + `train__time` (tiempo total del run). `epoch_time = train__time / num_epochs`.
- **Identificadores**: `unit` = un run; `experiment = dataset_task_net`; `experiment_id = "_".join(unit.split("_")[:-1])`.
- **Ingeniería de features pasadas** (referencia a replicar): `prepare_decision_data`/`extended_decision_data` en [../phmlc/.../curves/train.py:1113](../phmlc/src/phm_framework/optimization/curves/train.py#L1113) → `val_improvement`, `val_velocity` (diff), `val_ema` (ewm span 3), `expected_improvement`, `best_performance`, `best_val_at_epoch`, y verdad-terreno por época `should_continue_at_epoch = current_val <= best_val_at_same_epoch * 1.05`.
- **Patrón de simulación a imitar** (no a importar): `BOPredictiveSimulator.run_once` en [../phmlc/.../curves/fsldt.py:164](../phmlc/src/phm_framework/optimization/curves/fsldt.py#L164) — Optuna **TPESampler** fija el orden BO, replay por época, decisión con `patience=3`.
- **Reglas exactas de los baselines a replicar**:
  - `random` ([train.py:1993](../phmlc/src/phm_framework/optimization/curves/train.py#L1993)): continúa con prob. `random_pct`; decide una vez (ts_len=2).
  - `last-seen` ([train.py:1809-1817](../phmlc/src/phm_framework/optimization/curves/train.py#L1809)): `pred = último val parcial`; poda si `pred >= filter_best*2`.
  - `arima` ([train.py:1599-1632](../phmlc/src/phm_framework/optimization/curves/train.py#L1599)): ajusta ARIMA al tramo parcial, forecast a 100−ts_len; poda si `forecast >= filter_best*2`.

---

## 1) Estructura del repositorio nuevo `LUMES`

```
TFM/
├── phmlc/                      # READ-ONLY (framework)
└── LUMES/                     # NUEVO repo (todo el código del TFM)
    ├── pyproject.toml          # deps: phmd, requests, optuna, pandas, numpy, statsmodels, scikit-learn, matplotlib
    ├── README.md
    ├── .env.example            # LLM_BASE_URL, LLM_API_KEY, LLM_MODEL, PHMLC_SRC=../phmlc/src
    ├── docs/PLAN.md            # este documento
    ├── src/LUMES/
    │   ├── __init__.py
    │   ├── config.py           # dataclasses: LLMConfig, JudgeConfig, SimConfig, ExperimentConfig, ContextLevel(L0..L3)
    │   ├── phmlc_bridge.py      # ÚNICO módulo que importa phm_framework: load_curves, features, phmd
    │   ├── backend.py           # script provisto (Backend/OllamaBackend/ChatResponse/build_backend_from_env) + MockBackend
    │   ├── tracking.py          # LatencyTracker / CostTracker (agrega ChatResponse.latency_s + tokens)
    │   ├── scenario.py          # DecisionScenario + ScenarioBuilder (estrictamente pasado/presente)
    │   ├── prompting.py         # PromptBuilder (L0..L3) -> (system, user) + parse_decision()
    │   ├── policies/
    │   │   ├── base.py          # Policy ABC: decide(scenario) -> Decision
    │   │   ├── llm.py           # LLMPolicy (el juez; n_samples, confidence, caché)
    │   │   ├── random_policy.py # mirror del random de phmlc
    │   │   ├── last_seen.py     # mirror del last-seen de phmlc
    │   │   ├── arima_policy.py  # mirror del arima de phmlc
    │   │   └── oracle.py        # OraclePolicy (límite superior, regret 0)
    │   ├── simulator.py         # BOSimulator: TPE order + decisión por época + patience -> MetricsCollector
    │   └── metrics.py           # MetricsCollector: regret, ahorro A/B, FC/FP, confidence, tokens
    ├── experiments/
    │   ├── sanity_check.py      # curva manual -> juez (Fase 3)
    │   ├── exp1_select_llm.py   # varios LLMs sobre subconjunto reducido -> elegir mejor
    │   ├── exp2_vs_baselines.py # mejor LLM vs random/last-seen/arima/oracle, conjunto grande
    │   ├── exp3_determinism.py  # determinismo/confidence con el mismo LLM
    │   └── plots.py             # regret, ahorro A/B, FC/FP, confidence
    ├── results/                # CSVs + figuras (gitignore salvo resúmenes)
    └── tests/                  # no-fuga, parser, métricas, policies (con MockBackend)
```

### Acoplamiento a `phmlc` (sin modificarlo)
`phmlc` no tiene packaging (solo `src/` + `environment.txt`). En `phmlc_bridge.py`, antes de importar, se añade su `src` al path:
```python
import os, sys
sys.path.insert(0, os.environ.get("PHMLC_SRC", os.path.join("..", "phmlc", "src")))
from phm_framework.optimization.curves import load_curves
```
`phmd` se instala como dependencia normal (PyPI `phmd==2015.0.1`, igual que en `phmlc/environment.txt`). **`phmlc_bridge` es la única frontera**: todo el resto de `LUMES` ignora que `phmlc` existe.

---

## 2) Diseño de clases / interfaces clave

### `phmlc_bridge.py` — frontera única con el framework
```
load_experiment_curves(filters, test_dataset_names, random_state, num_folds, fold) -> dict[split -> DataFrame]
load_results() -> DataFrame                      # {"data":"results"}: hiperparámetros + train__time por unit
to_curve_dict(df) -> {unit: np.ndarray(n,2)}     # col0=train_loss, col1=val_loss
build_past_features(unit_curve, epoch) -> dict   # replica prepare_decision_data truncado a 'epoch'
```

### `backend.py` — transporte (script provisto, adoptado tal cual)
API nativa de Ollama (`POST {base_url}/api/chat`) tras el proxy Ollamus (auth `Bearer`), reintentos sobre `Timeout`/`ConnectionError`, credenciales por env. Interfaz: `Backend.chat(system, user) -> ChatResponse(text, input_tokens, output_tokens, latency_s, raw)`; `build_backend_from_env(backend, model, temperature, timeout)`. Añadidos:
- `MockBackend(Backend)`: `ChatResponse` guionizadas para tests y `--debug` sin red.
- Barrido de LLMs (Fase exp1) = cambiar `LLM_MODEL`, **sin tocar código**.

### `tracking.py` — `LatencyTracker` / `CostTracker`
`ChatResponse` ya trae `latency_s` y tokens → el tracker solo **agrega**: latencias (alimenta Variante B de ahorro) y tokens (coste por experimento). No re-mide tiempo.

### `scenario.py` — `DecisionScenario` (garantía de no-fuga)
Dataclass inmutable; **toda** la info que ve cualquier policy pasa por aquí; nada de `val[e+1:]`, final loss ni `num_epochs` real.
```
unit_id, experiment_id, epoch
partial_train: list[float]   # train_loss[:e+1]
partial_val:   list[float]   # val_loss[:e+1]
features: dict               # val_improvement, val_velocity, val_ema, expected_improvement (solo pasado)
bo_state: dict               # best_so_far_final_val, best_val_at_epoch, n_trials_done, in_warmup
meta: dict                   # dataset, task, net (+ hiperparámetros según ContextLevel)
```
`ScenarioBuilder` usa `phmlc_bridge.build_past_features`. Test obligatorio: ninguna clave depende de datos posteriores a `epoch`.

### `prompting.py` — `PromptBuilder` (niveles de contexto) + parser
`build(scenario, context_level) -> (system, user)`. Salida estricta `DECISION: CONTINUE|PRUNE`. Niveles para exp1:
- **L0** solo curva val parcial · **L1** + features derivadas · **L2** + estado BO (best-so-far, warmup) · **L3** + tarea/arquitectura/hiperparámetros.
`parse_decision(text) -> CONTINUE|PRUNE` robusto (default seguro = CONTINUE si no parsea).

### `policies/` — `Policy` común para TODOS los métodos (comparación justa)
```
class Policy(ABC): decide(scenario) -> Decision(action, confidence=1.0, latency_s=0.0, tokens=0)
```
- `LLMPolicy`: construye `(system,user)`, llama `backend.chat` `n_samples` veces (config, **def. 1**), voto mayoritario, `confidence = % acuerdo`, caché por hash de scenario.
- `RandomPolicy` / `LastSeenPolicy` / `ArimaPolicy`: replican las reglas exactas de `phmlc` (referencias arriba), evaluadas en el **mismo** harness/splits/cadencia.
- `OraclePolicy`: ve la curva completa → decisión perfecta (`filter_best == real_best`, **regret 0**) y máximo ahorro (cada run no-ganador se corta en el primer checkpoint; el ganador corre hasta su época argmin). Límite superior.

### `simulator.py` — `BOSimulator` (imita `BOPredictiveSimulator`, código propio)
Por experimento, Optuna **TPESampler** fija el orden BO; los **3 primeros trials = warmup** (sin poda, fijan best-so-far); desde `checkpoint_epoch` (def. 10) se consulta a la `Policy` **en cada época**, con `patience` para confirmar PRUNE (monotonicidad). Acumula vía `MetricsCollector`. Métodos `run_once(seed)`, `run_montecarlo`, `run_all_experiments`. La policy es inyectable → LLM, baselines y oracle comparten exactamente el mismo bucle.

### `metrics.py` — `MetricsCollector`
Por-experimento y global; emite CSV propio (limpio) con:
- **Regret** = `filter_best_loss - real_best_loss` (+ `rank_pct`), media/mediana/IC.
- **Ahorro Variante A** (sin latencia): `saved_time_A = epoch_time * epochs_avoided`.
- **Ahorro Variante B** (con latencia): `saved_time_B = saved_time_A - Σ min(llm_latency, epoch_time)` sobre PRUNEs.
- **False Continues / False Prunes**: decisión por época vs `should_continue_at_epoch`.
- **Confidence** (exp3): % acuerdo con `n_samples`. **Tokens/coste** por experimento.

---

## 3) Plan de implementación por fases (para tu OK)

- **Fase 0 — Scaffolding `LUMES`**: `pyproject.toml`, `.env.example`, `phmlc_bridge.py` (path a `../phmlc/src`), `config.py`. Smoke-test: `load_curves` devuelve curvas.
- **Fase 1 — Datos/no-fuga**: `DecisionScenario` + `ScenarioBuilder` reutilizando features de `phmlc`; tests de no-fuga.
- **Fase 2 — Backend + trackers**: integrar `backend.py` provisto, `MockBackend`, `LatencyTracker`/`CostTracker`; smoke-test real contra Ollamus con `LLM_*`.
- **Fase 3 — Prompt + LLMPolicy + `sanity_check.py`**: `(system,user)`; curvas guionizadas (creciente/plateau/divergente) → decisión esperada.
- **Fase 4 — BOSimulator + MetricsCollector + Policies baseline/oracle**: harness único; corrida `--debug` con `MockBackend`.
- **Fase 5 — Experimentos** (3 etapas exactas pedidas):
  1. **`exp1_select_llm.py`** — varios LLMs (cambiando `LLM_MODEL`) sobre **subconjunto reducido** de curvas (pocos experimentos / `--debug`-like) para no tardar; barrido de niveles de contexto L0..L3; se **elige el mejor LLM+contexto** por score (regret bajo + ahorro alto).
  2. **`exp2_vs_baselines.py`** — el **mejor LLM** vs `random/last-seen/arima/oracle` sobre **conjunto grande** (mismos splits), N seeds Monte Carlo → estudio estadístico robusto (medias, IC, tests de significancia, gráficas regret/ahorro A-B/FC-FP).
  3. **`exp3_determinism.py`** — **mismo LLM**, repetir escenarios idénticos `n_samples` veces (configurable) → confidence/determinismo.

### Limitaciones a documentar (tesis)
Modelos limitados a los de Ollamus; tiempos relativos a hardware + latencia de red; no se ejecutan todas las combinaciones por límite temporal; verdad-terreno de poda basada en la heurística `*1.05` de `phmlc`.

---

## 4) Verificación

- **Tests unitarios** (`tests/`, con `MockBackend`):
  - No-fuga: ninguna clave de `DecisionScenario`/prompt depende de info > época actual.
  - `parse_decision` robusto; default seguro CONTINUE.
  - Métricas: regret, FC/FP, `saved_time_A/B` sobre curvas sintéticas conocidas.
  - Baselines: `RandomPolicy/LastSeenPolicy/ArimaPolicy` reproducen las decisiones de `phmlc` en casos de prueba.
- **End-to-end debug**: `python experiments/exp2_vs_baselines.py --debug` con `MockBackend` → genera CSVs en `results/` con el esquema esperado.
- **Sanity check**: `experiments/sanity_check.py` con curvas guionizadas → decisiones esperadas.
- **Coherencia de cotas**: `oracle ≥ LLM ≥ baselines` en ahorro y `regret(oracle) ≈ 0`.
- **Determinismo**: repetir el mismo scenario `n_samples` veces y verificar cálculo/registro de `confidence`.

---

## Pendiente de tu parte (otro entorno)
- Variables de entorno reales: `LLM_BASE_URL`, `LLM_API_KEY`, `LLM_MODEL`, `PHMLC_SRC=../phmlc/src`.
- Lista de LLMs disponibles en Ollamus para el barrido de exp1.
- Confirmar tamaños de "subconjunto reducido" (exp1) y "conjunto grande" (exp2) y nº de seeds Monte Carlo.
