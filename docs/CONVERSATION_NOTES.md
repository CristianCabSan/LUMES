# Notas de conversación — Diseño del TFM "Juez LLM de Early Stopping" (repo `LUMES`)

Fecha: 2026-06-21
Participantes: Cristian (autor TFM) + asistente (rol: Senior ML Engineer / tutor)

## 1. Objetivo del TFM
Integrar un **LLM como juez de early stopping** para optimización bayesiana en mantenimiento
predictivo (PHM). El juez decide **CONTINUE / PRUNE** época a época sobre **curvas de aprendizaje
pre-entrenadas** (no se entrena ninguna red en tiempo de experimento), usando solo información
del **pasado/presente**, y se compara contra baselines y un Oracle.

Simulación de tiempo real: el modelo "entrena" hasta la época N, llama al LLM y empieza la N+1 en
paralelo. Si dice CONTINUE el tiempo perdido ≈ 0; si dice PRUNE el tiempo perdido ≤ latencia del LLM.

## 2. Análisis del framework base `phmlc` (READ-ONLY)
- No entrena redes: **replica curvas pre-computadas** (62k curvas, 59 datasets) de la meta-dataset
  `CURVES` cargada con `phmd`.
- Dos vistas de datos:
  - `{"data":"curves"}` → una fila por época: `unit, dataset, task, net, train_loss, val_loss, num_epochs`.
    En los generadores, cada curva es `np.array(n, 2)` (col0=train_loss, col1=val_loss).
  - `{"data":"results"}` → una fila por `unit`: hiperparámetros `model__*` + `train__time` (tiempo
    total del run). `epoch_time = train__time / num_epochs`.
- Identificadores: `unit` = un run; `experiment = dataset_task_net`;
  `experiment_id = "_".join(unit.split("_")[:-1])`. **Split por dataset** (generalización cross-dataset).
- Punto de carga reutilizable: `phm_framework.optimization.curves.load_curves(...)`.
- Mecánica común a todos los métodos: ordenar runs → reproducir curva → decidir CONTINUE/PRUNE →
  registrar `epochs_avoided`, `avoided_train_time`, `filter_best_losses` vs `real_best_losses`,
  `num_prunings`, `num_runs`.
- Mejor patrón a imitar para el juez: `BOPredictiveSimulator.run_once` (fsldt.py) — **Optuna TPESampler**
  fija el orden BO, replay por época, decisión con `patience=3`.
- Features pasadas ya implementadas (a replicar): `val_improvement`, `val_velocity` (diff),
  `val_ema` (ewm span 3), `expected_improvement`, `best_performance`, `best_val_at_epoch`, y la
  verdad-terreno por época `should_continue = current_val <= best_val_at_same_epoch * 1.05`.

## 3. Decisiones tomadas
- **Transporte LLM**: API nativa de **Ollama `/api/chat`** tras el proxy **Ollamus** (US), usando el
  `backend.py` que ya tiene Cristian (`Backend`/`OllamaBackend`/`ChatResponse`/`build_backend_from_env`).
  Credenciales por entorno: `LLM_BASE_URL`, `LLM_API_KEY`, `LLM_MODEL`. `ChatResponse` ya da
  `latency_s` y tokens.
- **Motor de simulación**: Optuna **TPE por época** (estilo `BOPredictiveSimulator`).
- **Cadencia del juez**: decisión en **cada época desde el checkpoint** (def. época 10), con `patience`.
- **Determinismo**: `n_samples` **configurable**, por defecto 1; el estudio de confidence se hace al
  final con **un único LLM**.
- **Repositorio separado `LUMES`**: proyecto Python independiente, hermano de `phmlc`. Consume
  `phmlc` como librería **sin modificarlo** (frontera única: `phmlc_bridge.py`, que añade
  `../phmlc/src` al `sys.path`).
- **Baselines**: solo **random, last-seen, arima** + **oracle**. Nada de hyperband/bohb/FSL/NN.
- **Comparación justa**: baselines y oracle se reimplementan como `Policy` dentro del **mismo harness**
  (mismo orden TPE, mismos splits, mismas métricas) que el juez LLM.

## 4. Arquitectura propuesta (resumen)
Repo `LUMES/` con `src/LUMES/`:
- `phmlc_bridge.py` — única frontera con `phm_framework` (carga curvas/resultados + features).
- `backend.py` — script provisto + `MockBackend` para tests.
- `tracking.py` — agrega `latency_s` y tokens de `ChatResponse`.
- `scenario.py` — `DecisionScenario` (bundle estrictamente pasado/presente, sin fugas) + builder.
- `prompting.py` — `PromptBuilder` con niveles de contexto **L0..L3** → `(system, user)` + `parse_decision`.
- `policies/` — `Policy` ABC + `LLMPolicy`, `RandomPolicy`, `LastSeenPolicy`, `ArimaPolicy`, `OraclePolicy`.
- `simulator.py` — `BOSimulator` (TPE order + decisión por época + patience), policy inyectable.
- `metrics.py` — `MetricsCollector`.
- `experiments/` — `sanity_check.py`, `exp1_select_llm.py`, `exp2_vs_baselines.py`, `exp3_determinism.py`, `plots.py`.
- `tests/` — no-fuga, parser, métricas, policies (con `MockBackend`).

## 5. Métricas
- **Regret** = `filter_best_loss - real_best_loss` (+ `rank_pct`).
- **Ahorro Variante A** (sin latencia LLM) y **Variante B** (restando `min(latencia, epoch_time)` por PRUNE).
- **False Continues / False Prunes** vs verdad-terreno por época.
- **Confidence/determinismo** = % de acuerdo con `n_samples`.
- **Coste**: tokens y latencia por experimento.

## 6. Proceso de experimentos (3 etapas)
1. Varios LLMs sobre un **subconjunto reducido** de curvas (rápido) + barrido de contexto L0..L3 →
   elegir el **mejor LLM+contexto**.
2. El **mejor LLM** vs **random/last-seen/arima/oracle** sobre un **conjunto grande**, N seeds Monte
   Carlo → estudio estadístico robusto.
3. **Estudio de determinismo** con el **mismo LLM** (`n_samples` configurable).

## 7. Pendientes (a fijar en el entorno de ejecución)
- Env vars reales: `LLM_BASE_URL`, `LLM_API_KEY`, `LLM_MODEL`, `PHMLC_SRC=../phmlc/src`.
- Lista de LLMs disponibles en Ollamus para exp1.
- Tamaños de "subconjunto reducido" (exp1) y "conjunto grande" + nº de seeds (exp2).

## 8. Reglas duras del proyecto
1. No modificar NADA de `phmlc`.
2. Todo el código nuevo vive en `LUMES`.
3. Baselines = solo random/last-seen/arima + oracle.
4. Plan detallado: `C:\Users\Cristian\.claude\plans\act-a-como-un-senior-compiled-coral.md`
   → copiar a `LUMES/docs/PLAN.md`.
