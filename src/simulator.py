"""``BOSimulator`` — the single harness shared by every policy.

It re-implements (in our own code, importing nothing from phmlc) the mechanics of
``BOPredictiveSimulator.run_once``:

* Optuna **TPESampler** fixes the BO order; suggested hyper-parameters are mapped to
  the **nearest real unit** (Euclidean over numeric ``model__*`` columns), whose
  pre-computed curve is replayed.
* The first ``warmup_trials`` trials are never pruned (they set the incumbent).
* From ``checkpoint_epoch`` the injected :class:`Policy` is queried **every epoch**;
  ``patience`` consecutive PRUNE votes confirm a prune (monotonicity guard).

Because the policy is injected, the LLM judge, the baselines and the oracle all run
through this identical loop — the comparison is apples to apples.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import optuna

import phmlc_bridge
from config import SimConfig
from metrics import ExperimentResult
from policies.base import Policy
from scenario import ScenarioBuilder

optuna.logging.set_verbosity(optuna.logging.WARNING)

# phmlc's categorical hyper-parameters (fsldt.py:104)
CATEGORICAL_PARAMS = [
    "model__activation", "model__batch_normalization", "model__conv_activation",
    "model__dense_activation", "model__kernel_size", "model__bidirectional",
    "model__cell_type",
]


@dataclass
class ExperimentData:
    """All pre-computed data for one experiment (a dataset_task_net group)."""

    experiment_id: str
    curves: Dict[str, np.ndarray]          # unit -> ndarray(n_epochs, 2)
    epoch_time: Dict[str, float]           # unit -> train__time / n_epochs
    meta: Dict[str, dict]                  # unit -> {dataset, task, net, hyperparameters}
    param_ranges: Dict[str, object]        # col -> (lo, hi) | list (categorical)
    categorical_params: List[str] = field(default_factory=list)
    numeric_cols: List[str] = field(default_factory=list)
    matrix_uids: np.ndarray = None
    matrix_values: np.ndarray = None

    @property
    def unit_ids(self) -> List[str]:
        return list(self.curves.keys())

    def nearest_unit(self, suggested: dict) -> str:
        if not self.numeric_cols or self.matrix_values is None or len(self.matrix_values) == 0:
            return self.matrix_uids[0]
        vec = np.array([suggested[c] for c in self.numeric_cols], dtype=float)
        d = np.sum((self.matrix_values - vec) ** 2, axis=1)
        return str(self.matrix_uids[int(np.argmin(d))])


class BOSimulator:
    def __init__(
        self,
        policy: Policy,
        sim: Optional[SimConfig] = None,
        scenario_builder: Optional[ScenarioBuilder] = None,
        dump_decisions: bool = False,
        full_trajectory: bool = False,
    ):
        self.policy = policy
        self.sim = sim or SimConfig()
        self.builder = scenario_builder or ScenarioBuilder()
        # dump_decisions: record every query made into ``decision_log``.
        # full_trajectory: keep querying past the effective stop to the end of the
        #   curve (lets any patience/checkpoint be re-derived offline, but ~Nx more
        #   LLM calls for aggressive policies). Default off -> record only up to stop.
        self.dump_decisions = dump_decisions
        self.full_trajectory = full_trajectory
        self.decision_log = []

    # ------------------------------------------------------------------ #
    def run_once(self, exp: ExperimentData, seed: int = 0) -> ExperimentResult:
        s = self.sim
        self.policy.new_run(seed)
        self.policy.set_experiment(exp)

        # incumbent state
        best_final = np.inf
        best_val_curve: Optional[np.ndarray] = None
        all_finals: List[float] = []
        n_trials_done = {"v": 0}

        # accumulators
        acc = {
            "total_epochs": 0, "epochs_avoided": 0, "total_train_time": 0.0,
            "saved_time_A": 0.0, "saved_B_penalty": 0.0,
            "num_runs": 0, "num_prunings": 0,
            "n_decisions": 0, "false_continues": 0, "false_prunes": 0,
            "confidences": [],
        }

        # cost/latency snapshot (LLM policies expose trackers)
        cost = getattr(self.policy, "cost", None)
        lat = getattr(self.policy, "latency", None)
        c0 = (cost.input_tokens, cost.output_tokens, cost.n_calls) if cost else (0, 0, 0)
        l0 = lat.total_s if lat else 0.0

        def objective(trial: optuna.Trial):
            nonlocal best_final, best_val_curve

            suggested = {}
            for k, v in exp.param_ranges.items():
                if k in exp.categorical_params:
                    suggested[k] = trial.suggest_categorical(k, v)
                else:
                    lo, hi = float(v[0]), float(v[1])
                    log = lo > 0 and hi > lo
                    suggested[k] = trial.suggest_float(k, lo, hi, log=log)

            unit = exp.nearest_unit(suggested)
            curve = exp.curves[unit]
            n_ep = curve.shape[0]
            final_val = float(curve[-1, phmlc_bridge.VAL_COL])
            all_finals.append(final_val)

            in_warmup = n_trials_done["v"] < s.warmup_trials
            epoch_time = float(exp.epoch_time.get(unit, 1.0))

            # checkpoint: fixed, or adaptive to this curve's length (so short curves
            # still get enough decision points for `patience` to be reachable)
            if s.checkpoint_frac:
                checkpoint = max(s.min_checkpoint, int(s.checkpoint_frac * n_ep))
            else:
                checkpoint = s.checkpoint_epoch

            is_pruned = False
            stop_epoch = n_ep
            prune_latency = 0.0

            if not in_warmup:
                stop_counter = 0
                best_known = best_final if np.isfinite(best_final) else None
                trial_idx = n_trials_done["v"]
                for e in range(checkpoint, n_ep):
                    bve = (
                        float(best_val_curve[min(e, len(best_val_curve) - 1)])
                        if best_val_curve is not None else None
                    )
                    bo_state = {
                        "best_so_far_final_val": best_known,
                        "best_val_at_epoch": bve,
                        "n_trials_done": n_trials_done["v"],
                        "in_warmup": False,
                    }
                    scenario = self.builder.build(
                        unit, exp.experiment_id, curve, e, bo_state, exp.meta.get(unit, {})
                    )
                    decision = self.policy.decide(scenario)
                    gt_continue = phmlc_bridge.ground_truth_continue(
                        scenario.partial_val[-1], bve
                    )

                    # full-granularity dump: record EVERY query (even past the effective
                    # stop), so any patience/checkpoint can be re-derived offline.
                    if self.dump_decisions:
                        self.decision_log.append({
                            "policy": self.policy.name, "experiment_id": exp.experiment_id,
                            "seed": seed, "unit_id": unit, "trial": trial_idx, "epoch": e,
                            "n_epochs": n_ep, "checkpoint": checkpoint,
                            "action": decision.action, "confidence": decision.confidence,
                            "latency_s": decision.latency_s, "tokens": decision.tokens,
                            "votes": "|".join(decision.votes) if decision.votes else decision.action,
                            "current_val": scenario.partial_val[-1],
                            "best_so_far_final_val": best_known, "best_val_at_epoch": bve,
                            "ground_truth_continue": gt_continue,
                            "label": ("FP" if (decision.is_prune() and gt_continue)
                                      else "FC" if (decision.is_continue() and not gt_continue)
                                      else "OK"),
                            "after_effective_stop": is_pruned,
                            "final_val": final_val, "epoch_time": epoch_time,
                            "raw": (decision.raw or "")[:2000],
                        })

                    # metrics: only the pre-stop decisions count (identical to non-dump)
                    if not is_pruned:
                        acc["n_decisions"] += 1
                        if decision.is_prune() and gt_continue:
                            acc["false_prunes"] += 1
                        elif decision.is_continue() and not gt_continue:
                            acc["false_continues"] += 1
                        acc["confidences"].append(decision.confidence)

                        if decision.is_prune():
                            stop_counter += 1
                        else:
                            stop_counter = 0
                        if stop_counter >= s.patience:
                            is_pruned = True
                            stop_epoch = e + 1
                            prune_latency = decision.latency_s
                            if not self.full_trajectory:
                                break  # only full-trajectory mode keeps querying

            epochs_run = stop_epoch if is_pruned else n_ep
            epochs_avoided = n_ep - epochs_run

            acc["total_epochs"] += n_ep
            acc["epochs_avoided"] += epochs_avoided
            acc["total_train_time"] += epoch_time * n_ep
            acc["saved_time_A"] += epoch_time * epochs_avoided
            acc["num_runs"] += 1
            if is_pruned:
                acc["num_prunings"] += 1
                acc["saved_B_penalty"] += min(prune_latency, epoch_time)

            n_trials_done["v"] += 1

            # report intermediate values for TPE bookkeeping (like phmlc)
            for step in range(epochs_run):
                trial.report(float(curve[step, phmlc_bridge.VAL_COL]), step=step + 1)

            if is_pruned:
                raise optuna.TrialPruned()

            # completed run -> may update incumbent
            if final_val < best_final:
                best_final = final_val
                best_val_curve = curve[:, phmlc_bridge.VAL_COL].copy()
            return final_val

        sampler = optuna.samplers.TPESampler(seed=seed)
        study = optuna.create_study(
            direction="minimize" if s.minimize else "maximize", sampler=sampler
        )
        n_trials = min(s.n_trials_cap, len(exp.unit_ids))
        if n_trials > 0:
            study.optimize(objective, n_trials=n_trials, catch=())

        # ---- finalize ---- #
        filter_best = best_final
        real_best = min(all_finals) if all_finals else np.inf
        if not np.isfinite(filter_best):                 # all pruned
            filter_best = max(all_finals) if all_finals else np.inf
        regret = float(filter_best - real_best)

        rank_pct = 0.0
        if all_finals:
            ordered = sorted(all_finals)
            # index of the first value >= filter_best (robust to ties / inf)
            idx = int(np.searchsorted(ordered, filter_best))
            rank_pct = idx / max(1, len(ordered) - 1)

        saved_B = acc["saved_time_A"] - acc["saved_B_penalty"]
        conf = float(np.mean(acc["confidences"])) if acc["confidences"] else 1.0

        return ExperimentResult(
            policy=self.policy.name,
            experiment_id=exp.experiment_id,
            seed=seed,
            filter_best_loss=float(filter_best),
            real_best_loss=float(real_best),
            regret=regret,
            rank_pct=float(rank_pct),
            num_runs=acc["num_runs"],
            num_prunings=acc["num_prunings"],
            total_epochs=acc["total_epochs"],
            epochs_avoided=acc["epochs_avoided"],
            total_train_time=acc["total_train_time"],
            saved_time_A=acc["saved_time_A"],
            saved_time_B=saved_B,
            n_decisions=acc["n_decisions"],
            false_continues=acc["false_continues"],
            false_prunes=acc["false_prunes"],
            llm_calls=(cost.n_calls - c0[2]) if cost else 0,
            input_tokens=(cost.input_tokens - c0[0]) if cost else 0,
            output_tokens=(cost.output_tokens - c0[1]) if cost else 0,
            total_latency_s=(lat.total_s - l0) if lat else 0.0,
            mean_confidence=conf,
        )

    # ------------------------------------------------------------------ #
    def run_montecarlo(self, exp: ExperimentData, seeds: List[int]) -> List[ExperimentResult]:
        return [self.run_once(exp, seed) for seed in seeds]

    def run_all_experiments(
        self, experiments: List[ExperimentData], seeds: List[int]
    ) -> List[ExperimentResult]:
        out: List[ExperimentResult] = []
        for exp in experiments:
            out.extend(self.run_montecarlo(exp, seeds))
        return out
