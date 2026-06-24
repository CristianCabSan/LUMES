"""Assemble :class:`ExperimentData` objects.

Two sources:

* :func:`build_experiments_from_phmlc` — real CURVES data (curves view + results
  view) grouped into ``dataset_task_net`` experiments, mirroring phmlc's hp handling
  (categorical columns, numeric nearest-neighbour matrix, ``epoch_time =
  train__time / n_epochs``).
* :func:`make_synthetic_experiments` — fully offline curves for unit tests, the
  ``--debug`` path and the sanity check (no TensorFlow, no network, no download).
"""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import pandas as pd

import phmlc_bridge
from simulator import CATEGORICAL_PARAMS, ExperimentData

EXCLUDE_HP = {"model__input_shape", "model__output", "model__net", "model__output_dim"}


# --------------------------------------------------------------------------- #
# Real phmlc data
# --------------------------------------------------------------------------- #
def _build_one(
    experiment_id: str,
    dataset: str,
    task: str,
    net: str,
    curves: Dict[str, np.ndarray],
    results: pd.DataFrame,
) -> Optional[ExperimentData]:
    units = [u for u in curves if u in results.index]
    if len(units) == 0:
        return None
    res = results.loc[units]

    hp_cols = [c for c in res.columns if "model__" in c and c not in EXCLUDE_HP]
    hp_cols = [c for c in hp_cols if not res[c].isnull().all()]
    categorical = [c for c in hp_cols if c in CATEGORICAL_PARAMS]
    numeric_cols = [
        c for c in hp_cols
        if c not in categorical and pd.api.types.is_numeric_dtype(res[c])
    ]

    param_ranges: Dict[str, object] = {}
    for c in hp_cols:
        if c in categorical:
            param_ranges[c] = sorted(res[c].dropna().unique().tolist())
        else:
            col = res[c].dropna().astype(float)
            if len(col) == 0:
                continue
            param_ranges[c] = (float(col.min()), float(col.max()))
    # keep only columns that ended up with a range
    categorical = [c for c in categorical if c in param_ranges]
    numeric_cols = [c for c in numeric_cols if c in param_ranges]

    matrix_uids = np.array(units, dtype=object)
    if numeric_cols:
        matrix_values = res[numeric_cols].astype(float).fillna(0.0).to_numpy()
    else:
        matrix_values = np.zeros((len(units), 0))

    epoch_time: Dict[str, float] = {}
    meta: Dict[str, dict] = {}
    for u in units:
        n_ep = max(1, curves[u].shape[0])
        tt = res.loc[u, "train__time"] if "train__time" in res.columns else np.nan
        tt = float(np.nanmean(np.atleast_1d(tt)))
        epoch_time[u] = (tt / n_ep) if np.isfinite(tt) and tt > 0 else 1.0
        hp = {c: res.loc[u, c] for c in hp_cols}
        meta[u] = {"dataset": dataset, "task": task, "net": net, "hyperparameters": hp}

    return ExperimentData(
        experiment_id=experiment_id,
        curves={u: curves[u] for u in units},
        epoch_time=epoch_time,
        meta=meta,
        param_ranges=param_ranges,
        categorical_params=categorical,
        numeric_cols=numeric_cols,
        matrix_uids=matrix_uids,
        matrix_values=matrix_values,
    )


def build_experiments_from_phmlc(
    curves_df: pd.DataFrame,
    results_df: pd.DataFrame,
    min_units: int = 2,
    n_experiments: Optional[int] = None,
    spread_datasets: bool = False,
) -> List[ExperimentData]:
    """Group a curves DataFrame into per-experiment :class:`ExperimentData`.

    With ``spread_datasets=True`` the selection is **round-robin across datasets**
    (one group per dataset, cycling) instead of the first-N groups — so a small
    ``n_experiments`` spans many datasets (cross-dataset generalisation) rather than
    clustering inside whichever dataset appears first.
    """
    results = results_df.copy()
    if results.index.name != "unit":
        results = results.groupby("unit").agg(
            lambda x: x.mean() if pd.api.types.is_numeric_dtype(x) else x.iloc[0]
        )

    group_cols = ["dataset", "task", "net"]
    grouped = curves_df.groupby(group_cols, sort=False)
    keys = list(grouped.groups.keys())

    if spread_datasets:
        from collections import OrderedDict
        by_ds = OrderedDict()
        for k in keys:
            by_ds.setdefault(k[0], []).append(k)
        ordered, pools = [], list(by_ds.values())
        while any(pools):
            for lst in pools:
                if lst:
                    ordered.append(lst.pop(0))
        keys = ordered

    experiments: List[ExperimentData] = []
    for key in keys:
        dataset, task, net = key
        g = grouped.get_group(key)
        experiment_id = f"{dataset}_{task}_{net}"
        curves = phmlc_bridge.to_curve_dict(g)
        exp = _build_one(experiment_id, dataset, task, net, curves, results)
        if exp is None or len(exp.unit_ids) < min_units:
            continue
        experiments.append(exp)
        if n_experiments is not None and len(experiments) >= n_experiments:
            break
    return experiments


# --------------------------------------------------------------------------- #
# Synthetic data (offline)
# --------------------------------------------------------------------------- #
def make_synthetic_experiment(
    experiment_id: str = "synthetic_exp",
    n_units: int = 12,
    n_epochs: int = 40,
    seed: int = 0,
    diverge_fraction: float = 0.35,
) -> ExperimentData:
    """One offline experiment with a mix of good, mediocre and diverging runs."""
    rng = np.random.default_rng(seed)
    curves: Dict[str, np.ndarray] = {}
    epoch_time: Dict[str, float] = {}
    meta: Dict[str, dict] = {}
    rows = []
    units = []

    for i in range(n_units):
        uid = f"{experiment_id}_u{i}"
        units.append(uid)
        lr = float(10 ** rng.uniform(-4, -1))        # numeric hp
        wd = float(10 ** rng.uniform(-6, -2))        # numeric hp
        diverging = rng.random() < diverge_fraction
        start = float(rng.uniform(0.8, 1.0))
        if diverging:
            final = float(rng.uniform(0.6, 1.2))
            tau = rng.uniform(8, 15)
            base = final + (start - final) * np.exp(-np.arange(n_epochs) / tau)
            drift = np.linspace(0, rng.uniform(0.2, 0.6), n_epochs)  # goes back up
            val = base + drift
        else:
            final = float(rng.uniform(0.05, 0.45))   # good runs reach low loss
            tau = rng.uniform(5, 12)
            val = final + (start - final) * np.exp(-np.arange(n_epochs) / tau)
        val = val + rng.normal(0, 0.01, n_epochs)
        val = np.clip(val, 1e-3, None)
        train = np.clip(val - rng.uniform(0.02, 0.1), 1e-3, None)
        curve = np.stack([train, val], axis=1)
        curves[uid] = curve
        epoch_time[uid] = float(rng.uniform(0.5, 2.0))
        meta[uid] = {
            "dataset": "synthetic", "task": "final_loss", "net": "synthnet",
            "hyperparameters": {"model__lr": lr, "model__weight_decay": wd},
        }
        rows.append((uid, lr, wd))

    numeric_cols = ["model__lr", "model__weight_decay"]
    matrix_uids = np.array(units, dtype=object)
    matrix_values = np.array([[r[1], r[2]] for r in rows], dtype=float)
    param_ranges = {
        "model__lr": (matrix_values[:, 0].min(), matrix_values[:, 0].max()),
        "model__weight_decay": (matrix_values[:, 1].min(), matrix_values[:, 1].max()),
    }
    return ExperimentData(
        experiment_id=experiment_id,
        curves=curves,
        epoch_time=epoch_time,
        meta=meta,
        param_ranges=param_ranges,
        categorical_params=[],
        numeric_cols=numeric_cols,
        matrix_uids=matrix_uids,
        matrix_values=matrix_values,
    )


def make_synthetic_experiments(
    n_experiments: int = 1, n_units: int = 12, n_epochs: int = 40, seed: int = 0
) -> List[ExperimentData]:
    return [
        make_synthetic_experiment(
            experiment_id=f"synthetic_exp{j}", n_units=n_units,
            n_epochs=n_epochs, seed=seed + j,
        )
        for j in range(n_experiments)
    ]
