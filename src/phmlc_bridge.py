"""The single boundary with the ``phmlc`` framework.

Hard rule of the project: **nothing in ``phmlc`` is modified**, it is consumed as
a library.  This module is the *only* place that touches ``phm_framework``.  Every
other module in ``LUMES`` is unaware that ``phmlc`` exists.

Two kinds of functions live here:

* Data loaders (``load_experiment_curves`` / ``load_results``) — these import
  ``phm_framework`` *lazily*, because importing that package pulls in TensorFlow
  (via ``phm_framework.typing``).  Keeping the import inside the function means the
  rest of ``LUMES`` (and the whole offline/synthetic path and unit tests) runs
  without TensorFlow installed.
* Pure helpers (``to_curve_dict`` / ``build_past_features``) — NumPy/pandas only,
  faithful re-implementations of ``phmlc``'s per-epoch feature engineering, safe to
  import and unit-test anywhere.
"""

from __future__ import annotations

import os
import sys
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

# Column convention used everywhere downstream: col0 = train_loss, col1 = val_loss.
TRAIN_COL = 0
VAL_COL = 1

# Ground-truth slack used by phmlc to label a per-epoch decision (train.py:1147).
GROUND_TRUTH_SLACK = 1.05


def _ensure_phmlc_on_path() -> str:
    """Prepend ``$PHMLC_SRC`` (default ``../phmlc/src``) to ``sys.path``."""
    src = os.environ.get("PHMLC_SRC", os.path.join("..", "phmlc", "src"))
    src = os.path.abspath(src)
    if src not in sys.path:
        sys.path.insert(0, src)
    return src


def _ensure_curves_unzipped() -> None:
    """phmd downloads ``curves.zip`` but ``load()`` defaults to ``unzip=False``.

    If the extracted ``curves/`` folder is missing, trigger phmd's own unzip so the
    reader finds the data.  Idempotent and offline once the zip is present.
    """
    try:
        from phmd.datasets import get_storage_dir
        from phmd.download import download
        storage = get_storage_dir()
        if not os.path.isdir(os.path.join(storage, "curves")):
            download("CURVES", force=False, unzip=True)
    except Exception as exc:  # pragma: no cover - best effort
        print(f"[phmlc_bridge] could not auto-unzip CURVES ({exc}); "
              f"ensure the dataset is extracted under ~/.phmd/datasets/curves/.")


_LOADER = None   # resolved once: phmlc's load_curves callable, or "phmd" sentinel


def _resolve_loader():
    """Resolve the curves loader once: phmlc's ``load_curves`` or the phmd replica.

    Importing ``phm_framework`` runs its package ``__init__`` which eagerly imports
    the whole training stack (models/trainers, ``einops``, a bare ``import utils``,
    TensorFlow) — none of which ``load_curves`` needs.  When that import fails we
    fall back to :func:`_load_curves_phmd`, a faithful re-implementation that uses
    only ``phmd`` (we never modify phmlc; we just avoid an unusable import path).
    """
    global _LOADER
    if _LOADER is not None:
        return _LOADER
    _ensure_phmlc_on_path()
    try:
        from phm_framework.optimization.curves import load_curves
        _LOADER = load_curves
    except Exception as exc:  # ImportError / ModuleNotFoundError / TF DLL / ...
        print(f"[phmlc_bridge] phm_framework import failed ({type(exc).__name__}: "
              f"{exc}); using phmd-only load_curves replica.")
        _LOADER = "phmd"
    return _LOADER


def _load_curves(fold, num_folds, filters, test_dataset_names, random_state,
                 normalize_output=False):
    loader = _resolve_loader()
    if loader == "phmd":
        return _load_curves_phmd(fold, num_folds, normalize_output, filters,
                                 random_state, test_dataset_names or [])
    return loader(
        fold=fold, num_folds=num_folds, normalize_output=normalize_output,
        filters=filters, random_state=random_state,
        test_dataset_names=test_dataset_names or [],
    )


def _load_curves_phmd(fold, num_folds, normalize_output, filters, random_state,
                      test_dataset_names):
    """phmd-only replica of ``phm_framework.optimization.curves.load_curves``.

    Mirrors that function line-for-line (same CURVES dataset, same normalisation,
    same KFold-by-dataset split); the only change is hard-coding the activation /
    RNN-cell name lists so the results branch needs no TensorFlow.
    """
    import random as _random
    from sklearn.model_selection import KFold
    from phmd import datasets

    _ensure_curves_unzipped()
    ds = datasets.Dataset("CURVES")
    task = ds["final_loss"]
    task.folds = num_folds
    task.filters = filters
    task.normalize_output = normalize_output
    task.random_state = random_state
    X = task.load()[0]

    if "train_loss" in X.columns:
        X.loc[X.train_loss > 8, "train_loss"] = 8.0
        X["max_train_loss"] = X.groupby(["unit", "dataset", "task", "net"])["val_loss"].transform("max")
        X["train_loss"] = X["train_loss"] / X["max_train_loss"]
        X.loc[X.train_loss > 1, "train_loss"] = 1.0
        X["val_loss"] = X["val_loss"] / X["max_train_loss"]
        del X["max_train_loss"]
        X["num_epochs"] = X["num_epochs"] / 100
    else:
        activations = ["relu", "LeakyReLU", "tanh", "sigmoid"]   # phmf.typing.ACTIVATIONS names
        rnn_cells = ["GRUCell", "LSTMCell"]                       # phmf.typing.RNN_CELLS names

        def get_code(x, elements):
            x = str(x)
            for i, a in enumerate(elements):
                if a in x:
                    return i
            return float(x)

        def _map(col, fn):
            if col in X.columns:
                X[col] = X[col].map(fn)

        _map("model__activation", lambda x: x if x is np.nan else round(get_code(x, activations)))
        _map("model__dense_activation", lambda x: x if x is np.nan else round(get_code(x, activations)))
        _map("model__conv_activation", lambda x: x if x is np.nan else round(get_code(x, activations)))
        _map("model__cell_type", lambda x: x if x is np.nan else round(get_code(x, rnn_cells)))
        _map("model__kernel_size",
             lambda x: (x if x is np.nan else float(x)) if "(" not in str(x) else eval(x)[0] + eval(x)[1] / 100)
        _map("model__batch_normalization", lambda x: x if x is np.nan else round(float(eval(str(x)))))
        _map("model__bidirectional", lambda x: x if x is np.nan else round(float(eval(str(x)))))

    if num_folds == 0:
        return X

    dataset_names = X.dataset.unique()
    if len(test_dataset_names) == 0:
        # phmlc shuffles with an *unseeded* random -> non-reproducible test split.
        # We seed it with random_state so runs are comparable (same algorithm, fixed
        # order); pass explicit test_dataset_names to bypass entirely.
        _random.seed(random_state)
        _random.shuffle(dataset_names)
        test_end_index = int(len(dataset_names) * 0.2)
        test_dataset_names = dataset_names[:test_end_index]
        dataset_names = dataset_names[test_end_index:]
    else:
        dataset_names = np.array([d for d in dataset_names if d not in test_dataset_names])

    skf = KFold(n_splits=num_folds, random_state=random_state, shuffle=True)
    folds = list(skf.split(dataset_names))
    train_idx, val_idx = folds[fold]

    return {
        "train": X[X.dataset.isin(dataset_names[train_idx])],
        "val": X[X.dataset.isin(dataset_names[val_idx])],
        "test": X[X.dataset.isin(test_dataset_names)],
    }


# --------------------------------------------------------------------------- #
# Processed-dataset cache (parse once, reuse the seeded/reproducible result)
# --------------------------------------------------------------------------- #
# phmd downloads curves.zip only once, but it RE-PARSES all ~58k curves on every
# load() call (the slow "Reading data" pass). The split is seeded/reproducible, so
# we pickle the processed result keyed by the load parameters and reuse it.
# Bump _CACHE_VERSION if the processing logic here changes; set LLM_ES_NO_CACHE=1
# to bypass, or delete the cache dir to invalidate.
_CACHE_VERSION = 1


def _cache_dir() -> str:
    d = os.environ.get(
        "LLM_ES_CACHE", os.path.join(os.path.expanduser("~"), ".cache", "LUMES")
    )
    os.makedirs(d, exist_ok=True)
    return d


def _cached_load(name: str, params: dict, compute):
    import hashlib
    import pickle

    if os.environ.get("LLM_ES_NO_CACHE"):
        return compute()
    key = hashlib.sha1(
        repr((_CACHE_VERSION, name, sorted(params.items()))).encode()
    ).hexdigest()[:16]
    path = os.path.join(_cache_dir(), f"{name}_{key}.pkl")
    if os.path.exists(path):
        try:
            with open(path, "rb") as f:
                print(f"[phmlc_bridge] using cached {name} -> {path}")
                return pickle.load(f)
        except Exception as exc:  # corrupt cache -> recompute
            print(f"[phmlc_bridge] cache read failed ({exc}); recomputing.")
    obj = compute()
    try:
        with open(path, "wb") as f:
            pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)
        print(f"[phmlc_bridge] cached {name} -> {path}")
    except Exception as exc:  # pragma: no cover - best effort
        print(f"[phmlc_bridge] cache write failed ({exc}).")
    return obj


# --------------------------------------------------------------------------- #
# Data loaders (lazy phm_framework import, cached)
# --------------------------------------------------------------------------- #
def load_experiment_curves(
    fold: int = 0,
    num_folds: int = 5,
    filters: Optional[dict] = None,
    test_dataset_names: Optional[List[str]] = None,
    random_state: int = 666,
) -> Dict[str, pd.DataFrame]:
    """Load the CURVES meta-dataset split into train/val/test DataFrames.

    Thin wrapper around ``phm_framework.optimization.curves.load_curves`` (or the
    phmd replica), **cached** by the load parameters so the ~58k-curve parse runs
    only once per (filters, split, seed).  Returns a dict with keys ``'train'``,
    ``'val'``, ``'test'``; each DataFrame has one row per epoch with at least
    ``unit, dataset, task, net, train_loss, val_loss, num_epochs``.
    """
    if filters is None:
        filters = {"data": "curves"}
    return _cached_load(
        "curves_split",
        dict(fold=fold, num_folds=num_folds, filters=tuple(sorted(filters.items())),
             test=tuple(test_dataset_names or []), rs=random_state),
        lambda: _load_curves(
            fold=fold, num_folds=num_folds, filters=filters,
            test_dataset_names=test_dataset_names or [], random_state=random_state,
        ),
    )


def load_results(random_state: int = 666) -> pd.DataFrame:
    """Load the per-run results view (one row per ``unit``), **cached**.

    Uses ``load_curves(..., num_folds=0, filters={"data": "results"})`` which
    returns the un-split frame with hyper-parameters (``model__*``) and
    ``train__time`` (total wall-clock of the run).  ``epoch_time`` downstream is
    ``train__time / num_epochs``.
    """
    return _cached_load(
        "results",
        dict(rs=random_state),
        lambda: _load_curves(
            fold=None, num_folds=0, filters={"data": "results"},
            test_dataset_names=[], random_state=random_state,
        ),
    )


# --------------------------------------------------------------------------- #
# Pure helpers (NumPy/pandas only)
# --------------------------------------------------------------------------- #
def to_curve_dict(
    df: pd.DataFrame,
    unit_col: str = "unit",
    train_col: str = "train_loss",
    val_col: str = "val_loss",
) -> Dict[str, np.ndarray]:
    """Turn a per-epoch curves DataFrame into ``{unit: ndarray(n_epochs, 2)}``.

    Mirrors how phmlc materialises curves (``groupby(unit)`` preserving row order,
    stacking the feature columns); col0=train_loss, col1=val_loss.  Row order
    within a unit is the curve's epoch order, exactly as stored upstream.
    """
    out: Dict[str, np.ndarray] = {}
    for unit, g in df.groupby(unit_col, sort=False):
        arr = np.stack(
            [g[train_col].to_numpy(dtype=float), g[val_col].to_numpy(dtype=float)],
            axis=1,
        )
        out[str(unit)] = arr
    return out


def ground_truth_continue(
    current_val: float, best_val_at_epoch: float, slack: float = GROUND_TRUTH_SLACK
) -> bool:
    """phmlc per-epoch ground truth: continue iff current <= best_at_epoch * slack."""
    if best_val_at_epoch is None or not np.isfinite(best_val_at_epoch):
        return True
    return bool(current_val <= best_val_at_epoch * slack)


def build_past_features(
    val_partial,
    train_partial=None,
    best_performance: Optional[float] = None,
    best_val_at_epoch: Optional[float] = None,
) -> dict:
    """Replicate phmlc's per-epoch decision features, **truncated to the present**.

    Faithful to ``prepare_decision_data`` / ``extended_decision_data``
    (train.py:1113, :1186) but evaluated only on ``val_partial`` (= ``val[:e+1]``),
    so it is *strictly past/present* — no value depends on epochs after the current
    one.  ``best_performance`` (BO incumbent final loss) and ``best_val_at_epoch``
    (incumbent's val at this epoch) come from the simulator's BO state; when absent
    the improvement features are ``None`` (e.g. during warmup).

    Notes
    -----
    * ``val_velocity`` = first difference (phmlc ``diff``), ``val_ema`` = ``ewm`` with
      span 3 — computed exactly as phmlc, then we take the value *at the current
      epoch*.
    * ``expected_improvement`` here is a transparent **past-only** heuristic
      (one-step EMA-velocity projection vs incumbent), *not* phmlc's surrogate-model
      EI — we have no surrogate; the LLM is the predictor.  It is provided only as an
      optional prompt feature.
    """
    val = pd.Series(np.asarray(val_partial, dtype=float))
    n = len(val)
    if n == 0:
        raise ValueError("val_partial must contain at least one value")

    current_val = float(val.iloc[-1])
    velocity_series = val.diff().fillna(0.0)
    val_velocity = float(velocity_series.iloc[-1])
    ema_series = val.ewm(span=3).mean()
    val_ema = float(ema_series.iloc[-1])
    ema_velocity = float(ema_series.diff().fillna(0.0).iloc[-1])

    feats = {
        "epoch": n - 1,
        "n_epochs_seen": n,
        "current_val": current_val,
        "current_train": (float(train_partial[-1]) if train_partial is not None else None),
        "val_velocity": val_velocity,
        "val_ema": val_ema,
        "val_min_so_far": float(val.min()),
        "epochs_since_best": int(n - 1 - int(val.values.argmin())),
        "best_performance": best_performance,
        "best_val_at_epoch": best_val_at_epoch,
        "val_improvement": None,
        "expected_improvement": None,
    }

    if best_performance is not None and np.isfinite(best_performance) and best_performance != 0:
        feats["val_improvement"] = float(
            np.clip((current_val - best_performance) / best_performance, -1.0, 1.0)
        )
        projected = current_val + ema_velocity
        feats["expected_improvement"] = float(
            np.clip((projected - best_performance) / best_performance, -1.0, 1.0)
        )

    return feats
