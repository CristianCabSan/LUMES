"""build_past_features replicates phmlc's velocity/EMA and stays past-only."""

import numpy as np
import pandas as pd

import phmlc_bridge


def test_velocity_and_ema_match_pandas():
    vals = [1.0, 0.8, 0.75, 0.7, 0.69]
    f = phmlc_bridge.build_past_features(vals)
    s = pd.Series(vals)
    assert abs(f["val_velocity"] - s.diff().fillna(0).iloc[-1]) < 1e-12
    assert abs(f["val_ema"] - s.ewm(span=3).mean().iloc[-1]) < 1e-12
    assert abs(f["current_val"] - 0.69) < 1e-12
    assert f["val_min_so_far"] == 0.69


def test_improvement_features_require_incumbent():
    vals = [1.0, 0.8, 0.6]
    assert phmlc_bridge.build_past_features(vals)["val_improvement"] is None
    f = phmlc_bridge.build_past_features(vals, best_performance=0.5)
    assert f["val_improvement"] is not None
    # (0.6 - 0.5) / 0.5 = 0.2, clipped to [-1, 1]
    assert abs(f["val_improvement"] - 0.2) < 1e-9


def test_to_curve_dict_orders_and_shapes():
    df = pd.DataFrame({
        "unit": ["a", "a", "a", "b", "b"],
        "train_loss": [0.9, 0.8, 0.7, 0.5, 0.4],
        "val_loss": [1.0, 0.85, 0.75, 0.6, 0.55],
    })
    d = phmlc_bridge.to_curve_dict(df)
    assert set(d) == {"a", "b"}
    assert d["a"].shape == (3, 2)
    assert np.allclose(d["a"][:, phmlc_bridge.VAL_COL], [1.0, 0.85, 0.75])
    assert np.allclose(d["b"][:, phmlc_bridge.TRAIN_COL], [0.5, 0.4])


def test_ground_truth_slack():
    # current 1.05x of best-at-epoch is still "continue"; above is not
    assert phmlc_bridge.ground_truth_continue(0.525, 0.5) is True
    assert phmlc_bridge.ground_truth_continue(0.60, 0.5) is False
    assert phmlc_bridge.ground_truth_continue(0.60, None) is True
