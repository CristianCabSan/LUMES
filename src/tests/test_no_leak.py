"""The central invariant: a scenario never depends on the future."""

import numpy as np

from scenario import ScenarioBuilder


def _curve(vals):
    vals = np.asarray(vals, dtype=float)
    train = np.clip(vals - 0.05, 1e-3, None)
    return np.stack([train, vals], axis=1)


def test_future_changes_do_not_affect_scenario():
    builder = ScenarioBuilder()
    base = [1.0, 0.8, 0.7, 0.65, 0.6, 0.55, 0.5, 0.48, 0.47, 0.46, 0.45, 0.44]
    epoch = 6
    bo = {"best_so_far_final_val": 0.4, "best_val_at_epoch": 0.5}

    sc1 = builder.build("u", "exp", _curve(base), epoch, bo, {})

    tampered = list(base)
    for i in range(epoch + 1, len(tampered)):
        tampered[i] = 99.0  # poison the future
    sc2 = builder.build("u", "exp", _curve(tampered), epoch, bo, {})

    assert sc1.partial_val == sc2.partial_val
    assert sc1.features == sc2.features
    assert sc1.cache_key() == sc2.cache_key()


def test_partial_arrays_have_exact_length():
    builder = ScenarioBuilder()
    curve = _curve(np.linspace(1.0, 0.2, 30))
    epoch = 9
    sc = builder.build("u", "exp", curve, epoch, {}, {})
    assert len(sc.partial_val) == epoch + 1
    assert len(sc.partial_train) == epoch + 1
    assert sc.features["n_epochs_seen"] == epoch + 1


def test_features_are_past_only_values():
    builder = ScenarioBuilder()
    curve = _curve([1.0, 0.9, 0.8, 0.7, 0.6])
    sc = builder.build("u", "exp", curve, 4, {"best_so_far_final_val": 0.5}, {})
    # current_val equals the last partial value, nothing from the (absent) future
    assert abs(sc.features["current_val"] - 0.6) < 1e-9
    assert sc.features["val_min_so_far"] == min(sc.partial_val)
