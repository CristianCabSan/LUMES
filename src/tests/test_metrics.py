import math

from metrics import ExperimentResult, MetricsCollector


def test_saved_time_b_never_exceeds_a():
    r = ExperimentResult(
        policy="llm", experiment_id="e", seed=0,
        total_train_time=100.0, saved_time_A=40.0, saved_time_B=37.5,
    )
    assert r.saved_time_B <= r.saved_time_A
    assert abs(r.saved_pct_A - 0.40) < 1e-9
    assert abs(r.saved_pct_B - 0.375) < 1e-9


def test_found_best_flag():
    assert ExperimentResult("o", "e", 0, filter_best_loss=0.5, real_best_loss=0.5).found_best
    assert not ExperimentResult("o", "e", 0, filter_best_loss=0.7, real_best_loss=0.5).found_best


def test_summary_rates_and_means():
    mc = MetricsCollector()
    mc.add(ExperimentResult("llm", "e1", 0, regret=0.0, real_best_loss=0.5,
                            filter_best_loss=0.5, total_train_time=100, saved_time_A=50,
                            saved_time_B=45, n_decisions=10, false_prunes=1, false_continues=2,
                            num_runs=5, num_prunings=2, mean_confidence=0.9))
    mc.add(ExperimentResult("llm", "e2", 0, regret=0.2, real_best_loss=0.4,
                            filter_best_loss=0.6, total_train_time=100, saved_time_A=30,
                            saved_time_B=28, n_decisions=10, false_prunes=3, false_continues=0,
                            num_runs=5, num_prunings=1, mean_confidence=0.8))
    s = mc.summary()
    row = s[s.policy == "llm"].iloc[0]
    assert abs(row["regret_mean"] - 0.1) < 1e-9
    assert abs(row["saved_pct_A_mean"] - 0.40) < 1e-9   # (0.5 + 0.3)/2
    assert abs(row["false_prune_rate"] - (4 / 20)) < 1e-9
    assert abs(row["false_continue_rate"] - (2 / 20)) < 1e-9
    assert abs(row["prune_rate"] - (3 / 10)) < 1e-9


def test_csv_roundtrip(tmp_path):
    mc = MetricsCollector()
    mc.add(ExperimentResult("random", "e1", 0, regret=0.1))
    path = mc.save_csv(str(tmp_path / "runs.csv"))
    import pandas as pd
    df = pd.read_csv(path)
    assert "regret" in df.columns and len(df) == 1
