"""Real-data smoke test for the phmlc bridge.

Imports phm_framework (pulls TensorFlow), downloads/loads the CURVES meta-dataset
via phmd, builds a couple of experiments and runs the offline-capable baselines on
them (no LLM/network needed).  Run this once in the execution environment to verify
the real data path end-to-end:

    python experiments/smoke_bridge.py --n-experiments 2 --seeds 0
"""

from __future__ import annotations

import argparse

from _cli import _SRC  # noqa: F401

import phmlc_bridge
from config import SimConfig
from dataset import build_experiments_from_phmlc
from policies import LastSeenPolicy, OraclePolicy
from simulator import BOSimulator


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-experiments", type=int, default=2)
    ap.add_argument("--split", default="test")
    ap.add_argument("--seeds", default="0")
    ap.add_argument("--min-units", type=int, default=3)
    args = ap.parse_args()
    seeds = [int(x) for x in args.seeds.split(",")]

    print("[smoke] loading CURVES (this imports TensorFlow and may download data)...")
    sets = phmlc_bridge.load_experiment_curves(fold=0, num_folds=5)
    for k, v in sets.items():
        print(f"  split {k:5s}: rows={len(v):>8d} units={v['unit'].nunique():>5d} "
              f"datasets={v['dataset'].nunique()}")

    results = phmlc_bridge.load_results()
    print(f"[smoke] results view: rows={len(results)} cols={results.shape[1]}")

    curves_df = sets[args.split]
    experiments = build_experiments_from_phmlc(
        curves_df, results, min_units=args.min_units, n_experiments=args.n_experiments
    )
    print(f"[smoke] built {len(experiments)} experiments from split '{args.split}'")
    for exp in experiments:
        print(f"  {exp.experiment_id}: units={len(exp.unit_ids)} "
              f"numeric_hp={len(exp.numeric_cols)} "
              f"sample_curve_shape={next(iter(exp.curves.values())).shape}")

    if not experiments:
        print("[smoke] no experiments built (check min-units); aborting policy run.")
        return

    for name, policy in [("last-seen", LastSeenPolicy()), ("oracle", OraclePolicy())]:
        sim = BOSimulator(policy=policy, sim=SimConfig())
        results_list = sim.run_all_experiments(experiments, seeds)
        import numpy as np
        regret = np.mean([r.regret for r in results_list])
        saved = np.mean([r.saved_pct_A for r in results_list])
        print(f"[smoke] {name:10s} -> mean_regret={regret:.4f} mean_saved_A={saved:.3f} "
              f"runs={sum(r.num_runs for r in results_list)} "
              f"prunes={sum(r.num_prunings for r in results_list)}")
    print("[smoke] OK")


if __name__ == "__main__":
    main()
