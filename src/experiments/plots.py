"""Plotting (Fase 5): regret, savings A/B, FC/FP, confidence.

Reads a per-run CSV produced by exp2 (``exp2_runs.csv``) or the determinism CSV
from exp3 (``exp3_determinism.csv``) and writes PNG figures.

    python experiments/plots.py --input results/exp2_runs.csv --outdir results/figs
    python experiments/plots.py --input results/exp3_determinism.csv --outdir results/figs
"""

from __future__ import annotations

import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402


def _save(fig, outdir, name):
    os.makedirs(outdir, exist_ok=True)
    path = os.path.join(outdir, name)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    print(f"  wrote {path}")


def plot_runs(df: pd.DataFrame, outdir: str):
    policies = sorted(df["policy"].unique())

    # regret: boxplot + jittered points (regret is often ~0 for every run, which
    # makes a bare boxplot collapse to a flat line; the points + mean markers keep
    # it readable and reveal the rare non-zero values).
    import numpy as np
    rng = np.random.default_rng(0)
    fig, ax = plt.subplots(figsize=(8, 4))
    data = [df[df.policy == p]["regret"].to_numpy() for p in policies]
    ax.boxplot(data, tick_labels=policies)
    for i, vals in enumerate(data, start=1):
        x = rng.normal(i, 0.05, size=len(vals))
        ax.scatter(x, vals, s=18, alpha=0.5, color="tab:blue", zorder=3)
        ax.scatter([i], [np.mean(vals)], marker="D", s=40, color="red", zorder=4,
                   label="mean" if i == 1 else None)
    allv = np.concatenate(data) if data else np.array([0.0])
    pad = max(1e-3, 0.1 * (allv.max() - allv.min()))
    ax.set_ylim(allv.min() - pad, allv.max() + pad)
    ax.set_ylabel("regret (filter_best - real_best)")
    ax.set_title("Regret by policy (points = individual runs; ~0 means it found the best)")
    ax.tick_params(axis="x", rotation=30)
    ax.legend(loc="best", fontsize=8)
    _save(fig, outdir, "regret_box.png")

    # savings A vs B
    fig, ax = plt.subplots(figsize=(8, 4))
    g = df.groupby("policy")[["saved_pct_A", "saved_pct_B"]].mean().reindex(policies)
    x = range(len(policies))
    ax.bar([i - 0.2 for i in x], g["saved_pct_A"], width=0.4, label="Variant A")
    ax.bar([i + 0.2 for i in x], g["saved_pct_B"], width=0.4, label="Variant B")
    ax.set_xticks(list(x)); ax.set_xticklabels(policies, rotation=30)
    ax.set_ylabel("mean saved fraction of train time"); ax.legend()
    ax.set_title("Time savings (A: no latency, B: with latency)")
    _save(fig, outdir, "savings_AB.png")

    # FC / FP rates
    fig, ax = plt.subplots(figsize=(8, 4))
    agg = df.groupby("policy").apply(
        lambda d: pd.Series({
            "false_prune_rate": d["false_prunes"].sum() / max(1, d["n_decisions"].sum()),
            "false_continue_rate": d["false_continues"].sum() / max(1, d["n_decisions"].sum()),
        })
    ).reindex(policies)
    ax.bar([i - 0.2 for i in range(len(policies))], agg["false_prune_rate"], width=0.4,
           label="false prune")
    ax.bar([i + 0.2 for i in range(len(policies))], agg["false_continue_rate"], width=0.4,
           label="false continue")
    ax.set_xticks(range(len(policies))); ax.set_xticklabels(policies, rotation=30)
    ax.set_ylabel("rate (per decision)"); ax.legend()
    ax.set_title("Per-epoch error rates vs ground truth")
    _save(fig, outdir, "fc_fp_rates.png")


def plot_determinism(df: pd.DataFrame, outdir: str):
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(df["confidence"], bins=10, range=(0, 1), edgecolor="black")
    ax.set_xlabel("confidence (agreement fraction)")
    ax.set_ylabel("# scenarios")
    ax.set_title(f"Determinism: mean conf {df['confidence'].mean():.2f}, "
                 f"{df['deterministic'].mean():.0%} deterministic")
    _save(fig, outdir, "confidence_hist.png")


def main():
    ap = argparse.ArgumentParser(description="plots for exp2/exp3 CSVs")
    ap.add_argument("--input", required=True)
    ap.add_argument("--outdir", default="results/figs")
    args = ap.parse_args()

    df = pd.read_csv(args.input)
    print(f"[plots] {args.input} ({len(df)} rows)")
    if "confidence" in df.columns and "deterministic" in df.columns:
        plot_determinism(df, args.outdir)
    else:
        plot_runs(df, args.outdir)


if __name__ == "__main__":
    main()
