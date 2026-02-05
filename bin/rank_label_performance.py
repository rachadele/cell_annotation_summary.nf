#!/usr/bin/env python3
"""
Rank-based analysis of cell type classification performance.

For each (cell type, taxonomy level), ranks (reference, method, subsample_ref)
combinations by mean F1 score within each study, then aggregates ranks across
studies to identify the best-performing parameter combination.

See docs/label_ranking_method.md for a full description of the approach.

Usage:
    python rank_label_performance.py \
        --label_results label_results.tsv \
        --outdir rankings
"""

import argparse
import os
import numpy as np
import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser(
        description="Rank reference/method/subsample_ref combinations per cell type"
    )
    parser.add_argument("--label_results", type=str, required=True,
                        help="Path to label_results.tsv")
    parser.add_argument("--cutoff", type=float, default=0,
                        help="Confidence cutoff to filter to (default: 0)")
    parser.add_argument("--outdir", type=str, default="rankings",
                        help="Output directory")
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    # ------------------------------------------------------------------
    # Load and filter
    # ------------------------------------------------------------------
    print(f"Loading {args.label_results}...")
    df = pd.read_csv(args.label_results, sep="\t", low_memory=False)
    print(f"  {len(df)} rows total")

    df = df[df["cutoff"] == args.cutoff].copy()
    print(f"  {len(df)} rows after cutoff == {args.cutoff} filter")

    if df.empty:
        print("No data after filtering. Exiting.")
        return

    # ------------------------------------------------------------------
    # Step 1: Within-study mean F1
    #   Average F1 across query samples within each
    #   (study, label, key, reference, method, subsample_ref) group.
    #   This marginalizes over sample-level covariates (sex, treatment, …).
    # ------------------------------------------------------------------
    combo_cols = ["reference", "method", "subsample_ref"]
    group_cols = ["study", "label", "key"] + combo_cols

    study_means = (
        df.groupby(group_cols)
        .agg(
            mean_f1=("f1_score", "mean"),
            n_samples=("query", "nunique"),
            mean_support=("support", "mean"),
        )
        .reset_index()
    )
    print(f"  {len(study_means)} (study, label, key, combo) groups")

    # ------------------------------------------------------------------
    # Step 2: Rank within (study, label, key)
    #   Rank 1 = highest mean F1 (best). Ties get average rank.
    # ------------------------------------------------------------------
    rank_group = ["study", "label", "key"]

    study_means["rank"] = (
        study_means.groupby(rank_group)["mean_f1"]
        .rank(method="average", ascending=False)
    )
    study_means["n_combos"] = (
        study_means.groupby(rank_group)["mean_f1"]
        .transform("count")
    )
    # Normalized rank in [0, 1]: 0 = best, 1 = worst
    denom = (study_means["n_combos"] - 1).clip(lower=1)
    study_means["norm_rank"] = (study_means["rank"] - 1) / denom

    # Flag whether this combo was the best for this (study, label, key)
    study_means["is_best"] = (
        study_means["rank"] == study_means.groupby(rank_group)["rank"].transform("min")
    )

    # ------------------------------------------------------------------
    # Step 3: Aggregate across studies
    # ------------------------------------------------------------------
    agg_cols = ["label", "key"] + combo_cols

    rankings = (
        study_means.groupby(agg_cols)
        .agg(
            mean_rank=("rank", "mean"),
            median_rank=("rank", "median"),
            mean_norm_rank=("norm_rank", "mean"),
            n_studies=("study", "nunique"),
            n_wins=("is_best", "sum"),
            mean_f1_across_studies=("mean_f1", "mean"),
            std_f1_across_studies=("mean_f1", "std"),
            mean_support=("mean_support", "mean"),
            total_samples=("n_samples", "sum"),
        )
        .reset_index()
    )

    rankings["win_fraction"] = rankings["n_wins"] / rankings["n_studies"]
    rankings = rankings.sort_values(["key", "label", "mean_rank"])

    # ------------------------------------------------------------------
    # Step 4: Best combo per (label, key)
    # ------------------------------------------------------------------
    best = (
        rankings
        .sort_values("mean_rank")
        .groupby(["label", "key"])
        .first()
        .reset_index()
    )

    # ------------------------------------------------------------------
    # Save outputs
    # ------------------------------------------------------------------
    study_means.to_csv(
        os.path.join(args.outdir, "rankings_per_study.tsv"),
        sep="\t", index=False,
    )
    rankings.to_csv(
        os.path.join(args.outdir, "rankings_detailed.tsv"),
        sep="\t", index=False,
    )
    best.to_csv(
        os.path.join(args.outdir, "rankings_best.tsv"),
        sep="\t", index=False,
    )

    print(f"\nSaved to {args.outdir}/:")
    print(f"  rankings_per_study.tsv  — {len(study_means)} rows "
          f"(per-study ranks for inspection)")
    print(f"  rankings_detailed.tsv   — {len(rankings)} rows "
          f"(all combos aggregated across studies)")
    print(f"  rankings_best.tsv       — {len(best)} rows "
          f"(best combo per cell type × key level)")

    # Summary
    for key in sorted(best["key"].unique()):
        key_best = best[best["key"] == key]
        multi = key_best[key_best["n_studies"] > 1]
        print(f"\n  {key}: {len(key_best)} cell types, "
              f"{len(multi)} with multi-study evidence")


if __name__ == "__main__":
    main()
