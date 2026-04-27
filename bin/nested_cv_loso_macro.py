#!/usr/bin/env python3
"""
Nested LOSO cross-validation for unbiased config selection on per-sample
macro F1.

For each `key` (taxonomy level) and each held-out study s_k:
    1. Inner CV on D_-k: pick c*_k = argmax over configs of
       [mean across train studies of (mean macro_f1 in that study)].
    2. Outer score: mean macro_f1 of c*_k on s_k.

Mean ± SD across outer folds is the unbiased estimate of the
selection-and-evaluation procedure's generalisation. See
docs/PLAN_nested_cv_macro.md for the full design.
"""
import argparse
import os
import numpy as np
import pandas as pd

CONFIG_COLS = ["method", "reference", "cutoff", "subsample_ref"]
SORT_COLS = ["study"] + CONFIG_COLS


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--sample_results", required=True,
                   help="Path to sample_results.tsv.gz")
    p.add_argument("--outdir", required=True,
                   help="Output directory; per-key subdirs will be created")
    p.add_argument("--keys", nargs="*", default=None,
                   help="Subset of keys to process (default: all in data)")
    p.add_argument("--min_study_n", type=int, default=0,
                   help="Drop studies with fewer than this many samples "
                        "in this key (default: 0)")
    return p.parse_args()


def run_key(df_k: pd.DataFrame, key: str, outdir: str, min_study_n: int):
    """Run nested LOSO for a single taxonomy level. Writes per-key outputs."""
    key_dir = os.path.join(outdir, key)
    os.makedirs(key_dir, exist_ok=True)

    # Per (study, config) score: mean macro_f1 over samples in the study.
    per_study = (
        df_k.groupby(["study"] + CONFIG_COLS, observed=True)["macro_f1"]
            .mean().rename("score").reset_index()
            .sort_values(SORT_COLS).reset_index(drop=True)
    )

    # Apply min_study_n filter on raw sample count per study (in this key).
    if min_study_n > 0:
        study_n = df_k.groupby("study", observed=True).size()
        keep = study_n[study_n >= min_study_n].index
        per_study = per_study[per_study.study.isin(keep)]
        print(f"  [{key}] kept {len(keep)} studies "
              f"with >= {min_study_n} samples")

    studies = sorted(per_study.study.unique())
    if len(studies) < 2:
        print(f"  [{key}] only {len(studies)} studies — skipping (need >=2)")
        return

    # Sample-count bookkeeping for the outer fold (per-study sample n in this key).
    n_per_study = df_k.groupby("study", observed=True).size().rename("n").to_dict()

    rows_outer, rows_inner = [], []
    for s_k in studies:
        train = per_study[per_study.study != s_k]
        test = per_study[per_study.study == s_k]

        # Inner: mean across train studies (skipna by default).
        inner = (train.groupby(CONFIG_COLS, observed=True)["score"]
                       .mean().rename("score_inner").reset_index()
                       .sort_values(CONFIG_COLS).reset_index(drop=True))

        if inner["score_inner"].isna().all():
            print(f"  [{key}] s_k={s_k}: all-NaN inner scores; skipping fold")
            continue

        best = inner.loc[[inner["score_inner"].idxmax()]]
        c_star = best[CONFIG_COLS]

        outer_match = test.merge(c_star, on=CONFIG_COLS)
        outer_score = float(outer_match["score"].iloc[0]) if len(outer_match) else float("nan")

        rows_outer.append({
            "held_out_study": s_k,
            **{col: best[col].iloc[0] for col in CONFIG_COLS},
            "outer_score": outer_score,
            "n_outer_test_samples": n_per_study.get(s_k, 0),
        })
        rows_inner.append(inner.assign(held_out_study=s_k))

    outer_df = pd.DataFrame(rows_outer)
    inner_df = pd.concat(rows_inner, ignore_index=True)

    outer_df.to_csv(os.path.join(key_dir, "outer_fold_results.tsv"),
                    sep="\t", index=False)
    inner_df.to_csv(os.path.join(key_dir, "inner_selection_log.tsv"),
                    sep="\t", index=False)

    # Full-data argmax (computed on all studies pooled).
    full = (per_study.groupby(CONFIG_COLS, observed=True)["score"]
                     .mean().rename("full_data_mean").reset_index()
                     .sort_values(CONFIG_COLS).reset_index(drop=True))
    full_best = full.loc[[full["full_data_mean"].idxmax()]].iloc[0]

    # Modal pick across outer folds: most common config tuple.
    pick_counts = (outer_df.groupby(CONFIG_COLS, observed=True)
                            .size().rename("n").reset_index()
                            .sort_values("n", ascending=False))
    modal_pick = pick_counts.iloc[0]
    modal_agreement = float(modal_pick["n"]) / len(outer_df)

    summary = {
        "key": key,
        "n_folds": len(outer_df),
        "n_folds_with_outer_score": int(outer_df["outer_score"].notna().sum()),
        "outer_mean": float(outer_df["outer_score"].mean(skipna=True)),
        "outer_sd": float(outer_df["outer_score"].std(skipna=True)),
        "full_data_mean": float(full_best["full_data_mean"]),
        "selection_bias": float(full_best["full_data_mean"]
                                - outer_df["outer_score"].mean(skipna=True)),
        "modal_agreement": modal_agreement,
    }
    for col in CONFIG_COLS:
        summary[f"full_data_{col}"] = full_best[col]
        summary[f"modal_{col}"] = modal_pick[col]

    pd.DataFrame([summary]).to_csv(
        os.path.join(key_dir, "unbiased_summary.tsv"),
        sep="\t", index=False,
    )
    print(f"  [{key}] outer mean = {summary['outer_mean']:.4f} "
          f"± {summary['outer_sd']:.4f} "
          f"(full-data mean = {summary['full_data_mean']:.4f}, "
          f"selection bias = {summary['selection_bias']:+.4f}, "
          f"modal agreement = {modal_agreement:.0%})")


def main():
    args = parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    print(f"Reading {args.sample_results} ...")
    df = pd.read_csv(args.sample_results, sep="\t", low_memory=False)
    print(f"  {len(df):,} rows, {df['study'].nunique()} studies, "
          f"{df['key'].nunique()} keys")

    keys = args.keys if args.keys else sorted(df["key"].unique())
    for key in keys:
        df_k = df[df["key"] == key].copy()
        if df_k.empty:
            print(f"  [{key}] no rows — skipping")
            continue
        print(f"Running {key} ({len(df_k):,} rows) ...")
        run_key(df_k, key, args.outdir, args.min_study_n)

    # Combined summary across keys for convenience.
    summaries = []
    for key in keys:
        path = os.path.join(args.outdir, key, "unbiased_summary.tsv")
        if os.path.exists(path):
            summaries.append(pd.read_csv(path, sep="\t"))
    if summaries:
        pd.concat(summaries, ignore_index=True).to_csv(
            os.path.join(args.outdir, "unbiased_summary_all_keys.tsv"),
            sep="\t", index=False,
        )
        print(f"\nWrote combined summary to "
              f"{args.outdir}/unbiased_summary_all_keys.tsv")


if __name__ == "__main__":
    main()
