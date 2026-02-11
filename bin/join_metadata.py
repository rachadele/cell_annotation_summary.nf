#!/usr/bin/env python3
"""Join study and reference metadata to aggregated results."""

import argparse
import pandas as pd


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample_results", default="sample_results.tsv")
    parser.add_argument("--label_results", default="label_results.tsv")
    parser.add_argument("--study_meta", default="study_metadata_mus_musculus.tsv")
    parser.add_argument("--ref_meta", default="reference_metadata_mus_musculus.tsv")
    parser.add_argument("--outdir", default=".")
    args, _ = parser.parse_known_args()

    study_meta = pd.read_csv(args.study_meta, sep="\t")
    study_meta = study_meta[["study", "suspension_type"]].rename(
        columns={"suspension_type": "query_suspension_type"}
    )

    ref_meta = pd.read_csv(args.ref_meta, sep="\t")
    ref_meta = ref_meta[["dataset_title", "suspension_type"]].rename(
        columns={"dataset_title": "reference", "suspension_type": "ref_suspension_type"}
    )

    # Join sample results
    sample_df = pd.read_csv(args.sample_results, sep="\t")
    sample_df = sample_df.merge(study_meta, on="study", how="left")
    sample_df = sample_df.merge(ref_meta, on="reference", how="left")
    sample_df.to_csv(f"{args.outdir}/sample_results.tsv", sep="\t", index=False)

    # Join label results
    label_df = pd.read_csv(args.label_results, sep="\t")
    label_df = label_df.merge(study_meta, on="study", how="left")
    label_df = label_df.merge(ref_meta, on="reference", how="left")
    label_df.to_csv(f"{args.outdir}/label_results.tsv", sep="\t", index=False)

    print(f"Sample results: {len(sample_df)} rows, "
          f"query_suspension_type null: {sample_df['query_suspension_type'].isna().sum()}, "
          f"ref_suspension_type null: {sample_df['ref_suspension_type'].isna().sum()}")
    print(f"Label results: {len(label_df)} rows, "
          f"query_suspension_type null: {label_df['query_suspension_type'].isna().sum()}, "
          f"ref_suspension_type null: {label_df['ref_suspension_type'].isna().sum()}")


if __name__ == "__main__":
    main()
