#!/bin/bash
# Run celltype trends plots for both human and mouse data

echo "=== Running human celltype trends ==="
python bin/post_hoc_plotting/plot_celltype_granularity.py \
    --label_f1_results "/space/grp/rschwartz/rschwartz/evaluation_summary.nf/2024-07-01/homo_sapiens/100/dataset_id/SCT/gap_false/aggregated_results/files/label_f1_results.tsv" \
    --key class \
    --cutoff 0 \
    --mapping_file "/space/grp/rschwartz/rschwartz/evaluation_summary.nf/assets/census_map_human.tsv" \
    --lineage_key global \
    --organism homo_sapiens \
    --outdir figures \
    --output_prefix celltype_trends

echo "=== Running mouse celltype trends ==="
python bin/post_hoc_plotting/plot_celltype_granularity.py \
    --label_f1_results "/space/grp/rschwartz/rschwartz/evaluation_summary.nf/2024-07-01/mmus_new_tabulamuris/100/dataset_id/SCT/gap_false/aggregated_results/files/label_f1_results.tsv" \
    --key class \
    --cutoff 0 \
    --mapping_file "/space/grp/rschwartz/rschwartz/evaluation_summary.nf/assets/census_map_mouse_author.tsv" \
    --lineage_key global \
    --organism mus_musculus \
    --outdir figures \
    --output_prefix celltype_trends

echo "=== Done ==="
