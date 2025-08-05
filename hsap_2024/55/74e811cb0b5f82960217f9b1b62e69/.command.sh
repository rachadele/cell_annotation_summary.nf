#!/bin/bash -ue
python /space/grp/rschwartz/rschwartz/evaluation_summary.nf/bin/plot_contrasts.py --emmeans_estimates subsample_ref_emmeans_estimates.tsv \
                --emmeans_summary subsample_ref_emmeans_summary.tsv \
                --key class \
                --weighted_f1_results weighted_f1_results.tsv
