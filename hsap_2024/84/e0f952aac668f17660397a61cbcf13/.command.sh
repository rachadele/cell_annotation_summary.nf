#!/bin/bash -ue
python /space/grp/rschwartz/rschwartz/evaluation_summary.nf/bin/plot_contrasts.py --emmeans_estimates disease_state_emmeans_estimates.tsv \
                --emmeans_summary disease_state_emmeans_summary.tsv \
                --key subclass \
                --weighted_f1_results weighted_f1_results.tsv
