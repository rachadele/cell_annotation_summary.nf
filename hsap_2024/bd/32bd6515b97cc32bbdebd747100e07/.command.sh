#!/bin/bash -ue
python /space/grp/rschwartz/rschwartz/evaluation_summary.nf/bin/label_dists.py --label_f1_results label_f1_results.tsv \
            --mapping_file /space/grp/rschwartz/rschwartz/evaluation_summary.nf/meta/census_map_human.tsv \
            --color_mapping_file /space/grp/rschwartz/rschwartz/evaluation_summary.nf/meta/color_mapping.tsv
