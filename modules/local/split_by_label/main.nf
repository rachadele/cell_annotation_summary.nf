process SPLIT_BY_LABEL {
    label 'process_single'

    input:
    path label_f1_results_aggregated

    output:
    path "**tsv", emit: label_f1_results_split
    path "**png"

    script:
    """
    python ${projectDir}/bin/split_by_label.py --label_f1_results ${label_f1_results_aggregated}
    """
}
