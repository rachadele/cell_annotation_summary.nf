process SPLIT_BY_LABEL {
    label 'process_single'

    input:
    path label_results_aggregated

    output:
    path "**tsv", emit: label_results_split
    path "**png"

    script:
    """
    python ${projectDir}/bin/split_by_label.py --label_results ${label_results_aggregated}
    """
}
