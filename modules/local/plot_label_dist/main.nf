process PLOT_LABEL_DIST {
    label 'process_single'

    input:
    path label_results_aggregated

    output:
    path "**png"

    script:
    """
    python ${projectDir}/bin/label_dists.py \\
        --label_results ${label_results_aggregated} \\
        --mapping_file ${params.mapping_file} \\
        --color_mapping_file ${params.color_mapping_file}
    """
}
