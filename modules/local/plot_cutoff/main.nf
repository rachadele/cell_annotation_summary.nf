process PLOT_CUTOFF {
    label 'process_single'

    input:
    path sample_results_aggregated
    path label_results_aggregated

    output:
    path "**png"

    script:
    def ref_keys = params.ref_keys.join(' ')
    """
    python ${projectDir}/bin/plot_cutoff.py \\
        --sample_results ${sample_results_aggregated} \\
        --label_results ${label_results_aggregated} \\
        --mapping_file ${params.mapping_file} \\
        --color_mapping_file ${params.color_mapping_file} \\
        --ref_keys ${ref_keys}
    """
}
