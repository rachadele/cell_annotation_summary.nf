process PLOT_CUTOFF {
    label 'process_single'
    conda '/home/rschwartz/anaconda3/envs/scanpyenv'

    input:
    path weighted_f1_results_aggregated
    path label_f1_results_aggregated

    output:
    path "**png"

    script:
    def ref_keys = params.ref_keys.join(' ')
    """
    python ${projectDir}/bin/plot_cutoff.py \\
        --weighted_f1_results ${weighted_f1_results_aggregated} \\
        --label_f1_results ${label_f1_results_aggregated} \\
        --mapping_file ${params.mapping_file} \\
        --color_mapping_file ${params.color_mapping_file} \\
        --ref_keys ${ref_keys}
    """
}
