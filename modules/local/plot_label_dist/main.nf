process PLOT_LABEL_DIST {
    label 'process_single'
    conda '/home/rschwartz/anaconda3/envs/scanpyenv'

    input:
    path label_f1_results_aggregated

    output:
    path "**png"

    script:
    """
    python ${projectDir}/bin/label_dists.py \\
        --label_f1_results ${label_f1_results_aggregated} \\
        --mapping_file ${params.mapping_file} \\
        --color_mapping_file ${params.color_mapping_file}
    """
}
