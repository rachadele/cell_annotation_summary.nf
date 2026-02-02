process PLOT_CELLTYPE_GRANULARITY {
    label 'process_single'

    input:
    path label_f1_results_aggregated

    output:
    path "**.png", emit: figures

    script:

    """
    python ${projectDir}/bin/plot_celltype_granularity.py \\
        --label_f1_results ${label_f1_results_aggregated} \\
        --cutoff ${params.cutoff} \\
        --mapping_file ${params.mapping_file} \\
        --organism ${params.organism} \\
        --output_prefix celltype_granularity \\
        --outdir .
    """
}
