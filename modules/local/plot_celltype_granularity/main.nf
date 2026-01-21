process PLOT_CELLTYPE_GRANULARITY {
    label 'process_single'

    input:
    path label_f1_results_aggregated

    output:
    path "figures/*.png", emit: figures

    script:
    def mapping_file = params.mapping_file ?: "${projectDir}/assets/census_map_${params.organism == 'homo_sapiens' ? 'human' : 'mouse_author'}.tsv"
    """
    python ${projectDir}/bin/post_hoc_plotting/plot_celltype_granularity.py \\
        --label_f1_results ${label_f1_results_aggregated} \\
        --cutoff ${params.cutoff} \\
        --mapping_file ${mapping_file} \\
        --organism ${params.organism} \\
        --outdir figures \\
        --output_prefix celltype_granularity \\
        --methods ${params.method}
    """
}
