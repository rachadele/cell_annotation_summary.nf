process PLOT_PUB_FIGURES {

    label 'process_single'

    input:
    val weighted_f1_results
    val cutoff_effects
    val reference_emmeans
    val method_emmeans_files
    val factor_emmeans_files

    output:
    path "**.png", emit: figures

    script:
    // Determine organism from weighted_f1 file
    // pass lists of files as space-separated strings
    def method_emmeans_str = method_emmeans_files.join(' ')
    def factor_emmeans_str = factor_emmeans_files.join(' ')
    """
    python ${projectDir}/bin/plot_pub_figures.py \\
        --cutoff_effects ${cutoff_effects} \\
        --reference_emmeans ${reference_emmeans} \\
        --method_emmeans ${method_emmeans_str} \\
        --factor_emmeans ${factor_emmeans_str} \\
        --organism ${params.organism} \\
        --outdir . \\
        --output_prefix pub_figure
    """
}
