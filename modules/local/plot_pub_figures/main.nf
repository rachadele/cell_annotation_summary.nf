process PLOT_PUB_FIGURES {

    label 'process_single'

    input:
    val weighted_f1_results
    val cutoff_effects
    val reference_emmeans
    val method_emmeans
    val all_emmeans_summary

    output:
    path "**.png", emit: figures

    script:
    // Combined files now contain all keys with 'key' column
    // Python script filters by key where needed
    def emmeans_str = all_emmeans_summary.join(' ')
    """
    python ${projectDir}/bin/plot_pub_figures.py \\
        --cutoff_effects ${cutoff_effects} \\
        --reference_emmeans ${reference_emmeans} \\
        --method_emmeans ${method_emmeans} \\
        --factor_emmeans ${emmeans_str} \\
        --organism ${params.organism} \\
        --outdir . \\
        --output_prefix pub_figure
    """
}
