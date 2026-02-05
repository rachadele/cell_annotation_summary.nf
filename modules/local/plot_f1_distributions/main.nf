process PLOT_F1_DISTRIBUTIONS {
    label 'process_single'

    input:
    path sample_results
    path label_results

    output:
    path "**png"
    path "**tsv"

    script:
    """
    python ${projectDir}/bin/plot_f1_distributions.py \\
        --sample_results ${sample_results} \\
        --label_results ${label_results} \\
        --outdir f1_distributions
    """
}
