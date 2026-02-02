process PLOT_F1_DISTRIBUTIONS {
    label 'process_single'

    input:
    path weighted_f1_results
    path label_f1_results

    output:
    path "**png"
    path "**tsv"

    script:
    """
    python ${projectDir}/bin/plot_f1_distributions.py \\
        --weighted_f1_results ${weighted_f1_results} \\
        --label_f1_results ${label_f1_results} \\
        --outdir f1_distributions
    """
}
