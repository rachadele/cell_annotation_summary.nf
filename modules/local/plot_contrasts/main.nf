process PLOT_CONTRASTS {
    tag "$key"
    label 'process_single'
    conda '/home/rschwartz/anaconda3/envs/scanpyenv'

    input:
    tuple val(key), val(contrast), path(emmeans_estimates), path(emmeans_summary)
    path weighted_f1_results_aggregated

    output:
    path "**png"

    script:
    """
    python ${projectDir}/bin/plot_contrasts.py \\
        --emmeans_estimates ${emmeans_estimates} \\
        --emmeans_summary ${emmeans_summary} \\
        --key ${key} \\
        --weighted_f1_results ${weighted_f1_results_aggregated}
    """
}
