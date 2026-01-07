process MODEL_EVAL_WEIGHTED {
    label 'process_medium'

    input:
    path weighted_f1_results_aggregated

    output:
    path "**png"
    path "**tsv"
    path "**emmeans_estimates.tsv"        , emit: emmeans_estimates
    path "**emmeans_summary.tsv"          , emit: emmeans_summary
    path "**model_summary_coefs_combined.tsv", emit: f1_model_summary_coefs
    path "**effects.tsv"                  , emit: continuous_effects

    script:
    """
    Rscript ${projectDir}/bin/model_performance_weighted.R --weighted_f1_results ${weighted_f1_results_aggregated}
    """
}
