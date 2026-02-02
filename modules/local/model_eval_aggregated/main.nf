process MODEL_EVAL_AGGREGATED {
    //label 'process_high'

    input:
    path aggregated_f1_results

    output:
    path "**/figures/**/*.png"            , emit: figures
    path "**/files/*.tsv"                 , emit: all_results
    path "**/files/model_coefs.tsv"       , emit: model_coefs
    path "**/files/method_cutoff_effects.tsv", emit: cutoff_effects
    path "**/files/reference_method_emmeans_summary.tsv", emit: reference_method_emmeans
    path "**/files/method_emmeans_summary.tsv", emit: method_emmeans
    path "**/files/*_emmeans_summary.tsv" , emit: all_emmeans_summary

    script:
    """
    Rscript ${projectDir}/bin/model_performance_aggregated.R --aggregated_f1_results ${aggregated_f1_results}
    """
}
