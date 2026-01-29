process MODEL_EVAL_LABEL {
    tag "${label}"
    label 'process_medium'
    beforeScript 'ulimit -Ss unlimited'

    input:
    tuple val(label), path(label_f1_results_split)

    output:
    path "**/figures/**/*.png", optional: true, emit: figures
    path "**/files/*.tsv"
    path "**/files/model_coefs.tsv"          , emit: f1_model_summary_coefs
    path "**/files/method_cutoff_effects.tsv", emit: continuous_effects
    path "**/files/*_emmeans_summary.tsv"    , emit: emmeans_summary

    script:
    """
    Rscript ${projectDir}/bin/model_performance_label.R \\
        --label_f1_results ${label_f1_results_split} \\
        --label "${label}"
    """
}
