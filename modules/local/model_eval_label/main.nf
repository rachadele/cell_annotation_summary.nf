process MODEL_EVAL_LABEL {
    tag "${key}_${label}"
    label 'process_medium'
    beforeScript 'ulimit -Ss unlimited'
    conda '/home/rschwartz/anaconda3/envs/r4.3'

    input:
    tuple val(key), val(label), path(label_f1_results_split)

    output:
    path "**png"
    path "**tsv"
    path "**model_summary_coefs_combined.tsv", emit: f1_model_summary_coefs
    path "**effects.tsv"                     , emit: continuous_effects

    script:
    """
    Rscript ${projectDir}/bin/model_performance_label.R \\
        --label_f1_results ${label_f1_results_split} \\
        --key ${key} \\
        --label ${label}
    """
}
