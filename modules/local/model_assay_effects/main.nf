process MODEL_ASSAY_EFFECTS {
    label 'process_medium'

    input:
    path sample_results

    output:
    path "**/figures/**/*.png"                      , emit: figures
    path "**/files/*.tsv"                           , emit: all_results
    path "**/files/assay_emmeans_contrasts.tsv"      , emit: contrasts

    script:
    """
    Rscript ${projectDir}/bin/model_assay_effects.R --sample_results ${sample_results}
    """
}
