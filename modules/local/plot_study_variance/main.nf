process PLOT_STUDY_VARIANCE {
    label 'process_single'

    input:
    path label_results

    output:
    path "study_variance/*.png",  emit: plots
    path "study_variance/*.pdf",  emit: pdfs
    path "study_variance/*.tsv",  emit: summary

    script:
    """
    python ${projectDir}/bin/plot_study_variance.py \
        --label_results ${label_results} \
        --organism ${params.organism} \
        --key subclass \
        --cutoff ${params.cutoff} \
        --outdir study_variance
    """
}
