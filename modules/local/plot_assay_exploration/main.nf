process PLOT_ASSAY_EXPLORATION {

    input:
    path sample_results
    path contrasts

    output:
    path "assay_exploration/*.png", emit: figures

    script:
    """
    Rscript ${projectDir}/bin/plot_assay_exploration.R --sample_results ${sample_results} --contrasts ${contrasts}
    """
}
