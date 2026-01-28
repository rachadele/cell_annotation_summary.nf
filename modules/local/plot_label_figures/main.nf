process PLOT_LABEL_FIGURES {
    label 'process_single'

    input:
    val emmeans_files

    output:
    path "figures/*.png", emit: figures

    script:
    emmeans_files_joined = emmeans_files.join(' ')
    """
    mkdir -p figures

    python ${projectDir}/bin/plot_label_figures.py \\
        --emmeans_files ${emmeans_files_joined} \\
        --outdir figures
    """
}
