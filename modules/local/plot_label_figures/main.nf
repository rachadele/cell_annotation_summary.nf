process PLOT_LABEL_FIGURES {
    label 'process_single'

    input:
    val emmeans_files
    path label_f1_results

    output:
    path "**.png", emit: figures

    script:
    emmeans_files_joined = emmeans_files.join(' ')
    """
    python ${projectDir}/bin/plot_label_figures.py \\
        --emmeans_files ${emmeans_files_joined} \\
        --label_f1_results ${label_f1_results} 
    """
}
