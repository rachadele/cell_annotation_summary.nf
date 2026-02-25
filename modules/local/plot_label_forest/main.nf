process PLOT_LABEL_FOREST {
    label 'process_single'

    input:
    path label_results

    output:
    path "forest_plots/*.png", emit: forest_plots

    script:
    """
    python ${projectDir}/bin/plot_label_forest.py \
        --label_results ${label_results} \
        --organism ${params.organism} \
        --cutoff ${params.cutoff} \
        --subsample_ref ${params.subsample_ref} \
        --outdir forest_plots
    """
}
