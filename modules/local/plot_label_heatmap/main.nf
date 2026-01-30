process PLOT_LABEL_HEATMAP {
    label 'process_single'

    input:
    path label_f1_results

    output:
    path "heatmaps/**/*.png", emit: heatmaps

    script:
    """
    python ${projectDir}/bin/plot_label_heatmap.py \
        --label_f1_results ${label_f1_results} \
        --organism ${params.organism} \
        --cutoff ${params.cutoff} \
        --subsample_ref ${params.subsample_ref} \
        --outdir heatmaps
    """
}
