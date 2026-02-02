process PLOT_PARAM_HEATMAP {
    label 'process_single'

    input:
    path rankings_detailed

    output:
    path "param_heatmaps/*.png", emit: heatmaps

    script:
    """
    python ${projectDir}/bin/plot_param_heatmap.py \
        --input ${rankings_detailed} \
        --outdir param_heatmaps
    """
}
