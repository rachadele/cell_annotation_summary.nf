process PLOT_PARAM_HEATMAP {

    input:
    path rankings_detailed

    output:
    path "param_heatmaps/*.png", emit: heatmaps

    script:
    """
    Rscript ${projectDir}/bin/plot_param_heatmap.R \
        --input ${rankings_detailed} \
        --outdir param_heatmaps
    """
}
