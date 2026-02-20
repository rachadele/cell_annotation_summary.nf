process PLOT_RANKING_RELIABILITY {
    label 'process_single'

    input:
    path rankings_best

    output:
    path "ranking_reliability/*.png", emit: reliability_plots

    script:
    """
    Rscript ${projectDir}/bin/plot_ranking_reliability.R \
        --input ${rankings_best} \
        --outdir ranking_reliability
    """
}
