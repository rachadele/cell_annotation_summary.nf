process PLOT_RANKING_SUMMARY {
    label 'process_single'

    input:
    path rankings_best

    output:
    path "ranking_summary/*.png", emit: summary_plots

    script:
    """
    python ${projectDir}/bin/plot_ranking_summary.py \
        --input ${rankings_best} \
        --outdir ranking_summary
    """
}
