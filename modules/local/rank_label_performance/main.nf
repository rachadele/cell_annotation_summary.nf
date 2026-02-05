process RANK_LABEL_PERFORMANCE {
    label 'process_single'

    input:
    path label_results

    output:
    path "rankings/*.tsv",                 emit: rankings
    path "rankings/rankings_detailed.tsv", emit: rankings_detailed
    path "rankings/rankings_best.tsv",     emit: rankings_best

    script:
    """
    python ${projectDir}/bin/rank_label_performance.py \
        --label_results ${label_results} \
        --cutoff ${params.cutoff} \
        --outdir rankings
    """
}
