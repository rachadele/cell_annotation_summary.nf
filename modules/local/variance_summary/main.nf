process VARIANCE_SUMMARY {
    label 'process_low'

    input:
    path label_results

    output:
    path "variance_summary/variance_dotplot_*.png",  emit: dotplots
    path "variance_summary/variance_scatter_*.png",  emit: scatters
    path "variance_summary/variance_summary.tsv",    emit: summary_tsv

    script:
    def cutoff_arg    = params.cutoff       != null ? "--cutoff ${params.cutoff}"                  : ""
    def subsample_arg = params.subsample_ref != null ? "--subsample_ref ${params.subsample_ref}"   : ""
    """
    python ${projectDir}/bin/variance_summary.py \
        --label_results ${label_results} \
        --organism ${params.organism} \
        --outdir variance_summary \
        ${cutoff_arg} \
        ${subsample_arg}
    """
}
