process GET_GRANT_SUMMARY {
    label 'process_single'

    input:
    path sample_results_aggregated
    path label_results_aggregated

    output:
    path "outliers**"

    script:
    def ref_keys = params.ref_keys.join(' ')
    def outlier_arg = ''
    if (params.remove_outliers && params.remove_outliers != null) {
        outlier_arg = '--remove_outliers ' + params.remove_outliers.join(' ')
    }
    """
    python ${projectDir}/bin/grant_summary.py \\
        --weighted_metrics ${sample_results_aggregated} \\
        --label_metrics ${label_results_aggregated} \\
        --ref_keys ${ref_keys} \\
        --subsample_ref ${params.subsample_ref} \\
        --cutoff ${params.cutoff} \\
        --reference '${params.reference}' \\
        --method ${params.method} \\
        --organism ${params.organism} \\
        ${outlier_arg}
    """
}
