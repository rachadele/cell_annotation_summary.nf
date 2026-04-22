process VARIANCE_SUMMARY {
    label 'process_low'

    input:
    path label_results

    output:
    path "variance_summary/study_variance_*.png",                      emit: heatmaps
    path "variance_summary/study_variance_*_summary.tsv",              emit: heatmap_summaries
    path "variance_summary/method_stability_summary.tsv",              emit: method_stability
    path "variance_summary/method_reference_stability_summary.tsv",    emit: method_reference_stability
    path "variance_summary/celltype_stability_summary.tsv",            emit: celltype_stability
    path "variance_summary/celltype_stability.png",                    emit: stability_plot

    script:
    def cutoff_arg      = params.cutoff       != null ? "--cutoff ${params.cutoff}"             : ""
    def subsample_arg   = params.subsample_ref != null ? "--subsample_ref ${params.subsample_ref}" : ""
    """
    python ${projectDir}/bin/variance_summary.py \
        --label_results ${label_results} \
        --organism ${params.organism} \
        --outdir variance_summary \
        ${cutoff_arg} \
        ${subsample_arg}
    """
}
