process PLOT_METHOD_STUDY_COMPARISON {
    label 'process_low'

    input:
    path sample_results

    output:
    path "method_study_comparison.png", emit: plot
    path "method_study_raw_scores.tsv",  emit: raw_scores

    script:
    def cutoff_arg      = params.emmeans_cutoff != null ? "--cutoff ${params.emmeans_cutoff}" : ""
    def subsample_arg   = params.subsample_ref  != null ? "--subsample_ref ${params.subsample_ref}" : ""
    """
    python ${projectDir}/bin/plot_method_study_comparison.py \\
        --results ${sample_results} \\
        --outpath method_study_comparison.png \\
        ${cutoff_arg} \\
        ${subsample_arg}
    """
}
