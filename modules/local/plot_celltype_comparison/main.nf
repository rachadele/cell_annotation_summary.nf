process PLOT_CELLTYPE_COMPARISON {
    label 'process_low'

    input:
    path label_results

    output:
    path "celltype_comparison/*.png", emit: plots

    script:
    def cutoff_arg    = params.emmeans_cutoff != null ? "--cutoff ${params.emmeans_cutoff}" : ""
    def subsample_arg = params.subsample_ref  != null ? "--subsample_ref ${params.subsample_ref}" : ""
    """
    mkdir -p celltype_comparison
    for key in global family class subclass; do
        python ${projectDir}/bin/plot_celltype_comparison.py \\
            --results ${label_results} \\
            --outpath celltype_comparison/celltype_comparison_\${key}.png \\
            --key \${key} \\
            ${cutoff_arg} \\
            ${subsample_arg}
    done
    """
}
