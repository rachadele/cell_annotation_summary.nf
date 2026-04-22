process PLOT_CELLTYPE_COMPARISON {
    label 'process_low'

    input:
    path label_results

    output:
    path "celltype_comparison/*.png", emit: plots

    script:
    def cutoff_arg    = params.emmeans_cutoff != null ? "--cutoff ${params.emmeans_cutoff}" : ""
    def subsample_arg = params.subsample_ref  != null ? "--subsample_ref ${params.subsample_ref}" : ""
    def organism_arg  = params.organism       != null ? "--organism ${params.organism}" : ""
    """
    mkdir -p celltype_comparison
    for key in global family class subclass; do
        python ${projectDir}/bin/plot_celltype_comparison.py \\
            --results ${label_results} \\
            --outdir celltype_comparison \\
            --key \${key} \\
            ${organism_arg} \\
            ${cutoff_arg} \\
            ${subsample_arg}
    done
    """
}
