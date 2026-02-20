process PLOT_CONFIG_PARETO {
    label 'process_single'

    input:
    path rankings_detailed
    path comptime_summary

    output:
    path "config_pareto/*.png", emit: pareto_plots
    path "config_pareto/config_pareto_table.tsv", emit: pareto_table

    script:
    """
    Rscript ${projectDir}/bin/plot_config_pareto.R \
        --rankings ${rankings_detailed} \
        --comptime ${comptime_summary} \
        --outdir config_pareto
    """
}
