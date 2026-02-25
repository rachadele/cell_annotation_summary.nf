process PLOT_COMPTIME {
    label 'process_single'

    input:
    path all_runs_dir

    output:
    path "comptime.png"
    path "comptime_summary.tsv", emit: comptime_summary

    script:
    """
    Rscript ${projectDir}/bin/plot_comptime.R --all_runs ${all_runs_dir}
    """
}
