process PLOT_COMPTIME {
    label 'process_single'
    conda '/home/rschwartz/anaconda3/envs/scanpyenv'

    input:
    path all_runs_dir

    output:
    path "comptime.png"
    path "comptime_summary.tsv"

    script:
    """
    python ${projectDir}/bin/plot_comptime.py --all_runs ${all_runs_dir}
    """
}
