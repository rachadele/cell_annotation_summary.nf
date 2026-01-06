process ADD_PARAMS {
    tag "$run_name"
    label 'process_single'
    conda '/home/rschwartz/anaconda3/envs/scanpyenv'

    input:
    tuple val(run_name), val(params_file), val(ref_obs), val(f1_results)

    output:
    path "*f1_results.tsv", emit: f1_results_params

    script:
    """
    python ${projectDir}/bin/add_params.py \\
        --run_name ${run_name} \\
        --ref_obs ${ref_obs} \\
        --f1_results ${f1_results} \\
        --params_file ${params_file}
    """
}
