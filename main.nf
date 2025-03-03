process addParams {
    conda '/home/rschwartz/anaconda3/envs/scanpyenv'
    publishDir "${params.outdir}/params_added", mode: 'copy'

    input:
    //val pipeline_run_dir_name
    tuple val(run_name), val(params_file), val(ref_obs), val(f1_results)


    output:
    path "*f1_results.tsv", emit: f1_results_params

    script:
    """
    python $projectDir/bin/add_params.py --run_name ${run_name}  --ref_obs ${ref_obs}  --f1_results ${f1_results} --params_file ${params_file}
    """

}

process aggregateResults {
    conda '/home/rschwartz/anaconda3/envs/scanpyenv'
    publishDir "${params.outdir}/aggregated_results", mode: 'copy'

    input:
    path f1_results_params

    output:
   // path "f1_results_all_pipeline_runs.tsv", emit: f1_results_aggregated
    path "weighted_f1_results.tsv", emit: weighted_f1_results_aggregated
    path "label_f1_results.tsv", emit: label_f1_results_aggregated
    path "**summary.tsv"
    path "**value_counts.tsv"
    path "**png"

    script:
    """
    python $projectDir/bin/aggregate_results.py --pipeline_results ${f1_results_params}
    """

}

process runAnova {
    conda '/home/rschwartz/anaconda3/envs/scanpyenv'
    publishDir "${params.outdir}/anova_results", mode: 'copy'

    input:
    path weighted_f1_results_aggregated
    path label_f1_results_aggregated

    output:

    path "*anova_*.tsv"
    path "*png"

    script:
    """
    python $projectDir/bin/run_anova.py --weighted_f1_results ${weighted_f1_results_aggregated} --label_f1_results ${label_f1_results_aggregated}
    """

}

process plotCutoff {
    conda '/home/rschwartz/anaconda3/envs/scanpyenv'
    publishDir "${params.outdir}/cutoff_plots", mode: 'copy'

    input:
    path weighted_f1_results_aggregated
    path label_f1_results_aggregated

    output:
    path "**png"

    script:
    """
    python $projectDir/bin/plot_cutoff.py --weighted_f1_results ${weighted_f1_results_aggregated} --label_f1_results ${label_f1_results_aggregated}
    """
}

process plotHeatmap {
    conda '/home/rschwartz/anaconda3/envs/scanpyenv'
    publishDir "${params.outdir}/heatmap_plots", mode: 'copy'

    input:
    path weighted_f1_results_aggregated

    output:
    path "**png"
    path "na_values.tsv"

    script:
    """
    python $projectDir/bin/plot_heatmaps.py --weighted_f1_results ${weighted_f1_results_aggregated}
    """
}

process plotComptime {
    conda '/home/rschwartz/anaconda3/envs/scanpyenv'
    publishDir "${params.outdir}/comptime_plots", mode: 'copy'

    input:
    path all_runs_dir

    output:
    path "comptime.png"
    path "comptime_summary.tsv"

    script:
    """
    python $projectDir/bin/plot_comptime.py --all_runs ${all_runs_dir}
    """
}

process plotLabelDist {
    conda '/home/rschwartz/anaconda3/envs/scanpyenv'
    publishDir "${params.outdir}/label_dists", mode: 'copy'

    input:
    path label_f1_results_aggregated

    output:
    path "**png"

    script:
    """
    python $projectDir/bin/label_dists.py --label_f1_results ${label_f1_results_aggregated} --mapping_file ${params.mapping_file}
    """

}

process modelEval {
    conda '/home/rschwartz/anaconda3/envs/r4.3' 
    publishDir "${params.outdir}/model_eval", mode: 'copy'

    input:
    path weighted_f1_results_aggregated
    path label_f1_results_aggregated

    output:
    path "**png"
    path "**model_summary_coefs_combined.tsv", emit: f1_model_summary_coefs


    script:
    """
    Rscript $projectDir/bin/model_performance.R --weighted_f1_results ${weighted_f1_results_aggregated}
    """
}

process plotContrasts {
    conda '/home/rschwartz/anaconda3/envs/scanpyenv'
    publishDir "${params.outdir}/contrasts", mode: 'copy'

    input:
    tuple path(f1_model_summary_coefs), val(f1_type), path(weighted_f1_results_aggregated)

    output:
    path "**png"
    path "**tsv"

    script:
    """
    python $projectDir/bin/plot_contrasts.py --model_summary_coefs ${f1_model_summary_coefs} \\
        --type ${f1_type} \\
        --f1_results ${weighted_f1_results_aggregated}
    """
}

workflow {

    Channel
    .fromPath("${params.results}/*", type: 'dir') // Get all subdirectories
    .map { pipeline_run_dir ->
        def pipeline_run_dirname = pipeline_run_dir.getName().toString()
        def params_file = "${pipeline_run_dir}/params.yaml"
        def ref_obs = "${pipeline_run_dir}/refs/"
        // Collect 'f1_results' directories
        def pipeline_results = []
        pipeline_run_dir.eachDirRecurse { dir ->
                if (dir.getName() == 'scvi' || dir.getName() == 'seurat') {
                        def dir_path = dir.toString()
                        pipeline_results << dir_path
            }
        }

        [pipeline_run_dirname, params_file, ref_obs, pipeline_results.flatten().join(' ')] // Return collected results for this pipeline_run_dir
    }

    .set { all_pipeline_results } 
    // add parameters to files  addParams(all_pipeline_results) 
    addParams(all_pipeline_results)

    aggregateResults(addParams.out.f1_results_params.flatten().toList())

    // plot aggregated results  aggregateResults(addParams.out.f1_results_params.flatten().toList())
    weighted_f1_results_aggregated = aggregateResults.out.weighted_f1_results_aggregated   
    label_f1_results_aggregated = aggregateResults.out.label_f1_results_aggregated 
    
    // run ANOVA on aggregated results
    plotCutoff(weighted_f1_results_aggregated, label_f1_results_aggregated)
    runAnova(weighted_f1_results_aggregated, label_f1_results_aggregated)
    plotHeatmap(weighted_f1_results_aggregated)
    plotLabelDist(label_f1_results_aggregated)
    
    // plot comptime
    all_runs_dir = "${params.results}"
    plotComptime(all_runs_dir) 

    // model evaluation
    modelEval(weighted_f1_results_aggregated, label_f1_results_aggregated)

    //f1_model_summary_coefs = modelEval.out.f1_model_summary_coefs
    //f1_model_summary_coefs.view()
    // get types of f1 score

    //f1_model_summary_coefs.map { path ->
        //def f1_type = path.getName().toString().split('_')[0]
        //[path, f1_type]
    //}.set { f1_model_summary_coefs_types }
    //f1_model_summary_coefs_types.view()
    //f1_model_summary_coefs_types.combine(weighted_f1_results_aggregated)
    //.set{f1_model_summary_coefs_results_aggregated}

    //f1_model_summary_coefs_results_aggregated.view()

    //plotContrasts(f1_model_summary_coefs_results_aggregated)

}

workflow.onError = {
println "Error: something went wrong, check the pipeline log at '.nextflow.log"
}