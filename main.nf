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
    publishDir "${params.outdir}", mode: 'copy'

    input:
    path f1_results_params

    output:
   // path "f1_results_all_pipeline_runs.tsv", emit: f1_results_aggregated
    path "weighted_f1_results.tsv", emit: weighted_f1_results_aggregated
    path "label_f1_results.tsv", emit: label_f1_results_aggregated
  //  path "label_distributions/*"
   // path "weighted_f1_distributions/*"

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
    path reports_ch

    output:
    path "**png"

    script:
    """
    python $projectDir/bin/plot_comptime.py --reports_dir ${reports_ch}
    """
}

process labelSupportCorr {
    conda '/home/rschwartz/anaconda3/envs/scanpyenv'
    publishDir "${params.outdir}/label_support_corr", mode: 'copy'

    input:
    path label_f1_results_aggregated

    output:
    path "**png"

    script:
    """
    python $projectDir/bin/label_support_corr.py --label_f1_results ${label_f1_results_aggregated}
    """

}

process modelEval {
    conda '/home/rschwartz/anaconda3/envs/scanpyenv'
    publishDir "${params.outdir}/model_eval", mode: 'copy'

    input:
    path weighted_f1_results_aggregated
    path label_f1_results_aggregated

    output:
    path "**png"
    path "**tsv"

    script:
    """
    python $projectDir/bin/model_performance.py --weighted_f1_results ${weighted_f1_results_aggregated} --label_f1_results ${label_f1_results_aggregated}
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
    all_pipeline_results.view()
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
    labelSupportCorr(label_f1_results_aggregated)
    
    // plot comptime
   // Channel
   // .fromPath("${params.reports_dir}")
   // .set { reports_ch }
    //plotComptime(reports_ch)
   // reports_ch.view()

    // plotComptime(reports_ch) 

    // model evaluation
    modelEval(weighted_f1_results_aggregated, label_f1_results_aggregated)

}

workflow.onError = {
println "Error: something went wrong, check the pipeline log at '.nextflow.log"
}