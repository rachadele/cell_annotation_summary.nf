
process aggregateMeta {
    conda '/home/rschwartz/anaconda3/envs/scanpyenv'
    publishDir "${params.outdir}/meta_aggregation", mode: 'copy'

    input:
    tuple val(run_name), val(params_file), val(ref_obs), val(results_dirs)
 

    output:
    path "**combined_meta.tsv", emit: combined_meta

    script:
    """
    python $projectDir/bin/aggregate_meta.py --run_name ${run_name} --params_file ${params_file} \\
            --ref_obs ${ref_obs} --results_dirs ${results_dirs}
    """
}


process individualCells {
    conda '/home/rschwartz/anaconda3/envs/scanpyenv'
    publishDir "${params.outdir}/cell_level_analysis", mode: 'copy'

    input:
    path(combined_meta)

    output:
    "**tsv"
    "**png"

    script:


    """
    python $projectDir/bin/cell_level_analysis.py --combined_meta ${combined_meta} \\
    """

}


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
    publishDir "${params.outdir}/aggregated_results/figs", mode: 'copy', pattern: "**png"
    publishDir "${params.outdir}/aggregated_results/files", mode: 'copy', pattern: "**tsv"

    input:
    path f1_results_params

    output:
   // path "f1_results_all_pipeline_runs.tsv", emit: f1_results_aggregated
    path "weighted_f1_results.tsv", emit: weighted_f1_results_aggregated
    path "label_f1_results.tsv", emit: label_f1_results_aggregated
    path "**factor**tsv"
    path "**summary.tsv"
    path "**png"

    script:
    """
    python $projectDir/bin/aggregate_results.py --pipeline_results ${f1_results_params}
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
    python $projectDir/bin/plot_cutoff.py \\
            --weighted_f1_results ${weighted_f1_results_aggregated} \\
            --label_f1_results ${label_f1_results_aggregated} \\
            --mapping_file ${params.mapping_file} \\
            --color_mapping_file ${params.color_mapping_file}
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
    python $projectDir/bin/label_dists.py --label_f1_results ${label_f1_results_aggregated} \\
                --mapping_file ${params.mapping_file} \\
                --color_mapping_file ${params.color_mapping_file}
    """

}

process modelEvalWeighted {
    conda '/home/rschwartz/anaconda3/envs/r4.3' 
    publishDir "${params.outdir}/models/weighted/files", mode: 'copy', pattern: "**tsv"
    publishDir "${params.outdir}/models/weighted/figs", mode: 'copy', pattern: "**png"

    input:
    path weighted_f1_results_aggregated

    output:
    path "**png"
    path "**tsv"
    path "**emmeans_estimates.tsv", emit: emmeans_estimates
    path "**emmeans_summary.tsv", emit: emmeans_summary
    path "**model_summary_coefs_combined.tsv", emit: f1_model_summary_coefs
    path "**effects.tsv", emit: continuous_effects

    script:
    """
    Rscript $projectDir/bin/model_performance_weighted.R --weighted_f1_results ${weighted_f1_results_aggregated}
    """
}
process split_by_label {
    conda '/home/rschwartz/anaconda3/envs/scanpyenv'
    publishDir "${params.outdir}/label_results", mode: 'copy'

    input:
    path label_f1_results_aggregated

    output:
    path "**tsv", emit: label_f1_results_split
    path "**png"

    script:
    """
    python $projectDir/bin/split_by_label.py --label_f1_results ${label_f1_results_aggregated}
    """
}

process modelEvalLabel {
    beforeScript 'ulimit -Ss unlimited' // Increase stack size limit for R script
    conda '/home/rschwartz/anaconda3/envs/r4.3' 
    publishDir "${params.outdir}/models/label/files", mode: 'copy', pattern: "**tsv"
    publishDir "${params.outdir}/models/label/figs", mode: 'copy', pattern: "**png"

    input:
    tuple val(key), val(label), path(label_f1_results_split)

    output:
    path "**png"
    path "**tsv"
    path "**model_summary_coefs_combined.tsv", emit: f1_model_summary_coefs
    path "**effects.tsv", emit: continuous_effects

    script:
    """
    Rscript $projectDir/bin/model_performance_label.R --label_f1_results ${label_f1_results_split} \\
            --key ${key} --label ${label}
    """
}


process plotContrasts {
    conda '/home/rschwartz/anaconda3/envs/scanpyenv'
    publishDir "${params.outdir}/contrast_figs/weighted", mode: 'copy'

    input:
    tuple val(key), val(contrast), path(emmeans_estimates), path(emmeans_summary)
    path weighted_f1_results_aggregated

    output:
    path "**png"

    script:
    """
    python $projectDir/bin/plot_contrasts.py --emmeans_estimates ${emmeans_estimates} \\
                    --emmeans_summary ${emmeans_summary} \\
                    --key ${key} \\
                    --weighted_f1_results ${weighted_f1_results_aggregated}
    """

}

process plot_continuous_contrast {
    conda '/home/rschwartz/anaconda3/envs/scanpyenv'
    publishDir "${params.outdir}/contrast_figs/${mode}/${key}/continuous_contrasts", mode: 'copy'

    input:
    tuple val(key), val(mode), path(continuous_effects) // mode can be 'weighted' or 'label'

    output:
    path "**png"

    script:
    """
    python $projectDir/bin/plot_continuous_contrasts.py --key ${key} --contrast ${continuous_effects} 
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
    
    plotCutoff(weighted_f1_results_aggregated, label_f1_results_aggregated)
    plotHeatmap(weighted_f1_results_aggregated)
    plotLabelDist(label_f1_results_aggregated)
    
    // plot comptime
    all_runs_dir = "${params.results}"
    plotComptime(all_runs_dir) 

    // model evaluation
    modelEvalWeighted(weighted_f1_results_aggregated)

    continuous_effects_weighted = modelEvalWeighted.out.continuous_effects
    emmeans_estimates = modelEvalWeighted.out.emmeans_estimates
    emmeans_summary = modelEvalWeighted.out.emmeans_summary
// need to get individual files in order to plot contrasts


    emmeans_estimates
        .flatMap { list ->
            // Iterate through each file in the ArrayList
            list.collect { file ->
                // def key = parent dir name two dirs up
                def key = file.getParent().getParent().getName()
 
                // Split the file name to extract the factors before "_emmeans_estimates.tsv"
                def factors = file.getName().toString().split("_emmeans_estimates.tsv")[0]
                // Return a map with factors and the file itself
                return [key, factors, file]
            }
        }
        .set { emmeans_estimates_map }

    emmeans_summary
        .flatMap { list ->
            // Iterate through each file in the ArrayList
            list.collect { file ->
                def key = file.getParent().getParent().getName()

                // Split the file name to extract the factors before "_emmeans_estimates.tsv"
                def factors = file.getName().toString().split("_emmeans_summary.tsv")[0]
                // Return a map with factors and the file itself
                return [key, factors, file]
            }
        }
        .set { emmeans_summary_map }

    emmeans_all = emmeans_estimates_map.join(emmeans_summary_map, by: [0,1])

    plotContrasts(emmeans_all, weighted_f1_results_aggregated)
    
    continuous_effects_weighted
        .flatMap { list ->
            list.collect { file ->
                // def key = parent dir name two dirs up
                def key = file.getParent().getParent().getName()
                def mode = 'weighted' // or 'label' based on the process
                return [key, mode, file]
            }
        }
        .set { continuous_effects_weighted_map }

    // split label results
    split_by_label(label_f1_results_aggregated)
    split_by_label.out.label_f1_results_split
    // label_f1_results_split = split_by_label.out.label_f
    .set { label_f1_results_split }

    // split label_f1_results_split into individual files
    label_f1_results_split.flatMap { list ->
            // Iterate through each file in the ArrayList
            list.collect { file ->
                def key = file.getParent().getParent().getName()
                def label = file.getParent().getName() // Assuming the label is the parent directory name
                def filepath = file
                return [key, label, filepath]
            }
        }.set { label_f1_results_split_map }


    modelEvalLabel(label_f1_results_split_map) 
    continuous_effects_label = modelEvalLabel.out.continuous_effects
    // flatMap the mode onto continuous_effects_label
    continuous_effects_label.map { file ->
                def key = file.getParent().getParent().getName()
                def mode = 'label' // or 'weighted' based on the process
                return [key, mode, file]
            }
        
        .set { continuous_effects_label_map }
    

    continuous_effects_all = continuous_effects_weighted_map.concat(continuous_effects_label_map)
    //continuous_effects_all.view()
    plot_continuous_contrast(continuous_effects_all)
}

workflow.onError = {
println "Error: something went wrong, check the pipeline log at '.nextflow.log"
}