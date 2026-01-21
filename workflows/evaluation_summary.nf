/*
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    IMPORT MODULES
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
*/

include { ADD_PARAMS             } from "$projectDir/modules/local/add_params/main"
include { AGGREGATE_RESULTS      } from "$projectDir/modules/local/aggregate_results/main"
include { PLOT_CUTOFF            } from "$projectDir/modules/local/plot_cutoff/main"
include { PLOT_COMPTIME          } from "$projectDir/modules/local/plot_comptime/main"
include { PLOT_LABEL_DIST        } from "$projectDir/modules/local/plot_label_dist/main"
include { MODEL_EVAL_WEIGHTED    } from "$projectDir/modules/local/model_eval_weighted/main"
include { SPLIT_BY_LABEL         } from "$projectDir/modules/local/split_by_label/main"
include { MODEL_EVAL_LABEL       } from "$projectDir/modules/local/model_eval_label/main"
include { PLOT_CONTRASTS         } from "$projectDir/modules/local/plot_contrasts/main"
include { PLOT_CONTINUOUS_CONTRAST } from "$projectDir/modules/local/plot_continuous_contrast/main"
include { GET_GRANT_SUMMARY      } from "$projectDir/modules/local/get_grant_summary/main"
include { PLOT_CELLTYPE_TRENDS   } from "$projectDir/modules/local/plot_celltype_trends/main"
include { PLOT_CELLTYPE_GRANULARITY } from "$projectDir/modules/local/plot_celltype_granularity/main"
/*
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    RUN MAIN WORKFLOW
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
*/

workflow EVALUATION_SUMMARY {

    //
    // SUBWORKFLOW: Prepare input channel from results directory
    //
    Channel
        .fromPath("${params.results}/*", type: 'dir')
        .map { pipeline_run_dir ->
            def pipeline_run_dirname = pipeline_run_dir.getName().toString()
            def params_file = "${pipeline_run_dir}/params.yaml"
            def ref_obs = "${pipeline_run_dir}/refs/"
            def pipeline_results = []
            pipeline_run_dir.eachDirRecurse { dir ->
                if (dir.getName() == 'scvi' || dir.getName() == 'seurat') {
                    def dir_path = dir.toString()
                    pipeline_results << dir_path
                }
            }
            [pipeline_run_dirname, params_file, ref_obs, pipeline_results.flatten().join(' ')]
        }
        .set { ch_all_pipeline_results }

    //
    // MODULE: Add parameters to results files
    //
    ADD_PARAMS(ch_all_pipeline_results)

    //
    // MODULE: Aggregate results across runs
    //
    AGGREGATE_RESULTS(ADD_PARAMS.out.f1_results_params.flatten().toList())

    ch_weighted_f1 = AGGREGATE_RESULTS.out.weighted_f1_results_aggregated
    ch_label_f1    = AGGREGATE_RESULTS.out.label_f1_results_aggregated

    //
    // MODULE: Plot cutoff analysis
    //
    PLOT_CUTOFF(ch_weighted_f1, ch_label_f1)

    //
    // MODULE: Generate grant summary
    //
    GET_GRANT_SUMMARY(ch_weighted_f1, ch_label_f1)

    //
    // MODULE: Plot label distributions
    //
    PLOT_LABEL_DIST(ch_label_f1)

    //
    // MODULE: Plot computation time
    //
    PLOT_COMPTIME("${params.results}")

    //
    // MODULE: Model evaluation for weighted results
    //
    MODEL_EVAL_WEIGHTED(ch_weighted_f1)

    ch_continuous_effects_weighted = MODEL_EVAL_WEIGHTED.out.continuous_effects
    ch_emmeans_estimates           = MODEL_EVAL_WEIGHTED.out.emmeans_estimates
    ch_emmeans_summary             = MODEL_EVAL_WEIGHTED.out.emmeans_summary

    //
    // CHANNEL: Parse emmeans estimates for plotting
    //
    ch_emmeans_estimates
        .flatMap { list ->
            list.collect { file ->
                def key = file.getParent().getParent().getName()
                def factors = file.getName().toString().split("_emmeans_estimates.tsv")[0]
                return [key, factors, file]
            }
        }
        .set { ch_emmeans_estimates_map }

    ch_emmeans_summary
        .flatMap { list ->
            list.collect { file ->
                def key = file.getParent().getParent().getName()
                def factors = file.getName().toString().split("_emmeans_summary.tsv")[0]
                return [key, factors, file]
            }
        }
        .set { ch_emmeans_summary_map }

    ch_emmeans_all = ch_emmeans_estimates_map.join(ch_emmeans_summary_map, by: [0,1])

    //
    // MODULE: Plot contrasts
    //
    PLOT_CONTRASTS(ch_emmeans_all, ch_weighted_f1)

    //
    // CHANNEL: Parse continuous effects for plotting
    //
    ch_continuous_effects_weighted
        .flatMap { list ->
            list.collect { file ->
                def key = file.getParent().getParent().getName()
                def mode = 'weighted'
                return [key, mode, file]
            }
        }
        .set { ch_continuous_effects_weighted_map }

    // NOTE: Label-level modeling is disabled due to segfault issues
    // Uncomment below to enable label-level analysis:
    // SPLIT_BY_LABEL(ch_label_f1)
    // MODEL_EVAL_LABEL(ch_label_f1_results_split_map)
    // PLOT_CONTINUOUS_CONTRAST(ch_continuous_effects_all)

    //
    // MODULE: Plot cell type trends (post-hoc)
    //
    PLOT_CELLTYPE_TRENDS(ch_label_f1)

    //
    // MODULE: Plot cell type granularity comparison (post-hoc)
    //
    PLOT_CELLTYPE_GRANULARITY(ch_label_f1)
}

/*
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    COMPLETION HANDLERS
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
*/

workflow.onError = {
    println "Error: something went wrong, check the pipeline log at '.nextflow.log"
}

workflow.onComplete = {
    println "Pipeline completed successfully!"
    println "Results are available in: ${params.outdir}"
}
