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
include { PLOT_F1_DISTRIBUTIONS  } from "$projectDir/modules/local/plot_f1_distributions/main"
include { MODEL_EVAL_AGGREGATED  } from "$projectDir/modules/local/model_eval_aggregated/main"
include { GET_GRANT_SUMMARY      } from "$projectDir/modules/local/get_grant_summary/main"
include { PLOT_CELLTYPE_GRANULARITY } from "$projectDir/modules/local/plot_celltype_granularity/main"
include { PLOT_PUB_FIGURES       } from "$projectDir/modules/local/plot_pub_figures/main"
include { PLOT_LABEL_HEATMAP     } from "$projectDir/modules/local/plot_label_heatmap/main"
include { PLOT_LABEL_FOREST     } from "$projectDir/modules/local/plot_label_forest/main"
include { RANK_LABEL_PERFORMANCE   } from "$projectDir/modules/local/rank_label_performance/main"
include { PLOT_PARAM_HEATMAP       } from "$projectDir/modules/local/plot_param_heatmap/main"
include { PLOT_RANKING_SUMMARY     } from "$projectDir/modules/local/plot_ranking_summary/main"
include { PLOT_RANKING_RELIABILITY } from "$projectDir/modules/local/plot_ranking_reliability/main"
include { PLOT_CONFIG_PARETO      } from "$projectDir/modules/local/plot_config_pareto/main"
include { JOIN_METADATA           } from "$projectDir/modules/local/join_metadata/main"
include { MODEL_ASSAY_EFFECTS     } from "$projectDir/modules/local/model_assay_effects/main"
include { PLOT_ASSAY_EXPLORATION  } from "$projectDir/modules/local/plot_assay_exploration/main"

/*
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    RUN MAIN WORKFLOW
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
*/

workflow EVALUATION_SUMMARY {

    //
    // CHANNEL: Prepare input channel from results directory
    //
    Channel
        .fromPath("${params.results}/*/*", type: 'dir')
        .filter { file("${it}/params.yaml").exists() }
        .map { pipeline_run_dir ->
            def pipeline_run_dirname = pipeline_run_dir.getParent().getName().toString() + "_" + pipeline_run_dir.getName().toString()
            def params_file = "${pipeline_run_dir}/params.yaml"
            def ref_obs = "${pipeline_run_dir}/refs/"
            def pipeline_results = []
            pipeline_run_dir.eachDir { dir ->
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

    //
    // MODULE: Join study and reference metadata (mouse only)
    //
    if (params.organism == 'mus_musculus') {
        JOIN_METADATA(
            AGGREGATE_RESULTS.out.sample_results_aggregated,
            AGGREGATE_RESULTS.out.label_results_aggregated,
            file("${projectDir}/assets/study_metadata_mus_musculus.tsv"),
            file("${projectDir}/assets/reference_metadata_mus_musculus.tsv")
        )

        ch_sample_results = JOIN_METADATA.out.sample_results
        ch_label_results  = JOIN_METADATA.out.label_results

        //
        // MODULE: Model assay effects (mouse only)
        //
        MODEL_ASSAY_EFFECTS(ch_sample_results)

        //
        // MODULE: Plot assay exploration (mouse only)
        //
        PLOT_ASSAY_EXPLORATION(ch_sample_results, MODEL_ASSAY_EFFECTS.out.contrasts)
    } else {
        ch_sample_results = AGGREGATE_RESULTS.out.sample_results_aggregated
        ch_label_results  = AGGREGATE_RESULTS.out.label_results_aggregated
    }

    //
    // MODULE: Plot cutoff analysis
    //
    PLOT_CUTOFF(ch_sample_results, ch_label_results)

    //
    // MODULE: Generate grant summary
    //
    GET_GRANT_SUMMARY(ch_sample_results, ch_label_results)

    //
    // MODULE: Plot label distributions
    //
    //PLOT_LABEL_DIST(ch_label_results)

    //
    // MODULE: Plot F1 score distributions (macro and per-label)
    //
    PLOT_F1_DISTRIBUTIONS(ch_sample_results, ch_label_results)

    //
    // MODULE: Plot computation time
    //
    PLOT_COMPTIME("${params.results}")

    //
    // MODULE: Model evaluation for aggregated results (macro F1)
    //
    MODEL_EVAL_AGGREGATED(ch_sample_results)

    // Combined files now contain all keys with a 'key' column
    ch_cutoff_effects      = MODEL_EVAL_AGGREGATED.out.cutoff_effects
    ch_reference_emmeans   = MODEL_EVAL_AGGREGATED.out.reference_method_emmeans
    ch_method_emmeans      = MODEL_EVAL_AGGREGATED.out.method_emmeans
    ch_all_emmeans_summary = MODEL_EVAL_AGGREGATED.out.all_emmeans_summary

    //
    // MODULE: Generate publication figures
    //
    // Note: Combined files contain all keys - filtering by key happens in R script
    PLOT_PUB_FIGURES(
        ch_sample_results,
        ch_cutoff_effects,
        ch_reference_emmeans,
        ch_method_emmeans,
        ch_all_emmeans_summary
    )

    //
    // MODULE: Plot per-study label F1 heatmaps
    //
    //PLOT_LABEL_HEATMAP(ch_label_results)

    //
    // MODULE: Plot per-study label F1 forest plots
    //
    PLOT_LABEL_FOREST(ch_label_results)


    //
    // MODULE: Rank label performance across studies
    //
    RANK_LABEL_PERFORMANCE(ch_label_results)

    //
    // MODULE: Plot parameter performance heatmaps
    //
    PLOT_PARAM_HEATMAP(RANK_LABEL_PERFORMANCE.out.rankings_detailed)

    //
    // MODULE: Plot ranking summary dot plots
    //
    PLOT_RANKING_SUMMARY(RANK_LABEL_PERFORMANCE.out.rankings_best)

    //
    // MODULE: Plot ranking reliability scatter
    //
    PLOT_RANKING_RELIABILITY(RANK_LABEL_PERFORMANCE.out.rankings_best)

    //
    // MODULE: Plot configuration Pareto front (F1 vs compute cost)
    //
    PLOT_CONFIG_PARETO(
        RANK_LABEL_PERFORMANCE.out.rankings_detailed,
        PLOT_COMPTIME.out.comptime_summary
    )

    //
    // MODULE: Plot cell type granularity comparison (post-hoc)
    //
    // PLOT_CELLTYPE_GRANULARITY(ch_label_results)
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
