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
include { GET_GRANT_SUMMARY      } from "$projectDir/modules/local/get_grant_summary/main"
include { PLOT_CELLTYPE_GRANULARITY } from "$projectDir/modules/local/plot_celltype_granularity/main"
include { PLOT_PUB_FIGURES       } from "$projectDir/modules/local/plot_pub_figures/main"
include { PLOT_LABEL_FIGURES    } from "$projectDir/modules/local/plot_label_figures/main"

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

    ch_continuous_effects = MODEL_EVAL_WEIGHTED.out.continuous_effects
    ch_emmeans_summary    = MODEL_EVAL_WEIGHTED.out.emmeans_summary

    //
    // CHANNEL: Prepare files for publication figures
    //

    // Get cutoff effects file for subclass (primary key)
    ch_cutoff_effects_subclass = ch_continuous_effects
        .flatMap { list -> list }
        .filter { file ->
            file.getParent().getParent().getName() == 'subclass' &&
            file.getName() == 'method_cutoff_effects.tsv'
        }
        .first()

    // Get reference_method emmeans for subclass
    ch_reference_emmeans_subclass = ch_emmeans_summary
        .flatMap { list -> list }
        .filter { file ->
            file.getParent().getParent().getName() == 'subclass' &&
            file.getName() == 'reference_method_emmeans_summary.tsv'
        }
        .first()

    // Get method emmeans for all taxonomy levels (for slope chart)
    ch_method_emmeans = ch_emmeans_summary
        .flatMap { list -> list }
        .filter { file -> file.getName() == 'method_emmeans_summary.tsv' }
        .collect()
    // Get factor emmeans for subclass (disease_state, sex, region_match, treatment)
    ch_factor_emmeans = ch_emmeans_summary
        .flatMap { list -> list }
        .filter { file ->
            file.getParent().getParent().getName() == 'subclass' &&
            (file.getName() =~ /^(disease_state|disease|sex|region_match|treatment_state|treatment)_emmeans_summary\.tsv$/)
        }
        .collect()

    //
    // MODULE: Generate publication figures
    //

    // view all channels used for publication figures
    


    PLOT_PUB_FIGURES(
        ch_weighted_f1,
        ch_cutoff_effects_subclass,
        ch_reference_emmeans_subclass,
        ch_method_emmeans,
        ch_factor_emmeans
    )

    //
    // MODULE: Split label F1 results by key and label
    //
    SPLIT_BY_LABEL(ch_label_f1)

    //
    // CHANNEL: Map split label files to (label, file) tuples
    //
    ch_label_f1_results_split_map = SPLIT_BY_LABEL.out.label_f1_results_split
        .flatten()
        .map { file ->
            def label = file.getParent().getName()
            [label, file]
        }

    //
    // MODULE: Model evaluation for label-level results (fixed-effects beta regression)
    //
    MODEL_EVAL_LABEL(ch_label_f1_results_split_map)

    //
    // MODULE: Plot label-level model results (forest plots)
    //
    ch_label_emmeans = MODEL_EVAL_LABEL.out.emmeans_summary
        .collect()
    PLOT_LABEL_FIGURES(ch_label_emmeans)

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
