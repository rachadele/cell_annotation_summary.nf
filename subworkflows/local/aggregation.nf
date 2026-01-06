/*
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    SUBWORKFLOW: AGGREGATION
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    Handles aggregation of results from multiple pipeline runs
----------------------------------------------------------------------------------------
*/

include { ADD_PARAMS        } from "$projectDir/modules/local/add_params/main"
include { AGGREGATE_RESULTS } from "$projectDir/modules/local/aggregate_results/main"

workflow AGGREGATION {

    take:
    ch_pipeline_results  // channel: [ run_name, params_file, ref_obs, f1_results ]

    main:
    ADD_PARAMS(ch_pipeline_results)
    AGGREGATE_RESULTS(ADD_PARAMS.out.f1_results_params.flatten().toList())

    emit:
    weighted_f1 = AGGREGATE_RESULTS.out.weighted_f1_results_aggregated
    label_f1    = AGGREGATE_RESULTS.out.label_f1_results_aggregated
}
