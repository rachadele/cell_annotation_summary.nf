process AGGREGATE_RESULTS {
    label 'process_low'

    input:
    path f1_results_params

    output:
    path "sample_results.tsv", emit: sample_results_aggregated
    path "label_results.tsv" , emit: label_results_aggregated
    path "**factor**tsv"
    path "**summary.tsv"

    script:
    """
    python ${projectDir}/bin/aggregate_results.py --pipeline_results ${f1_results_params}
    """
}
