process AGGREGATE_RESULTS {
    label 'process_low'

    input:
    path f1_results_params
    val  metadata_dir

    output:
    path "sample_results.tsv.gz", emit: sample_results_aggregated
    path "label_results.tsv.gz" , emit: label_results_aggregated
    path "**factor**tsv.gz"
    path "**summary.tsv.gz"

    script:
    def metadata_arg = metadata_dir ? "--metadata_dir ${metadata_dir}" : ""
    """
    python ${projectDir}/bin/aggregate_results.py --pipeline_results ${f1_results_params} ${metadata_arg}
    """
}
