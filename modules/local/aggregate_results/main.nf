process AGGREGATE_RESULTS {
    label 'process_low'
    storeDir "${params.outdir}/aggregated_results/files"

    input:
    path f1_results_params
    val  metadata_dir
    val  remove_outliers

    output:
    path "sample_results.tsv.gz", emit: sample_results_aggregated
    path "label_results.tsv.gz" , emit: label_results_aggregated
    path "**factor**tsv.gz"
    path "**summary.tsv.gz"
    path "contamination.tsv", optional: true

    script:
    def metadata_arg  = metadata_dir  ? "--metadata_dir ${metadata_dir}"           : ""
    def outliers_arg  = (remove_outliers && remove_outliers != 'null') ? "--remove_outliers ${remove_outliers.join(' ')}" : ""
    """
    python ${projectDir}/bin/aggregate_results.py --pipeline_results ${f1_results_params} ${metadata_arg} ${outliers_arg}
    """
}
