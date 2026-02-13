process JOIN_METADATA {
    label 'process_single'

    input:
    path sample_results
    path label_results
    path study_meta
    path ref_meta

    output:
    path "sample_results_meta.tsv", emit: sample_results
    path "label_results_meta.tsv",  emit: label_results

    script:
    """
    python ${projectDir}/bin/join_metadata.py \
        --sample_results ${sample_results} \
        --label_results ${label_results} \
        --study_meta ${study_meta} \
        --ref_meta ${ref_meta}
    """
}
