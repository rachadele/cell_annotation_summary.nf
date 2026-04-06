#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

for params_file in "${SCRIPT_DIR}"/params.*.test_knn.json; do
    echo "==> Running pipeline with ${params_file}"
    nextflow run "${SCRIPT_DIR}/main.nf" \
        -params-file "${params_file}" \
        -profile conda \
        -resume
    echo "==> Done: ${params_file}"
done
