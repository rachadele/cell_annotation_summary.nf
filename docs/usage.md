# nf-core/evaluation_summary: Usage

## Introduction

This pipeline aggregates and analyzes results from the cell-type classification benchmarking pipeline (`nextflow_eval_pipeline`).

## Running the pipeline

The typical command for running the pipeline is:

```bash
nextflow run main.nf -params-file params.json
```

### Using profiles

```bash
# Run with conda
nextflow run main.nf -profile conda -params-file params.json

# Run with test data
nextflow run main.nf -profile test
```

## Parameters

### Required parameters

| Parameter | Description |
|-----------|-------------|
| `--results` | Path to directory containing pipeline run results |
| `--outdir` | Output directory for results |

### Reference options

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--organism` | `mus_musculus` | Organism (`homo_sapiens` or `mus_musculus`) |
| `--census_version` | `2024-07-01` | CellxGene Census version |
| `--ref_keys` | `[subclass, class, family, global]` | Hierarchy levels for evaluation |
| `--mapping_file` | - | Cell type mapping file |
| `--color_mapping_file` | - | Color mapping for plots |

### Filtering options

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--subsample_ref` | `500` | Reference subsampling count |
| `--cutoff` | `0.0` | F1 score cutoff |
| `--method` | `scvi` | Method to analyze (`scvi` or `seurat`) |
| `--remove_outliers` | `null` | Dataset IDs to exclude |

## Output

Results are organized in subdirectories:

- `aggregated_results/` - Combined metrics across runs
- `cutoff_plots/` - F1 threshold analysis
- `grant_summary/` - Summary statistics
- `weighted_models/` - Statistical model outputs
- `contrast_figs/` - Effect visualizations
