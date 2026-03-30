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
| `--method` | `scvi_rf` | Method to analyze (`scvi_rf`, `scvi_knn`, or `seurat`) |
| `--remove_outliers` | `null` | Dataset IDs to exclude |

## Modeling

By default, an ordered beta GLMM is fit to macro F1 scores using `glmmTMB`. Variables with no variation (e.g. a single cutoff value) are automatically dropped from the model formula. To skip modeling entirely:

```json
{ "skip_modeling": true }
```

The cutoff value at which emmeans marginal means are evaluated defaults to `0` and can be changed:

```json
{ "emmeans_cutoff": 0.1 }
```

## Test runs

Test parameter files for the kNN results are provided:

```bash
# Mouse
nextflow run main.nf -params-file params.mm.test_knn.json -profile conda -resume

# Human
nextflow run main.nf -params-file params.hs.test_knn.json -profile conda -resume

# Both
./run_test_knn.sh
```

## Output

Results are organized in subdirectories under `{census_version}/{organism}/{N}/dataset_id/{normalization}/gap_{bool}/`:

- `aggregated_results/` - Combined metrics across runs
- `cutoff_plots/` - F1 threshold analysis
- `f1_distributions/` - Macro and per-label F1 histograms by method and taxonomy level
- `aggregated_models/` - GLMM outputs (emmeans, contrasts, coefficients)
- `post-hoc-figures/` - Multi-panel publication figures
- `label_forest_plots/` - Per-label F1 forest plots across studies
- `celltype_rankings/` - Rankings, param heatmaps, Pareto configs, reliability plots
- `comptime_plots/` - Computational time comparisons
- `study_variance/` - Cell-type F1 variance across studies
- `assay_exploration/` - Mouse only: single-cell vs single-nuclei comparison
