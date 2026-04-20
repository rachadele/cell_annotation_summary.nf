# evaluation_summary.nf

## Project Overview
This repository implements a modular Nextflow pipeline (DSL2) for summarizing and evaluating cell annotation results across multiple datasets and methods. It compares three classifiers — **scVI RF** (random forest on scVI embeddings), **scVI kNN** (k-nearest neighbours on scVI embeddings), and **Seurat** — across GEO studies for mouse and human brain data. It integrates R and Python scripts for data aggregation, statistical modelling, and visualization.

## Architecture & Data Flow
- **Entry Point:** `main.nf` includes the main workflow from `workflows/evaluation_summary.nf`.
- **Modules:** Each analysis step is a DSL2 module in `modules/local/` (e.g., `add_params`, `aggregate_results`, `plot_cutoff`).
- **Data Flow:** Results directories are scanned, parameters are added, results are aggregated, and various plots and summaries are generated. Nextflow channels pass data between modules.
- **Configuration:** Pipeline parameters and profiles are set in `nextflow.config` and `conf/`.
- **Scripts:** Analysis and plotting scripts are in `bin/` (Python and R).

## Developer Workflows
- **Run Pipeline:**
  ```bash
  nextflow run main.nf --results <results_dir> --organism <organism> [...other params]
  ```
- **Using a Parameters File:**
```bash
  nextflow run main.nf --params-file <path_to_params_file.json>
  ```
- **Cleanup:** Use `nf-cleanup` to remove intermediate files.
- **Profiles:** Use `-profile conda` for environment management.
- **Add/Update Modules:** Place new modules in `modules/local/` and include them in the workflow as needed.

## Conventions & Patterns
- **Module Naming:** All modules are in `modules/local/` and named after their function.
- **Scripts:** Python and R scripts in `bin/` are called by modules for data processing and plotting.
- **Results Structure:** Output is organized under `{census_version}/{organism}/{N}/dataset_id/{normalization}/gap_{bool}/` (e.g., `2024-07-01/mus_musculus/100/dataset_id/SCT/gap_false/`). `outdir` is auto-derived from `params.results` by stripping the path up to `nextflow[_-]eval[_-]pipeline/(results/)?`.
- **Data formats:** All aggregated outputs are gzip-compressed TSV (`.tsv.gz`). Two main dataframes flow through the pipeline: `sample_results.tsv.gz` (sample-level macro F1) and `label_results.tsv.gz` (per-cell-type F1).
- **Parameter Passing:** Use Nextflow channels to pass structured data between modules.
- **Aggregation:** Aggregation and summary files are written to `{outdir}/aggregated_results/files/` via `storeDir`.

## Integration Points
- **External Tools:**
  - Python (`scanpyenv` conda env): `pandas`, `numpy`, `matplotlib`, `seaborn`, `scipy` — used for aggregation, ranking, and all Python-based plots.
  - R (`r4.3` conda env): `glmmTMB`, `emmeans`, `ggplot2`, `broom.mixed` — used for ordered beta GLMMs, marginal means, and publication figures.
- **Environment:** Conda only (active profile). Docker and Singularity profiles are defined but commented out.
- **Compute:** SLURM executor; `workDir` is hardcoded to `/cosmos/data/evaluation-summary-work`.

## Key Files & Directories
- `main.nf` – Pipeline entry point
- `workflows/evaluation_summary.nf` – Main workflow logic
- `modules/local/` – DSL2 modules for each step
- `bin/` – Python and R scripts for analysis/plotting
- `bin/model_functions.R` – Shared R helpers for GLMM fitting and emmeans
- `bin/plot_utils.py` – Shared Python plot utilities (`set_pub_style()`, color palettes)
- `nextflow.config` – Pipeline configuration and parameter defaults
- `conf/modules.config` – Per-module `publishDir` and conda env settings
- `assets/` – Color mappings, census maps, mouse study/reference metadata TSVs
- `params/params.mm.json` – Mouse full-run parameter file
- `params/params.hs.json` – Human full-run parameter file

## Pipeline Parameter Definitions
The following are key parameters defined in `nextflow.config`:

- **results**: Path to the input results directory (required).
- **organism**: Organism name (`mus_musculus` or `homo_sapiens`).
- **census_version**: Census version string (e.g., `2024-07-01`).
- **outdir**: Output directory (auto-derived from `results` path; do not set manually).
- **ref_keys**: List of taxonomy levels (default: `[subclass, class, family, global]`).
- **mapping_file**: Path to census label mapping TSV.
- **color_mapping_file**: Path to color mapping TSV (default: `assets/color_mapping.tsv`).
- **subsample_ref**: Reference subsampling size used in upstream eval (default: `500`).
- **cutoff**: Confidence cutoff used in upstream eval (default: `0.0`).
- **method**: Default method for single-method plots (default: `scvi_rf`).
- **metadata_dir**: Path to standardized metadata directory for human runs (default: null).
- **skip_modeling**: Skip GLMM fitting and emmeans steps (default: `false`). Use when data has only one cutoff or subsample_ref value.
- **emmeans_cutoff**: Cutoff value at which emmeans marginal means are evaluated (default: `0`).
- **remove_outliers**: List of study IDs to exclude before aggregation (default: null).
- **max_cpus**: Maximum CPUs (default: 16).
- **max_memory**: Maximum memory (default: `128.GB`).
- **max_time**: Maximum runtime (default: `240.h`).

## Statistical Modeling

The pipeline uses **ordered beta regression** (via `glmmTMB(..., family = ordbeta())`) to model macro F1 score distributions. Standard beta regression assumes the response lies strictly in the open interval (0, 1), but F1 scores are bounded on [0, 1] and can include exact 0s and 1s (boundary inflation). Ordered beta regression handles this by modeling boundary values via ordinal cutpoints and the continuous interior via a beta distribution. The implementation follows Kubinec (2023), which demonstrates that this approach provides better calibration and fit for bounded continuous outcomes compared to alternatives such as zero/one-inflated beta models.

When `mixed=TRUE` (default for sample-level analysis), study is included as a random intercept, yielding a generalized linear mixed model (GLMM). Marginal means and contrasts are estimated via the `emmeans` package on the response scale.

## References
- Kubinec, R. (2023). Ordered Beta Regression: A Parsimonious, Well-Fitting Model for Continuous Data with Lower and Upper Bounds. *Political Analysis*, 31(4), 519–536. https://doi.org/10.1017/pan.2022.20
- [nf-core Nextflow DSL2 documentation](https://nf-co.re/developers/dsl2)
