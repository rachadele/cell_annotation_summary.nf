
# evaluation_summary.nf

## Project Overview
This repository implements a modular Nextflow pipeline (DSL2) for summarizing and evaluating cell annotation results across multiple datasets and methods. It integrates R and Python scripts for data aggregation, analysis, and visualization.

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
- **Results Structure:** Output is organized by organism, dataset, and method under dated directories (e.g., `2024-07-01/homo_sapiens/100/dataset_id/SCT/`).
- **Parameter Passing:** Use Nextflow channels to pass structured data between modules.
- **Aggregation:** Aggregation and summary files are written to `full_queries/aggregated_results/`.

## Integration Points
- **External Tools:**
  - Python: `scanpy`, `anndata`, `pandas`, `numpy`, etc.
  - R: Used for some statistical summaries and plots.
- **Environment:** Supports Conda, Docker, and Singularity via Nextflow profiles.

## Key Files & Directories
- `main.nf` – Pipeline entry point
- `workflows/evaluation_summary.nf` – Main workflow logic
- `modules/local/` – DSL2 modules for each step
- `bin/` – Python and R scripts for analysis/plotting
- `nextflow.config` – Pipeline configuration
- `full_queries/aggregated_results/` – Aggregated output summaries


## Pipeline Parameter Definitions
The following are key parameters defined in `nextflow.config`:

- **results**: Path to the input results directory (required).
- **organism**: Organism name (e.g., `mus_musculus`, `homo_sapiens`).
- **census_version**: Census version string (e.g., `2024-07-01`).
- **outdir**: Output directory (default: derived from results path).
- **ref_keys**: List of reference label types (default: `[subclass, class, family, global]`).
- **mapping_file**: Optional mapping file for labels.
- **color_mapping_file**: Optional color mapping file for plots.
- **subsample_ref**: Number of reference cells to subsample (default: 500).
- **cutoff**: Cutoff value for filtering (default: 0.0).
- **reference**: Reference dataset (default: `whole_cortex`).
- **method**: Annotation method (default: `scvi`).
- **remove_outliers**: Option to remove outliers (default: null).
- **max_cpus**: Maximum CPUs to use (default: 16).
- **max_memory**: Maximum memory (default: `128.GB`).
- **max_time**: Maximum runtime (default: `240.h`).

## References
- [nf-core Nextflow DSL2 documentation](https://nf-co.re/developers/dsl2)
