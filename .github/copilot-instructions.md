# Copilot Instructions for evaluation_summary.nf

## Project Overview
This repository implements a Nextflow pipeline for summarizing and evaluating cell annotation results across multiple datasets and methods. The pipeline is modular, using DSL2, and integrates R and Python scripts for data aggregation, analysis, and visualization.

## Architecture & Data Flow
- **Entry Point:** `main.nf` includes the main workflow from `workflows/evaluation_summary.nf`.
- **Modules:** Each analysis step is a DSL2 module in `modules/local/`, e.g., `add_params`, `aggregate_results`, `plot_cutoff`, etc.
- **Data Flow:** Results directories are scanned, parameters are added, results are aggregated, and various plots and summaries are generated. Channels are used to pass data between modules.
- **Config:** Pipeline parameters and profiles are set in `nextflow.config` and `conf/`.
- **Scripts:** Analysis and plotting scripts are in `bin/` (Python and R).

## Developer Workflows
- **Run Pipeline:**
  ```bash
  nextflow run main.nf --results <results_dir> --organism <organism> [...other params]
  ```
- **Cleanup:** Use `nf-cleanup` to remove intermediate files.
- **Profiles:** Use `-profile conda|docker|singularity` for environment management.
- **Add/Update Modules:** Place new modules in `modules/local/` and include in the workflow as needed.

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

## Example: Adding a New Analysis Step
1. Create a new module in `modules/local/<new_module>/main.nf`.
2. Add any supporting scripts to `bin/`.
3. Include the module in `workflows/evaluation_summary.nf`.
4. Pass data via channels as per existing patterns.

## References
- [nf-core Nextflow DSL2 documentation](https://nf-co.re/developers/dsl2)
- See `README.md` for any additional usage notes (currently minimal).
