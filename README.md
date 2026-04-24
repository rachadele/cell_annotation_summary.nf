# evaluation_summary.nf

Nextflow DSL2 pipeline for summarizing and evaluating single-cell RNA-seq cell-type annotation benchmarking results. Takes F1 outputs from an upstream evaluation pipeline (`nextflow_eval_pipeline`) and produces aggregated metrics, ordered beta GLMM fits, and publication-quality figures comparing two classifiers — **scVI** and **Seurat** — across GEO studies for mouse and human brain data.

A three-classifier variant (scVI RF + scVI kNN + Seurat) lives on `feature/integrate-scvi-knn`.

## Running the pipeline

```bash
# Full mouse run
nextflow run main.nf -params-file params.mm.json -profile conda

# Full human run
nextflow run main.nf -params-file params.hs.json -profile conda

# Resume after a failure
nextflow run main.nf -params-file params.mm.json -profile conda -resume
```

- `workDir` is hardcoded to `/cosmos/data/evaluation-summary-work` in `nextflow.config`.
- `outdir` is auto-derived from `params.results` by stripping the path up to `nextflow[_-]eval[_-]pipeline/(results/)?`. Override with `--outdir` if you want to write to a non-default location (e.g. for a test run).
- Default executor is SLURM (see `conf/base.config`). For local execution, add `-process.executor local`.

## Inputs

`params.results` should point to a directory matching:

```
{params.results}/{run_name}/{run_name}/
    params.yaml          # per-run params from the upstream pipeline
    refs/                # reference obs data
    scvi/                # per-cutoff F1 results
    seurat/
```

Only subdirectories named `scvi` and `seurat` are picked up. The main per-run artefact is the set of F1 TSVs under each method directory.

## Parameters

Defined in `nextflow.config` and per-organism params files (`params.mm.json`, `params.hs.json`).

| Parameter | Default | Purpose |
|---|---|---|
| `results` | *(required)* | Upstream evaluation-pipeline output directory |
| `outdir` | auto-derived | Where this pipeline writes its outputs |
| `organism` | `mus_musculus` | `mus_musculus` or `homo_sapiens` |
| `census_version` | `2024-07-01` | Census version (for output pathing) |
| `ref_keys` | `[subclass, class, family, global]` | Taxonomy levels to model |
| `remove_outliers` | `null` | List of study names to exclude before aggregation |
| `metadata_dir` | `null` | (Human) per-study sample metadata TSVs for filling missing sex/disease |
| `subsample_ref_emmeans` | `"500"` | `subsample_ref` level at which emmeans are evaluated |
| `emmeans_cutoff` | `0` | `cutoff` value at which emmeans are evaluated |
| `skip_modeling` | `false` | Skip GLMM + publication figures (useful when the data has only one cutoff/subsample level) |
| `mapping_file` | `null` | Cell-type taxonomy mapping TSV |
| `color_mapping_file` | `assets/color_mapping.tsv` | Cell-type colour palette |

## Architecture

```
main.nf
  └─ workflows/evaluation_summary.nf
       ADD_PARAMS              # attach params.yaml values to each F1 TSV
       AGGREGATE_RESULTS       # stream, concatenate, normalise, filter support>0
       JOIN_METADATA           # mouse-only: attach study/reference metadata
       MODEL_ASSAY_EFFECTS     # mouse-only: single-cell vs single-nuclei contrasts
       PLOT_ASSAY_EXPLORATION  # mouse-only
       PLOT_STUDY_VARIANCE
       PLOT_CUTOFF
       PLOT_F1_DISTRIBUTIONS
       PLOT_COMPTIME           # from upstream trace files
       MODEL_EVAL_AGGREGATED   # ordered beta GLMM in R; per-key emmeans
       PLOT_PUB_FIGURES        # multi-panel publication figure
       RANK_LABEL_PERFORMANCE
       PLOT_PARAM_HEATMAP
       PLOT_RANKING_SUMMARY
       PLOT_RANKING_RELIABILITY
       PLOT_CONFIG_PARETO
```

Each module in `modules/local/<name>/main.nf` calls a script via `python ${projectDir}/bin/script.py` or `Rscript ${projectDir}/bin/script.R`. Shared helpers: `bin/plot_utils.py` (Python plot style/palettes), `bin/model_functions.R` (GLMM fit, emmeans helpers).

## Statistical modelling

Ordered beta GLMM (`glmmTMB` with `family = ordbeta()`) on macro F1, with `study` as a random intercept. Marginal means via `emmeans`. Formulas, organism-specific exclusions (`sex` for mouse, `region_match` for human), and emmeans defaults are defined in `bin/model_performance_aggregated.R` and `bin/model_functions.R`. Outlier studies are removed via `params.remove_outliers` before aggregation.

## Output directory structure

Under `{outdir}/` (which is `{census_version}/{organism}/{N}/dataset_id/{normalization}/gap_{bool}/` by default):

| Directory | Contents |
|---|---|
| `params_added/` | per-run F1 TSVs with params.yaml values attached (`.tsv.gz`, cached via `storeDir`) |
| `aggregated_results/files/` | `sample_results.tsv.gz`, `label_results.tsv.gz`, `*_summary.tsv.gz`, plus `contamination.tsv` (mouse only) |
| `cutoff_plots/` | F1/precision/recall vs confidence cutoff curves |
| `f1_distributions/` | macro and per-label F1 histograms |
| `comptime_plots/` | computational time comparisons |
| `aggregated_models/{formula}/{key}/files/` | GLMM outputs — emmeans summaries, pairwise estimates, model coefficients |
| `post-hoc-figures/` | multi-panel publication figures |
| `celltype_rankings/` | per-cell-type rankings, Pareto configs, reliability plots |
| `parameter_heatmaps/` | heatmaps across parameter combinations |
| `assay_exploration/` | mouse only: single-cell vs single-nuclei comparison |
| `study_variance/` | cell-type F1 variation across studies |

## Key files

| File | Purpose |
|---|---|
| `main.nf` | Pipeline entry point |
| `workflows/evaluation_summary.nf` | Main workflow |
| `modules/local/` | Process definitions (one dir per module) |
| `bin/` | Analysis and plotting scripts |
| `bin/model_functions.R` | Shared R helpers for GLMM fits and emmeans |
| `bin/plot_utils.py` | Shared Python plot style and palettes |
| `bin/standalone/` | Retired pipeline scripts kept for ad-hoc use (grant summary, cell-type granularity) |
| `nextflow.config` | Top-level config and parameter defaults |
| `conf/base.config` | Executor and (unused) resource labels |
| `conf/modules.config` | Per-module `publishDir`/conda settings |
| `assets/` | Colour mappings, census taxonomy maps, mouse study/reference metadata |
| `params.mm.json`, `params.hs.json` | Organism params files |

## Environments

Conda only. Python modules use `scanpyenv`; R modules use `r4.3`. Docker and Singularity profiles exist in `nextflow.config` but are commented out.

Python scripts rely on `pandas`, `numpy`, `matplotlib`, `seaborn`, `scipy`, `scanpy`. R scripts rely on `glmmTMB`, `emmeans`, `broom.mixed`, `ggplot2`, `argparse`.

## Branches

- **`main`** — two-classifier pipeline (scVI + Seurat) with the 2026-04-24 cleanup (AGGREGATE_RESULTS streaming, configurable emmeans params, orphan-module removal, process-label stripping).
- **`feature/integrate-scvi-knn`** — three-classifier variant (scVI RF + scVI kNN + Seurat); results land in non-`_main_branch` directories. Not yet synced with `main`.
- **`feature/split-aggregate-results`, `feature/plotting-exploration`** — upstream development branches for the efficiency and plotting work already merged into `main`.

Baseline two-classifier figures and summaries are preserved under `2024-07-01/*_main_branch/` and `combined_orgs/main_branch/`. Do not overwrite these on the kNN branch — its params files point to `mus_musculus/` / `homo_sapiens/` eval paths instead.
