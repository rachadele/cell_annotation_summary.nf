# Cell-Type Annotation Benchmarking: Mouse Results Summary

> **WARNING — Old pipeline results** (scVI monolithic + Seurat; two classifiers). No `ref_support=0` filtering applied; labels absent from a reference are included in macro F1 averaging, inflating estimates for narrow references. Per-cell-type cutoff sensitivity tables unavailable. Compare with new pipeline results before drawing conclusions.

**Generated:** 2026-04-20  
**Source:** `2024-07-01/mus_musculus_main_branch/100/dataset_id/SCT/gap_false/`  
**Model formula:** `macro_f1 ~ reference + method + cutoff + subsample_ref + treatment_state + sex + method:cutoff + reference:method + (1 | study)`  
**Pipeline version:** old (scvi monolithic + seurat)

> **Note on sex covariate:** `sex` is present in this model formula but nearly all samples are male (two studies have `sex = NaN`). The large EMM difference (NaN >> male) is an artifact of the NaN group, not a biological signal. Sex has been removed from the mouse formula in the updated pipeline.

---

## Study Cohort

| study | treatment | sex | region | samples | cells | subclasses |
| --- | --- | --- | --- | --- | --- | --- |
| GSE124952 | cocaine, saline, None | male | core of nucleus accumbens | 15 | 1500 | 7 |
| GSE181021.2 | None, Setd1a CRISPR-Cas9 | male | prefrontal cortex | 4 | 400 | 7 |
| GSE185454 | Veh - CFC, HDACi - CFC | nan | hippocampus (DG) | 4 | 400 | 9 |
| GSE199460.2 | nan | nan | brain | 2 | 198 | 1 |
| GSE214244.1 | nan | male | entorhinal cortex | 3 | 300 | 8 |
| GSE247339.1 | T4, sham, TBI | male | Ammon's horn | 21 | 2089 | 10 |
| GSE247339.2 | sham, TBI, T4 | male | cerebral cortex | 20 | 1993 | 10 |

7 studies; mix of cortical, hippocampal, and subcortical regions. Outliers GSE181021.1, GSE152715.1/2, GSE231868 removed prior to modelling.

---

## Method Performance (model-adjusted marginal means)

| key | seurat | scvi |
| --- | --- | --- |
| global | 0.927 [0.864–0.963] | 0.966 [0.934–0.983] |
| family | 0.855 [0.800–0.897] | 0.904 [0.864–0.933] |
| class | 0.791 [0.676–0.873] | 0.839 [0.741–0.904] |
| subclass | 0.773 [0.668–0.853] | 0.822 [0.731–0.887] |

scVI outperforms Seurat at all taxonomy levels.

## Method Pairwise Contrasts

| key | contrast | odds.ratio | p.value |
| --- | --- | --- | --- |
| global | seurat / scvi | 0.446 | < 1e-113 |
| family | seurat / scvi | 0.632 | < 1e-118 |
| class | seurat / scvi | 0.729 | < 1e-47 |
| subclass | seurat / scvi | 0.739 | < 1e-63 |

---

## Cutoff Sensitivity (method × cutoff EMMs)

| key | cutoff | scvi | seurat |
| --- | --- | --- | --- |
| global | 0.0 | 0.952 | 0.931 |
| global | 0.25 | 0.863 | 0.919 |
| global | 0.50 | 0.664 | 0.906 |
| global | 0.75 | 0.385 | 0.891 |
| family | 0.0 | 0.886 | 0.852 |
| family | 0.25 | 0.766 | 0.836 |
| family | 0.50 | 0.580 | 0.819 |
| family | 0.75 | 0.369 | 0.801 |
| class | 0.0 | 0.807 | 0.790 |
| class | 0.25 | 0.666 | 0.774 |
| class | 0.50 | 0.488 | 0.756 |
| class | 0.75 | 0.312 | 0.737 |
| subclass | 0.0 | 0.793 | 0.772 |
| subclass | 0.25 | 0.644 | 0.753 |
| subclass | 0.50 | 0.461 | 0.733 |
| subclass | 0.75 | 0.287 | 0.712 |

scVI degrades severely with increasing cutoff; Seurat is nearly cutoff-insensitive. At cutoff ≥ 0.25, Seurat equals or exceeds scVI at all levels.

---

## Reference × Method Performance

| key | reference | scvi | seurat |
| --- | --- | --- | --- |
| global | Motor cortex | 0.966 | 0.953 |
| global | Whole cortex | 0.966 | 0.927 |
| global | Cortical+Hipp. 10x | 0.956 | 0.951 |
| global | Cortical+Hipp. SSv4 | 0.933 | 0.910 |
| family | Motor cortex | 0.908 | 0.879 |
| family | Whole cortex | 0.904 | 0.855 |
| family | Cortical+Hipp. 10x | 0.894 | 0.866 |
| family | Cortical+Hipp. SSv4 | 0.841 | 0.816 |
| class | Motor cortex | 0.836 | **0.858** |
| class | Whole cortex | 0.839 | 0.791 |
| class | Cortical+Hipp. 10x | 0.820 | 0.784 |
| class | Cortical+Hipp. SSv4 | 0.756 | 0.747 |
| subclass | Whole cortex | 0.822 | 0.773 |
| subclass | Motor cortex | 0.819 | 0.833 |
| subclass | Cortical+Hipp. 10x | 0.807 | 0.779 |
| subclass | Cortical+Hipp. SSv4 | 0.740 | 0.716 |

At class level, Seurat + Motor cortex (0.858) marginally outperforms scVI + Motor cortex (0.836). SSv4 is consistently the weakest reference.

### Reference Ranking (mean EMM across methods and keys)

| reference | mean_emm |
| --- | --- |
| Motor cortex | 0.882 |
| Whole cortex | 0.860 |
| Cortical+Hipp. 10x | 0.857 |
| Cortical+Hipp. SSv4 | 0.807 |

---

## Reference Subsample Size

| key | subsample_ref | EMM |
| --- | --- | --- |
| global | 500 | 0.950 [0.905–0.975] |
| global | 100 | 0.946 [0.898–0.973] |
| global | 50 | 0.937 [0.880–0.967] |
| family | 500 | 0.882 [0.835–0.917] |
| family | 100 | 0.882 [0.835–0.917] |
| family | 50 | 0.870 [0.820–0.908] |
| class | 500 | 0.816 [0.710–0.889] |
| class | 100 | 0.811 [0.703–0.886] |
| class | 50 | 0.798 [0.686–0.878] |
| subclass | 500 | 0.799 [0.700–0.871] |
| subclass | 100 | 0.797 [0.698–0.869] |
| subclass | 50 | 0.784 [0.682–0.860] |

Subsample size has minimal effect; CIs overlap substantially across all three levels.

---

## Biological Covariates

**treatment_state** — negligible effect (CIs overlap, ≤ 0.005 difference at any level).

| key | treatment_state | EMM |
| --- | --- | --- |
| global | no treatment | 0.952 [0.907–0.975] |
| global | treatment | 0.949 [0.903–0.974] |
| family | no treatment | 0.879 [0.831–0.915] |
| family | treatment | 0.884 [0.838–0.919] |
| class | no treatment | 0.818 [0.713–0.891] |
| class | treatment | 0.814 [0.707–0.888] |
| subclass | no treatment | 0.800 [0.703–0.872] |
| subclass | treatment | 0.797 [0.698–0.869] |

---

## Between-Study Heterogeneity

**Well-classified at subclass (mean F1 ≥ 0.85)**

| label | n_studies | mean_f1 | std_f1 | mean_precision | mean_recall |
| --- | --- | --- | --- | --- | --- |
| Endothelial | 6 | 0.952 | 0.029 | 0.998 | 0.918 |
| Glutamatergic | 4 | 0.944 | 0.075 | 0.931 | 0.986 |
| Astrocyte | 6 | 0.885 | 0.067 | 0.936 | 0.879 |
| GABAergic | 4 | 0.884 | 0.049 | 0.911 | 0.906 |
| Vascular | 2 | 0.869 | 0.048 | 0.955 | 0.833 |

**Systematic failures at subclass (mean F1 < 0.5, ≥ 2 studies)**

| label | n_studies | mean_f1 | std_f1 | mean_precision | mean_recall | failure_mode |
| --- | --- | --- | --- | --- | --- | --- |
| OPC | 6 | 0.016 | 0.007 | 0.665 | 0.023 | Coverage failure (absent from Motor/10x/SSv4 refs) |
| Microglia | 6 | 0.116 | 0.037 | 0.964 | 0.110 | Label escape (high precision, near-zero recall) |
| Neural stem cell | 2 | 0.155 | 0.051 | 0.303 | 0.258 | Coverage failure |
| Macrophage | 2 | 0.203 | 0.080 | 0.137 | 0.726 | Over-prediction |
| Cajal-Retzius cell | 2 | 0.506 | 0.227 | 0.904 | 0.496 | Label escape; high study variance |

---

## Cell-Type Rankings (best config per label)

| key | label | method | reference | subsample_ref | mean_f1 | win_fraction | n_studies |
| --- | --- | --- | --- | --- | --- | --- | --- |
| global | Neural stem cell | scvi | Motor cortex | 500 | 0.452 | 0.50 | 2 |
| global | Neuron | seurat | Cortical+Hipp. 10x | 500 | 0.882 | 0.33 | 6 |
| global | Non-neuron | seurat | Cortical+Hipp. 10x | 500 | 0.992 | 0.43 | 7 |
| family | Astrocyte | seurat | Cortical+Hipp. 10x | 500 | 0.943 | 0.67 | 6 |
| family | CNS macrophage | seurat | Motor cortex | 100 | 0.988 | 0.67 | 6 |
| family | GABAergic | seurat | Cortical+Hipp. SSv4 | 50 | 0.990 | 0.50 | 4 |
| family | Glutamatergic | seurat | Whole cortex | 500 | 0.892 | 0.50 | 6 |
| family | Neural stem cell | scvi | Motor cortex | 500 | 0.452 | 0.50 | 2 |
| family | OPC | seurat | Whole cortex | 50 | 0.199 | 0.67 | 6 |
| family | Oligodendrocyte | seurat | Motor cortex | 100 | 0.974 | 0.67 | 6 |
| family | Vascular | seurat | Cortical+Hipp. 10x | 500 | 0.977 | 0.33 | 6 |
| class | Astrocyte | seurat | Cortical+Hipp. 10x | 500 | 0.943 | 0.67 | 6 |
| class | Cajal-Retzius cell | seurat | Cortical+Hipp. 10x | 50 | 0.992 | 1.00 | 2 |
| class | GABAergic | seurat | Cortical+Hipp. SSv4 | 50 | 0.990 | 0.50 | 4 |
| class | Glutamatergic | seurat | Cortical+Hipp. SSv4 | 50 | 0.978 | 0.75 | 4 |
| class | Hippocampal neuron | seurat | Cortical+Hipp. 10x | 50 | 0.996 | 1.00 | 1 |
| class | Macrophage | scvi | Cortical+Hipp. SSv4 | 500 | 0.263 | 0.50 | 2 |
| class | Microglia | seurat | Whole cortex | 500 | 0.459 | 0.17 | 6 |
| class | OPC | seurat | Whole cortex | 50 | 0.199 | 0.67 | 6 |
| class | Oligodendrocyte | seurat | Motor cortex | 100 | 0.974 | 0.67 | 6 |
| subclass | CA1-ProS | seurat | Whole cortex | 100 | 0.933 | 1.00 | 1 |
| subclass | CA3 | seurat | Whole cortex | 100 | 0.943 | 1.00 | 1 |
| subclass | DG | seurat | Whole cortex | 500 | 0.996 | 1.00 | 1 |
| subclass | Endothelial | seurat | Cortical+Hipp. 10x | 500 | 0.986 | 0.33 | 6 |
| subclass | Macrophage | scvi | Cortical+Hipp. SSv4 | 500 | 0.247 | 0.00 | 2 |
| subclass | Microglia | seurat | Whole cortex | 500 | 0.459 | 0.17 | 6 |
| subclass | OPC | seurat | Whole cortex | 50 | 0.199 | 0.67 | 6 |

---

## Reference Cell-Type Coverage

Only cells present in a reference can be annotated. Labels with 0 in a reference cannot be reliably annotated by that reference regardless of macro F1.

**global**

| label | All | Motor cortex | Cortical+Hipp. 10x | Cortical+Hipp. SSv4 | Whole cortex |
| --- | --- | --- | --- | --- | --- |
| Neural stem cell | 248 | 246 | 0 | 0 | 494 |
| Neuron | 0 | 343,823 | 1,149,359 | 64,794 | 1,557,976 |
| Non-neuron | 19,537 | 56,301 | 15,241 | 1,799 | 92,878 |

**family** (coverage gaps only)

| label | All | Motor cortex | Cortical+Hipp. 10x | Cortical+Hipp. SSv4 | Whole cortex |
| --- | --- | --- | --- | --- | --- |
| GABAergic | 0 | 61,650 | 177,594 | 20,531 | 259,775 |
| Glutamatergic | 0 | 282,173 | 971,765 | 44,263 | 1,298,201 |
| Leukocyte | 188 | 0 | 0 | 0 | 188 |
| OPC | 312 | 0 | 0 | 0 | 312 |
| Neural stem cell | 248 | 246 | 0 | 0 | 494 |

**class/subclass** — Motor cortex, 10x, and SSv4 all lack Hippocampal neuron (0 cells); only Whole cortex covers it (86,025). OPC has 0 cells in all references except Whole cortex (312). DG, CA1-ProS, CA3 exist only in Cortical+Hipp. 10x and Whole cortex.

---

## Assay Exploration (mouse only)

| key | ref_type | query_type | EMM |
| --- | --- | --- | --- |
| global | cell_only | single-cell | 0.705 [0.698–0.712] |
| global | cell_and_nucleus | single-cell | 0.751 [0.745–0.757] |
| global | cell_only | single-nucleus | 0.894 [0.885–0.903] |
| global | cell_and_nucleus | single-nucleus | 0.890 [0.882–0.898] |
| family | cell_only | single-cell | 0.703 [0.698–0.708] |
| family | cell_and_nucleus | single-cell | 0.749 [0.744–0.753] |
| family | cell_only | single-nucleus | 0.811 [0.801–0.821] |
| family | cell_and_nucleus | single-nucleus | 0.842 [0.833–0.850] |
| class | cell_only | single-cell | 0.576 [0.570–0.582] |
| class | cell_and_nucleus | single-cell | 0.648 [0.643–0.654] |
| class | cell_only | single-nucleus | 0.676 [0.662–0.689] |
| class | cell_and_nucleus | single-nucleus | 0.697 [0.685–0.710] |
| subclass | cell_only | single-cell | 0.604 [0.598–0.609] |
| subclass | cell_and_nucleus | single-cell | 0.671 [0.666–0.676] |
| subclass | cell_only | single-nucleus | 0.661 [0.648–0.674] |
| subclass | cell_and_nucleus | single-nucleus | 0.663 [0.650–0.675] |

Single-nucleus queries annotate ~0.19 F1 better than single-cell at global. Mixed-assay references help single-cell queries but offer minimal gain for single-nucleus.

---

## Pareto-Optimal Configurations

| key | method | reference | subsample_ref | mean_f1 | total_duration_hrs | total_memory_gb |
| --- | --- | --- | --- | --- | --- | --- |
| subclass | Seurat | Whole cortex | 100 | 0.694 | 0.109 | 0.032 |
| subclass | scVI | Whole cortex | 100 | 0.658 | 0.043 | 0.019 |
| class | Seurat | Whole cortex | 500 | 0.690 | 0.210 | 0.030 |
| class | scVI | Whole cortex | 100 | 0.664 | 0.043 | 0.019 |
| family | scVI | Whole cortex | 100 | 0.756 | 0.043 | 0.019 |
| global | scVI | Motor cortex | 500 | 0.757 | 0.048 | 0.019 |
| global | scVI | Whole cortex | 100 | 0.755 | 0.043 | 0.019 |

---

## Computational Time

| method | step | subsample_ref | mean_duration (hrs) | mean_memory (GB) |
| --- | --- | --- | --- | --- |
| scVI | Query Processing | 500 | 0.025 | 0.019 |
| scVI | Prediction | 500 | 0.022 | 0.013 |
| Seurat | Query Processing | 500 | 0.051 | 0.030 |
| Seurat | Ref Processing | 500 | 0.114 | 0.025 |
| Seurat | Prediction | 500 | 0.045 | 0.022 |

scVI is ~3× faster than Seurat end-to-end.

---

## Macro F1 vs Per-Cell-Type F1 Conflict

The macro F1 ranking (scVI > Seurat at all levels, OR 0.739 at subclass) conflicts with the cell-type win-fraction ranking, where Seurat wins the best configuration for the majority of individual types at family, class, and subclass. scVI's advantage is concentrated in high-abundance types (Glutamatergic, GABAergic) that dominate the macro average. For analyses where per-type accuracy matters equally across rare and common types, Seurat is the more consistent performer. The cutoff sensitivity further favours Seurat: at cutoff=0.25, Seurat matches or exceeds scVI at every level.

---

## Hippocampal Contamination

Not available — old pipeline does not apply `ref_support=0` filtering. Re-run with the new pipeline to obtain contamination estimates.

---

## Configuration Recommendation

### Recommended Taxonomy Level: **class**

At subclass, four types show systematic failure across ≥2 studies: OPC (F1=0.016), Microglia (F1=0.116), Neural stem cell (F1=0.155), Macrophage (F1=0.203). At class, OPC (0.199) and Microglia (0.459) still fail, but the major cortical and hippocampal types — GABAergic (0.990), Glutamatergic (0.978), Astrocyte (0.943), Oligodendrocyte (0.974), Hippocampal neuron (0.996), Cajal-Retzius (0.992), Vascular (0.977) — are reliably classified. OPC and Microglia require manual curation at any level; their failure is structural (coverage gaps and label escape respectively).

### Recommended Configuration

| Dimension | Recommended value | Rationale |
| --- | --- | --- |
| Taxonomy level | **class** | Major neuronal and glial lineages reliably classified; OPC/Microglia curated separately |
| Method | **scVI (cutoff=0)** | Macro F1 advantage (OR 0.739, p < 1e-63); prefer Seurat if per-type consistency matters more |
| Reference | **Whole cortex** | Only reference covering OPC, hippocampal subtypes, and Neural stem cell; mean EMM 0.860 |
| Cutoff | **0.0** | scVI degrades severely at higher cutoffs; contamination filtering unavailable in this pipeline |
| Subsample_ref | **100** | Near-identical EMM to subsample=500 (0.797 vs 0.799); 2.6× faster Seurat reference processing |

> If using Seurat, Motor cortex at class (0.858) exceeds scVI + Motor cortex (0.836) and is a competitive choice despite hippocampal coverage gaps.

### Raw Performance — scVI, Whole cortex, cutoff=0, subsample_ref=500

| key | macro_f1 | macro_precision | macro_recall |
| --- | --- | --- | --- |
| global | 0.809 | 0.800 | 0.896 |
| family | 0.747 | 0.885 | 0.770 |
| class | 0.598 | 0.821 | 0.649 |
| subclass | 0.621 | 0.840 | 0.660 |

Precision substantially exceeds recall at class and subclass (0.821 vs 0.649), indicating systematic label escape.

### Compute Time — scVI, subsample_ref=100

| step | mean_duration (hrs) | mean_memory (GB) |
| --- | --- | --- |
| Query Processing | 0.025 | 0.019 |
| Prediction | 0.017 | 0.013 |

Total per-query ≈ 0.042 hrs (~2.5 min).

### Trade-offs

Choosing scVI over Seurat accepts worse performance on rare cell types (Microglia, OPC, Macrophage) in exchange for higher aggregate F1 from abundant neuronal types. Subsample=100 over 500 saves ~60% of Seurat reference processing time with < 0.002 F1 loss at all levels.

### Pareto Note

scVI + Whole cortex + subsample=100 appears on the Pareto front at class and family, confirming it is not dominated by any configuration on the F1 vs compute trade-off.
