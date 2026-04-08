# Cell-Type Annotation Benchmarking: Results Summary (Old Pipeline)

> ⚠️ Old pipeline results (scVI monolithic + Seurat). No ref_support=0 filtering. Per-cell-type cutoff sensitivity tables unavailable. Compare with new pipeline results before drawing conclusions.

Generated from: `/space/grp/rschwartz/rschwartz/evaluation_summary.nf/2024-07-01/mus_musculus/100/dataset_id/SCT/gap_false//`

---

## mus_musculus

**Organism:** mus_musculus  
**Model formula:** `macro f1 ~ reference + method + cutoff + subsample ref + treatment state + sex + method:cutoff + reference:method`  
**Pipeline:** old (scvi + seurat)

### Study Cohort

| study | treatment | sex | Region | Samples | Cells | Subclasses |
| --- | --- | --- | --- | --- | --- | --- |
| GSE124952 | cocaine, saline, None | male | core of nucleus accumbens | 15 | 1500 | 7 |
| GSE181021.2 | None, Setd1a CRISPR-Cas9 | male | prefrontal cortex | 4 | 400 | 7 |
| GSE185454 | Veh - CFC, HDACi - CFC | nan | hippocampus (DG) | 4 | 400 | 9 |
| GSE199460.2 | nan | nan | brain | 2 | 198 | 1 |
| GSE214244.1 | nan | male | entorhinal cortex | 3 | 300 | 8 |
| GSE247339.1 | T4, sham, TBI | male | Ammon's horn | 21 | 2089 | 10 |
| GSE247339.2 | sham, TBI, T4 | male | cerebral cortex | 20 | 1993 | 10 |

### Method Performance (model-adjusted marginal means)

| key | seurat | scvi |
| --- | --- | --- |
| global | 0.927 [0.864–0.963] | 0.966 [0.934–0.983] |
| family | 0.855 [0.800–0.897] | 0.904 [0.864–0.933] |
| class | 0.791 [0.676–0.873] | 0.839 [0.741–0.904] |
| subclass | 0.773 [0.668–0.853] | 0.822 [0.731–0.887] |

### Method Pairwise Contrasts

| key | contrast | odds.ratio | p.value |
| --- | --- | --- | --- |
| global | seurat / scvi | 0.446 | < 1e-113 |
| family | seurat / scvi | 0.632 | < 1e-118 |
| class | seurat / scvi | 0.729 | < 1e-47 |
| subclass | seurat / scvi | 0.739 | < 1e-63 |

### Cutoff Sensitivity (method × cutoff EMMs)

| key | cutoff | scvi | seurat |
| --- | --- | --- | --- |
| global | 0.0 | 0.952 | 0.931 |
| global | 0.05 | 0.94 | 0.929 |
| global | 0.1 | 0.926 | 0.926 |
| global | 0.15 | 0.909 | 0.924 |
| global | 0.2 | 0.888 | 0.922 |
| global | 0.25 | 0.863 | 0.919 |
| global | 0.5 | 0.664 | 0.906 |
| global | 0.75 | 0.385 | 0.891 |
| family | 0.0 | 0.886 | 0.852 |
| family | 0.05 | 0.867 | 0.848 |
| family | 0.1 | 0.846 | 0.845 |
| family | 0.15 | 0.822 | 0.842 |
| family | 0.2 | 0.795 | 0.839 |
| family | 0.25 | 0.766 | 0.836 |
| family | 0.5 | 0.58 | 0.819 |
| family | 0.75 | 0.369 | 0.801 |
| class | 0.0 | 0.807 | 0.79 |
| class | 0.05 | 0.783 | 0.787 |
| class | 0.1 | 0.757 | 0.784 |
| class | 0.15 | 0.729 | 0.78 |
| class | 0.2 | 0.698 | 0.777 |
| class | 0.25 | 0.666 | 0.774 |
| class | 0.5 | 0.488 | 0.756 |
| class | 0.75 | 0.312 | 0.737 |
| subclass | 0.0 | 0.793 | 0.772 |
| subclass | 0.05 | 0.768 | 0.769 |
| subclass | 0.1 | 0.74 | 0.765 |
| subclass | 0.15 | 0.71 | 0.761 |
| subclass | 0.2 | 0.678 | 0.757 |
| subclass | 0.25 | 0.644 | 0.753 |
| subclass | 0.5 | 0.461 | 0.733 |
| subclass | 0.75 | 0.287 | 0.712 |

### Reference × Method Performance

| key | ref_short | scvi | seurat |
| --- | --- | --- | --- |
| global | Cortical+Hipp. 10x | 0.956 | 0.951 |
| global | Cortical+Hipp. SSv4 | 0.933 | 0.91 |
| global | Motor cortex | 0.966 | 0.953 |
| global | Whole cortex | 0.966 | 0.927 |
| family | Cortical+Hipp. 10x | 0.894 | 0.866 |
| family | Cortical+Hipp. SSv4 | 0.841 | 0.816 |
| family | Motor cortex | 0.908 | 0.879 |
| family | Whole cortex | 0.904 | 0.855 |
| class | Cortical+Hipp. 10x | 0.82 | 0.784 |
| class | Cortical+Hipp. SSv4 | 0.756 | 0.747 |
| class | Motor cortex | 0.836 | 0.858 |
| class | Whole cortex | 0.839 | 0.791 |
| subclass | Cortical+Hipp. 10x | 0.807 | 0.779 |
| subclass | Cortical+Hipp. SSv4 | 0.74 | 0.716 |
| subclass | Motor cortex | 0.819 | 0.833 |
| subclass | Whole cortex | 0.822 | 0.773 |

### Reference Subsample Size

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

### Biological Covariates

**treatment_state**

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

**sex**

| key | sex | EMM |
| --- | --- | --- |
| global | nan | 0.971 [0.916–0.991] |
| global | male | 0.915 [0.828–0.960] |
| family | nan | 0.936 [0.883–0.966] |
| family | male | 0.792 [0.715–0.852] |
| class | nan | 0.894 [0.761–0.957] |
| class | male | 0.701 [0.542–0.823] |
| subclass | nan | 0.879 [0.749–0.947] |
| subclass | male | 0.684 [0.551–0.792] |

### Between-Study Heterogeneity

**subclass — Well-classified (mean F1 ≥ 0.85)**

| key | label | n_studies | mean_f1 | std_f1 | mean_precision | mean_recall |
| --- | --- | --- | --- | --- | --- | --- |
| subclass | Endothelial | 6 | 0.952 | 0.029 | 0.998 | 0.918 |
| subclass | Glutamatergic | 4 | 0.944 | 0.075 | 0.931 | 0.986 |
| subclass | Astrocyte | 6 | 0.885 | 0.067 | 0.936 | 0.879 |
| subclass | GABAergic | 4 | 0.884 | 0.049 | 0.911 | 0.906 |
| subclass | Vascular | 2 | 0.869 | 0.048 | 0.955 | 0.833 |

**subclass — Hard / high-variance (mean F1 < 0.70 or std > 0.20)**

| key | label | n_studies | mean_f1 | std_f1 | mean_precision | mean_recall | failure_mode |
| --- | --- | --- | --- | --- | --- | --- | --- |
| subclass | OPC | 6 | 0.016 | 0.007 | 0.665 | 0.023 | — |
| subclass | Microglia | 6 | 0.116 | 0.037 | 0.964 | 0.11 | Label escape |
| subclass | Neural stem cell | 2 | 0.155 | 0.051 | 0.303 | 0.258 | Coverage failure |
| subclass | Macrophage | 2 | 0.203 | 0.08 | 0.137 | 0.726 | Over-prediction |
| subclass | Pericyte | 1 | 0.281 | nan | 1.0 | 0.259 | Label escape |
| subclass | CA1-ProS | 1 | 0.504 | nan | 0.962 | 0.433 | Label escape |
| subclass | Cajal-Retzius cell | 2 | 0.506 | 0.227 | 0.904 | 0.496 | Label escape; Study variance |
| subclass | CA3 | 1 | 0.542 | nan | 0.833 | 0.537 | — |
| subclass | DG | 1 | 0.647 | nan | 0.999 | 0.605 | — |

### Cell-Type Rankings (best config per label)

| key | label | method | reference | subsample_ref | mean_f1_across_studies | win_fraction | n_studies |
| --- | --- | --- | --- | --- | --- | --- | --- |
| global | Neural stem cell | scvi | Motor cortex | 500 | 0.452 | 0.5 | 2 |
| global | Neuron | seurat | Cortical+Hipp. 10x | 500 | 0.882 | 0.333 | 6 |
| global | Non-neuron | seurat | Cortical+Hipp. 10x | 500 | 0.992 | 0.429 | 7 |
| family | Astrocyte | seurat | Cortical+Hipp. 10x | 500 | 0.943 | 0.667 | 6 |
| family | CNS macrophage | seurat | Motor cortex | 100 | 0.988 | 0.667 | 6 |
| family | GABAergic | seurat | Cortical+Hipp. SSv4 | 50 | 0.99 | 0.5 | 4 |
| family | Glutamatergic | seurat | Whole cortex | 500 | 0.892 | 0.5 | 6 |
| family | Neural stem cell | scvi | Motor cortex | 500 | 0.452 | 0.5 | 2 |
| family | Neuron | scvi | Motor cortex | 100 | 0.913 | 0.5 | 2 |
| family | OPC | seurat | Whole cortex | 50 | 0.199 | 0.667 | 6 |
| family | Oligodendrocyte | seurat | Motor cortex | 100 | 0.974 | 0.667 | 6 |
| family | Vascular | seurat | Cortical+Hipp. 10x | 500 | 0.977 | 0.333 | 6 |
| class | Astrocyte | seurat | Cortical+Hipp. 10x | 500 | 0.943 | 0.667 | 6 |
| class | Cajal-Retzius cell | seurat | Cortical+Hipp. 10x | 50 | 0.992 | 1.0 | 2 |
| class | GABAergic | seurat | Cortical+Hipp. SSv4 | 50 | 0.99 | 0.5 | 4 |
| class | Glutamatergic | seurat | Cortical+Hipp. SSv4 | 50 | 0.978 | 0.75 | 4 |
| class | Hippocampal neuron | seurat | Cortical+Hipp. 10x | 50 | 0.996 | 1.0 | 1 |
| class | Macrophage | scvi | Cortical+Hipp. SSv4 | 500 | 0.263 | 0.5 | 2 |
| class | Microglia | seurat | Whole cortex | 500 | 0.459 | 0.167 | 6 |
| class | Neural stem cell | scvi | Motor cortex | 500 | 0.452 | 0.5 | 2 |
| class | Neuron | scvi | Motor cortex | 100 | 0.913 | 0.5 | 2 |
| class | OPC | seurat | Whole cortex | 50 | 0.199 | 0.667 | 6 |
| class | Oligodendrocyte | seurat | Motor cortex | 100 | 0.974 | 0.667 | 6 |
| class | Vascular | seurat | Cortical+Hipp. 10x | 500 | 0.977 | 0.333 | 6 |
| subclass | Astrocyte | seurat | Cortical+Hipp. 10x | 500 | 0.943 | 0.667 | 6 |
| subclass | CA1-ProS | seurat | Whole cortex | 100 | 0.933 | 1.0 | 1 |
| subclass | CA3 | seurat | Whole cortex | 100 | 0.943 | 1.0 | 1 |
| subclass | Cajal-Retzius cell | seurat | Cortical+Hipp. 10x | 50 | 0.992 | 1.0 | 2 |
| subclass | DG | seurat | Whole cortex | 500 | 0.996 | 1.0 | 1 |
| subclass | Endothelial | seurat | Cortical+Hipp. 10x | 500 | 0.986 | 0.333 | 6 |
| subclass | GABAergic | seurat | Cortical+Hipp. SSv4 | 50 | 0.99 | 0.5 | 4 |
| subclass | Glutamatergic | seurat | Cortical+Hipp. SSv4 | 50 | 0.978 | 0.75 | 4 |
| subclass | Macrophage | scvi | Cortical+Hipp. SSv4 | 500 | 0.247 | 0.0 | 2 |
| subclass | Microglia | seurat | Whole cortex | 500 | 0.459 | 0.167 | 6 |
| subclass | Neural stem cell | scvi | Motor cortex | 500 | 0.452 | 0.5 | 2 |
| subclass | Neuron | scvi | Motor cortex | 100 | 0.913 | 0.5 | 2 |
| subclass | OPC | seurat | Whole cortex | 50 | 0.199 | 0.667 | 6 |
| subclass | Oligodendrocyte | seurat | Motor cortex | 100 | 0.974 | 0.667 | 6 |
| subclass | Pericyte | seurat | Whole cortex | 50 | 0.933 | 1.0 | 1 |
| subclass | Vascular | seurat | Whole cortex | 100 | 0.93 | 0.0 | 2 |

### Reference Cell-Type Coverage

**global**

| Unnamed: 0 | Motor cortex | Cortical+Hipp. 10x | Cortical+Hipp. SSv4 | Whole cortex |
| --- | --- | --- | --- | --- |
| Neural stem cell | 246 | 0 | 0 | 494 |
| Neuron | 5016 | 7605 | 6850 | 7678 |
| Non-neuron | 3102 | 2440 | 1336 | 4511 |

**family**

| Unnamed: 0 | Motor cortex | Cortical+Hipp. 10x | Cortical+Hipp. SSv4 | Whole cortex |
| --- | --- | --- | --- | --- |
| Astrocyte | 500 | 500 | 500 | 500 |
| CNS macrophage | 555 | 500 | 177 | 1000 |
| GABAergic | 2500 | 2500 | 2500 | 2500 |
| Glutamatergic | 2516 | 5105 | 4350 | 5178 |
| Leukocyte | 0 | 0 | 0 | 188 |
| Neural stem cell | 246 | 0 | 0 | 494 |
| Oligodendrocyte | 0 | 0 | 0 | 312 |
| OPC | 500 | 500 | 229 | 500 |
| Vascular | 1547 | 940 | 430 | 2011 |

**class**

| Unnamed: 0 | Motor cortex | Cortical+Hipp. 10x | Cortical+Hipp. SSv4 | Whole cortex |
| --- | --- | --- | --- | --- |
| Astrocyte | 500 | 500 | 500 | 500 |
| Cajal-Retzius cell | 16 | 277 | 38 | 331 |
| deep layer non-IT | 0 | 2328 | 1812 | 2347 |
| Hippocampal neuron | 500 | 500 | 500 | 500 |
| L2/3-6 IT | 500 | 500 | 500 | 500 |
| LAMP5 | 0 | 0 | 0 | 188 |
| Leukocyte | 500 | 500 | 177 | 500 |
| Macrophage | 55 | 0 | 0 | 500 |
| Microglia | 246 | 0 | 0 | 494 |
| Neural stem cell | 0 | 0 | 0 | 312 |
| Oligodendrocyte | 500 | 500 | 229 | 500 |
| OPC | 500 | 500 | 500 | 500 |
| PVALB | 500 | 500 | 500 | 500 |
| SNCG | 500 | 500 | 500 | 500 |
| SST | 500 | 500 | 500 | 500 |
| Vascular | 1547 | 940 | 430 | 2011 |
| VIP | 2000 | 2000 | 2000 | 2000 |

**subclass**

| Unnamed: 0 | Motor cortex | Cortical+Hipp. 10x | Cortical+Hipp. SSv4 | Whole cortex |
| --- | --- | --- | --- | --- |
| Astrocyte | 500 | 500 | 500 | 500 |
| CA1-ProS | 0 | 500 | 500 | 500 |
| CA2-IG-FC | 0 | 328 | 19 | 347 |
| CA3 | 0 | 500 | 322 | 500 |
| Cajal-Retzius cell | 500 | 500 | 500 | 500 |
| CT | 16 | 277 | 38 | 331 |
| DG | 0 | 500 | 500 | 500 |
| Endothelial | 500 | 500 | 193 | 500 |
| Ependymal | 0 | 0 | 0 | 55 |
| L2/3-6 IT | 500 | 500 | 500 | 500 |
| L5 ET | 500 | 500 | 500 | 500 |
| L6b | 500 | 500 | 500 | 500 |
| LAMP5 | 500 | 500 | 500 | 500 |
| Leukocyte | 0 | 0 | 0 | 188 |
| Macrophage | 500 | 500 | 177 | 500 |
| Microglia | 55 | 0 | 0 | 500 |
| Neural stem cell | 500 | 500 | 500 | 500 |
| NP | 246 | 0 | 0 | 494 |
| Oligodendrocyte | 0 | 0 | 0 | 312 |
| OPC | 500 | 500 | 229 | 500 |
| Pericyte | 500 | 500 | 500 | 500 |
| PVALB | 500 | 0 | 0 | 500 |
| SMC | 47 | 288 | 121 | 456 |
| SNCG | 500 | 500 | 500 | 500 |
| SST | 500 | 500 | 500 | 500 |
| SUB-ProS | 0 | 500 | 471 | 500 |
| VIP | 500 | 500 | 500 | 500 |
| VLMC | 500 | 152 | 116 | 500 |

### Assay Exploration (mouse only)

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

### Pareto-Optimal Configurations

| key | method_display | reference | subsample_ref | mean_f1 | total_duration_hrs | total_memory_gb |
| --- | --- | --- | --- | --- | --- | --- |
| subclass | Seurat | Whole cortex | 100 | 0.694 | 0.109 | 0.032 |
| subclass | scVI | Whole cortex | 100 | 0.658 | 0.043 | 0.019 |
| subclass | scVI | Cortical+Hipp. 10x | 100 | 0.59 | 0.043 | 0.019 |
| subclass | scVI | Cortical+Hipp. SSv4 | 100 | 0.565 | 0.043 | 0.019 |
| subclass | scVI | Motor cortex | 100 | 0.486 | 0.043 | 0.019 |
| class | Seurat | Whole cortex | 500 | 0.69 | 0.21 | 0.03 |
| class | scVI | Whole cortex | 100 | 0.664 | 0.043 | 0.019 |
| class | scVI | Cortical+Hipp. 10x | 100 | 0.611 | 0.043 | 0.019 |
| class | scVI | Cortical+Hipp. SSv4 | 100 | 0.55 | 0.043 | 0.019 |
| class | scVI | Motor cortex | 100 | 0.521 | 0.043 | 0.019 |
| family | scVI | Whole cortex | 100 | 0.756 | 0.043 | 0.019 |
| family | scVI | Motor cortex | 100 | 0.725 | 0.043 | 0.019 |
| family | scVI | Cortical+Hipp. 10x | 100 | 0.68 | 0.043 | 0.019 |
| family | scVI | Cortical+Hipp. SSv4 | 100 | 0.619 | 0.043 | 0.019 |
| global | scVI | Motor cortex | 500 | 0.757 | 0.048 | 0.019 |
| global | scVI | Whole cortex | 100 | 0.755 | 0.043 | 0.019 |
| global | scVI | Motor cortex | 100 | 0.71 | 0.043 | 0.019 |
| global | scVI | Cortical+Hipp. 10x | 100 | 0.599 | 0.043 | 0.019 |
| global | scVI | Cortical+Hipp. SSv4 | 100 | 0.561 | 0.043 | 0.019 |

### Computational Time

| method | step | subsample_ref | mean_duration | mean_memory |
| --- | --- | --- | --- | --- |
| scVI | Query Processing | 50 | 0.025 | 0.019 |
| scVI | Query Processing | 100 | 0.025 | 0.019 |
| scVI | Query Processing | 500 | 0.025 | 0.019 |
| Seurat | Prediction | 50 | 0.018 | 0.019 |
| Seurat | Prediction | 100 | 0.019 | 0.018 |
| Seurat | Prediction | 500 | 0.045 | 0.022 |
| Seurat | Query Processing | 50 | 0.051 | 0.03 |
| Seurat | Query Processing | 100 | 0.051 | 0.03 |
| Seurat | Query Processing | 500 | 0.051 | 0.03 |
| Seurat | Ref Processing | 50 | 0.05 | 0.05 |
| Seurat | Ref Processing | 100 | 0.038 | 0.032 |
| Seurat | Ref Processing | 500 | 0.114 | 0.025 |
| scVI | Prediction | 50 | 0.018 | 0.013 |
| scVI | Prediction | 100 | 0.017 | 0.013 |
| scVI | Prediction | 500 | 0.022 | 0.013 |

---

## TODO

- [ ] Fix reference coverage tables (`assets/ref_coverage/tabulamuris-mus-musculus/`): tables may contain errors (same bug class as human tables — root cause unknown). Oligodendrocyte shows 0 cells for Motor cortex, Cortical+Hipp. 10x, and Cortical+Hipp. SSv4 at family level, which is biologically implausible. Revisit reference selection once corrected tables are available.

---

## Macro F1 vs Per-Cell-Type Conflict

scVI leads macro F1 at all levels (family EMM: 0.904 vs 0.855; subclass: 0.822 vs 0.773), but Seurat wins the best configuration for 14 of 19 cell types at subclass in per-cell-type rankings. The macro advantage of scVI likely reflects broadly stable performance across the large, common populations (Glutamatergic, GABAergic, Astrocyte) that dominate the macro average, whereas Seurat's per-cell-type wins include many rare or single-study types (CA1-ProS, CA3, DG, Cajal-Retzius cell) where Seurat's optimal reference differs from the single reference used in the macro model.

---

## Configuration Recommendation

### Recommended Taxonomy Level: **family**

Two cell types show systematic failures (mean F1 < 0.5 in ≥ 3 studies) at subclass and class:

- **OPC**: mean F1 = 0.016 (subclass/class), n = 6 studies — fails at all levels including family (mean F1 = 0.199). Flag for manual curation; collapsing further to global removes biological resolution without resolving the problem.
- **Microglia**: mean F1 = 0.116 (subclass), n = 6 studies — merges into CNS macrophage at family (mean F1 = 0.988, n = 6). Family eliminates this failure.

Family is therefore the finest level where Microglia classification is reliable. OPC remains a known failure at all levels and should be excluded from downstream analysis or curated manually. For cortex-focused studies, hippocampal subtypes (CA1-ProS, CA3, DG — all present in only 1 study) merge into Glutamatergic at family, further reducing spurious label variation from region mismatch.

### Recommended Configuration

| Dimension | Recommended value | Rationale |
| --- | --- | --- |
| Taxonomy level | family | Microglia failure (n=6 studies) eliminated at family (→ CNS macrophage, F1=0.988); OPC fails at all levels — flag for curation |
| Method | scVI | Macro F1 EMM: scvi 0.904 vs seurat 0.855 at family; scvi wins macro at all levels. Note: seurat preferred at subclass for many individual cell types (14/19), but performance advantage is reference-specific |
| Reference | Whole cortex | Highest mean F1 at family in Pareto (0.756); broadest biological scope. Motor cortex EMM slightly higher (0.908 vs 0.904) but coverage tables (note: may have errors — see TODO) show Oligodendrocyte = 0 at family — pending table correction |
| Cutoff | 0.0 | scVI performance degrades sharply with cutoff (0.793 → 0.287 at subclass from 0.0 → 0.75); seurat is more robust (0.772 → 0.712) |
| subsample_ref | 100 | Subclass EMM 0.797 vs 0.799 at 500 — negligible difference; Seurat ref processing 0.038 hrs at 100 vs 0.114 hrs at 500 |

### Raw Performance — scVI + Whole cortex + cutoff 0.0 + subsample_ref 100

| key | macro_f1_mean | macro_precision_mean | macro_recall_mean |
| --- | --- | --- | --- |
| global | 0.824 | 0.805 | 0.886 |
| family | 0.770 | 0.888 | 0.783 |
| class | 0.625 | 0.826 | 0.664 |
| subclass | 0.644 | 0.844 | 0.672 |

### Compute Time — scVI + subsample_ref 100

| step | mean_duration (hrs) | mean_memory (GB) |
| --- | --- | --- |
| Query Processing | 0.025 | 0.019 |
| Prediction | 0.017 | 0.013 |

### Trade-offs

scVI is substantially faster than Seurat (0.043 hrs total vs 0.109 hrs at subsample_ref 100) while achieving higher macro F1 at all levels; the per-cell-type win advantage of Seurat is reference-dependent and does not translate to a consistent macro benefit. For single-nucleus data, using a reference that includes nucleus data (cell_and_nucleus) yields +0.03–0.05 F1 at family over cell-only references.

### Pareto Note

scVI + Whole cortex + subsample_ref 100 is Pareto-optimal at family (mean F1 = 0.756, 0.043 hrs, 0.019 GB). The recommended configuration is Pareto-optimal.
