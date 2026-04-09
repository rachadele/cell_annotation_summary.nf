# Cell-Type Annotation Benchmarking: Results Summary


Generated from: `/space/grp/rschwartz/rschwartz/evaluation_summary.nf/2024-07-01/mus_musculus/100/dataset_id/SCT/gap_false/`

---

## mus_musculus

**Organism:** mus_musculus  
**Model formula:** `macro f1 ~ reference + method + cutoff + subsample ref + treatment state + sex + method:cutoff + reference:method`  
**Pipeline:** new (scvi_rf + scvi_knn + seurat)

### Study Cohort

| study | treatment | sex | Region | Samples | Cells | Subclasses |
| --- | --- | --- | --- | --- | --- | --- |
| GSE124952 | saline, None, cocaine | male | prefrontal cortex | 15 | 1500 | 5 |
| GSE181021.2 | Setd1a CRISPR-Cas9, None | male | prefrontal cortex | 4 | 400 | 5 |
| GSE185454 | HDACi - CFC, Veh - CFC | nan | hippocampus (DG) | 4 | 400 | 7 |
| GSE199460.2 | nan | nan | brain | 2 | 198 | 1 |
| GSE214244.1 | nan | male | entorhinal cortex | 3 | 300 | 6 |
| GSE247339.1 | sham, TBI, T4 | male | Ammon's horn | 21 | 2089 | 8 |
| GSE247339.2 | T4, TBI, sham | male | cerebral cortex | 20 | 1993 | 8 |

### Method Performance (model-adjusted marginal means)

| key | seurat | scvi_rf | scvi_knn |
| --- | --- | --- | --- |
| global | 0.940 [0.900–0.965] | 0.958 [0.929–0.976] | 0.948 [0.913–0.970] |
| family | 0.906 [0.882–0.925] | 0.923 [0.903–0.940] | 0.918 [0.896–0.935] |
| class | 0.860 [0.801–0.904] | 0.859 [0.799–0.904] | 0.850 [0.786–0.896] |
| subclass | 0.834 [0.755–0.891] | 0.829 [0.749–0.888] | 0.821 [0.739–0.882] |

### Method Pairwise Contrasts

| key | contrast | odds.ratio | p.value |
| --- | --- | --- | --- |
| global | seurat / scvi_rf | 0.687 | < 1e-300 |
| global | seurat / scvi_knn | 0.855 | < 1e-10 |
| global | scvi_rf / scvi_knn | 1.245 | < 1e-13 |
| family | seurat / scvi_rf | 0.797 | < 1e-300 |
| family | seurat / scvi_knn | 0.86 | < 1e-13 |
| family | scvi_rf / scvi_knn | 1.079 | < 1e-4 |
| class | seurat / scvi_rf | 1.008 | 0.896 |
| class | seurat / scvi_knn | 1.092 | < 1e-5 |
| class | scvi_rf / scvi_knn | 1.083 | < 1e-4 |
| subclass | seurat / scvi_rf | 1.035 | 0.055 |
| subclass | seurat / scvi_knn | 1.093 | < 1e-8 |
| subclass | scvi_rf / scvi_knn | 1.056 | < 1e-3 |

### Cutoff Sensitivity (method × cutoff EMMs)

| key | cutoff | scvi_knn | scvi_rf | seurat |
| --- | --- | --- | --- | --- |
| global | 0.0 | 0.948 | 0.958 | 0.94 |
| global | 0.05 | 0.945 | 0.947 | 0.938 |
| global | 0.1 | 0.94 | 0.934 | 0.936 |
| global | 0.15 | 0.936 | 0.918 | 0.933 |
| global | 0.2 | 0.931 | 0.898 | 0.931 |
| global | 0.25 | 0.926 | 0.873 | 0.928 |
| global | 0.5 | 0.896 | 0.675 | 0.914 |
| global | 0.75 | 0.855 | 0.384 | 0.898 |
| family | 0.0 | 0.918 | 0.923 | 0.906 |
| family | 0.05 | 0.915 | 0.909 | 0.904 |
| family | 0.1 | 0.912 | 0.892 | 0.901 |
| family | 0.15 | 0.909 | 0.872 | 0.899 |
| family | 0.2 | 0.906 | 0.849 | 0.897 |
| family | 0.25 | 0.903 | 0.823 | 0.895 |
| family | 0.5 | 0.886 | 0.641 | 0.882 |
| family | 0.75 | 0.867 | 0.408 | 0.868 |
| class | 0.0 | 0.85 | 0.859 | 0.86 |
| class | 0.05 | 0.847 | 0.841 | 0.859 |
| class | 0.1 | 0.845 | 0.82 | 0.858 |
| class | 0.15 | 0.842 | 0.797 | 0.857 |
| class | 0.2 | 0.84 | 0.772 | 0.855 |
| class | 0.25 | 0.837 | 0.745 | 0.854 |
| class | 0.5 | 0.823 | 0.582 | 0.848 |
| class | 0.75 | 0.809 | 0.399 | 0.841 |
| subclass | 0.0 | 0.821 | 0.829 | 0.834 |
| subclass | 0.05 | 0.819 | 0.808 | 0.833 |
| subclass | 0.1 | 0.816 | 0.785 | 0.831 |
| subclass | 0.15 | 0.813 | 0.76 | 0.83 |
| subclass | 0.2 | 0.81 | 0.733 | 0.828 |
| subclass | 0.25 | 0.808 | 0.704 | 0.827 |
| subclass | 0.5 | 0.793 | 0.537 | 0.82 |
| subclass | 0.75 | 0.777 | 0.362 | 0.812 |

### Reference × Method Performance

| key | ref_short | scvi_knn | scvi_rf | seurat |
| --- | --- | --- | --- | --- |
| global | Cortical+Hipp. 10x | 0.953 | 0.965 | 0.97 |
| global | Cortical+Hipp. SSv4 | 0.937 | 0.941 | 0.919 |
| global | Motor cortex | 0.951 | 0.961 | 0.942 |
| global | Whole cortex | 0.951 | 0.961 | 0.912 |
| family | Cortical+Hipp. 10x | 0.94 | 0.95 | 0.946 |
| family | Cortical+Hipp. SSv4 | 0.911 | 0.894 | 0.892 |
| family | Motor cortex | 0.934 | 0.942 | 0.924 |
| family | Whole cortex | 0.875 | 0.891 | 0.828 |
| class | Cortical+Hipp. 10x | 0.914 | 0.925 | 0.917 |
| class | Cortical+Hipp. SSv4 | 0.88 | 0.844 | 0.867 |
| class | Motor cortex | 0.807 | 0.849 | 0.892 |
| class | Whole cortex | 0.758 | 0.789 | 0.708 |
| subclass | Cortical+Hipp. 10x | 0.889 | 0.901 | 0.904 |
| subclass | Cortical+Hipp. SSv4 | 0.851 | 0.81 | 0.829 |
| subclass | Motor cortex | 0.781 | 0.821 | 0.864 |
| subclass | Whole cortex | 0.733 | 0.759 | 0.689 |

### Reference Subsample Size

| key | subsample_ref | EMM |
| --- | --- | --- |
| global | 500 | 0.954 [0.922–0.973] |
| global | 100 | 0.951 [0.918–0.971] |
| global | 50 | 0.943 [0.904–0.966] |
| family | 500 | 0.918 [0.897–0.936] |
| family | 100 | 0.919 [0.898–0.936] |
| family | 50 | 0.911 [0.888–0.929] |
| class | 500 | 0.860 [0.801–0.904] |
| class | 100 | 0.858 [0.797–0.902] |
| class | 50 | 0.851 [0.789–0.898] |
| subclass | 500 | 0.831 [0.751–0.889] |
| subclass | 100 | 0.831 [0.751–0.889] |
| subclass | 50 | 0.823 [0.741–0.884] |

### Biological Covariates

**treatment_state**

| key | treatment_state | EMM |
| --- | --- | --- |
| global | no treatment | 0.952 [0.919–0.972] |
| global | treatment | 0.947 [0.910–0.969] |
| family | no treatment | 0.915 [0.893–0.933] |
| family | treatment | 0.917 [0.895–0.935] |
| class | no treatment | 0.849 [0.786–0.896] |
| class | treatment | 0.863 [0.805–0.906] |
| subclass | no treatment | 0.820 [0.736–0.881] |
| subclass | treatment | 0.837 [0.759–0.893] |

**sex**

| key | sex | EMM |
| --- | --- | --- |
| global | nan | 0.969 [0.927–0.987] |
| global | male | 0.918 [0.855–0.955] |
| family | nan | 0.947 [0.921–0.965] |
| family | male | 0.870 [0.835–0.897] |
| class | nan | 0.910 [0.834–0.953] |
| class | male | 0.779 [0.683–0.852] |
| subclass | nan | 0.883 [0.768–0.945] |
| subclass | male | 0.756 [0.648–0.839] |

### Between-Study Heterogeneity

**subclass — Well-classified (mean F1 ≥ 0.85)**

| key | label | n_studies | mean_f1 | std_f1 | mean_precision | mean_recall |
| --- | --- | --- | --- | --- | --- | --- |
| subclass | Endothelial | 6 | 0.953 | 0.025 | 0.999 | 0.918 |
| subclass | DG | 1 | 0.894 | nan | 0.998 | 0.846 |
| subclass | Astrocyte | 6 | 0.893 | 0.068 | 0.943 | 0.884 |

**subclass — Hard / high-variance (mean F1 < 0.70 or std > 0.20)**

| key | label | n_studies | mean_f1 | std_f1 | mean_precision | mean_recall | failure_mode |
| --- | --- | --- | --- | --- | --- | --- | --- |
| subclass | OPC | 6 | 0.131 | 0.109 | 0.633 | 0.137 | — |
| subclass | Microglia | 6 | 0.154 | 0.049 | 0.964 | 0.146 | Label escape |
| subclass | Macrophage | 2 | 0.205 | 0.079 | 0.134 | 0.734 | Over-prediction |
| subclass | Neural stem cell | 2 | 0.372 | 0.114 | 0.371 | 0.545 | — |
| subclass | Cajal-Retzius cell | 2 | 0.57 | 0.254 | 0.935 | 0.559 | Study variance |
| subclass | Pericyte | 1 | 0.637 | nan | 1.0 | 0.593 | — |
| subclass | CA1-ProS | 1 | 0.679 | nan | 0.961 | 0.571 | — |
| subclass | CA3 | 1 | 0.686 | nan | 0.802 | 0.663 | — |

### Cell-Type Rankings (best config per label)

| key | label | method | reference | subsample_ref | mean_f1_across_studies | win_fraction | n_studies |
| --- | --- | --- | --- | --- | --- | --- | --- |
| global | Neural stem cell | scvi_knn | Motor cortex | 500 | 0.514 | 0.5 | 2 |
| global | Neuron | seurat | Cortical+Hipp. 10x | 500 | 0.882 | 0.333 | 6 |
| global | Non-neuron | seurat | Cortical+Hipp. 10x | 500 | 0.992 | 0.429 | 7 |
| family | Astrocyte | seurat | Cortical+Hipp. 10x | 500 | 0.943 | 0.667 | 6 |
| family | CNS macrophage | scvi_knn | Cortical+Hipp. 10x | 100 | 0.983 | 0.5 | 6 |
| family | GABAergic | seurat | Cortical+Hipp. SSv4 | 50 | 0.99 | 0.5 | 4 |
| family | Glutamatergic | seurat | Whole cortex | 500 | 0.892 | 0.5 | 6 |
| family | Neural stem cell | scvi_knn | Motor cortex | 500 | 0.514 | 0.5 | 2 |
| family | OPC | scvi_knn | Whole cortex | 100 | 0.277 | 0.333 | 6 |
| family | Oligodendrocyte | seurat | Motor cortex | 100 | 0.974 | 0.5 | 6 |
| family | Vascular | seurat | Cortical+Hipp. 10x | 500 | 0.977 | 0.333 | 6 |
| class | Astrocyte | seurat | Cortical+Hipp. 10x | 500 | 0.943 | 0.667 | 6 |
| class | Cajal-Retzius cell | seurat | Cortical+Hipp. 10x | 50 | 0.992 | 1.0 | 2 |
| class | Hippocampal neuron | scvi_knn | Cortical+Hipp. 10x | 100 | 0.996 | 1.0 | 1 |
| class | Macrophage | scvi_rf | Cortical+Hipp. SSv4 | 500 | 0.263 | 0.5 | 2 |
| class | Microglia | seurat | Whole cortex | 500 | 0.459 | 0.167 | 6 |
| class | Neural stem cell | scvi_knn | Motor cortex | 500 | 0.514 | 0.5 | 2 |
| class | OPC | scvi_knn | Whole cortex | 100 | 0.277 | 0.333 | 6 |
| class | Oligodendrocyte | seurat | Motor cortex | 100 | 0.974 | 0.5 | 6 |
| class | Vascular | seurat | Cortical+Hipp. 10x | 500 | 0.977 | 0.333 | 6 |
| subclass | Astrocyte | seurat | Cortical+Hipp. 10x | 500 | 0.943 | 0.667 | 6 |
| subclass | CA1-ProS | seurat | Whole cortex | 100 | 0.933 | 1.0 | 1 |
| subclass | CA3 | seurat | Whole cortex | 100 | 0.943 | 1.0 | 1 |
| subclass | Cajal-Retzius cell | seurat | Cortical+Hipp. 10x | 50 | 0.992 | 1.0 | 2 |
| subclass | DG | seurat | Whole cortex | 500 | 0.996 | 1.0 | 1 |
| subclass | Endothelial | seurat | Cortical+Hipp. 10x | 500 | 0.986 | 0.333 | 6 |
| subclass | Macrophage | scvi_rf | Cortical+Hipp. SSv4 | 100 | 0.247 | 0.0 | 2 |
| subclass | Microglia | seurat | Whole cortex | 500 | 0.459 | 0.167 | 6 |
| subclass | Neural stem cell | scvi_knn | Motor cortex | 100 | 0.555 | 0.0 | 2 |
| subclass | OPC | scvi_knn | Whole cortex | 100 | 0.277 | 0.333 | 6 |
| subclass | Oligodendrocyte | seurat | Motor cortex | 100 | 0.974 | 0.5 | 6 |
| subclass | Pericyte | seurat | Whole cortex | 50 | 0.933 | 1.0 | 1 |

### Reference Cell-Type Coverage

**global**

| label | All - A single-cell transcriptomic… | Motor cortex | Cortical+Hipp. 10x | Cortical+Hipp. SSv4 | Whole cortex |
| --- | --- | --- | --- | --- | --- |
| Neural stem cell | 248 | 246 | 0 | 0 | 494 |
| Neuron | 0 | 343823 | 1149359 | 64794 | 1557976 |
| Non-neuron | 19537 | 56301 | 15241 | 1799 | 92878 |

**family**

| label | All - A single-cell transcriptomic… | Motor cortex | Cortical+Hipp. 10x | Cortical+Hipp. SSv4 | Whole cortex |
| --- | --- | --- | --- | --- | --- |
| Astrocyte | 592 | 18905 | 3899 | 963 | 24359 |
| CNS macrophage | 13580 | 8463 | 955 | 177 | 23175 |
| GABAergic | 0 | 61650 | 177594 | 20531 | 259775 |
| Glutamatergic | 0 | 282173 | 971765 | 44263 | 1298201 |
| Leukocyte | 188 | 0 | 0 | 0 | 188 |
| Neural stem cell | 248 | 246 | 0 | 0 | 494 |
| OPC | 312 | 0 | 0 | 0 | 312 |
| Oligodendrocyte | 2094 | 21549 | 8987 | 229 | 32859 |
| Vascular | 2771 | 7384 | 1400 | 430 | 11985 |

**class**

| label | All - A single-cell transcriptomic… | Motor cortex | Cortical+Hipp. 10x | Cortical+Hipp. SSv4 | Whole cortex |
| --- | --- | --- | --- | --- | --- |
| Astrocyte | 592 | 18905 | 3899 | 963 | 24359 |
| Cajal-Retzius cell | 0 | 16 | 277 | 38 | 331 |
| Hippocampal neuron | 0 | 0 | 81035 | 4990 | 86025 |
| L2/3-6 IT | 0 | 194564 | 645985 | 27175 | 867724 |
| LAMP5 | 0 | 12656 | 42144 | 4430 | 59230 |
| Leukocyte | 188 | 0 | 0 | 0 | 188 |
| Macrophage | 312 | 8408 | 955 | 177 | 9852 |
| Microglia | 13268 | 55 | 0 | 0 | 13323 |
| Neural stem cell | 248 | 246 | 0 | 0 | 494 |
| OPC | 312 | 0 | 0 | 0 | 312 |
| Oligodendrocyte | 2094 | 21549 | 8987 | 229 | 32859 |
| PVALB | 0 | 14706 | 30461 | 3581 | 48748 |
| SNCG | 0 | 2594 | 13877 | 1419 | 17890 |
| SST | 0 | 15733 | 47428 | 5366 | 68527 |
| VIP | 0 | 15961 | 43684 | 5735 | 65380 |
| Vascular | 2771 | 7384 | 1400 | 430 | 11985 |
| deep layer non-IT | 0 | 87593 | 244468 | 12060 | 344121 |

**subclass**

| label | All - A single-cell transcriptomic… | Motor cortex | Cortical+Hipp. 10x | Cortical+Hipp. SSv4 | Whole cortex |
| --- | --- | --- | --- | --- | --- |
| Astrocyte | 592 | 18905 | 3899 | 963 | 24359 |
| CA1-ProS | 0 | 0 | 15897 | 1704 | 17601 |
| CA2-IG-FC | 0 | 0 | 328 | 19 | 347 |
| CA3 | 0 | 0 | 1675 | 322 | 1997 |
| CT | 0 | 62674 | 154521 | 5501 | 222696 |
| Cajal-Retzius cell | 0 | 16 | 277 | 38 | 331 |
| DG | 0 | 0 | 58948 | 2474 | 61422 |
| Endothelial | 2232 | 3483 | 960 | 193 | 6868 |
| Ependymal | 55 | 0 | 0 | 0 | 55 |
| L2/3-6 IT | 0 | 194564 | 645985 | 27175 | 867724 |
| L5 ET | 0 | 6643 | 18443 | 1787 | 26873 |
| L6b | 0 | 5801 | 35319 | 2348 | 43468 |
| LAMP5 | 0 | 12656 | 42144 | 4430 | 59230 |
| Leukocyte | 188 | 0 | 0 | 0 | 188 |
| Macrophage | 312 | 8408 | 955 | 177 | 9852 |
| Microglia | 13268 | 55 | 0 | 0 | 13323 |
| NP | 0 | 12475 | 36185 | 2424 | 51084 |
| Neural stem cell | 248 | 246 | 0 | 0 | 494 |
| OPC | 312 | 0 | 0 | 0 | 312 |
| Oligodendrocyte | 2094 | 21549 | 8987 | 229 | 32859 |
| PVALB | 0 | 14706 | 30461 | 3581 | 48748 |
| Pericyte | 484 | 1450 | 0 | 0 | 1934 |
| SMC | 0 | 47 | 288 | 121 | 456 |
| SNCG | 0 | 2594 | 13877 | 1419 | 17890 |
| SST | 0 | 15733 | 47428 | 5366 | 68527 |
| SUB-ProS | 0 | 0 | 4187 | 471 | 4658 |
| VIP | 0 | 15961 | 43684 | 5735 | 65380 |
| VLMC | 0 | 2404 | 152 | 116 | 2672 |

### Per-Cell-Type Cutoff Sensitivity

### Global

| label | method | F1(0.0) | F1(0.25) | F1(0.5) | F1(0.75) |
| --- | --- | --- | --- | --- | --- |
| Non-neuron | scvi_knn | 0.948 | 0.947 | 0.928 | 0.879 |
| Non-neuron | scvi_rf | 0.931 | 0.862 | 0.617 | 0.31 |
| Non-neuron | seurat | 0.926 | 0.925 | 0.92 | 0.878 |
| Neuron | scvi_knn | 0.717 | 0.715 | 0.602 | 0.355 |
| Neuron | scvi_rf | 0.711 | 0.668 | 0.222 | 0.032 |
| Neuron | seurat | 0.707 | 0.708 | 0.672 | 0.523 |
| Neural stem cell | scvi_knn | 0.515 | 0.518 | 0.499 | 0.413 |
| Neural stem cell | scvi_rf | 0.386 | 0.468 | 0.39 | 0.151 |
| Neural stem cell | seurat | 0.256 | 0.254 | 0.215 | 0.135 |

**Most cutoff-sensitive cell types (F1(0) → F1(0.75) drop):**

| method | label | F1(0.0) | F1(0.75) | Drop |
| --- | --- | --- | --- | --- |
| scvi_knn | Neuron | 0.717 | 0.355 | 0.362 |
| scvi_knn | Neural stem cell | 0.515 | 0.413 | 0.102 |
| scvi_knn | Non-neuron | 0.948 | 0.879 | 0.069 |
| scvi_rf | Neuron | 0.711 | 0.032 | 0.679 |
| scvi_rf | Non-neuron | 0.931 | 0.31 | 0.621 |
| scvi_rf | Neural stem cell | 0.386 | 0.151 | 0.235 |
| seurat | Neuron | 0.707 | 0.523 | 0.184 |
| seurat | Neural stem cell | 0.256 | 0.135 | 0.121 |
| seurat | Non-neuron | 0.926 | 0.878 | 0.048 |

### Family

| label | method | F1(0.0) | F1(0.25) | F1(0.5) | F1(0.75) |
| --- | --- | --- | --- | --- | --- |
| CNS macrophage | scvi_knn | 0.96 | 0.96 | 0.951 | 0.927 |
| Vascular | scvi_rf | 0.938 | 0.878 | 0.635 | 0.329 |
| Vascular | scvi_knn | 0.929 | 0.929 | 0.91 | 0.856 |
| CNS macrophage | scvi_rf | 0.928 | 0.944 | 0.761 | 0.355 |
| CNS macrophage | seurat | 0.928 | 0.928 | 0.923 | 0.814 |
| GABAergic | seurat | 0.916 | 0.916 | 0.876 | 0.705 |
| Vascular | seurat | 0.906 | 0.906 | 0.898 | 0.848 |
| Astrocyte | scvi_knn | 0.869 | 0.869 | 0.852 | 0.79 |
| GABAergic | scvi_knn | 0.862 | 0.863 | 0.86 | 0.742 |
| Astrocyte | seurat | 0.861 | 0.861 | 0.863 | 0.858 |
| Oligodendrocyte | scvi_knn | 0.853 | 0.854 | 0.861 | 0.861 |
| Astrocyte | scvi_rf | 0.834 | 0.751 | 0.517 | 0.225 |
| Oligodendrocyte | seurat | 0.833 | 0.833 | 0.831 | 0.831 |
| GABAergic | scvi_rf | 0.82 | 0.783 | 0.429 | 0.126 |
| Glutamatergic | seurat | 0.798 | 0.798 | 0.771 | 0.634 |
| Glutamatergic | scvi_rf | 0.788 | 0.761 | 0.316 | 0.047 |
| Oligodendrocyte | scvi_rf | 0.787 | 0.744 | 0.466 | 0.21 |
| Glutamatergic | scvi_knn | 0.78 | 0.778 | 0.623 | 0.371 |
| Neural stem cell | scvi_knn | 0.515 | 0.518 | 0.499 | 0.413 |
| Neural stem cell | scvi_rf | 0.386 | 0.468 | 0.39 | 0.151 |
| OPC | scvi_knn | 0.262 | 0.261 | 0.175 | 0.048 |
| Neural stem cell | seurat | 0.256 | 0.254 | 0.215 | 0.135 |
| OPC | seurat | 0.079 | 0.078 | 0.062 | 0.038 |
| OPC | scvi_rf | 0.06 | 0.038 | 0.0 | 0.0 |

**Most cutoff-sensitive cell types (F1(0) → F1(0.75) drop):**

| method | label | F1(0.0) | F1(0.75) | Drop |
| --- | --- | --- | --- | --- |
| scvi_knn | Glutamatergic | 0.78 | 0.371 | 0.409 |
| scvi_knn | OPC | 0.262 | 0.048 | 0.214 |
| scvi_knn | GABAergic | 0.862 | 0.742 | 0.12 |
| scvi_knn | Neural stem cell | 0.515 | 0.413 | 0.102 |
| scvi_knn | Astrocyte | 0.869 | 0.79 | 0.079 |
| scvi_rf | Glutamatergic | 0.788 | 0.047 | 0.741 |
| scvi_rf | GABAergic | 0.82 | 0.126 | 0.694 |
| scvi_rf | Vascular | 0.938 | 0.329 | 0.609 |
| scvi_rf | Astrocyte | 0.834 | 0.225 | 0.609 |
| scvi_rf | Oligodendrocyte | 0.787 | 0.21 | 0.577 |
| seurat | GABAergic | 0.916 | 0.705 | 0.211 |
| seurat | Glutamatergic | 0.798 | 0.634 | 0.164 |
| seurat | Neural stem cell | 0.256 | 0.135 | 0.121 |
| seurat | CNS macrophage | 0.928 | 0.814 | 0.114 |
| seurat | Vascular | 0.906 | 0.848 | 0.058 |

### Class

| label | method | F1(0.0) | F1(0.25) | F1(0.5) | F1(0.75) |
| --- | --- | --- | --- | --- | --- |
| Hippocampal neuron | scvi_knn | 0.972 | 0.971 | 0.895 | 0.717 |
| Vascular | scvi_rf | 0.938 | 0.878 | 0.635 | 0.329 |
| Hippocampal neuron | scvi_rf | 0.937 | 0.906 | 0.49 | 0.101 |
| Vascular | scvi_knn | 0.929 | 0.929 | 0.91 | 0.856 |
| Vascular | seurat | 0.906 | 0.906 | 0.898 | 0.848 |
| Astrocyte | scvi_knn | 0.869 | 0.869 | 0.852 | 0.79 |
| Astrocyte | seurat | 0.861 | 0.861 | 0.863 | 0.858 |
| Oligodendrocyte | scvi_knn | 0.853 | 0.854 | 0.861 | 0.861 |
| Hippocampal neuron | seurat | 0.836 | 0.836 | 0.825 | 0.781 |
| Astrocyte | scvi_rf | 0.834 | 0.751 | 0.517 | 0.225 |
| Oligodendrocyte | seurat | 0.833 | 0.833 | 0.831 | 0.831 |
| Oligodendrocyte | scvi_rf | 0.787 | 0.744 | 0.466 | 0.21 |
| Cajal-Retzius cell | scvi_knn | 0.564 | 0.564 | 0.445 | 0.235 |
| Neural stem cell | scvi_knn | 0.515 | 0.518 | 0.499 | 0.413 |
| Microglia | seurat | 0.456 | 0.456 | 0.453 | 0.352 |
| Cajal-Retzius cell | seurat | 0.453 | 0.453 | 0.42 | 0.243 |
| Neural stem cell | scvi_rf | 0.386 | 0.468 | 0.39 | 0.151 |
| Cajal-Retzius cell | scvi_rf | 0.365 | 0.242 | 0.02 | 0.0 |
| OPC | scvi_knn | 0.262 | 0.261 | 0.175 | 0.048 |
| Neural stem cell | seurat | 0.256 | 0.254 | 0.215 | 0.135 |
| Macrophage | scvi_rf | 0.242 | 0.224 | 0.149 | 0.057 |
| Macrophage | scvi_knn | 0.212 | 0.212 | 0.204 | 0.172 |
| Macrophage | seurat | 0.179 | 0.179 | 0.17 | 0.146 |
| OPC | seurat | 0.079 | 0.078 | 0.062 | 0.038 |
| OPC | scvi_rf | 0.06 | 0.038 | 0.0 | 0.0 |
| Microglia | scvi_rf | 0.01 | 0.006 | 0.0 | 0.0 |
| Microglia | scvi_knn | 0.0 | 0.0 | 0.0 | 0.0 |

**Most cutoff-sensitive cell types (F1(0) → F1(0.75) drop):**

| method | label | F1(0.0) | F1(0.75) | Drop |
| --- | --- | --- | --- | --- |
| scvi_knn | Cajal-Retzius cell | 0.564 | 0.235 | 0.329 |
| scvi_knn | Hippocampal neuron | 0.972 | 0.717 | 0.255 |
| scvi_knn | OPC | 0.262 | 0.048 | 0.214 |
| scvi_knn | Neural stem cell | 0.515 | 0.413 | 0.102 |
| scvi_knn | Astrocyte | 0.869 | 0.79 | 0.079 |
| scvi_rf | Hippocampal neuron | 0.937 | 0.101 | 0.836 |
| scvi_rf | Vascular | 0.938 | 0.329 | 0.609 |
| scvi_rf | Astrocyte | 0.834 | 0.225 | 0.609 |
| scvi_rf | Oligodendrocyte | 0.787 | 0.21 | 0.577 |
| scvi_rf | Cajal-Retzius cell | 0.365 | 0.0 | 0.365 |
| seurat | Cajal-Retzius cell | 0.453 | 0.243 | 0.21 |
| seurat | Neural stem cell | 0.256 | 0.135 | 0.121 |
| seurat | Microglia | 0.456 | 0.352 | 0.104 |
| seurat | Vascular | 0.906 | 0.848 | 0.058 |
| seurat | Hippocampal neuron | 0.836 | 0.781 | 0.055 |

### Subclass

| label | method | F1(0.0) | F1(0.25) | F1(0.5) | F1(0.75) |
| --- | --- | --- | --- | --- | --- |
| DG | scvi_knn | 0.958 | 0.958 | 0.95 | 0.885 |
| Endothelial | scvi_knn | 0.949 | 0.949 | 0.937 | 0.905 |
| Endothelial | scvi_rf | 0.939 | 0.892 | 0.66 | 0.349 |
| Endothelial | seurat | 0.935 | 0.935 | 0.932 | 0.901 |
| DG | seurat | 0.881 | 0.881 | 0.881 | 0.877 |
| Astrocyte | scvi_knn | 0.869 | 0.869 | 0.852 | 0.79 |
| Astrocyte | seurat | 0.861 | 0.861 | 0.863 | 0.858 |
| Oligodendrocyte | scvi_knn | 0.853 | 0.854 | 0.861 | 0.861 |
| DG | scvi_rf | 0.844 | 0.837 | 0.49 | 0.127 |
| Astrocyte | scvi_rf | 0.834 | 0.751 | 0.517 | 0.225 |
| Oligodendrocyte | seurat | 0.833 | 0.833 | 0.831 | 0.831 |
| Pericyte | scvi_knn | 0.789 | 0.789 | 0.622 | 0.469 |
| Oligodendrocyte | scvi_rf | 0.787 | 0.744 | 0.466 | 0.21 |
| CA3 | seurat | 0.765 | 0.765 | 0.756 | 0.603 |
| CA1-ProS | scvi_rf | 0.727 | 0.727 | 0.265 | 0.019 |
| CA1-ProS | scvi_knn | 0.693 | 0.693 | 0.541 | 0.238 |
| CA3 | scvi_rf | 0.681 | 0.612 | 0.371 | 0.118 |
| CA1-ProS | seurat | 0.616 | 0.616 | 0.596 | 0.527 |
| CA3 | scvi_knn | 0.613 | 0.611 | 0.509 | 0.301 |
| Pericyte | scvi_rf | 0.567 | 0.567 | 0.276 | 0.028 |
| Cajal-Retzius cell | scvi_knn | 0.561 | 0.563 | 0.445 | 0.235 |
| Pericyte | seurat | 0.556 | 0.556 | 0.481 | 0.283 |
| Neural stem cell | scvi_knn | 0.515 | 0.518 | 0.499 | 0.413 |
| Microglia | seurat | 0.456 | 0.456 | 0.453 | 0.352 |
| Cajal-Retzius cell | seurat | 0.453 | 0.453 | 0.42 | 0.243 |
| Neural stem cell | scvi_rf | 0.386 | 0.468 | 0.39 | 0.151 |
| Cajal-Retzius cell | scvi_rf | 0.355 | 0.241 | 0.02 | 0.0 |
| OPC | scvi_knn | 0.262 | 0.261 | 0.175 | 0.048 |
| Neural stem cell | seurat | 0.256 | 0.254 | 0.215 | 0.135 |
| Macrophage | scvi_rf | 0.235 | 0.223 | 0.149 | 0.057 |
| Macrophage | scvi_knn | 0.212 | 0.212 | 0.204 | 0.172 |
| Macrophage | seurat | 0.178 | 0.178 | 0.17 | 0.146 |
| OPC | seurat | 0.079 | 0.078 | 0.062 | 0.038 |
| OPC | scvi_rf | 0.06 | 0.038 | 0.0 | 0.0 |
| Microglia | scvi_rf | 0.01 | 0.006 | 0.0 | 0.0 |
| Microglia | scvi_knn | 0.0 | 0.0 | 0.0 | 0.0 |

**Most cutoff-sensitive cell types (F1(0) → F1(0.75) drop):**

| method | label | F1(0.0) | F1(0.75) | Drop |
| --- | --- | --- | --- | --- |
| scvi_knn | CA1-ProS | 0.693 | 0.238 | 0.455 |
| scvi_knn | Cajal-Retzius cell | 0.561 | 0.235 | 0.326 |
| scvi_knn | Pericyte | 0.789 | 0.469 | 0.32 |
| scvi_knn | CA3 | 0.613 | 0.301 | 0.312 |
| scvi_knn | OPC | 0.262 | 0.048 | 0.214 |
| scvi_rf | DG | 0.844 | 0.127 | 0.717 |
| scvi_rf | CA1-ProS | 0.727 | 0.019 | 0.708 |
| scvi_rf | Astrocyte | 0.834 | 0.225 | 0.609 |
| scvi_rf | Endothelial | 0.939 | 0.349 | 0.59 |
| scvi_rf | Oligodendrocyte | 0.787 | 0.21 | 0.577 |
| seurat | Pericyte | 0.556 | 0.283 | 0.273 |
| seurat | Cajal-Retzius cell | 0.453 | 0.243 | 0.21 |
| seurat | CA3 | 0.765 | 0.603 | 0.162 |
| seurat | Neural stem cell | 0.256 | 0.135 | 0.121 |
| seurat | Microglia | 0.456 | 0.352 | 0.104 |

### Hippocampal Contamination

| cutoff | method | mean_spurious_per_query | mean_recall_non_hippo |
| --- | --- | --- | --- |
| 0.0 | scvi_knn | 1.093 | 0.696 |
| 0.05 | scvi_knn | 1.093 | 0.696 |
| 0.1 | scvi_knn | 1.093 | 0.696 |
| 0.15 | scvi_knn | 1.094 | 0.696 |
| 0.2 | scvi_knn | 1.094 | 0.696 |
| 0.25 | scvi_knn | 1.094 | 0.696 |
| 0.5 | scvi_knn | 1.045 | 0.65 |
| 0.75 | scvi_knn | 1.0 | 0.547 |
| 0.0 | scvi_rf | 1.045 | 0.66 |
| 0.05 | scvi_rf | 1.045 | 0.66 |
| 0.1 | scvi_rf | 1.045 | 0.66 |
| 0.15 | scvi_rf | 1.034 | 0.657 |
| 0.2 | scvi_rf | 1.0 | 0.634 |
| 0.25 | scvi_rf | 1.0 | 0.579 |
| 0.0 | seurat | 1.667 | 0.655 |
| 0.05 | seurat | 1.667 | 0.655 |
| 0.1 | seurat | 1.667 | 0.655 |
| 0.15 | seurat | 1.667 | 0.655 |
| 0.2 | seurat | 1.686 | 0.654 |
| 0.25 | seurat | 1.636 | 0.654 |
| 0.5 | seurat | 1.429 | 0.634 |
| 0.75 | seurat | 1.0 | 0.553 |

### Assay Exploration (mouse only)

| key | ref_type | query_type | EMM |
| --- | --- | --- | --- |
| global | cell_only | single-cell | 0.794 [0.790–0.798] |
| global | cell_and_nucleus | single-cell | 0.789 [0.784–0.793] |
| global | cell_only | single-nucleus | 0.923 [0.917–0.928] |
| global | cell_and_nucleus | single-nucleus | 0.918 [0.913–0.923] |
| family | cell_only | single-cell | 0.835 [0.831–0.838] |
| family | cell_and_nucleus | single-cell | 0.803 [0.799–0.806] |
| family | cell_only | single-nucleus | 0.907 [0.902–0.912] |
| family | cell_and_nucleus | single-nucleus | 0.889 [0.883–0.894] |
| class | cell_only | single-cell | 0.750 [0.746–0.754] |
| class | cell_and_nucleus | single-cell | 0.644 [0.639–0.649] |
| class | cell_only | single-nucleus | 0.876 [0.870–0.882] |
| class | cell_and_nucleus | single-nucleus | 0.667 [0.655–0.679] |
| subclass | cell_only | single-cell | 0.751 [0.747–0.755] |
| subclass | cell_and_nucleus | single-cell | 0.646 [0.641–0.651] |
| subclass | cell_only | single-nucleus | 0.838 [0.830–0.846] |
| subclass | cell_and_nucleus | single-nucleus | 0.659 [0.648–0.671] |

### Pareto-Optimal Configurations

| key | method_display | reference | subsample_ref | mean_f1 | total_duration_hrs | total_memory_gb |
| --- | --- | --- | --- | --- | --- | --- |
| subclass | Seurat | Cortical+Hipp. 10x | 100 | 0.852 | 0.102 | 0.04 |
| subclass | Seurat | Cortical+Hipp. 10x | 50 | 0.85 | 0.096 | 0.044 |
| subclass | scVI kNN | Cortical+Hipp. SSv4 | 100 | 0.761 | 0.053 | 0.02 |
| subclass | scVI RF | Cortical+Hipp. 10x | 100 | 0.738 | 0.053 | 0.02 |
| subclass | scVI kNN | Cortical+Hipp. 10x | 100 | 0.737 | 0.053 | 0.02 |
| subclass | scVI RF | Cortical+Hipp. SSv4 | 100 | 0.701 | 0.053 | 0.02 |
| subclass | scVI kNN | Whole cortex | 100 | 0.63 | 0.053 | 0.02 |
| subclass | scVI RF | Whole cortex | 100 | 0.572 | 0.053 | 0.02 |
| subclass | scVI kNN | Motor cortex | 100 | 0.572 | 0.053 | 0.02 |
| subclass | scVI RF | Motor cortex | 100 | 0.532 | 0.053 | 0.02 |
| class | Seurat | Cortical+Hipp. 10x | 100 | 0.826 | 0.102 | 0.04 |
| class | Seurat | Cortical+Hipp. 10x | 50 | 0.815 | 0.096 | 0.044 |
| class | scVI kNN | Cortical+Hipp. 10x | 100 | 0.802 | 0.053 | 0.02 |
| class | scVI RF | Cortical+Hipp. 10x | 100 | 0.781 | 0.053 | 0.02 |
| class | scVI kNN | Cortical+Hipp. SSv4 | 100 | 0.733 | 0.053 | 0.02 |
| class | scVI RF | Cortical+Hipp. SSv4 | 100 | 0.683 | 0.053 | 0.02 |
| class | scVI kNN | Whole cortex | 100 | 0.615 | 0.053 | 0.02 |
| class | scVI RF | Whole cortex | 100 | 0.581 | 0.053 | 0.02 |
| class | scVI kNN | Motor cortex | 100 | 0.568 | 0.053 | 0.02 |
| class | scVI RF | Motor cortex | 100 | 0.522 | 0.053 | 0.02 |
| family | Seurat | Cortical+Hipp. 10x | 500 | 0.912 | 0.173 | 0.05 |
| family | Seurat | Cortical+Hipp. 10x | 100 | 0.912 | 0.102 | 0.04 |
| family | Seurat | Cortical+Hipp. 10x | 50 | 0.907 | 0.096 | 0.044 |
| family | scVI kNN | Cortical+Hipp. 10x | 500 | 0.904 | 0.056 | 0.02 |
| family | scVI kNN | Cortical+Hipp. 10x | 100 | 0.895 | 0.053 | 0.02 |
| family | scVI RF | Cortical+Hipp. 10x | 100 | 0.891 | 0.053 | 0.02 |
| family | scVI kNN | Cortical+Hipp. SSv4 | 100 | 0.866 | 0.053 | 0.02 |
| family | scVI kNN | Motor cortex | 100 | 0.839 | 0.053 | 0.02 |
| family | scVI RF | Motor cortex | 100 | 0.802 | 0.053 | 0.02 |
| family | scVI RF | Cortical+Hipp. SSv4 | 100 | 0.799 | 0.053 | 0.02 |
| family | scVI kNN | Whole cortex | 100 | 0.768 | 0.053 | 0.02 |
| family | scVI RF | Whole cortex | 100 | 0.742 | 0.053 | 0.02 |
| global | Seurat | Cortical+Hipp. 10x | 100 | 0.94 | 0.102 | 0.04 |
| global | Seurat | Cortical+Hipp. 10x | 50 | 0.939 | 0.096 | 0.044 |
| global | scVI kNN | Cortical+Hipp. 10x | 100 | 0.911 | 0.053 | 0.02 |
| global | scVI RF | Cortical+Hipp. 10x | 100 | 0.899 | 0.053 | 0.02 |
| global | scVI kNN | Cortical+Hipp. SSv4 | 100 | 0.878 | 0.053 | 0.02 |
| global | scVI RF | Cortical+Hipp. SSv4 | 100 | 0.841 | 0.053 | 0.02 |
| global | scVI kNN | Motor cortex | 100 | 0.784 | 0.053 | 0.02 |
| global | scVI kNN | Whole cortex | 100 | 0.763 | 0.053 | 0.02 |
| global | scVI RF | Whole cortex | 100 | 0.755 | 0.053 | 0.02 |
| global | scVI RF | Motor cortex | 100 | 0.71 | 0.053 | 0.02 |

### Computational Time

| method | step | subsample_ref | mean_duration | mean_memory |
| --- | --- | --- | --- | --- |
| scVI RF/kNN | Query Processing | 50 | 0.018 | 0.02 |
| scVI RF/kNN | Query Processing | 100 | 0.018 | 0.02 |
| scVI RF/kNN | Query Processing | 500 | 0.018 | 0.02 |
| Seurat | Ref Processing | 50 | 0.046 | 0.044 |
| Seurat | Ref Processing | 100 | 0.052 | 0.04 |
| Seurat | Ref Processing | 500 | 0.117 | 0.05 |
| scVI RF/kNN | Prediction | 50 | 0.017 | 0.013 |
| scVI RF/kNN | Prediction | 100 | 0.017 | 0.013 |
| scVI RF/kNN | Prediction | 500 | 0.017 | 0.013 |
| scVI RF/kNN | Embedding | 50 | 0.021 | 0.016 |
| scVI RF/kNN | Embedding | 100 | 0.017 | 0.016 |
| scVI RF/kNN | Embedding | 500 | 0.02 | 0.016 |
| Seurat | Prediction | 50 | 0.018 | 0.02 |
| Seurat | Prediction | 100 | 0.019 | 0.021 |
| Seurat | Prediction | 500 | 0.023 | 0.021 |
| Seurat | Query Processing | 50 | 0.032 | 0.025 |
| Seurat | Query Processing | 100 | 0.032 | 0.025 |
| Seurat | Query Processing | 500 | 0.032 | 0.025 |


---
## Macro F1 vs Per-Cell-Type F1 Conflict

At global and family levels, scvi_rf leads macro F1 (global: 0.958 vs seurat 0.940; family: 0.923 vs seurat 0.906), but seurat wins most individual cell types at family (5/8: Astrocyte, GABAergic, Glutamatergic, Oligodendrocyte, Vascular) while scvi_knn wins 3/8 (CNS macrophage, OPC, Neural stem cell). At class and subclass, seurat leads both macro F1 and per-type wins. The scvi_rf macro F1 advantage at coarser levels reflects higher per-type F1 at cutoff=0 on majority neuron classes, but this advantage collapses entirely at any cutoff > 0.20 — making scvi_rf impractical as a cutoff-robust method.

## Configuration Recommendation

### Recommended Taxonomy Level: family

**Systematic failures (mean F1 < 0.50, n >= 3 studies):**

| Taxonomy level | Failing types |
| --- | --- |
| subclass | OPC (F1=0.131, n=6), Microglia (F1=0.154, n=6) |
| class | OPC (F1=0.277, n=6), Microglia (F1=0.459, n=6) |
| family | OPC (F1=0.277, n=6) |

Microglia fails at subclass and class but collapses into CNS macrophage at family (F1=0.960), resolving the failure. OPC persists as a systematic failure at family (F1=0.262–0.277 across methods, n=6) and should be excluded or manually curated. For cortex-focused studies, hippocampal subtypes (DG, CA1-ProS, CA3) also merge into Glutamatergic/GABAergic at family, eliminating spurious hippocampal assignments without requiring a cutoff.

### Recommended Configuration

| Dimension | Recommended value | Rationale |
| --- | --- | --- |
| Taxonomy level | family | Finest level without systematic failures; Microglia resolves into CNS macrophage (F1=0.960); hippocampal contamination eliminated structurally |
| Method | Seurat | Wins 5/8 cell types at family; cutoff-stable (Astrocyte 0.861→0.858 at 0.75); scvi_rf wins on macro F1 (0.923 vs 0.906) but collapses catastrophically with any cutoff |
| Reference | Cortical+Hipp. 10x | Broadest family-level coverage including GABAergic and Glutamatergic; highest mean EMM at family (seurat: 0.946); Motor cortex has zero coverage for GABAergic/Glutamatergic |
| Cutoff | 0 | scvi_rf is non-viable at any cutoff > 0.20 (Glutamatergic drops 0.741); seurat and scvi_knn are stable but cutoff offers no contamination benefit at family level |
| subsample_ref | 100 | EMM difference from 500 negligible at family (0.919 vs 0.918); Seurat ref processing 2x faster (0.052 vs 0.117 hrs) |

### Raw Performance — Recommended Configuration (Seurat, Cortical+Hipp. 10x, cutoff=0, subsample_ref=100)

| key | macro_f1_mean | macro_precision_mean | macro_recall_mean |
| --- | --- | --- | --- |
| global | 0.897 | 0.866 | 0.976 |
| family | 0.908 | 0.894 | 0.956 |
| class | 0.828 | 0.797 | 0.948 |
| subclass | 0.830 | 0.802 | 0.948 |

### Compute Time — Recommended Configuration (Seurat, subsample_ref=100)

| step | mean_duration (hrs) | mean_memory (GB) |
| --- | --- | --- |
| Query Processing | 0.032 | 0.025 |
| Ref Processing | 0.052 | 0.040 |
| Prediction | 0.019 | 0.021 |
| **Total** | **0.103** | **0.086** |

### Trade-off Narrative

scvi_knn offers comparable macro F1 at family (0.918 vs 0.906 Seurat) at half the compute time (0.053 vs 0.103 hrs total), making it the preferred alternative when throughput is the constraint. Seurat is recommended over scvi_knn when per-cell-type reliability matters — particularly for Astrocyte, Oligodendrocyte, and the major neuron families where seurat is more cutoff-stable and wins more studies.

### Pareto Note

The recommended configuration (Seurat, Cortical+Hipp. 10x, subsample_ref=100) appears in the Pareto table at family level (mean_f1=0.912, 0.102 hrs), confirming it is not dominated on both performance and compute.
