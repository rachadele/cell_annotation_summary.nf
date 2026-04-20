# Cell-Type Annotation Benchmarking: Results Summary


Generated from: `/space/grp/rschwartz/rschwartz/evaluation_summary.nf/2024-07-01/mus_musculus/100/dataset_id/SCT/gap_false/`

---

## mus_musculus

**Organism:** mus_musculus  
**Model formula:** `macro f1 ~ reference + method + cutoff + subsample ref + treatment state + method:cutoff + reference:method`  
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
| global | 0.913 [0.841–0.954] | 0.939 [0.886–0.968] | 0.924 [0.860–0.960] |
| family | 0.886 [0.839–0.921] | 0.907 [0.867–0.936] | 0.901 [0.858–0.931] |
| class | 0.811 [0.711–0.882] | 0.815 [0.716–0.885] | 0.802 [0.698–0.876] |
| subclass | 0.806 [0.710–0.876] | 0.801 [0.702–0.872] | 0.792 [0.691–0.866] |

### Method Pairwise Contrasts

| key | contrast | odds.ratio | p.value |
| --- | --- | --- | --- |
| global | seurat / scvi_rf | 0.686 | < 1e-300 |
| global | seurat / scvi_knn | 0.867 | < 1e-13 |
| global | scvi_rf / scvi_knn | 1.263 | < 1e-300 |
| family | seurat / scvi_rf | 0.797 | < 1e-300 |
| family | seurat / scvi_knn | 0.86 | < 1e-13 |
| family | scvi_rf / scvi_knn | 1.079 | < 1e-4 |
| class | seurat / scvi_rf | 0.974 | 0.197 |
| class | seurat / scvi_knn | 1.062 | < 1e-3 |
| class | scvi_rf / scvi_knn | 1.091 | < 1e-7 |
| subclass | seurat / scvi_rf | 1.035 | 0.055 |
| subclass | seurat / scvi_knn | 1.093 | < 1e-8 |
| subclass | scvi_rf / scvi_knn | 1.056 | < 1e-3 |

### Cutoff Sensitivity (method × cutoff EMMs)

| key | cutoff | scvi_knn | scvi_rf | seurat |
| --- | --- | --- | --- | --- |
| global | 0.0 | 0.924 | 0.939 | 0.913 |
| global | 0.05 | 0.919 | 0.924 | 0.91 |
| global | 0.1 | 0.914 | 0.906 | 0.907 |
| global | 0.15 | 0.908 | 0.884 | 0.903 |
| global | 0.2 | 0.902 | 0.859 | 0.9 |
| global | 0.25 | 0.896 | 0.828 | 0.896 |
| global | 0.5 | 0.86 | 0.602 | 0.877 |
| global | 0.75 | 0.813 | 0.323 | 0.854 |
| family | 0.0 | 0.901 | 0.907 | 0.886 |
| family | 0.05 | 0.897 | 0.89 | 0.884 |
| family | 0.1 | 0.894 | 0.87 | 0.881 |
| family | 0.15 | 0.89 | 0.846 | 0.878 |
| family | 0.2 | 0.887 | 0.82 | 0.876 |
| family | 0.25 | 0.883 | 0.79 | 0.873 |
| family | 0.5 | 0.863 | 0.591 | 0.858 |
| family | 0.75 | 0.84 | 0.358 | 0.842 |
| class | 0.0 | 0.802 | 0.815 | 0.811 |
| class | 0.05 | 0.798 | 0.792 | 0.809 |
| class | 0.1 | 0.795 | 0.766 | 0.808 |
| class | 0.15 | 0.792 | 0.738 | 0.806 |
| class | 0.2 | 0.789 | 0.709 | 0.804 |
| class | 0.25 | 0.786 | 0.677 | 0.802 |
| class | 0.5 | 0.769 | 0.5 | 0.794 |
| class | 0.75 | 0.751 | 0.322 | 0.785 |
| subclass | 0.0 | 0.792 | 0.801 | 0.806 |
| subclass | 0.05 | 0.789 | 0.777 | 0.805 |
| subclass | 0.1 | 0.786 | 0.751 | 0.803 |
| subclass | 0.15 | 0.783 | 0.723 | 0.801 |
| subclass | 0.2 | 0.779 | 0.694 | 0.8 |
| subclass | 0.25 | 0.776 | 0.663 | 0.798 |
| subclass | 0.5 | 0.76 | 0.49 | 0.79 |
| subclass | 0.75 | 0.743 | 0.319 | 0.781 |

### Reference × Method Performance

| key | ref_short | scvi_knn | scvi_rf | seurat |
| --- | --- | --- | --- | --- |
| global | Cortical+Hipp. 10x | 0.935 | 0.95 | 0.96 |
| global | Cortical+Hipp. SSv4 | 0.907 | 0.916 | 0.874 |
| global | Motor cortex | 0.926 | 0.943 | 0.91 |
| global | Whole cortex | 0.926 | 0.942 | 0.88 |
| family | Cortical+Hipp. 10x | 0.926 | 0.938 | 0.934 |
| family | Cortical+Hipp. SSv4 | 0.892 | 0.872 | 0.87 |
| family | Motor cortex | 0.919 | 0.93 | 0.908 |
| family | Whole cortex | 0.85 | 0.869 | 0.796 |
| class | Cortical+Hipp. 10x | 0.882 | 0.896 | 0.889 |
| class | Cortical+Hipp. SSv4 | 0.831 | 0.792 | 0.805 |
| class | Motor cortex | 0.754 | 0.804 | 0.844 |
| class | Whole cortex | 0.703 | 0.738 | 0.655 |
| subclass | Cortical+Hipp. 10x | 0.869 | 0.882 | 0.886 |
| subclass | Cortical+Hipp. SSv4 | 0.825 | 0.779 | 0.8 |
| subclass | Motor cortex | 0.747 | 0.791 | 0.84 |
| subclass | Whole cortex | 0.694 | 0.722 | 0.647 |

### Reference Ranking (mean EMM across methods and keys)

| reference | mean_emm |
| --- | --- |
| Cortical+Hipp. 10x | 0.912 |
| Motor cortex | 0.86 |
| Cortical+Hipp. SSv4 | 0.847 |
| Whole cortex | 0.785 |

### Reference Subsample Size

| key | subsample_ref | EMM |
| --- | --- | --- |
| global | 500 | 0.931 [0.873–0.964] |
| global | 100 | 0.929 [0.868–0.963] |
| global | 50 | 0.917 [0.848–0.956] |
| family | 500 | 0.901 [0.859–0.932] |
| family | 100 | 0.902 [0.860–0.932] |
| family | 50 | 0.892 [0.846–0.925] |
| class | 500 | 0.812 [0.712–0.883] |
| class | 100 | 0.812 [0.712–0.883] |
| class | 50 | 0.804 [0.701–0.877] |
| subclass | 500 | 0.803 [0.705–0.874] |
| subclass | 100 | 0.802 [0.705–0.873] |
| subclass | 50 | 0.794 [0.694–0.868] |

### Biological Covariates

**treatment_state**

| key | treatment_state | EMM |
| --- | --- | --- |
| global | no treatment | 0.930 [0.870–0.963] |
| global | treatment | 0.922 [0.856–0.959] |
| family | no treatment | 0.897 [0.853–0.929] |
| family | treatment | 0.900 [0.857–0.931] |
| class | no treatment | 0.800 [0.697–0.875] |
| class | treatment | 0.818 [0.720–0.887] |
| subclass | no treatment | 0.790 [0.688–0.865] |
| subclass | treatment | 0.809 [0.714–0.878] |

### Between-Study Heterogeneity

**global — Well-classified (mean F1 ≥ 0.85)**

| key | label | n_studies | mean_f1 | std_f1 | mean_precision | mean_recall |
| --- | --- | --- | --- | --- | --- | --- |
| global | Non-neuron | 7 | 0.953 | 0.028 | 0.993 | 0.924 |

**global — Hard / high-variance (mean F1 < 0.75 or std > 0.20)**

| key | label | n_studies | mean_f1 | std_f1 | mean_precision | mean_recall | failure_mode |
| --- | --- | --- | --- | --- | --- | --- | --- |
| global | Neural stem cell | 2 | 0.372 | 0.114 | 0.371 | 0.545 | — |
| global | Neuron | 6 | 0.832 | 0.229 | 0.788 | 0.964 | Study variance |

**family — Well-classified (mean F1 ≥ 0.85)**

| key | label | n_studies | mean_f1 | std_f1 | mean_precision | mean_recall |
| --- | --- | --- | --- | --- | --- | --- |
| family | CNS macrophage | 6 | 0.958 | 0.032 | 0.983 | 0.947 |
| family | Vascular | 6 | 0.928 | 0.047 | 0.982 | 0.906 |
| family | Astrocyte | 6 | 0.893 | 0.068 | 0.943 | 0.884 |
| family | GABAergic | 4 | 0.874 | 0.051 | 0.887 | 0.911 |

**family — Hard / high-variance (mean F1 < 0.75 or std > 0.20)**

| key | label | n_studies | mean_f1 | std_f1 | mean_precision | mean_recall | failure_mode |
| --- | --- | --- | --- | --- | --- | --- | --- |
| family | OPC | 6 | 0.131 | 0.109 | 0.633 | 0.137 | — |
| family | Neural stem cell | 2 | 0.372 | 0.114 | 0.371 | 0.545 | — |
| family | Glutamatergic | 6 | 0.762 | 0.341 | 0.733 | 0.955 | Study variance |

**class — Well-classified (mean F1 ≥ 0.85)**

| key | label | n_studies | mean_f1 | std_f1 | mean_precision | mean_recall |
| --- | --- | --- | --- | --- | --- | --- |
| class | Vascular | 6 | 0.928 | 0.047 | 0.982 | 0.906 |
| class | Hippocampal neuron | 1 | 0.915 | nan | 0.999 | 0.877 |
| class | Astrocyte | 6 | 0.893 | 0.068 | 0.943 | 0.884 |

**class — Hard / high-variance (mean F1 < 0.75 or std > 0.20)**

| key | label | n_studies | mean_f1 | std_f1 | mean_precision | mean_recall | failure_mode |
| --- | --- | --- | --- | --- | --- | --- | --- |
| class | OPC | 6 | 0.131 | 0.109 | 0.633 | 0.137 | — |
| class | Microglia | 6 | 0.154 | 0.049 | 0.964 | 0.146 | Label escape |
| class | Macrophage | 2 | 0.207 | 0.081 | 0.136 | 0.734 | Over-prediction |
| class | Neural stem cell | 2 | 0.372 | 0.114 | 0.371 | 0.545 | — |
| class | Cajal-Retzius cell | 2 | 0.573 | 0.25 | 0.99 | 0.559 | Study variance |

**subclass — Well-classified (mean F1 ≥ 0.85)**

| key | label | n_studies | mean_f1 | std_f1 | mean_precision | mean_recall |
| --- | --- | --- | --- | --- | --- | --- |
| subclass | Endothelial | 6 | 0.953 | 0.025 | 0.999 | 0.918 |
| subclass | DG | 1 | 0.894 | nan | 0.998 | 0.846 |
| subclass | Astrocyte | 6 | 0.893 | 0.068 | 0.943 | 0.884 |

**subclass — Hard / high-variance (mean F1 < 0.75 or std > 0.20)**

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

| key | label | method | reference | subsample_ref | mean_f1_across_studies | win_fraction | n_studies | mean_support |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| global | Neural stem cell | scvi_knn | Motor cortex | 500 | 0.514 | 0.5 | 2 | 3.702991452991453 |
| global | Neuron | seurat | Cortical+Hipp. 10x | 500 | 0.882 | 0.333 | 6 | 39.80992063492064 |
| global | Non-neuron | seurat | Cortical+Hipp. 10x | 500 | 0.992 | 0.429 | 7 | 64.7561224489796 |
| family | Astrocyte | seurat | Cortical+Hipp. 10x | 500 | 0.943 | 0.667 | 6 | 10.332539682539682 |
| family | CNS macrophage | scvi_knn | Cortical+Hipp. 10x | 100 | 0.983 | 0.5 | 6 | 8.907142857142857 |
| family | GABAergic | seurat | Cortical+Hipp. SSv4 | 50 | 0.99 | 0.5 | 4 | 6.245833333333334 |
| family | Glutamatergic | seurat | Whole cortex | 500 | 0.892 | 0.5 | 6 | 33.645370370370365 |
| family | Neural stem cell | scvi_knn | Motor cortex | 500 | 0.514 | 0.5 | 2 | 3.702991452991453 |
| family | OPC | scvi_knn | Whole cortex | 100 | 0.277 | 0.333 | 6 | 8.003535353535353 |
| family | Oligodendrocyte | seurat | Motor cortex | 100 | 0.974 | 0.5 | 6 | 17.315079365079367 |
| family | Vascular | seurat | Cortical+Hipp. 10x | 500 | 0.977 | 0.333 | 6 | 31.713492063492065 |
| class | Astrocyte | seurat | Cortical+Hipp. 10x | 500 | 0.943 | 0.667 | 6 | 10.332539682539682 |
| class | Cajal-Retzius cell | seurat | Cortical+Hipp. 10x | 50 | 0.992 | 1.0 | 2 | 1.4444444444444444 |
| class | Hippocampal neuron | scvi_knn | Cortical+Hipp. 10x | 100 | 0.996 | 1.0 | 1 | 65.5 |
| class | Macrophage | scvi_rf | Cortical+Hipp. SSv4 | 500 | 0.263 | 0.5 | 2 | 2.1319444444444446 |
| class | Microglia | seurat | Whole cortex | 500 | 0.459 | 0.167 | 6 | 8.315873015873017 |
| class | Neural stem cell | scvi_knn | Motor cortex | 500 | 0.514 | 0.5 | 2 | 3.702991452991453 |
| class | OPC | scvi_knn | Whole cortex | 100 | 0.277 | 0.333 | 6 | 8.003535353535353 |
| class | Oligodendrocyte | seurat | Motor cortex | 100 | 0.974 | 0.5 | 6 | 17.315079365079367 |
| class | Vascular | seurat | Cortical+Hipp. 10x | 500 | 0.977 | 0.333 | 6 | 31.713492063492065 |
| subclass | Astrocyte | seurat | Cortical+Hipp. 10x | 500 | 0.943 | 0.667 | 6 | 10.332539682539682 |
| subclass | CA1-ProS | seurat | Whole cortex | 100 | 0.933 | 1.0 | 1 | 19.5 |
| subclass | CA3 | seurat | Whole cortex | 100 | 0.943 | 1.0 | 1 | 11.5 |
| subclass | Cajal-Retzius cell | seurat | Cortical+Hipp. 10x | 50 | 0.992 | 1.0 | 2 | 1.4444444444444444 |
| subclass | DG | seurat | Whole cortex | 500 | 0.996 | 1.0 | 1 | 34.5 |
| subclass | Endothelial | seurat | Cortical+Hipp. 10x | 500 | 0.986 | 0.333 | 6 | 27.807142857142853 |
| subclass | Macrophage | scvi_rf | Cortical+Hipp. SSv4 | 100 | 0.247 | 0.0 | 2 | 2.1319444444444446 |
| subclass | Microglia | seurat | Whole cortex | 500 | 0.459 | 0.167 | 6 | 8.315873015873017 |
| subclass | Neural stem cell | scvi_knn | Motor cortex | 100 | 0.555 | 0.0 | 2 | 3.702991452991453 |
| subclass | OPC | scvi_knn | Whole cortex | 100 | 0.277 | 0.333 | 6 | 8.003535353535353 |
| subclass | Oligodendrocyte | seurat | Motor cortex | 100 | 0.974 | 0.5 | 6 | 17.315079365079367 |
| subclass | Pericyte | seurat | Whole cortex | 50 | 0.933 | 1.0 | 1 | 2.0 |

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

**Low-F1 cell types at cutoff=0 (F1 < 0.5) — precision/recall across all cutoffs:**

| label | method | cutoff | F1 | precision | recall |
| --- | --- | --- | --- | --- | --- |
| Neural stem cell | scvi_knn | 0.0 | 0.515 | 0.515 | 0.617 |
| Neural stem cell | scvi_knn | 0.05 | 0.515 | 0.515 | 0.617 |
| Neural stem cell | scvi_knn | 0.1 | 0.515 | 0.515 | 0.617 |
| Neural stem cell | scvi_knn | 0.15 | 0.515 | 0.515 | 0.617 |
| Neural stem cell | scvi_knn | 0.2 | 0.516 | 0.517 | 0.617 |
| Neural stem cell | scvi_knn | 0.25 | 0.518 | 0.522 | 0.617 |
| Neural stem cell | scvi_knn | 0.5 | 0.499 | 0.608 | 0.524 |
| Neural stem cell | scvi_knn | 0.75 | 0.413 | 0.7 | 0.368 |
| Neural stem cell | scvi_rf | 0.0 | 0.386 | 0.288 | 0.748 |
| Neural stem cell | scvi_rf | 0.05 | 0.386 | 0.288 | 0.748 |
| Neural stem cell | scvi_rf | 0.1 | 0.386 | 0.288 | 0.748 |
| Neural stem cell | scvi_rf | 0.15 | 0.4 | 0.304 | 0.746 |
| Neural stem cell | scvi_rf | 0.2 | 0.43 | 0.346 | 0.726 |
| Neural stem cell | scvi_rf | 0.25 | 0.468 | 0.409 | 0.681 |
| Neural stem cell | scvi_rf | 0.5 | 0.39 | 0.727 | 0.335 |
| Neural stem cell | scvi_rf | 0.75 | 0.151 | 0.797 | 0.118 |
| Neural stem cell | seurat | 0.0 | 0.256 | 0.342 | 0.305 |
| Neural stem cell | seurat | 0.05 | 0.256 | 0.342 | 0.305 |
| Neural stem cell | seurat | 0.1 | 0.256 | 0.342 | 0.305 |
| Neural stem cell | seurat | 0.15 | 0.256 | 0.342 | 0.305 |
| Neural stem cell | seurat | 0.2 | 0.256 | 0.342 | 0.304 |
| Neural stem cell | seurat | 0.25 | 0.254 | 0.342 | 0.302 |
| Neural stem cell | seurat | 0.5 | 0.215 | 0.367 | 0.229 |
| Neural stem cell | seurat | 0.75 | 0.135 | 0.424 | 0.124 |

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

**Low-F1 cell types at cutoff=0 (F1 < 0.5) — precision/recall across all cutoffs:**

| label | method | cutoff | F1 | precision | recall |
| --- | --- | --- | --- | --- | --- |
| Neural stem cell | scvi_knn | 0.0 | 0.515 | 0.515 | 0.617 |
| Neural stem cell | scvi_knn | 0.05 | 0.515 | 0.515 | 0.617 |
| Neural stem cell | scvi_knn | 0.1 | 0.515 | 0.515 | 0.617 |
| Neural stem cell | scvi_knn | 0.15 | 0.515 | 0.515 | 0.617 |
| Neural stem cell | scvi_knn | 0.2 | 0.516 | 0.517 | 0.617 |
| Neural stem cell | scvi_knn | 0.25 | 0.518 | 0.522 | 0.617 |
| Neural stem cell | scvi_knn | 0.5 | 0.499 | 0.608 | 0.524 |
| Neural stem cell | scvi_knn | 0.75 | 0.413 | 0.7 | 0.368 |
| Neural stem cell | scvi_rf | 0.0 | 0.386 | 0.288 | 0.748 |
| Neural stem cell | scvi_rf | 0.05 | 0.386 | 0.288 | 0.748 |
| Neural stem cell | scvi_rf | 0.1 | 0.386 | 0.288 | 0.748 |
| Neural stem cell | scvi_rf | 0.15 | 0.4 | 0.304 | 0.746 |
| Neural stem cell | scvi_rf | 0.2 | 0.43 | 0.346 | 0.726 |
| Neural stem cell | scvi_rf | 0.25 | 0.468 | 0.409 | 0.681 |
| Neural stem cell | scvi_rf | 0.5 | 0.39 | 0.727 | 0.335 |
| Neural stem cell | scvi_rf | 0.75 | 0.151 | 0.797 | 0.118 |
| Neural stem cell | seurat | 0.0 | 0.256 | 0.342 | 0.305 |
| Neural stem cell | seurat | 0.05 | 0.256 | 0.342 | 0.305 |
| Neural stem cell | seurat | 0.1 | 0.256 | 0.342 | 0.305 |
| Neural stem cell | seurat | 0.15 | 0.256 | 0.342 | 0.305 |
| Neural stem cell | seurat | 0.2 | 0.256 | 0.342 | 0.304 |
| Neural stem cell | seurat | 0.25 | 0.254 | 0.342 | 0.302 |
| Neural stem cell | seurat | 0.5 | 0.215 | 0.367 | 0.229 |
| Neural stem cell | seurat | 0.75 | 0.135 | 0.424 | 0.124 |
| OPC | scvi_knn | 0.0 | 0.262 | 0.902 | 0.218 |
| OPC | scvi_knn | 0.05 | 0.262 | 0.902 | 0.218 |
| OPC | scvi_knn | 0.1 | 0.262 | 0.902 | 0.218 |
| OPC | scvi_knn | 0.15 | 0.262 | 0.902 | 0.218 |
| OPC | scvi_knn | 0.2 | 0.262 | 0.902 | 0.218 |
| OPC | scvi_knn | 0.25 | 0.261 | 0.901 | 0.217 |
| OPC | scvi_knn | 0.5 | 0.175 | 0.849 | 0.148 |
| OPC | scvi_knn | 0.75 | 0.048 | 0.847 | 0.036 |
| OPC | scvi_rf | 0.0 | 0.06 | 0.996 | 0.044 |
| OPC | scvi_rf | 0.05 | 0.06 | 0.996 | 0.044 |
| OPC | scvi_rf | 0.1 | 0.06 | 0.996 | 0.044 |
| OPC | scvi_rf | 0.15 | 0.058 | 0.996 | 0.042 |
| OPC | scvi_rf | 0.2 | 0.05 | 0.994 | 0.037 |
| OPC | scvi_rf | 0.25 | 0.038 | 1.0 | 0.029 |
| OPC | scvi_rf | 0.5 | 0.0 | nan | 0.0 |
| OPC | scvi_rf | 0.75 | 0.0 | nan | 0.0 |
| OPC | seurat | 0.0 | 0.079 | 0.252 | 0.083 |
| OPC | seurat | 0.05 | 0.079 | 0.252 | 0.083 |
| OPC | seurat | 0.1 | 0.079 | 0.252 | 0.083 |
| OPC | seurat | 0.15 | 0.079 | 0.252 | 0.083 |
| OPC | seurat | 0.2 | 0.078 | 0.252 | 0.083 |
| OPC | seurat | 0.25 | 0.078 | 0.252 | 0.083 |
| OPC | seurat | 0.5 | 0.062 | 0.275 | 0.065 |
| OPC | seurat | 0.75 | 0.038 | 0.809 | 0.034 |

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

**Low-F1 cell types at cutoff=0 (F1 < 0.5) — precision/recall across all cutoffs:**

| label | method | cutoff | F1 | precision | recall |
| --- | --- | --- | --- | --- | --- |
| Cajal-Retzius cell | scvi_knn | 0.0 | 0.564 | 1.0 | 0.537 |
| Cajal-Retzius cell | scvi_knn | 0.05 | 0.564 | 1.0 | 0.537 |
| Cajal-Retzius cell | scvi_knn | 0.1 | 0.564 | 1.0 | 0.537 |
| Cajal-Retzius cell | scvi_knn | 0.15 | 0.564 | 1.0 | 0.537 |
| Cajal-Retzius cell | scvi_knn | 0.2 | 0.564 | 1.0 | 0.537 |
| Cajal-Retzius cell | scvi_knn | 0.25 | 0.564 | 1.0 | 0.537 |
| Cajal-Retzius cell | scvi_knn | 0.5 | 0.445 | 1.0 | 0.419 |
| Cajal-Retzius cell | scvi_knn | 0.75 | 0.235 | 1.0 | 0.22 |
| Cajal-Retzius cell | scvi_rf | 0.0 | 0.365 | 1.0 | 0.337 |
| Cajal-Retzius cell | scvi_rf | 0.05 | 0.365 | 1.0 | 0.337 |
| Cajal-Retzius cell | scvi_rf | 0.1 | 0.365 | 1.0 | 0.337 |
| Cajal-Retzius cell | scvi_rf | 0.15 | 0.365 | 1.0 | 0.337 |
| Cajal-Retzius cell | scvi_rf | 0.2 | 0.343 | 1.0 | 0.312 |
| Cajal-Retzius cell | scvi_rf | 0.25 | 0.242 | 1.0 | 0.216 |
| Cajal-Retzius cell | scvi_rf | 0.5 | 0.02 | 1.0 | 0.015 |
| Cajal-Retzius cell | scvi_rf | 0.75 | 0.0 | nan | 0.0 |
| Cajal-Retzius cell | seurat | 0.0 | 0.453 | 0.929 | 0.437 |
| Cajal-Retzius cell | seurat | 0.05 | 0.453 | 0.929 | 0.437 |
| Cajal-Retzius cell | seurat | 0.1 | 0.453 | 0.929 | 0.437 |
| Cajal-Retzius cell | seurat | 0.15 | 0.453 | 0.929 | 0.437 |
| Cajal-Retzius cell | seurat | 0.2 | 0.453 | 0.929 | 0.437 |
| Cajal-Retzius cell | seurat | 0.25 | 0.453 | 0.929 | 0.437 |
| Cajal-Retzius cell | seurat | 0.5 | 0.42 | 0.901 | 0.404 |
| Cajal-Retzius cell | seurat | 0.75 | 0.243 | 0.878 | 0.229 |
| Macrophage | scvi_knn | 0.0 | 0.212 | 0.131 | 0.752 |
| Macrophage | scvi_knn | 0.05 | 0.212 | 0.131 | 0.752 |
| Macrophage | scvi_knn | 0.1 | 0.212 | 0.131 | 0.752 |
| Macrophage | scvi_knn | 0.15 | 0.212 | 0.131 | 0.752 |
| Macrophage | scvi_knn | 0.2 | 0.212 | 0.131 | 0.752 |
| Macrophage | scvi_knn | 0.25 | 0.212 | 0.131 | 0.752 |
| Macrophage | scvi_knn | 0.5 | 0.204 | 0.127 | 0.719 |
| Macrophage | scvi_knn | 0.75 | 0.172 | 0.11 | 0.586 |
| Macrophage | scvi_rf | 0.0 | 0.242 | 0.148 | 0.874 |
| Macrophage | scvi_rf | 0.05 | 0.242 | 0.148 | 0.874 |
| Macrophage | scvi_rf | 0.1 | 0.242 | 0.148 | 0.874 |
| Macrophage | scvi_rf | 0.15 | 0.242 | 0.148 | 0.874 |
| Macrophage | scvi_rf | 0.2 | 0.24 | 0.148 | 0.865 |
| Macrophage | scvi_rf | 0.25 | 0.224 | 0.138 | 0.807 |
| Macrophage | scvi_rf | 0.5 | 0.149 | 0.098 | 0.495 |
| Macrophage | scvi_rf | 0.75 | 0.057 | 0.046 | 0.124 |
| Macrophage | seurat | 0.0 | 0.179 | 0.128 | 0.583 |
| Macrophage | seurat | 0.05 | 0.179 | 0.128 | 0.583 |
| Macrophage | seurat | 0.1 | 0.179 | 0.128 | 0.583 |
| Macrophage | seurat | 0.15 | 0.179 | 0.128 | 0.583 |
| Macrophage | seurat | 0.2 | 0.179 | 0.128 | 0.583 |
| Macrophage | seurat | 0.25 | 0.179 | 0.129 | 0.583 |
| Macrophage | seurat | 0.5 | 0.17 | 0.121 | 0.559 |
| Macrophage | seurat | 0.75 | 0.146 | 0.111 | 0.475 |
| Microglia | scvi_knn | 0.0 | 0.0 | nan | 0.0 |
| Microglia | scvi_knn | 0.05 | 0.0 | nan | 0.0 |
| Microglia | scvi_knn | 0.1 | 0.0 | nan | 0.0 |
| Microglia | scvi_knn | 0.15 | 0.0 | nan | 0.0 |
| Microglia | scvi_knn | 0.2 | 0.0 | nan | 0.0 |
| Microglia | scvi_knn | 0.25 | 0.0 | nan | 0.0 |
| Microglia | scvi_knn | 0.5 | 0.0 | nan | 0.0 |
| Microglia | scvi_knn | 0.75 | 0.0 | nan | 0.0 |
| Microglia | scvi_rf | 0.0 | 0.01 | 1.0 | 0.006 |
| Microglia | scvi_rf | 0.05 | 0.01 | 1.0 | 0.006 |
| Microglia | scvi_rf | 0.1 | 0.01 | 1.0 | 0.006 |
| Microglia | scvi_rf | 0.15 | 0.009 | 1.0 | 0.006 |
| Microglia | scvi_rf | 0.2 | 0.009 | 1.0 | 0.006 |
| Microglia | scvi_rf | 0.25 | 0.006 | 1.0 | 0.004 |
| Microglia | scvi_rf | 0.5 | 0.0 | nan | 0.0 |
| Microglia | scvi_rf | 0.75 | 0.0 | nan | 0.0 |
| Microglia | seurat | 0.0 | 0.456 | 0.951 | 0.439 |
| Microglia | seurat | 0.05 | 0.456 | 0.951 | 0.439 |
| Microglia | seurat | 0.1 | 0.456 | 0.951 | 0.439 |
| Microglia | seurat | 0.15 | 0.456 | 0.951 | 0.439 |
| Microglia | seurat | 0.2 | 0.456 | 0.951 | 0.439 |
| Microglia | seurat | 0.25 | 0.456 | 0.952 | 0.439 |
| Microglia | seurat | 0.5 | 0.453 | 0.954 | 0.435 |
| Microglia | seurat | 0.75 | 0.352 | 0.964 | 0.331 |
| Neural stem cell | scvi_knn | 0.0 | 0.515 | 0.515 | 0.617 |
| Neural stem cell | scvi_knn | 0.05 | 0.515 | 0.515 | 0.617 |
| Neural stem cell | scvi_knn | 0.1 | 0.515 | 0.515 | 0.617 |
| Neural stem cell | scvi_knn | 0.15 | 0.515 | 0.515 | 0.617 |
| Neural stem cell | scvi_knn | 0.2 | 0.516 | 0.517 | 0.617 |
| Neural stem cell | scvi_knn | 0.25 | 0.518 | 0.522 | 0.617 |
| Neural stem cell | scvi_knn | 0.5 | 0.499 | 0.608 | 0.524 |
| Neural stem cell | scvi_knn | 0.75 | 0.413 | 0.7 | 0.368 |
| Neural stem cell | scvi_rf | 0.0 | 0.386 | 0.288 | 0.748 |
| Neural stem cell | scvi_rf | 0.05 | 0.386 | 0.288 | 0.748 |
| Neural stem cell | scvi_rf | 0.1 | 0.386 | 0.288 | 0.748 |
| Neural stem cell | scvi_rf | 0.15 | 0.4 | 0.304 | 0.746 |
| Neural stem cell | scvi_rf | 0.2 | 0.43 | 0.346 | 0.726 |
| Neural stem cell | scvi_rf | 0.25 | 0.468 | 0.409 | 0.681 |
| Neural stem cell | scvi_rf | 0.5 | 0.39 | 0.727 | 0.335 |
| Neural stem cell | scvi_rf | 0.75 | 0.151 | 0.797 | 0.118 |
| Neural stem cell | seurat | 0.0 | 0.256 | 0.342 | 0.305 |
| Neural stem cell | seurat | 0.05 | 0.256 | 0.342 | 0.305 |
| Neural stem cell | seurat | 0.1 | 0.256 | 0.342 | 0.305 |
| Neural stem cell | seurat | 0.15 | 0.256 | 0.342 | 0.305 |
| Neural stem cell | seurat | 0.2 | 0.256 | 0.342 | 0.304 |
| Neural stem cell | seurat | 0.25 | 0.254 | 0.342 | 0.302 |
| Neural stem cell | seurat | 0.5 | 0.215 | 0.367 | 0.229 |
| Neural stem cell | seurat | 0.75 | 0.135 | 0.424 | 0.124 |
| OPC | scvi_knn | 0.0 | 0.262 | 0.902 | 0.218 |
| OPC | scvi_knn | 0.05 | 0.262 | 0.902 | 0.218 |
| OPC | scvi_knn | 0.1 | 0.262 | 0.902 | 0.218 |
| OPC | scvi_knn | 0.15 | 0.262 | 0.902 | 0.218 |
| OPC | scvi_knn | 0.2 | 0.262 | 0.902 | 0.218 |
| OPC | scvi_knn | 0.25 | 0.261 | 0.901 | 0.217 |
| OPC | scvi_knn | 0.5 | 0.175 | 0.849 | 0.148 |
| OPC | scvi_knn | 0.75 | 0.048 | 0.847 | 0.036 |
| OPC | scvi_rf | 0.0 | 0.06 | 0.996 | 0.044 |
| OPC | scvi_rf | 0.05 | 0.06 | 0.996 | 0.044 |
| OPC | scvi_rf | 0.1 | 0.06 | 0.996 | 0.044 |
| OPC | scvi_rf | 0.15 | 0.058 | 0.996 | 0.042 |
| OPC | scvi_rf | 0.2 | 0.05 | 0.994 | 0.037 |
| OPC | scvi_rf | 0.25 | 0.038 | 1.0 | 0.029 |
| OPC | scvi_rf | 0.5 | 0.0 | nan | 0.0 |
| OPC | scvi_rf | 0.75 | 0.0 | nan | 0.0 |
| OPC | seurat | 0.0 | 0.079 | 0.252 | 0.083 |
| OPC | seurat | 0.05 | 0.079 | 0.252 | 0.083 |
| OPC | seurat | 0.1 | 0.079 | 0.252 | 0.083 |
| OPC | seurat | 0.15 | 0.079 | 0.252 | 0.083 |
| OPC | seurat | 0.2 | 0.078 | 0.252 | 0.083 |
| OPC | seurat | 0.25 | 0.078 | 0.252 | 0.083 |
| OPC | seurat | 0.5 | 0.062 | 0.275 | 0.065 |
| OPC | seurat | 0.75 | 0.038 | 0.809 | 0.034 |

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

**Low-F1 cell types at cutoff=0 (F1 < 0.5) — precision/recall across all cutoffs:**

| label | method | cutoff | F1 | precision | recall |
| --- | --- | --- | --- | --- | --- |
| Cajal-Retzius cell | scvi_knn | 0.0 | 0.561 | 0.982 | 0.537 |
| Cajal-Retzius cell | scvi_knn | 0.05 | 0.561 | 0.982 | 0.537 |
| Cajal-Retzius cell | scvi_knn | 0.1 | 0.561 | 0.982 | 0.537 |
| Cajal-Retzius cell | scvi_knn | 0.15 | 0.561 | 0.982 | 0.537 |
| Cajal-Retzius cell | scvi_knn | 0.2 | 0.561 | 0.994 | 0.537 |
| Cajal-Retzius cell | scvi_knn | 0.25 | 0.563 | 0.997 | 0.537 |
| Cajal-Retzius cell | scvi_knn | 0.5 | 0.445 | 1.0 | 0.419 |
| Cajal-Retzius cell | scvi_knn | 0.75 | 0.235 | 1.0 | 0.22 |
| Cajal-Retzius cell | scvi_rf | 0.0 | 0.355 | 0.843 | 0.337 |
| Cajal-Retzius cell | scvi_rf | 0.05 | 0.355 | 0.843 | 0.337 |
| Cajal-Retzius cell | scvi_rf | 0.1 | 0.355 | 0.843 | 0.337 |
| Cajal-Retzius cell | scvi_rf | 0.15 | 0.361 | 0.924 | 0.337 |
| Cajal-Retzius cell | scvi_rf | 0.2 | 0.339 | 0.916 | 0.312 |
| Cajal-Retzius cell | scvi_rf | 0.25 | 0.241 | 0.923 | 0.216 |
| Cajal-Retzius cell | scvi_rf | 0.5 | 0.02 | 1.0 | 0.015 |
| Cajal-Retzius cell | scvi_rf | 0.75 | 0.0 | nan | 0.0 |
| Cajal-Retzius cell | seurat | 0.0 | 0.453 | 0.895 | 0.437 |
| Cajal-Retzius cell | seurat | 0.05 | 0.453 | 0.895 | 0.437 |
| Cajal-Retzius cell | seurat | 0.1 | 0.453 | 0.895 | 0.437 |
| Cajal-Retzius cell | seurat | 0.15 | 0.453 | 0.895 | 0.437 |
| Cajal-Retzius cell | seurat | 0.2 | 0.453 | 0.895 | 0.437 |
| Cajal-Retzius cell | seurat | 0.25 | 0.453 | 0.895 | 0.437 |
| Cajal-Retzius cell | seurat | 0.5 | 0.42 | 0.901 | 0.404 |
| Cajal-Retzius cell | seurat | 0.75 | 0.243 | 0.878 | 0.229 |
| Macrophage | scvi_knn | 0.0 | 0.212 | 0.131 | 0.752 |
| Macrophage | scvi_knn | 0.05 | 0.212 | 0.131 | 0.752 |
| Macrophage | scvi_knn | 0.1 | 0.212 | 0.131 | 0.752 |
| Macrophage | scvi_knn | 0.15 | 0.212 | 0.131 | 0.752 |
| Macrophage | scvi_knn | 0.2 | 0.212 | 0.131 | 0.752 |
| Macrophage | scvi_knn | 0.25 | 0.212 | 0.131 | 0.752 |
| Macrophage | scvi_knn | 0.5 | 0.204 | 0.127 | 0.719 |
| Macrophage | scvi_knn | 0.75 | 0.172 | 0.11 | 0.586 |
| Macrophage | scvi_rf | 0.0 | 0.235 | 0.143 | 0.874 |
| Macrophage | scvi_rf | 0.05 | 0.235 | 0.143 | 0.874 |
| Macrophage | scvi_rf | 0.1 | 0.235 | 0.143 | 0.874 |
| Macrophage | scvi_rf | 0.15 | 0.236 | 0.144 | 0.874 |
| Macrophage | scvi_rf | 0.2 | 0.237 | 0.145 | 0.865 |
| Macrophage | scvi_rf | 0.25 | 0.223 | 0.137 | 0.807 |
| Macrophage | scvi_rf | 0.5 | 0.149 | 0.098 | 0.495 |
| Macrophage | scvi_rf | 0.75 | 0.057 | 0.046 | 0.124 |
| Macrophage | seurat | 0.0 | 0.178 | 0.128 | 0.583 |
| Macrophage | seurat | 0.05 | 0.178 | 0.128 | 0.583 |
| Macrophage | seurat | 0.1 | 0.178 | 0.128 | 0.583 |
| Macrophage | seurat | 0.15 | 0.178 | 0.128 | 0.583 |
| Macrophage | seurat | 0.2 | 0.178 | 0.128 | 0.583 |
| Macrophage | seurat | 0.25 | 0.178 | 0.128 | 0.583 |
| Macrophage | seurat | 0.5 | 0.17 | 0.121 | 0.559 |
| Macrophage | seurat | 0.75 | 0.146 | 0.111 | 0.475 |
| Microglia | scvi_knn | 0.0 | 0.0 | nan | 0.0 |
| Microglia | scvi_knn | 0.05 | 0.0 | nan | 0.0 |
| Microglia | scvi_knn | 0.1 | 0.0 | nan | 0.0 |
| Microglia | scvi_knn | 0.15 | 0.0 | nan | 0.0 |
| Microglia | scvi_knn | 0.2 | 0.0 | nan | 0.0 |
| Microglia | scvi_knn | 0.25 | 0.0 | nan | 0.0 |
| Microglia | scvi_knn | 0.5 | 0.0 | nan | 0.0 |
| Microglia | scvi_knn | 0.75 | 0.0 | nan | 0.0 |
| Microglia | scvi_rf | 0.0 | 0.01 | 1.0 | 0.006 |
| Microglia | scvi_rf | 0.05 | 0.01 | 1.0 | 0.006 |
| Microglia | scvi_rf | 0.1 | 0.01 | 1.0 | 0.006 |
| Microglia | scvi_rf | 0.15 | 0.009 | 1.0 | 0.006 |
| Microglia | scvi_rf | 0.2 | 0.009 | 1.0 | 0.006 |
| Microglia | scvi_rf | 0.25 | 0.006 | 1.0 | 0.004 |
| Microglia | scvi_rf | 0.5 | 0.0 | nan | 0.0 |
| Microglia | scvi_rf | 0.75 | 0.0 | nan | 0.0 |
| Microglia | seurat | 0.0 | 0.456 | 0.951 | 0.439 |
| Microglia | seurat | 0.05 | 0.456 | 0.951 | 0.439 |
| Microglia | seurat | 0.1 | 0.456 | 0.951 | 0.439 |
| Microglia | seurat | 0.15 | 0.456 | 0.951 | 0.439 |
| Microglia | seurat | 0.2 | 0.456 | 0.951 | 0.439 |
| Microglia | seurat | 0.25 | 0.456 | 0.952 | 0.439 |
| Microglia | seurat | 0.5 | 0.453 | 0.954 | 0.435 |
| Microglia | seurat | 0.75 | 0.352 | 0.964 | 0.331 |
| Neural stem cell | scvi_knn | 0.0 | 0.515 | 0.515 | 0.617 |
| Neural stem cell | scvi_knn | 0.05 | 0.515 | 0.515 | 0.617 |
| Neural stem cell | scvi_knn | 0.1 | 0.515 | 0.515 | 0.617 |
| Neural stem cell | scvi_knn | 0.15 | 0.515 | 0.515 | 0.617 |
| Neural stem cell | scvi_knn | 0.2 | 0.516 | 0.517 | 0.617 |
| Neural stem cell | scvi_knn | 0.25 | 0.518 | 0.522 | 0.617 |
| Neural stem cell | scvi_knn | 0.5 | 0.499 | 0.608 | 0.524 |
| Neural stem cell | scvi_knn | 0.75 | 0.413 | 0.7 | 0.368 |
| Neural stem cell | scvi_rf | 0.0 | 0.386 | 0.288 | 0.748 |
| Neural stem cell | scvi_rf | 0.05 | 0.386 | 0.288 | 0.748 |
| Neural stem cell | scvi_rf | 0.1 | 0.386 | 0.288 | 0.748 |
| Neural stem cell | scvi_rf | 0.15 | 0.4 | 0.304 | 0.746 |
| Neural stem cell | scvi_rf | 0.2 | 0.43 | 0.346 | 0.726 |
| Neural stem cell | scvi_rf | 0.25 | 0.468 | 0.409 | 0.681 |
| Neural stem cell | scvi_rf | 0.5 | 0.39 | 0.727 | 0.335 |
| Neural stem cell | scvi_rf | 0.75 | 0.151 | 0.797 | 0.118 |
| Neural stem cell | seurat | 0.0 | 0.256 | 0.342 | 0.305 |
| Neural stem cell | seurat | 0.05 | 0.256 | 0.342 | 0.305 |
| Neural stem cell | seurat | 0.1 | 0.256 | 0.342 | 0.305 |
| Neural stem cell | seurat | 0.15 | 0.256 | 0.342 | 0.305 |
| Neural stem cell | seurat | 0.2 | 0.256 | 0.342 | 0.304 |
| Neural stem cell | seurat | 0.25 | 0.254 | 0.342 | 0.302 |
| Neural stem cell | seurat | 0.5 | 0.215 | 0.367 | 0.229 |
| Neural stem cell | seurat | 0.75 | 0.135 | 0.424 | 0.124 |
| OPC | scvi_knn | 0.0 | 0.262 | 0.902 | 0.218 |
| OPC | scvi_knn | 0.05 | 0.262 | 0.902 | 0.218 |
| OPC | scvi_knn | 0.1 | 0.262 | 0.902 | 0.218 |
| OPC | scvi_knn | 0.15 | 0.262 | 0.902 | 0.218 |
| OPC | scvi_knn | 0.2 | 0.262 | 0.902 | 0.218 |
| OPC | scvi_knn | 0.25 | 0.261 | 0.901 | 0.217 |
| OPC | scvi_knn | 0.5 | 0.175 | 0.849 | 0.148 |
| OPC | scvi_knn | 0.75 | 0.048 | 0.847 | 0.036 |
| OPC | scvi_rf | 0.0 | 0.06 | 0.996 | 0.044 |
| OPC | scvi_rf | 0.05 | 0.06 | 0.996 | 0.044 |
| OPC | scvi_rf | 0.1 | 0.06 | 0.996 | 0.044 |
| OPC | scvi_rf | 0.15 | 0.058 | 0.996 | 0.042 |
| OPC | scvi_rf | 0.2 | 0.05 | 0.994 | 0.037 |
| OPC | scvi_rf | 0.25 | 0.038 | 1.0 | 0.029 |
| OPC | scvi_rf | 0.5 | 0.0 | nan | 0.0 |
| OPC | scvi_rf | 0.75 | 0.0 | nan | 0.0 |
| OPC | seurat | 0.0 | 0.079 | 0.252 | 0.083 |
| OPC | seurat | 0.05 | 0.079 | 0.252 | 0.083 |
| OPC | seurat | 0.1 | 0.079 | 0.252 | 0.083 |
| OPC | seurat | 0.15 | 0.079 | 0.252 | 0.083 |
| OPC | seurat | 0.2 | 0.078 | 0.252 | 0.083 |
| OPC | seurat | 0.25 | 0.078 | 0.252 | 0.083 |
| OPC | seurat | 0.5 | 0.062 | 0.275 | 0.065 |
| OPC | seurat | 0.75 | 0.038 | 0.809 | 0.034 |

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

At global and family levels, scVI RF achieves the highest model-adjusted macro F1 (global: 0.939, family: 0.907), significantly outperforming Seurat (0.913, 0.886). However, in per-cell-type rankings, Seurat wins the best configuration for 5/8 family-level cell types (Astrocyte, GABAergic, Glutamatergic, Oligodendrocyte, Vascular) while scVI RF wins none. This discrepancy arises because scVI RF's macro advantage is driven by Glutamatergic and GABAergic neurons, which dominate cell counts and thus macro averaging, even though Seurat achieves higher F1 for these same types at their individually optimal configurations.

At class and subclass, scVI RF and Seurat are statistically indistinguishable (class: 0.815 vs 0.811, p=0.197; subclass: 0.801 vs 0.806, p=0.055), yet Seurat wins 5/9 class-level and 9/12 subclass-level cell types. scVI RF's macro EMM advantage at global/family thus does not reflect uniformly superior per-type performance.

Critically, scVI RF is far more cutoff-sensitive: at family, Glutamatergic drops from 0.788 → 0.047 and GABAergic from 0.820 → 0.126 by cutoff=0.75, compared to Seurat drops of 0.798 → 0.634 and 0.916 → 0.705. scVI kNN is intermediate. Any pipeline applying a confidence cutoff should strongly prefer Seurat or scVI kNN over scVI RF.

---

## Configuration Recommendation

**Date:** 2026-04-20  
**Goal:** Identify the best method, reference, cutoff, and subsample_ref for mouse brain cell-type annotation.

### Recommended Taxonomy Level: Family

Systematic failures (mean F1 < 0.5, ≥ 3 studies) at subclass and class: **OPC** (F1 = 0.131, 6 studies) and **Microglia** (F1 = 0.154, 6 studies). At family, Microglia collapses into CNS macrophage (F1 = 0.958), eliminating that failure. OPC remains a systematic failure at family (F1 = 0.131, 6 studies) due to near-zero reference support in most references — this is a coverage problem, not a model problem, and cannot be resolved by changing the classifier. Hippocampal subtypes (DG, CA1-ProS, CA3) collapse into Glutamatergic/GABAergic at family, which is appropriate for cortex-focused studies.

### Recommended Configuration

| Dimension | Recommended value | Rationale |
|-----------|------------------|-----------|
| Taxonomy level | **family** | Eliminates Microglia failure (→ CNS macrophage, F1=0.958); OPC remains unresolvable at all levels due to missing reference support |
| Method | **Seurat** (macro F1 alt: scVI RF) | Seurat wins 5/8 family-level cell types; scVI RF has higher macro EMM (0.907 vs 0.886) but wins 0/8 per-type and collapses at cutoff > 0.1. Use scVI RF only at cutoff=0 if macro F1 is the sole criterion. |
| Reference | **Cortical+Hipp. 10x** | Highest mean EMM (0.912); includes hippocampal types (DG, CA1-ProS, CA3). OPC and Neural stem cell absent from this reference but fail in all references regardless. |
| Cutoff | **0** | Any cutoff degrades scVI RF performance sharply; Seurat is more robust but cutoff=0 still maximises F1 across all methods. |
| Subsample_ref | **100** | Indistinguishable from 500 at family (EMM: 0.902 vs 0.901); Seurat ref processing is 2× faster at 100 vs 500. |

### Raw Performance: Seurat + Cortical+Hipp. 10x + cutoff=0 + subsample_ref=100

| key | macro_f1_mean | macro_precision_mean | macro_recall_mean |
|-----|--------------|---------------------|------------------|
| global | 0.897 | 0.866 | 0.976 |
| family | 0.908 | 0.894 | 0.956 |
| class | 0.828 | 0.797 | 0.948 |
| subclass | 0.830 | 0.802 | 0.948 |

### Compute Time: Seurat + subsample_ref=100

| step | mean_duration (hrs) | mean_memory (GB) |
|------|--------------------|--------------------|
| Ref Processing | 0.052 | 0.040 |
| Query Processing | 0.032 | 0.025 |
| Prediction | 0.019 | 0.021 |

Total per-query wall time: ~0.10 hrs.

### Trade-offs

scVI RF achieves 0.021 higher macro F1 at family (0.907 vs 0.886, significant) but wins zero individual cell types and is highly cutoff-sensitive; Seurat's per-type wins and robustness make it more reliable across diverse query datasets. scVI kNN (family macro EMM: 0.901) is a viable middle ground — outperforms Seurat on CNS macrophage and Neural stem cell while being more robust than scVI RF, at roughly half Seurat's compute cost.

### Pareto Note

Seurat + Cortical+Hipp. 10x + subsample_ref=100 appears in the Pareto-optimal table at family (mean_f1=0.912, 0.102 hrs) — the recommended configuration is Pareto-efficient.
