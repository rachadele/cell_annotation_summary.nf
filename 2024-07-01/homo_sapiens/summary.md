# Cell-Type Annotation Benchmarking: Results Summary


Generated from: `/space/grp/rschwartz/rschwartz/evaluation_summary.nf/2024-07-01/homo_sapiens/50/dataset_id/SCT/gap_false/`

---

## homo_sapiens

**Organism:** homo_sapiens  
**Model formula:** `macro f1 ~ reference + method + cutoff + subsample ref + disease state + sex + region match + method:cutoff + reference:method`  
**Pipeline:** new (scvi_rf + scvi_knn + seurat)

### Study Cohort

| study | disease | sex | Region | Samples | Cells | Subclasses |
| --- | --- | --- | --- | --- | --- | --- |
| CMC | control | nan | nan | 100 | 5000 | 20 |
| DevBrain | control | nan | nan | 16 | 800 | 20 |
| GSE144136 | control | nan | nan | 34 | 1571 | 12 |
| GSE157827 | reference subject role, Alzheimer disease | nan | prefrontal cortex | 21 | 1050 | 4 |
| GSE174332 | FTLD, control, ALS | male, female | primary motor cortex | 66 | 3300 | 14 |
| GSE180670 | control | nan | nan | 4 | 200 | 2 |
| GSE180928.1 | control | nan | brain | 12 | 530 | 9 |
| GSE211870 | control | male | adult prefrontal cortex | 4 | 200 | 4 |
| GSE237718 | control | nan | nan | 56 | 2800 | 4 |
| Ling-2024 | Affected, Unaffected | male, female | nan | 191 | 9550 | 15 |
| Mathys-2023 | control | nan | nan | 427 | 21350 | 19 |
| MultiomeBrain | control | nan | nan | 21 | 1050 | 18 |
| PTSDBrainomics | control | nan | nan | 19 | 950 | 20 |
| SZBDMulti-Seq | control | nan | nan | 72 | 3600 | 20 |
| UCLA-ASD | control | nan | nan | 51 | 2550 | 20 |
| Velmeshev-2019.1 | control | nan | nan | 10 | 500 | 17 |
| Velmeshev-2019.2 | control | nan | nan | 17 | 850 | 18 |

### Method Performance (model-adjusted marginal means)

| key | seurat | scvi_rf | scvi_knn |
| --- | --- | --- | --- |
| global | 0.974 [0.957–0.984] | 0.982 [0.970–0.989] | 0.942 [0.906–0.964] |
| family | 0.934 [0.855–0.972] | 0.948 [0.884–0.978] | 0.906 [0.801–0.959] |
| class | 0.830 [0.685–0.917] | 0.852 [0.719–0.928] | 0.791 [0.627–0.895] |
| subclass | 0.806 [0.639–0.908] | 0.827 [0.669–0.918] | 0.780 [0.601–0.893] |

### Method Pairwise Contrasts

| key | contrast | odds.ratio | p.value |
| --- | --- | --- | --- |
| global | seurat / scvi_rf | 0.704 | < 1e-300 |
| global | seurat / scvi_knn | 2.327 | < 1e-300 |
| global | scvi_rf / scvi_knn | 3.308 | < 1e-300 |
| family | seurat / scvi_rf | 0.776 | < 1e-300 |
| family | seurat / scvi_knn | 1.468 | < 1e-300 |
| family | scvi_rf / scvi_knn | 1.892 | < 1e-300 |
| class | seurat / scvi_rf | 0.85 | < 1e-300 |
| class | seurat / scvi_knn | 1.294 | < 1e-300 |
| class | scvi_rf / scvi_knn | 1.522 | < 1e-300 |
| subclass | seurat / scvi_rf | 0.873 | < 1e-300 |
| subclass | seurat / scvi_knn | 1.174 | < 1e-300 |
| subclass | scvi_rf / scvi_knn | 1.346 | < 1e-300 |

### Cutoff Sensitivity (method × cutoff EMMs)

| key | cutoff | scvi_knn | scvi_rf | seurat |
| --- | --- | --- | --- | --- |
| global | 0.0 | 0.942 | 0.982 | 0.974 |
| global | 0.05 | 0.941 | 0.978 | 0.972 |
| global | 0.1 | 0.939 | 0.975 | 0.971 |
| global | 0.15 | 0.938 | 0.97 | 0.969 |
| global | 0.2 | 0.936 | 0.965 | 0.967 |
| global | 0.25 | 0.935 | 0.959 | 0.964 |
| global | 0.5 | 0.927 | 0.91 | 0.951 |
| global | 0.75 | 0.918 | 0.815 | 0.932 |
| family | 0.0 | 0.906 | 0.948 | 0.934 |
| family | 0.05 | 0.904 | 0.942 | 0.931 |
| family | 0.1 | 0.901 | 0.934 | 0.927 |
| family | 0.15 | 0.898 | 0.926 | 0.923 |
| family | 0.2 | 0.895 | 0.917 | 0.919 |
| family | 0.25 | 0.892 | 0.907 | 0.915 |
| family | 0.5 | 0.877 | 0.839 | 0.89 |
| family | 0.75 | 0.859 | 0.736 | 0.86 |
| class | 0.0 | 0.791 | 0.852 | 0.83 |
| class | 0.05 | 0.789 | 0.842 | 0.826 |
| class | 0.1 | 0.788 | 0.831 | 0.822 |
| class | 0.15 | 0.787 | 0.82 | 0.817 |
| class | 0.2 | 0.786 | 0.809 | 0.813 |
| class | 0.25 | 0.784 | 0.797 | 0.809 |
| class | 0.5 | 0.778 | 0.728 | 0.785 |
| class | 0.75 | 0.771 | 0.646 | 0.76 |
| subclass | 0.0 | 0.78 | 0.827 | 0.806 |
| subclass | 0.05 | 0.778 | 0.816 | 0.802 |
| subclass | 0.1 | 0.776 | 0.806 | 0.798 |
| subclass | 0.15 | 0.774 | 0.794 | 0.794 |
| subclass | 0.2 | 0.772 | 0.783 | 0.79 |
| subclass | 0.25 | 0.77 | 0.77 | 0.786 |
| subclass | 0.5 | 0.76 | 0.702 | 0.763 |
| subclass | 0.75 | 0.749 | 0.623 | 0.739 |

### Reference × Method Performance

| key | ref_short | scvi_knn | scvi_rf | seurat |
| --- | --- | --- | --- | --- |
| global | Dissection A1 | 0.956 | 0.984 | 0.98 |
| global | Dissection ACC | 0.929 | 0.982 | 0.979 |
| global | Dissection AnG | 0.96 | 0.984 | 0.983 |
| global | Dissection DFC | 0.911 | 0.954 | 0.95 |
| global | Dissection S1 | 0.95 | 0.985 | 0.981 |
| global | Dissection V1 | 0.947 | 0.981 | 0.982 |
| global | Human MC SMART-seq | 0.914 | 0.974 | 0.944 |
| global | SEA-AD DLPFC | 0.936 | 0.987 | 0.979 |
| global | SEA-AD MTG | 0.952 | 0.987 | 0.979 |
| global | Single-nucleus transcriptome data … | 0.952 | 0.983 | 0.978 |
| global | Whole cortex | 0.935 | 0.985 | 0.952 |
| family | Dissection A1 | 0.933 | 0.959 | 0.957 |
| family | Dissection ACC | 0.914 | 0.958 | 0.955 |
| family | Dissection AnG | 0.937 | 0.963 | 0.959 |
| family | Dissection DFC | 0.722 | 0.854 | 0.703 |
| family | Dissection S1 | 0.929 | 0.961 | 0.957 |
| family | Dissection V1 | 0.925 | 0.954 | 0.958 |
| family | Human MC SMART-seq | 0.884 | 0.936 | 0.912 |
| family | SEA-AD DLPFC | 0.92 | 0.966 | 0.956 |
| family | SEA-AD MTG | 0.933 | 0.966 | 0.956 |
| family | Single-nucleus transcriptome data … | 0.89 | 0.913 | 0.884 |
| family | Whole cortex | 0.886 | 0.937 | 0.902 |
| class | Dissection A1 | 0.82 | 0.863 | 0.859 |
| class | Dissection ACC | 0.796 | 0.866 | 0.858 |
| class | Dissection AnG | 0.83 | 0.866 | 0.861 |
| class | Dissection DFC | 0.679 | 0.777 | 0.65 |
| class | Dissection S1 | 0.827 | 0.864 | 0.859 |
| class | Dissection V1 | 0.803 | 0.854 | 0.859 |
| class | Human MC SMART-seq | 0.699 | 0.823 | 0.745 |
| class | SEA-AD DLPFC | 0.809 | 0.882 | 0.867 |
| class | SEA-AD MTG | 0.827 | 0.878 | 0.867 |
| class | Single-nucleus transcriptome data … | 0.795 | 0.824 | 0.81 |
| class | Whole cortex | 0.775 | 0.848 | 0.819 |
| subclass | Dissection A1 | 0.81 | 0.835 | 0.832 |
| subclass | Dissection ACC | 0.784 | 0.842 | 0.834 |
| subclass | Dissection AnG | 0.818 | 0.84 | 0.837 |
| subclass | Dissection DFC | 0.672 | 0.754 | 0.632 |
| subclass | Dissection S1 | 0.809 | 0.837 | 0.831 |
| subclass | Dissection V1 | 0.792 | 0.833 | 0.832 |
| subclass | Human MC SMART-seq | 0.713 | 0.818 | 0.769 |
| subclass | SEA-AD DLPFC | 0.799 | 0.858 | 0.847 |
| subclass | SEA-AD MTG | 0.817 | 0.851 | 0.845 |
| subclass | Single-nucleus transcriptome data … | 0.782 | 0.794 | 0.776 |
| subclass | Whole cortex | 0.756 | 0.814 | 0.78 |

### Reference Subsample Size

| key | subsample_ref | EMM |
| --- | --- | --- |
| global | 500 | 0.972 [0.954–0.983] |
| global | 100 | 0.968 [0.947–0.980] |
| family | 500 | 0.930 [0.847–0.970] |
| family | 100 | 0.933 [0.852–0.971] |
| class | 500 | 0.829 [0.683–0.916] |
| class | 100 | 0.822 [0.673–0.912] |
| subclass | 500 | 0.808 [0.640–0.908] |
| subclass | 100 | 0.803 [0.633–0.906] |

### Biological Covariates

**sex**

| key | sex | EMM |
| --- | --- | --- |
| global | female | 0.972 [0.943–0.987] |
| global | male | 0.973 [0.945–0.987] |
| global | nan | 0.962 [0.947–0.973] |
| family | female | 0.923 [0.764–0.978] |
| family | male | 0.927 [0.776–0.979] |
| family | nan | 0.943 [0.907–0.966] |
| class | female | 0.809 [0.562–0.933] |
| class | male | 0.808 [0.561–0.933] |
| class | nan | 0.856 [0.784–0.907] |
| subclass | female | 0.779 [0.501–0.925] |
| subclass | male | 0.783 [0.507–0.927] |
| subclass | nan | 0.847 [0.766–0.904] |

**disease_state**

| key | disease_state | EMM |
| --- | --- | --- |
| global | control | 0.969 [0.949–0.981] |
| global | disease | 0.971 [0.952–0.982] |
| family | control | 0.928 [0.843–0.969] |
| family | disease | 0.935 [0.856–0.972] |
| class | control | 0.823 [0.674–0.913] |
| class | disease | 0.828 [0.681–0.915] |
| subclass | control | 0.808 [0.641–0.908] |
| subclass | disease | 0.803 [0.633–0.906] |

**region_match**

| key | region_match | EMM |
| --- | --- | --- |
| global | False | 0.967 [0.946–0.980] |
| global | True | 0.972 [0.954–0.983] |
| family | False | 0.914 [0.815–0.962] |
| family | True | 0.946 [0.879–0.977] |
| class | False | 0.852 [0.720–0.928] |
| class | True | 0.795 [0.633–0.897] |
| subclass | False | 0.822 [0.661–0.916] |
| subclass | True | 0.788 [0.611–0.897] |

### Between-Study Heterogeneity

**subclass — Well-classified (mean F1 ≥ 0.85)**

| key | label | n_studies | mean_f1 | std_f1 | mean_precision | mean_recall |
| --- | --- | --- | --- | --- | --- | --- |
| subclass | Chandelier | 10 | 0.974 | 0.029 | 0.977 | 0.987 |
| subclass | Astrocyte | 16 | 0.952 | 0.058 | 0.968 | 0.956 |
| subclass | Oligodendrocyte | 17 | 0.922 | 0.113 | 0.946 | 0.936 |
| subclass | VIP | 12 | 0.915 | 0.042 | 0.958 | 0.923 |
| subclass | L2/3-6 IT | 13 | 0.914 | 0.066 | 0.942 | 0.915 |
| subclass | SST | 13 | 0.898 | 0.132 | 0.941 | 0.914 |
| subclass | LAMP5 | 10 | 0.885 | 0.051 | 0.987 | 0.872 |
| subclass | Microglia | 16 | 0.884 | 0.06 | 0.957 | 0.888 |
| subclass | PAX6 | 8 | 0.876 | 0.101 | 0.923 | 0.91 |
| subclass | OPC | 16 | 0.87 | 0.24 | 0.921 | 0.872 |

**subclass — Hard / high-variance (mean F1 < 0.75 or std > 0.20)**

| key | label | n_studies | mean_f1 | std_f1 | mean_precision | mean_recall | failure_mode |
| --- | --- | --- | --- | --- | --- | --- | --- |
| subclass | L5 ET | 11 | 0.518 | 0.435 | 0.896 | 0.512 | Study variance |
| subclass | SMC | 7 | 0.68 | 0.165 | 0.957 | 0.703 | — |
| subclass | Endothelial | 14 | 0.698 | 0.301 | 0.982 | 0.695 | Study variance |
| subclass | VLMC | 6 | 0.707 | 0.137 | 0.848 | 0.772 | — |
| subclass | Pericyte | 9 | 0.719 | 0.13 | 0.948 | 0.73 | — |
| subclass | L6 CT | 11 | 0.764 | 0.261 | 0.88 | 0.764 | Study variance |
| subclass | L5/6 NP | 12 | 0.818 | 0.337 | 0.914 | 0.809 | Study variance |
| subclass | PVALB | 13 | 0.833 | 0.203 | 0.945 | 0.825 | Study variance |
| subclass | OPC | 16 | 0.87 | 0.24 | 0.921 | 0.872 | Study variance |

### Cell-Type Rankings (best config per label)

| key | label | method | reference | subsample_ref | mean_f1_across_studies | win_fraction | n_studies | mean_support |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| global | GABAergic | seurat | Dissection V1 | 100 | 0.972 | 0.312 | 16 | 8.00676953975703 |
| global | Glutamatergic | seurat | Dissection S1 | 500 | 0.976 | 0.438 | 16 | 20.3638255438186 |
| global | Non-neuron | seurat | Dissection AnG | 100 | 0.98 | 0.353 | 17 | 23.80842339147121 |
| family | Astrocyte | seurat | Single-nucleus transcriptome data … | 500 | 0.979 | 0.75 | 16 | 5.588300017279261 |
| family | GABAergic | seurat | Dissection V1 | 100 | 0.972 | 0.312 | 16 | 8.00676953975703 |
| family | Glutamatergic | seurat | Dissection S1 | 500 | 0.976 | 0.438 | 16 | 20.3638255438186 |
| family | Immune | scvi_knn | Single-nucleus transcriptome data … | 500 | 0.725 | 0.222 | 9 | 1.1610985632724764 |
| family | Microglia | scvi_rf | Dissection AnG | 500 | 0.956 | 0.375 | 16 | 2.742660932981272 |
| family | OPC | scvi_rf | Dissection S1 | 500 | 0.917 | 0.688 | 16 | 3.481878787956467 |
| family | Oligodendrocyte | scvi_knn | Single-nucleus transcriptome data … | 100 | 0.963 | 0.412 | 17 | 13.478734617380344 |
| family | Vascular | seurat | Single-nucleus transcriptome data … | 500 | 0.917 | 0.733 | 15 | 1.9154452546560283 |
| class | Astrocyte | seurat | Single-nucleus transcriptome data … | 500 | 0.978 | 0.688 | 16 | 5.588300017279261 |
| class | Chandelier | scvi_rf | Dissection AnG | 100 | 0.982 | 0.7 | 10 | 1.1218869365928188 |
| class | Immune | scvi_knn | Single-nucleus transcriptome data … | 500 | 0.722 | 0.222 | 9 | 1.1610985632724764 |
| class | L2/3-6 IT | seurat | Dissection AnG | 500 | 0.955 | 0.154 | 13 | 17.487360766394843 |
| class | LAMP5 | seurat | SEA-AD DLPFC | 100 | 0.951 | 0.3 | 10 | 1.7879067840694411 |
| class | Microglia | scvi_rf | Dissection AnG | 500 | 0.956 | 0.375 | 16 | 2.742660932981272 |
| class | OPC | scvi_rf | Dissection S1 | 500 | 0.916 | 0.625 | 16 | 3.481878787956467 |
| class | Oligodendrocyte | scvi_rf | SEA-AD DLPFC | 500 | 0.926 | 0.353 | 17 | 13.478734617380344 |
| class | PAX6 | seurat | Dissection S1 | 500 | 0.927 | 0.75 | 8 | 1.146355946684894 |
| class | PVALB | seurat | Dissection ACC | 100 | 0.873 | 0.231 | 13 | 2.3730777805693792 |
| class | SNCG | seurat | Dissection ACC | 500 | 0.862 | 0.444 | 9 | 1.2037360756832705 |
| class | SST | seurat | Whole cortex | 100 | 0.92 | 0.385 | 13 | 2.16202783090141 |
| class | VIP | seurat | Dissection DFC | 500 | 0.944 | 0.417 | 12 | 2.5145926436205808 |
| class | Vascular | seurat | Single-nucleus transcriptome data … | 500 | 0.917 | 0.8 | 15 | 1.9154452546560283 |
| class | deep layer non-IT | seurat | Dissection AnG | 500 | 0.836 | 0.308 | 13 | 2.948497139324551 |
| subclass | Astrocyte | seurat | Single-nucleus transcriptome data … | 500 | 0.978 | 0.688 | 16 | 5.588300017279261 |
| subclass | Chandelier | scvi_rf | Dissection AnG | 100 | 0.982 | 0.7 | 10 | 1.1218869365928188 |
| subclass | Endothelial | scvi_rf | SEA-AD DLPFC | 100 | 0.867 | 0.5 | 14 | 1.685776880271759 |
| subclass | L2/3-6 IT | seurat | Dissection A1 | 500 | 0.955 | 0.154 | 13 | 17.487360766394843 |
| subclass | L5 ET | scvi_knn | Dissection ACC | 100 | 0.675 | 0.545 | 11 | 1.8674045651318376 |
| subclass | L5/6 NP | seurat | SEA-AD DLPFC | 500 | 0.831 | 1.0 | 12 | 1.4098620823620822 |
| subclass | L6 CT | scvi_knn | Whole cortex | 500 | 0.818 | 0.091 | 11 | 1.4451763550883785 |
| subclass | L6b | seurat | Dissection AnG | 500 | 0.851 | 0.364 | 11 | 1.384934823753456 |
| subclass | LAMP5 | seurat | SEA-AD DLPFC | 100 | 0.951 | 0.3 | 10 | 1.7879067840694411 |
| subclass | Microglia | scvi_rf | Dissection AnG | 500 | 0.956 | 0.375 | 16 | 2.742660932981272 |
| subclass | OPC | scvi_rf | Dissection S1 | 500 | 0.916 | 0.625 | 16 | 3.481878787956467 |
| subclass | Oligodendrocyte | scvi_rf | SEA-AD DLPFC | 500 | 0.926 | 0.353 | 17 | 13.478734617380344 |
| subclass | PAX6 | seurat | Dissection S1 | 500 | 0.927 | 0.75 | 8 | 1.146355946684894 |
| subclass | PVALB | seurat | Dissection ACC | 100 | 0.873 | 0.231 | 13 | 2.3730777805693792 |
| subclass | Pericyte | scvi_knn | Single-nucleus transcriptome data … | 100 | 0.94 | 0.667 | 9 | 1.1770339697875931 |
| subclass | SMC | seurat | Single-nucleus transcriptome data … | 500 | 0.865 | 0.571 | 7 | 1.0404761904761906 |
| subclass | SNCG | seurat | Dissection ACC | 500 | 0.862 | 0.444 | 9 | 1.2037360756832705 |
| subclass | SST | seurat | Whole cortex | 100 | 0.92 | 0.385 | 13 | 2.16202783090141 |
| subclass | T Cell | scvi_rf | Whole cortex | 500 | 0.952 | 1.0 | 2 | 1.0869565217391304 |
| subclass | VIP | seurat | Dissection DFC | 500 | 0.944 | 0.417 | 12 | 2.5145926436205808 |
| subclass | VLMC | scvi_knn | Single-nucleus transcriptome data … | 100 | 0.98 | 0.667 | 6 | 1.0888888888888888 |

### Reference Cell-Type Coverage

**global**

| label | Dissection: Angular gyrus (AnG) | Dissection: Anterior cingulate cor… | Dissection: Dorsolateral prefronta… | Dissection: Primary auditory corte… | Dissection: Primary somatosensory … | Dissection: Primary visual cortex(… | Human MC SMART-seq | Single-nucleus transcriptome data … | Whole Taxonomy - DLPFC: Seattle Al… | Whole Taxonomy - MTG: Seattle Alzh… | Whole cortex |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| GABAergic | 700 | 700 | 616 | 700 | 700 | 700 | 500 | 500 | 700 | 700 | 700 |
| Glutamatergic | 500 | 500 | 500 | 500 | 500 | 500 | 500 | 500 | 500 | 500 | 500 |
| Non-neuron | 586 | 600 | 40 | 580 | 600 | 592 | 513 | 1234 | 600 | 600 | 1234 |

**family**

| label | Dissection: Angular gyrus (AnG) | Dissection: Anterior cingulate cor… | Dissection: Dorsolateral prefronta… | Dissection: Primary auditory corte… | Dissection: Primary somatosensory … | Dissection: Primary visual cortex(… | Human MC SMART-seq | Single-nucleus transcriptome data … | Whole Taxonomy - DLPFC: Seattle Al… | Whole Taxonomy - MTG: Seattle Alzh… | Whole cortex |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Astrocyte | 100 | 100 | 10 | 100 | 100 | 100 | 100 | 100 | 100 | 100 | 100 |
| GABAergic | 700 | 700 | 616 | 700 | 700 | 700 | 500 | 500 | 700 | 700 | 700 |
| Glutamatergic | 500 | 500 | 500 | 500 | 500 | 500 | 500 | 500 | 500 | 500 | 500 |
| Immune | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 334 | 0 | 0 | 334 |
| Microglia | 100 | 100 | 8 | 100 | 100 | 100 | 100 | 100 | 100 | 100 | 100 |
| OPC | 100 | 100 | 3 | 100 | 100 | 100 | 100 | 100 | 100 | 100 | 100 |
| Oligodendrocyte | 100 | 100 | 13 | 100 | 100 | 100 | 100 | 100 | 100 | 100 | 100 |
| Vascular | 186 | 200 | 6 | 180 | 200 | 192 | 113 | 500 | 200 | 200 | 500 |

**class**

| label | Dissection: Angular gyrus (AnG) | Dissection: Anterior cingulate cor… | Dissection: Dorsolateral prefronta… | Dissection: Primary auditory corte… | Dissection: Primary somatosensory … | Dissection: Primary visual cortex(… | Human MC SMART-seq | Single-nucleus transcriptome data … | Whole Taxonomy - DLPFC: Seattle Al… | Whole Taxonomy - MTG: Seattle Alzh… | Whole cortex |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Astrocyte | 100 | 100 | 10 | 100 | 100 | 100 | 100 | 100 | 100 | 100 | 100 |
| Chandelier | 100 | 100 | 100 | 100 | 100 | 100 | 0 | 100 | 100 | 100 | 100 |
| Immune | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 334 | 0 | 0 | 334 |
| L2/3-6 IT | 100 | 100 | 100 | 100 | 100 | 100 | 100 | 100 | 100 | 100 | 100 |
| LAMP5 | 100 | 100 | 100 | 100 | 100 | 100 | 100 | 100 | 100 | 100 | 100 |
| Microglia | 100 | 100 | 8 | 100 | 100 | 100 | 100 | 100 | 100 | 100 | 100 |
| OPC | 100 | 100 | 3 | 100 | 100 | 100 | 100 | 100 | 100 | 100 | 100 |
| Oligodendrocyte | 100 | 100 | 13 | 100 | 100 | 100 | 100 | 100 | 100 | 100 | 100 |
| PAX6 | 100 | 100 | 66 | 100 | 100 | 100 | 100 | 0 | 100 | 100 | 100 |
| PVALB | 100 | 100 | 100 | 100 | 100 | 100 | 100 | 100 | 100 | 100 | 100 |
| SNCG | 100 | 100 | 50 | 100 | 100 | 100 | 0 | 0 | 100 | 100 | 100 |
| SST | 100 | 100 | 100 | 100 | 100 | 100 | 100 | 100 | 100 | 100 | 100 |
| VIP | 100 | 100 | 100 | 100 | 100 | 100 | 100 | 100 | 100 | 100 | 100 |
| Vascular | 186 | 200 | 6 | 180 | 200 | 192 | 113 | 500 | 200 | 200 | 500 |
| deep layer non-IT | 400 | 400 | 400 | 400 | 400 | 400 | 400 | 400 | 400 | 400 | 400 |

**subclass**

| label | Dissection: Angular gyrus (AnG) | Dissection: Anterior cingulate cor… | Dissection: Dorsolateral prefronta… | Dissection: Primary auditory corte… | Dissection: Primary somatosensory … | Dissection: Primary visual cortex(… | Human MC SMART-seq | Single-nucleus transcriptome data … | Whole Taxonomy - DLPFC: Seattle Al… | Whole Taxonomy - MTG: Seattle Alzh… | Whole cortex |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Astrocyte | 100 | 100 | 10 | 100 | 100 | 100 | 100 | 100 | 100 | 100 | 100 |
| B Cell | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 53 | 0 | 0 | 53 |
| Chandelier | 100 | 100 | 100 | 100 | 100 | 100 | 0 | 100 | 100 | 100 | 100 |
| Endothelial | 97 | 100 | 3 | 94 | 100 | 100 | 70 | 100 | 100 | 100 | 100 |
| Erythroid | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 100 | 0 | 0 | 100 |
| L2/3-6 IT | 100 | 100 | 100 | 100 | 100 | 100 | 100 | 100 | 100 | 100 | 100 |
| L5 ET | 100 | 100 | 100 | 100 | 100 | 100 | 100 | 100 | 100 | 100 | 100 |
| L5/6 NP | 100 | 100 | 100 | 100 | 100 | 100 | 100 | 100 | 100 | 100 | 100 |
| L6 CT | 100 | 100 | 100 | 100 | 100 | 100 | 100 | 100 | 100 | 100 | 100 |
| L6b | 100 | 100 | 100 | 100 | 100 | 100 | 100 | 100 | 100 | 100 | 100 |
| LAMP5 | 100 | 100 | 100 | 100 | 100 | 100 | 100 | 100 | 100 | 100 | 100 |
| Macrophage | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 100 | 0 | 0 | 100 |
| Microglia | 100 | 100 | 8 | 100 | 100 | 100 | 100 | 100 | 100 | 100 | 100 |
| Myeloid | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 81 | 0 | 0 | 81 |
| OPC | 100 | 100 | 3 | 100 | 100 | 100 | 100 | 100 | 100 | 100 | 100 |
| Oligodendrocyte | 100 | 100 | 13 | 100 | 100 | 100 | 100 | 100 | 100 | 100 | 100 |
| PAX6 | 100 | 100 | 66 | 100 | 100 | 100 | 100 | 0 | 100 | 100 | 100 |
| PVALB | 100 | 100 | 100 | 100 | 100 | 100 | 100 | 100 | 100 | 100 | 100 |
| Pericyte | 0 | 0 | 0 | 0 | 0 | 0 | 32 | 100 | 0 | 0 | 100 |
| SMC | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 100 | 0 | 0 | 100 |
| SNCG | 100 | 100 | 50 | 100 | 100 | 100 | 0 | 0 | 100 | 100 | 100 |
| SST | 100 | 100 | 100 | 100 | 100 | 100 | 100 | 100 | 100 | 100 | 100 |
| T Cell | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 100 | 0 | 0 | 100 |
| VIP | 100 | 100 | 100 | 100 | 100 | 100 | 100 | 100 | 100 | 100 | 100 |
| VLMC | 89 | 100 | 3 | 86 | 100 | 92 | 11 | 100 | 100 | 100 | 100 |

### Per-Cell-Type Cutoff Sensitivity

### Global

| label | method | F1(0.0) | F1(0.25) | F1(0.5) | F1(0.75) |
| --- | --- | --- | --- | --- | --- |
| Glutamatergic | scvi_rf | 0.985 | 0.982 | 0.938 | 0.738 |
| GABAergic | scvi_rf | 0.983 | 0.978 | 0.917 | 0.79 |
| Non-neuron | scvi_rf | 0.983 | 0.979 | 0.944 | 0.839 |
| Non-neuron | seurat | 0.983 | 0.983 | 0.971 | 0.926 |
| Glutamatergic | seurat | 0.982 | 0.982 | 0.972 | 0.946 |
| GABAergic | seurat | 0.979 | 0.979 | 0.974 | 0.944 |
| GABAergic | scvi_knn | 0.969 | 0.969 | 0.969 | 0.936 |
| Glutamatergic | scvi_knn | 0.954 | 0.953 | 0.948 | 0.887 |
| Non-neuron | scvi_knn | 0.948 | 0.949 | 0.959 | 0.938 |

**Most cutoff-sensitive cell types (F1(0) → F1(0.75) drop):**

| method | label | F1(0.0) | F1(0.75) | Drop |
| --- | --- | --- | --- | --- |
| scvi_knn | Glutamatergic | 0.954 | 0.887 | 0.067 |
| scvi_knn | GABAergic | 0.969 | 0.936 | 0.033 |
| scvi_knn | Non-neuron | 0.948 | 0.938 | 0.01 |
| scvi_rf | Glutamatergic | 0.985 | 0.738 | 0.247 |
| scvi_rf | GABAergic | 0.983 | 0.79 | 0.193 |
| scvi_rf | Non-neuron | 0.983 | 0.839 | 0.144 |
| seurat | Non-neuron | 0.983 | 0.926 | 0.057 |
| seurat | Glutamatergic | 0.982 | 0.946 | 0.036 |
| seurat | GABAergic | 0.979 | 0.944 | 0.035 |

### Family

| label | method | F1(0.0) | F1(0.25) | F1(0.5) | F1(0.75) |
| --- | --- | --- | --- | --- | --- |
| Glutamatergic | scvi_rf | 0.985 | 0.982 | 0.938 | 0.738 |
| Oligodendrocyte | scvi_knn | 0.984 | 0.984 | 0.982 | 0.97 |
| GABAergic | scvi_rf | 0.983 | 0.978 | 0.917 | 0.79 |
| Glutamatergic | seurat | 0.982 | 0.982 | 0.972 | 0.946 |
| OPC | scvi_rf | 0.979 | 0.976 | 0.942 | 0.847 |
| GABAergic | seurat | 0.979 | 0.979 | 0.974 | 0.944 |
| Astrocyte | scvi_knn | 0.976 | 0.976 | 0.972 | 0.93 |
| Oligodendrocyte | scvi_rf | 0.975 | 0.974 | 0.949 | 0.84 |
| Astrocyte | scvi_rf | 0.974 | 0.972 | 0.933 | 0.831 |
| Astrocyte | seurat | 0.97 | 0.97 | 0.932 | 0.898 |
| GABAergic | scvi_knn | 0.969 | 0.969 | 0.969 | 0.936 |
| Oligodendrocyte | seurat | 0.969 | 0.969 | 0.957 | 0.933 |
| Glutamatergic | scvi_knn | 0.954 | 0.953 | 0.948 | 0.887 |
| OPC | scvi_knn | 0.94 | 0.94 | 0.906 | 0.898 |
| OPC | seurat | 0.936 | 0.936 | 0.928 | 0.857 |
| Microglia | scvi_rf | 0.916 | 0.917 | 0.867 | 0.679 |
| Microglia | scvi_knn | 0.91 | 0.909 | 0.907 | 0.773 |
| Vascular | seurat | 0.895 | 0.895 | 0.844 | 0.698 |
| Vascular | scvi_rf | 0.894 | 0.866 | 0.65 | 0.421 |
| Microglia | seurat | 0.887 | 0.887 | 0.842 | 0.758 |
| Vascular | scvi_knn | 0.826 | 0.826 | 0.796 | 0.671 |
| Immune | scvi_knn | 0.708 | 0.708 | 0.722 | 0.662 |
| Immune | scvi_rf | 0.689 | 0.665 | 0.629 | 0.469 |
| Immune | seurat | 0.387 | 0.387 | 0.359 | 0.342 |

**Most cutoff-sensitive cell types (F1(0) → F1(0.75) drop):**

| method | label | F1(0.0) | F1(0.75) | Drop |
| --- | --- | --- | --- | --- |
| scvi_knn | Vascular | 0.826 | 0.671 | 0.155 |
| scvi_knn | Microglia | 0.91 | 0.773 | 0.137 |
| scvi_knn | Glutamatergic | 0.954 | 0.887 | 0.067 |
| scvi_knn | Astrocyte | 0.976 | 0.93 | 0.046 |
| scvi_knn | Immune | 0.708 | 0.662 | 0.046 |
| scvi_rf | Vascular | 0.894 | 0.421 | 0.473 |
| scvi_rf | Glutamatergic | 0.985 | 0.738 | 0.247 |
| scvi_rf | Microglia | 0.916 | 0.679 | 0.237 |
| scvi_rf | Immune | 0.689 | 0.469 | 0.22 |
| scvi_rf | GABAergic | 0.983 | 0.79 | 0.193 |
| seurat | Vascular | 0.895 | 0.698 | 0.197 |
| seurat | Microglia | 0.887 | 0.758 | 0.129 |
| seurat | OPC | 0.936 | 0.857 | 0.079 |
| seurat | Astrocyte | 0.97 | 0.898 | 0.072 |
| seurat | Immune | 0.387 | 0.342 | 0.045 |

**Low-F1 cell types at cutoff=0 (F1 < 0.5) — precision/recall across all cutoffs:**

| label | method | cutoff | F1 | precision | recall |
| --- | --- | --- | --- | --- | --- |
| Immune | scvi_knn | 0.0 | 0.708 | 0.732 | 0.849 |
| Immune | scvi_knn | 0.05 | 0.708 | 0.732 | 0.849 |
| Immune | scvi_knn | 0.1 | 0.708 | 0.732 | 0.849 |
| Immune | scvi_knn | 0.15 | 0.708 | 0.732 | 0.849 |
| Immune | scvi_knn | 0.2 | 0.708 | 0.732 | 0.849 |
| Immune | scvi_knn | 0.25 | 0.708 | 0.732 | 0.849 |
| Immune | scvi_knn | 0.5 | 0.722 | 0.756 | 0.849 |
| Immune | scvi_knn | 0.75 | 0.662 | 0.848 | 0.697 |
| Immune | scvi_rf | 0.0 | 0.689 | 0.742 | 0.794 |
| Immune | scvi_rf | 0.05 | 0.689 | 0.742 | 0.794 |
| Immune | scvi_rf | 0.1 | 0.689 | 0.742 | 0.794 |
| Immune | scvi_rf | 0.15 | 0.687 | 0.741 | 0.791 |
| Immune | scvi_rf | 0.2 | 0.683 | 0.74 | 0.786 |
| Immune | scvi_rf | 0.25 | 0.665 | 0.74 | 0.765 |
| Immune | scvi_rf | 0.5 | 0.629 | 0.767 | 0.699 |
| Immune | scvi_rf | 0.75 | 0.469 | 0.952 | 0.463 |
| Immune | seurat | 0.0 | 0.387 | 0.497 | 0.481 |
| Immune | seurat | 0.05 | 0.387 | 0.497 | 0.481 |
| Immune | seurat | 0.1 | 0.387 | 0.497 | 0.481 |
| Immune | seurat | 0.15 | 0.387 | 0.497 | 0.481 |
| Immune | seurat | 0.2 | 0.387 | 0.497 | 0.481 |
| Immune | seurat | 0.25 | 0.387 | 0.497 | 0.481 |
| Immune | seurat | 0.5 | 0.359 | 0.383 | 0.448 |
| Immune | seurat | 0.75 | 0.342 | 0.492 | 0.373 |

### Class

| label | method | F1(0.0) | F1(0.25) | F1(0.5) | F1(0.75) |
| --- | --- | --- | --- | --- | --- |
| Chandelier | seurat | 0.987 | 0.987 | 0.988 | 0.948 |
| OPC | scvi_rf | 0.979 | 0.976 | 0.942 | 0.847 |
| Oligodendrocyte | scvi_rf | 0.975 | 0.974 | 0.949 | 0.84 |
| Astrocyte | scvi_rf | 0.974 | 0.972 | 0.933 | 0.831 |
| Oligodendrocyte | seurat | 0.969 | 0.969 | 0.957 | 0.933 |
| Astrocyte | scvi_knn | 0.969 | 0.969 | 0.969 | 0.929 |
| Chandelier | scvi_rf | 0.969 | 0.967 | 0.938 | 0.814 |
| Astrocyte | seurat | 0.969 | 0.969 | 0.932 | 0.897 |
| L2/3-6 IT | scvi_rf | 0.963 | 0.961 | 0.935 | 0.751 |
| Chandelier | scvi_knn | 0.963 | 0.963 | 0.96 | 0.957 |
| L2/3-6 IT | seurat | 0.945 | 0.945 | 0.937 | 0.916 |
| OPC | scvi_knn | 0.94 | 0.94 | 0.905 | 0.898 |
| OPC | seurat | 0.936 | 0.936 | 0.928 | 0.857 |
| PAX6 | seurat | 0.933 | 0.933 | 0.934 | 0.906 |
| Oligodendrocyte | scvi_knn | 0.929 | 0.929 | 0.955 | 0.957 |
| PVALB | seurat | 0.926 | 0.926 | 0.925 | 0.892 |
| PVALB | scvi_knn | 0.926 | 0.926 | 0.929 | 0.893 |
| VIP | scvi_rf | 0.921 | 0.919 | 0.874 | 0.743 |
| Microglia | scvi_rf | 0.916 | 0.917 | 0.867 | 0.679 |
| PVALB | scvi_rf | 0.914 | 0.912 | 0.878 | 0.743 |
| VIP | seurat | 0.913 | 0.913 | 0.92 | 0.878 |
| Microglia | scvi_knn | 0.909 | 0.909 | 0.907 | 0.773 |
| VIP | scvi_knn | 0.907 | 0.907 | 0.912 | 0.894 |
| SST | scvi_rf | 0.899 | 0.899 | 0.875 | 0.752 |
| SST | scvi_knn | 0.895 | 0.895 | 0.906 | 0.891 |
| Vascular | scvi_rf | 0.893 | 0.866 | 0.65 | 0.421 |
| PAX6 | scvi_rf | 0.892 | 0.892 | 0.792 | 0.612 |
| SST | seurat | 0.889 | 0.889 | 0.888 | 0.866 |
| L2/3-6 IT | scvi_knn | 0.888 | 0.888 | 0.923 | 0.879 |
| Vascular | seurat | 0.888 | 0.888 | 0.841 | 0.697 |
| PAX6 | scvi_knn | 0.887 | 0.887 | 0.884 | 0.789 |
| Microglia | seurat | 0.886 | 0.886 | 0.841 | 0.758 |
| LAMP5 | seurat | 0.858 | 0.858 | 0.871 | 0.836 |
| deep layer non-IT | seurat | 0.857 | 0.857 | 0.856 | 0.827 |
| LAMP5 | scvi_rf | 0.857 | 0.855 | 0.817 | 0.702 |
| deep layer non-IT | scvi_rf | 0.854 | 0.855 | 0.82 | 0.65 |
| LAMP5 | scvi_knn | 0.839 | 0.839 | 0.84 | 0.812 |
| Vascular | scvi_knn | 0.82 | 0.82 | 0.795 | 0.671 |
| deep layer non-IT | scvi_knn | 0.758 | 0.758 | 0.822 | 0.784 |
| SNCG | seurat | 0.753 | 0.753 | 0.765 | 0.673 |
| SNCG | scvi_knn | 0.738 | 0.738 | 0.739 | 0.671 |
| SNCG | scvi_rf | 0.727 | 0.721 | 0.638 | 0.469 |
| Immune | scvi_knn | 0.702 | 0.702 | 0.721 | 0.662 |
| Immune | scvi_rf | 0.687 | 0.665 | 0.629 | 0.469 |
| Immune | seurat | 0.386 | 0.386 | 0.359 | 0.342 |

**Most cutoff-sensitive cell types (F1(0) → F1(0.75) drop):**

| method | label | F1(0.0) | F1(0.75) | Drop |
| --- | --- | --- | --- | --- |
| scvi_knn | Vascular | 0.82 | 0.671 | 0.149 |
| scvi_knn | Microglia | 0.909 | 0.773 | 0.136 |
| scvi_knn | PAX6 | 0.887 | 0.789 | 0.098 |
| scvi_knn | SNCG | 0.738 | 0.671 | 0.067 |
| scvi_knn | OPC | 0.94 | 0.898 | 0.042 |
| scvi_rf | Vascular | 0.893 | 0.421 | 0.472 |
| scvi_rf | PAX6 | 0.892 | 0.612 | 0.28 |
| scvi_rf | SNCG | 0.727 | 0.469 | 0.258 |
| scvi_rf | Microglia | 0.916 | 0.679 | 0.237 |
| scvi_rf | Immune | 0.687 | 0.469 | 0.218 |
| seurat | Vascular | 0.888 | 0.697 | 0.191 |
| seurat | Microglia | 0.886 | 0.758 | 0.128 |
| seurat | SNCG | 0.753 | 0.673 | 0.08 |
| seurat | OPC | 0.936 | 0.857 | 0.079 |
| seurat | Astrocyte | 0.969 | 0.897 | 0.072 |

**Low-F1 cell types at cutoff=0 (F1 < 0.5) — precision/recall across all cutoffs:**

| label | method | cutoff | F1 | precision | recall |
| --- | --- | --- | --- | --- | --- |
| Immune | scvi_knn | 0.0 | 0.702 | 0.725 | 0.849 |
| Immune | scvi_knn | 0.05 | 0.702 | 0.725 | 0.849 |
| Immune | scvi_knn | 0.1 | 0.702 | 0.725 | 0.849 |
| Immune | scvi_knn | 0.15 | 0.702 | 0.725 | 0.849 |
| Immune | scvi_knn | 0.2 | 0.702 | 0.725 | 0.849 |
| Immune | scvi_knn | 0.25 | 0.702 | 0.725 | 0.849 |
| Immune | scvi_knn | 0.5 | 0.721 | 0.755 | 0.849 |
| Immune | scvi_knn | 0.75 | 0.662 | 0.848 | 0.697 |
| Immune | scvi_rf | 0.0 | 0.687 | 0.739 | 0.794 |
| Immune | scvi_rf | 0.05 | 0.687 | 0.739 | 0.794 |
| Immune | scvi_rf | 0.1 | 0.687 | 0.739 | 0.794 |
| Immune | scvi_rf | 0.15 | 0.685 | 0.739 | 0.791 |
| Immune | scvi_rf | 0.2 | 0.682 | 0.738 | 0.786 |
| Immune | scvi_rf | 0.25 | 0.665 | 0.74 | 0.765 |
| Immune | scvi_rf | 0.5 | 0.629 | 0.767 | 0.699 |
| Immune | scvi_rf | 0.75 | 0.469 | 0.952 | 0.463 |
| Immune | seurat | 0.0 | 0.386 | 0.469 | 0.481 |
| Immune | seurat | 0.05 | 0.386 | 0.469 | 0.481 |
| Immune | seurat | 0.1 | 0.386 | 0.469 | 0.481 |
| Immune | seurat | 0.15 | 0.386 | 0.469 | 0.481 |
| Immune | seurat | 0.2 | 0.386 | 0.469 | 0.481 |
| Immune | seurat | 0.25 | 0.386 | 0.469 | 0.481 |
| Immune | seurat | 0.5 | 0.359 | 0.377 | 0.448 |
| Immune | seurat | 0.75 | 0.342 | 0.492 | 0.373 |

### Subclass

| label | method | F1(0.0) | F1(0.25) | F1(0.5) | F1(0.75) |
| --- | --- | --- | --- | --- | --- |
| Chandelier | seurat | 0.987 | 0.987 | 0.988 | 0.948 |
| OPC | scvi_rf | 0.979 | 0.976 | 0.942 | 0.847 |
| Oligodendrocyte | scvi_rf | 0.975 | 0.974 | 0.949 | 0.84 |
| Astrocyte | scvi_rf | 0.974 | 0.972 | 0.933 | 0.831 |
| Oligodendrocyte | seurat | 0.969 | 0.969 | 0.957 | 0.933 |
| Astrocyte | scvi_knn | 0.969 | 0.969 | 0.969 | 0.929 |
| Astrocyte | seurat | 0.969 | 0.969 | 0.932 | 0.897 |
| Chandelier | scvi_rf | 0.967 | 0.966 | 0.938 | 0.814 |
| Chandelier | scvi_knn | 0.962 | 0.962 | 0.96 | 0.956 |
| L2/3-6 IT | scvi_rf | 0.962 | 0.961 | 0.935 | 0.751 |
| L2/3-6 IT | seurat | 0.945 | 0.945 | 0.936 | 0.916 |
| OPC | scvi_knn | 0.94 | 0.94 | 0.905 | 0.898 |
| OPC | seurat | 0.936 | 0.936 | 0.928 | 0.857 |
| PAX6 | seurat | 0.933 | 0.933 | 0.934 | 0.906 |
| Oligodendrocyte | scvi_knn | 0.929 | 0.929 | 0.955 | 0.957 |
| PVALB | seurat | 0.926 | 0.926 | 0.925 | 0.892 |
| PVALB | scvi_knn | 0.926 | 0.926 | 0.929 | 0.893 |
| VIP | scvi_rf | 0.921 | 0.919 | 0.874 | 0.743 |
| Microglia | scvi_rf | 0.916 | 0.917 | 0.867 | 0.679 |
| PVALB | scvi_rf | 0.914 | 0.912 | 0.878 | 0.743 |
| VIP | seurat | 0.913 | 0.913 | 0.92 | 0.878 |
| Microglia | scvi_knn | 0.909 | 0.909 | 0.907 | 0.773 |
| VIP | scvi_knn | 0.907 | 0.907 | 0.912 | 0.894 |
| L6b | seurat | 0.903 | 0.903 | 0.898 | 0.866 |
| SST | scvi_rf | 0.899 | 0.899 | 0.875 | 0.752 |
| SST | scvi_knn | 0.895 | 0.895 | 0.906 | 0.891 |
| PAX6 | scvi_rf | 0.892 | 0.892 | 0.792 | 0.612 |
| T Cell | scvi_rf | 0.891 | 0.891 | 0.881 | 0.721 |
| SST | seurat | 0.889 | 0.889 | 0.888 | 0.866 |
| PAX6 | scvi_knn | 0.887 | 0.887 | 0.884 | 0.789 |
| L2/3-6 IT | scvi_knn | 0.887 | 0.887 | 0.923 | 0.879 |
| Microglia | seurat | 0.886 | 0.886 | 0.841 | 0.758 |
| L6 CT | seurat | 0.886 | 0.886 | 0.875 | 0.823 |
| T Cell | scvi_knn | 0.881 | 0.881 | 0.881 | 0.881 |
| L6 CT | scvi_knn | 0.877 | 0.877 | 0.88 | 0.804 |
| SMC | scvi_rf | 0.862 | 0.846 | 0.667 | 0.394 |
| LAMP5 | seurat | 0.857 | 0.857 | 0.871 | 0.836 |
| LAMP5 | scvi_rf | 0.856 | 0.855 | 0.817 | 0.702 |
| L6b | scvi_rf | 0.843 | 0.842 | 0.789 | 0.592 |
| L6 CT | scvi_rf | 0.839 | 0.838 | 0.763 | 0.567 |
| LAMP5 | scvi_knn | 0.837 | 0.837 | 0.839 | 0.812 |
| L6b | scvi_knn | 0.833 | 0.833 | 0.838 | 0.735 |
| L5/6 NP | seurat | 0.816 | 0.816 | 0.817 | 0.807 |
| L5/6 NP | scvi_knn | 0.811 | 0.811 | 0.822 | 0.808 |
| L5/6 NP | scvi_rf | 0.804 | 0.802 | 0.801 | 0.766 |
| Endothelial | scvi_rf | 0.772 | 0.758 | 0.636 | 0.466 |
| VLMC | seurat | 0.761 | 0.762 | 0.771 | 0.649 |
| Pericyte | scvi_rf | 0.761 | 0.72 | 0.314 | 0.088 |
| SMC | scvi_knn | 0.76 | 0.762 | 0.744 | 0.589 |
| SNCG | seurat | 0.753 | 0.753 | 0.765 | 0.673 |
| Pericyte | scvi_knn | 0.747 | 0.747 | 0.61 | 0.359 |
| SNCG | scvi_knn | 0.738 | 0.738 | 0.739 | 0.671 |
| SNCG | scvi_rf | 0.727 | 0.721 | 0.638 | 0.469 |
| Endothelial | seurat | 0.727 | 0.727 | 0.696 | 0.606 |
| Endothelial | scvi_knn | 0.703 | 0.703 | 0.686 | 0.638 |
| Pericyte | seurat | 0.693 | 0.693 | 0.697 | 0.581 |
| VLMC | scvi_rf | 0.65 | 0.626 | 0.293 | 0.14 |
| SMC | seurat | 0.627 | 0.627 | 0.705 | 0.48 |
| VLMC | scvi_knn | 0.598 | 0.598 | 0.552 | 0.407 |
| T Cell | seurat | 0.45 | 0.45 | 0.441 | 0.422 |
| L5 ET | seurat | 0.418 | 0.417 | 0.419 | 0.375 |
| L5 ET | scvi_knn | 0.408 | 0.408 | 0.424 | 0.351 |
| L5 ET | scvi_rf | 0.38 | 0.376 | 0.393 | 0.287 |

**Most cutoff-sensitive cell types (F1(0) → F1(0.75) drop):**

| method | label | F1(0.0) | F1(0.75) | Drop |
| --- | --- | --- | --- | --- |
| scvi_knn | Pericyte | 0.747 | 0.359 | 0.388 |
| scvi_knn | VLMC | 0.598 | 0.407 | 0.191 |
| scvi_knn | SMC | 0.76 | 0.589 | 0.171 |
| scvi_knn | Microglia | 0.909 | 0.773 | 0.136 |
| scvi_knn | PAX6 | 0.887 | 0.789 | 0.098 |
| scvi_rf | Pericyte | 0.761 | 0.088 | 0.673 |
| scvi_rf | VLMC | 0.65 | 0.14 | 0.51 |
| scvi_rf | SMC | 0.862 | 0.394 | 0.468 |
| scvi_rf | Endothelial | 0.772 | 0.466 | 0.306 |
| scvi_rf | PAX6 | 0.892 | 0.612 | 0.28 |
| seurat | SMC | 0.627 | 0.48 | 0.147 |
| seurat | Microglia | 0.886 | 0.758 | 0.128 |
| seurat | Endothelial | 0.727 | 0.606 | 0.121 |
| seurat | VLMC | 0.761 | 0.649 | 0.112 |
| seurat | Pericyte | 0.693 | 0.581 | 0.112 |

**Low-F1 cell types at cutoff=0 (F1 < 0.5) — precision/recall across all cutoffs:**

| label | method | cutoff | F1 | precision | recall |
| --- | --- | --- | --- | --- | --- |
| L5 ET | scvi_knn | 0.0 | 0.408 | 0.964 | 0.391 |
| L5 ET | scvi_knn | 0.05 | 0.408 | 0.964 | 0.391 |
| L5 ET | scvi_knn | 0.1 | 0.408 | 0.964 | 0.391 |
| L5 ET | scvi_knn | 0.15 | 0.408 | 0.964 | 0.391 |
| L5 ET | scvi_knn | 0.2 | 0.408 | 0.964 | 0.391 |
| L5 ET | scvi_knn | 0.25 | 0.408 | 0.964 | 0.39 |
| L5 ET | scvi_knn | 0.5 | 0.424 | 0.97 | 0.412 |
| L5 ET | scvi_knn | 0.75 | 0.351 | 0.967 | 0.342 |
| L5 ET | scvi_rf | 0.0 | 0.38 | 0.949 | 0.369 |
| L5 ET | scvi_rf | 0.05 | 0.38 | 0.949 | 0.369 |
| L5 ET | scvi_rf | 0.1 | 0.38 | 0.949 | 0.369 |
| L5 ET | scvi_rf | 0.15 | 0.38 | 0.949 | 0.369 |
| L5 ET | scvi_rf | 0.2 | 0.38 | 0.951 | 0.368 |
| L5 ET | scvi_rf | 0.25 | 0.376 | 0.954 | 0.365 |
| L5 ET | scvi_rf | 0.5 | 0.393 | 0.975 | 0.385 |
| L5 ET | scvi_rf | 0.75 | 0.287 | 0.99 | 0.278 |
| L5 ET | seurat | 0.0 | 0.418 | 0.961 | 0.402 |
| L5 ET | seurat | 0.05 | 0.418 | 0.961 | 0.402 |
| L5 ET | seurat | 0.1 | 0.418 | 0.961 | 0.402 |
| L5 ET | seurat | 0.15 | 0.418 | 0.961 | 0.402 |
| L5 ET | seurat | 0.2 | 0.418 | 0.961 | 0.402 |
| L5 ET | seurat | 0.25 | 0.417 | 0.961 | 0.402 |
| L5 ET | seurat | 0.5 | 0.419 | 0.961 | 0.406 |
| L5 ET | seurat | 0.75 | 0.375 | 0.955 | 0.364 |
| T Cell | scvi_knn | 0.0 | 0.881 | 1.0 | 0.877 |
| T Cell | scvi_knn | 0.05 | 0.881 | 1.0 | 0.877 |
| T Cell | scvi_knn | 0.1 | 0.881 | 1.0 | 0.877 |
| T Cell | scvi_knn | 0.15 | 0.881 | 1.0 | 0.877 |
| T Cell | scvi_knn | 0.2 | 0.881 | 1.0 | 0.877 |
| T Cell | scvi_knn | 0.25 | 0.881 | 1.0 | 0.877 |
| T Cell | scvi_knn | 0.5 | 0.881 | 1.0 | 0.877 |
| T Cell | scvi_knn | 0.75 | 0.881 | 1.0 | 0.877 |
| T Cell | scvi_rf | 0.0 | 0.891 | 1.0 | 0.886 |
| T Cell | scvi_rf | 0.05 | 0.891 | 1.0 | 0.886 |
| T Cell | scvi_rf | 0.1 | 0.891 | 1.0 | 0.886 |
| T Cell | scvi_rf | 0.15 | 0.891 | 1.0 | 0.886 |
| T Cell | scvi_rf | 0.2 | 0.891 | 1.0 | 0.886 |
| T Cell | scvi_rf | 0.25 | 0.891 | 1.0 | 0.886 |
| T Cell | scvi_rf | 0.5 | 0.881 | 1.0 | 0.877 |
| T Cell | scvi_rf | 0.75 | 0.721 | 1.0 | 0.71 |
| T Cell | seurat | 0.0 | 0.45 | 1.0 | 0.448 |
| T Cell | seurat | 0.05 | 0.45 | 1.0 | 0.448 |
| T Cell | seurat | 0.1 | 0.45 | 1.0 | 0.448 |
| T Cell | seurat | 0.15 | 0.45 | 1.0 | 0.448 |
| T Cell | seurat | 0.2 | 0.45 | 1.0 | 0.448 |
| T Cell | seurat | 0.25 | 0.45 | 1.0 | 0.448 |
| T Cell | seurat | 0.5 | 0.441 | 1.0 | 0.438 |
| T Cell | seurat | 0.75 | 0.422 | 1.0 | 0.42 |

### Pareto-Optimal Configurations

| key | method_display | reference | subsample_ref | mean_f1 | total_duration_hrs | total_memory_gb |
| --- | --- | --- | --- | --- | --- | --- |
| subclass | Seurat | SEA-AD DLPFC | 500 | 0.876 | 0.132 | 0.031 |
| subclass | Seurat | SEA-AD DLPFC | 100 | 0.876 | 0.097 | 0.039 |
| subclass | Seurat | SEA-AD MTG | 100 | 0.875 | 0.097 | 0.039 |
| subclass | Seurat | Dissection AnG | 100 | 0.87 | 0.097 | 0.039 |
| subclass | Seurat | Dissection ACC | 100 | 0.865 | 0.097 | 0.039 |
| subclass | scVI kNN | SEA-AD DLPFC | 500 | 0.861 | 0.064 | 0.021 |
| subclass | scVI kNN | SEA-AD MTG | 500 | 0.86 | 0.064 | 0.021 |
| subclass | scVI RF | SEA-AD DLPFC | 500 | 0.858 | 0.064 | 0.021 |
| subclass | scVI kNN | Dissection S1 | 500 | 0.857 | 0.064 | 0.021 |
| subclass | scVI RF | SEA-AD MTG | 500 | 0.856 | 0.064 | 0.021 |
| subclass | scVI RF | Dissection AnG | 500 | 0.855 | 0.064 | 0.021 |
| subclass | scVI RF | Dissection S1 | 500 | 0.851 | 0.064 | 0.021 |
| subclass | scVI kNN | Whole cortex | 500 | 0.848 | 0.064 | 0.021 |
| subclass | scVI RF | Dissection A1 | 500 | 0.846 | 0.064 | 0.021 |
| subclass | scVI RF | Dissection ACC | 500 | 0.844 | 0.064 | 0.021 |
| subclass | scVI kNN | Single-nucleus transcriptome data … | 500 | 0.84 | 0.064 | 0.021 |
| subclass | scVI kNN | Dissection AnG | 500 | 0.84 | 0.064 | 0.021 |
| subclass | scVI RF | Single-nucleus transcriptome data … | 500 | 0.839 | 0.064 | 0.021 |
| subclass | scVI kNN | Dissection A1 | 500 | 0.838 | 0.064 | 0.021 |
| subclass | scVI kNN | Dissection ACC | 500 | 0.837 | 0.064 | 0.021 |
| subclass | scVI RF | Whole cortex | 500 | 0.836 | 0.064 | 0.021 |
| subclass | scVI RF | Dissection V1 | 500 | 0.833 | 0.064 | 0.021 |
| subclass | scVI kNN | Dissection V1 | 500 | 0.827 | 0.064 | 0.021 |
| subclass | scVI RF | Human MC SMART-seq | 500 | 0.769 | 0.064 | 0.021 |
| subclass | scVI kNN | Human MC SMART-seq | 500 | 0.744 | 0.064 | 0.021 |
| subclass | scVI RF | Dissection DFC | 500 | 0.735 | 0.064 | 0.021 |
| subclass | scVI kNN | Dissection DFC | 500 | 0.707 | 0.064 | 0.021 |
| class | Seurat | SEA-AD DLPFC | 100 | 0.917 | 0.097 | 0.039 |
| class | Seurat | SEA-AD MTG | 100 | 0.912 | 0.097 | 0.039 |
| class | scVI RF | SEA-AD MTG | 500 | 0.905 | 0.064 | 0.021 |
| class | scVI RF | SEA-AD DLPFC | 500 | 0.905 | 0.064 | 0.021 |
| class | scVI kNN | SEA-AD MTG | 500 | 0.902 | 0.064 | 0.021 |
| class | scVI RF | Dissection AnG | 500 | 0.901 | 0.064 | 0.021 |
| class | scVI kNN | Dissection S1 | 500 | 0.901 | 0.064 | 0.021 |
| class | scVI kNN | SEA-AD DLPFC | 500 | 0.901 | 0.064 | 0.021 |
| class | scVI RF | Dissection S1 | 500 | 0.9 | 0.064 | 0.021 |
| class | scVI kNN | Dissection AnG | 500 | 0.898 | 0.064 | 0.021 |
| class | scVI RF | Dissection ACC | 500 | 0.897 | 0.064 | 0.021 |
| class | scVI RF | Dissection A1 | 500 | 0.896 | 0.064 | 0.021 |
| class | scVI kNN | Dissection A1 | 500 | 0.892 | 0.064 | 0.021 |
| class | scVI kNN | Dissection V1 | 500 | 0.888 | 0.064 | 0.021 |
| class | scVI RF | Dissection V1 | 500 | 0.887 | 0.064 | 0.021 |
| class | scVI kNN | Dissection ACC | 500 | 0.885 | 0.064 | 0.021 |
| class | scVI RF | Whole cortex | 500 | 0.879 | 0.064 | 0.021 |
| class | scVI kNN | Single-nucleus transcriptome data … | 500 | 0.874 | 0.064 | 0.021 |
| class | scVI kNN | Whole cortex | 500 | 0.871 | 0.064 | 0.021 |
| class | scVI RF | Single-nucleus transcriptome data … | 500 | 0.861 | 0.064 | 0.021 |
| class | scVI RF | Human MC SMART-seq | 500 | 0.854 | 0.064 | 0.021 |
| class | scVI kNN | Human MC SMART-seq | 500 | 0.828 | 0.064 | 0.021 |
| class | scVI RF | Dissection DFC | 500 | 0.805 | 0.064 | 0.021 |
| class | scVI kNN | Dissection DFC | 500 | 0.768 | 0.064 | 0.021 |
| family | Seurat | SEA-AD MTG | 100 | 0.948 | 0.097 | 0.039 |
| family | Seurat | Dissection A1 | 100 | 0.947 | 0.097 | 0.039 |
| family | scVI RF | Dissection AnG | 500 | 0.946 | 0.064 | 0.021 |
| family | scVI kNN | SEA-AD MTG | 500 | 0.945 | 0.064 | 0.021 |
| family | scVI kNN | SEA-AD DLPFC | 500 | 0.944 | 0.064 | 0.021 |
| family | scVI RF | SEA-AD DLPFC | 500 | 0.943 | 0.064 | 0.021 |
| family | scVI RF | Dissection S1 | 500 | 0.941 | 0.064 | 0.021 |
| family | scVI kNN | Dissection S1 | 500 | 0.941 | 0.064 | 0.021 |
| family | scVI RF | SEA-AD MTG | 500 | 0.94 | 0.064 | 0.021 |
| family | scVI kNN | Dissection A1 | 500 | 0.94 | 0.064 | 0.021 |
| family | scVI RF | Dissection ACC | 500 | 0.939 | 0.064 | 0.021 |
| family | scVI kNN | Dissection AnG | 500 | 0.939 | 0.064 | 0.021 |
| family | scVI RF | Dissection A1 | 500 | 0.936 | 0.064 | 0.021 |
| family | scVI kNN | Dissection V1 | 500 | 0.933 | 0.064 | 0.021 |
| family | scVI RF | Dissection V1 | 500 | 0.932 | 0.064 | 0.021 |
| family | scVI kNN | Dissection ACC | 500 | 0.932 | 0.064 | 0.021 |
| family | scVI RF | Human MC SMART-seq | 500 | 0.91 | 0.064 | 0.021 |
| family | scVI RF | Whole cortex | 500 | 0.892 | 0.064 | 0.021 |
| family | scVI kNN | Single-nucleus transcriptome data … | 500 | 0.889 | 0.064 | 0.021 |
| family | scVI kNN | Human MC SMART-seq | 500 | 0.885 | 0.064 | 0.021 |
| family | scVI kNN | Whole cortex | 500 | 0.88 | 0.064 | 0.021 |
| family | scVI RF | Single-nucleus transcriptome data … | 500 | 0.869 | 0.064 | 0.021 |
| family | scVI RF | Dissection DFC | 500 | 0.8 | 0.064 | 0.021 |
| family | scVI kNN | Dissection DFC | 500 | 0.689 | 0.064 | 0.021 |
| global | Seurat | Dissection A1 | 100 | 0.976 | 0.097 | 0.039 |
| global | Seurat | Dissection V1 | 100 | 0.975 | 0.097 | 0.039 |
| global | Seurat | Dissection AnG | 100 | 0.974 | 0.097 | 0.039 |
| global | Seurat | Dissection S1 | 100 | 0.974 | 0.097 | 0.039 |
| global | Seurat | SEA-AD DLPFC | 100 | 0.972 | 0.097 | 0.039 |
| global | Seurat | SEA-AD MTG | 100 | 0.972 | 0.097 | 0.039 |
| global | Seurat | Dissection ACC | 100 | 0.972 | 0.097 | 0.039 |
| global | scVI RF | Human MC SMART-seq | 100 | 0.97 | 0.071 | 0.021 |
| global | scVI RF | Human MC SMART-seq | 500 | 0.969 | 0.064 | 0.021 |
| global | scVI kNN | Dissection V1 | 500 | 0.968 | 0.064 | 0.021 |
| global | scVI RF | Single-nucleus transcriptome data … | 500 | 0.968 | 0.064 | 0.021 |
| global | scVI RF | Whole cortex | 500 | 0.968 | 0.064 | 0.021 |
| global | scVI kNN | Dissection A1 | 500 | 0.968 | 0.064 | 0.021 |
| global | scVI kNN | Dissection S1 | 500 | 0.968 | 0.064 | 0.021 |
| global | scVI RF | SEA-AD DLPFC | 500 | 0.965 | 0.064 | 0.021 |
| global | scVI RF | Dissection ACC | 500 | 0.965 | 0.064 | 0.021 |
| global | scVI kNN | Dissection AnG | 500 | 0.964 | 0.064 | 0.021 |
| global | scVI RF | Dissection AnG | 500 | 0.963 | 0.064 | 0.021 |
| global | scVI kNN | SEA-AD MTG | 500 | 0.963 | 0.064 | 0.021 |
| global | scVI RF | SEA-AD MTG | 500 | 0.958 | 0.064 | 0.021 |
| global | scVI RF | Dissection V1 | 500 | 0.958 | 0.064 | 0.021 |
| global | scVI kNN | Single-nucleus transcriptome data … | 500 | 0.958 | 0.064 | 0.021 |
| global | scVI kNN | SEA-AD DLPFC | 500 | 0.956 | 0.064 | 0.021 |
| global | scVI kNN | Whole cortex | 500 | 0.955 | 0.064 | 0.021 |
| global | scVI RF | Dissection S1 | 500 | 0.954 | 0.064 | 0.021 |
| global | scVI RF | Dissection A1 | 500 | 0.953 | 0.064 | 0.021 |
| global | scVI kNN | Dissection ACC | 500 | 0.945 | 0.064 | 0.021 |
| global | scVI kNN | Human MC SMART-seq | 500 | 0.927 | 0.064 | 0.021 |
| global | scVI RF | Dissection DFC | 500 | 0.895 | 0.064 | 0.021 |
| global | scVI kNN | Dissection DFC | 500 | 0.884 | 0.064 | 0.021 |

### Computational Time

| method | step | subsample_ref | mean_duration | mean_memory |
| --- | --- | --- | --- | --- |
| scVI RF/kNN | Query Processing | 100 | 0.019 | 0.021 |
| scVI RF/kNN | Query Processing | 500 | 0.019 | 0.021 |
| Seurat | Ref Processing | 100 | 0.038 | 0.039 |
| Seurat | Ref Processing | 500 | 0.075 | 0.031 |
| scVI RF/kNN | Prediction | 100 | 0.026 | 0.013 |
| scVI RF/kNN | Prediction | 500 | 0.02 | 0.013 |
| scVI RF/kNN | Embedding | 100 | 0.026 | 0.016 |
| scVI RF/kNN | Embedding | 500 | 0.025 | 0.016 |
| Seurat | Prediction | 100 | 0.035 | 0.018 |
| Seurat | Prediction | 500 | 0.034 | 0.021 |
| Seurat | Query Processing | 100 | 0.024 | 0.03 |
| Seurat | Query Processing | 500 | 0.024 | 0.03 |


---
