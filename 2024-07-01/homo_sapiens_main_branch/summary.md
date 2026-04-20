# Cell-Type Annotation Benchmarking: Results Summary (Old Pipeline)

> WARNING: Old pipeline results (scVI monolithic + Seurat). No ref_support=0 filtering. Per-cell-type cutoff sensitivity tables unavailable. Compare with new pipeline results before drawing conclusions.

Generated from: `/space/grp/rschwartz/rschwartz/evaluation_summary.nf/2024-07-01/homo_sapiens_main_branch/100/dataset_id/SCT/gap_false/`

---

## homo_sapiens_main_branch

**Organism:** homo_sapiens  
**Model formula:** `macro f1 ~ reference + method + cutoff + subsample ref + disease state + sex + method:cutoff + reference:method`  
**Pipeline:** old (scvi + seurat)

### Method Performance (model-adjusted marginal means)

| key | seurat | scvi |
| --- | --- | --- |
| global | 0.976 [0.971–0.980] | 0.989 [0.987–0.991] |
| family | 0.950 [0.942–0.956] | 0.973 [0.968–0.976] |
| class | 0.931 [0.906–0.949] | 0.950 [0.932–0.964] |
| subclass | 0.900 [0.860–0.930] | 0.929 [0.898–0.950] |

### Method Pairwise Contrasts

| key | contrast | odds.ratio | p.value |
| --- | --- | --- | --- |
| global | seurat / scvi | 0.45 | < 1e-300 |
| family | seurat / scvi | 0.532 | < 1e-300 |
| class | seurat / scvi | 0.709 | < 1e-289 |
| subclass | seurat / scvi | 0.696 | < 1e-300 |

### Cutoff Sensitivity (method × cutoff EMMs)

| key | cutoff | scvi | seurat |
| --- | --- | --- | --- |
| global | 0.0 | 0.983 | 0.976 |
| global | 0.05 | 0.979 | 0.974 |
| global | 0.1 | 0.973 | 0.972 |
| global | 0.15 | 0.965 | 0.97 |
| global | 0.2 | 0.956 | 0.968 |
| global | 0.25 | 0.944 | 0.966 |
| global | 0.5 | 0.827 | 0.951 |
| global | 0.75 | 0.577 | 0.931 |
| family | 0.0 | 0.961 | 0.95 |
| family | 0.05 | 0.953 | 0.947 |
| family | 0.1 | 0.944 | 0.943 |
| family | 0.15 | 0.934 | 0.94 |
| family | 0.2 | 0.921 | 0.936 |
| family | 0.25 | 0.907 | 0.933 |
| family | 0.5 | 0.794 | 0.91 |
| family | 0.75 | 0.606 | 0.881 |
| class | 0.0 | 0.934 | 0.914 |
| class | 0.05 | 0.925 | 0.911 |
| class | 0.1 | 0.916 | 0.909 |
| class | 0.15 | 0.905 | 0.906 |
| class | 0.2 | 0.892 | 0.903 |
| class | 0.25 | 0.879 | 0.9 |
| class | 0.5 | 0.787 | 0.883 |
| class | 0.75 | 0.654 | 0.863 |
| subclass | 0.0 | 0.909 | 0.885 |
| subclass | 0.05 | 0.898 | 0.882 |
| subclass | 0.1 | 0.886 | 0.879 |
| subclass | 0.15 | 0.873 | 0.875 |
| subclass | 0.2 | 0.859 | 0.872 |
| subclass | 0.25 | 0.844 | 0.868 |
| subclass | 0.5 | 0.745 | 0.849 |
| subclass | 0.75 | 0.613 | 0.827 |

### Reference × Method Performance

| key | ref_short | scvi | seurat |
| --- | --- | --- | --- |
| global | Dissection A1 | 0.984 | 0.978 |
| global | Dissection ACC | 0.982 | 0.977 |
| global | Dissection AnG | 0.985 | 0.978 |
| global | Dissection DFC | 0.966 | 0.955 |
| global | Dissection S1 | 0.985 | 0.977 |
| global | Dissection V1 | 0.979 | 0.979 |
| global | Human MC SMART-seq | 0.964 | 0.964 |
| global | SEA-AD DLPFC | 0.988 | 0.979 |
| global | SEA-AD MTG | 0.988 | 0.977 |
| global | Whole cortex | 0.989 | 0.976 |
| family | Dissection A1 | 0.959 | 0.955 |
| family | Dissection ACC | 0.963 | 0.953 |
| family | Dissection AnG | 0.962 | 0.955 |
| family | Dissection DFC | 0.904 | 0.816 |
| family | Dissection S1 | 0.964 | 0.953 |
| family | Dissection V1 | 0.956 | 0.958 |
| family | Human MC SMART-seq | 0.927 | 0.947 |
| family | SEA-AD DLPFC | 0.971 | 0.958 |
| family | SEA-AD MTG | 0.969 | 0.955 |
| family | Whole cortex | 0.973 | 0.95 |
| class | Dissection A1 | 0.941 | 0.926 |
| class | Dissection ACC | 0.944 | 0.928 |
| class | Dissection AnG | 0.945 | 0.93 |
| class | Dissection DFC | 0.909 | 0.864 |
| class | Dissection S1 | 0.944 | 0.926 |
| class | Dissection V1 | 0.932 | 0.932 |
| class | Human MC SMART-seq | 0.86 | 0.822 |
| class | SEA-AD DLPFC | 0.951 | 0.934 |
| class | SEA-AD MTG | 0.951 | 0.93 |
| class | Whole cortex | 0.95 | 0.931 |
| subclass | Dissection A1 | 0.913 | 0.895 |
| subclass | Dissection ACC | 0.918 | 0.899 |
| subclass | Dissection AnG | 0.919 | 0.902 |
| subclass | Dissection DFC | 0.872 | 0.824 |
| subclass | Dissection S1 | 0.914 | 0.892 |
| subclass | Dissection V1 | 0.91 | 0.895 |
| subclass | Human MC SMART-seq | 0.852 | 0.842 |
| subclass | SEA-AD DLPFC | 0.931 | 0.907 |
| subclass | SEA-AD MTG | 0.926 | 0.903 |
| subclass | Whole cortex | 0.929 | 0.9 |

### Reference Subsample Size

| key | subsample_ref | EMM |
| --- | --- | --- |
| global | 500 | 0.984 [0.980–0.986] |
| global | 100 | 0.986 [0.984–0.989] |
| global | 50 | 0.983 [0.979–0.986] |
| family | 500 | 0.963 [0.957–0.968] |
| family | 100 | 0.968 [0.963–0.972] |
| family | 50 | 0.964 [0.958–0.968] |
| class | 500 | 0.941 [0.920–0.957] |
| class | 100 | 0.943 [0.922–0.958] |
| class | 50 | 0.933 [0.909–0.951] |
| subclass | 500 | 0.916 [0.880–0.941] |
| subclass | 100 | 0.918 [0.883–0.943] |
| subclass | 50 | 0.907 [0.868–0.935] |

### Biological Covariates

**sex**

| key | sex | EMM |
| --- | --- | --- |
| global | female | 0.983 [0.979–0.986] |
| global | male | 0.984 [0.981–0.987] |
| family | female | 0.962 [0.957–0.967] |
| family | male | 0.963 [0.957–0.968] |
| class | female | 0.941 [0.920–0.957] |
| class | male | 0.941 [0.920–0.957] |
| subclass | female | 0.915 [0.879–0.941] |
| subclass | male | 0.916 [0.881–0.942] |

**disease_state**

| key | disease_state | EMM |
| --- | --- | --- |
| global | control | 0.982 [0.979–0.985] |
| global | disease | 0.985 [0.981–0.987] |
| family | control | 0.962 [0.956–0.967] |
| family | disease | 0.964 [0.958–0.968] |
| class | control | 0.937 [0.915–0.954] |
| class | disease | 0.945 [0.925–0.960] |
| subclass | control | 0.911 [0.874–0.938] |
| subclass | disease | 0.920 [0.887–0.944] |

### Between-Study Heterogeneity

**subclass — Well-classified (mean F1 ≥ 0.85)**

| key | label | n_studies | mean_f1 | std_f1 | mean_precision | mean_recall |
| --- | --- | --- | --- | --- | --- | --- |
| subclass | Glutamatergic | 1 | 0.992 | nan | 0.992 | 0.993 |
| subclass | Oligodendrocyte | 8 | 0.982 | 0.01 | 0.983 | 0.987 |
| subclass | Non-neuron | 6 | 0.975 | 0.027 | 1.0 | 0.972 |
| subclass | Astrocyte | 8 | 0.974 | 0.013 | 0.977 | 0.981 |
| subclass | L2/3-6 IT | 7 | 0.956 | 0.022 | 0.99 | 0.938 |
| subclass | OPC | 8 | 0.951 | 0.019 | 0.99 | 0.946 |
| subclass | Microglia | 8 | 0.936 | 0.034 | 0.966 | 0.943 |
| subclass | L5/6 NP | 7 | 0.929 | 0.059 | 1.0 | 0.925 |
| subclass | SST | 7 | 0.902 | 0.024 | 0.898 | 0.948 |
| subclass | LAMP5 | 7 | 0.896 | 0.033 | 0.971 | 0.881 |
| subclass | GABAergic | 2 | 0.891 | 0.123 | 0.958 | 0.906 |
| subclass | VIP | 7 | 0.891 | 0.033 | 0.946 | 0.89 |
| subclass | PVALB | 7 | 0.889 | 0.037 | 0.963 | 0.866 |
| subclass | Chandelier | 6 | 0.866 | 0.024 | 0.966 | 0.883 |
| subclass | Vascular | 7 | 0.86 | 0.115 | 0.986 | 0.863 |

**subclass — Hard / high-variance (mean F1 < 0.70 or std > 0.20)**

| key | label | n_studies | mean_f1 | std_f1 | mean_precision | mean_recall | failure_mode |
| --- | --- | --- | --- | --- | --- | --- | --- |
| subclass | Pericyte | 6 | 0.074 | 0.027 | 0.967 | 0.073 | Label escape |
| subclass | L5 ET | 7 | 0.574 | 0.322 | 0.89 | 0.553 | Study variance |
| subclass | SNCG | 7 | 0.653 | 0.086 | 0.79 | 0.711 | — |
| subclass | VLMC | 6 | 0.686 | 0.147 | 0.797 | 0.769 | — |

### Cell-Type Rankings (best config per label)

| key | label | method | reference | subsample_ref | mean_f1_across_studies | win_fraction | n_studies |
| --- | --- | --- | --- | --- | --- | --- | --- |
| global | GABAergic | scvi | Whole cortex | 500 | 0.988 | 0.375 | 8 |
| global | Glutamatergic | scvi | Whole cortex | 500 | 0.992 | 0.5 | 8 |
| global | Non-neuron | scvi | Whole cortex | 50 | 0.997 | 0.25 | 8 |
| family | Astrocyte | scvi | SEA-AD MTG | 100 | 0.988 | 0.25 | 8 |
| family | GABAergic | scvi | Whole cortex | 500 | 0.988 | 0.375 | 8 |
| family | Glutamatergic | scvi | Whole cortex | 500 | 0.992 | 0.5 | 8 |
| family | Microglia | scvi | Dissection AnG | 500 | 0.98 | 0.75 | 8 |
| family | Non-neuron | scvi | Dissection A1 | 100 | 1.0 | 1.0 | 6 |
| family | OPC | scvi | Dissection S1 | 500 | 0.991 | 0.75 | 8 |
| family | Oligodendrocyte | scvi | Dissection AnG | 500 | 0.993 | 0.375 | 8 |
| family | Vascular | scvi | Whole cortex | 100 | 0.972 | 0.375 | 8 |
| class | Astrocyte | scvi | SEA-AD MTG | 100 | 0.988 | 0.25 | 8 |
| class | Chandelier | scvi | Whole cortex | 100 | 0.976 | 0.5 | 6 |
| class | GABAergic | scvi | SEA-AD DLPFC | 500 | 0.999 | 1.0 | 2 |
| class | Glutamatergic | scvi | Dissection AnG | 500 | 0.999 | 1.0 | 1 |
| class | L2/3-6 IT | scvi | Dissection AnG | 500 | 0.986 | 0.0 | 7 |
| class | LAMP5 | seurat | Human MC SMART-seq | 100 | 0.94 | 0.0 | 7 |
| class | Microglia | scvi | Dissection AnG | 500 | 0.98 | 0.75 | 8 |
| class | Non-neuron | seurat | Dissection V1 | 500 | 1.0 | 1.0 | 6 |
| class | OPC | scvi | Dissection S1 | 500 | 0.99 | 0.625 | 8 |
| class | Oligodendrocyte | scvi | Dissection AnG | 500 | 0.993 | 0.5 | 8 |
| class | PAX6 | scvi | Whole cortex | 500 | 0.874 | 0.5 | 6 |
| class | PVALB | seurat | Dissection DFC | 50 | 0.918 | 0.286 | 7 |
| class | SNCG | scvi | Whole cortex | 50 | 0.766 | 0.0 | 7 |
| class | SST | seurat | Whole cortex | 100 | 0.931 | 0.143 | 7 |
| class | VIP | scvi | Dissection AnG | 500 | 0.93 | 0.0 | 7 |
| class | Vascular | scvi | Whole cortex | 100 | 0.971 | 0.375 | 8 |
| class | deep layer non-IT | seurat | Dissection A1 | 100 | 0.888 | 0.0 | 7 |
| subclass | Astrocyte | scvi | SEA-AD MTG | 100 | 0.988 | 0.25 | 8 |
| subclass | Chandelier | scvi | Whole cortex | 100 | 0.976 | 0.5 | 6 |
| subclass | Endothelial | scvi | SEA-AD DLPFC | 100 | 0.921 | 0.429 | 7 |
| subclass | GABAergic | scvi | SEA-AD DLPFC | 500 | 0.999 | 1.0 | 2 |
| subclass | Glutamatergic | scvi | Dissection AnG | 500 | 0.999 | 1.0 | 1 |
| subclass | L2/3-6 IT | scvi | SEA-AD DLPFC | 500 | 0.985 | 0.143 | 7 |
| subclass | L5 ET | seurat | Human MC SMART-seq | 100 | 0.709 | 0.571 | 7 |
| subclass | L5/6 NP | seurat | Human MC SMART-seq | 50 | 0.968 | 0.714 | 7 |
| subclass | L6 CT | seurat | Dissection ACC | 50 | 0.846 | 0.143 | 7 |
| subclass | L6b | seurat | Dissection A1 | 50 | 0.851 | 0.0 | 7 |
| subclass | LAMP5 | seurat | Human MC SMART-seq | 100 | 0.94 | 0.0 | 7 |
| subclass | Microglia | scvi | Dissection AnG | 500 | 0.98 | 0.75 | 8 |
| subclass | Non-neuron | seurat | Dissection S1 | 50 | 1.0 | 1.0 | 6 |
| subclass | OPC | scvi | Dissection S1 | 500 | 0.99 | 0.625 | 8 |
| subclass | Oligodendrocyte | scvi | Dissection AnG | 500 | 0.993 | 0.5 | 8 |
| subclass | PAX6 | scvi | Whole cortex | 500 | 0.874 | 0.5 | 6 |
| subclass | PVALB | seurat | Dissection DFC | 50 | 0.918 | 0.286 | 7 |
| subclass | Pericyte | seurat | Human MC SMART-seq | 50 | 0.819 | 0.833 | 6 |
| subclass | SNCG | scvi | Whole cortex | 50 | 0.766 | 0.0 | 7 |
| subclass | SST | seurat | Whole cortex | 100 | 0.931 | 0.143 | 7 |
| subclass | VIP | scvi | Dissection AnG | 500 | 0.93 | 0.0 | 7 |
| subclass | VLMC | seurat | Human MC SMART-seq | 50 | 0.928 | 1.0 | 6 |
| subclass | Vascular | scvi | Dissection A1 | 500 | 0.991 | 0.857 | 7 |

### Reference Cell-Type Coverage

**global**

| label | Dissection: Angular gyrus (AnG) | Dissection: Anterior cingulate cor… | Dissection: Dorsolateral prefronta… | Dissection: Primary auditory corte… | Dissection: Primary somatosensory … | Dissection: Primary visual cortex(… | Human MC SMART-seq | Whole Taxonomy - DLPFC: Seattle Al… | Whole Taxonomy - MTG: Seattle Alzh… | Whole cortex |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| GABAergic | 700 | 700 | 616 | 700 | 700 | 700 | 500 | 700 | 700 | 700 |
| Glutamatergic | 500 | 500 | 500 | 500 | 500 | 500 | 500 | 500 | 500 | 500 |
| Non-neuron | 586 | 600 | 40 | 580 | 600 | 592 | 513 | 600 | 600 | 632 |

**family**

| label | Dissection: Angular gyrus (AnG) | Dissection: Anterior cingulate cor… | Dissection: Dorsolateral prefronta… | Dissection: Primary auditory corte… | Dissection: Primary somatosensory … | Dissection: Primary visual cortex(… | Human MC SMART-seq | Whole Taxonomy - DLPFC: Seattle Al… | Whole Taxonomy - MTG: Seattle Alzh… | Whole cortex |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Astrocyte | 100 | 100 | 10 | 100 | 100 | 100 | 100 | 100 | 100 | 100 |
| GABAergic | 700 | 700 | 616 | 700 | 700 | 700 | 500 | 700 | 700 | 700 |
| Glutamatergic | 500 | 500 | 500 | 500 | 500 | 500 | 500 | 500 | 500 | 500 |
| Microglia | 100 | 100 | 8 | 100 | 100 | 100 | 100 | 100 | 100 | 100 |
| OPC | 100 | 100 | 3 | 100 | 100 | 100 | 100 | 100 | 100 | 100 |
| Oligodendrocyte | 100 | 100 | 13 | 100 | 100 | 100 | 100 | 100 | 100 | 100 |
| Vascular | 186 | 200 | 6 | 180 | 200 | 192 | 113 | 200 | 200 | 232 |

**class**

| label | Dissection: Angular gyrus (AnG) | Dissection: Anterior cingulate cor… | Dissection: Dorsolateral prefronta… | Dissection: Primary auditory corte… | Dissection: Primary somatosensory … | Dissection: Primary visual cortex(… | Human MC SMART-seq | Whole Taxonomy - DLPFC: Seattle Al… | Whole Taxonomy - MTG: Seattle Alzh… | Whole cortex |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Astrocyte | 100 | 100 | 10 | 100 | 100 | 100 | 100 | 100 | 100 | 100 |
| Chandelier | 100 | 100 | 100 | 100 | 100 | 100 | 0 | 100 | 100 | 100 |
| L2/3-6 IT | 100 | 100 | 100 | 100 | 100 | 100 | 100 | 100 | 100 | 100 |
| LAMP5 | 100 | 100 | 100 | 100 | 100 | 100 | 100 | 100 | 100 | 100 |
| Microglia | 100 | 100 | 8 | 100 | 100 | 100 | 100 | 100 | 100 | 100 |
| OPC | 100 | 100 | 3 | 100 | 100 | 100 | 100 | 100 | 100 | 100 |
| Oligodendrocyte | 100 | 100 | 13 | 100 | 100 | 100 | 100 | 100 | 100 | 100 |
| PAX6 | 100 | 100 | 66 | 100 | 100 | 100 | 100 | 100 | 100 | 100 |
| PVALB | 100 | 100 | 100 | 100 | 100 | 100 | 100 | 100 | 100 | 100 |
| SNCG | 100 | 100 | 50 | 100 | 100 | 100 | 0 | 100 | 100 | 100 |
| SST | 100 | 100 | 100 | 100 | 100 | 100 | 100 | 100 | 100 | 100 |
| VIP | 100 | 100 | 100 | 100 | 100 | 100 | 100 | 100 | 100 | 100 |
| Vascular | 186 | 200 | 6 | 180 | 200 | 192 | 113 | 200 | 200 | 232 |
| deep layer non-IT | 400 | 400 | 400 | 400 | 400 | 400 | 400 | 400 | 400 | 400 |

**subclass**

| label | Dissection: Angular gyrus (AnG) | Dissection: Anterior cingulate cor… | Dissection: Dorsolateral prefronta… | Dissection: Primary auditory corte… | Dissection: Primary somatosensory … | Dissection: Primary visual cortex(… | Human MC SMART-seq | Whole Taxonomy - DLPFC: Seattle Al… | Whole Taxonomy - MTG: Seattle Alzh… | Whole cortex |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Astrocyte | 100 | 100 | 10 | 100 | 100 | 100 | 100 | 100 | 100 | 100 |
| Chandelier | 100 | 100 | 100 | 100 | 100 | 100 | 0 | 100 | 100 | 100 |
| Endothelial | 97 | 100 | 3 | 94 | 100 | 100 | 70 | 100 | 100 | 100 |
| L2/3-6 IT | 100 | 100 | 100 | 100 | 100 | 100 | 100 | 100 | 100 | 100 |
| L5 ET | 100 | 100 | 100 | 100 | 100 | 100 | 100 | 100 | 100 | 100 |
| L5/6 NP | 100 | 100 | 100 | 100 | 100 | 100 | 100 | 100 | 100 | 100 |
| L6 CT | 100 | 100 | 100 | 100 | 100 | 100 | 100 | 100 | 100 | 100 |
| L6b | 100 | 100 | 100 | 100 | 100 | 100 | 100 | 100 | 100 | 100 |
| LAMP5 | 100 | 100 | 100 | 100 | 100 | 100 | 100 | 100 | 100 | 100 |
| Microglia | 100 | 100 | 8 | 100 | 100 | 100 | 100 | 100 | 100 | 100 |
| OPC | 100 | 100 | 3 | 100 | 100 | 100 | 100 | 100 | 100 | 100 |
| Oligodendrocyte | 100 | 100 | 13 | 100 | 100 | 100 | 100 | 100 | 100 | 100 |
| PAX6 | 100 | 100 | 66 | 100 | 100 | 100 | 100 | 100 | 100 | 100 |
| PVALB | 100 | 100 | 100 | 100 | 100 | 100 | 100 | 100 | 100 | 100 |
| Pericyte | 0 | 0 | 0 | 0 | 0 | 0 | 32 | 0 | 0 | 32 |
| SNCG | 100 | 100 | 50 | 100 | 100 | 100 | 0 | 100 | 100 | 100 |
| SST | 100 | 100 | 100 | 100 | 100 | 100 | 100 | 100 | 100 | 100 |
| VIP | 100 | 100 | 100 | 100 | 100 | 100 | 100 | 100 | 100 | 100 |
| VLMC | 89 | 100 | 3 | 86 | 100 | 92 | 11 | 100 | 100 | 100 |

### Pareto-Optimal Configurations

| key | method_display | reference | subsample_ref | mean_f1 | total_duration_hrs | total_memory_gb |
| --- | --- | --- | --- | --- | --- | --- |
| subclass | scVI | SEA-AD DLPFC | 500 | 0.868 | 0.04 | 0.02 |
| subclass | scVI | Whole cortex | 500 | 0.867 | 0.04 | 0.02 |
| subclass | scVI | SEA-AD DLPFC | 100 | 0.866 | 0.04 | 0.02 |
| subclass | scVI | Whole cortex | 50 | 0.86 | 0.039 | 0.02 |
| subclass | scVI | SEA-AD MTG | 50 | 0.856 | 0.039 | 0.02 |
| subclass | scVI | SEA-AD DLPFC | 50 | 0.854 | 0.039 | 0.02 |
| subclass | scVI | Dissection AnG | 50 | 0.849 | 0.039 | 0.02 |
| subclass | scVI | Dissection ACC | 50 | 0.847 | 0.039 | 0.02 |
| subclass | scVI | Dissection V1 | 50 | 0.843 | 0.039 | 0.02 |
| subclass | scVI | Dissection S1 | 50 | 0.839 | 0.039 | 0.02 |
| subclass | scVI | Dissection A1 | 50 | 0.836 | 0.039 | 0.02 |
| subclass | scVI | Dissection DFC | 50 | 0.762 | 0.039 | 0.02 |
| subclass | scVI | Human MC SMART-seq | 50 | 0.732 | 0.039 | 0.02 |
| class | scVI | SEA-AD DLPFC | 500 | 0.945 | 0.04 | 0.02 |
| class | scVI | SEA-AD MTG | 500 | 0.941 | 0.04 | 0.02 |
| class | scVI | SEA-AD MTG | 100 | 0.941 | 0.04 | 0.02 |
| class | scVI | SEA-AD DLPFC | 100 | 0.941 | 0.04 | 0.02 |
| class | scVI | SEA-AD MTG | 50 | 0.933 | 0.039 | 0.02 |
| class | scVI | Dissection S1 | 50 | 0.927 | 0.039 | 0.02 |
| class | scVI | Whole cortex | 50 | 0.927 | 0.039 | 0.02 |
| class | scVI | Dissection AnG | 50 | 0.924 | 0.039 | 0.02 |
| class | scVI | Dissection ACC | 50 | 0.922 | 0.039 | 0.02 |
| class | scVI | SEA-AD DLPFC | 50 | 0.921 | 0.039 | 0.02 |
| class | scVI | Dissection A1 | 50 | 0.917 | 0.039 | 0.02 |
| class | scVI | Dissection V1 | 50 | 0.907 | 0.039 | 0.02 |
| class | scVI | Dissection DFC | 50 | 0.881 | 0.039 | 0.02 |
| class | scVI | Human MC SMART-seq | 50 | 0.745 | 0.039 | 0.02 |
| family | scVI | SEA-AD DLPFC | 500 | 0.986 | 0.04 | 0.02 |
| family | scVI | Whole cortex | 500 | 0.985 | 0.04 | 0.02 |
| family | scVI | SEA-AD MTG | 500 | 0.985 | 0.04 | 0.02 |
| family | scVI | SEA-AD MTG | 100 | 0.984 | 0.04 | 0.02 |
| family | scVI | Whole cortex | 100 | 0.984 | 0.04 | 0.02 |
| family | scVI | Dissection S1 | 100 | 0.983 | 0.04 | 0.02 |
| family | scVI | SEA-AD DLPFC | 100 | 0.983 | 0.04 | 0.02 |
| family | scVI | Dissection A1 | 100 | 0.983 | 0.04 | 0.02 |
| family | scVI | Whole cortex | 50 | 0.982 | 0.039 | 0.02 |
| family | scVI | SEA-AD MTG | 50 | 0.981 | 0.039 | 0.02 |
| family | scVI | Dissection ACC | 50 | 0.98 | 0.039 | 0.02 |
| family | scVI | Dissection A1 | 50 | 0.98 | 0.039 | 0.02 |
| family | scVI | Dissection S1 | 50 | 0.978 | 0.039 | 0.02 |
| family | scVI | Dissection V1 | 50 | 0.977 | 0.039 | 0.02 |
| family | scVI | SEA-AD DLPFC | 50 | 0.975 | 0.039 | 0.02 |
| family | scVI | Dissection AnG | 50 | 0.975 | 0.039 | 0.02 |
| family | scVI | Dissection DFC | 50 | 0.932 | 0.039 | 0.02 |
| family | scVI | Human MC SMART-seq | 50 | 0.927 | 0.039 | 0.02 |
| global | scVI | Whole cortex | 500 | 0.992 | 0.04 | 0.02 |
| global | scVI | SEA-AD DLPFC | 500 | 0.992 | 0.04 | 0.02 |
| global | scVI | Whole cortex | 50 | 0.992 | 0.039 | 0.02 |
| global | scVI | SEA-AD MTG | 50 | 0.99 | 0.039 | 0.02 |
| global | scVI | Dissection A1 | 50 | 0.989 | 0.039 | 0.02 |
| global | scVI | SEA-AD DLPFC | 50 | 0.989 | 0.039 | 0.02 |
| global | scVI | Dissection S1 | 50 | 0.989 | 0.039 | 0.02 |
| global | scVI | Dissection ACC | 50 | 0.987 | 0.039 | 0.02 |
| global | scVI | Dissection AnG | 50 | 0.984 | 0.039 | 0.02 |
| global | scVI | Dissection V1 | 50 | 0.982 | 0.039 | 0.02 |
| global | scVI | Dissection DFC | 50 | 0.971 | 0.039 | 0.02 |
| global | scVI | Human MC SMART-seq | 50 | 0.953 | 0.039 | 0.02 |


---

## Macro F1 vs Per-Cell-Type F1 Conflict

scVI has higher marginal mean macro F1 at every taxonomy level (class: 0.950 vs 0.931, OR=0.709, p < 10^-289), but at the per-label level at class and subclass, Seurat is the best-performing config for several GABAergic/interneuron subtypes — Non-neuron (win fraction 1.0), SST (0.143), PVALB (0.286), LAMP5 (0.0 wins but highest mean F1), and deep layer non-IT (0.0 wins). scVI's macro F1 advantage is driven by majority cell types (excitatory neurons, oligodendrocytes, glia); for studies specifically focused on interneuron biology, Seurat may be preferable at these labels.

---

## Configuration Recommendation

### Recommended Taxonomy Level: **class**

At subclass, two cell types show systematic or near-systematic failures across studies:

- **Pericyte** (subclass): mean F1 = 0.074 ± 0.027 across 6 studies — **label escape** (absent from 8/10 references; 0 cells in all dissection references and SEA-AD atlases). Collapses to **Vascular** at class (mean F1 = 0.971).
- **L5 ET** (subclass): mean F1 = 0.574 ± 0.322 across 7 studies — extreme study variance (best config with Seurat still yields 0.709). Collapses to **deep layer non-IT** at class (mean F1 = 0.888).

At class, all labels have mean F1 ≥ 0.766 (SNCG). No label meets the systematic failure criterion (mean F1 < 0.5 in ≥ 3 studies). Class is the recommended level.

### Recommended Configuration

| Dimension | Recommended value | Rationale |
| --- | --- | --- |
| Taxonomy level | **class** | Subclass failures (Pericyte: F1=0.074, L5 ET: F1=0.574±0.322) resolve to F1 ≥ 0.888 at class |
| Method | **scVI** | EMM 0.950 vs 0.931 for Seurat (OR=0.709, p < 10^-289); note Seurat outperforms on interneuron subtypes (Non-neuron, SST, PVALB) |
| Reference | **SEA-AD DLPFC** | Highest class EMM with scVI (0.951); Human MC SMART-seq excluded (0 coverage for Chandelier, SNCG); Dissection DFC excluded (near-zero Microglia/OPC/Astrocyte coverage) |
| Cutoff | **0.0** | scVI degrades sharply with cutoff (class F1: 0.934 → 0.879 at cutoff=0.25 → 0.787 at cutoff=0.5); no hippocampal contamination concern for human |
| Subsample_ref | **100** | Class EMM 0.943 [0.922–0.958] vs 0.941 [0.920–0.957] at 500 — effectively identical; smaller subsample is more compute-efficient |

### Raw Performance — Recommended Configuration (scVI, Whole cortex, cutoff=0.0, subsample_ref=100)

| key | macro_f1_mean | macro_precision_mean | macro_recall_mean |
| --- | --- | --- | --- |
| global | 0.995 | 0.995 | 0.995 |
| family | 0.989 | 0.992 | 0.990 |
| class | 0.939 | 0.963 | 0.942 |
| subclass | 0.913 | 0.965 | 0.913 |

*Note: Whole cortex used here; SEA-AD DLPFC recommended for production use based on model-adjusted EMMs.*

### Trade-Off Narrative

scVI at cutoff=0 maximises recall but applies no confidence filter — if a downstream use case requires high-confidence labels, Seurat's robustness to higher cutoffs (class F1 = 0.883 at cutoff=0.5 vs scVI's 0.787) becomes an advantage. For interneuron-focused analyses (SST, PVALB, LAMP5, deep layer non-IT), Seurat produces higher per-type F1 at these labels despite lower overall macro F1.

### Pareto Note

The recommended reference (SEA-AD DLPFC, scVI, subsample_ref=500) is the top Pareto-optimal configuration at class (mean F1 = 0.945, total duration = 0.04 hrs, memory = 0.02 GB). Subsample_ref=100 is not separately listed but performs equivalently within model CIs.
