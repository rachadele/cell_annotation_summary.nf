# Cell-Type Annotation Benchmarking: Results Summary (Old Pipeline)

> WARNING: Old pipeline results (scVI monolithic + Seurat). No ref_support=0 filtering. Per-cell-type cutoff sensitivity tables unavailable. Compare with new pipeline results before drawing conclusions.

Generated from: `/space/grp/rschwartz/rschwartz/evaluation_summary.nf/2024-07-01/homo_sapiens_new/100/dataset_id/SCT/gap_false/`

---

## homo_sapiens_new

**Organism:** homo_sapiens  
**Model formula:** `macro f1 ~ reference + method + cutoff + subsample ref + disease state + sex + region match + method:cutoff + reference:method`  
**Pipeline:** old (scvi + seurat)

### Study Cohort

| study | disease | sex | Region | Samples | Cells | Subclasses |
| --- | --- | --- | --- | --- | --- | --- |
| CMC | control | nan | nan | 100 | 10000 | 21 |
| DevBrain | control | nan | nan | 16 | 1600 | 21 |
| GSE180670 | control | nan | nan | 4 | 400 | 2 |
| GSE237718 | control | nan | nan | 56 | 5600 | 7 |
| Ling-2024 | Affected, Unaffected | male, female | nan | 191 | 19100 | 16 |
| MultiomeBrain | control | nan | nan | 21 | 2061 | 21 |
| PTSDBrainomics | control | nan | nan | 19 | 1900 | 21 |
| SZBDMulti-Seq | control | nan | nan | 72 | 7194 | 21 |
| UCLA-ASD | control | nan | nan | 51 | 5100 | 21 |

### Method Performance (model-adjusted marginal means)

| key | seurat | scvi |
| --- | --- | --- |
| global | 0.972 [0.946–0.986] | 0.986 [0.973–0.993] |
| family | 0.939 [0.809–0.982] | 0.971 [0.901–0.992] |
| class | 0.906 [0.742–0.970] | 0.941 [0.826–0.981] |
| subclass | 0.882 [0.703–0.959] | 0.923 [0.792–0.974] |

### Method Pairwise Contrasts

| key | contrast | odds.ratio | p.value |
| --- | --- | --- | --- |
| global | seurat / scvi | 0.488 | < 1e-300 |
| family | seurat / scvi | 0.463 | < 1e-300 |
| class | seurat / scvi | 0.606 | < 1e-300 |
| subclass | seurat / scvi | 0.62 | < 1e-300 |

### Cutoff Sensitivity (method × cutoff EMMs)

| key | cutoff | scvi | seurat |
| --- | --- | --- | --- |
| global | 0.0 | 0.98 | 0.973 |
| global | 0.05 | 0.975 | 0.971 |
| global | 0.1 | 0.968 | 0.968 |
| global | 0.15 | 0.959 | 0.966 |
| global | 0.2 | 0.948 | 0.963 |
| global | 0.25 | 0.934 | 0.96 |
| global | 0.5 | 0.803 | 0.943 |
| global | 0.75 | 0.54 | 0.918 |
| family | 0.0 | 0.96 | 0.941 |
| family | 0.05 | 0.951 | 0.937 |
| family | 0.1 | 0.941 | 0.933 |
| family | 0.15 | 0.929 | 0.929 |
| family | 0.2 | 0.914 | 0.925 |
| family | 0.25 | 0.897 | 0.92 |
| family | 0.5 | 0.759 | 0.893 |
| family | 0.75 | 0.534 | 0.857 |
| class | 0.0 | 0.924 | 0.889 |
| class | 0.05 | 0.913 | 0.885 |
| class | 0.1 | 0.9 | 0.882 |
| class | 0.15 | 0.886 | 0.878 |
| class | 0.2 | 0.87 | 0.874 |
| class | 0.25 | 0.852 | 0.87 |
| class | 0.5 | 0.733 | 0.848 |
| class | 0.75 | 0.565 | 0.823 |
| subclass | 0.0 | 0.905 | 0.869 |
| subclass | 0.05 | 0.892 | 0.865 |
| subclass | 0.1 | 0.879 | 0.861 |
| subclass | 0.15 | 0.863 | 0.857 |
| subclass | 0.2 | 0.846 | 0.853 |
| subclass | 0.25 | 0.828 | 0.848 |
| subclass | 0.5 | 0.709 | 0.825 |
| subclass | 0.75 | 0.551 | 0.798 |

### Reference × Method Performance

| key | ref_short | scvi | seurat |
| --- | --- | --- | --- |
| global | Dissection A1 | 0.981 | 0.975 |
| global | Dissection ACC | 0.979 | 0.973 |
| global | Dissection AnG | 0.981 | 0.975 |
| global | Dissection DFC | 0.958 | 0.948 |
| global | Dissection S1 | 0.982 | 0.974 |
| global | Dissection V1 | 0.974 | 0.975 |
| global | Human MC SMART-seq | 0.959 | 0.958 |
| global | SEA-AD DLPFC | 0.985 | 0.975 |
| global | SEA-AD MTG | 0.985 | 0.972 |
| global | Whole cortex | 0.986 | 0.972 |
| family | Dissection A1 | 0.958 | 0.945 |
| family | Dissection ACC | 0.96 | 0.943 |
| family | Dissection AnG | 0.961 | 0.945 |
| family | Dissection DFC | 0.889 | 0.772 |
| family | Dissection S1 | 0.961 | 0.942 |
| family | Dissection V1 | 0.952 | 0.948 |
| family | Human MC SMART-seq | 0.921 | 0.933 |
| family | SEA-AD DLPFC | 0.969 | 0.949 |
| family | SEA-AD MTG | 0.967 | 0.945 |
| family | Whole cortex | 0.971 | 0.939 |
| class | Dissection A1 | 0.93 | 0.901 |
| class | Dissection ACC | 0.931 | 0.902 |
| class | Dissection AnG | 0.933 | 0.905 |
| class | Dissection DFC | 0.882 | 0.81 |
| class | Dissection S1 | 0.933 | 0.9 |
| class | Dissection V1 | 0.919 | 0.908 |
| class | Human MC SMART-seq | 0.84 | 0.773 |
| class | SEA-AD DLPFC | 0.942 | 0.911 |
| class | SEA-AD MTG | 0.941 | 0.906 |
| class | Whole cortex | 0.941 | 0.906 |
| subclass | Dissection A1 | 0.907 | 0.875 |
| subclass | Dissection ACC | 0.91 | 0.879 |
| subclass | Dissection AnG | 0.911 | 0.883 |
| subclass | Dissection DFC | 0.854 | 0.785 |
| subclass | Dissection S1 | 0.908 | 0.872 |
| subclass | Dissection V1 | 0.902 | 0.877 |
| subclass | Human MC SMART-seq | 0.842 | 0.815 |
| subclass | SEA-AD DLPFC | 0.926 | 0.889 |
| subclass | SEA-AD MTG | 0.92 | 0.884 |
| subclass | Whole cortex | 0.923 | 0.882 |

### Reference Subsample Size

| key | subsample_ref | EMM |
| --- | --- | --- |
| global | 500 | 0.980 [0.962–0.990] |
| global | 100 | 0.984 [0.969–0.992] |
| global | 50 | 0.981 [0.962–0.990] |
| family | 500 | 0.958 [0.861–0.988] |
| family | 100 | 0.965 [0.885–0.990] |
| family | 50 | 0.963 [0.877–0.989] |
| class | 500 | 0.925 [0.787–0.976] |
| class | 100 | 0.929 [0.797–0.978] |
| class | 50 | 0.923 [0.782–0.976] |
| subclass | 500 | 0.904 [0.750–0.968] |
| subclass | 100 | 0.910 [0.762–0.969] |
| subclass | 50 | 0.904 [0.750–0.968] |

### Biological Covariates

**sex**

| key | sex | EMM |
| --- | --- | --- |
| global | nan | 0.979 [0.967–0.986] |
| global | female | 0.981 [0.936–0.995] |
| global | male | 0.983 [0.940–0.995] |
| family | nan | 0.944 [0.878–0.976] |
| family | female | 0.967 [0.719–0.997] |
| family | male | 0.969 [0.731–0.997] |
| class | nan | 0.915 [0.828–0.960] |
| class | female | 0.933 [0.590–0.993] |
| class | male | 0.934 [0.594–0.993] |
| subclass | nan | 0.884 [0.781–0.943] |
| subclass | female | 0.919 [0.566–0.990] |
| subclass | male | 0.924 [0.581–0.991] |

**disease_state**

| key | disease_state | EMM |
| --- | --- | --- |
| global | control | 0.979 [0.967–0.986] |
| global | disease | 0.982 [0.938–0.995] |
| family | control | 0.944 [0.878–0.976] |
| family | disease | 0.968 [0.725–0.997] |
| class | control | 0.915 [0.828–0.960] |
| class | disease | 0.934 [0.592–0.993] |
| subclass | control | 0.884 [0.781–0.943] |
| subclass | disease | 0.921 [0.574–0.990] |

### Between-Study Heterogeneity

**subclass — Well-classified (mean F1 ≥ 0.85)**

| key | label | n_studies | mean_f1 | std_f1 | mean_precision | mean_recall |
| --- | --- | --- | --- | --- | --- | --- |
| subclass | Glutamatergic | 1 | 0.992 | nan | 0.992 | 0.993 |
| subclass | Non-neuron | 6 | 0.975 | 0.027 | 1.0 | 0.972 |
| subclass | Astrocyte | 8 | 0.974 | 0.013 | 0.977 | 0.981 |
| subclass | L2/3-6 IT | 7 | 0.956 | 0.022 | 0.99 | 0.938 |
| subclass | Microglia | 8 | 0.936 | 0.034 | 0.966 | 0.943 |
| subclass | Oligodendrocyte | 9 | 0.929 | 0.16 | 0.97 | 0.923 |
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
| subclass | OPC | 9 | 0.849 | 0.307 | 0.894 | 0.846 | Study variance |

### Cell-Type Rankings (best config per label)

| key | label | method | reference | subsample_ref | mean_f1_across_studies | win_fraction | n_studies |
| --- | --- | --- | --- | --- | --- | --- | --- |
| global | GABAergic | scvi | Whole cortex | 500 | 0.988 | 0.375 | 8 |
| global | Glutamatergic | scvi | Whole cortex | 500 | 0.992 | 0.5 | 8 |
| global | Non-neuron | scvi | Whole cortex | 50 | 0.975 | 0.222 | 9 |
| family | Astrocyte | scvi | SEA-AD MTG | 100 | 0.988 | 0.25 | 8 |
| family | GABAergic | scvi | Whole cortex | 500 | 0.988 | 0.375 | 8 |
| family | Glutamatergic | scvi | Whole cortex | 500 | 0.992 | 0.5 | 8 |
| family | Microglia | scvi | Dissection AnG | 500 | 0.98 | 0.75 | 8 |
| family | Non-neuron | scvi | SEA-AD DLPFC | 500 | 1.0 | 1.0 | 6 |
| family | OPC | scvi | Dissection S1 | 500 | 0.889 | 0.667 | 9 |
| family | Oligodendrocyte | scvi | Human MC SMART-seq | 50 | 0.974 | 0.0 | 9 |
| family | Vascular | scvi | Whole cortex | 100 | 0.972 | 0.375 | 8 |
| class | Astrocyte | scvi | SEA-AD MTG | 100 | 0.988 | 0.25 | 8 |
| class | Chandelier | scvi | Whole cortex | 100 | 0.976 | 0.5 | 6 |
| class | GABAergic | scvi | SEA-AD DLPFC | 500 | 0.999 | 1.0 | 2 |
| class | Glutamatergic | scvi | Dissection AnG | 500 | 0.999 | 1.0 | 1 |
| class | L2/3-6 IT | scvi | Dissection AnG | 500 | 0.986 | 0.0 | 7 |
| class | LAMP5 | seurat | Human MC SMART-seq | 100 | 0.94 | 0.0 | 7 |
| class | Microglia | scvi | Dissection AnG | 500 | 0.98 | 0.75 | 8 |
| class | Non-neuron | scvi | SEA-AD MTG | 50 | 1.0 | 1.0 | 6 |
| class | OPC | scvi | Dissection S1 | 500 | 0.887 | 0.556 | 9 |
| class | Oligodendrocyte | scvi | Dissection AnG | 500 | 0.944 | 0.444 | 9 |
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
| subclass | Non-neuron | seurat | Dissection ACC | 500 | 1.0 | 1.0 | 6 |
| subclass | OPC | scvi | Dissection S1 | 500 | 0.887 | 0.556 | 9 |
| subclass | Oligodendrocyte | scvi | Dissection AnG | 500 | 0.944 | 0.444 | 9 |
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
| subclass | scVI | SEA-AD DLPFC | 500 | 0.86 | 0.04 | 0.02 |
| subclass | scVI | Whole cortex | 500 | 0.86 | 0.04 | 0.02 |
| subclass | scVI | SEA-AD DLPFC | 100 | 0.859 | 0.04 | 0.02 |
| subclass | scVI | Whole cortex | 50 | 0.853 | 0.039 | 0.02 |
| subclass | scVI | SEA-AD MTG | 50 | 0.847 | 0.039 | 0.02 |
| subclass | scVI | SEA-AD DLPFC | 50 | 0.847 | 0.039 | 0.02 |
| subclass | scVI | Dissection AnG | 50 | 0.841 | 0.039 | 0.02 |
| subclass | scVI | Dissection ACC | 50 | 0.84 | 0.039 | 0.02 |
| subclass | scVI | Dissection V1 | 50 | 0.835 | 0.039 | 0.02 |
| subclass | scVI | Dissection S1 | 50 | 0.832 | 0.039 | 0.02 |
| subclass | scVI | Dissection A1 | 50 | 0.828 | 0.039 | 0.02 |
| subclass | scVI | Dissection DFC | 50 | 0.754 | 0.039 | 0.02 |
| subclass | scVI | Human MC SMART-seq | 50 | 0.726 | 0.039 | 0.02 |
| class | scVI | SEA-AD DLPFC | 500 | 0.935 | 0.04 | 0.02 |
| class | scVI | SEA-AD DLPFC | 100 | 0.932 | 0.04 | 0.02 |
| class | scVI | SEA-AD MTG | 100 | 0.93 | 0.04 | 0.02 |
| class | scVI | Dissection A1 | 100 | 0.926 | 0.04 | 0.02 |
| class | scVI | SEA-AD MTG | 50 | 0.922 | 0.039 | 0.02 |
| class | scVI | Whole cortex | 50 | 0.918 | 0.039 | 0.02 |
| class | scVI | Dissection S1 | 50 | 0.916 | 0.039 | 0.02 |
| class | scVI | Dissection AnG | 50 | 0.913 | 0.039 | 0.02 |
| class | scVI | SEA-AD DLPFC | 50 | 0.912 | 0.039 | 0.02 |
| class | scVI | Dissection ACC | 50 | 0.912 | 0.039 | 0.02 |
| class | scVI | Dissection A1 | 50 | 0.906 | 0.039 | 0.02 |
| class | scVI | Dissection V1 | 50 | 0.897 | 0.039 | 0.02 |
| class | scVI | Dissection DFC | 50 | 0.87 | 0.039 | 0.02 |
| class | scVI | Human MC SMART-seq | 50 | 0.737 | 0.039 | 0.02 |
| family | scVI | Whole cortex | 500 | 0.967 | 0.04 | 0.02 |
| family | scVI | Dissection A1 | 100 | 0.967 | 0.04 | 0.02 |
| family | scVI | Whole cortex | 50 | 0.965 | 0.039 | 0.02 |
| family | scVI | Dissection ACC | 50 | 0.96 | 0.039 | 0.02 |
| family | scVI | Dissection A1 | 50 | 0.958 | 0.039 | 0.02 |
| family | scVI | Dissection S1 | 50 | 0.956 | 0.039 | 0.02 |
| family | scVI | SEA-AD DLPFC | 50 | 0.956 | 0.039 | 0.02 |
| family | scVI | SEA-AD MTG | 50 | 0.956 | 0.039 | 0.02 |
| family | scVI | Dissection V1 | 50 | 0.954 | 0.039 | 0.02 |
| family | scVI | Dissection AnG | 50 | 0.951 | 0.039 | 0.02 |
| family | scVI | Human MC SMART-seq | 50 | 0.911 | 0.039 | 0.02 |
| family | scVI | Dissection DFC | 50 | 0.909 | 0.039 | 0.02 |
| global | Seurat | Dissection A1 | 50 | 0.99 | 0.066 | 0.03 |
| global | Seurat | Dissection AnG | 50 | 0.99 | 0.066 | 0.03 |
| global | Seurat | Dissection S1 | 50 | 0.989 | 0.066 | 0.03 |
| global | Seurat | SEA-AD DLPFC | 50 | 0.989 | 0.066 | 0.03 |
| global | Seurat | Dissection V1 | 50 | 0.988 | 0.066 | 0.03 |
| global | Seurat | Dissection ACC | 50 | 0.988 | 0.066 | 0.03 |
| global | Seurat | SEA-AD MTG | 50 | 0.987 | 0.066 | 0.03 |
| global | Seurat | Human MC SMART-seq | 50 | 0.987 | 0.066 | 0.03 |
| global | Seurat | Whole cortex | 50 | 0.986 | 0.066 | 0.03 |
| global | scVI | Dissection ACC | 100 | 0.985 | 0.04 | 0.02 |
| global | scVI | Whole cortex | 50 | 0.984 | 0.039 | 0.02 |
| global | scVI | SEA-AD DLPFC | 50 | 0.982 | 0.039 | 0.02 |
| global | scVI | Dissection S1 | 50 | 0.976 | 0.039 | 0.02 |
| global | scVI | Dissection ACC | 50 | 0.973 | 0.039 | 0.02 |
| global | scVI | Dissection A1 | 50 | 0.972 | 0.039 | 0.02 |
| global | scVI | Dissection V1 | 50 | 0.97 | 0.039 | 0.02 |
| global | scVI | SEA-AD MTG | 50 | 0.965 | 0.039 | 0.02 |
| global | scVI | Dissection AnG | 50 | 0.957 | 0.039 | 0.02 |
| global | scVI | Human MC SMART-seq | 50 | 0.952 | 0.039 | 0.02 |
| global | scVI | Dissection DFC | 50 | 0.942 | 0.039 | 0.02 |

### Computational Time

| method | step | subsample_ref | mean_duration | mean_memory |
| --- | --- | --- | --- | --- |
| scVI | Query Processing | 50 | 0.022 | 0.02 |
| scVI | Query Processing | 100 | 0.022 | 0.02 |
| scVI | Query Processing | 500 | 0.022 | 0.02 |
| Seurat | Prediction | 50 | 0.018 | 0.019 |
| Seurat | Prediction | 100 | 0.019 | 0.017 |
| Seurat | Prediction | 500 | 0.028 | 0.02 |
| Seurat | Query Processing | 50 | 0.021 | 0.03 |
| Seurat | Query Processing | 100 | 0.021 | 0.03 |
| Seurat | Query Processing | 500 | 0.021 | 0.03 |
| Seurat | Ref Processing | 50 | 0.027 | 0.03 |
| Seurat | Ref Processing | 100 | 0.042 | 0.036 |
| Seurat | Ref Processing | 500 | 0.112 | 0.032 |
| scVI | Prediction | 50 | 0.018 | 0.013 |
| scVI | Prediction | 100 | 0.018 | 0.013 |
| scVI | Prediction | 500 | 0.019 | 0.013 |


---

## Macro F1 vs Per-Cell-Type Conflict

scVI leads Seurat at every taxonomy level by macro F1 (subclass: 0.923 vs 0.882, OR = 0.620, p < 10^−300), and also wins more cell types at subclass (13/23 vs 10/23 by best-config win fraction). The methods diverge most for sparse or region-specific types: Seurat achieves better best-config F1 for L5 ET (0.709 vs lower for scVI), L5/6 NP, LAMP5, PVALB, SST, VLMC, and Pericyte — all types where the Human MC SMART-seq reference, which Seurat favors, provides specialized coverage. The macro and cell-type rankings are broadly consistent; scVI's macro advantage does not mask systematic per-type losses.

---

## Configuration Recommendation

**Recommended taxonomy level: subclass.** The only systematic failure (mean F1 < 0.5 in ≥ 3 studies) is Pericyte (mean F1 = 0.074, 6 studies), but this is a **reference coverage artifact**: Pericyte has 0 cells in all Dissection and both SEA-AD references — only Whole cortex (32 cells) and Human MC SMART-seq (32 cells) include it. With the recommended reference (Whole cortex), Pericyte annotation is attempted but limited by the small reference size. L5 ET is borderline (mean F1 = 0.574, SD = 0.322, 7 studies) and highly study-variable — not a systematic failure but warrants caution. All other subclasses achieve mean F1 > 0.65.

| Dimension | Recommended value | Rationale |
|---|---|---|
| Taxonomy level | subclass | Only failure is Pericyte (coverage artifact); L5 ET high-variance but not systematic |
| Method | scVI | Higher macro F1 at all levels (subclass Δ = +0.042); wins 13/23 cell types |
| Reference | Whole cortex | Only reference with complete subclass coverage (all 23 labels > 0); EMM 0.923 — competitive with SEA-AD DLPFC (0.926) and SEA-AD MTG (0.920), which both have 0 Pericyte cells |
| Cutoff | 0.0 | scVI subclass F1 drops from 0.905 → 0.551 between cutoff 0.0 and 0.75; no benefit to filtering |
| Subsample_ref | 100 | Marginally higher EMM than 500 (0.910 vs 0.904); identical compute time |

**Raw performance (scVI × Whole cortex × cutoff 0.0 × subsample_ref 100):**

| key | macro_f1 | macro_precision | macro_recall |
|---|---|---|---|
| global | 0.993 | 0.995 | 0.992 |
| family | 0.983 | 0.988 | 0.984 |
| class | 0.934 | 0.959 | 0.937 |
| subclass | 0.908 | 0.961 | 0.908 |

**Compute time (scVI × subsample_ref 100):**

| step | mean_duration (hrs) | mean_memory (GB) |
|---|---|---|
| Query Processing | 0.022 | 0.020 |
| Prediction | 0.018 | 0.013 |
| **Total** | **0.040** | **0.033** |

SEA-AD DLPFC × 500 and Whole cortex × 500 are the top Pareto-optimal configs; the recommended Whole cortex × 100 is not in the Pareto table, but the EMM difference vs 500 cells is negligible (0.910 vs 0.904) at identical compute cost. If Pericyte annotation is not required, SEA-AD DLPFC × 100 (EMM 0.926, Pareto-optimal) is a stronger choice.
