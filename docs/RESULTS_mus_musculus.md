# Mouse Brain Cell-Type Annotation Benchmarking Results

**Organism:** *Mus musculus*
**Date:** 2026-04-06
**Pipeline output:** `2024-07-01/mus_musculus/100/dataset_id/SCT/gap_false/`
**Model formula:** `macro_f1 ~ reference + method + cutoff + subsample_ref + treatment_state + sex + method:cutoff + reference:method + (1 | study)`
**Goal:** Determine which combination of method, reference, cutoff, and subsample_ref best optimises F1, precision, recall, and compute time for mouse brain cell-type annotation.

Reference abbreviations used throughout:
- **Motor cortex** = An integrated transcriptomic and epigenomic atlas of mouse primary motor cortex cell types
- **Cortical+Hipp. 10x** = Single-cell RNA-seq for all cortical & hippocampal regions (10x)
- **Cortical+Hipp. SSv4** = Single-cell RNA-seq for all cortical & hippocampal regions (SMART-Seq v4)
- **Whole cortex** = whole cortex

---

## Overall Performance

Model-estimated marginal means (ordered beta GLMM) averaged over all references and covariates. Values are back-transformed response-scale EMMs with 95% asymptotic CIs.

| key | method | EMM | 95% LCL | 95% UCL |
|-----|--------|-----|---------|---------|
| subclass | seurat | 0.834 | 0.755 | 0.891 |
| subclass | scvi_rf | 0.829 | 0.749 | 0.888 |
| subclass | scvi_knn | 0.821 | 0.739 | 0.882 |
| class | seurat | 0.860 | 0.801 | 0.904 |
| class | scvi_rf | 0.859 | 0.799 | 0.904 |
| class | scvi_knn | 0.850 | 0.786 | 0.896 |
| family | scvi_rf | 0.923 | 0.903 | 0.940 |
| family | scvi_knn | 0.918 | 0.896 | 0.935 |
| family | seurat | 0.906 | 0.882 | 0.925 |
| global | scvi_rf | 0.958 | 0.929 | 0.976 |
| global | scvi_knn | 0.948 | 0.913 | 0.970 |
| global | seurat | 0.940 | 0.900 | 0.965 |

**Best reference per key** (mean EMM across methods from `reference_method_emmeans_summary.tsv`):
- Subclass: **Cortical+Hipp. 10x** — mean EMM 0.898
- Class: **Cortical+Hipp. 10x** — mean EMM 0.919
- Family: **Cortical+Hipp. 10x** — mean EMM 0.945
- Global: **Cortical+Hipp. 10x** — mean EMM 0.963

Cortical+Hipp. 10x is the top reference at every taxonomy level.

---

## Macro F1 vs Per-Cell-Type F1 Conflict

This section compares two rankings: (1) methods ranked by model-estimated macro F1 (EMMs above), and (2) methods ranked by the fraction of individual cell types they win across studies.

**Macro F1 ranking** (from `method_emmeans_summary.tsv`):

| key | Rank 1 | Rank 2 | Rank 3 |
|-----|--------|--------|--------|
| subclass | seurat (0.834) | scvi_rf (0.829) | scvi_knn (0.821) |
| class | seurat (0.860) | scvi_rf (0.859) | scvi_knn (0.850) |
| family | scvi_rf (0.923) | scvi_knn (0.918) | seurat (0.906) |
| global | scvi_rf (0.958) | scvi_knn (0.948) | seurat (0.940) |

**Cell-type win counts** (from `celltype_rankings/rankings/rankings_best.tsv`, sum of `n_wins` per method × key):

| key | total labels | seurat wins | scvi_knn wins | scvi_rf wins |
|-----|-------------|-------------|---------------|-------------|
| subclass | 12 | **16** | 2 | 0 |
| class | 9 | **12** | 4 | 1 |
| family | 8 | **14** | 6 | 0 |
| global | 3 | **5** | 1 | 0 |

**Conflict summary:** At the family and global levels, scvi_rf ranks first on macro F1 but wins zero individual cell-type comparisons across all labels and studies. Seurat wins the most individual cell types at every taxonomy level despite ranking third at family and global on macro F1.

The discrepancy at family/global is explained by cutoff sensitivity: scvi_rf achieves high F1 at cutoff=0 but its performance collapses sharply at any confidence filtering, and in per-cell-type comparisons with best configurations its mean F1 is lower for nearly all labels. Comparing per-cell-type precision and recall at cutoff=0 (averaged over all cell types, references, and subsample_ref values):

| key | method | avg F1 | avg precision | avg recall |
|-----|--------|--------|---------------|------------|
| class | scvi_knn | 0.647 | 0.782 | 0.721 |
| class | seurat | 0.544 | 0.696 | 0.596 |
| class | scvi_rf | 0.507 | 0.792 | 0.603 |
| family | scvi_knn | 0.754 | 0.845 | 0.787 |
| family | scvi_rf | 0.692 | 0.818 | 0.757 |
| family | seurat | 0.697 | 0.750 | 0.730 |
| global | scvi_knn | 0.726 | 0.715 | 0.827 |
| global | scvi_rf | 0.676 | 0.638 | 0.860 |
| global | seurat | 0.629 | 0.660 | 0.707 |
| subclass | scvi_knn | 0.661 | 0.813 | 0.693 |
| subclass | seurat | 0.602 | 0.762 | 0.626 |
| subclass | scvi_rf | 0.536 | 0.802 | 0.595 |

These per-cell-type averages include difficult cell types (OPC, Microglia, Neural stem cell, Macrophage) that bring all methods' averages down, and are not directly comparable to the model EMMs which weight each study equally. scvi_knn leads per-cell-type average F1 at every level, driven by higher recall. scvi_rf's precision is high but its recall and per-cell-type F1 are low for many cell types, especially at subclass level.

**Conclusion:** Seurat is the recommended method. It dominates cell-type win counts at every level. scvi_rf's superior macro EMM at family/global is fragile — it is driven by performance at cutoff=0 and collapses at any confidence filtering. scvi_knn performs well per-cell-type but is consistently below seurat in head-to-head comparisons.

---

## Method Comparison: Pairwise Contrasts

Odds ratios from the ordered beta GLMM (OR > 1 means the numerator method is better).

| key | contrast | OR | p-value |
|-----|----------|----|---------|
| class | seurat / scvi_rf | 1.008 | 0.896 |
| class | seurat / scvi_knn | 1.092 | < 10⁻⁵ |
| class | scvi_rf / scvi_knn | 1.083 | < 10⁻⁴ |
| family | seurat / scvi_rf | 0.797 | < 10⁻¹⁶ |
| family | seurat / scvi_knn | 0.860 | < 10⁻¹³ |
| family | scvi_rf / scvi_knn | 1.079 | < 10⁻⁴ |
| global | seurat / scvi_rf | 0.687 | < 10⁻¹⁶ |
| global | seurat / scvi_knn | 0.855 | < 10⁻¹⁰ |
| global | scvi_rf / scvi_knn | 1.245 | < 10⁻¹³ |
| subclass | seurat / scvi_rf | 1.035 | 0.055 |
| subclass | seurat / scvi_knn | 1.093 | < 10⁻⁸ |
| subclass | scvi_rf / scvi_knn | 1.056 | < 10⁻³ |

**Key findings:** At class and subclass — the primary annotation levels — seurat and scvi_rf are statistically indistinguishable (p > 0.05), and both are significantly better than scvi_knn. At family and global, scvi_rf significantly outperforms seurat on macro F1, but this advantage does not translate to per-cell-type wins and collapses with any confidence filtering (see above).

---

## Confidence Cutoff Sensitivity (Method-Level EMMs)

Model-estimated macro F1 at each cutoff, averaged over all references, subsample_ref, and covariates. scvi_rf shows extreme sensitivity — EMM drops from ~0.86–0.96 to ~0.36–0.40 at cutoff=0.75. Seurat and scvi_knn are substantially more stable.

| key | method | cutoff=0 | cutoff=0.10 | cutoff=0.25 | cutoff=0.50 | cutoff=0.75 |
|-----|--------|----------|-------------|-------------|-------------|-------------|
| class | seurat | 0.860 | 0.858 | 0.854 | 0.848 | 0.841 |
| class | scvi_rf | 0.859 | 0.820 | 0.745 | 0.582 | 0.399 |
| class | scvi_knn | 0.850 | 0.845 | 0.837 | 0.823 | 0.809 |
| family | seurat | 0.906 | 0.901 | 0.895 | 0.882 | 0.868 |
| family | scvi_rf | 0.923 | 0.892 | 0.823 | 0.641 | 0.408 |
| family | scvi_knn | 0.918 | 0.912 | 0.903 | 0.886 | 0.867 |
| global | seurat | 0.940 | 0.936 | 0.928 | 0.914 | 0.898 |
| global | scvi_rf | 0.958 | 0.934 | 0.873 | 0.675 | 0.384 |
| global | scvi_knn | 0.948 | 0.940 | 0.926 | 0.896 | 0.855 |
| subclass | seurat | 0.834 | 0.831 | 0.827 | 0.820 | 0.812 |
| subclass | scvi_rf | 0.829 | 0.785 | 0.704 | 0.537 | 0.362 |
| subclass | scvi_knn | 0.821 | 0.816 | 0.808 | 0.793 | 0.777 |

By cutoff=0.50, scvi_rf has lost ~33–57% of its macro F1 relative to its cutoff=0 value. Seurat retains ~95–98% of its macro F1 at the same cutoff.

---

## Per-Cell-Type Cutoff Sensitivity

F1 values averaged over all references and subsample_ref levels. Sorted by descending F1(0) within each level.

### Subclass

| label | method | F1(0) | F1(0.25) | F1(0.50) | F1(0.75) |
|-------|--------|-------|----------|----------|----------|
| DG | seurat | 0.991 | 0.991 | 0.991 | 0.987 |
| DG | scvi_knn | 0.958 | 0.958 | 0.950 | 0.885 |
| DG | scvi_rf | 0.844 | 0.837 | 0.490 | 0.191 |
| Endothelial | scvi_knn | 0.949 | 0.949 | 0.937 | 0.905 |
| Endothelial | scvi_rf | 0.939 | 0.892 | 0.660 | 0.349 |
| Endothelial | seurat | 0.935 | 0.935 | 0.932 | 0.901 |
| CA3 | seurat | 0.861 | 0.861 | 0.850 | 0.678 |
| Astrocyte | scvi_knn | 0.869 | 0.869 | 0.852 | 0.790 |
| Astrocyte | seurat | 0.861 | 0.861 | 0.863 | 0.858 |
| Astrocyte | scvi_rf | 0.834 | 0.751 | 0.564 | 0.301 |
| Oligodendrocyte | scvi_knn | 0.853 | 0.854 | 0.861 | 0.861 |
| Oligodendrocyte | seurat | 0.833 | 0.833 | 0.831 | 0.831 |
| Oligodendrocyte | scvi_rf | 0.787 | 0.744 | 0.466 | 0.210 |
| CA1-ProS | scvi_rf | 0.727 | 0.727 | 0.265 | 0.034 |
| CA1-ProS | scvi_knn | 0.693 | 0.693 | 0.541 | 0.238 |
| CA1-ProS | seurat | 0.616 | 0.616 | 0.670 | 0.677 |
| Pericyte | scvi_knn | 0.789 | 0.789 | 0.622 | 0.469 |
| Pericyte | seurat | 0.667 | 0.667 | 0.578 | 0.567 |
| Pericyte | scvi_rf | 0.567 | 0.567 | 0.276 | 0.167 |
| Cajal-Retzius cell | scvi_knn | 0.561 | 0.563 | 0.534 | 0.313 |
| Cajal-Retzius cell | seurat | 0.494 | 0.494 | 0.458 | 0.325 |
| Cajal-Retzius cell | scvi_rf | 0.355 | 0.263 | 0.061 | N/A |
| Neural stem cell | scvi_knn | 0.515 | 0.518 | 0.499 | 0.413 |
| Neural stem cell | scvi_rf | 0.386 | 0.468 | 0.390 | 0.227 |
| Neural stem cell | seurat | 0.256 | 0.254 | 0.215 | 0.162 |
| Microglia | seurat | 0.456 | 0.456 | 0.453 | 0.423 |
| Microglia | scvi_rf | 0.015 | 0.013 | N/A | N/A |
| Macrophage | scvi_rf | 0.235 | 0.223 | 0.149 | 0.057 |
| Macrophage | scvi_knn | 0.212 | 0.212 | 0.204 | 0.172 |
| Macrophage | seurat | 0.178 | 0.178 | 0.170 | 0.146 |
| OPC | scvi_knn | 0.262 | 0.261 | 0.175 | 0.048 |
| OPC | seurat | 0.079 | 0.078 | 0.062 | 0.113 |
| OPC | scvi_rf | 0.060 | 0.057 | N/A | N/A |

**Most cutoff-sensitive at subclass (top 5 per method, by drop F1(0)→F1(0.75)):**

| method | label | F1(0) | F1(0.75) | Drop |
|--------|-------|-------|----------|------|
| seurat | CA3 | 0.861 | 0.678 | 0.183 |
| seurat | Cajal-Retzius cell | 0.494 | 0.325 | 0.169 |
| seurat | Pericyte | 0.667 | 0.567 | 0.100 |
| seurat | Neural stem cell | 0.256 | 0.162 | 0.093 |
| seurat | Endothelial | 0.935 | 0.901 | 0.034 |
| scvi_rf | CA1-ProS | 0.727 | 0.034 | 0.693 |
| scvi_rf | DG | 0.844 | 0.191 | 0.653 |
| scvi_rf | Endothelial | 0.939 | 0.349 | 0.590 |
| scvi_rf | Oligodendrocyte | 0.787 | 0.210 | 0.577 |
| scvi_rf | Astrocyte | 0.834 | 0.301 | 0.533 |
| scvi_knn | CA1-ProS | 0.693 | 0.238 | 0.455 |
| scvi_knn | Pericyte | 0.789 | 0.469 | 0.320 |
| scvi_knn | CA3 | 0.613 | 0.301 | 0.312 |
| scvi_knn | Cajal-Retzius cell | 0.561 | 0.313 | 0.248 |
| scvi_knn | OPC | 0.262 | 0.048 | 0.214 |

### Class

| label | method | F1(0) | F1(0.25) | F1(0.50) | F1(0.75) |
|-------|--------|-------|----------|----------|----------|
| Hippocampal neuron | scvi_knn | 0.972 | 0.971 | 0.895 | 0.717 |
| Hippocampal neuron | scvi_rf | 0.937 | 0.906 | 0.490 | 0.151 |
| Hippocampal neuron | seurat | 0.836 | 0.836 | 0.928 | 0.879 |
| Vascular | scvi_rf | 0.938 | 0.878 | 0.635 | 0.329 |
| Vascular | scvi_knn | 0.929 | 0.929 | 0.910 | 0.856 |
| Vascular | seurat | 0.906 | 0.906 | 0.898 | 0.848 |
| Astrocyte | scvi_knn | 0.869 | 0.869 | 0.852 | 0.790 |
| Astrocyte | seurat | 0.861 | 0.861 | 0.863 | 0.858 |
| Astrocyte | scvi_rf | 0.834 | 0.751 | 0.564 | 0.301 |
| Oligodendrocyte | scvi_knn | 0.853 | 0.854 | 0.861 | 0.861 |
| Oligodendrocyte | seurat | 0.833 | 0.833 | 0.831 | 0.831 |
| Oligodendrocyte | scvi_rf | 0.787 | 0.744 | 0.466 | 0.210 |
| Microglia | seurat | 0.456 | 0.456 | 0.453 | 0.423 |
| Microglia | scvi_rf | 0.015 | 0.013 | N/A | N/A |
| Cajal-Retzius cell | scvi_knn | 0.564 | 0.564 | 0.534 | 0.313 |
| Cajal-Retzius cell | seurat | 0.494 | 0.494 | 0.458 | 0.325 |
| Cajal-Retzius cell | scvi_rf | 0.365 | 0.264 | 0.061 | N/A |
| Neural stem cell | scvi_knn | 0.515 | 0.518 | 0.499 | 0.413 |
| Neural stem cell | scvi_rf | 0.386 | 0.468 | 0.390 | 0.227 |
| Neural stem cell | seurat | 0.256 | 0.254 | 0.215 | 0.162 |
| Macrophage | scvi_rf | 0.242 | 0.224 | 0.149 | 0.057 |
| Macrophage | scvi_knn | 0.212 | 0.212 | 0.204 | 0.172 |
| Macrophage | seurat | 0.179 | 0.179 | 0.170 | 0.146 |
| OPC | scvi_knn | 0.262 | 0.261 | 0.175 | 0.048 |
| OPC | seurat | 0.079 | 0.078 | 0.062 | 0.113 |
| OPC | scvi_rf | 0.060 | 0.057 | N/A | N/A |

**Most cutoff-sensitive at class (top 5 per method, by drop F1(0)→F1(0.75)):**

| method | label | F1(0) | F1(0.75) | Drop |
|--------|-------|-------|----------|------|
| seurat | Cajal-Retzius cell | 0.494 | 0.325 | 0.169 |
| seurat | Neural stem cell | 0.256 | 0.162 | 0.093 |
| seurat | Vascular | 0.906 | 0.848 | 0.058 |
| seurat | Microglia | 0.456 | 0.423 | 0.033 |
| seurat | Macrophage | 0.179 | 0.146 | 0.033 |
| scvi_rf | Hippocampal neuron | 0.937 | 0.151 | 0.786 |
| scvi_rf | Vascular | 0.938 | 0.329 | 0.609 |
| scvi_rf | Oligodendrocyte | 0.787 | 0.210 | 0.577 |
| scvi_rf | Astrocyte | 0.834 | 0.301 | 0.533 |
| scvi_rf | Macrophage | 0.242 | 0.057 | 0.185 |
| scvi_knn | Hippocampal neuron | 0.972 | 0.717 | 0.255 |
| scvi_knn | Cajal-Retzius cell | 0.564 | 0.313 | 0.251 |
| scvi_knn | OPC | 0.262 | 0.048 | 0.214 |
| scvi_knn | Neural stem cell | 0.515 | 0.413 | 0.102 |
| scvi_knn | Astrocyte | 0.869 | 0.790 | 0.079 |

### Family

| label | method | F1(0) | F1(0.25) | F1(0.50) | F1(0.75) |
|-------|--------|-------|----------|----------|----------|
| CNS macrophage | scvi_knn | 0.960 | 0.960 | 0.951 | 0.927 |
| CNS macrophage | scvi_rf | 0.928 | 0.944 | 0.761 | 0.355 |
| CNS macrophage | seurat | 0.928 | 0.928 | 0.923 | 0.814 |
| Vascular | scvi_rf | 0.938 | 0.878 | 0.635 | 0.329 |
| Vascular | scvi_knn | 0.929 | 0.929 | 0.910 | 0.856 |
| Vascular | seurat | 0.906 | 0.906 | 0.898 | 0.848 |
| GABAergic | seurat | 0.916 | 0.916 | 0.876 | 0.705 |
| GABAergic | scvi_knn | 0.862 | 0.863 | 0.860 | 0.742 |
| GABAergic | scvi_rf | 0.820 | 0.783 | 0.429 | 0.126 |
| Astrocyte | scvi_knn | 0.869 | 0.869 | 0.852 | 0.790 |
| Astrocyte | seurat | 0.861 | 0.861 | 0.863 | 0.858 |
| Astrocyte | scvi_rf | 0.834 | 0.751 | 0.564 | 0.301 |
| Glutamatergic | seurat | 0.798 | 0.798 | 0.771 | 0.634 |
| Glutamatergic | scvi_rf | 0.788 | 0.761 | 0.316 | 0.052 |
| Glutamatergic | scvi_knn | 0.780 | 0.778 | 0.623 | 0.371 |
| Oligodendrocyte | scvi_knn | 0.853 | 0.854 | 0.861 | 0.861 |
| Oligodendrocyte | seurat | 0.833 | 0.833 | 0.831 | 0.831 |
| Oligodendrocyte | scvi_rf | 0.787 | 0.744 | 0.466 | 0.210 |
| Neural stem cell | scvi_knn | 0.515 | 0.518 | 0.499 | 0.413 |
| Neural stem cell | scvi_rf | 0.386 | 0.468 | 0.390 | 0.227 |
| Neural stem cell | seurat | 0.256 | 0.254 | 0.215 | 0.162 |
| OPC | scvi_knn | 0.262 | 0.261 | 0.175 | 0.048 |
| OPC | seurat | 0.079 | 0.078 | 0.062 | 0.113 |
| OPC | scvi_rf | 0.060 | 0.057 | N/A | N/A |

**Most cutoff-sensitive at family (top 5 per method, by drop F1(0)→F1(0.75)):**

| method | label | F1(0) | F1(0.75) | Drop |
|--------|-------|-------|----------|------|
| seurat | GABAergic | 0.916 | 0.705 | 0.212 |
| seurat | Glutamatergic | 0.798 | 0.634 | 0.165 |
| seurat | CNS macrophage | 0.928 | 0.814 | 0.114 |
| seurat | Neural stem cell | 0.256 | 0.162 | 0.093 |
| seurat | Vascular | 0.906 | 0.848 | 0.058 |
| scvi_rf | Glutamatergic | 0.788 | 0.052 | 0.736 |
| scvi_rf | GABAergic | 0.820 | 0.126 | 0.694 |
| scvi_rf | Vascular | 0.938 | 0.329 | 0.609 |
| scvi_rf | Oligodendrocyte | 0.787 | 0.210 | 0.577 |
| scvi_rf | CNS macrophage | 0.928 | 0.355 | 0.573 |
| scvi_knn | Glutamatergic | 0.780 | 0.371 | 0.409 |
| scvi_knn | OPC | 0.262 | 0.048 | 0.214 |
| scvi_knn | GABAergic | 0.862 | 0.742 | 0.120 |
| scvi_knn | Neural stem cell | 0.515 | 0.413 | 0.102 |
| scvi_knn | Astrocyte | 0.869 | 0.790 | 0.079 |

### Global

| label | method | F1(0) | F1(0.25) | F1(0.50) | F1(0.75) |
|-------|--------|-------|----------|----------|----------|
| Non-neuron | scvi_knn | 0.948 | 0.947 | 0.928 | 0.879 |
| Non-neuron | scvi_rf | 0.931 | 0.862 | 0.617 | 0.310 |
| Non-neuron | seurat | 0.926 | 0.925 | 0.920 | 0.878 |
| Neuron | scvi_knn | 0.717 | 0.715 | 0.602 | 0.355 |
| Neuron | scvi_rf | 0.711 | 0.668 | 0.222 | 0.032 |
| Neuron | seurat | 0.707 | 0.708 | 0.672 | 0.523 |
| Neural stem cell | scvi_knn | 0.515 | 0.518 | 0.499 | 0.413 |
| Neural stem cell | scvi_rf | 0.386 | 0.468 | 0.390 | 0.227 |
| Neural stem cell | seurat | 0.256 | 0.254 | 0.215 | 0.162 |

**Most cutoff-sensitive at global (top 5 per method, by drop F1(0)→F1(0.75)):**

| method | label | F1(0) | F1(0.75) | Drop |
|--------|-------|-------|----------|------|
| seurat | Neuron | 0.707 | 0.523 | 0.184 |
| seurat | Neural stem cell | 0.256 | 0.162 | 0.093 |
| seurat | Non-neuron | 0.926 | 0.878 | 0.048 |
| scvi_rf | Neuron | 0.711 | 0.032 | 0.679 |
| scvi_rf | Non-neuron | 0.931 | 0.310 | 0.621 |
| scvi_rf | Neural stem cell | 0.386 | 0.227 | 0.159 |
| scvi_knn | Neuron | 0.717 | 0.355 | 0.361 |
| scvi_knn | Neural stem cell | 0.515 | 0.413 | 0.102 |
| scvi_knn | Non-neuron | 0.948 | 0.879 | 0.069 |

---

## Reference Subsample Size

Model-estimated EMMs per subsample_ref level.

| key | subsample_ref | EMM | 95% LCL | 95% UCL |
|-----|--------------|-----|---------|---------|
| class | 500 | 0.860 | 0.801 | 0.904 |
| class | 100 | 0.858 | 0.797 | 0.902 |
| class | 50 | 0.851 | 0.789 | 0.898 |
| family | 500 | 0.918 | 0.897 | 0.936 |
| family | 100 | 0.919 | 0.898 | 0.936 |
| family | 50 | 0.911 | 0.888 | 0.929 |
| global | 500 | 0.954 | 0.922 | 0.973 |
| global | 100 | 0.951 | 0.918 | 0.971 |
| global | 50 | 0.943 | 0.904 | 0.966 |
| subclass | 500 | 0.831 | 0.751 | 0.889 |
| subclass | 100 | 0.831 | 0.751 | 0.889 |
| subclass | 50 | 0.823 | 0.741 | 0.884 |

**Pairwise subsample_ref contrasts:**

| key | contrast | OR | p-value |
|-----|----------|----|---------|
| class | 500 / 100 | 1.023 | 0.182 |
| class | 500 / 50 | 1.075 | < 10⁻⁷ |
| class | 100 / 50 | 1.051 | < 10⁻³ |
| family | 500 / 100 | 0.995 | 0.911 |
| family | 500 / 50 | 1.106 | < 10⁻¹³ |
| family | 100 / 50 | 1.111 | < 10⁻¹³ |
| global | 500 / 100 | 1.066 | < 10⁻⁴ |
| global | 500 / 50 | 1.261 | < 10⁻¹⁶ |
| global | 100 / 50 | 1.182 | < 10⁻¹⁶ |
| subclass | 500 / 100 | 1.002 | 0.983 |
| subclass | 500 / 50 | 1.055 | < 10⁻⁶ |
| subclass | 100 / 50 | 1.053 | < 10⁻⁶ |

**Summary:** subsample_ref=100 and subsample_ref=500 are statistically indistinguishable at class, family, and subclass (all p > 0.05). At global, 500 > 100 (OR=1.07) but the absolute EMM difference is <0.003. subsample_ref=50 is consistently inferior at all levels. **subsample_ref=100 is sufficient** and avoids the 2.3× higher reference processing time required by 500.

---

## Biological Covariates

> **Interpretation note:** Treatment, sex, and assay type are not randomised across studies — they are study-level properties that co-vary with other unmeasured factors (tissue source, protocol, lab context). Any observed differences should be attributed to study-level confounding rather than true biological effects on annotation accuracy.

### Treatment State

| key | treatment_state | EMM | 95% LCL | 95% UCL |
|-----|----------------|-----|---------|---------|
| class | treatment | 0.863 | 0.805 | 0.906 |
| class | no treatment | 0.849 | 0.786 | 0.896 |
| family | treatment | 0.917 | 0.895 | 0.935 |
| family | no treatment | 0.915 | 0.893 | 0.933 |
| global | no treatment | 0.952 | 0.919 | 0.972 |
| global | treatment | 0.947 | 0.910 | 0.969 |
| subclass | treatment | 0.837 | 0.759 | 0.893 |
| subclass | no treatment | 0.820 | 0.736 | 0.881 |

Treatment contrasts (no treatment / treatment OR): class 0.893 (p < 10⁻⁷), family 0.973 (p = 0.220), global 1.122 (p < 10⁻⁵), subclass 0.886 (p < 10⁻⁹). Significant differences at class, global, and subclass most likely reflect study-specific protocols or biological composition differences between treated and untreated study cohorts rather than a direct effect of pharmacological treatment on annotation accuracy.

### Sex

| key | sex | EMM | 95% LCL | 95% UCL |
|-----|-----|-----|---------|---------|
| class | female/unspecified | 0.910 | 0.834 | 0.953 |
| class | male | 0.779 | 0.683 | 0.852 |
| family | female/unspecified | 0.947 | 0.921 | 0.965 |
| family | male | 0.870 | 0.835 | 0.897 |
| global | female/unspecified | 0.969 | 0.927 | 0.987 |
| global | male | 0.918 | 0.855 | 0.955 |
| subclass | female/unspecified | 0.883 | 0.768 | 0.945 |
| subclass | male | 0.756 | 0.648 | 0.839 |

Studies with predominantly female or unspecified-sex samples show higher annotation F1 across all levels (OR: class 2.87, family 2.68, global 2.80, subclass 2.43; all p ≤ 0.07). The very large ORs with wide CIs reflect high uncertainty, likely driven by few studies per sex category. These differences reflect study-level confounding — female-dominant datasets in this benchmark likely differ from male-dominant ones in tissue processing, region sampled, and cell type composition, not in any inherent sex-related annotability.

---

## Between-Study Heterogeneity

Random effect standard deviations (SD of the study-level random intercept) from the GLMM.

| key | random effect SD |
|-----|-----------------|
| class | 0.503 |
| family | 0.310 |
| global | 0.654 |
| subclass | 0.594 |

Heterogeneity is highest at global and subclass levels. Family-level annotation is the most consistent across studies.

### Subclass — Per-Cell-Type Study Variance

(From `study_variance/study_variance/study_variance_summary.tsv`, cutoff=0. Only subclass data available.)

| label | n_studies / total | mean_f1 | std_f1 |
|-------|------------------|---------|--------|
| Endothelial | 6/7 | 0.953 | 0.025 |
| DG | 1/7 | 0.894 | N/A |
| Astrocyte | 6/7 | 0.893 | 0.068 |
| Oligodendrocyte | 6/7 | 0.845 | 0.066 |
| CA3 | 1/7 | 0.686 | N/A |
| CA1-ProS | 1/7 | 0.679 | N/A |
| Pericyte | 1/7 | 0.637 | N/A |
| Cajal-Retzius cell | 2/7 | 0.570 | 0.254 |
| Neural stem cell | 2/7 | 0.372 | 0.114 |
| Macrophage | 2/7 | 0.205 | 0.079 |
| Microglia | 6/7 | 0.154 | 0.049 |
| OPC | 6/7 | 0.131 | 0.109 |

Endothelial, Astrocyte, and Oligodendrocyte are consistently well-annotated. Microglia and OPC have very low mean F1 with high variability across studies. Hippocampal subtypes (DG, CA3, CA1-ProS) appear in only 1 of 7 studies, reflecting their regional specificity.

---

## Cell-Type Rankings

Best configuration per cell type (highest mean F1 across studies). Reference names abbreviated.

### Subclass

| label | method | reference | subsample_ref | mean_f1 | win_fraction |
|-------|--------|-----------|--------------|---------|-------------|
| DG | seurat | Whole cortex | 500 | 0.996 | 1.000 |
| CA3 | seurat | Whole cortex | 100 | 0.943 | 1.000 |
| Endothelial | seurat | Cortical+Hipp. 10x | 500 | 0.986 | 0.333 |
| CA1-ProS | seurat | Whole cortex | 100 | 0.933 | 1.000 |
| Cajal-Retzius cell | seurat | Cortical+Hipp. 10x | 50 | 0.992 | 1.000 |
| Oligodendrocyte | seurat | Motor cortex | 100 | 0.974 | 0.500 |
| Astrocyte | seurat | Cortical+Hipp. 10x | 500 | 0.943 | 0.667 |
| Pericyte | seurat | Whole cortex | 50 | 0.933 | 1.000 |
| Neural stem cell | scvi_knn | Motor cortex | 100 | 0.555 | 0.000 |
| Macrophage | scvi_rf | Cortical+Hipp. SSv4 | 100 | 0.247 | 0.000 |
| Microglia | seurat | Whole cortex | 500 | 0.459 | 0.167 |
| OPC | scvi_knn | Whole cortex | 100 | 0.277 | 0.333 |

### Class

| label | method | reference | subsample_ref | mean_f1 | win_fraction |
|-------|--------|-----------|--------------|---------|-------------|
| Cajal-Retzius cell | seurat | Cortical+Hipp. 10x | 50 | 0.992 | 1.000 |
| Hippocampal neuron | scvi_knn | Cortical+Hipp. 10x | 100 | 0.996 | 1.000 |
| Vascular | seurat | Cortical+Hipp. 10x | 500 | 0.977 | 0.333 |
| Oligodendrocyte | seurat | Motor cortex | 100 | 0.974 | 0.500 |
| Astrocyte | seurat | Cortical+Hipp. 10x | 500 | 0.943 | 0.667 |
| Microglia | seurat | Whole cortex | 500 | 0.459 | 0.167 |
| Neural stem cell | scvi_knn | Motor cortex | 500 | 0.514 | 0.500 |
| Macrophage | scvi_rf | Cortical+Hipp. SSv4 | 500 | 0.263 | 0.500 |
| OPC | scvi_knn | Whole cortex | 100 | 0.277 | 0.333 |

### Family

| label | method | reference | subsample_ref | mean_f1 | win_fraction |
|-------|--------|-----------|--------------|---------|-------------|
| CNS macrophage | scvi_knn | Cortical+Hipp. 10x | 100 | 0.983 | 0.500 |
| GABAergic | seurat | Cortical+Hipp. SSv4 | 50 | 0.990 | 0.500 |
| Vascular | seurat | Cortical+Hipp. 10x | 500 | 0.977 | 0.333 |
| Oligodendrocyte | seurat | Motor cortex | 100 | 0.974 | 0.500 |
| Astrocyte | seurat | Cortical+Hipp. 10x | 500 | 0.943 | 0.667 |
| Glutamatergic | seurat | Whole cortex | 500 | 0.892 | 0.500 |
| Neural stem cell | scvi_knn | Motor cortex | 500 | 0.514 | 0.500 |
| OPC | scvi_knn | Whole cortex | 100 | 0.277 | 0.333 |

### Global

| label | method | reference | subsample_ref | mean_f1 | win_fraction |
|-------|--------|-----------|--------------|---------|-------------|
| Non-neuron | seurat | Cortical+Hipp. 10x | 500 | 0.992 | 0.429 |
| Neuron | seurat | Cortical+Hipp. 10x | 500 | 0.882 | 0.333 |
| Neural stem cell | scvi_knn | Motor cortex | 500 | 0.514 | 0.500 |

---

## Hippocampal Contamination (Mouse Only)

Contamination is defined as predicted_support > 0 for a hippocampal label (Hippocampal neuron, DG, CA1-ProS, CA3) where support = 0 in the query ground truth (spurious predictions in non-hippocampal samples). Filtered to `key == "class"`.

| cutoff | method | mean_spurious_per_query | mean_recall_non_hippo |
|--------|--------|------------------------|-----------------------|
| 0.00 | seurat | 2.727 | 0.655 |
| 0.00 | scvi_rf | 3.286 | 0.660 |
| 0.00 | scvi_knn | 5.222 | 0.696 |
| 0.10 | seurat | 2.727 | 0.655 |
| 0.10 | scvi_rf | 3.286 | 0.660 |
| 0.10 | scvi_knn | 5.222 | 0.696 |
| 0.20 | seurat | 2.682 | 0.654 |
| 0.20 | scvi_rf | 2.692 | 0.634 |
| 0.20 | scvi_knn | 5.148 | 0.696 |
| 0.25 | seurat | 2.455 | 0.654 |
| 0.25 | scvi_rf | 2.222 | 0.579 |
| 0.25 | scvi_knn | 5.148 | 0.696 |
| 0.50 | seurat | 2.000 | 0.634 |
| 0.50 | scvi_rf | N/A | 0.321 |
| 0.50 | scvi_knn | 3.889 | 0.650 |
| 0.75 | seurat | 1.500 | 0.553 |
| 0.75 | scvi_rf | N/A | 0.117 |
| 0.75 | scvi_knn | 1.818 | 0.547 |

**Narrative:**

- **Seurat:** Contamination starts at 2.7 spurious predictions per query and decreases gradually. At cutoff=0.25, spurious predictions drop to 2.5 with virtually no recall impact (0.655→0.654). At cutoff=0.50, contamination falls further to 2.0 but recall drops 2 pp. **Recommended cutoff: 0.25** — meaningful reduction with negligible recall cost.

- **scvi_rf:** Contamination starts at 3.3 spurious per query. Filtering below cutoff=0.10 has no effect. At cutoff=0.25, contamination falls to 2.2 but non-hippocampal recall drops 8 pp (0.660→0.579). By cutoff=0.50, no hippocampal contamination rows remain but recall collapses to 0.321. **No good cutoff:** meaningful contamination reduction requires cutoffs that severely penalise recall.

- **scvi_knn:** Contamination is highest at baseline (5.2 spurious per query) and almost completely insensitive to cutoffs up to 0.25. At cutoff=0.50, contamination drops to 3.9 with recall 0.650; at cutoff=0.75 it falls to 1.8 but recall drops to 0.547. **No clean cutoff** — contamination profile is worst across the full cutoff range.

---

## Computational Time

From `comptime_plots/comptime_summary.tsv`. Seurat reference processing scales strongly with subsample_ref; all other steps are effectively constant. Rows with identical values across all subsample_ref levels are collapsed.

| method | step | subsample_ref | mean_duration (hrs) | mean_memory (GB) |
|--------|------|--------------|---------------------|-----------------|
| scVI RF/kNN | Query Processing | 50/100/500 | 0.018 | 0.020 |
| scVI RF/kNN | Embedding | 50 | 0.021 | 0.016 |
| scVI RF/kNN | Embedding | 100 | 0.017 | 0.016 |
| scVI RF/kNN | Embedding | 500 | 0.020 | 0.016 |
| scVI RF/kNN | Prediction | 50/100/500 | 0.017 | 0.013 |
| Seurat | Query Processing | 50/100/500 | 0.032 | 0.025 |
| Seurat | Ref Processing | 50 | 0.046 | 0.044 |
| Seurat | Ref Processing | 100 | 0.052 | 0.040 |
| **Seurat** | **Ref Processing** | **500** | **0.117** | **0.050** |
| Seurat | Prediction | 50 | 0.018 | 0.020 |
| Seurat | Prediction | 100 | 0.019 | 0.021 |
| Seurat | Prediction | 500 | 0.023 | 0.021 |

Seurat reference processing at subsample_ref=500 (0.117 hrs) is 2.3× slower than at subsample_ref=100 (0.052 hrs), with no meaningful gain in annotation performance at most taxonomy levels. scVI RF/kNN steps are uniformly fast and do not scale with subsample_ref.

---

## Assay Exploration (Mouse Only)

Model-estimated EMMs comparing references built from cell-only vs. cell-and-nucleus data, and queries that are single-cell vs. single-nucleus. Differences likely reflect study-level confounding (single-cell and single-nucleus studies differ in dissociation protocol, gene detection, and overall biology) rather than a direct assay effect.

| ref_type | query_type | key | EMM | 95% LCL | 95% UCL |
|----------|------------|-----|-----|---------|---------|
| cell_only | single-nucleus | subclass | 0.838 | 0.830 | 0.846 |
| cell_and_nucleus | single-nucleus | subclass | 0.659 | 0.648 | 0.671 |
| cell_only | single-cell | subclass | 0.751 | 0.747 | 0.755 |
| cell_and_nucleus | single-cell | subclass | 0.646 | 0.641 | 0.651 |
| cell_only | single-nucleus | class | 0.876 | 0.870 | 0.882 |
| cell_and_nucleus | single-nucleus | class | 0.667 | 0.655 | 0.679 |
| cell_only | single-cell | class | 0.750 | 0.746 | 0.754 |
| cell_and_nucleus | single-cell | class | 0.644 | 0.639 | 0.649 |
| cell_only | single-nucleus | family | 0.907 | 0.902 | 0.912 |
| cell_and_nucleus | single-nucleus | family | 0.889 | 0.883 | 0.894 |
| cell_only | single-cell | family | 0.835 | 0.831 | 0.838 |
| cell_and_nucleus | single-cell | family | 0.803 | 0.799 | 0.806 |
| cell_only | single-nucleus | global | 0.923 | 0.917 | 0.928 |
| cell_and_nucleus | single-nucleus | global | 0.918 | 0.913 | 0.923 |
| cell_only | single-cell | global | 0.794 | 0.790 | 0.798 |
| cell_and_nucleus | single-cell | global | 0.789 | 0.784 | 0.793 |

Single-nucleus queries consistently outperform single-cell queries, and cell-only references outperform cell-and-nucleus references. These patterns almost certainly reflect study-level confounds: single-nucleus datasets in this benchmark come from distinct GEO studies with different tissue regions, cell type compositions, and processing workflows relative to single-cell studies, rather than indicating a direct assay effect on annotation performance.

---

## Pareto-Optimal Configurations

Configurations not dominated in both mean F1 and compute time (filter: `pareto == TRUE`). Reference names abbreviated; showing configurations at subsample_ref=100 unless otherwise noted.

| key | method | reference | subsample_ref | mean_f1 | total_duration_hrs | total_memory_gb |
|-----|--------|-----------|--------------|---------|-------------------|-----------------|
| subclass | Seurat | Cortical+Hipp. 10x | 100 | 0.852 | 0.102 | 0.040 |
| subclass | Seurat | Cortical+Hipp. 10x | 50 | 0.850 | 0.096 | 0.044 |
| subclass | scVI kNN | Cortical+Hipp. SSv4 | 100 | 0.761 | 0.053 | 0.020 |
| subclass | scVI RF | Cortical+Hipp. 10x | 100 | 0.738 | 0.053 | 0.020 |
| subclass | scVI kNN | Cortical+Hipp. 10x | 100 | 0.737 | 0.053 | 0.020 |
| subclass | scVI RF | Cortical+Hipp. SSv4 | 100 | 0.701 | 0.053 | 0.020 |
| subclass | scVI kNN | Whole cortex | 100 | 0.630 | 0.053 | 0.020 |
| subclass | scVI RF | Whole cortex | 100 | 0.572 | 0.053 | 0.020 |
| subclass | scVI kNN | Motor cortex | 100 | 0.572 | 0.053 | 0.020 |
| subclass | scVI RF | Motor cortex | 100 | 0.532 | 0.053 | 0.020 |
| class | Seurat | Cortical+Hipp. 10x | 100 | 0.826 | 0.102 | 0.040 |
| class | Seurat | Cortical+Hipp. 10x | 50 | 0.815 | 0.096 | 0.044 |
| class | scVI kNN | Cortical+Hipp. 10x | 100 | 0.802 | 0.053 | 0.020 |
| class | scVI RF | Cortical+Hipp. 10x | 100 | 0.781 | 0.053 | 0.020 |
| class | scVI kNN | Cortical+Hipp. SSv4 | 100 | 0.733 | 0.053 | 0.020 |
| class | scVI RF | Cortical+Hipp. SSv4 | 100 | 0.683 | 0.053 | 0.020 |
| class | scVI kNN | Whole cortex | 100 | 0.615 | 0.053 | 0.020 |
| class | scVI RF | Whole cortex | 100 | 0.581 | 0.053 | 0.020 |
| class | scVI kNN | Motor cortex | 100 | 0.568 | 0.053 | 0.020 |
| class | scVI RF | Motor cortex | 100 | 0.522 | 0.053 | 0.020 |
| family | Seurat | Cortical+Hipp. 10x | 100 | 0.912 | 0.102 | 0.040 |
| family | Seurat | Cortical+Hipp. 10x | 50 | 0.907 | 0.096 | 0.044 |
| family | scVI kNN | Cortical+Hipp. 10x | 500 | 0.904 | 0.056 | 0.020 |
| family | scVI kNN | Cortical+Hipp. 10x | 100 | 0.895 | 0.053 | 0.020 |
| family | scVI RF | Cortical+Hipp. 10x | 100 | 0.891 | 0.053 | 0.020 |
| family | scVI kNN | Cortical+Hipp. SSv4 | 100 | 0.866 | 0.053 | 0.020 |
| family | scVI kNN | Motor cortex | 100 | 0.839 | 0.053 | 0.020 |
| family | scVI RF | Motor cortex | 100 | 0.802 | 0.053 | 0.020 |
| family | scVI RF | Cortical+Hipp. SSv4 | 100 | 0.799 | 0.053 | 0.020 |
| family | scVI kNN | Whole cortex | 100 | 0.768 | 0.053 | 0.020 |
| family | scVI RF | Whole cortex | 100 | 0.742 | 0.053 | 0.020 |
| global | Seurat | Cortical+Hipp. 10x | 100 | 0.940 | 0.102 | 0.040 |
| global | Seurat | Cortical+Hipp. 10x | 50 | 0.939 | 0.096 | 0.044 |
| global | scVI kNN | Cortical+Hipp. 10x | 100 | 0.911 | 0.053 | 0.020 |
| global | scVI RF | Cortical+Hipp. 10x | 100 | 0.899 | 0.053 | 0.020 |
| global | scVI kNN | Cortical+Hipp. SSv4 | 100 | 0.878 | 0.053 | 0.020 |
| global | scVI RF | Cortical+Hipp. SSv4 | 100 | 0.841 | 0.053 | 0.020 |
| global | scVI kNN | Motor cortex | 100 | 0.784 | 0.053 | 0.020 |
| global | scVI kNN | Whole cortex | 100 | 0.763 | 0.053 | 0.020 |
| global | scVI RF | Whole cortex | 100 | 0.755 | 0.053 | 0.020 |
| global | scVI RF | Motor cortex | 100 | 0.710 | 0.053 | 0.020 |

---

## Configuration Recommendation

### Recommended Configuration

| Dimension | Recommended value | Rationale |
|-----------|------------------|-----------|
| **Method** | **Seurat** | Wins the most individual cell types at every taxonomy level (subclass: 16 wins vs 2 for scvi_knn, 0 for scvi_rf). Statistically indistinguishable from scvi_rf at class and subclass (p > 0.05), the two most important levels. Has the lowest hippocampal contamination and most stable cutoff behaviour. scvi_rf's higher macro EMMs at family/global are fragile — they collapse completely at any confidence filtering and do not correspond to per-cell-type superiority. |
| **Reference** | **Cortical+Hipp. 10x** | Highest mean EMM across all methods at every taxonomy level (subclass 0.898, class 0.919, family 0.945, global 0.963). Appears in the Pareto front as the top-performing reference for all methods at all levels. |
| **Cutoff** | **0.25** | For Seurat: reduces hippocampal contamination from 2.73 to 2.46 spurious predictions per query with negligible recall impact (0.655→0.654 at class level). Model-estimated macro F1 drops only ~0.6 pp at class (0.860→0.854). Avoids the severe contamination seen at cutoff=0 while preserving recall. |
| **Subsample_ref** | **100** | Statistically indistinguishable from 500 at class, family, and subclass (all p > 0.05). Small advantage for 500 at global level (OR=1.07) but absolute EMM difference is <0.003. Seurat ref processing at subsample_ref=100 takes 0.052 hrs vs 0.117 hrs at 500 — a 2.3× speed-up for no meaningful accuracy cost. |

### Raw Performance for the Recommended Configuration

Observed mean values from `aggregated_results/files/sample_results_summary.tsv`: Seurat, Cortical+Hipp. 10x, cutoff=0.25, subsample_ref=100.

| key | macro_f1_mean | macro_precision_mean | macro_recall_mean |
|-----|--------------|---------------------|------------------|
| class | 0.828 | 0.797 | 0.948 |
| family | 0.908 | 0.894 | 0.956 |
| global | 0.898 | 0.867 | 0.975 |
| subclass | 0.830 | 0.802 | 0.948 |

### Compute Time for the Recommended Configuration

Seurat, subsample_ref=100.

| step | mean_duration (hrs) | mean_memory (GB) |
|------|---------------------|-----------------|
| Query Processing | 0.032 | 0.025 |
| Ref Processing | 0.052 | 0.040 |
| Prediction | 0.019 | 0.021 |

### Trade-Off Narrative

Choosing Seurat over scvi_rf sacrifices the higher macro F1 EMMs at family (0.906 vs 0.923) and global (0.940 vs 0.958) levels when evaluated at cutoff=0, but this trade-off is illusory in practice: scvi_rf's advantage disappears at any cutoff above zero and it wins zero individual cell-type comparisons in the rankings. Seurat requires ~0.10 hrs total per query set vs ~0.05 hrs for scVI methods — roughly 2× slower, though the absolute difference is small in practice. Using cutoff=0.25 reduces macro recall modestly relative to cutoff=0 (observed recall ~0.95 at class) but eliminates some hippocampal contamination in cortex-only samples without a meaningful macro F1 cost for Seurat. Rare and difficult cell types (OPC, Microglia, Neural stem cell, Macrophage) remain poorly annotated regardless of method or configuration — these likely require specialised models or additional reference data.

> **Reference coverage caveat:** References differ in which cell types they include. A reference that lacks a label entirely (ref_support=0 in label results) cannot annotate that cell type, forcing those cells into incorrect categories and inflating false positives for other labels. This biases macro F1 comparisons in favour of references that happen to cover the cell types present in each query dataset. Notably, the cell-type rankings reveal this indirectly: Neural stem cell is best annotated by the Motor cortex reference (not Cortical+Hipp. 10x), and hippocampal subtypes (DG, CA1-ProS, CA3) are best annotated by the Whole cortex reference — suggesting these labels are absent or underrepresented in Cortical+Hipp. 10x. The recommendation of Cortical+Hipp. 10x as the overall best reference holds for cortex-focused studies, but for datasets containing Neural stem cells or hippocampal subtypes a supplementary or alternative reference should be considered. A full reference coverage table (which labels have ref_support=0 per reference) is pending.

### Pareto Note

The recommended configuration (Seurat, Cortical+Hipp. 10x, subsample_ref=100) appears in the Pareto-optimal front for **all four taxonomy levels** (subclass mean_f1=0.852, class mean_f1=0.826, family mean_f1=0.912, global mean_f1=0.940), confirming it as the best F1/compute trade-off among Seurat configurations. The cutoff of 0.25 is evaluated separately from the Pareto analysis (which uses raw performance at the best observed configuration), but model EMM data confirm that a 0.25 cutoff introduces negligible F1 loss for Seurat while providing contamination reduction.
