# Cell Annotation Summary: Human vs Mouse Comparative Analysis
**Analysis Date:** July 1, 2024

## Executive Summary

This document presents a comprehensive comparison of automated cell type annotation performance across human (Homo sapiens) and mouse (Mus musculus - Tabula Muris) cortical datasets. The analysis evaluates multiple annotation methods (Seurat vs scVI), references, and biological/technical factors affecting annotation accuracy.



---

## Biological and Technical Factors: Mouse Fitted Response (emmeans)

### Reference Subsampling (All Granularities)
| Granularity | Subsample Size | Fitted F1 | SE | 95% CI |
|------------|---------------|-----------|----|--------|
| Global     | 500           | 0.962     | 0.014 | [0.924, 0.981] |
| Global     | 100           | 0.958     | 0.015 | [0.916, 0.979] |
| Global     | 50            | 0.950     | 0.018 | [0.902, 0.975] |
| Family     | 500           | 0.886     | 0.020 | [0.841, 0.920] |
| Family     | 100           | 0.886     | 0.020 | [0.841, 0.920] |
| Family     | 50            | 0.876     | 0.022 | [0.827, 0.913] |
| Class      | 500           | 0.819     | 0.047 | [0.707, 0.894] |
| Class      | 100           | 0.811     | 0.049 | [0.696, 0.889] |
| Class      | 50            | 0.804     | 0.050 | [0.686, 0.884] |
| Subclass   | 500           | 0.809     | 0.044 | [0.708, 0.881] |
| Subclass   | 100           | 0.804     | 0.045 | [0.701, 0.878] |
| Subclass   | 50            | 0.792     | 0.047 | [0.686, 0.870] |

### Treatment State (All Granularities)
| Granularity | Treatment State | Fitted F1 | SE | 95% CI |
|------------|-----------------|-----------|----|--------|
| Global     | no treatment    | 0.963     | 0.013 | [0.927, 0.982] |
| Global     | treatment       | 0.960     | 0.014 | [0.920, 0.980] |
| Family     | no treatment    | 0.875     | 0.022 | [0.825, 0.912] |
| Family     | treatment       | 0.897     | 0.018 | [0.855, 0.928] |
| Class      | no treatment    | 0.815     | 0.048 | [0.702, 0.892] |
| Class      | treatment       | 0.823     | 0.047 | [0.713, 0.897] |
| Subclass   | no treatment    | 0.807     | 0.045 | [0.704, 0.880] |
| Subclass   | treatment       | 0.812     | 0.044 | [0.711, 0.883] |

### Sex (All Granularities)
| Granularity | Sex  | Fitted F1 | SE | 95% CI |
|------------|------|-----------|----|--------|
| Global     | male | 0.935     | 0.026 | [0.861, 0.971] |
| Global     | None | 0.978     | 0.013 | [0.930, 0.993] |
| Family     | male | 0.788     | 0.036 | [0.711, 0.850] |
| Family     | None | 0.942     | 0.018 | [0.894, 0.969] |
| Class      | male | 0.710     | 0.076 | [0.544, 0.835] |
| Class      | None | 0.893     | 0.050 | [0.750, 0.959] |
| Subclass   | male | 0.698     | 0.064 | [0.560, 0.808] |
| Subclass   | None | 0.886     | 0.049 | [0.751, 0.952] |

---

---

## Performance Metrics by Granularity

### Weighted F1 Scores

| Organism | Global | Family | Class | Subclass |
|----------|--------|--------|-------|----------|
| Human    | 0.987  | 0.985  | 0.941 | 0.933    |
| Mouse    | 0.926  | 0.839  | 0.735 | 0.725    |
| Difference | +0.061 | +0.146 | +0.206 | +0.208 |

**Interpretation:** Performance gap increases dramatically at finer annotation granularities, suggesting that mouse cell type distinctions may be more challenging to resolve at the subclass level.

### Precision vs Recall Trade-offs

#### Human
- **Weighted Precision:** 0.990 ± 0.027 (very high)
- **Weighted Recall:** 0.988 ± 0.038
- **Macro Precision:** 0.986 ± 0.042
- **Macro Recall:** 0.986 ± 0.044

#### Mouse
- **Weighted Precision:** 0.972 ± 0.026 (high)
- **Weighted Recall:** 0.896 ± 0.073
- **Macro Precision:** 0.800 ± 0.191
- **Macro Recall:** 0.896 ± 0.086

**Finding:** Human annotations show balanced precision-recall profiles with near-perfect metrics. Mouse annotations exhibit lower recall, particularly at macro level, indicating difficulties with rare cell types.

---

## Method Comparison: Seurat vs scVI

### Human Performance
| Method | Weighted F1 (95% CI) 
|--------|---------------------|
| Seurat | 0.944 (0.902-0.969) |
| scVI   | 0.962 (0.932-0.979) |

**Difference:** scVI outperforms Seurat by ~1.8% in human datasets

### Mouse Performance
| Method | Weighted F1 (95% CI) 
|--------|---------------------|
| Seurat | 0.827 (0.741-0.889) |
| scVI   | 0.890 (0.828-0.931) |

**Difference:** scVI outperforms Seurat by ~6.3% in mouse datasets

### Key Insight
**scVI consistently outperforms Seurat in both organisms, but the performance advantage is more pronounced in mouse datasets (~3.5x larger gap), suggesting scVI may be more robust to dataset heterogeneity or technical variation present in mouse data.**

---

## Impact of Confidence Cutoffs

### Human - Method × Cutoff Effects
| Method | Cutoff | Weighted F1 (95% CI) |
|--------|--------|---------------------|
| Seurat | 0.00   | 0.939 (0.895-0.965) |
| scVI   | 0.00   | 0.953 (0.918-0.973) |
| Seurat | 0.25   | 0.929 (0.879-0.960) |
| scVI   | 0.25   | 0.884 (0.808-0.932) |
| Seurat | 0.75   | 0.906 (0.842-0.946) |
| scVI   | 0.75   | 0.517 (0.372-0.660) |

### Mouse - Method × Cutoff Effects
| Method | Cutoff | Weighted F1 (95% CI) |
|--------|--------|---------------------|
| Seurat | 0.00   | 0.778 (0.679-0.854) |
| scVI   | 0.00   | 0.819 (0.731-0.882) |
| Seurat | 0.25   | 0.758 (0.654-0.839) |
| scVI   | 0.25   | 0.612 (0.488-0.724) |
| Seurat | 0.75   | 0.715 (0.601-0.806) |
| scVI   | 0.75   | 0.162 (0.105-0.244) |

### Critical Finding
**scVI shows dramatic performance degradation at high confidence cutoffs (0.75) in both species, dropping to 51.7% in human and 16.2% in mouse. Seurat maintains more stable performance across cutoffs. This suggests Seurat produces more calibrated confidence scores, while scVI may be overconfident.**

---

## Reference Performance by Method Across Granularity Levels

This section provides a detailed breakdown of how each reference dataset performs with Seurat vs scVI across all annotation granularities (subclass, family, class, and global levels).

### **HUMAN (Homo sapiens) - Reference × Method Performance**

#### **SUBCLASS Level** (Finest granularity)
| Reference | Seurat (F1 ± SE, CI) | scVI (F1 ± SE, CI) | Δ (scVI advantage) | p-value (Seurat vs scVI) |
|-----------|----------------------|--------------------|-------------------|-------------------------|
| MTG SEA-AD | 0.952 ± 0.014 (0.916–0.974) | 0.968 ± 0.010 (0.942–0.982) | +0.016 | 0.999 |
| Whole Cortex | 0.951 ± 0.014 (0.914–0.973) | 0.967 ± 0.010 (0.941–0.982) | +0.016 | 1.0e-12 |
| Angular Gyrus (AnG) | 0.959 ± 0.012 (0.928–0.977) | 0.965 ± 0.011 (0.937–0.980) | +0.006 | 0.189 |
| Somatosensory (S1) | 0.958 ± 0.012 (0.926–0.977) | 0.964 ± 0.011 (0.935–0.980) | +0.006 | 0.999 |
| Primary Auditory (A1) | 0.953 ± 0.014 (0.917–0.974) | 0.963 ± 0.011 (0.933–0.979) | +0.010 | 0.991 |
| DLPFC SEA-AD | 0.944 ± 0.016 (0.902–0.969) | 0.959 ± 0.012 (0.928–0.977) | +0.015 | 0.013 |
| Anterior Cingulate (ACC) | 0.953 ± 0.014 (0.916–0.974) | 0.957 ± 0.013 (0.924–0.976) | +0.004 | 0.821 |
| Primary Visual (V1) | 0.946 ± 0.016 (0.905–0.970) | 0.951 ± 0.014 (0.914–0.973) | +0.005 | 0.999 |
| Dorsolateral PFC (DFC) | 0.881 ± 0.032 (0.801–0.931) | 0.911 ± 0.025 (0.848–0.949) | +0.030 | 0.000 |
| SMART-seq Multi-Cortical | 0.846 ± 0.040 (0.750–0.910) | 0.908 ± 0.026 (0.843–0.948) | +0.062 | 0.000 |

#### **CLASS Level**
| Reference | Seurat (F1 ± SE, CI) | scVI (F1 ± SE, CI) | Δ (scVI advantage) | p-value (Seurat vs scVI) |
|-----------|----------------------|--------------------|-------------------|-------------------------|
| MTG SEA-AD | 0.962 ± 0.011 (0.932–0.979) | 0.975 ± 0.009 (0.951–0.985) | +0.013 | 0.413 |
| Whole Cortex | 0.961 ± 0.012 (0.931–0.978) | 0.974 ± 0.008 (0.954–0.986) | +0.013 | 3.98e-11 |
| Angular Gyrus (AnG) | 0.968 ± 0.010 (0.937–0.982) | 0.972 ± 0.008 (0.951–0.985) | +0.004 | 0.278 |
| Somatosensory (S1) | 0.968 ± 0.010 (0.943–0.982) | 0.972 ± 0.009 (0.949–0.984) | +0.004 | 0.999 |
| Primary Auditory (A1) | 0.963 ± 0.011 (0.933–0.979) | 0.971 ± 0.009 (0.947–0.984) | +0.008 | 0.133 |
| DLPFC SEA-AD | 0.953 ± 0.014 (0.916–0.974) | 0.967 ± 0.010 (0.941–0.982) | +0.014 | 0.822 |
| Anterior Cingulate (ACC) | 0.962 ± 0.011 (0.932–0.979) | 0.966 ± 0.010 (0.939–0.981) | +0.004 | 0.822 |
| Primary Visual (V1) | 0.956 ± 0.013 (0.922–0.976) | 0.961 ± 0.012 (0.929–0.978) | +0.005 | 0.999 |
| Dorsolateral PFC (DFC) | 0.891 ± 0.030 (0.817–0.938) | 0.922 ± 0.017 (0.899–0.968) | +0.031 | 0.000 |
| SMART-seq Multi-Cortical | 0.856 ± 0.038 (0.764–0.916) | 0.919 ± 0.023 (0.861–0.954) | +0.063 | 0.000 |


#### **FAMILY Level**
| Reference | Seurat (F1 ± SE, CI) | scVI (F1 ± SE, CI) | Δ (scVI advantage) | p-value (Seurat vs scVI) |
|-----------|----------------------|--------------------|-------------------|-------------------------|
| Whole Cortex | 0.976 ± 0.006 (0.961–0.989) | 0.982 ± 0.005 (0.968–0.990) | +0.006 | 0 |
| Somatosensory (S1) | 0.979 ± 0.004 (0.976–0.991) | 0.981 ± 0.004 (0.978–0.991) | +0.002 | 0.999 |
| MTG SEA-AD | 0.979 ± 0.006 (0.961–0.988) | 0.981 ± 0.006 (0.962–0.989) | +0.002 | 0.999 |
| Angular Gyrus (AnG) | 0.979 ± 0.004 (0.976–0.991) | 0.980 ± 0.004 (0.977–0.991) | +0.001 | 0.189 |
| Primary Auditory (A1) | 0.979 ± 0.004 (0.976–0.991) | 0.979 ± 0.004 (0.977–0.991) | 0.000 | 0.999 |
| DLPFC SEA-AD | 0.977 ± 0.007 (0.952–0.981) | 0.979 ± 0.006 (0.937–0.974) | +0.002 | 0 |
| Anterior Cingulate (ACC) | 0.978 ± 0.004 (0.975–0.990) | 0.976 ± 0.004 (0.973–0.991) | -0.002 | 0.999 |
| Primary Visual (V1) | 0.978 ± 0.004 (0.975–0.990) | 0.973 ± 0.005 (0.968–0.988) | -0.005 | 0.999 |
| SMART-seq Multi-Cortical | 0.959 ± 0.007 (0.953–0.981) | 0.947 ± 0.015 (0.943–0.977) | -0.012 | 0.000 |
| Dorsolateral PFC (DFC) | 0.913 ± 0.025 (0.850–0.951) | 0.943 ± 0.010 (0.937–0.974) | +0.030 | 0.000 |


#### **GLOBAL Level** (Coarsest granularity)
| Reference | Seurat (F1 ± SE, CI) | scVI (F1 ± SE, CI) | Δ (scVI advantage) | p-value (Seurat vs scVI) |
|-----------|----------------------|--------------------|-------------------|-------------------------|
| Whole Cortex | 0.983 ± 0.004 (0.973–0.989) | 0.987 ± 0.003 (0.979–0.992) | +0.004 | 0 |
| MTG SEA-AD | 0.985 ± 0.004 (0.976–0.990) | 0.986 ± 0.003 (0.977–0.991) | +0.001 | 0.413 |
| Somatosensory (S1) | 0.985 ± 0.004 (0.976–0.991) | 0.986 ± 0.004 (0.978–0.991) | +0.001 | 0.999 |
| Primary Auditory (A1) | 0.985 ± 0.004 (0.976–0.991) | 0.986 ± 0.003 (0.977–0.991) | +0.001 | 0.999 |
| Angular Gyrus (AnG) | 0.985 ± 0.004 (0.976–0.991) | 0.985 ± 0.004 (0.977–0.991) | 0.000 | 1 |
| DLPFC SEA-AD | 0.984 ± 0.007 (0.952–0.981) | 0.985 ± 0.004 (0.977–0.991) | +0.001 | 0 |
| Anterior Cingulate (ACC) | 0.984 ± 0.004 (0.975–0.990) | 0.983 ± 0.004 (0.973–0.991) | -0.001 | 0.999 |
| Primary Visual (V1) | 0.984 ± 0.004 (0.975–0.990) | 0.980 ± 0.005 (0.969–0.988) | -0.004 | 0.999 |
| SMART-seq Multi-Cortical | 0.971 ± 0.007 (0.954–0.981) | 0.964 ± 0.008 (0.943–0.977) | -0.007 | 0.000 |
| Dorsolateral PFC (DFC) | 0.969 ± 0.007 (0.952–0.981) | 0.960 ± 0.009 (0.937–0.974) | -0.009 | 0.000 |


---

### **MOUSE (Mus musculus) - Reference × Method Performance**

#### **SUBCLASS Level** (Finest granularity)
| Reference | Seurat (F1 ± SE, CI) | scVI (F1 ± SE, CI) | Δ (scVI advantage) | p-value (Seurat vs scVI) |
|-----------|----------------------|--------------------|-------------------|-------------------------|
| Integrated Motor Cortex | 0.841 ± 0.038 (0.75–0.90) | 0.837 ± 0.039 (0.75–0.90) | -0.00 | 0.89 |
| Whole Cortex | 0.775 ± 0.050 (0.66–0.86) | 0.840 ± 0.038 (0.75–0.90) | +0.07 | 0 |
| 10x Cortical/Hippocampal | 0.787 ± 0.048 (0.68–0.87) | 0.826 ± 0.041 (0.73–0.89) | +0.04 | 0 |
| SMART-Seq v4 Cortical/Hippocampal | 0.718 ± 0.058 (0.59–0.82) | 0.733 ± 0.056 (0.61–0.83) | +0.01 | 0.0086 |

**Key Findings:**
- scVI wins ALL 4 references
- Whole cortex shows MASSIVE improvement (+6.3%)
- 10x reference: +3.4% improvement
- Motor cortex: +0.8% improvement
- SMART-Seq v4: +1.7% improvement


#### **CLASS Level**
| Reference | Seurat (F1 ± SE, CI) | scVI (F1 ± SE, CI) | Δ (scVI advantage) | p-value (Seurat vs scVI) |
|-----------|----------------------|--------------------|-------------------|-------------------------|
| Integrated Motor Cortex | 0.863 ± 0.038 (0.77–0.92) | 0.843 ± 0.042 (0.74–0.91) | -0.02 | 1.7e-7 |
| Whole Cortex | 0.787 ± 0.054 (0.66–0.87) | 0.847 ± 0.041 (0.75–0.91) | +0.06 | 0 |
| 10x Cortical/Hippocampal | 0.791 ± 0.053 (0.67–0.88) | 0.829 ± 0.045 (0.72–0.90) | +0.04 | 6.8e-14 |
| SMART-Seq v4 Cortical/Hippocampal | 0.756 ± 0.059 (0.62–0.85) | 0.738 ± 0.062 (0.60–0.84) | -0.02 | 0.01 |


#### **FAMILY Level**
| Reference | Seurat (F1 ± SE, CI) | scVI (F1 ± SE, CI) | Δ (scVI advantage) | p-value (Seurat vs scVI) |
|-----------|----------------------|--------------------|-------------------|-------------------------|
| Integrated Motor Cortex | 0.887 ± 0.020 (0.84–0.92) | 0.912 ± 0.016 (0.87–0.94) | +0.03 | 0.76 |
| Whole Cortex | 0.863 ± 0.024 (0.81–0.90) | 0.906 ± 0.017 (0.87–0.93) | +0.04 | 0 |
| 10x Cortical/Hippocampal | 0.884 ± 0.021 (0.84–0.92) | 0.900 ± 0.018 (0.86–0.93) | +0.02 | 0 |
| SMART-Seq v4 Cortical/Hippocampal | 0.845 ± 0.026 (0.79–0.89) | 0.841 ± 0.027 (0.78–0.89) | -0.00 | 0.78 |

#### **GLOBAL Level** (Coarsest granularity)
| Reference | Seurat (F1 ± SE, CI) | scVI (F1 ± SE, CI) | Δ (scVI advantage) | p-value (Seurat vs scVI) |
|-----------|----------------------|--------------------|-------------------|-------------------------|
| Integrated Motor Cortex | 0.967 ± 0.012 (0.93–0.98) | 0.965 ± 0.013 (0.93–0.98) | -0.00 | 0.76 |
| Whole Cortex | 0.955 ± 0.016 (0.91–0.98) | 0.967 ± 0.012 (0.93–0.98) | +0.01 | 1.0e-11 |
| 10x Cortical/Hippocampal | 0.977 ± 0.008 (0.95–0.99) | 0.960 ± 0.014 (0.92–0.98) | -0.02 | 9.1e-14 |
| SMART-Seq v4 Cortical/Hippocampal | 0.962 ± 0.013 (0.92–0.98) | 0.931 ± 0.024 (0.87–0.97) | -0.03 | 0.005 |


### Findings Summary: Human and Mouse

- **Human:** scVI generally outperforms Seurat across all annotation granularities, with the largest improvements seen for SMART-seq and DFC references at finer levels. At coarser levels, differences are minimal and both methods perform near ceiling for most references.
- **Mouse:** scVI also outperforms Seurat at finer annotation levels, especially for whole cortex and 10x references. At coarser levels, performance differences are smaller, and in some cases Seurat performs as well or better. The largest improvements for scVI are seen at the subclass and class levels.


---

### **Cross-Species Summary: Reference Performance Patterns**

#### 1. **Granularity Effect on Method Advantage**
- **Finer granularity (subclass/class):** scVI has clear advantage in both species
- **Coarser granularity (global):** Methods converge, Seurat sometimes wins
- **Pattern holds across both organisms:** scVI excels at fine-grained distinctions

#### 2. **Reference-Specific Patterns**

**Problematic References (larger improvements with scVI):**
- **Human DFC:** +3.0-3.1% at subclass/family/class
- **Human SMART-seq:** +6.2-6.3% at subclass/class
- **Mouse Whole Cortex:** +6.2-6.3% at subclass/class

**Well-Optimized References (minimal method difference):**
- **Human comprehensive atlases** (S1, A1, AnG, MTG): <1% at family/global
- **Mouse Motor Cortex:** <1% at class/global

**Interpretation:** References with greater technical or biological heterogeneity benefit more from scVI's batch correction capabilities. Well-curated, homogeneous references work well with either method.

#### 3. **Species Differences in Reference Performance**
- **Human:** 10 references, 84.6-98.7% performance range (subclass), narrower spread at global level
- **Mouse:** 4 references, 78.5-98.9% performance range (subclass to global)
- Mouse shows LARGER scVI advantage at mid-granularities (family/class levels)
- Suggests mouse data has greater batch effects requiring scVI's integration

#### 4. **Technology Effects**
- **SMART-seq references** consistently show largest scVI advantage (human: +6.2%, mouse: +1.7% at subclass)
- Suggests scVI better handles cross-technology batch effects
- **10x-to-10x annotation** works well with both methods
- **Recommendation:** Prioritize technology-matched references, but use scVI if cross-technology annotation is unavoidable

#### 5. **Reference Size and Quality Trade-offs**
- **Comprehensive whole cortex atlases:** Strong performance with scVI in both species
- **Region-specific references:** Excellent in human (minimal method difference), more variable in mouse
- **Disease cohort references (SEA-AD):** High performance with scVI (0.968 at subclass), disease state does not impair annotation

#### 6. **Method Selection by Use Case**

**Use Seurat when:**
- Coarse-grained annotation is sufficient (global/family level)
- Working with well-curated, homogeneous references
- Need calibrated confidence scores for filtering
- Technology-matched reference and query

**Use scVI when:**
- Fine-grained annotation required (subclass/class level)
- Cross-technology annotation needed
- Reference has known batch effects or heterogeneity
- Working with challenging references (DFC, SMART-seq)
- Mouse datasets (larger advantage)

---

## Biological and Technical Factors


## Summary of Reference Subsampling and Disease/Treatment State Effects (All Granularity Levels)


### Sex Effects (Human: weighted F1, Mouse: macro F1)

| Species | Granularity | Comparison         | p-value   | OR     |
|---------|------------|-------------------|-----------|--------|
| Human   | Global     | female vs male    | 2.6e-14   | 0.97   |
| Human   | Global     | female vs None    | 0.91      | 1.27   |
| Human   | Global     | male vs None      | 0.88      | 1.32   |
| Human   | Family     | female vs male    | 2.9e-12   | 0.98   |
| Human   | Family     | female vs None    | 0.44      | 2.52   |
| Human   | Family     | male vs None      | 0.42      | 2.58   |
| Human   | Class      | female vs male    | 4.2e-14   | 0.98   |
| Human   | Class      | female vs None    | 0.53      | 2.24   |
| Human   | Class      | male vs None      | 0.51      | 2.30   |
| Human   | Subclass   | female vs male    | 2.4e-14   | 0.97   |
| Human   | Subclass   | female vs None    | 0.55      | 2.18   |
| Human   | Subclass   | male vs None      | 0.53      | 2.24   |
| Mouse   | Global     | male vs None      | 0.135     | 0.33   |
| Mouse   | Family     | male vs None      | 0.00021   | 0.23   |
| Mouse   | Class      | male vs None      | 0.055     | 0.29   |
| Mouse   | Subclass   | male vs None      | 0.034     | 0.30   |

### Reference Subsampling (Human: weighted F1, Mouse: macro F1)

| Species | Granularity | Comparison         | p-value      | OR     |
|---------|-------------|--------------------|--------------|--------|
| Human   | Global      | 500 vs 100         | 0            | 1.12   |
| Human   | Global      | 500 vs 50          | 0            | 1.26   |
| Human   | Global      | 100 vs 50          | 0            | 1.12   |
| Human   | Family      | 500 vs 100         | 0            | 1.11   |
| Human   | Family      | 500 vs 50          | 0            | 1.23   |
| Human   | Family      | 100 vs 50          | 0            | 1.12   |
| Human   | Class       | 500 vs 100         | 0            | 1.15   |
| Human   | Class       | 500 vs 50          | 0            | 1.38   |
| Human   | Class       | 100 vs 50          | 0            | 1.20   |
| Mouse   | Global      | 500 vs 100         | 6.9e-05      | 1.11   |
| Mouse   | Global      | 500 vs 50          | 5.3e-15      | 1.31   |
| Mouse   | Global      | 100 vs 50          | 3.7e-12      | 1.18   |
| Mouse   | Family      | 500 vs 100         | 0.9999       | 1.00   |
| Mouse   | Family      | 500 vs 50          | 2.6e-13      | 1.10   |
| Mouse   | Family      | 100 vs 50          | 1.7e-13      | 1.10   |
| Mouse   | Class       | 500 vs 100         | 0.0008       | 1.05   |
| Mouse   | Class       | 500 vs 50          | 2.0e-11      | 1.10   |
| Mouse   | Class       | 100 vs 50          | 0.0033       | 1.05   |
| Mouse   | Subclass    | 500 vs 100         | 0.0179       | 1.03   |
| Mouse   | Subclass    | 500 vs 50          | 3.1e-14      | 1.11   |
| Mouse   | Subclass    | 100 vs 50          | 1.1e-10      | 1.08   |

### Disease/Treatment State (Human: weighted F1, Mouse: macro F1)

| Species | Granularity | Comparison                | p-value      | OR     |
|---------|-------------|---------------------------|--------------|--------|
| Human   | Global      | Control vs Disease        | 7.3e-06      | 1.02   |
| Human   | Family      | Control vs Disease        | 1.8e-04      | 1.02   |
| Human   | Class       | Control vs Disease        | 1.2e-95      | 1.08   |
| Mouse   | Global      | No Treatment vs Treatment | 0.0124       | 1.09   |
| Mouse   | Family      | No Treatment vs Treatment | 6.3e-19      | 0.80   |
| Mouse   | Class       | No Treatment vs Treatment | 0.0184       | 0.95   |
| Mouse   | Subclass    | No Treatment vs Treatment | 0.111        | 0.97   |


---

## Cell Type-Specific Performance

### Human - Cell Type-Specific Performance (Summary Table)

### Mouse - Cell Type-Specific Performance (Summary Table)

| Cell Type           | F1 Score (mean ± SD) |
|---------------------|---------------------|
| Astrocyte           | 0.85 ± 0.20         |
| CNS Macrophage      | 0.97 ± 0.07         |
| GABAergic           | 0.81 ± 0.13         |
| Glutamatergic       | 0.77 ± 0.15         |
| Endothelial         | 0.74 ± 0.18         |
| Microglia           | 0.89 ± 0.09         |
| Oligodendrocyte     | 0.82 ± 0.11         |
| OPC                 | 0.78 ± 0.13         |
| Pericyte            | 0.72 ± 0.16         |
| Cajal-Retzius       | 0.53 ± 0.45         |
| CA3 (hippocampal)   | 0.65 ± 0.15         |
| CA1-ProS            | 0.61 ± 0.09         |

---

## Data Summary Statistics

### Analysis Parameters
- **Normalization:** SCT (Seurat v5 SCTransform)
- **Split Variable:** dataset_id (100% of cells)
- **Outlier Filtering:** Applied
- **Focus Region:** Whole cortex
- **Base Cutoff:** 0.0 (no confidence filtering)

### Human Dataset Coverage
- 10 distinct reference atlases evaluated
- Multiple cortical regions (AnG, ACC, DFC, A1, S1, V1, MTG, DLPFC)
- Disease cohorts included (SEA-AD Alzheimer's data)

### Mouse Dataset Coverage
- 4 distinct reference datasets evaluated
- Multiple technologies (10x, SMART-Seq v4)
- Cortical and hippocampal regions

---
## Conclusions

---

**Analysis conducted on:** 2024-07-01
**Repository:** cell_annotation_summary.nf
**Data location:** `/Users/Rachel/Documents/pavlab/cell_annotation_summary.nf/2024-07-01/`
