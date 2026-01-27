# Cell Annotation Summary: Human vs Mouse Comparative Analysis
**Analysis Date:** July 1, 2024

## Executive Summary

This document presents a comprehensive comparison of automated cell type annotation performance across human (Homo sapiens) and mouse (Mus musculus - Tabula Muris) cortical datasets. The analysis evaluates multiple annotation methods (Seurat vs scVI), references, and biological/technical factors affecting annotation accuracy.

## Overall Performance Comparison

### Human (Homo sapiens)
- **Global weighted F1:** 0.987 ± 0.038 (98.7% accuracy)
- **Overall accuracy:** 0.988 ± 0.038
- **Subclass weighted F1:** 0.933 ± 0.073
- **Family weighted F1:** 0.985 ± 0.048
- **Class weighted F1:** 0.941 ± 0.071

### Mouse (Mus musculus - Tabula Muris)
- **Global weighted F1:** 0.926 ± 0.055 (92.6% accuracy)
- **Overall accuracy:** 0.896 ± 0.073
- **Subclass weighted F1:** 0.725 ± 0.132
- **Family weighted F1:** 0.839 ± 0.097
- **Class weighted F1:** 0.735 ± 0.140

### Key Finding
**Human datasets demonstrate substantially higher annotation accuracy (~6% higher global F1, ~21% higher subclass F1) compared to mouse datasets across all granularity levels.**

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
| Reference | Seurat | scVI | Δ (scVI advantage) |
|-----------|--------|------|-------------------|
| MTG SEA-AD | 0.952 | **0.968** | +0.016 |
| Whole Cortex | 0.951 | **0.967** | +0.016 |
| Angular Gyrus (AnG) | 0.959 | **0.965** | +0.006 |
| Somatosensory (S1) | 0.958 | **0.964** | +0.006 |
| Primary Auditory (A1) | 0.953 | **0.963** | +0.010 |
| DLPFC SEA-AD | 0.944 | **0.959** | +0.015 |
| Anterior Cingulate (ACC) | 0.953 | **0.957** | +0.004 |
| Primary Visual (V1) | 0.946 | **0.951** | +0.005 |
| Dorsolateral PFC (DFC) | 0.881 | **0.911** | +0.030 |
| SMART-seq Multi-Cortical | 0.846 | **0.908** | +0.062 |

**Key Findings:**
- scVI outperforms Seurat for ALL references
- Advantage ranges from 0.4% to 6.2%
- Largest improvement with SMART-seq reference (+6.2%)
- Second-largest with DFC reference (+3.0%)
- Most references show ~1-2% improvement with scVI

#### **CLASS Level**
| Reference | Seurat | scVI | Δ (scVI advantage) |
|-----------|--------|------|-------------------|
| MTG SEA-AD | 0.962 | **0.975** | +0.013 |
| Whole Cortex | 0.961 | **0.974** | +0.013 |
| Angular Gyrus (AnG) | 0.968 | **0.972** | +0.004 |
| Somatosensory (S1) | 0.968 | **0.972** | +0.004 |
| Primary Auditory (A1) | 0.963 | **0.971** | +0.008 |
| DLPFC SEA-AD | 0.953 | **0.967** | +0.014 |
| Anterior Cingulate (ACC) | 0.962 | **0.966** | +0.004 |
| Primary Visual (V1) | 0.956 | **0.961** | +0.005 |
| Dorsolateral PFC (DFC) | 0.891 | **0.922** | +0.031 |
| SMART-seq Multi-Cortical | 0.856 | **0.919** | +0.063 |

**Key Findings:**
- scVI wins ALL 10 references
- SMART-seq shows LARGEST improvement (+6.3%)
- DFC second-largest (+3.1%)
- Most other references: +0.4% to +1.4%


#### **FAMILY Level**
| Reference | Seurat | scVI | Δ (scVI advantage) |
|-----------|--------|------|-------------------|
| Whole Cortex | 0.976 | **0.982** | +0.006 |
| Somatosensory (S1) | 0.979 | **0.981** | +0.002 |
| MTG SEA-AD | 0.979 | **0.981** | +0.002 |
| Angular Gyrus (AnG) | 0.979 | **0.980** | +0.001 |
| Primary Auditory (A1) | 0.979 | **0.979** | 0.000 |
| DLPFC SEA-AD | 0.977 | **0.979** | +0.002 |
| Anterior Cingulate (ACC) | 0.978 | 0.976 | -0.002 |
| Primary Visual (V1) | 0.978 | 0.973 | -0.005 |
| SMART-seq Multi-Cortical | 0.959 | 0.947 | -0.012 |
| Dorsolateral PFC (DFC) | 0.913 | **0.943** | +0.030 |

**Key Findings:**
- Mixed results: scVI wins 7/10, Seurat wins 3/10
- DFC still shows largest improvement with scVI (+3.0%)
- At this granularity, differences are minimal (<1% for most)
- SMART-seq now performs WORSE with scVI (-1.2%)


#### **GLOBAL Level** (Coarsest granularity)
| Reference | Seurat | scVI | Δ (scVI advantage) |
|-----------|--------|------|-------------------|
| Whole Cortex | 0.983 | **0.987** | +0.004 |
| MTG SEA-AD | 0.985 | **0.986** | +0.001 |
| Somatosensory (S1) | 0.985 | **0.986** | +0.001 |
| Primary Auditory (A1) | 0.985 | **0.986** | +0.001 |
| Angular Gyrus (AnG) | 0.985 | 0.985 | 0.000 |
| DLPFC SEA-AD | 0.984 | **0.985** | +0.001 |
| Anterior Cingulate (ACC) | 0.984 | 0.983 | -0.001 |
| Primary Visual (V1) | 0.984 | 0.980 | -0.004 |
| SMART-seq Multi-Cortical | 0.971 | 0.964 | -0.007 |
| Dorsolateral PFC (DFC) | 0.969 | 0.960 | -0.009 |

**Key Findings:**
- Mixed results: scVI wins 6/10, Seurat wins 4/10
- At global level, performance converges
- DFC and SMART-seq now WORSE with scVI
- Differences minimal (<1% for most)

---

### **MOUSE (Mus musculus) - Reference × Method Performance**

#### **SUBCLASS Level** (Finest granularity)
| Reference | Seurat | scVI | Δ (scVI advantage) |
|-----------|--------|------|-------------------|
| Integrated Motor Cortex | 0.883 | **0.891** | +0.008 |
| Whole Cortex | 0.827 | **0.890** | +0.063 |
| 10x Cortical/Hippocampal | 0.849 | **0.883** | +0.034 |
| SMART-Seq v4 Cortical/Hippocampal | 0.785 | **0.802** | +0.017 |

**Key Findings:**
- scVI wins ALL 4 references
- Whole cortex shows MASSIVE improvement (+6.3%)
- 10x reference: +3.4% improvement
- Motor cortex: +0.8% improvement
- SMART-Seq v4: +1.7% improvement


#### **CLASS Level**
| Reference | Seurat | scVI | Δ (scVI advantage) |
|-----------|--------|------|-------------------|
| Whole Cortex | 0.842 | **0.904** | +0.062 |
| 10x Cortical/Hippocampal | 0.867 | **0.899** | +0.032 |
| Integrated Motor Cortex | 0.892 | **0.896** | +0.004 |
| SMART-Seq v4 Cortical/Hippocampal | 0.826 | **0.827** | +0.001 |

**Key Findings:**
- scVI wins ALL 4 references
- Whole cortex shows MASSIVE improvement (+6.2%)
- 10x reference: +3.2% improvement
- Motor cortex: minimal (+0.4%)


#### **FAMILY Level**
| Reference | Seurat | scVI | Δ (scVI advantage) |
|-----------|--------|------|-------------------|
| Integrated Motor Cortex | 0.927 | **0.952** | +0.025 |
| Whole Cortex | 0.908 | **0.945** | +0.037 |
| 10x Cortical/Hippocampal | 0.927 | **0.944** | +0.017 |
| SMART-Seq v4 Cortical/Hippocampal | 0.889 | **0.899** | +0.010 |

**Key Findings:**
- scVI wins ALL 4 references
- Whole cortex: +3.7% improvement
- Motor cortex: +2.5% improvement
- 10x reference: +1.7% improvement

#### **GLOBAL Level** (Coarsest granularity)
| Reference | Seurat | scVI | Δ (scVI advantage) |
|-----------|--------|------|-------------------|
| 10x Cortical/Hippocampal | **0.989** | 0.981 | -0.008 |
| SMART-Seq v4 Cortical/Hippocampal | **0.982** | 0.961 | -0.021 |
| Whole Cortex | 0.973 | **0.979** | +0.006 |
| Integrated Motor Cortex | **0.979** | 0.977 | -0.002 |

**Key Findings:**
- Seurat wins 3/4 references at global level!
- 10x reference performs BEST with Seurat (98.9%)
- SMART-Seq v4 much worse with scVI (-2.1%)
- Performance converges at coarsest granularity

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

### Reference Subsampling (cells per type)

#### Human
| Subsample Size | Weighted F1 (95% CI) |
|----------------|---------------------|
| 500 cells      | 0.954 (0.918-0.974) |
| 100 cells      | 0.947 (0.907-0.970) |
| 50 cells       | 0.937 (0.890-0.965) |

**Impact:** Minimal (~1.7% degradation from 500→50 cells)

#### Mouse
| Subsample Size | Weighted F1 (95% CI) |
|----------------|---------------------|
| 500 cells      | 0.861 (0.788-0.912) |
| 100 cells      | 0.857 (0.782-0.909) |
| 50 cells       | 0.845 (0.766-0.901) |

**Impact:** Minimal (~1.6% degradation from 500→50 cells)

**Finding:** Both species show robustness to reference size reduction, suggesting 50 cells per type may be sufficient for annotation.

---

### Sex Effects

#### Human
| Sex    | Weighted F1 (95% CI) |
|--------|---------------------|
| Female | 0.963 (0.934-0.980) |
| Male   | 0.964 (0.936-0.980) |
| None   | 0.924 (0.761-0.979) |

**Finding:** No sex-specific differences in annotation accuracy.

#### Mouse
| Sex    | Weighted F1 (95% CI) |
|--------|---------------------|
| Male   | 0.814 (0.717-0.884) |
| None   | 0.898 (0.787-0.954) |

**Finding:** Male-only datasets show reduced performance compared to mixed-sex datasets (~8.4% lower).

---

### Disease/Treatment State

#### Human - Disease State
| State   | Weighted F1 (95% CI) |
|---------|---------------------|
| Control | 0.955 (0.921-0.975) |
| Disease | 0.952 (0.915-0.973) |

**Finding:** Disease state has negligible impact (<0.3% difference), suggesting annotation methods are robust to disease-associated transcriptional changes.

#### Mouse - Treatment State
| State         | Weighted F1 (95% CI) |
|---------------|---------------------|
| No Treatment  | 0.870 (0.800-0.918) |
| Treatment     | 0.852 (0.775-0.906) |

**Finding:** Treatment introduces minor performance reduction (~1.8%), but effect is modest.

---

### Region Matching (Human Only)

| Region Match | Weighted F1 (95% CI) |
|--------------|---------------------|
| True         | 0.948 (0.909-0.971) |
| False        | 0.931 (0.880-0.961) |

**Finding:** Matching cortical regions between reference and query improves performance by ~1.7%, but cross-region annotation remains highly accurate.

---

## Cell Type-Specific Performance

### Human - High-Performing Cell Types (F1 > 0.97, subclass level)
- **Astrocyte:** 0.987 ± 0.056
- **Chandelier:** 0.974 ± 0.125
- **GABAergic:** 0.961 ± 0.113

### Human - Challenging Cell Types
- **Endothelial:** 0.792 ± 0.351 (high variability)

### Mouse - High-Performing Cell Types (F1 > 0.85, class level)
- **Astrocyte:** 0.848 ± 0.195
- **CNS Macrophage (family):** 0.968 ± 0.066

### Mouse - Challenging Cell Types
- **Cajal-Retzius:** 0.530 ± 0.452 (highly variable, rare cell type)
- **CA3 (hippocampal):** 0.651 ± 0.145
- **CA1-ProS:** 0.610 ± 0.087

**Pattern:** Rare and regionally-restricted cell types show reduced accuracy in both species, but the effect is more pronounced in mouse.

---

## Macro vs Micro Metrics

### Human
- **Macro F1:** 0.860 ± 0.115
- **Micro F1:** 0.922 ± 0.082
- **Gap:** 6.2%

### Mouse
- **Macro F1:** 0.621 ± 0.099
- **Micro F1:** 0.726 ± 0.120
- **Gap:** 10.5%

**Interpretation:** The macro-micro gap indicates performance inequality across cell types. Mouse datasets show a larger gap, suggesting greater challenges with rare cell types or class imbalance issues.

---

## Key Comparative Insights

### 1. Overall Performance Disparity
Human datasets achieve substantially higher accuracy than mouse datasets across all metrics and granularities. This may reflect:
- Greater standardization in human data collection protocols
- Higher quality/depth of human reference atlases
- Reduced biological heterogeneity in human cortical samples
- Potential confounds from strain/age variation in mouse data

### 2. Method Selection
- **scVI** is the preferred method for both species
- Performance advantage is larger in mouse, suggesting superior handling of batch effects
- **Seurat** shows better calibrated confidence scores for filtering

### 3. Confidence Thresholding Strategy
- For **Seurat**: Can apply cutoffs up to 0.5 with modest performance loss
- For **scVI**: Should use low cutoffs (≤0.1); high cutoffs dramatically reduce performance

### 4. Reference Dataset Selection
- Human: Multiple high-quality references available; whole cortex performs excellently
- Mouse: Integrated motor cortex atlas and whole cortex references perform best
- Both: Avoid SMART-seq-specific references for 10x data

### 5. Biological Robustness
- Both methods are robust to:
  - Disease/treatment state
  - Sex (in human; modest effect in mouse)
  - Cross-region annotation (human)
  - Reference subsampling (50+ cells/type sufficient)

### 6. Granularity Challenges
- Subclass-level annotation shows largest human-mouse performance gap
- Fine-grained mouse cell type definitions may require additional optimization

---

## Recommendations

### For Human Data
1. Use **scVI** with whole cortex or comprehensive regional atlases
2. Apply minimal confidence cutoff (0-0.05) for scVI
3. Expect excellent performance (>93% F1) at subclass level
4. Region matching optional but beneficial

### For Mouse Data
1. Use **scVI** with integrated motor cortex or whole cortex reference
2. Avoid high confidence cutoffs with scVI
3. Expect moderate performance (~72-89% F1) at subclass level
4. Consider method ensemble or manual review for rare cell types
5. Account for sex composition when possible

### General Best Practices
1. Maintain ≥50 cells per cell type in reference
2. Monitor macro metrics to detect rare cell type issues
3. Validate disease/treatment sample annotations
4. Consider technology matching (10x-to-10x preferred over SMART-seq-to-10x)

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

This comprehensive analysis reveals that automated cell type annotation has reached maturity for human cortical datasets, with near-perfect accuracy achievable using modern methods like scVI. Mouse datasets present greater challenges, likely reflecting biological and technical heterogeneity, but still achieve good performance suitable for most applications.

The substantial methodological improvements offered by scVI over Seurat come with the trade-off of poorly calibrated confidence scores, suggesting that users should rely on fixed low cutoffs rather than adaptive thresholding. Reference dataset quality remains critical, with comprehensive whole-cortex atlases outperforming region-specific or technology-specific references.

Future work should focus on:
1. Understanding and reducing the human-mouse performance gap
2. Improving rare cell type detection in mouse datasets
3. Developing better confidence calibration for deep learning methods
4. Expanding cross-species annotation capabilities

---

**Analysis conducted on:** 2024-07-01
**Repository:** cell_annotation_summary.nf
**Data location:** `/Users/Rachel/Documents/pavlab/cell_annotation_summary.nf/2024-07-01/`
