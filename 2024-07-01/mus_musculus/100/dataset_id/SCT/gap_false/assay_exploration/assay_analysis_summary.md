# Assay Type Analysis: Single-Cell vs Single-Nucleus

## Overview

We tested whether benchmarking performance (macro F1) differs between single-cell (sc) and single-nucleus (sn) query samples, and whether the reference atlas suspension type (cell-only vs cell+nucleus) matters.

## Data

- **Query studies**: 5 single-cell (GSE124952, GSE181021.2, GSE199460.2, GSE247339.1, GSE247339.2), 2 single-nucleus (GSE185454, GSE214244.1)
- **References**: 2 cell-only (10x, SMART-Seq v4 Allen atlas), 2 cell+nucleus (BICCN MOp, whole cortex)
- **Sample sizes**: 11,208 sc samples, 1,260 sn samples (at cutoff=0: 1,656 total per key)

## Model

Ordinal beta regression (Kubinec 2023) fit separately per taxonomy level:

```
macro_f1 ~ ref_type * query_type + cutoff
```

No random effect for study since query_type is perfectly nested within study (each study is either sc or sn), and only 7 studies total.

## Results

### Estimated Marginal Means (at cutoff=0)

| Key | Query | cell only | cell + nucleus | FDR-adj p |
|-----|-------|-----------|----------------|-----------|
| subclass | single-cell | 0.603 | 0.668 | *** |
| subclass | single-nucleus | 0.658 | 0.658 | ns |
| class | single-cell | 0.575 | 0.645 | *** |
| class | single-nucleus | 0.672 | 0.694 | ns |
| family | single-cell | 0.702 | 0.746 | *** |
| family | single-nucleus | 0.808 | 0.839 | *** |
| global | single-cell | 0.706 | 0.748 | *** |
| global | single-nucleus | 0.892 | 0.888 | ns |

P-values are FDR-corrected across all 8 within-query-type contrasts (2 per key x 4 keys).

### Key Findings

1. **Cell+nucleus references improve performance for single-cell queries** across all taxonomy levels (+0.04 to +0.07 F1).

2. **Reference type does not matter for single-nucleus queries** at subclass, class, or global levels (ns after FDR correction). Only at family level do sn queries benefit from cell+nucleus references.

3. **The interaction (ref_type:query_type) is significant** at subclass (p=1.7e-10), class (p=1.8e-5), and global (p=4.5e-5), but not at family (p=0.82). This means the reference type advantage is specific to single-cell queries at most granularity levels.

4. **Single-nucleus queries consistently outperform single-cell queries**, especially at coarser levels (global: ~0.89 vs ~0.73). This likely reflects lower transcriptomic complexity in nuclei (fewer genes, less ambient RNA contamination).

5. **Cutoff has a strong negative effect** on F1 across all levels (coefficient ~ -1.5 to -1.8 on logit scale), confirming that higher confidence thresholds reduce overall performance.

### Interpretation

The cell+nucleus references likely help single-cell queries because these references contain both cell and nucleus expression profiles, providing better coverage of expression variation. Single-nucleus queries already perform well regardless of reference type, possibly because nuclear expression profiles are a subset of whole-cell profiles and thus adequately represented in either reference type. The performance ceiling for sn queries may also leave less room for improvement.

## Files

- `files/assay_model_coefs.tsv` - Model coefficients per key
- `files/assay_emmeans_summary.tsv` - Estimated marginal means per key
- `files/assay_emmeans_contrasts.tsv` - All pairwise contrasts
- `macro_f1_query_x_ref.png` - Boxplot with significance brackets
- `macro_f1_by_study.png` - Per-study boxplots colored by assay type
- `macro_f1_by_study_query_ref.png` - Per-study, faceted by query type, colored by ref type
- `figures/<key>/` - QQ plots, dispersion plots, emmeans plots, coefficient plots per key
