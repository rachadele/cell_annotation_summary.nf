# Results Summary

## Dataset overview

The benchmarking evaluation spans two organisms with distinct study compositions:

- **Mouse:** 7 GEO studies after outlier removal (4 excluded), covering nucleus accumbens, prefrontal cortex, hippocampus, entorhinal cortex, and cerebral cortex. All male or unspecified sex. Includes treatment conditions (cocaine, HDAC inhibitors, TBI) and genetic perturbations (Setd1a, App knock-in).
- **Human:** 16 GEO studies after outlier removal (1 excluded), covering primarily prefrontal cortex with some temporal, cingulate, and motor cortex. Includes neuropsychiatric and neurodegenerative conditions (ASD, Schizophrenia, Bipolar Disorder, AD, HD, ALS, FTLD, PTSD, MDD). Both sexes represented. Age spans infant through late adult.

## Overall annotation performance

Mean performance under default parameters (whole cortex reference, cutoff = 0.0, subsample_ref = 500) across both methods:

| Taxonomy level | Mouse weighted F1 | Mouse macro F1 | Human weighted F1 | Human macro F1 |
|----------------|-------------------|----------------|-------------------|----------------|
| Global         | 0.93 (0.05)       | 0.81 (0.16)    | 0.99 (0.04)       | 0.98 (0.05)    |
| Family         | 0.84 (0.10)       | 0.75 (0.09)    | 0.99 (0.05)       | 0.98 (0.06)    |
| Class          | 0.73 (0.14)       | 0.60 (0.12)    | 0.94 (0.07)       | 0.90 (0.11)    |
| Subclass       | 0.72 (0.13)       | 0.62 (0.10)    | 0.93 (0.07)       | 0.86 (0.11)    |

*Values are mean (SD). Weighted F1 weights cell types by support; macro F1 gives equal weight to each type.*

Performance degrades at finer taxonomy levels as expected: distinguishing subclass-level cell types (e.g., specific GABAergic subtypes) is harder than global-level categories (e.g., neuronal vs. non-neuronal). Human studies show substantially higher F1 than mouse, likely reflecting differences in study composition and reference atlas quality.

## Method comparison: scVI vs. Seurat

Estimated marginal means from the ordered beta regression model (at cutoff = 0, subsample_ref = 500, whole cortex reference) show scVI outperforming Seurat at all taxonomy levels in mouse:

| Taxonomy level | Seurat EMM (95% CI) | scVI EMM (95% CI) | Odds ratio (Seurat/scVI) | p-value |
|----------------|---------------------|--------------------|-----------------------------|---------|
| Global         | 0.927 (0.864, 0.963) | 0.966 (0.934, 0.983) | 0.446 | < 10^-113 |
| Family         | 0.855 (0.800, 0.897) | 0.904 (0.864, 0.933) | 0.632 | < 10^-119 |
| Class          | 0.788 (0.677, 0.868) | 0.835 (0.740, 0.900) | 0.734 | < 10^-33  |
| Subclass       | 0.773 (0.668, 0.853) | 0.822 (0.731, 0.887) | 0.739 | < 10^-63  |

*Odds ratios < 1 indicate scVI advantage. All contrasts FDR-significant.*

The scVI advantage is largest at the global level (odds ratio 0.45) and more modest at finer levels. In human data, the same pattern holds: scVI yields higher estimated marginal means than Seurat at the subclass level (0.962 vs. 0.944) and global level (0.987 vs. 0.982).

## Confidence cutoff sensitivity

The method-by-cutoff interaction is the strongest effect in the model (coefficient magnitudes -2.5 to -3.9 on the link scale, all p = 0). scVI's macro F1 degrades sharply at high confidence cutoffs, while Seurat remains relatively stable:

**Mouse, subclass level (EMMs):**

| Cutoff | Seurat | scVI |
|--------|--------|------|
| 0.00   | 0.772  | 0.793 |
| 0.10   | 0.765  | 0.740 |
| 0.25   | 0.753  | 0.644 |
| 0.50   | 0.733  | 0.461 |
| 0.75   | 0.712  | 0.287 |

At cutoff = 0, scVI outperforms Seurat. By cutoff = 0.10, the methods are approximately equivalent. Beyond 0.25, Seurat substantially outperforms scVI. This crossover pattern is consistent across all taxonomy levels and both organisms, and reflects the fact that scVI's softmax-based confidence scores are more uniformly distributed and thus more aggressively filtered by higher thresholds.

## Reference subsample size

Reducing the reference from 500 to 100 cells per type has minimal impact on estimated macro F1 (< 1 percentage point at most levels). Reducing to 50 cells produces a more noticeable decline, particularly at coarser levels:

**Mouse EMMs by subsample_ref:**

| Subsample | Global | Family | Class | Subclass |
|-----------|--------|--------|-------|----------|
| 500       | 0.950  | 0.882  | 0.813 | 0.799    |
| 100       | 0.946  | 0.882  | 0.807 | 0.797    |
| 50        | 0.937  | 0.870  | 0.794 | 0.784    |

The 500-to-50 difference is ~1.3 percentage points at global and ~1.5 at subclass — statistically significant but modest in magnitude. This suggests that 100 cells per type is a practical minimum for reference construction with limited performance loss.

## Biological covariates

**Treatment state (mouse):** No significant effect at most taxonomy levels (FDR > 0.05 for class, global, subclass). At the family level, treatment was marginally associated with slightly higher F1 (EMM: treatment 0.884 vs. no treatment 0.879, FDR = 0.010). The small magnitude and inconsistency across levels suggest treatment status does not meaningfully impact annotation quality.

**Disease state (human):** Negligible effect. At the global level, control and disease samples yield nearly identical estimated F1 (0.985 vs. 0.984).

**Sex (mouse):** Studies with recorded sex ("male") showed lower F1 than those with unspecified sex ("None"), but this is confounded with study identity — the "None" category corresponds to specific studies rather than a biological variable. The effect should be interpreted as residual study-level variation not captured by the random intercept.

## Boundary inflation in F1 distributions

F1 score distributions show notable boundary pile-ups that justify the ordered beta regression approach:

**Mouse:**
- scVI produces exact 0 F1 scores at subclass and class levels (~1.6% of observations), reflecting cell types that are completely misannotated.
- Seurat never produces exact 0s but produces exact 1s at coarser levels (global: 18.4% of observations), reflecting cell types that are perfectly classified.
- These boundary observations would be excluded or mishandled by standard beta regression.

## Cell-type ranking analysis

Rank aggregation across mouse studies shows that optimal parameter configurations are cell-type specific. Key patterns:

- **Seurat dominates rankings** for most cell types — the majority of "best" configurations per cell type use Seurat, often with 10x or SMART-Seq references.
- **Common cell types** (Astrocyte, Glutamatergic, GABAergic) achieve high win fractions (0.50-0.75) with consistent best configurations across studies.
- **Rare cell types** (e.g., Macrophage, mean F1 ~0.26) show lower performance and less consensus on optimal parameters, with scVI occasionally preferred.
- **Reference atlas choice matters per cell type** — some types perform best with 10x references while others favor SMART-Seq or whole cortex, reflecting technology-specific biases in the reference.

## Pareto analysis: performance vs. compute cost

We summarize configurations using a Pareto framework that jointly optimizes mean F1 (higher is better) and total compute cost (lower is better). For each taxonomy level (`subclass`, `class`, `family`, `global`), we compute mean F1 per configuration and mark a configuration as Pareto-optimal if no other configuration has both higher F1 and lower compute time. Compute cost is method-level (from `comptime_summary.tsv`), so all configurations within a method share the same cost value.

This framing emphasizes the tradeoff between accuracy and runtime. Because cost is fixed within each method, Pareto-optimal points typically correspond to the top-performing configurations within each method for a given taxonomy level. The resulting plots provide a compact view of which configurations are most competitive at a given cost, and which improvements in F1 would require switching to a higher-cost method.

## Between-study heterogeneity

The random intercept standard deviation ranges from 0.47 (family) to 0.82 (global) on the link scale, indicating substantial study-level variation in annotation quality. This justifies the mixed-effects approach: ignoring study-level clustering would produce overconfident estimates of method and covariate effects.
