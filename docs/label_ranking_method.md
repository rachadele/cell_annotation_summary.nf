# Cell Type Label Ranking Method

## Goal

For each cell type at each taxonomy level (subclass, class, family, global), identify the best-performing combination of pipeline parameters — specifically **(reference, method, subsample_ref)** — by aggregating evidence across studies.

## Why not parametric regression?

We initially attempted beta regression (logit-link GLM) on per-label F1 scores. This approach was abandoned because:

1. **Boundary values.** F1 scores of exactly 0 and 1 are common, especially for rare cell types. The beta distribution has support on (0, 1) and does not accommodate boundary values without zero/one inflation, which adds model complexity and instability.
2. **Heterogeneous distributions.** The shape of the F1 distribution varies dramatically across cell types. Abundant, well-separated types cluster near 1; rare or ambiguous types are often bimodal (many 0s plus a spread of non-zero values). No single parametric family fits all cases.
3. **Small within-group sample sizes.** Within a single (study, cell type) group, there may be only a handful of query samples, making maximum-likelihood estimation unreliable and convergence difficult.
4. **Study-level confounding.** Studies differ in difficulty due to tissue composition, sequencing depth, and experimental design. Mixed-effects models with study random intercepts partially address this but compound the convergence issues above.

## Approach: within-study ranking with cross-study aggregation

We use a non-parametric, rank-based approach that avoids distributional assumptions entirely.

### Step 1 — Within-study mean F1

For each **(study, cell type, key, reference, method, subsample_ref)** group, compute the mean F1 score across all query samples in that study. This averaging step marginalizes over sample-level covariates (sex, treatment, genotype, etc.) that vary within a study but are not parameters we are optimizing over.

### Step 2 — Rank within study

For each **(study, cell type, key)** group, rank the parameter combinations by their mean F1 from Step 1. Rank 1 is assigned to the combination with the highest mean F1. Ties receive the average of the tied ranks (fractional ranking).

A normalized rank is also computed:

```
norm_rank = (rank - 1) / (n_combos - 1)
```

This maps ranks to [0, 1] (0 = best, 1 = worst), making ranks comparable across studies that may have different numbers of available parameter combinations.

### Step 3 — Aggregate across studies

For each **(cell type, key, reference, method, subsample_ref)** combination, aggregate the per-study ranks:

| Metric | Description |
|--------|-------------|
| `mean_rank` | Mean raw rank across studies (lower is better) |
| `median_rank` | Median raw rank across studies |
| `mean_norm_rank` | Mean normalized rank across studies (0–1 scale) |
| `n_studies` | Number of studies contributing evidence for this cell type |
| `n_wins` | Number of studies where this combination achieved rank 1 |
| `win_fraction` | `n_wins / n_studies` |
| `mean_f1_across_studies` | Mean of the within-study mean F1 values |
| `std_f1_across_studies` | Standard deviation of the within-study mean F1 values |
| `mean_support` | Mean cell type proportion (fraction of cells in query samples) |
| `total_samples` | Total number of query samples across all studies |

### Step 4 — Select best combination

For each **(cell type, key)** pair, the combination with the lowest `mean_rank` is selected as the recommended parameter set.

## Outputs

| File | Contents |
|------|----------|
| `rankings_per_study.tsv` | Per-study ranks for every (study, cell type, key, combo). Useful for inspecting individual study behavior. |
| `rankings_detailed.tsv` | All parameter combinations with cross-study aggregated metrics. |
| `rankings_best.tsv` | One row per (cell type, key): the best-performing parameter combination. |

## Interpreting the results

- **`n_studies`**: Cell types appearing in only one study have no cross-study replication. Treat their rankings as provisional.
- **`win_fraction`**: A combination that wins in 80% of studies is a stronger recommendation than one that wins 50% but has a slightly lower mean rank.
- **`mean_support`**: Provides context on cell type abundance. Rankings for very rare cell types (low support) may be noisier.
- **`mean_f1_across_studies`**: Gives the absolute performance level. A combination can be "best" with mean rank 1 but still have low F1 if the cell type is inherently difficult to classify.

## Assumptions and limitations

- **Equal study weighting.** Each study contributes equally regardless of the number of query samples. This could be refined by weighting studies by sample count or inverse variance.
- **No support weighting.** Rankings do not account for cell type abundance. A future extension could weight by support so that recommendations are optimized for the most abundant types.
- **Missing combinations.** Not all parameter combinations may be present in every study. Rankings are computed over available combinations only, and the normalized rank accounts for differing numbers of combinations.
- **No formal significance testing.** The current implementation reports descriptive statistics (mean rank, win fraction) without p-values. A Friedman test could be added for cell types with sufficient multi-study evidence.

## Fixed parameters

- **cutoff = 0**: Confidence-based cell filtering is not applied, as prior analysis showed minimal benefit.
