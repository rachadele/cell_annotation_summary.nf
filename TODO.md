# TODO

## 1. Fix reference coverage tables

**Affects:** both organisms

The `get_ref_support.py` script generates incorrect coverage counts for some cell types. Root cause is unknown. Known issues:

- **homo_sapiens** (`assets/ref_coverage/no-ma-et-al-homo-sapiens/`): PVALB shows 0 cells at subclass for SEA-AD DLPFC and SEA-AD MTG, despite being present in the actual reference data (confirmed via h5ad files). This is biologically impossible.
- **mus_musculus** (`assets/ref_coverage/tabulamuris-mus-musculus/`): Oligodendrocyte shows 0 cells for Motor cortex, Cortical+Hipp. 10x, and Cortical+Hipp. SSv4 at family level — also biologically implausible.

Leading hypothesis: `disease == 'normal'` filter in `get_filtered_obs()` (in `nextflow_eval_pipeline/bin/utils.py`) may be excluding cells from disease-labeled donors (e.g., Alzheimer's in SEA-AD), but root cause is unconfirmed.

**Action:** Debug `get_ref_support.py`, regenerate tables, then revisit reference selection for both organisms.

---

## 2. Revisit configuration recommendations after coverage table fix

**Affects:** `2024-07-01/homo_sapiens_new/summary.md`, `2024-07-01/mus_musculus/summary.md`

Both summaries currently recommend **whole cortex** as a conservative default pending correct coverage tables. Once tables are fixed:

- **homo_sapiens_new**: SEA-AD DLPFC may be preferable (previously Pareto-optimal at subclass: mean F1 = 0.859, 0.040 hrs, 0.020 GB). Revisit reference selection and check PVALB coverage.
- **mus_musculus**: Motor cortex EMM is marginally higher than whole cortex (0.908 vs 0.904 at family) but shows Oligodendrocyte = 0 — may become preferred once corrected.

---

## 3. Add missing human studies to homo_sapiens_new results

**Affects:** `2024-07-01/homo_sapiens_new/`

The current results only include 9 studies. The following studies are present in `get_gemma_data.nf/all_homo_sapiens_samples/metadata_standardized/` but missing from the results:

| Study | Notes |
|---|---|
| GSE144136 | — |
| GSE157827 | — |
| GSE174332 | — |
| GSE180928.1 | — |
| GSE211870 | — |
| Mathys-2023 | — |
| Velmeshev-2019.1 | — |
| Velmeshev-2019.2 | — |

These were added on the `scvi_knn` branch — use that branch's query set as the source. After adding, regenerate the summary.

---

## 4. Fix disease label propagation in homo_sapiens_new

**Affects:** `2024-07-01/homo_sapiens_new/`

The `disease`, `sex`, and `dev_stage` columns are empty in the per-sample scores TSVs produced by `nextflow_eval_pipeline`. These fields come from the run-level `params.yaml`, which only contains pipeline-wide parameters — no per-sample metadata. As a result, `aggregate_results.py` line 149 maps all null disease values to `"control"`, silently misclassifying disease samples.

**Known misclassified studies:**

| Study | Condition | Misclassified samples |
|---|---|---|
| CMC | Schizophrenia | 47 |
| SZBDMulti-Seq | Schizophrenia (24), Bipolar Disorder (24) | 48 |
| UCLA-ASD | ASD | 27 |
| PTSDBrainomics | PTSD (6), MDD (4) | 10 |

Correct metadata is available at:
`get_gemma_data.nf/all_homo_sapiens_samples/metadata_standardized/<study>/<study>_sample_meta_std.tsv`

**Actions:**
- In the upstream pipeline: join per-sample metadata at the scores-writing step so `disease`, `sex`, `dev_stage` are populated from the standardized metadata TSVs.
- In `aggregate_results.py` line 149: do not map null disease → `"control"`; leave as null so missing metadata is distinguishable from confirmed controls.
- After fix: rerun `evaluation_summary.nf` and re-evaluate the `disease_state` covariate effect.
