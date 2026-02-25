# Cell-Type Annotation Benchmarking: Results Summary

We benchmarked scVI and Seurat label transfer for cell-type annotation across GEO studies of mouse and human brain data, evaluating performance at four taxonomy levels (subclass → class → family → global). All numbers below are macro F1 scores averaged across studies (mean ± SD), using SCT normalization with no gap filtering.

---

## Overall Performance by Taxonomy Level

| Organism | Subclass | Class | Family | Global |
|---|---|---|---|---|
| Mouse | 0.621 ± 0.099 | 0.598 ± 0.119 | 0.747 ± 0.088 | 0.809 ± 0.163 |
| Human | 0.860 ± 0.115 | 0.897 ± 0.107 | 0.980 ± 0.064 | 0.985 ± 0.048 |

Performance improves substantially at coarser taxonomy levels for both organisms. Human datasets consistently outperform mouse, likely reflecting greater homogeneity across human cortical dissection protocols and more complete reference coverage.

---

## Poorly Performing Cell Types

### Mouse (F1 < 0.75 at finest resolved level)

| Cell Type | Level | Mean F1 | SD |
|---|---|---|---|
| Microglia | subclass | 0.010 | 0.054 |
| OPC | subclass | 0.024 | 0.104 |
| Macrophage | subclass | 0.231 | 0.133 |
| Neural stem cell | subclass | 0.337 | 0.169 |
| Cajal-Retzius cell | subclass | 0.485 | 0.425 |
| Pericyte | subclass | 0.500 | 0.500 |

### Human (F1 < 0.75 at finest resolved level)

| Cell Type | Level | Mean F1 | SD |
|---|---|---|---|
| Pericyte | subclass | 0.000 | — |
| L5 ET | subclass | 0.360 | 0.435 |
| SNCG | subclass | 0.515 | 0.438 |
| deep layer non-IT | class | 0.679 | 0.304 |
| L6b | subclass | 0.707 | 0.352 |
| L6 CT | subclass | 0.726 | 0.360 |
| L5/6 NP | subclass | 0.749 | 0.425 |
| LAMP5 | subclass | 0.750 | 0.341 |

---

## Biological Hypotheses for Poor Performance

**Microglia (mouse only — F1 = 0.010)**
Mouse microglia performs catastrophically despite human microglia performing well (F1 = 0.971). Brain-resident immune cells activate rapidly during tissue dissociation, dramatically shifting their transcriptome. Mouse microglia may be especially sensitive to this artifact, or they may be systematically absent/underrepresented in the cortical query datasets used here, leaving the classifier with near-zero signal.

**OPC (mouse — F1 = 0.024)**
The near-zero F1 with perfect precision (1.0) and near-zero recall (0.016) indicates that OPC predictions are almost never made — the model effectively abstains. Oligodendrocyte precursors share substantial transcriptional overlap with mature oligodendrocytes, and subclass-level distinction may require marker genes poorly preserved across protocols. Mouse OPCs may be at differentiation states not represented in the reference.

**Macrophage (mouse — F1 = 0.231)**
High recall (0.83) but very low precision (0.14) suggests cells are broadly assigned to macrophage but the reference boundary between brain-resident macrophages and circulating/infiltrating macrophages is blurred. In cortical datasets, this class may include contaminating blood-derived cells that the reference does not cleanly separate.

**Neural stem cell (mouse — F1 = 0.337)**
Adult neural stem cells are restricted to neurogenic niches (SVZ, dentate gyrus) and are effectively absent from most cortical query datasets. The model recalls ~78% of true NSCs when they occur, but precision is low (0.23), indicating frequent false positive calls. Rarity and transcriptional proximity to radial glia make confident detection difficult.

**Cajal-Retzius cell (mouse — F1 = 0.485)**
Transient neurons of the marginal zone that largely disappear postnatally. Their presence in adult datasets depends on the exact brain region sampled; the high SD (0.43) reflects near-binary detection across studies. Reference coverage of this rare population may be inconsistent.

**Pericyte (mouse F1 = 0.500; human F1 = 0.000)**
Small, low-abundance vascular mural cells frequently lost during single-cell dissociation. Only a handful of pericytes are captured per dataset, making F1 highly unstable (SD = 0.50 in mouse; complete failure in human with no true positives predicted). Small annotation inconsistencies between reference and query are sufficient to cause total F1 collapse at this scale.

**L5 ET (human — F1 = 0.360)**
Layer 5 extratelencephalic neurons project subcortically and spinally. High precision (0.89) but near-zero recall (0.36) suggests the classifier correctly identifies the rare L5 ET cells it predicts but misses most true L5 ETs, likely classifying them as L5 IT or other L5 subtypes. Underrepresentation of deep cortical layers in standard dissections further limits signal.

**SNCG (human — F1 = 0.515)**
CGE-derived interneuron subtype (synuclein-gamma expressing). The large SD (0.44) indicates binary behavior across studies: detectable when this rare interneuron is well sampled, invisible otherwise. Transcriptional overlap with LAMP5 and other small-diameter inhibitory neurons contributes to misclassification.

**deep layer non-IT (human — F1 = 0.679)**
A coarse class grouping neurons not projecting intracortically (L5 ET, L6 CT, L6b, L5/6 NP). Its poor performance at the class level directly reflects the difficulty of classifying its constituent subtypes, compounded by transcriptional similarity among deep-layer projection neurons.

**L6b, L6 CT, L5/6 NP (human — F1 = 0.707, 0.726, 0.749)**
Deep-layer and near-white-matter neurons among the least abundant in standard cortical dissections. Their transcriptional profiles partially overlap with adjacent layer subtypes (especially L6 IT), causing systematic misclassification. L6b represents an evolutionarily ancient cortical layer that may be inconsistently annotated across reference datasets. The high variance for L5/6 NP (SD = 0.42) suggests it is well detected in some studies but entirely missed in others.

**LAMP5 (human — F1 = 0.750)**
CGE-derived interneuron with partial transcriptional overlap with SNCG and other small-diameter inhibitory neurons. Moderate mean performance with high variance (SD = 0.34) suggests LAMP5 is routinely detected in well-powered datasets but merged with related subtypes in sparser studies.
