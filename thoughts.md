why do individual labels perform better with seurat, but fitted values from the random effect model favor scvi?
we are fitting to weighted average f1 scores, which give more weight to cell types with more support
>fit to macro average f1 instead?
>somehow incorporate number of total cells per cell type into the plot, like maybe with size of dots on the forest plot, so we can see which cell types get more weight?

still don't understand why performance is overall better in scvi thatn scvi for sample-level weighted f1, but when we look at fitted values for individual cell types seurat is clearly better

maybe we need to look at raw values instead of fitted values?


> fitting to macro average f1 didn't change scvi improvement over seurat for sample-level f1, so the question remains why seurat is better for individual cell types but scvi is better overall

changes made to make results more interpretable:

- write fitted emmeans results per contrast to a single tsv file instead of separate files per model, and add the key (level of granularity) as a column
- this will impact PLOT_PUB_FIGURES module, which will need to read from the combined tsv file