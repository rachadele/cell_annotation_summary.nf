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


Paul thoughts:
- plot raw data
- stop label modeling
- variation sample to sample?
- e.g. 100 samples
- score for each sample for f1 for astrocytes
- variation across samples?
- build dataset-by-dataset models instead of cell-type-by-cell-type models
- do cell type analyses per dataset
- do sample-level analyses per dataset?
- plot inter-sample variation in an indiviudal study
- maybe this will explain why seurat is better for individual cell types but scvi is better overall?
- maybe my model fits are just bad for label models
- make a heatmap for each dataset: rows are samples, columns are cell types, values are f1 scores
- look for patterns in the heatmaps
- order by sex, treatment, disease status, etc

                                                                                                                                                                                              
																																												
❯ ok so what if i want to fit a model to each cell type within each study to find the best reference/method combination. i was having trouble with model fits, the beta-logit glm seemed ill conditioned which is why i abandoned it. the distriubtion of f1 scores for each cell type looks really different so i'm not sure what model to pick.                                      
                                                                                                                                                                                                
