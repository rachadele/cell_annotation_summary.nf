library(glmmTMB)
library(dplyr)
library(broom.mixed)
library(multcomp)
library(argparse)
library(dplyr)
library(readr)
library(dplyr)
library(tidyr)
library(ggplot2)
library(gridExtra)
library(stringr)
library(DHARMa)
library(effects)
library(emmeans)
source("/space/grp/rschwartz/rschwartz/evaluation_summary.nf/bin/model_functions.R")
#library(multcomp)
# Set global theme for background
theme_set(
  theme_minimal(base_size =30 ) +  # Base theme
    theme(
      plot.background = element_rect(fill = "white", color = NA),  # Plot background color
      panel.background = element_rect(fill = "white", color = NA), # Panel background color
      legend.background = element_rect(fill = "white", color = NA) # Legend background color
    )
)

parser <- argparse::ArgumentParser()
parser$add_argument("--weighted_f1_results", help = "Path to the weighted_f1_results file", 
  default="/space/grp/rschwartz/rschwartz/evaluation_summary.nf/2024-07-01/mmus_new_tabulamuris/100/dataset_id/SCT/gap_false/aggregated_results/weighted_f1_results.tsv")
args <- parser$parse_args()


# Reading the weighted_f1_results file
weighted_f1_results <- read.table(args$weighted_f1_results, sep="\t", header=TRUE, stringsAsFactors = TRUE)
# fill NA with none
weighted_f1_results[is.na(weighted_f1_results)] <- "None"
# Extract organism (assuming only one unique value in the 'organism' column)
organism <- unique(weighted_f1_results$organism)[1]

# Defining factor names
factor_names <- c("reference", "method", "cutoff", "subsample_ref")



if (organism == "homo_sapiens") {
  all_factors = c(factor_names, "disease_state","sex","region_match")
  # Defining the formulas
  formulas <- list(
    #paste("weighted_f1 ~", paste(c(factor_names, "method:cutoff"), collapse = " + ")),
    #paste("weighted_f1 ~", paste(c(factor_names, "method:cutoff","disease_state"),collapse = "+")),
    #paste("weighted_f1 ~", paste(c(factor_names, "method:cutoff","sex"),collapse = "+")),
    #paste("weighted_f1 ~", paste(c(factor_names, "method:cutoff","region_match"),collapse = "+")),
    #paste("weighted_f1 ~", paste(c(factor_names, "method:cutoff", "reference:method"), collapse = " + ")),
    paste("weighted_f1 ~", paste(c(all_factors, "method:cutoff", "reference:method"), collapse = " + "))
    )
} else if (organism == "mus_musculus") {
    # full interactive model
  all_factors <- c(factor_names, "treatment_state","sex")
  formulas <- list(
    #paste("weighted_f1 ~", paste(c(factor_names, "method:cutoff"), collapse = " + ")),
    #paste("weighted_f1 ~", paste(c(factor_names, "method:cutoff","treatment_state"),collapse = " + ")),
    #paste("weighted_f1 ~", paste(c(factor_names, "method:cutoff","sex"),collapse = " + ")),
    #paste("weighted_f1 ~", paste(c(factor_names, "method:cutoff", "reference:method"), collapse = " + ")),
    paste("weighted_f1 ~", paste(c(all_factors, "method:cutoff", "reference:method"), collapse = " + "))

    
  )
}

weighted_f1_results$weighted_f1 <-  pmax(pmin(weighted_f1_results$weighted_f1, 1 - 1e-6), 1e-6)
weighted_f1_results$subsample_ref <- weighted_f1_results$subsample_ref %>% factor(levels = c("500","100","50"))

# Grouping the data by 'key' column and creating a list of data frames
df_list <- split(weighted_f1_results, weighted_f1_results$key)

#plot_model_metrics(df_list, formulas)

for (df in df_list) {
  for (formula in formulas) {
    formula_dir <- formula %>% gsub(" ", "_", .)

    dir.create(formula_dir, showWarnings = FALSE,recursive=TRUE)
    df$method <- factor(df$method, levels=c("seurat","scvi"))
    
    key = df$key[1]
    key_dir = file.path(formula_dir, key)
    # make dir
    dir.create(key_dir, showWarnings = FALSE,recursive=TRUE)

    run_and_store_model(df, formula, key_dir = key_dir, key = key, type="weighted")
  }
}
