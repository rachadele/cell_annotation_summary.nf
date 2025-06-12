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
parser$add_argument("--label_f1_results", help = "Path to the label_f1_results file", 
  default="/space/grp/rschwartz/rschwartz/evaluation_summary.nf/work/8b/c346da8db7b29ed2f3f3564a0700d2/label_f1_results.tsv")
args <- parser$parse_args()


# Reading the label_f1_results file
label_f1_results <- read.table(args$label_f1_results, sep="\t", header=TRUE, stringsAsFactors = TRUE)
# fill NA with none
label_f1_results[is.na(label_f1_results)] <- "None"
# Extract organism (assuming only one unique value in the 'organism' column)
organism <- unique(label_f1_results$organism)[1]

# Defining factor names
#factor_names <- c("reference", "method", "cutoff", "subsample_ref")
factor_names <- c("label", "support", "cutoff", "method")

formulas <- list(
  paste("f1_score ~", paste(c(factor_names, "method:cutoff", "method:support"), collapse = " + "))
)


label_f1_results$f1_score <-  pmax(pmin(label_f1_results$f1_score, 1 - 1e-6), 1e-6)
label_f1_results$subsample_ref <- label_f1_results$subsample_ref %>% factor(levels = c("500","100","50"))

# Grouping the data by 'key' column and creating a list of data frames
df_list <- split(label_f1_results, label_f1_results$key)

#plot_model_metrics(df_list, formulas)

for (df in df_list) {
  for (formula in formulas) {
    formula_dir <- formula %>% gsub(" ", "_", .)

    dir.create(formula_dir, showWarnings = FALSE,recursive=TRUE)
    df$method <- factor(df$method, levels=c("seurat","scvi"))
    
  # need to set baseline cell types
    df$label <- factor(df$label)
    print(levels(df$label)[1])

    key = df$key[1]
    key_dir = file.path(formula_dir, key)
    # make dir
    dir.create(key_dir, showWarnings = FALSE,recursive=TRUE)

    run_and_store_model(df, formula, key_dir = key_dir, key = key, type="label")
  }
}
