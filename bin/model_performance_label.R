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
  default="/space/grp/rschwartz/rschwartz/evaluation_summary.nf/work/6f/dea5391518a01f43e6c8891c9cadd9/f1_results.tsv")
parser$add_argument("--key", help = "Level of granularity", default="class")
parser$add_argument("--label", help = "Label subset", default="GABAergic")
args <- parser$parse_args()

key <- args$key
label <- args$label
# Reading the label_f1_results file
label_f1_results <- read.table(args$label_f1_results, sep="\t", header=TRUE, stringsAsFactors = TRUE)
# fill NA with none
label_f1_results[is.na(label_f1_results)] <- "None"
# Extract organism (assuming only one unique value in the 'organism' column)
organism <- unique(label_f1_results$organism)[1]

# Defining factor names
#factor_names <- c("reference", "method", "cutoff", "subsample_ref")
factor_names <- c("cutoff", "method", "reference", "method:cutoff", "method:reference", "subsample_ref")

formulas <- list(
  paste("f1_score ~", paste(c(factor_names), collapse = " + "))
)


label_f1_results$f1_score <-  pmax(pmin(label_f1_results$f1_score, 1 - 1e-6), 1e-6)
label_f1_results$subsample_ref <- label_f1_results$subsample_ref %>% factor(levels = c("500","100","50"))
label_f1_results$method <- factor(label_f1_results$method, levels=c("seurat","scvi"))

dir.create(key, showWarnings = FALSE, recursive = TRUE)
label_dir <- file.path(key, label)
dir.create(label_dir, showWarnings = FALSE, recursive = TRUE)
#plot_model_metrics(df_list, formulas)

  for (formula in formulas) {
    formula_str <- formula %>% gsub(" ", "_", .)
    formula_dir <- file.path(label_dir, formula_str)
    dir.create(formula_dir, showWarnings = FALSE,recursive=TRUE)
    # make dir
    df = label_f1_results 
    run_and_store_model(df, formula, key_dir = formula_dir, key = key, type="label", group_var="study")
  }