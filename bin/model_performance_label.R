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
parser$add_argument("--label", help = "Label subset", default="GABAergic")
args <- parser$parse_args()

label <- args$label
# Reading the label_f1_results file
label_f1_results <- read.table(args$label_f1_results, sep="\t", header=TRUE, stringsAsFactors = TRUE)
# fill NA with none
label_f1_results[is.na(label_f1_results)] <- "None"
# Extract organism (assuming only one unique value in the 'organism' column)
organism <- unique(label_f1_results$organism)[1]

# Defining factor names (key = granularity level is now a factor, not a split variable)
factor_names <- c("key", "cutoff", "method", "reference", "method:cutoff", "method:reference", "subsample_ref")

formulas <- list(
  paste("f1_score ~", paste(c(factor_names), collapse = " + "))
)

label_f1_results$f1_score <-  pmax(pmin(label_f1_results$f1_score, 1 - 1e-6), 1e-6)
label_f1_results$subsample_ref <- label_f1_results$subsample_ref %>% factor(levels = c("500","100","50"))
label_f1_results$method <- factor(label_f1_results$method, levels=c("seurat","scvi"))

label_dir <- label
dir.create(label_dir, showWarnings = FALSE, recursive = TRUE)

for (formula in formulas) {
  formula_str <- formula %>% gsub(" ", "_", .)
  formula_dir <- file.path(label_dir, formula_str)
  dir.create(formula_dir, showWarnings = FALSE, recursive=TRUE)
  file.dir <- file.path(formula_dir, "files")
  dir.create(file.dir, showWarnings = FALSE, recursive = TRUE)
  df = label_f1_results
  tryCatch({
    run_and_store_model(df, formula, key_dir = formula_dir, key = "all", type="label", mixed=FALSE)
  }, error = function(e) {
    message(paste0("Model failed for label ", label, ": ", e$message))
    # Write an error summary so the process still produces output
    error_df <- data.frame(
      term = "ERROR", estimate = NA, std.error = NA, statistic = NA,
      p.value = NA, FDR = NA, formula = formula, key = "all",
      LogLik = NA, AIC = NA, BIC = NA
    )
    write.table(error_df, file = file.path(file.dir, "all_model_summary_coefs_combined.tsv"),
                sep = "\t", row.names = FALSE)
    # Write a placeholder effects file so Nextflow output glob matches
    write.table(data.frame(note = "model_failed"), file = file.path(file.dir, "method_cutoff_effects.tsv"),
                sep = "\t", row.names = FALSE)
  })
}