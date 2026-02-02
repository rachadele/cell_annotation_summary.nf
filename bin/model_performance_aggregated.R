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
parser$add_argument("--aggregated_f1_results", help = "Path to the aggregated F1 results file",
  default="/space/grp/rschwartz/rschwartz/evaluation_summary.nf/work/e8/0ee4b04d0441800e9a092e9885599a/aggregated_f1_results.tsv")
args <- parser$parse_args()


# Reading the aggregated F1 results file
aggregated_f1_results <- read.table(args$aggregated_f1_results, sep="\t", header=TRUE, stringsAsFactors = TRUE)
# fill NA with none
aggregated_f1_results[is.na(aggregated_f1_results)] <- "None"
# Extract organism (assuming only one unique value in the 'organism' column)
organism <- unique(aggregated_f1_results$organism)[1]

# Defining factor names
factor_names <- c("reference", "method", "cutoff", "subsample_ref")



if (organism == "homo_sapiens") {
  all_factors = c(factor_names, "disease_state","sex","region_match")
  # Defining the formulas
  formulas <- list( 
    paste("macro_f1 ~", paste(c(all_factors, "method:cutoff", "reference:method"), collapse = " + "))
    )
} else if (organism == "mus_musculus") {
    # full interactive model
  all_factors <- c(factor_names, "treatment_state","sex")
  formulas <- list(
    paste("macro_f1 ~", paste(c(all_factors, "method:cutoff", "reference:method"), collapse = " + "))

    
  )
}

aggregated_f1_results$macro_f1 <-  pmax(pmin(aggregated_f1_results$macro_f1, 1 - 1e-6), 1e-6)
aggregated_f1_results$subsample_ref <- aggregated_f1_results$subsample_ref %>% factor(levels = c("500","100","50"))

# Grouping the data by 'key' column and creating a list of data frames
df_list <- split(aggregated_f1_results, aggregated_f1_results$key)

# Initialize list to collect all results across keys
all_collected_results <- list()

for (df in df_list) {
  for (formula in formulas) {
    formula_dir <- formula %>% gsub(" ", "_", .)
    dir.create(formula_dir, showWarnings = FALSE, recursive = TRUE)

    df$method <- factor(df$method, levels = c("seurat", "scvi"))
    key <- df$key[1]

    # Create figures directory for this key
    fig_dir <- file.path(formula_dir, "figures", key)
    dir.create(fig_dir, showWarnings = FALSE, recursive = TRUE)

    # Run model and get results
    key_results <- run_and_store_model(df, formula, fig_dir = fig_dir, key = key, type = "weighted")

    # Collect results by name
    for (result_name in names(key_results)) {
      if (is.null(all_collected_results[[result_name]])) {
        all_collected_results[[result_name]] <- list()
      }
      all_collected_results[[result_name]][[key]] <- key_results[[result_name]]
    }
  }
}

# Combine and write results
files_dir <- file.path(formula_dir, "files")
dir.create(files_dir, showWarnings = FALSE, recursive = TRUE)

for (result_name in names(all_collected_results)) {
  combined_df <- bind_rows(all_collected_results[[result_name]])
  write.table(combined_df, file = file.path(files_dir, paste0(result_name, ".tsv")), sep = "\t", row.names = FALSE)
}
