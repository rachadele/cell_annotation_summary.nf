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
parser$add_argument("--emmeans_cutoff", type = "double", default = 0,
  help = "Cutoff value at which to evaluate emmeans (default: 0)")
args <- parser$parse_args()


# Reading the aggregated F1 results file
aggregated_f1_results <- read.table(args$aggregated_f1_results, sep = "\t",
                                    header = TRUE, stringsAsFactors = FALSE)
# Fill NA only for character columns (avoid coercing numeric responses)
char_cols <- vapply(aggregated_f1_results, is.character, logical(1))
aggregated_f1_results[char_cols] <- lapply(
  aggregated_f1_results[char_cols],
  function(x) {
    x[is.na(x)] <- "None"
    x
  }
)

# Ensure response is numeric and remove rows with NA
aggregated_f1_results$macro_f1 <- as.numeric(aggregated_f1_results$macro_f1)
aggregated_f1_results <- aggregated_f1_results[!is.na(aggregated_f1_results$macro_f1), ]
aggregated_f1_results <- droplevels(aggregated_f1_results)
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

METHOD_COLORS <- c(seurat = "#ff7f0e", scvi_rf = "#1f77b4", scvi_knn = "#2ca02c")
METHOD_NAMES  <- c(seurat = "Seurat",  scvi_rf = "scVI RF",  scvi_knn = "scVI kNN")

aggregated_f1_results$subsample_ref <- aggregated_f1_results$subsample_ref %>% factor(levels = c("500","100","50"))

plot_method_boxplot <- function(df, key, outdir, cutoff_val) {
  df_plot <- df[df$cutoff == cutoff_val, ]
  df_plot$method_label <- METHOD_NAMES[df_plot$method]
  df_plot$method_label <- factor(df_plot$method_label, levels = METHOD_NAMES)

  p <- ggplot(df_plot, aes(x = method_label, y = macro_f1, fill = method)) +
    geom_boxplot(outlier.shape = NA, alpha = 0.7) +
    geom_jitter(width = 0.15, alpha = 0.3, size = 1.5) +
    scale_fill_manual(values = METHOD_COLORS, guide = "none") +
    labs(x = NULL, y = "Macro F1", title = paste0(key, " — method (cutoff=", cutoff_val, ")")) +
    theme(axis.text.x = element_text(angle = 30, hjust = 1),
          plot.title = element_text(size = 18),
          strip.text = element_text(size = 18))

  ggsave(file.path(outdir, paste0(key, "_method_boxplot.png")), p, width = 12, height = 9, dpi = 200)
}

plot_reference_method_boxplot <- function(df, key, outdir, cutoff_val) {
  df_plot <- df[df$cutoff == cutoff_val, ]
  df_plot$method_label <- METHOD_NAMES[df_plot$method]
  df_plot$method_label <- factor(df_plot$method_label, levels = METHOD_NAMES)

  p <- ggplot(df_plot, aes(x = method_label, y = macro_f1, fill = method)) +
    geom_boxplot(outlier.shape = NA, alpha = 0.7) +
    geom_jitter(width = 0.15, alpha = 0.3, size = 1.5) +
    scale_fill_manual(values = METHOD_COLORS, guide = "none") +
    facet_wrap(~ reference, labeller = label_wrap_gen(width = 25)) +
    labs(x = NULL, y = "Macro F1", title = paste0(key, " — reference × method (cutoff=", cutoff_val, ")")) +
    theme(axis.text.x = element_text(angle = 30, hjust = 1),
          plot.title = element_text(size = 18),
          strip.text = element_text(size = 18))

  ggsave(file.path(outdir, paste0(key, "_reference_method_boxplot.png")), p, width = 18, height = 12, dpi = 200)
}

# Grouping the data by 'key' column and creating a list of data frames
df_list <- split(aggregated_f1_results, aggregated_f1_results$key)

# Initialize list to collect all results across keys
all_collected_results <- list()

for (df in df_list) {
  for (formula in formulas) {
    formula_dir <- formula %>% gsub(" ", "_", .)
    dir.create(formula_dir, showWarnings = FALSE, recursive = TRUE)

    df$method <- factor(df$method, levels = c("seurat", "scvi_rf", "scvi_knn"))
    key <- df$key[1]

    # Create figures directory for this key
    fig_dir <- file.path(formula_dir, "figures", key)
    dir.create(fig_dir, showWarnings = FALSE, recursive = TRUE)

    # Raw value box plots for method and reference×method contrasts
    plot_method_boxplot(df, key, fig_dir, args$emmeans_cutoff)
    plot_reference_method_boxplot(df, key, fig_dir, args$emmeans_cutoff)

    # Run model and get results
    key_results <- run_and_store_model(df, formula, fig_dir = fig_dir, key = key, type = "weighted", cutoff_ref = args$emmeans_cutoff)

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
