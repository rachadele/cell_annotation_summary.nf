#!/usr/bin/env Rscript
library(glmmTMB)
library(dplyr)
library(readr)
library(tidyr)
library(broom.mixed)
library(emmeans)
library(ggplot2)
library(DHARMa)
library(argparse)
library(stringr)
source("/space/grp/rschwartz/rschwartz/evaluation_summary.nf/bin/model_functions.R")

parser <- ArgumentParser()
parser$add_argument("--sample_results",
                    default = "2024-07-01/mus_musculus/100/dataset_id/SCT/gap_false/aggregated_results/files/sample_results.tsv")
parser$add_argument("--outdir", default = "assay_model")
args <- parser$parse_args()

dir.create(args$outdir, showWarnings = FALSE, recursive = TRUE)

df <- read_tsv(args$sample_results, show_col_types = FALSE)

# Simplify suspension labels
df <- df %>%
  mutate(
    query_type = case_when(
      query_suspension_type == "cell" ~ "single-cell",
      query_suspension_type == "nucleus" ~ "single-nucleus",
      TRUE ~ query_suspension_type
    ),
    ref_type = case_when(
      ref_suspension_type == "cell" ~ "cell_only",
      ref_suspension_type == "cell, nucleus" ~ "cell_and_nucleus",
      TRUE ~ ref_suspension_type
    )
  )

df$query_type <- factor(df$query_type, levels = c("single-cell", "single-nucleus"))
df$ref_type <- factor(df$ref_type, levels = c("cell_only", "cell_and_nucleus"))

KEY_ORDER <- c("subclass", "class", "family", "global")

# Split by key
df_list <- split(df, df$key)

all_coefs <- list()
all_emmeans <- list()
all_contrasts <- list()

for (k in KEY_ORDER) {
  df_k <- df_list[[k]]
  cat(sprintf("\n=== Fitting model for key: %s (n=%d) ===\n", k, nrow(df_k)))

  fig_dir <- file.path(args$outdir, "figures", k)
  dir.create(fig_dir, showWarnings = FALSE, recursive = TRUE)

  model_formula <- "macro_f1 ~ ref_type * query_type + cutoff"

  model <- glmmTMB(as.formula(model_formula), data = df_k, family = ordbeta())

  # Coefficients
  coefs <- tidy(model) %>%
    mutate(FDR = p.adjust(p.value, method = "fdr"),
           key = k,
           formula = model_formula,
           LogLik = as.numeric(logLik(model)),
           AIC = AIC(model),
           BIC = BIC(model))
  all_coefs[[k]] <- coefs

  cat("Coefficients:\n")
  print(coefs %>% select(term, estimate, std.error, p.value, FDR))

  # QQ and dispersion plots
  plot_qq(model, fig_dir)

  # Emmeans: ref_type * query_type interaction at cutoff=0
  emm <- emmeans(model, specs = ~ ref_type * query_type, at = list(cutoff = 0), type = "response")
  emm_summary <- as.data.frame(summary(emm)) %>% mutate(key = k)
  emm_pairs <- as.data.frame(pairs(emm)) %>% mutate(key = k)
  all_emmeans[[k]] <- emm_summary
  all_contrasts[[k]] <- emm_pairs

  cat("\nEmmeans (at cutoff=0):\n")
  print(emm_summary)
  cat("\nPairwise contrasts:\n")
  print(emm_pairs)

  # Plot emmeans
  p <- ggplot(emm_summary, aes(x = query_type, y = response,
                                ymin = asymp.LCL, ymax = asymp.UCL,
                                color = ref_type)) +
    geom_point(size = 3, position = position_dodge(width = 0.3)) +
    geom_errorbar(width = 0.2, position = position_dodge(width = 0.3)) +
    labs(title = k, x = "Query assay", y = "Estimated Macro F1",
         color = "Reference type") +
    theme_bw(base_size = 14) +
    theme(legend.position = "bottom")

  ggsave(file.path(fig_dir, "emmeans_ref_x_query.png"), p,
         width = 6, height = 5, bg = "white")

  # Plot coefficient summary
  plot_model_summary(model_summary = coefs, outdir = fig_dir, key = k)
}

# Write combined results
files_dir <- file.path(args$outdir, "files")
dir.create(files_dir, showWarnings = FALSE, recursive = TRUE)

bind_rows(all_coefs) %>%
  write_tsv(file.path(files_dir, "assay_model_coefs.tsv"))

bind_rows(all_emmeans) %>%
  write_tsv(file.path(files_dir, "assay_emmeans_summary.tsv"))

bind_rows(all_contrasts) %>%
  write_tsv(file.path(files_dir, "assay_emmeans_contrasts.tsv"))
