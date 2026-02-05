#!/usr/bin/env Rscript
# F1 vs Compute Cost Scatter (Configuration Pareto Front)
#
# Each point is one of the 24 configurations.
# x = mean F1 (averaged across all cell types)
# y = total compute cost (hours, from comptime_summary.tsv)
# Colour = method, shape = subsample_ref.
# Since cost is method-level only, all scVI configs share one y-value
# and all Seurat configs share another. Pareto-optimal configs are labelled.
#
# Usage:
#   Rscript plot_config_pareto.R \
#     --rankings rankings_detailed.tsv \
#     --comptime comptime_summary.tsv \
#     --outdir config_pareto

library(argparse)
library(dplyr)
library(tidyr)
library(readr)
library(ggplot2)
library(ggrepel)

# -- Constants ----------------------------------------------------------------

METHOD_COLORS <- c(scVI = "#1f77b4", Seurat = "#ff7f0e")
METHOD_NAMES  <- c(scvi = "scVI", seurat = "Seurat")
KEY_ORDER     <- c("subclass", "class", "family", "global")

# -- Helpers ------------------------------------------------------------------

identify_pareto <- function(df, x_col, y_col, minimize_y = TRUE) {
  # Identify Pareto-optimal points (maximize x, minimize y)
  df <- df %>% arrange(desc(.data[[x_col]]))
  pareto <- logical(nrow(df))
  if (minimize_y) {
    best_y <- Inf
    for (i in seq_len(nrow(df))) {
      if (df[[y_col]][i] <= best_y) {
        pareto[i] <- TRUE
        best_y <- df[[y_col]][i]
      }
    }
  } else {
    best_y <- -Inf
    for (i in seq_len(nrow(df))) {
      if (df[[y_col]][i] >= best_y) {
        pareto[i] <- TRUE
        best_y <- df[[y_col]][i]
      }
    }
  }
  pareto
}

# -- Main ---------------------------------------------------------------------

main <- function() {
  parser <- ArgumentParser(
    description = "F1 vs compute cost scatter with Pareto front"
  )
  parser$add_argument(
    "--rankings",
    default = "2024-07-01/mus_musculus/100/dataset_id/SCT/gap_false/celltype_rankings/rankings/rankings_detailed.tsv",
    help = "Path to rankings_detailed.tsv"
  )
  parser$add_argument(
    "--comptime",
    default = "2024-07-01/mus_musculus/100/dataset_id/SCT/gap_false/comptime_plots/comptime_summary.tsv",
    help = "Path to comptime_summary.tsv"
  )
  parser$add_argument("--outdir", default = "config_pareto",
                      help = "Output directory")
  args <- parser$parse_args()

  dir.create(args$outdir, showWarnings = FALSE, recursive = TRUE)

  cat("Loading data...\n")
  rankings <- read_tsv(args$rankings, show_col_types = FALSE)
  comptime <- read_tsv(args$comptime, show_col_types = FALSE)

  # Compute total cost per method (and subsample_ref if available)
  if ("subsample_ref" %in% names(comptime)) {
    cost <- comptime %>%
      mutate(subsample_ref = as.character(subsample_ref)) %>%
      group_by(method, subsample_ref) %>%
      summarise(total_duration_hrs = sum(mean_duration, na.rm = TRUE),
                total_memory_gb   = max(mean_memory, na.rm = TRUE),
                .groups = "drop")
  } else {
    cost <- comptime %>%
      group_by(method) %>%
      summarise(total_duration_hrs = sum(mean_duration, na.rm = TRUE),
                total_memory_gb   = max(mean_memory, na.rm = TRUE),
                .groups = "drop")
  }

  cat("Compute costs:\n")
  print(as.data.frame(cost))

  # Add display method name
  rankings <- rankings %>%
    mutate(method_display = METHOD_NAMES[method])

  keys <- intersect(KEY_ORDER, unique(rankings$key))

  # Compute mean F1 per config and key (unweighted across cell types)
  config_f1 <- rankings %>%
    filter(key %in% keys) %>%
    group_by(key, method, method_display, reference, subsample_ref) %>%
    summarise(
      mean_f1 = mean(mean_f1_across_studies, na.rm = TRUE),
      n_celltypes = n(),
      .groups = "drop"
    )

  # Join with cost (match on method display name and subsample_ref if present)
  if ("subsample_ref" %in% names(cost)) {
    config_f1 <- config_f1 %>%
      mutate(subsample_ref = as.character(subsample_ref)) %>%
      left_join(cost, by = c("method_display" = "method",
                             "subsample_ref" = "subsample_ref"))
  } else {
    config_f1 <- config_f1 %>%
      left_join(cost, by = c("method_display" = "method"))
  }

  # Build short reference labels
  config_f1 <- config_f1 %>%
    mutate(
      ref_short = case_when(
        grepl("10x", reference) ~ "10x",
        grepl("SMART", reference) ~ "SMART-Seq",
        grepl("motor", reference, ignore.case = TRUE) ~ "Motor Ctx",
        grepl("whole", reference, ignore.case = TRUE) ~ "Whole Ctx",
        TRUE ~ substr(reference, 1, 12)
      ),
      config_label = paste0(method_display, "/", ref_short, "/",
                            subsample_ref),
      subsample_ref = factor(subsample_ref),
      key = factor(key, levels = KEY_ORDER)
    )

  # Identify Pareto front within each taxonomy level
  config_f1 <- config_f1 %>%
    group_by(key) %>%
    mutate(
      pareto = identify_pareto(cur_data(), "mean_f1",
                               "total_duration_hrs",
                               minimize_y = TRUE)
    ) %>%
    ungroup()

  n_keys <- length(keys)

  # Pareto front line data: sort by mean_f1 ascending for geom_step
  pareto_df <- config_f1 %>%
    filter(pareto) %>%
    arrange(key, mean_f1)

  p <- ggplot(config_f1, aes(x = mean_f1, y = total_duration_hrs)) +
    geom_step(data = pareto_df,
              direction = "vh",
              linetype = "dashed", color = "grey30", linewidth = 0.5) +
    geom_point(aes(color = method_display, shape = subsample_ref),
               size = 3.6, alpha = 0.85) +
    geom_text_repel(
      data = filter(config_f1, pareto),
      aes(label = config_label),
      size = 3.6, max.overlaps = 20,
      segment.color = "grey60", segment.size = 0.3,
      box.padding = 0.4, point.padding = 0.3
    ) +
    facet_wrap(~ key, ncol = 1, scales = "fixed",
               labeller = labeller(key = tools::toTitleCase)) +
    scale_x_continuous(breaks = scales::pretty_breaks(n = 7)) +
    scale_color_manual(values = METHOD_COLORS, name = "Method") +
    scale_shape_discrete(name = "Subsample Ref") +
    labs(
      x = "Mean F1 (across cell types)",
      y = "Total Compute Cost (hours)",
      title = "Performance vs Cost by Taxonomy Level",
      subtitle = "Pareto-optimal configurations labelled"
    ) +
    theme_bw(base_size = 14) +
    theme(
      plot.title    = element_text(face = "bold", hjust = 0),
      plot.subtitle = element_text(size = 11, color = "grey40"),
      plot.background  = element_rect(fill = "white", color = NA),
      panel.background = element_rect(fill = "white", color = NA),
      legend.position  = "bottom",
      legend.box = "horizontal"
    )

  outpath_png <- file.path(args$outdir, "config_pareto_all_keys.png")

  ggsave(outpath_png, p, width = 10.6, height = max(6.8, 3.0 * n_keys),
         dpi = 300,
         bg = "white")
  cat(sprintf("  Saved %s\n", outpath_png))

  cat(sprintf("\nDone. Figures in %s/\n", args$outdir))
}

main()
