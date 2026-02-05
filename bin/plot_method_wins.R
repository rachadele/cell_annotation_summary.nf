#!/usr/bin/env Rscript
# Win-fraction Bar Chart by Method
#
# From rankings_best.tsv, counts the number of cell types where scVI vs
# Seurat is the best method, stratified by taxonomy level. Overlays the
# mean F1 of the winning configuration as text annotations.
#
# Usage:
#   Rscript plot_method_wins.R \
#     --rankings_best rankings_best.tsv \
#     --outdir method_wins

library(argparse)
library(dplyr)
library(tidyr)
library(readr)
library(ggplot2)

# -- Constants ----------------------------------------------------------------

METHOD_COLORS <- c(scVI = "#1f77b4", Seurat = "#ff7f0e")
METHOD_NAMES  <- c(scvi = "scVI", seurat = "Seurat")
KEY_ORDER     <- c("subclass", "class", "family", "global")

# -- Main ---------------------------------------------------------------------

main <- function() {
  parser <- ArgumentParser(
    description = "Bar chart of win counts by method and taxonomy level"
  )
  parser$add_argument("--rankings_best", required = TRUE,
                      help = "Path to rankings_best.tsv")
  parser$add_argument("--outdir", default = "method_wins",
                      help = "Output directory")
  args <- parser$parse_args()

  dir.create(args$outdir, showWarnings = FALSE, recursive = TRUE)

  cat("Loading rankings_best data...\n")
  df <- read_tsv(args$rankings_best, show_col_types = FALSE)
  cat(sprintf("Loaded %d rows\n", nrow(df)))

  df <- df %>%
    mutate(
      method_display = METHOD_NAMES[method],
      key = factor(key, levels = KEY_ORDER)
    )

  # Count wins and compute mean F1 per method per key
  win_counts <- df %>%
    group_by(key, method_display) %>%
    summarise(
      n_wins = n(),
      mean_f1_of_wins = mean(mean_f1_across_studies, na.rm = TRUE),
      .groups = "drop"
    )

  # Total cell types per key (for proportion context)
  totals <- df %>%
    group_by(key) %>%
    summarise(total = n(), .groups = "drop")

  win_counts <- win_counts %>%
    left_join(totals, by = "key") %>%
    mutate(proportion = n_wins / total)

  cat("\n--- Win counts ---\n")
  print(as.data.frame(win_counts))

  # -- Plot 1: Grouped bar chart of win counts --
  p1 <- ggplot(win_counts,
               aes(x = key, y = n_wins, fill = method_display)) +
    geom_col(position = position_dodge(width = 0.7), width = 0.6,
             alpha = 0.9) +
    geom_text(
      aes(label = sprintf("%d (F1=%.2f)", n_wins, mean_f1_of_wins)),
      position = position_dodge(width = 0.7),
      vjust = -0.5, size = 3
    ) +
    scale_fill_manual(values = METHOD_COLORS, name = "Method") +
    scale_y_continuous(expand = expansion(mult = c(0, 0.15))) +
    labs(
      x = "Taxonomy Level",
      y = "Number of Cell Types Won",
      title = "Best Configuration Wins by Method",
      subtitle = "Labels show count and mean F1 of winning configurations"
    ) +
    theme_bw(base_size = 11) +
    theme(
      plot.title    = element_text(face = "bold", hjust = 0),
      plot.subtitle = element_text(size = 9, color = "grey40"),
      plot.background  = element_rect(fill = "white", color = NA),
      panel.background = element_rect(fill = "white", color = NA),
      panel.grid.major.x = element_blank(),
      legend.position  = "bottom"
    )

  outpath_png <- file.path(args$outdir, "method_wins.png")
  ggsave(outpath_png, p1, width = 7.1, height = 5.0, dpi = 300,
         bg = "white")
  cat(sprintf("Saved %s\n", outpath_png))

  # -- Plot 2: Stacked proportion bar chart --
  p2 <- ggplot(win_counts,
               aes(x = key, y = proportion, fill = method_display)) +
    geom_col(width = 0.6, alpha = 0.9) +
    geom_text(
      aes(label = sprintf("%d", n_wins)),
      position = position_stack(vjust = 0.5),
      size = 3.5, color = "white", fontface = "bold"
    ) +
    geom_hline(yintercept = 0.5, linetype = "dashed", color = "grey40",
               linewidth = 0.4) +
    scale_fill_manual(values = METHOD_COLORS, name = "Method") +
    scale_y_continuous(labels = scales::percent_format(),
                       expand = expansion(mult = c(0, 0.02))) +
    labs(
      x = "Taxonomy Level",
      y = "Proportion of Cell Types",
      title = "Method Win Share by Taxonomy Level",
      subtitle = "Dashed line = 50%"
    ) +
    theme_bw(base_size = 11) +
    theme(
      plot.title    = element_text(face = "bold", hjust = 0),
      plot.subtitle = element_text(size = 9, color = "grey40"),
      plot.background  = element_rect(fill = "white", color = NA),
      panel.background = element_rect(fill = "white", color = NA),
      panel.grid.major.x = element_blank(),
      legend.position  = "bottom"
    )

  outpath_png2 <- file.path(args$outdir, "method_wins_proportion.png")
  ggsave(outpath_png2, p2, width = 7.1, height = 5.0, dpi = 300,
         bg = "white")
  cat(sprintf("Saved %s\n", outpath_png2))

  cat(sprintf("\nDone. Figures in %s/\n", args$outdir))
}

main()
