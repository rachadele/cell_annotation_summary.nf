#!/usr/bin/env Rscript
# Reliability vs Performance Scatter Plot
#
# Scatter plot of win_fraction vs mean_f1_across_studies from rankings_best.tsv.
# Cell types in the upper-right are both high-performing and consistently best
# across studies. Faceted by taxonomy key level. Labels placed with ggrepel to
# avoid overlap.
#
# Usage:
#   Rscript plot_ranking_reliability.R \
#     --input rankings_best.tsv \
#     --outdir ranking_reliability

library(argparse)
library(dplyr)
library(readr)
library(ggplot2)
library(ggrepel)

# -- Constants ----------------------------------------------------------------

KEY_ORDER <- c("subclass", "class", "family", "global")

KEY_COLORS <- c(
  subclass = "#e41a1c",
  class    = "#377eb8",
  family   = "#4daf4a",
  global   = "#984ea3"
)

# -- Main ---------------------------------------------------------------------

main <- function() {
  parser <- ArgumentParser(
    description = "Win fraction vs F1 scatter plot (reliability vs performance)"
  )
  parser$add_argument("--input", default = "rankings_best.tsv",
                      help = "Path to rankings_best.tsv")
  parser$add_argument("--outdir", default = "ranking_reliability",
                      help = "Output directory")
  args <- parser$parse_args()

  dir.create(args$outdir, showWarnings = FALSE, recursive = TRUE)

  df <- read_tsv(args$input, show_col_types = FALSE)
  cat(sprintf("Loaded %d rows from %s\n", nrow(df), args$input))

  keys <- intersect(KEY_ORDER, unique(df$key))
  if (length(keys) == 0) {
    cat("No matching key levels found. Exiting.\n")
    quit(status = 0)
  }

  df <- df %>%
    filter(key %in% keys) %>%
    mutate(key = factor(key, levels = keys))

  # Normalise n_studies to [2, 8] point-size range
  ns_min <- min(df$n_studies, na.rm = TRUE)
  ns_max <- max(df$n_studies, na.rm = TRUE)
  if (ns_max > ns_min) {
    df <- df %>%
      mutate(pt_size = 2 + (n_studies - ns_min) / (ns_max - ns_min) * 6)
  } else {
    df <- df %>% mutate(pt_size = 5)
  }

  n_cols <- min(length(keys), 4)
  n_rows <- ceiling(length(keys) / n_cols)

  p <- ggplot(df, aes(x = mean_f1_across_studies, y = win_fraction)) +
    geom_hline(yintercept = 0.5, linetype = "dashed",
               color = "grey60", linewidth = 0.4) +
    geom_vline(xintercept = 0.5, linetype = "dashed",
               color = "grey60", linewidth = 0.4) +
    geom_point(aes(size = n_studies, color = key), alpha = 0.75) +
    geom_text_repel(
      aes(label = label, color = key),
      size          = 3.0,
      max.overlaps  = Inf,
      box.padding   = 0.4,
      point.padding = 0.3,
      segment.size  = 0.3,
      segment.color = "grey50",
      show.legend   = FALSE
    ) +
    facet_wrap(~ key, ncol = n_cols,
               labeller = labeller(key = tools::toTitleCase)) +
    scale_color_manual(values = KEY_COLORS, guide = "none") +
    scale_size_continuous(
      name   = "N Studies",
      range  = c(2, 8),
      breaks = scales::pretty_breaks(n = 3)
    ) +
    scale_x_continuous(limits = c(-0.05, 1.05),
                       breaks = scales::pretty_breaks(n = 5)) +
    scale_y_continuous(limits = c(-0.05, 1.15),
                       breaks = scales::pretty_breaks(n = 5)) +
    labs(
      x     = "Mean F1 Across Studies",
      y     = "Win Fraction",
      title = "Reliability vs Performance"
    ) +
    theme_bw(base_size = 13) +
    theme(
      plot.title       = element_text(face = "bold", hjust = 0),
      plot.background  = element_rect(fill = "white", color = NA),
      panel.background = element_rect(fill = "white", color = NA),
      legend.position  = "bottom"
    )

  outpath <- file.path(args$outdir, "ranking_reliability.png")
  ggsave(outpath, p,
         width  = 5 * n_cols,
         height = 5.5 * n_rows + 0.5,
         dpi    = 200,
         bg     = "white")
  cat(sprintf("Saved %s\n", outpath))
  cat(sprintf("\nDone. Figures in %s/\n", args$outdir))
}

main()
