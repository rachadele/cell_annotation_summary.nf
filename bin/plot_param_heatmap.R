#!/usr/bin/env Rscript
# Combined parameter heatmaps with annotation color bars.
#
# - Rows = parameter combinations (reference x method x subsample_ref)
# - Columns = cell types
# - Faceted by taxonomy level (key)
# - Left-side color bars encode method, reference, and subsample_ref
#
# Usage:
#   Rscript plot_param_heatmap.R \
#     --input rankings_detailed.tsv \
#     --outdir param_heatmaps

suppressPackageStartupMessages({
  library(argparse)
  library(dplyr)
  library(tidyr)
  library(readr)
  library(stringr)
  library(ggplot2)
  library(patchwork)
  library(scales)
})

KEY_ORDER <- c("subclass", "class", "family", "global")
METHOD_COLORS <- c(scvi = "#1f77b4", seurat = "#ff7f0e")
METHOD_NAMES <- c(scvi = "scVI", seurat = "Seurat")

REF_PALETTE <- c(
  "#66c2a5", "#fc8d62", "#8da0cb", "#e78ac3",
  "#a6d854", "#ffd92f", "#e5c494", "#b3b3b3"
)
SUB_PALETTE <- c("#feedde", "#fdae6b", "#e6550d")

build_ref_short_names <- function(references) {
  refs_sorted <- sort(unique(references))
  short <- case_when(
    grepl("10x", refs_sorted) ~ "10x",
    grepl("SMART", refs_sorted) ~ "SMART-Seq",
    grepl("motor", refs_sorted, ignore.case = TRUE) ~ "Motor Ctx",
    grepl("whole", refs_sorted, ignore.case = TRUE) ~ "Whole Ctx",
    TRUE ~ substr(refs_sorted, 1, 15)
  )
  setNames(short, refs_sorted)
}

cluster_order_cols <- function(df_key) {
  mat <- df_key %>%
    select(row_id, label, mean_f1_across_studies) %>%
    pivot_wider(names_from = label, values_from = mean_f1_across_studies)

  if (nrow(mat) == 0) return(character())

  labels <- setdiff(colnames(mat), "row_id")
  if (length(labels) <= 1) return(labels)

  m <- as.matrix(mat[, labels, drop = FALSE])
  keep <- colSums(!is.na(m)) > 0
  labels <- labels[keep]
  m <- m[, keep, drop = FALSE]

  if (length(labels) <= 1) return(labels)

  for (j in seq_len(ncol(m))) {
    col_mean <- mean(m[, j], na.rm = TRUE)
    if (is.nan(col_mean)) col_mean <- 0
    m[is.na(m[, j]), j] <- col_mean
  }

  cor_mat <- suppressWarnings(cor(m, use = "pairwise.complete.obs"))
  cor_mat[is.na(cor_mat)] <- 0
  dist_mat <- as.dist(1 - cor_mat)
  hc <- hclust(dist_mat, method = "average")
  labels[hc$order]
}

main <- function() {
  parser <- argparse::ArgumentParser(
    description = "Combined parameter heatmaps with annotation color bars"
  )
  parser$add_argument("--input", default = "rankings_detailed.tsv",
                      help = "Path to rankings_detailed.tsv")
  parser$add_argument("--outdir", default = "param_heatmaps",
                      help = "Output directory")
  args <- parser$parse_args()

  dir.create(args$outdir, showWarnings = FALSE, recursive = TRUE)

  df <- read_tsv(args$input, show_col_types = FALSE)
  df$key <- tolower(df$key)

  keys <- intersect(KEY_ORDER, unique(df$key))
  if (length(keys) == 0) {
    message("No keys found. Exiting.")
    return()
  }

  # Build short reference names and palettes
  ref_short <- build_ref_short_names(df$reference)
  refs_sorted <- sort(unique(df$reference))
  ref_colors <- setNames(REF_PALETTE[seq_along(refs_sorted)], refs_sorted)

  subs_sorted <- sort(unique(as.character(df$subsample_ref)))
  sub_colors <- setNames(SUB_PALETTE[seq_along(subs_sorted)], subs_sorted)

  # Create a unique row ID per config, sorted by method then reference then subsample
  df <- df %>%
    mutate(
      method = factor(method, levels = names(METHOD_NAMES)),
      subsample_ref = factor(subsample_ref, levels = subs_sorted),
      row_id = paste0(method, ":", reference, ":", subsample_ref)
    )

  row_meta <- df %>%
    distinct(row_id, method, reference, subsample_ref) %>%
    arrange(method, reference, subsample_ref)
  row_levels <- row_meta$row_id

  df <- df %>%
    mutate(row_id = factor(row_id, levels = row_levels))

  # Cluster x-axis (cell type labels) per key
  label_levels <- c()
  for (k in keys) {
    key_df <- df %>% filter(key == k)
    labels_ordered <- cluster_order_cols(key_df)
    label_levels <- c(label_levels, paste(k, labels_ordered, sep = "___"))
  }

  df <- df %>%
    mutate(
      label_key = paste(key, label, sep = "___"),
      label_key = factor(label_key, levels = label_levels),
      key = factor(key, levels = KEY_ORDER)
    )

  # -- Annotation bar data --
  anno <- row_meta %>%
    mutate(
      row_id = factor(row_id, levels = row_levels),
      subsample_ref = as.character(subsample_ref)
    )

  # Shared y-axis theme for annotation strips
  strip_theme <- theme_void() +
    theme(
      plot.margin = margin(0, 0, 0, 0),
      axis.text.y = element_blank(),
      axis.ticks.y = element_blank()
    )

  # Method annotation bar
  p_method <- ggplot(anno, aes(x = 1, y = row_id, fill = method)) +
    geom_tile(color = "white", linewidth = 0.25) +
    scale_fill_manual(values = METHOD_COLORS,
                      labels = METHOD_NAMES,
                      name = "Method") +
    strip_theme +
    labs(x = NULL, y = NULL)

  # Reference annotation bar
  p_ref <- ggplot(anno, aes(x = 1, y = row_id, fill = reference)) +
    geom_tile(color = "white", linewidth = 0.25) +
    scale_fill_manual(values = ref_colors,
                      labels = ref_short,
                      name = "Reference") +
    strip_theme +
    labs(x = NULL, y = NULL)

  # Subsample annotation bar
  p_sub <- ggplot(anno, aes(x = 1, y = row_id, fill = subsample_ref)) +
    geom_tile(color = "white", linewidth = 0.25) +
    scale_fill_manual(values = sub_colors,
                      name = "Subsample\nRef") +
    strip_theme +
    labs(x = NULL, y = NULL)

  # -- Main heatmap --
  p_heat <- ggplot(df, aes(x = label_key, y = row_id,
                            fill = mean_f1_across_studies)) +
    geom_tile(color = "white", linewidth = 0.25) +
    scale_fill_viridis_c(limits = c(0, 1), name = "Mean F1") +
    scale_x_discrete(labels = function(x) sub(".*___", "", x)) +
    facet_grid(. ~ key, scales = "free_x", space = "free_x") +
    labs(x = NULL, y = NULL) +
    theme_minimal(base_size = 12) +
    theme(
      axis.text.x = element_text(angle = 90, vjust = 0.5, hjust = 1, size = 8),
      axis.text.y = element_blank(),
      axis.ticks.y = element_blank(),
      panel.grid = element_blank(),
      strip.text = element_text(face = "bold", size = 11),
      legend.position = "right"
    )

  # -- Assemble with patchwork --
  combined <- p_method + p_ref + p_sub + p_heat +
    plot_layout(widths = c(0.02, 0.02, 0.02, 1),
                guides = "collect") +
    plot_annotation(
      title = "Parameter Performance Heatmaps",
      theme = theme(
        plot.title = element_text(face = "bold", size = 16, hjust = 0)
      )
    ) &
    theme(legend.position = "right")

  outpath <- file.path(args$outdir, "param_heatmaps_combined.png")
  ggsave(outpath, combined, width = 16, height = 8, dpi = 200, bg = "white")
  message("Saved: ", outpath)
}

main()
