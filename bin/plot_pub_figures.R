#!/usr/bin/env Rscript
# Publication Figures for Cell Type Annotation Benchmarking
#
# Creates a multi-panel publication figure (A-D) using ggplot2 + patchwork:
#   A: Cutoff sensitivity curves (F1 vs cutoff by method)
#   B: Taxonomy level slope chart (subclass -> global)
#   C: Experimental factor contrasts (disease/sex/treatment/region)
#   D: Reference atlas comparison (emmeans by reference x method)
#
# Layout: top row = A | B | C, bottom row = D (full width)
#
# Usage:
#   Rscript plot_pub_figures.R \
#       --cutoff_effects path/to/method_cutoff_effects.tsv \
#       --reference_emmeans path/to/reference_method_emmeans_summary.tsv \
#       --method_emmeans path/to/method_emmeans_summary.tsv \
#       --factor_emmeans treatment_emmeans.tsv sex_emmeans.tsv ... \
#       --organism mus_musculus \
#       --outdir figures

library(argparse)
library(dplyr)
library(tidyr)
library(readr)
library(stringr)
library(ggplot2)
library(patchwork)

# -- Constants ----------------------------------------------------------------

METHOD_COLORS <- c(scvi = "#1f77b4", seurat = "#ff7f0e")
METHOD_NAMES  <- c(scvi = "scVI", seurat = "Seurat")

KEY_ORDER <- c("subclass", "class", "family", "global")

FULL_WIDTH      <- 7.1
STANDARD_HEIGHT <- 5.9

FACTOR_DISPLAY_NAMES <- c(
  disease_state  = "Disease State",
  disease        = "Disease State",
  sex            = "Sex",
  region_match   = "Region Match",
  treatment_state = "Treatment",
  treatment      = "Treatment",
  subsample_ref  = "Reference Size"
)

FACTOR_COLORS <- c(
  disease_state   = "#2ecc71",
  disease         = "#2ecc71",
  sex             = "#9b59b6",
  region_match    = "#e74c3c",
  treatment_state = "#3498db",
  treatment       = "#3498db",
  subsample_ref   = "#f39c12"
)

HUMAN_FACTORS <- c("disease_state", "disease", "sex", "region_match")
MOUSE_FACTORS <- c("treatment_state", "treatment", "sex")

# -- Theme --------------------------------------------------------------------

pub_theme <- function() {
  theme_bw(base_size = 20) +
    theme(
      panel.grid.major = element_blank(),
      panel.grid.minor = element_blank(),
      legend.position  = "none",
      plot.background  = element_rect(fill = "white", colour = NA),
      panel.background = element_rect(fill = "white", colour = NA)
    )
}

# -- Helpers ------------------------------------------------------------------

extract_factor_from_path <- function(filepath) {
  basename <- basename(filepath)
  m <- regmatches(basename, regexec("(.+)_emmeans_summary\\.tsv", basename))
  if (length(m[[1]]) >= 2) return(m[[1]][2])
  return(NA_character_)
}

# -- Panel functions ----------------------------------------------------------

create_panel_a <- function(cutoff_data) {
  # Panel A: Cutoff sensitivity — line + ribbon by method
  cutoff_data <- cutoff_data %>%
    mutate(method = factor(method, levels = names(METHOD_NAMES)))

  ggplot(cutoff_data, aes(x = cutoff, y = fit, colour = method, fill = method)) +
    geom_ribbon(aes(ymin = lower, ymax = upper), alpha = 0.2, colour = NA) +
    geom_line(linewidth = 0.8) +
    geom_point(size = 2) +
    scale_colour_manual(values = METHOD_COLORS, labels = METHOD_NAMES) +
    scale_fill_manual(values = METHOD_COLORS, labels = METHOD_NAMES) +
    scale_y_continuous(limits = c(0, 1)) +
    labs(x = "Confidence Cutoff", y = "Estimated F1 Score") +
    pub_theme()
}

create_panel_b <- function(taxonomy_emmeans) {
  # Panel B: Taxonomy slope chart — connected dots across key levels
  if (nrow(taxonomy_emmeans) == 0 || !"key" %in% colnames(taxonomy_emmeans)) {
    return(
      ggplot() +
        annotate("text", x = 0.5, y = 0.5, label = "Taxonomy data\nnot available",
                 colour = "gray", size = 4) +
        labs(x = "Taxonomy Level", y = "Estimated Marginal Mean F1") +
        pub_theme() +
        theme(axis.text = element_blank(), axis.ticks = element_blank())
    )
  }

  taxonomy_emmeans <- taxonomy_emmeans %>%
    mutate(
      method = factor(method, levels = names(METHOD_NAMES)),
      key    = factor(key, levels = KEY_ORDER)
    )

  ggplot(taxonomy_emmeans, aes(x = key, y = response,
                               colour = method, group = method)) +
    geom_line(linewidth = 0.8, alpha = 0.8) +
    geom_point(size = 3) +
    scale_colour_manual(values = METHOD_COLORS, labels = METHOD_NAMES) +
    scale_y_continuous(limits = c(0.5, 1)) +
    labs(x = "Taxonomy Level", y = "Estimated Marginal Mean F1") +
    pub_theme()
}

create_panel_c <- function(factor_emmeans, organism) {
  # Panel C: Experimental factor forest plot
  if (length(factor_emmeans) == 0) {
    return(
      ggplot() +
        annotate("text", x = 0.5, y = 0.5, label = "No factor data available",
                 colour = "gray", size = 4) +
        pub_theme() +
        theme(axis.text = element_blank(), axis.ticks = element_blank())
    )
  }

  factor_order <- if (organism == "homo_sapiens") HUMAN_FACTORS else MOUSE_FACTORS
  factor_order <- factor_order[factor_order %in% names(factor_emmeans)]

  if (length(factor_order) == 0) {
    return(
      ggplot() +
        annotate("text", x = 0.5, y = 0.5, label = "No factor data available",
                 colour = "gray", size = 4) +
        pub_theme() +
        theme(axis.text = element_blank(), axis.ticks = element_blank())
    )
  }

  # Build a combined data frame with y positions
  plot_rows <- list()
  separator_positions <- c()
  current_pos <- 0

  for (i in seq_along(factor_order)) {
    fct <- factor_order[i]
    df <- factor_emmeans[[fct]]
    standard_cols <- c("response", "SE", "df", "asymp.LCL", "asymp.UCL", "key")
    factor_cols <- setdiff(colnames(df), standard_cols)
    if (length(factor_cols) == 0) next
    primary_col <- factor_cols[1]
    df <- df %>% arrange(desc(response))

    for (j in seq_len(nrow(df))) {
      level_val <- as.character(df[[primary_col]][j])
      if (tolower(level_val) %in% c("none", "nan", "") || is.na(level_val)) next
      level_display <- str_to_title(str_replace_all(level_val, "_", " "))
      plot_rows[[length(plot_rows) + 1]] <- tibble(
        factor   = fct,
        level    = level_display,
        response = df$response[j],
        lower    = df$`asymp.LCL`[j],
        upper    = df$`asymp.UCL`[j],
        y_pos    = current_pos
      )
      current_pos <- current_pos + 1
    }

    if (i < length(factor_order)) {
      separator_positions <- c(separator_positions, current_pos - 0.5)
      current_pos <- current_pos + 0.5
    }
  }

  if (length(plot_rows) == 0) {
    return(
      ggplot() +
        annotate("text", x = 0.5, y = 0.5, label = "No factor data available",
                 colour = "gray", size = 4) +
        pub_theme() +
        theme(axis.text = element_blank(), axis.ticks = element_blank())
    )
  }

  plot_df <- bind_rows(plot_rows)
  grand_mean <- mean(plot_df$response)

  # Factor group label positions
  factor_labels <- plot_df %>%
    group_by(factor) %>%
    summarise(mid_pos = mean(y_pos), .groups = "drop") %>%
    mutate(display_name = FACTOR_DISPLAY_NAMES[factor],
           colour       = FACTOR_COLORS[factor])

  p <- ggplot(plot_df, aes(x = response, y = y_pos)) +
    geom_vline(xintercept = grand_mean, colour = "gray", linetype = "dashed",
               linewidth = 0.5, alpha = 0.6) +
    geom_segment(aes(x = lower, xend = upper, yend = y_pos, colour = factor),
                 linewidth = 1.2) +
    geom_point(aes(colour = factor), size = 3) +
    scale_colour_manual(values = FACTOR_COLORS) +
    scale_y_continuous(
      breaks = plot_df$y_pos,
      labels = plot_df$level,
      expand = expansion(mult = c(0.05, 0.05))
    ) +
    labs(x = "Estimated Marginal Mean F1", y = "") +
    pub_theme()

  # Add separator lines
  for (sp in separator_positions) {
    p <- p + geom_hline(yintercept = sp, colour = "lightgray",
                        linewidth = 0.5, alpha = 0.7)
  }

  # Add factor group labels in right margin
  for (k in seq_len(nrow(factor_labels))) {
    p <- p + annotate(
      "text",
      x    = max(plot_df$upper, na.rm = TRUE) + 0.01,
      y    = factor_labels$mid_pos[k],
      label = factor_labels$display_name[k],
      hjust = 0, vjust = 0.5,
      fontface = "bold", size = 3.5,
      colour   = factor_labels$colour[k]
    )
  }

  # Extend x-axis to make room for factor labels
  x_range <- range(c(plot_df$lower, plot_df$upper), na.rm = TRUE)
  x_pad <- diff(x_range) * 0.35
  p <- p + coord_cartesian(xlim = c(x_range[1] - diff(x_range) * 0.05,
                                     x_range[2] + x_pad),
                            clip = "off")

  p
}

create_panel_d <- function(reference_emmeans) {
  # Panel D: Reference atlas forest plot (dodged by method)
  methods <- sort(unique(reference_emmeans$method))
  n_methods <- length(methods)

  # Order references by mean response
  ref_order <- reference_emmeans %>%
    group_by(reference) %>%
    summarise(mean_resp = mean(response), .groups = "drop") %>%
    arrange(mean_resp) %>%
    pull(reference)

  reference_emmeans <- reference_emmeans %>%
    mutate(
      method    = factor(method, levels = names(METHOD_NAMES)),
      reference = factor(reference, levels = ref_order)
    )

  # Compute dodge offset manually for segment + point
  dodge_width <- 0.25
  reference_emmeans <- reference_emmeans %>%
    mutate(
      ref_num = as.numeric(reference),
      method_num = as.numeric(method),
      offset  = (method_num - (n_methods + 1) / 2) * dodge_width,
      y_dodge = ref_num + offset
    )

  grand_mean <- mean(reference_emmeans$response)

  # Truncate long reference labels
  ref_labels <- levels(reference_emmeans$reference)
  ref_labels_trunc <- ifelse(nchar(ref_labels) > 60,
                              paste0(substr(ref_labels, 1, 57), "..."),
                              ref_labels)

  ggplot(reference_emmeans, aes(x = response, y = y_dodge, colour = method)) +
    geom_vline(xintercept = grand_mean, colour = "gray", linetype = "dashed",
               linewidth = 0.5, alpha = 0.6) +
    geom_segment(aes(x = `asymp.LCL`, xend = `asymp.UCL`, yend = y_dodge),
                 linewidth = 0.8) +
    geom_point(size = 2) +
    scale_colour_manual(values = METHOD_COLORS, labels = METHOD_NAMES) +
    scale_y_continuous(
      breaks = seq_along(ref_labels),
      labels = ref_labels_trunc,
      expand = expansion(mult = c(0.02, 0.02))
    ) +
    labs(x = "Estimated Marginal Mean F1", y = "Reference Datasets") +
    pub_theme()
}

# -- Main ---------------------------------------------------------------------

parse_arguments <- function() {
  parser <- ArgumentParser(
    description = "Generate publication figures for benchmarking results"
  )
  parser$add_argument("--cutoff_effects", type = "character", default = NULL,
                      help = "Path to method_cutoff_effects.tsv")
  parser$add_argument("--reference_emmeans", type = "character", default = NULL,
                      help = "Path to reference_method_emmeans_summary.tsv")
  parser$add_argument("--method_emmeans", type = "character", default = NULL,
                      help = "Path to method_emmeans_summary.tsv")
  parser$add_argument("--factor_emmeans", type = "character", default = "",
                      help = "Space-separated paths to factor emmeans summary files")
  parser$add_argument("--primary_key", type = "character", default = "subclass",
                      help = "Primary taxonomy key for filtering")
  parser$add_argument("--organism", type = "character", default = "homo_sapiens",
                      help = "Organism (homo_sapiens or mus_musculus)")
  parser$add_argument("--outdir", type = "character", default = ".",
                      help = "Output directory")
  parser$add_argument("--output_prefix", type = "character", default = "pub_figure",
                      help = "Prefix for output filenames")
  parser$parse_args()
}

load_factor_emmeans <- function(filepaths, organism, primary_key) {
  target_factors <- if (organism == "homo_sapiens") HUMAN_FACTORS else MOUSE_FACTORS
  factor_data <- list()

  for (fp in filepaths) {
    if (!file.exists(fp)) {
      message("  Warning: File not found: ", fp)
      next
    }
    fct <- extract_factor_from_path(fp)
    if (is.na(fct) || !(fct %in% target_factors)) next

    df <- read_tsv(fp, show_col_types = FALSE)
    if ("key" %in% colnames(df)) {
      df <- df %>% filter(key == primary_key)
    }
    if (nrow(df) == 0) {
      message("  Warning: No data for ", fct, " with key=", primary_key)
      next
    }
    factor_data[[fct]] <- df
    message("  Loaded ", fct, ": ", nrow(df), " levels (key=", primary_key, ")")
  }
  factor_data
}

main <- function() {
  args <- parse_arguments()
  dir.create(args$outdir, recursive = TRUE, showWarnings = FALSE)
  primary_key <- args$primary_key

  # -- Load data ---------------------------------------------------------------
  message("Loading data (filtering by key=", primary_key, ")...")

  cutoff_data <- read_tsv(args$cutoff_effects, show_col_types = FALSE)
  if ("key" %in% colnames(cutoff_data)) {
    cutoff_data <- cutoff_data %>% filter(key == primary_key)
  }
  message("  Cutoff effects: ", nrow(cutoff_data), " rows")

  reference_emmeans <- read_tsv(args$reference_emmeans, show_col_types = FALSE)
  if ("key" %in% colnames(reference_emmeans)) {
    reference_emmeans <- reference_emmeans %>% filter(key == primary_key)
  }
  message("  Reference emmeans: ", nrow(reference_emmeans), " rows")

  taxonomy_emmeans <- if (!is.null(args$method_emmeans) && file.exists(args$method_emmeans)) {
    df <- read_tsv(args$method_emmeans, show_col_types = FALSE)
    message("  Loaded method emmeans: ", nrow(df), " rows, keys: ",
            paste(unique(df$key), collapse = ", "))
    df
  } else {
    tibble()
  }

  message("Loading experimental factor emmeans (key=", primary_key, ")...")
  factor_files <- if (nzchar(args$factor_emmeans)) {
    strsplit(trimws(args$factor_emmeans), "\\s+")[[1]]
  } else {
    character(0)
  }
  factor_emmeans <- load_factor_emmeans(factor_files, args$organism, primary_key)

  # -- Build panels ------------------------------------------------------------
  message("\nCreating panels...")

  p_a <- create_panel_a(cutoff_data)
  p_b <- if (nrow(taxonomy_emmeans) > 0 && "key" %in% colnames(taxonomy_emmeans) &&
             n_distinct(taxonomy_emmeans$key) > 1) {
    create_panel_b(taxonomy_emmeans)
  } else {
    create_panel_b(tibble())
  }
  p_c <- create_panel_c(factor_emmeans, args$organism)
  p_d <- create_panel_d(reference_emmeans)

  # -- Compose layout ----------------------------------------------------------
  message("Composing layout...")

  # Enable legend on panel A so patchwork can collect it
  p_a <- p_a + theme(legend.position = "top") +
    guides(colour = guide_legend(title = NULL,
                                  override.aes = list(size = 3)),
           fill   = "none")

  combined <- (guide_area() /
    ((p_a | p_b | p_c) + plot_layout(widths = c(1, 1, 1.2))) /
    p_d) +
    plot_layout(heights = c(0.06, 1, 1), guides = "collect") +
    plot_annotation(tag_levels = list(c("", "A", "B", "C", "D"))) &
    theme(legend.position = "top",
          legend.text = element_text(size = 16))

  # -- Save --------------------------------------------------------------------
  out_path <- file.path(args$outdir, paste0(args$output_prefix, "_combined.png"))
  message("Saving to ", out_path, " ...")
  ggsave(out_path, combined,
         width  = FULL_WIDTH * 3.7,
         height = STANDARD_HEIGHT * 2.4,
         dpi = 300, bg = "white")

  message("Done!")
}

main()
