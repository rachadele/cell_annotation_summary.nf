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

create_panel_c <- function(factor_emmeans, organism, raw_data = NULL) {
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

  # Build raw strip points aligned to y_pos
  raw_points <- NULL
  if (!is.null(raw_data) && nrow(raw_data) > 0) {
    raw_rows <- list()
    for (fct in unique(plot_df$factor)) {
      if (!fct %in% colnames(raw_data)) next
      fct_ypos <- plot_df %>% filter(factor == fct) %>% select(level, y_pos)
      rd <- raw_data %>%
        { if (fct != "subsample_ref" && "subsample_ref" %in% colnames(.)) filter(., subsample_ref == "500") else . } %>%
        mutate(level = str_to_title(str_replace_all(as.character(.data[[fct]]), "_", " "))) %>%
        inner_join(fct_ypos, by = "level") %>%
        select(y_pos, macro_f1)
      if (nrow(rd) > 0) raw_rows[[length(raw_rows) + 1]] <- rd
    }
    if (length(raw_rows) > 0) raw_points <- bind_rows(raw_rows)
  }

  # Factor group label positions
  factor_labels <- plot_df %>%
    group_by(factor) %>%
    summarise(mid_pos = mean(y_pos), .groups = "drop") %>%
    mutate(display_name = FACTOR_DISPLAY_NAMES[factor],
           colour       = FACTOR_COLORS[factor])

  p <- ggplot(plot_df, aes(x = response, y = y_pos)) +
    geom_vline(xintercept = grand_mean, colour = "gray", linetype = "dashed",
               linewidth = 0.5, alpha = 0.6)

  if (!is.null(raw_points) && nrow(raw_points) > 0) {
    p <- p + geom_boxplot(data = raw_points, aes(x = macro_f1, y = y_pos, group = factor(y_pos)),
                          inherit.aes = FALSE, width = 0.35, alpha = 0.3,
                          outlier.size = 0.5, colour = "gray50", fill = "gray80",
                          linewidth = 0.4)
  }

  p <- p +
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

  # Add factor group labels in right margin (pinned to just past x=1)
  for (k in seq_len(nrow(factor_labels))) {
    p <- p + annotate(
      "text",
      x    = 1.01,
      y    = factor_labels$mid_pos[k],
      label = factor_labels$display_name[k],
      hjust = 0, vjust = 0.5,
      fontface = "bold", size = 3.5,
      colour   = factor_labels$colour[k]
    )
  }

  p <- p + coord_cartesian(xlim = c(0, 1), clip = "off")

  p
}

create_panel_e <- function(coef_data, primary_key) {
  # Panel E: Effect magnitude comparison (logit scale)
  # Shows study random intercept SD vs fixed-effect predictor magnitudes,
  # making visible that study of origin dominates unexplained variance.
  if (is.null(coef_data) || nrow(coef_data) == 0) return(NULL)

  df <- coef_data %>% filter(key == primary_key)

  study_row <- df %>%
    filter(effect == "ran_pars", group == "study", term == "sd__(Intercept)")
  if (nrow(study_row) == 0) return(NULL)
  study_sd <- study_row$estimate[1]

  # Fixed main effects only (no interactions, no intercept)
  fixed <- df %>%
    filter(effect == "fixed", term != "(Intercept)", !str_detect(term, ":")) %>%
    mutate(
      predictor = case_when(
        str_starts(term, "reference")                            ~ "Reference",
        str_starts(term, "method")                              ~ "Method",
        term == "cutoff"                                         ~ "Cutoff",
        str_starts(term, "subsample_ref")                       ~ "Subsampling",
        str_starts(term, "sex")                                 ~ "Sex",
        str_starts(term, "disease_state") | str_starts(term, "treatment") ~ "Disease/Treatment",
        str_starts(term, "region_match")                        ~ "Region match",
        TRUE ~ NA_character_
      )
    ) %>%
    filter(!is.na(predictor)) %>%
    group_by(predictor) %>%
    summarise(
      # For multi-level predictors use SD of coefs; for single-level use |coef|
      magnitude = if (n() > 1) sd(estimate, na.rm = TRUE) else abs(estimate[1]),
      .groups = "drop"
    )

  plot_df <- bind_rows(
    tibble(predictor = "Study\n(random)", magnitude = study_sd, type = "random"),
    fixed %>% mutate(type = "fixed")
  ) %>%
    arrange(desc(magnitude)) %>%
    mutate(predictor = factor(predictor, levels = rev(predictor)))

  PANEL_E_COLORS <- c(random = "#e74c3c", fixed = "#888888")

  ggplot(plot_df, aes(x = magnitude, y = predictor, colour = type)) +
    geom_segment(aes(x = 0, xend = magnitude, yend = predictor), linewidth = 1.5) +
    geom_point(size = 4) +
    scale_colour_manual(values = PANEL_E_COLORS) +
    labs(x = "Effect magnitude (logit scale)", y = "",
         caption = "Multi-level predictors: SD of coefficients\nSingle-level: |coefficient|") +
    pub_theme() +
    theme(plot.caption = element_text(size = 7, colour = "gray50"),
          axis.text.y = element_text(size = 9))
}

create_panel_d <- function(reference_emmeans, raw_data = NULL) {
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

  raw_d <- NULL

  # Truncate long reference labels
  ref_labels <- levels(reference_emmeans$reference)
  ref_labels_trunc <- ifelse(nchar(ref_labels) > 60,
                              paste0(substr(ref_labels, 1, 57), "..."),
                              ref_labels)

  p_d <- ggplot(reference_emmeans, aes(x = response, y = y_dodge, colour = method)) +
    geom_vline(xintercept = grand_mean, colour = "gray", linetype = "dashed",
               linewidth = 0.5, alpha = 0.6)

  if (!is.null(raw_data) && nrow(raw_data) > 0) {
    raw_d <- raw_data %>%
      mutate(reference = as.character(reference), method = as.character(method)) %>%
      { if ("subsample_ref" %in% colnames(.)) filter(., subsample_ref == "500") else . } %>%
      inner_join(
        reference_emmeans %>% distinct(reference = as.character(reference),
                                       method = as.character(method), y_dodge),
        by = c("reference", "method")
      )
  }
  if (!is.null(raw_d) && nrow(raw_d) > 0) {
    p_d <- p_d + geom_boxplot(data = raw_d, aes(x = macro_f1, y = y_dodge, group = factor(y_dodge)),
                               inherit.aes = FALSE, width = 0.2, alpha = 0.3,
                               outlier.size = 0.5, colour = "gray50", fill = "gray80",
                               linewidth = 0.4)
  }

  p_d <- p_d +
    geom_segment(aes(x = `asymp.LCL`, xend = `asymp.UCL`, yend = y_dodge),
                 linewidth = 0.8) +
    geom_point(size = 2) +
    scale_colour_manual(values = METHOD_COLORS, labels = METHOD_NAMES) +
    scale_x_continuous(limits = c(0, 1)) +
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
  parser$add_argument("--model_coefs", type = "character", default = NULL,
                      help = "Path to model_coefs.tsv for variance source panel")
  parser$add_argument("--primary_key", type = "character", default = "subclass",
                      help = "Primary taxonomy key for filtering")
  parser$add_argument("--organism", type = "character", default = "homo_sapiens",
                      help = "Organism (homo_sapiens or mus_musculus)")
  parser$add_argument("--sample_results", type = "character", default = NULL,
                      help = "Path to sample_results.tsv.gz for raw data overlay")
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

  raw_sample_data <- if (!is.null(args$sample_results) && file.exists(args$sample_results)) {
    df <- read_tsv(args$sample_results, show_col_types = FALSE,
                   col_types = cols(.default = col_character()))
    if ("key"          %in% colnames(df)) df <- df %>% filter(key          == primary_key)
    if ("cutoff" %in% colnames(df)) df <- df %>% filter(as.numeric(cutoff) == 0)
    df <- df %>% mutate(macro_f1 = as.numeric(macro_f1))
    message("  Sample results (raw, cutoff=0, subsample_ref=500): ", nrow(df), " rows")
    df
  } else {
    NULL
  }

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

  coef_data <- if (!is.null(args$model_coefs) && file.exists(args$model_coefs)) {
    df <- read_tsv(args$model_coefs, show_col_types = FALSE)
    message("  Model coefs: ", nrow(df), " rows")
    df
  } else {
    message("  Model coefs: not available (skipping panel E)")
    NULL
  }

  # -- Build panels ------------------------------------------------------------
  message("\nCreating panels...")

  p_a <- create_panel_a(cutoff_data)
  p_b <- if (nrow(taxonomy_emmeans) > 0 && "key" %in% colnames(taxonomy_emmeans) &&
             n_distinct(taxonomy_emmeans$key) > 1) {
    create_panel_b(taxonomy_emmeans)
  } else {
    create_panel_b(tibble())
  }
  p_c <- create_panel_c(factor_emmeans, args$organism, raw_data = raw_sample_data)
  p_d <- create_panel_d(reference_emmeans, raw_data = raw_sample_data)
  p_e <- create_panel_e(coef_data, primary_key)

  # -- Compose layout ----------------------------------------------------------
  message("Composing layout...")

  # Enable legend on panel A so patchwork can collect it
  p_a <- p_a + theme(legend.position = "top") +
    guides(colour = guide_legend(title = NULL,
                                  override.aes = list(size = 3)),
           fill   = "none")

  if (!is.null(p_e)) {
    top_row <- (p_a | p_b | p_c | p_e) + plot_layout(widths = c(1, 1, 1.2, 0.9))
    combined <- (guide_area() / top_row / p_d) +
      plot_layout(heights = c(0.06, 1, 1), guides = "collect") +
      plot_annotation(tag_levels = list(c("", "A", "B", "C", "E", "D"))) &
      theme(legend.position = "top",
            legend.text = element_text(size = 16))
  } else {
    combined <- (guide_area() /
      ((p_a | p_b | p_c) + plot_layout(widths = c(1, 1, 1.2))) /
      p_d) +
      plot_layout(heights = c(0.06, 1, 1), guides = "collect") +
      plot_annotation(tag_levels = list(c("", "A", "B", "C", "D"))) &
      theme(legend.position = "top",
            legend.text = element_text(size = 16))
  }

  # -- Save --------------------------------------------------------------------
  out_path <- file.path(args$outdir, paste0(args$output_prefix, "_combined.png"))
  message("Saving to ", out_path, " ...")
  ggsave(out_path, combined,
         width  = FULL_WIDTH * 5.0,
         height = STANDARD_HEIGHT * 2.8,
         dpi = 300, bg = "white")

  message("Done!")
}

main()
