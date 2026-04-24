#!/usr/bin/env Rscript
# Combined Human + Mouse Publication Figure
#
# Creates a 4-panel combined figure (A-D) for both organisms using:
#   colour = organism (Human / Mouse)
#   linetype = method (scVI / Seurat)
#
#   A: Cutoff sensitivity curves (F1 vs cutoff)
#   B: Taxonomy level slope chart (subclass -> global)
#   C: Experimental factor contrasts (disease/treatment/sex)
#   D: Reference atlas comparison
#
# Usage:
#   Rscript bin/plot_combined_figures.R \
#       --hs_cutoff_effects   path/to/hs/method_cutoff_effects.tsv \
#       --mm_cutoff_effects   path/to/mm/method_cutoff_effects.tsv \
#       --hs_reference_emmeans path/to/hs/reference_method_emmeans_summary.tsv \
#       --mm_reference_emmeans path/to/mm/reference_method_emmeans_summary.tsv \
#       --hs_method_emmeans   path/to/hs/method_emmeans_summary.tsv \
#       --mm_method_emmeans   path/to/mm/method_emmeans_summary.tsv \
#       --hs_factor_emmeans   "path/sex.tsv path/disease_state.tsv" \
#       --mm_factor_emmeans   "path/sex.tsv path/treatment.tsv" \
#       --primary_key subclass \
#       --outdir figures \
#       --output_prefix combined_figure

library(argparse)
library(dplyr)
library(tidyr)
library(readr)
library(stringr)
library(ggplot2)
library(patchwork)

# -- Constants ----------------------------------------------------------------

ORGANISM_COLORS  <- c(Human = "#2166ac", Mouse = "#d6604d")
METHOD_LINETYPES <- c(scvi_rf = "solid", scvi_knn = "dotted", seurat = "dashed", scvi = "solid")
METHOD_NAMES     <- c(scvi_rf = "scVI RF", scvi_knn = "scVI kNN", seurat = "Seurat", scvi = "scVI")

KEY_ORDER <- c("subclass", "class", "family", "global")

FULL_WIDTH      <- 7.1
STANDARD_HEIGHT <- 5.9

FACTOR_DISPLAY_NAMES <- c(
  disease_state   = "Disease State",
  treatment_state = "Treatment",
  treatment       = "Treatment",
  sex             = "Sex"
)

# Factor section order for Panel C (top to bottom)
FACTOR_ORDER <- c("disease_state", "treatment_state", "sex")

# Reference abbreviations for Panel D
REFERENCE_ABBREVS <- c(
  # Human — Allen Brain Cell Atlas dissection regions
  "Dissection Angular gyrus AnG"                                   = "AnG",
  "Dissection Anterior cingulate cortex ACC"                       = "ACC",
  "Dissection Dorsolateral prefrontal cortex DFC"                  = "DFC",
  "Dissection Primary auditory cortexA1"                           = "A1",
  "Dissection Primary somatosensory cortex S1"                     = "S1",
  "Dissection Primary visual cortexV1"                             = "V1",
  "Human Multiple Cortical Areas SMART-seq"                        = "MCA (SMART-seq)",
  "Whole Taxonomy - DLPFC Seattle Alzheimers Disease Atlas SEA-AD" = "SEA-AD DLPFC",
  "Whole Taxonomy - MTG Seattle Alzheimers Disease Atlas SEA-AD"   = "SEA-AD MTG",
  # Contrast strings use "/" instead of "-" for SEA-AD
  "Whole Taxonomy / DLPFC Seattle Alzheimers Disease Atlas SEA-AD" = "SEA-AD DLPFC",
  "Whole Taxonomy / MTG Seattle Alzheimers Disease Atlas SEA-AD"   = "SEA-AD MTG",
  # Mouse
  "An integrated transcriptomic and epigenomic atlas of mouse primary motor cortex cell types" =
    "Motor Cortex Atlas",
  "Single-cell RNA-seq for all cortical  hippocampal regions 10x"          = "Ctx & Hippo (10x)",
  "Single-cell RNA-seq for all cortical  hippocampal regions SMART-Seq v4" = "Ctx & Hippo (SS v4)",
  # Shared
  "whole cortex"                                                   = "Aggregated"
)

# -- Theme --------------------------------------------------------------------

pub_theme <- function() {
  theme_bw(base_size = 36) +
    theme(
      panel.grid.major = element_blank(),
      panel.grid.minor = element_blank(),
      legend.position  = "none",
      axis.text        = element_text(size = 34),
      axis.title       = element_text(size = 40),
      strip.text       = element_text(size = 34),
      plot.background  = element_rect(fill = "white", colour = NA),
      panel.background = element_rect(fill = "white", colour = NA)
    )
}

make_legend_plot <- function() {
  # Single plot with both aesthetics; panel data clipped out of view so only
  # the legend renders. override.aes prevents colour/linetype cross-contamination.
  dummy <- expand.grid(
    x        = c(0, 1),
    organism = factor(names(ORGANISM_COLORS), levels = names(ORGANISM_COLORS)),
    method   = factor(names(METHOD_NAMES),    levels = names(METHOD_NAMES))
  )
  dummy$y <- as.numeric(interaction(dummy$organism, dummy$method))

  ggplot(dummy, aes(x = x, y = y,
                    colour   = organism,
                    linetype = method,
                    group    = interaction(organism, method))) +
    geom_line(linewidth = 1.5) +
    scale_colour_manual(
      values = ORGANISM_COLORS, name = "Organism",
      guide  = guide_legend(override.aes = list(linetype = "solid"))
    ) +
    scale_linetype_manual(
      values = METHOD_LINETYPES, labels = unname(METHOD_NAMES), name = "Method",
      guide  = guide_legend(override.aes = list(colour = "black"))
    ) +
    coord_cartesian(xlim = c(10, 11)) +   # push dummy lines off-screen
    labs(caption = "Points + CI: fitted values (estimated marginal mean)\nBoxplots: raw data") +
    theme_void() +
    theme(
      legend.position  = c(0.6, 0.6),
      legend.direction = "vertical",
      legend.text      = element_text(size = 36),
      legend.title     = element_text(size = 40, face = "bold"),
      legend.key.width = unit(3, "cm"),
      plot.caption     = element_text(size = 28, hjust = 0, colour = "gray30",
                                      margin = margin(t = 10))
    )
}

# -- Helpers ------------------------------------------------------------------

extract_factor_from_path <- function(filepath) {
  base <- basename(filepath)
  m <- regmatches(base, regexec("(.+)_emmeans_summary\\.tsv", base))
  if (length(m[[1]]) >= 2) return(m[[1]][2])
  return(NA_character_)
}

filter_key <- function(df, pk) {
  if ("key" %in% colnames(df)) df <- df %>% filter(key == pk)
  df
}

sig_stars <- function(p) {
  dplyr::case_when(
    p < 0.001 ~ "***",
    p < 0.01  ~ "**",
    p < 0.05  ~ "*",
    TRUE      ~ "ns"
  )
}

# -- Panel A: Cutoff Sensitivity ----------------------------------------------

create_panel_a <- function(cutoff_data) {
  cutoff_data <- cutoff_data %>%
    mutate(
      method   = factor(method, levels = names(METHOD_NAMES)),
      organism = factor(organism, levels = names(ORGANISM_COLORS))
    )

  ggplot(cutoff_data, aes(x = cutoff, y = fit,
                           colour = organism, linetype = method)) +
    geom_line(linewidth = 0.8) +
    geom_point(size = 1.5) +
    scale_colour_manual(values = ORGANISM_COLORS, name = "Organism") +
    scale_linetype_manual(values = METHOD_LINETYPES, labels = METHOD_NAMES,
                          name = "Method") +
    scale_y_continuous(limits = c(0, 1)) +
    labs(x = "Confidence Cutoff", y = "Estimated F1 Score") +
    pub_theme()
}

# -- Panel B: Taxonomy Slope Chart --------------------------------------------

create_panel_b <- function(taxonomy_data) {
  if (nrow(taxonomy_data) == 0 || !"key" %in% colnames(taxonomy_data)) {
    return(
      ggplot() +
        annotate("text", x = 0.5, y = 0.5, label = "Taxonomy data\nnot available",
                 colour = "gray", size = 4) +
        labs(x = "Taxonomy Level", y = "Estimated Marginal Mean F1") +
        pub_theme() +
        theme(axis.text = element_blank(), axis.ticks = element_blank())
    )
  }

  taxonomy_data <- taxonomy_data %>%
    mutate(
      method   = factor(method, levels = names(METHOD_NAMES)),
      organism = factor(organism, levels = names(ORGANISM_COLORS)),
      key      = factor(key, levels = KEY_ORDER)
    )

  ggplot(taxonomy_data,
         aes(x = key, y = response,
             colour = organism, linetype = method,
             group = interaction(organism, method))) +
    geom_line(linewidth = 0.8, alpha = 0.8) +
    geom_point(size = 3) +
    scale_colour_manual(values = ORGANISM_COLORS, name = "Organism") +
    scale_linetype_manual(values = METHOD_LINETYPES, labels = METHOD_NAMES,
                          name = "Method") +
    scale_y_continuous(limits = c(0.5, 1)) +
    labs(x = "Taxonomy Level", y = "Estimated Marginal Mean F1") +
    pub_theme()
}

# -- Panel C: Combined Factor Effects Forest Plot -----------------------------

create_panel_c <- function(factor_data, sig_data = NULL, raw_data = NULL) {
  if (nrow(factor_data) == 0) {
    return(
      ggplot() +
        annotate("text", x = 0.5, y = 0.5, label = "No factor data available",
                 colour = "gray", size = 4) +
        pub_theme() +
        theme(axis.text = element_blank(), axis.ticks = element_blank())
    )
  }

  factor_data <- factor_data %>%
    mutate(organism = factor(organism, levels = names(ORGANISM_COLORS)))

  sections <- list(
    list(factor = "disease_state",   organisms = "Human"),
    list(factor = "treatment",       organisms = "Mouse"),
    list(factor = "sex",             organisms = c("Human", "Mouse"))
  )

  plot_rows           <- list()
  separator_positions <- c()
  current_pos         <- 0
  n_sections_added    <- 0

  for (sec in sections) {
    fct  <- sec$factor
    orgs <- sec$organisms

    sec_df <- factor_data %>%
      filter(factor_name == fct, organism %in% orgs) %>%
      mutate(organism = factor(organism, levels = orgs))

    sec_df <- sec_df %>% arrange(organism, desc(response))

    if (nrow(sec_df) == 0) next
    n_sections_added <- n_sections_added + 1

    if (n_sections_added > 1) {
      separator_positions <- c(separator_positions, current_pos - 0.5)
      current_pos <- current_pos + 0.5
    }

    for (r in seq_len(nrow(sec_df))) {
      plot_rows[[length(plot_rows) + 1]] <- tibble(
        factor_name = fct,
        organism    = as.character(sec_df$organism[r]),
        label       = sec_df$level[r],   # no organism suffix
        response    = sec_df$response[r],
        lower       = sec_df$`asymp.LCL`[r],
        upper       = sec_df$`asymp.UCL`[r],
        y_pos       = current_pos
      )
      current_pos <- current_pos + 1
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

  plot_df    <- bind_rows(plot_rows)
  grand_mean <- mean(plot_df$response, na.rm = TRUE)

  # Build raw strip points aligned to y_pos
  raw_points_c <- NULL
  if (!is.null(raw_data) && nrow(raw_data) > 0) {
    raw_rows_c <- list()
    for (i in seq_len(nrow(plot_df))) {
      org <- plot_df$organism[i]
      fct <- plot_df$factor_name[i]
      lbl <- plot_df$label[i]
      yp  <- plot_df$y_pos[i]
      # "treatment" emmeans factor maps to raw column "treatment_state" (binary status)
      raw_col <- if (fct == "treatment" && "treatment_state" %in% colnames(raw_data)) "treatment_state" else fct
      if (!raw_col %in% colnames(raw_data)) next
      rd <- raw_data %>%
        filter(organism == org) %>%
        { if ("subsample_ref" %in% colnames(.)) filter(., subsample_ref == "500") else . } %>%
        mutate(level_raw = str_to_title(str_replace_all(as.character(.data[[raw_col]]), "_", " "))) %>%
        filter(level_raw == lbl) %>%
        mutate(y_pos = yp) %>%
        select(y_pos, macro_f1, organism)
      if (nrow(rd) > 0) raw_rows_c[[length(raw_rows_c) + 1]] <- rd
    }
    if (length(raw_rows_c) > 0) raw_points_c <- bind_rows(raw_rows_c)
  }

  p <- ggplot(plot_df, aes(x = response, y = y_pos, colour = organism)) +
    geom_vline(xintercept = grand_mean, colour = "gray", linetype = "dashed",
               linewidth = 0.5, alpha = 0.6)

  if (!is.null(raw_points_c) && nrow(raw_points_c) > 0) {
    p <- p + geom_boxplot(data = raw_points_c,
                          aes(x = macro_f1, y = y_pos, group = factor(y_pos)),
                          inherit.aes = FALSE, width = 0.35, alpha = 0.3,
                          outlier.size = 0.5, colour = "gray50", fill = "gray80",
                          linewidth = 0.4)
  }

  p <- p +
    geom_segment(aes(x = lower, xend = upper, yend = y_pos), linewidth = 1.2) +
    geom_point(size = 3) +
    scale_colour_manual(values = ORGANISM_COLORS, name = "Organism") +
    scale_x_continuous(breaks = c(0, 0.25, 0.5, 0.75, 1.0)) +
    scale_y_continuous(
      breaks = plot_df$y_pos,
      labels = plot_df$label,
      expand = expansion(mult = c(0.05, 0.05))
    ) +
    labs(x = "Agreement with Author Labels (Macro F1)", y = "") +
    pub_theme()

  for (sp in separator_positions) {
    p <- p + geom_hline(yintercept = sp, colour = "lightgray",
                        linewidth = 0.5, alpha = 0.7)
  }


  # Significance brackets: one per individual pairwise contrast
  if (!is.null(sig_data) && nrow(sig_data) > 0) {
    y_lookup <- plot_df %>% select(factor_name, organism, label, y_pos)

    bracket_df <- sig_data %>%
      inner_join(y_lookup, by = c("factor_name", "organism", "level1" = "label")) %>%
      rename(y1 = y_pos) %>%
      inner_join(y_lookup, by = c("factor_name", "organism", "level2" = "label")) %>%
      rename(y2 = y_pos) %>%
      mutate(
        y_lo     = pmin(y1, y2),
        y_hi     = pmax(y1, y2),
        y_mid    = (y_lo + y_hi) / 2,
        span     = y_hi - y_lo,
        organism = factor(organism, levels = names(ORGANISM_COLORS))
      )

    if (nrow(bracket_df) > 0) {
      min_span <- min(bracket_df$span)

      # Assign a unique rank per factor+organism group so every bracket gets its own x lane
      bracket_df <- bracket_df %>%
        group_by(factor_name, organism) %>%
        arrange(span, y_lo, .by_group = TRUE) %>%
        mutate(bracket_rank = row_number()) %>%
        ungroup() %>%
        mutate(
          x_spine = 1.06 + (bracket_rank - 1) * 0.09,
          x_tick  = x_spine - 0.012,
          x_star  = x_spine + 0.016
        )

      p <- p +
        geom_segment(
          data = bracket_df,
          aes(x = x_spine, xend = x_spine, y = y_lo, yend = y_hi, colour = organism),
          inherit.aes = FALSE, linewidth = 0.6
        ) +
        geom_segment(
          data = bracket_df,
          aes(x = x_tick, xend = x_spine, y = y_lo, yend = y_lo, colour = organism),
          inherit.aes = FALSE, linewidth = 0.6
        ) +
        geom_segment(
          data = bracket_df,
          aes(x = x_tick, xend = x_spine, y = y_hi, yend = y_hi, colour = organism),
          inherit.aes = FALSE, linewidth = 0.6
        ) +
        geom_text(
          data = bracket_df,
          aes(x = x_star, y = y_mid, label = sig, colour = organism),
          inherit.aes = FALSE, hjust = 0, size = 12
        )
    }
  }

  p + coord_cartesian(xlim = c(0, 1.30), clip = "off")
}

# -- Panel E: Variance Source Comparison (logit scale) ------------------------

create_panel_e <- function(hs_coef_path, mm_coef_path, primary_key) {
  # Shows study random intercept SD vs fixed-effect predictor magnitudes,
  # demonstrating that study of origin dominates unexplained variance.
  # Uses facet_wrap per organism to avoid position_dodge segment artifacts.
  PANEL_E_COLORS <- c(random = "#e74c3c", fixed = "#888888")

  load_one <- function(path, organism_label) {
    if (is.null(path) || !file.exists(path)) return(NULL)
    df <- read_tsv(path, show_col_types = FALSE) %>% filter(key == primary_key)

    study_row <- df %>%
      filter(effect == "ran_pars", group == "study", term == "sd__(Intercept)")
    if (nrow(study_row) == 0) return(NULL)
    study_sd <- study_row$estimate[1]

    fixed <- df %>%
      filter(effect == "fixed", term != "(Intercept)", !str_detect(term, ":")) %>%
      mutate(
        predictor = case_when(
          str_starts(term, "reference")                                      ~ "Reference",
          str_starts(term, "method")                                        ~ "Method",
          term == "cutoff"                                                   ~ "Cutoff",
          str_starts(term, "subsample_ref")                                 ~ "Subsampling",
          str_starts(term, "sex")                                           ~ "Sex",
          str_starts(term, "disease_state") | str_starts(term, "treatment") ~ "Disease/Treatment",
          str_starts(term, "region_match")                                  ~ "Region match",
          TRUE ~ NA_character_
        )
      ) %>%
      filter(!is.na(predictor), !is.na(estimate)) %>%
      group_by(predictor) %>%
      summarise(
        # Multi-level predictors: SD of coefficients; single-level: |coefficient|
        magnitude = if (n() > 1) sd(estimate, na.rm = TRUE) else abs(estimate[1]),
        .groups = "drop"
      )

    bind_rows(
      tibble(predictor = "Study (random)", magnitude = study_sd, type = "random"),
      fixed %>% mutate(type = "fixed")
    ) %>%
      mutate(organism = organism_label)
  }

  hs_df <- load_one(hs_coef_path, "Human")
  mm_df <- load_one(mm_coef_path, "Mouse")
  plot_df <- bind_rows(hs_df, mm_df)
  if (is.null(plot_df) || nrow(plot_df) == 0) return(NULL)

  # Order predictors by mean magnitude across organisms
  pred_order <- plot_df %>%
    group_by(predictor) %>%
    summarise(mean_mag = mean(magnitude), .groups = "drop") %>%
    arrange(mean_mag) %>%
    pull(predictor)

  plot_df <- plot_df %>%
    mutate(
      predictor = factor(predictor, levels = pred_order),
      organism  = factor(organism, levels = names(ORGANISM_COLORS))
    )

  # Compute explicit dodged y positions to avoid position_dodge segment artifacts
  dodge_offset <- 0.18
  plot_df <- plot_df %>%
    mutate(
      pred_num = as.numeric(predictor),
      y_pos    = pred_num + ifelse(organism == "Human", dodge_offset, -dodge_offset)
    )

  ggplot(plot_df, aes(colour = organism)) +
    geom_segment(aes(x = 0, xend = magnitude, y = y_pos, yend = y_pos),
                 linewidth = 1.2) +
    geom_point(aes(x = magnitude, y = y_pos), size = 3.5) +
    scale_colour_manual(values = ORGANISM_COLORS) +
    scale_y_continuous(
      breaks = seq_along(levels(plot_df$predictor)),
      labels = levels(plot_df$predictor),
      expand = expansion(mult = c(0.05, 0.05))
    ) +
    scale_x_continuous(expand = expansion(mult = c(0, 0.1))) +
    labs(x = "Effect magnitude (logit scale)", y = "", colour = NULL) +
    pub_theme() +
    theme(legend.position = "none")
}

# -- Panel D: Reference Atlas Forest Plot (faceted by organism) ---------------

create_panel_d <- function(reference_data, ref_sig = NULL, raw_data = NULL,
                           organism_filter = NULL) {
  if (!is.null(organism_filter)) {
    reference_data <- reference_data %>% filter(organism == organism_filter)
    if (!is.null(raw_data) && nrow(raw_data) > 0) {
      raw_data <- raw_data %>% filter(organism == organism_filter)
    }
    if (!is.null(ref_sig) && nrow(ref_sig) > 0) {
      ref_sig <- ref_sig %>% filter(organism == organism_filter)
    }
  }

  # Apply abbreviations
  reference_data <- reference_data %>%
    mutate(ref_abbr = dplyr::recode(reference, !!!REFERENCE_ABBREVS,
                                    .default = reference))

  methods   <- sort(unique(reference_data$method))
  n_methods <- length(methods)
  dodge_width <- 0.25

  # Compute mean response per organism × reference for ordering
  reference_data <- reference_data %>%
    group_by(organism, ref_abbr) %>%
    mutate(mean_resp = mean(response)) %>%
    ungroup() %>%
    mutate(
      method   = factor(method, levels = names(METHOD_NAMES)),
      organism = factor(organism, levels = names(ORGANISM_COLORS))
    )

  # Order within each organism by mean response ascending
  mouse_order <- reference_data %>%
    filter(organism == "Mouse") %>%
    arrange(mean_resp) %>%
    pull(ref_abbr) %>%
    unique()

  human_order <- reference_data %>%
    filter(organism == "Human") %>%
    arrange(mean_resp) %>%
    pull(ref_abbr) %>%
    unique()

  # Unique factor levels: "Organism##Label" keeps mouse (1..N_mm) then human (N_mm+1..N)
  ref_level_order <- c(
    paste0("Mouse##", mouse_order),
    paste0("Human##", human_order)
  )

  reference_data <- reference_data %>%
    mutate(
      ref_unique = factor(paste0(organism, "##", ref_abbr),
                          levels = ref_level_order),
      y_raw      = as.numeric(ref_unique),
      method_num = as.numeric(method),
      y_dodge    = y_raw + (method_num - (n_methods + 1) / 2) * dodge_width
    )

  grand_mean <- mean(reference_data$response, na.rm = TRUE)

  # Y-axis breaks/labels (strip "Organism##" prefix)
  y_tbl   <- reference_data %>%
    distinct(ref_unique, y_raw) %>%
    arrange(y_raw)
  y_labels <- sub(".*##", "", as.character(y_tbl$ref_unique))

  # Build raw strip points aligned to y_dodge (filter to subsample_ref=500 to match emmeans)
  raw_d <- NULL
  if (!is.null(raw_data) && nrow(raw_data) > 0) {
    dodge_map <- reference_data %>%
      distinct(organism = as.character(organism),
               reference = as.character(reference),
               method = as.character(method),
               y_dodge)
    raw_d <- raw_data %>%
      mutate(organism  = as.character(organism),
             reference = as.character(reference),
             method    = as.character(method)) %>%
      { if ("subsample_ref" %in% colnames(.)) filter(., subsample_ref == "500") else . } %>%
      inner_join(dodge_map, by = c("organism", "reference", "method"))
  }

  p <- ggplot(reference_data, aes(x = response, y = y_dodge, colour = organism)) +
    geom_vline(xintercept = grand_mean, colour = "gray", linetype = "dashed",
               linewidth = 0.5, alpha = 0.6)

  if (!is.null(raw_d) && nrow(raw_d) > 0) {
    p <- p + geom_boxplot(data = raw_d,
                          aes(x = macro_f1, y = y_dodge, group = factor(y_dodge)),
                          inherit.aes = FALSE, width = 0.7, alpha = 0.3,
                          outlier.size = 1.2, colour = "gray50", fill = "gray80",
                          linewidth = 0.8)
  }

  p <- p +
    geom_segment(aes(x = `asymp.LCL`, xend = `asymp.UCL`, yend = y_dodge,
                     linetype = method),
                 linewidth = 1.5) +
    geom_point(size = 4) +
    facet_wrap(~ organism, scales = "free_y", ncol = 2) +
    scale_colour_manual(values = ORGANISM_COLORS, name = "Organism") +
    scale_linetype_manual(values = METHOD_LINETYPES, labels = METHOD_NAMES,
                          name = "Method") +
    scale_y_continuous(
      breaks = y_tbl$y_raw,
      labels = y_labels,
      expand = expansion(mult = c(0.05, 0.05))
    ) +
    labs(x = "Agreement with Author Labels (Macro F1)", y = "Reference Datasets") +
    scale_x_continuous(breaks = c(0, 0.5, 1.0)) +
    coord_cartesian(xlim = c(0, 1.25), clip = "off") +
    pub_theme()

  # Pairwise method significance brackets (one per method pair per reference),
  # stacked in x-lanes when a reference has >1 pairwise contrast (3+ methods).
  if (!is.null(ref_sig) && nrow(ref_sig) > 0) {
    ref_sig_abbr <- ref_sig %>%
      mutate(
        ref_abbr = dplyr::recode(reference, !!!REFERENCE_ABBREVS, .default = reference),
        organism = as.character(organism)
      )

    # y_dodge lookup per (organism, ref_abbr, method)
    dodge_lookup <- reference_data %>%
      distinct(organism, ref_abbr, method, y_dodge) %>%
      mutate(
        organism = as.character(organism),
        method   = as.character(method)
      )

    bracket_df <- ref_sig_abbr %>%
      inner_join(dodge_lookup,
                 by = c("organism", "ref_abbr", "method1" = "method")) %>%
      rename(y1 = y_dodge) %>%
      inner_join(dodge_lookup,
                 by = c("organism", "ref_abbr", "method2" = "method")) %>%
      rename(y2 = y_dodge) %>%
      mutate(
        y_lo     = pmin(y1, y2),
        y_hi     = pmax(y1, y2),
        y_mid    = (y_lo + y_hi) / 2,
        span     = y_hi - y_lo,
        organism = factor(organism, levels = names(ORGANISM_COLORS))
      )

    if (nrow(bracket_df) > 0) {
      # One x-lane per (organism, ref) so 3+ method pairs don't overlap
      bracket_df <- bracket_df %>%
        group_by(organism, ref_abbr) %>%
        arrange(span, y_lo, .by_group = TRUE) %>%
        mutate(bracket_rank = row_number()) %>%
        ungroup() %>%
        mutate(
          x_spine = 1.05 + (bracket_rank - 1) * 0.04,
          x_tick  = x_spine - 0.012,
          x_star  = x_spine + 0.016
        )

      p <- p +
        geom_segment(data = bracket_df,
                     aes(x = x_spine, xend = x_spine, y = y_lo, yend = y_hi),
                     inherit.aes = FALSE, colour = "black", linewidth = 0.5) +
        geom_segment(data = bracket_df,
                     aes(x = x_tick, xend = x_spine, y = y_lo, yend = y_lo),
                     inherit.aes = FALSE, colour = "black", linewidth = 0.5) +
        geom_segment(data = bracket_df,
                     aes(x = x_tick, xend = x_spine, y = y_hi, yend = y_hi),
                     inherit.aes = FALSE, colour = "black", linewidth = 0.5) +
        geom_text(data = bracket_df,
                  aes(x = x_star, y = y_mid, label = sig),
                  inherit.aes = FALSE, hjust = 0, size = 12)
    }
  }

  p
}

# -- Data Loading -------------------------------------------------------------

load_factor_data <- function(hs_files, mm_files, primary_key) {
  load_organism_factors <- function(files, organism_name) {
    rows <- list()

    for (fp in files) {
      if (!file.exists(fp)) {
        message("  Warning: File not found: ", fp)
        next
      }
      fct <- extract_factor_from_path(fp)
      if (is.na(fct)) {
        message("  Warning: Could not extract factor name from: ", fp)
        next
      }

      df <- read_tsv(fp, show_col_types = FALSE)
      df <- filter_key(df, primary_key)
      if (nrow(df) == 0) {
        message("  Warning: No data for key=", primary_key, " in: ", fp)
        next
      }

      # Identify the primary factor column (non-standard columns)
      standard_cols <- c("response", "SE", "df", "asymp.LCL", "asymp.UCL", "key")
      factor_cols   <- setdiff(colnames(df), standard_cols)
      if (length(factor_cols) == 0) next
      primary_col <- factor_cols[1]

      # Human sex has a disease_state interaction column: average by sex level
      if (fct == "sex" && "disease_state" %in% colnames(df)) {
        df <- df %>%
          group_by(!!sym(primary_col)) %>%
          summarise(
            response  = mean(response, na.rm = TRUE),
            `asymp.LCL` = mean(`asymp.LCL`, na.rm = TRUE),
            `asymp.UCL` = mean(`asymp.UCL`, na.rm = TRUE),
            .groups = "drop"
          )
      }

      # Standardize schema: (level, response, asymp.LCL, asymp.UCL)
      df <- df %>%
        select(level = !!sym(primary_col), response, `asymp.LCL`, `asymp.UCL`) %>%
        mutate(level = as.character(level)) %>%
        # Replace empty string (MM sex baseline = female)
        mutate(level = if_else(level == "", "female", level)) %>%
        # Drop uninformative levels
        filter(!tolower(level) %in% c("none", "nan", "na")) %>%
        mutate(
          level       = str_to_title(str_replace_all(level, "_", " ")),
          factor_name = fct,
          organism    = organism_name
        )

      if (nrow(df) == 0) next
      rows[[length(rows) + 1]] <- df
      message("  Loaded ", fct, " (", organism_name, "): ", nrow(df), " levels")
    }

    bind_rows(rows)
  }

  hs_data <- load_organism_factors(hs_files, "Human")
  mm_data <- load_organism_factors(mm_files, "Mouse")
  bind_rows(hs_data, mm_data)
}

load_reference_significance <- function(hs_file, mm_file, primary_key) {
  # Accepts methods: scvi, scvi_rf, scvi_knn, seurat
  method_alt <- "(scvi_rf|scvi_knn|scvi|seurat)"
  pat_no_paren <- paste0("^(.+) ", method_alt, " / (.+) ", method_alt, "$")
  pat_parens   <- paste0("^\\((.+) ", method_alt, "\\) / \\((.+) ", method_alt, "\\)$")

  parse_contrast <- function(contrast) {
    m <- regmatches(contrast, regexec(pat_parens, contrast, perl = TRUE))[[1]]
    if (length(m) < 5) {
      m <- regmatches(contrast, regexec(pat_no_paren, contrast, perl = TRUE))[[1]]
    }
    if (length(m) < 5) return(c(NA_character_, NA_character_, NA_character_, NA_character_))
    c(trimws(m[2]), m[3], trimws(m[4]), m[5])
  }

  rows <- list()
  for (item in list(list(fp = hs_file, org = "Human"),
                    list(fp = mm_file, org = "Mouse"))) {
    fp <- item$fp; org <- item$org
    if (!file.exists(fp)) {
      message("  Warning: File not found: ", fp)
      next
    }
    df <- read_tsv(fp, show_col_types = FALSE) %>%
      filter(key == primary_key)
    if (nrow(df) == 0) next

    parsed <- t(vapply(df$contrast, parse_contrast, character(4L)))
    df <- df %>%
      mutate(
        ref1     = parsed[, 1],
        method1  = parsed[, 2],
        ref2     = parsed[, 3],
        method2  = parsed[, 4],
        organism = org,
        sig      = sig_stars(p.value)
      ) %>%
      filter(!is.na(ref1), !is.na(ref2), ref1 == ref2, method1 != method2) %>%
      mutate(reference = ref1) %>%
      select(reference, method1, method2, organism, p_value = p.value, sig)
    if (nrow(df) == 0) next
    rows[[length(rows) + 1]] <- df
    message("  Loaded reference sig (", org, "): ", nrow(df), " within-ref method contrasts")
  }
  bind_rows(rows)
}

load_factor_significance <- function(hs_files, mm_files, primary_key) {
  # Normalise a raw contrast level name to match the plot row labels
  normalize_level <- function(raw, factor_name) {
    raw <- trimws(raw)
    if (!nzchar(raw)) raw <- "female"               # empty string = female (mouse sex baseline)
    raw <- sub("^subsample_ref", "", raw)            # strip numeric-prefix for subsample_ref
    if (factor_name == "sex") raw <- strsplit(raw, " ")[[1]][1]  # strip disease_state suffix (human sex)
    str_to_title(str_replace_all(raw, "_", " "))
  }

  load_org_contrasts <- function(files, organism_name) {
    rows <- list()
    for (fp in files) {
      if (!file.exists(fp)) next
      base <- basename(fp)
      m <- regmatches(base, regexec("(.+)_emmeans_estimates\\.tsv", base))
      if (length(m[[1]]) < 2) next
      fct <- m[[1]][2]

      df <- read_tsv(fp, show_col_types = FALSE) %>% filter(key == primary_key)
      if (nrow(df) == 0) next

      # Human sex: keep only the sex-within-disease contrast (contains both female and male)
      if (fct == "sex" && organism_name == "Human") {
        df <- df %>% filter(
          grepl("female", contrast, ignore.case = TRUE),
          grepl("male",   contrast, ignore.case = TRUE)
        )
      }
      if (nrow(df) == 0) next

      df <- df %>%
        mutate(
          raw1        = trimws(sub(" / .+$", "", contrast)),
          raw2        = trimws(sub("^.+ / ", "", contrast)),
          level1      = vapply(raw1, normalize_level, character(1L), factor_name = fct),
          level2      = vapply(raw2, normalize_level, character(1L), factor_name = fct),
          factor_name = fct,
          organism    = organism_name,
          sig         = sig_stars(p.value)
        ) %>%
        select(factor_name, organism, level1, level2, p_value = p.value, sig)

      rows[[length(rows) + 1]] <- df
    }
    bind_rows(rows)
  }

  bind_rows(
    load_org_contrasts(hs_files, "Human"),
    load_org_contrasts(mm_files, "Mouse")
  )
}

# -- CLI Arguments ------------------------------------------------------------

parse_args_cli <- function() {
  p <- ArgumentParser(description = "Combined Human + Mouse publication figure")

  p$add_argument(
    "--hs_model_dir",
    default = "2024-07-01/homo_sapiens_main_branch/100/dataset_id/SCT/gap_false/aggregated_models/macro_f1_~_reference_+_method_+_cutoff_+_subsample_ref_+_disease_state_+_sex_+_method:cutoff_+_reference:method/files",
    help    = "Path to HS aggregated_models/.../files directory"
  )
  p$add_argument(
    "--mm_model_dir",
    default = "2024-07-01/mus_musculus_main_branch/100/dataset_id/SCT/gap_false/aggregated_models/macro_f1_~_reference_+_method_+_cutoff_+_subsample_ref_+_treatment_state_+_method:cutoff_+_reference:method/files",
    help    = "Path to MM aggregated_models/.../files directory"
  )
  p$add_argument(
    "--hs_sample_results",
    default = "2024-07-01/homo_sapiens_main_branch/100/dataset_id/SCT/gap_false/aggregated_results/files/sample_results.tsv.gz",
    help    = "Path to HS sample_results.tsv.gz (optional; used for raw data overlays)"
  )
  p$add_argument(
    "--mm_sample_results",
    default = "2024-07-01/mus_musculus_main_branch/100/dataset_id/SCT/gap_false/aggregated_results/files/sample_results.tsv.gz",
    help    = "Path to MM sample_results.tsv.gz (optional; used for raw data overlays)"
  )
  p$add_argument(
    "--hs_factors",
    nargs   = "+",
    default = c("disease_state", "sex"),
    help    = "Factor names for human Panel C (files derived from --hs_model_dir)"
  )
  p$add_argument(
    "--mm_factors",
    nargs   = "+",
    default = c("treatment"),
    help    = "Factor names for mouse Panel C (files derived from --mm_model_dir)"
  )
  p$add_argument("--primary_key",    default = "subclass")
  p$add_argument("--outdir",         default = "combined_orgs/main_branch")
  p$add_argument("--output_prefix",  default = "combined_figure")
  p$add_argument(
    "--methods",
    nargs   = "+",
    default = c("scvi", "seurat"),
    help    = "Methods to include in the legend (default: main-branch two-classifier set)"
  )

  a      <- p$parse_args()
  hs_dir <- a$hs_model_dir
  mm_dir <- a$mm_model_dir

  list(
    hs_cutoff_effects      = file.path(hs_dir, "method_cutoff_effects.tsv"),
    mm_cutoff_effects      = file.path(mm_dir, "method_cutoff_effects.tsv"),
    hs_reference_emmeans   = file.path(hs_dir, "reference_method_emmeans_summary.tsv"),
    mm_reference_emmeans   = file.path(mm_dir, "reference_method_emmeans_summary.tsv"),
    hs_method_emmeans      = file.path(hs_dir, "method_emmeans_summary.tsv"),
    mm_method_emmeans      = file.path(mm_dir, "method_emmeans_summary.tsv"),
    hs_factor_emmeans      = paste(
      file.path(hs_dir, paste0(a$hs_factors, "_emmeans_summary.tsv")),
      collapse = " "
    ),
    mm_factor_emmeans      = paste(
      file.path(mm_dir, paste0(a$mm_factors, "_emmeans_summary.tsv")),
      collapse = " "
    ),
    hs_factor_estimates    = paste(
      file.path(hs_dir, paste0(a$hs_factors, "_emmeans_estimates.tsv")),
      collapse = " "
    ),
    mm_factor_estimates    = paste(
      file.path(mm_dir, paste0(a$mm_factors, "_emmeans_estimates.tsv")),
      collapse = " "
    ),
    hs_reference_estimates = file.path(hs_dir, "reference_method_emmeans_estimates.tsv"),
    mm_reference_estimates = file.path(mm_dir, "reference_method_emmeans_estimates.tsv"),
    hs_model_coefs         = file.path(hs_dir, "model_coefs.tsv"),
    mm_model_coefs         = file.path(mm_dir, "model_coefs.tsv"),
    hs_sample_results      = a$hs_sample_results,
    mm_sample_results      = a$mm_sample_results,
    primary_key            = a$primary_key,
    outdir                 = a$outdir,
    output_prefix          = a$output_prefix,
    methods                = a$methods
  )
}

# -- Main ---------------------------------------------------------------------

main <- function() {
  args <- parse_args_cli()
  dir.create(args$outdir, recursive = TRUE, showWarnings = FALSE)
  pk <- args$primary_key

  # Restrict method legend/aesthetics to the requested set
  missing_methods <- setdiff(args$methods, names(METHOD_NAMES))
  if (length(missing_methods) > 0) {
    stop("Unknown method(s) in --methods: ", paste(missing_methods, collapse = ", "),
         ". Known: ", paste(names(METHOD_NAMES), collapse = ", "))
  }
  METHOD_NAMES     <<- METHOD_NAMES[args$methods]
  METHOD_LINETYPES <<- METHOD_LINETYPES[args$methods]

  message("Loading data (key=", pk, ")...")

  # Raw sample results for strip overlay
  load_raw <- function(path, organism_label, pk) {
    if (is.null(path) || !file.exists(path)) return(NULL)
    df <- read_tsv(path, show_col_types = FALSE, col_types = cols(.default = col_character()))
    if ("key" %in% colnames(df)) df <- df %>% filter(key == pk)
    # Filter to cutoff=0 (reference configuration for emmeans)
    if ("cutoff" %in% colnames(df)) df <- df %>% filter(as.numeric(cutoff) == 0)
    keep_cols <- intersect(
      c("macro_f1", "reference", "method", "key",
        "disease_state", "treatment_state", "treatment", "sex", "subsample_ref"),
      colnames(df)
    )
    df %>%
      select(all_of(keep_cols)) %>%
      mutate(macro_f1 = as.numeric(macro_f1), organism = organism_label)
  }
  hs_raw  <- load_raw(args$hs_sample_results, "Human", pk)
  mm_raw  <- load_raw(args$mm_sample_results, "Mouse", pk)
  raw_data <- bind_rows(hs_raw, mm_raw)
  if (nrow(raw_data) > 0) {
    message("  Raw sample data: ", nrow(raw_data), " rows")
  } else {
    raw_data <- NULL
  }

  # Cutoff effects
  hs_cutoff <- read_tsv(args$hs_cutoff_effects, show_col_types = FALSE) %>%
    filter_key(pk) %>%
    mutate(organism = "Human")
  mm_cutoff <- read_tsv(args$mm_cutoff_effects, show_col_types = FALSE) %>%
    filter_key(pk) %>%
    mutate(organism = "Mouse")
  cutoff_data <- bind_rows(hs_cutoff, mm_cutoff)
  message("  Cutoff data: ", nrow(cutoff_data), " rows")

  # Method emmeans (taxonomy)
  hs_tax <- read_tsv(args$hs_method_emmeans, show_col_types = FALSE) %>%
    mutate(organism = "Human")
  mm_tax <- read_tsv(args$mm_method_emmeans, show_col_types = FALSE) %>%
    mutate(organism = "Mouse")
  taxonomy_data <- bind_rows(hs_tax, mm_tax)
  message("  Taxonomy data: ", nrow(taxonomy_data), " rows, keys: ",
          paste(sort(unique(taxonomy_data$key)), collapse = ", "))

  # Reference emmeans
  hs_ref <- read_tsv(args$hs_reference_emmeans, show_col_types = FALSE) %>%
    filter_key(pk) %>%
    mutate(organism = "Human")
  mm_ref <- read_tsv(args$mm_reference_emmeans, show_col_types = FALSE) %>%
    filter_key(pk) %>%
    mutate(organism = "Mouse")
  reference_data <- bind_rows(hs_ref, mm_ref)
  message("  Reference data: ", nrow(reference_data), " rows")

  # Factor emmeans
  hs_factor_files <- if (nzchar(trimws(args$hs_factor_emmeans))) {
    strsplit(trimws(args$hs_factor_emmeans), "\\s+")[[1]]
  } else {
    character(0)
  }
  mm_factor_files <- if (nzchar(trimws(args$mm_factor_emmeans))) {
    strsplit(trimws(args$mm_factor_emmeans), "\\s+")[[1]]
  } else {
    character(0)
  }

  message("Loading factor emmeans (key=", pk, ")...")
  factor_data <- load_factor_data(hs_factor_files, mm_factor_files, pk)
  message("  Factor data: ", nrow(factor_data), " rows")

  # Factor significance (Panel C brackets)
  hs_factor_est_files <- strsplit(trimws(args$hs_factor_estimates), "\\s+")[[1]]
  mm_factor_est_files <- strsplit(trimws(args$mm_factor_estimates), "\\s+")[[1]]
  message("Loading factor significance (key=", pk, ")...")
  factor_sig <- load_factor_significance(hs_factor_est_files, mm_factor_est_files, pk)
  message("  Factor sig: ", nrow(factor_sig), " rows")

  # Reference significance (Panel D brackets)
  message("Loading reference significance (key=", pk, ")...")
  ref_sig <- load_reference_significance(
    args$hs_reference_estimates, args$mm_reference_estimates, pk
  )
  message("  Reference sig: ", nrow(ref_sig), " rows")

  # -- Build panels -----------------------------------------------------------
  message("\nCreating panels...")
  p_a <- create_panel_a(cutoff_data)
  p_b <- create_panel_b(taxonomy_data)
  p_c <- create_panel_c(factor_data, factor_sig, raw_data = raw_data)
  p_d_human <- create_panel_d(reference_data, ref_sig, raw_data = raw_data,
                              organism_filter = "Human")
  p_d_mouse <- create_panel_d(reference_data, ref_sig, raw_data = raw_data,
                              organism_filter = "Mouse")
  p_e <- create_panel_e(args$hs_model_coefs, args$mm_model_coefs, pk)

  # -- Save individual panels -------------------------------------------------
  save_panel <- function(p, suffix, width, height) {
    path <- file.path(args$outdir, paste0(args$output_prefix, "_", suffix, ".png"))
    message("Saving ", path, " ...")
    ggsave(path, p, width = width, height = height, dpi = 300, bg = "white")
  }

  save_panel(p_a, "panel_A_cutoff",    width = 13, height = 9)
  save_panel(p_b, "panel_B_taxonomy",  width = 13, height = 12)
  save_panel(p_c, "panel_C_factors",   width = 21, height = 14)
  save_panel(p_d_human, "panel_D_reference_human", width = 18, height = 16)
  save_panel(p_d_mouse, "panel_D_reference_mouse", width = 18, height = 10)
  if (!is.null(p_e)) save_panel(p_e, "panel_E_variance", width = 14, height = 10)

  legend_path <- file.path(args$outdir, paste0(args$output_prefix, "_legend.png"))
  message("Saving legend to ", legend_path, " ...")
  p_legend <- make_legend_plot()
  ggsave(legend_path, p_legend, width = 12, height = 8, dpi = 300, bg = "white")

  message("Done! Panels saved to ", args$outdir)
}

main()
