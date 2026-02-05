#!/usr/bin/env Rscript
# Computational Performance Figure (Supplementary S1)
#
# Creates point-and-line plots with error bars showing:
#   Panel A: Mean runtime (hours) by process step
#   Panel B: Mean peak memory (GB) by process step
#
# Usage:
#   Rscript plot_comptime.R --all_runs path/to/runs_directory --outdir figures

library(argparse)
library(dplyr)
library(tidyr)
library(readr)
library(stringr)
library(ggplot2)
library(patchwork)
library(yaml)

# -- Constants ----------------------------------------------------------------

# Map each Nextflow process to its methodology and pipeline stage
PROCESS_METHOD <- c(
  MAP_QUERY            = "scVI",
  RF_PREDICT           = "scVI",
  REF_PROCESS_SEURAT   = "Seurat",
  QUERY_PROCESS_SEURAT = "Seurat",
  PREDICT_SEURAT       = "Seurat"
)

PROCESS_STEP <- c(
  MAP_QUERY            = "Query Processing",
  RF_PREDICT           = "Prediction",
  REF_PROCESS_SEURAT   = "Ref Processing",
  QUERY_PROCESS_SEURAT = "Query Processing",
  PREDICT_SEURAT       = "Prediction"
)

# Pipeline stage order (processing → prediction)
STEP_ORDER <- c("Ref Processing", "Query Processing", "Prediction")

# Method colors (consistent with plot_utils.py METHOD_COLORS)
METHOD_COLORS <- c(scVI = "#1f77b4", Seurat = "#ff7f0e")

PROCESS_ORDER <- names(PROCESS_METHOD)

KEYS_TO_DROP <- c("ref_collections", "ref_keys", "outdir",
                  "batch_keys", "relabel_r", "relabel_q",
                  "tree_file", "queries_adata")

# -- Helpers ------------------------------------------------------------------

convert_time <- function(x) {
  # Convert Nextflow time strings (e.g. "1h30m45.2s", "200ms") to hours
  ifelse(is.na(x) | x == "-", NA_real_, {
    x <- tolower(gsub(" ", "", x))
    hrs  <- as.numeric(ifelse(grepl("(\\d+)h", x), sub(".*?(\\d+)h.*", "\\1", x), "0"))
    mins <- as.numeric(ifelse(grepl("(\\d+)m(?!s)", x, perl = TRUE),
                              sub(".*?(\\d+)m(?!s).*", "\\1", x, perl = TRUE), "0"))
    secs <- as.numeric(ifelse(grepl("([\\d.]+)s(?!.*ms)", x, perl = TRUE),
                              sub(".*?([\\d.]+)s.*", "\\1", x, perl = TRUE), "0"))
    ms   <- as.numeric(ifelse(grepl("(\\d+)ms", x), sub(".*?(\\d+)ms.*", "\\1", x), "0"))
    (hrs * 3600 + mins * 60 + secs + ms / 1000) / 3600
  })
}

convert_memory <- function(x) {
  # Convert memory string (first token, assumed MB) to GB
  ifelse(is.na(x) | x == "-", NA_real_, {
    val <- suppressWarnings(as.numeric(str_extract(x, "[\\d.]+")))
    val / 1024
  })
}

# -- Data loading -------------------------------------------------------------

load_trace_data <- function(all_runs_dir) {
  trace_files <- list.files(all_runs_dir, pattern = "^trace\\.txt$",
                            recursive = TRUE, full.names = TRUE)
  if (length(trace_files) == 0) return(tibble())

  traces <- lapply(trace_files, function(tf) {
    tryCatch({
      trace <- read_tsv(tf, show_col_types = FALSE)
      params_path <- file.path(dirname(tf), "params.yaml")
      if (file.exists(params_path)) {
        params <- yaml::read_yaml(params_path)
        params[KEYS_TO_DROP] <- NULL
        # Keep only scalar params
        params <- params[vapply(params, function(v) length(v) == 1 && !is.list(v), logical(1))]
        for (nm in names(params)) trace[[nm]] <- params[[nm]]
      }
      trace
    }, error = function(e) {
      message("Warning: Could not load ", tf, ": ", conditionMessage(e))
      NULL
    })
  })

  bind_rows(traces)
}

# -- Plotting -----------------------------------------------------------------

make_panel <- function(stats_df, mean_col, sd_col, ylabel, title_label,
                       show_legend = FALSE) {
  has_subsample <- "subsample_ref" %in% names(stats_df)
  group_col <- if (has_subsample) {
    interaction(stats_df$method, as.character(stats_df$subsample_ref), drop = TRUE)
  } else {
    stats_df$method
  }

  p <- ggplot(stats_df, aes(x = step, y = .data[[mean_col]],
                             color = method, group = group_col)) +
    geom_line(aes(linetype = if (has_subsample) factor(subsample_ref) else NULL),
              linewidth = 0.7) +
    geom_pointrange(
      aes(ymin = .data[[mean_col]] - .data[[sd_col]],
          ymax = .data[[mean_col]] + .data[[sd_col]]),
      size = 0.6, linewidth = 0.8
    ) +
    scale_color_manual(values = METHOD_COLORS, name = "Method") +
    scale_y_continuous(expand = expansion(mult = c(0, 0.05)), limits = c(0, NA)) +
    labs(y = ylabel, x = NULL, title = title_label) +
    theme_bw(base_size = 11) +
    theme(
      plot.title = element_text(face = "bold", hjust = 0),
      axis.text.x = element_text(angle = 45, hjust = 1),
      plot.background  = element_rect(fill = "white", color = NA),
      panel.background = element_rect(fill = "white", color = NA),
      panel.grid.major.x = element_blank(),
      legend.position = if (show_legend) "right" else "none"
    )

  if (has_subsample) {
    p <- p + scale_linetype_discrete(name = "Subsample Ref")
  }

  p
}

# -- Main ---------------------------------------------------------------------

main <- function() {
  parser <- ArgumentParser(description = "Generate computational performance figures")
  parser$add_argument("--all_runs", required = TRUE, help = "Path to runs directory")
  parser$add_argument("--outdir", default = ".", help = "Output directory")
  parser$add_argument("--output_prefix", default = "comptime", help = "Output filename prefix")
  args <- parser$parse_args()

  dir.create(args$outdir, showWarnings = FALSE, recursive = TRUE)

  cat("Loading trace data...\n")
  reports <- load_trace_data(args$all_runs)
  if (nrow(reports) == 0) { stop("No trace data found") }

  # Convert columns
  if ("duration"  %in% names(reports)) reports$duration_hours <- convert_time(reports$duration)
  if ("realtime"  %in% names(reports)) reports$realtime_hours <- convert_time(reports$realtime)
  if ("peak_vmem" %in% names(reports)) reports$memory_gb     <- convert_memory(reports$peak_vmem)

  # Extract process name: "PIPELINE:PROCESS_NAME (sample)" -> "PROCESS_NAME"
  reports$process <- str_trim(str_extract(reports$name, "[^:]+$")) |>
    str_remove("\\s*\\(.*\\)$")

  # Filter to relevant processes
  reports <- reports %>% filter(process %in% PROCESS_ORDER)
  if (nrow(reports) == 0) stop("No relevant processes found in trace data")

  # Add method and pipeline step
  reports$method <- PROCESS_METHOD[reports$process]
  reports$step   <- PROCESS_STEP[reports$process]

  # Summarise per process (optionally by subsample_ref)
  group_cols <- c("process", "method", "step")
  if ("subsample_ref" %in% names(reports)) {
    group_cols <- c(group_cols, "subsample_ref")
  }

  stats <- reports %>%
    group_by(across(all_of(group_cols))) %>%
    summarise(
      mean_duration = mean(duration_hours, na.rm = TRUE),
      sd_duration   = sd(duration_hours,   na.rm = TRUE),
      mean_realtime = mean(realtime_hours,  na.rm = TRUE),
      sd_realtime   = sd(realtime_hours,    na.rm = TRUE),
      mean_memory   = mean(memory_gb,       na.rm = TRUE),
      sd_memory     = sd(memory_gb,         na.rm = TRUE),
      count         = n(),
      .groups = "drop"
    ) %>%
    mutate(across(where(is.numeric), ~ replace_na(., 0)))

  # Ordered factor for x-axis (pipeline order: processing → prediction)
  stats$step <- factor(stats$step, levels = STEP_ORDER)

  cat("Found", nrow(stats), "processes with data\n")
  print(as.data.frame(stats))

  # Save summary table
  write_tsv(stats, file.path(args$outdir, paste0(args$output_prefix, "_summary.tsv")))

  # Build figure (two stacked panels via patchwork; legend on bottom panel only)
  p1 <- make_panel(stats, "mean_duration", "sd_duration",
                    "Duration (hours)", "A. Runtime", show_legend = FALSE)
  p2 <- make_panel(stats, "mean_memory", "sd_memory",
                    "Peak Memory (GB)", "B. Peak Memory", show_legend = TRUE)

  fig <- p1 / p2

  # ~7.1 x 5.9 inches (matching FULL_WIDTH x STANDARD_HEIGHT from plot_utils.py)
  outpath <- file.path(args$outdir, paste0(args$output_prefix, ".png"))
  ggsave(outpath, fig, width = 7.1, height = 5.9, dpi = 300, bg = "white")
  cat("Saved", outpath, "\n")

  outpath_pdf <- file.path(args$outdir, paste0(args$output_prefix, ".pdf"))
  ggsave(outpath_pdf, fig, width = 7.1, height = 5.9, bg = "white")
  cat("Saved", outpath_pdf, "\n")

  cat("Done!\n")
}

main()
