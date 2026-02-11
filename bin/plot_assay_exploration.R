#!/usr/bin/env Rscript
library(argparse)
library(tidyverse)

parser <- ArgumentParser()
parser$add_argument("--sample_results",
                    default = "2024-07-01/mus_musculus/100/dataset_id/SCT/gap_false/aggregated_results/files/sample_results.tsv")
parser$add_argument("--contrasts",
                    default = "2024-07-01/mus_musculus/100/dataset_id/SCT/gap_false/assay_exploration/files/assay_emmeans_contrasts.tsv")
parser$add_argument("--outdir", default = "assay_exploration")
args <- parser$parse_args()

dir.create(args$outdir, showWarnings = FALSE, recursive = TRUE)

df <- read_tsv(args$sample_results, show_col_types = FALSE) %>%
  filter(cutoff == 0)

# Simplify labels
df <- df %>%
  mutate(
    query_type = case_when(
      query_suspension_type == "cell" ~ "single-cell",
      query_suspension_type == "nucleus" ~ "single-nucleus",
      TRUE ~ query_suspension_type
    ),
    ref_type = case_when(
      ref_suspension_type == "cell" ~ "cell only",
      ref_suspension_type == "cell, nucleus" ~ "cell + nucleus",
      TRUE ~ ref_suspension_type
    )
  )

KEY_ORDER <- c("subclass", "class", "family", "global")
df$key <- factor(df$key, levels = KEY_ORDER)

theme_set(theme_bw(base_size = 12))

# --- Read contrasts and build annotation data ---
contrasts_df <- read_tsv(args$contrasts, show_col_types = FALSE)

# Parse contrast strings to extract the two groups being compared
# Format: "(ref_type query_type) / (ref_type query_type)"
# Map model factor names back to plot labels
ref_label <- c(cell_only = "cell only", cell_and_nucleus = "cell + nucleus")
query_label <- c("single-cell" = "single-cell", "single-nucleus" = "single-nucleus")

parse_contrast <- function(contrast_str) {
  # Extract the two groups from e.g. "(cell_only single-cell) / (cell_and_nucleus single-cell)"
  groups <- str_match_all(contrast_str, "\\(([^ ]+) ([^)]+)\\)")[[1]]
  tibble(
    ref1 = ref_label[groups[1, 2]], query1 = groups[1, 3],
    ref2 = ref_label[groups[2, 2]], query2 = groups[2, 3]
  )
}

annotations <- contrasts_df %>%
  rowwise() %>%
  mutate(parsed = list(parse_contrast(contrast))) %>%
  unnest(parsed) %>%
  ungroup()

# Format p-values
format_p <- function(p) {
  case_when(
    p < 0.001 ~ "***",
    p < 0.01  ~ "**",
    p < 0.05  ~ "*",
    TRUE      ~ "ns"
  )
}

# Filter to only within-query-group comparisons (ref type effect within each query type)
annotations <- annotations %>%
  filter(query1 == query2)

# FDR correction across all contrasts (across keys)
annotations <- annotations %>%
  mutate(p_adj = p.adjust(p.value, method = "fdr"),
         p_label = format_p(p_adj))

# For the boxplot, x positions are: single-cell=1, single-nucleus=2
# Within each x, dodge positions for fill are: cell+nucleus=-0.1875, cell only=+0.1875
# (ggplot2 default dodge width for boxplot is 0.75, so offset = 0.75/4 = 0.1875)
dodge <- 0.75
hw <- dodge / 4  # half-width offset for dodged boxes

get_x <- function(query, ref) {
  base <- ifelse(query == "single-cell", 1, 2)
  # fill order: "cell + nucleus" first (left), "cell only" second (right)
  offset <- ifelse(ref == "cell + nucleus", -hw, hw)
  base + offset
}

annotations <- annotations %>%
  mutate(
    x1 = get_x(query1, ref1),
    x2 = get_x(query2, ref2),
    key = factor(key, levels = KEY_ORDER)
  )

# Assign y positions for brackets (stagger within each key)
annotations <- annotations %>%
  group_by(key) %>%
  arrange(key, x1) %>%
  mutate(y_pos = 1.02 + 0.05 * (row_number() - 1)) %>%
  ungroup()

# --- Plot 1: Macro F1 faceted by query assay, colored by reference type, with contrasts ---
p <- ggplot(df, aes(x = query_type, y = macro_f1, fill = ref_type)) +
  geom_boxplot(alpha = 0.7, outlier.size = 0.5) +
  facet_wrap(~ key, nrow = 1) +
  scale_fill_manual(values = c("cell only" = "#9467bd", "cell + nucleus" = "#8c564b")) +
  # Significance brackets
  geom_segment(data = annotations, inherit.aes = FALSE,
               aes(x = x1, xend = x2, y = y_pos, yend = y_pos),
               linewidth = 0.3) +
  geom_segment(data = annotations, inherit.aes = FALSE,
               aes(x = x1, xend = x1, y = y_pos, yend = y_pos - 0.01),
               linewidth = 0.3) +
  geom_segment(data = annotations, inherit.aes = FALSE,
               aes(x = x2, xend = x2, y = y_pos, yend = y_pos - 0.01),
               linewidth = 0.3) +
  geom_text(data = annotations, inherit.aes = FALSE,
            aes(x = (x1 + x2) / 2, y = y_pos + 0.01, label = p_label),
            size = 3) +
  scale_y_continuous(expand = expansion(mult = c(0.05, 0.15))) +
  labs(x = "Query assay", y = "Macro F1", fill = "Reference type") +
  theme(legend.position = "bottom")

ggsave(file.path(args$outdir, "macro_f1_query_x_ref.png"),
       p, width = 12, height = 5.5, bg = "white")

# --- Plot 2: Macro F1 by study, colored by query assay type ---
p2 <- ggplot(df, aes(x = reorder(study, macro_f1, FUN = median),
                      y = macro_f1, fill = query_type)) +
  geom_boxplot(alpha = 0.7, outlier.size = 0.5) +
  facet_wrap(~ key, nrow = 1) +
  scale_fill_manual(values = c("single-cell" = "#2ca02c", "single-nucleus" = "#d62728")) +
  labs(x = "Study", y = "Macro F1", fill = "Query assay") +
  theme(legend.position = "bottom",
        axis.text.x = element_text(angle = 45, hjust = 1, size = 7))

ggsave(file.path(args$outdir, "macro_f1_by_study.png"),
       p2, width = 14, height = 5, bg = "white")

# --- Plot 3: Macro F1 by study, faceted by query assay, colored by reference type ---
p3 <- ggplot(df, aes(x = study, y = macro_f1, fill = ref_type)) +
  geom_boxplot(alpha = 0.7, outlier.size = 0.5) +
  facet_grid(key ~ query_type, scales = "free_x", space = "free_x") +
  scale_fill_manual(values = c("cell only" = "#9467bd", "cell + nucleus" = "#8c564b")) +
  labs(x = "Study", y = "Macro F1", fill = "Reference type") +
  theme(legend.position = "bottom",
        axis.text.x = element_text(angle = 45, hjust = 1, size = 7))

ggsave(file.path(args$outdir, "macro_f1_by_study_query_ref.png"),
       p3, width = 12, height = 10, bg = "white")
