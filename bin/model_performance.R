library(glmmTMB)
library(dplyr)
library(broom.mixed)
library(multcomp)
library(argparse)
library(dplyr)
library(readr)
library(dplyr)
library(tidyr)
library(ggplot2)
library(gridExtra)
library(stringr)

# Set global theme for background
theme_set(
  theme_minimal(base_size =30 ) +  # Base theme
    theme(
      plot.background = element_rect(fill = "white", color = NA),  # Plot background color
      panel.background = element_rect(fill = "white", color = NA), # Panel background color
      legend.background = element_rect(fill = "white", color = NA) # Legend background color
    )
)

run_beta_model <- function(df, formula, group_var = "study") {
  # Ensure the outcome is within (0,1) for Beta regression
  outcome_var <- all.vars(as.formula(formula))[1]
  df[[outcome_var]] <- pmax(pmin(df[[outcome_var]], 1 - 1e-6), 1e-6)

  random_effect_formula <- paste(formula, "+ (1 |", group_var, ")")
    # Add random effects directly to the formula
  model <- glmmTMB(as.formula(random_effect_formula), data = df, family = beta_family(link = "logit"))

  
  # Extract coefficients and p-values
  summary_df <- tidy(model, effects = "fixed") %>%
    mutate(FDR = p.adjust(p.value, method = "fdr"))
  
  # Model fit statistics
  model_stats <- tibble(
    LogLik = logLik(model),
    AIC = AIC(model),
    BIC = BIC(model)
  )
  
  return(list(model = model, summary = summary_df, stats = model_stats, formula = random_effect_formula))
}


run_and_store_model <- function(df, formula, formula_dir, key) {
  # Run the beta model using the run_beta_model function
  result <- run_beta_model(df, formula, group_var = "study")  # Adjust group_var as needed
  
  # Extract model summary coefficients and add additional info
  model_summary_coefs <- result$summary
  model_summary_coefs$formula <- formula
  model_summary_coefs$key <- key
  model_summary_coefs$LogLik <- result$stats$LogLik
  model_summary_coefs$AIC <- result$stats$AIC
  model_summary_coefs$BIC <- result$stats$BIC

  # Save the model summary and coefficients summary to files
  write.table(model_summary_coefs, file = file.path(formula_dir, paste0(key,"_model_summary_coefs_combined.tsv")), sep = "\t", row.names = FALSE)
 # write.table(result$stats, file = file.path(formula_dir, paste0(key,"_model_stats_.tsv")), sep = "\t", row.names = FALSE)
  
  # Plot the model summary (assuming plot_model_summary is defined)
  plot_model_summary(model_summary = model_summary_coefs, outdir = formula_dir, key = key)
  
  # Return the model summary coefficients
  return(model_summary_coefs)
}

# Function to plot model metrics
plot_model_metrics <- function(df_list, formulas) {
  # Initialize an empty list to store results
  results <- list()
  
  # Iterate over the data frames in df_list and formulas
  for (df in df_list) {
    for (formula in formulas) {
      # Run the model (replace with your function to fit the model)
      model_output <- run_beta_model(df, formula)
      stats <- model_output$stats
      # Add key column for each unique 'key' in df
      for (key in unique(df$key)) {
        key_df <- stats %>% 
          mutate(key = key) %>% 
          mutate(formula = model_output$formula)
        
        results <- append(results, list(key_df))
      }
    }
  }
  # Combine the results into a single data frame
  results_df <- do.call(rbind, results)
  library(patchwork)
  results_df$formula_wrapped <- factor(str_wrap(results_df$formula, width = 30))  # Adjust width as needed
  # Plot AIC vs LogLik
  p <- ggplot(results_df, aes(x = AIC, y = LogLik, color = formula_wrapped, shape = key)) +
  geom_point(size = 10, stroke = 2, position = position_jitter(width = 1, height = 1)) +
    labs(
      title = "AIC vs LogLik Across Models",
      x = "AIC",
      y = "LogLik",
      color = "formula",
      shape = "key"
    ) +
    theme(
      legend.title = element_text(size = 30),
      legend.text = element_text(size = 15),
      plot.title = element_text(size = 30),
      axis.title = element_text(size = 30),
    ) +
   # scale_shape_manual(values = c(16, 17, 18)) +  # Customize shapes if needed
  #  scale_color_brewer(palette = "Set2") +  # Customize colors
    theme(legend.position = "right") + plot_annotation()
   # guides(color = guide_legend(title = "key"), shape = guide_legend(title = "Formula"))
    

  # Save the plot
  ggsave("model_metrics.png", p, width = 20, height = 15, dpi = 250)
}


plot_model_summary <- function(model_summary, outdir, key) {
  # Add an FDR < 0.05 column
  model_summary$`FDR < 0.01` <- model_summary$FDR < 0.01
  model_summary$`FDR < 0.05` <- model_summary$FDR < 0.05
  #formula <- unique(model_summary$formula)
  formula <- unique(model_summary$formula)
  formula_wrapped <- unique(factor(str_wrap(formula, width = 10)))  # Adjust width as needed

  model_summary$term <- model_summary$term %>% gsub("reference"," ", .) %>%
                                          gsub("method"," ", .)
  model_summary <- model_summary[order(model_summary$estimate, decreasing=TRUE), ]
  # Create the base plot
  p <- ggplot(model_summary, aes(x = estimate, y = reorder(term, estimate), fill = `FDR < 0.01`)) +
    geom_bar(stat = "identity", show.legend = TRUE) +
    theme(
      axis.text.y = element_text(color = "black"),  # Y-axis text
      axis.text.x = element_text(color = "black"),  # X-axis text
      axis.title = element_text(size = 25),        # Axis title size
      plot.title = element_text(size = 25, hjust = 0.5),  # Title size
      plot.subtitle = element_text(size = 25),      # Subtitle size
      plot.caption = element_text(size = 25),       # Caption size
      axis.ticks = element_line(size = 1),          # Axis tick size
      legend.text = element_text(size = 25),        # Legend text size
      legend.title = element_text(size = 25)        # Legend title size
    ) +
    labs(title = paste(key, " - ", formula_wrapped),
         x = "Coefficient",
         y = "Term")
        
  
  # Highlight significant terms (FDR < 0.05)
  #p <- p + geom_bar(data = model_summary[model_summary$FDR_05, ], 
                     #stat = "identity", 
                     #color = "red", 
                     #size = 1.5,
                     #show.legend = FALSE) +
    #geom_bar(data = model_summary[!model_summary$FDR_05, ], 
             #stat = "identity", 
             #color = "lightgray", 
              #size = 1.5,
             #show.legend = FALSE)
  
  # Add error bars
  p <- p + geom_errorbar(aes(xmin = estimate - 2 * std.error, xmax = estimate + 2 * std.error), width = 0.2)

  # Save the plot to the specified directory
  ggsave(filename = file.path(outdir, paste0(key, "_", formula, "_lm_coefficients.png")), 
  plot = p, width = 35, height = max(10, nrow(model_summary) * 0.5))
}



parser <- argparse::ArgumentParser()
parser$add_argument("--weighted_f1_results", help = "Path to the weighted_f1_results file", default="/space/grp/rschwartz/rschwartz/evaluation_summary.nf/SCT_integrated_hsap/aggregated_results/weighted_f1_results.tsv")
args <- parser$parse_args()


# Reading the weighted_f1_results file
weighted_f1_results <- read.table(args$weighted_f1_results, sep="\t", header=TRUE)
# fill NA with none
weighted_f1_results[is.na(weighted_f1_results)] <- "None"
# Extract organism (assuming only one unique value in the 'organism' column)
organism <- unique(weighted_f1_results$organism)[1]

# Defining factor names
factor_names <- c("study", "reference", "method", "cutoff")

if (organism == "homo_sapiens") {
  # Defining the formulas
  formulas <- list(
    paste("weighted_f1 ~", paste(c(factor_names, "reference:method", "method:cutoff"), collapse = " + ")),
    paste("weighted_f1 ~", paste(c(factor_names, "reference:method", "method:cutoff", "disease_state"), collapse = " + ")),
    paste("weighted_f1 ~", paste(c(factor_names, "reference:method", "method:cutoff", "disease_state", "sex"), collapse = " + "))
  )
} else if (organism == "mus_musculus") {
  formulas <- list(
    paste("weighted_f1 ~", paste(c(factor_names, "reference:method", "method:cutoff"), collapse = " + ")),
    paste("weighted_f1 ~", paste(c(factor_names, "reference:method", "method:cutoff", "treatment"), collapse = " + ")),
    paste("weighted_f1 ~", paste(c(factor_names, "reference:method", "method:cutoff", "treatment", "genotype"), collapse = " + ")),
    paste("weighted_f1 ~", paste(c(factor_names, "reference:method", "method:cutoff", "treatment", "genotype", "strain"), collapse = " + ")),
    paste("weighted_f1 ~", paste(c(factor_names, "reference:method", "method:cutoff", "treatment", "genotype", "strain","age"), collapse = " + "))
  )
}
# Grouping the data by 'key' column and creating a list of data frames
df_list <- split(weighted_f1_results, weighted_f1_results$key)

plot_model_metrics(df_list, formulas)

formula <- formulas[[1]]

for (df in df_list) {
  for (formula in formulas) {
    formula_dir <- formula %>% gsub(" ", "_", .)
    dir.create(formula_dir, showWarnings = FALSE,recursive=TRUE)
    df$method <- factor(df$method, levels=c("seurat","scvi"))
    # set baseline reference
    if (organism == "homo_sapiens") {
      df$reference <- factor(df$reference)
        all_levels <- levels(df$reference)
        df$reference <- factor(df$reference, levels = c("Human Multiple Cortical Areas SMART-seq", 
                                  all_levels[!all_levels %in% "Human Multiple Cortical Areas SMART-seq"]))

    } # need to set default levels for mmus
    if (organism == "mus_musculus") {
      df$reference <- factor(df$reference)
      all_levels <- levels(df$reference)
      ref_ref <- "An integrated transcriptomic and epigenomic atlas of mouse primary motor cortex cell types"
      df$reference <- factor(df$reference, levels = c(ref_ref, 
                                  all_levels[!all_levels %in% ref_ref]))
      df$study <- factor(df$study)
      study_levels <- levels(df$study)
      study_ref <- "GSE214244"
      df$study <- factor(df$study, levels = c(study_ref,
                                              study_levels[!study_levels %in% study_ref]))

    }

    run_and_store_model(df, formula, formula_dir = formula_dir, key = df$key[1])
  }
}
