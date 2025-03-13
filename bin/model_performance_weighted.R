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
library(DHARMa)
library(effects)
library(emmeans)
library(multcomp)
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

  random_effect_formula <- paste(formula, "+ (1 |", group_var, ")")
    # Add random effects directly to the formula
  model <- glmmTMB(as.formula(random_effect_formula), data = df, family = beta_family(link = "logit"))

  
  # Extract coefficients and p-values
  summary_df <- tidy(model) %>%
    mutate(FDR = p.adjust(p.value, method = "fdr"))
  
  # Model fit statistics
  model_stats <- tibble(
    LogLik = logLik(model),
    AIC = AIC(model),
    BIC = BIC(model)
  )
  return(list(model = model, summary = summary_df, stats = model_stats, formula = random_effect_formula))
}


run_multcomp <- function(model, contrast) {
  # Run the multcomp analysis
  contrast_var <- as.name(contrast)  # Convert string to symbol
  mc <- glht(model, linfct = mcp(method = "Tukey", interaction_average=TRUE))
  summary_mc <- summary(mc)
  return(summary_mc)
}

run_emmeans <- function(model, model_summary_coefs, key_dir) {

  terms <- model_summary_coefs$term


  # After running the model:
  emm <- emmeans(model, specs = ~ reference * method * cutoff, at = list(cutoff = 0), type = "response")
  
  
  summary_emm <- summary(emm)
  estimate <- pairs(emm)

  estimate_df <- as.data.frame(estimate)
  summary_emm_df <- as.data.frame(summary_emm) 

   # Save the emmeans summary to a file
  write.table(summary_emm_df, file = file.path(key_dir, "emmeans_summary.tsv"), sep = "\t", row.names = FALSE)
  # save emmeans estimates
  write.table(estimate_df, file = file.path(key_dir, "emmeans_estimates.tsv"), sep = "\t", row.names = FALSE)


}

plot_contrasts <- function(ae, key_dir, key) {


  for (contrast in names(ae)) {
    contrast_df = as.data.frame(ae[[contrast]])

    # Checking the number of factors (columns before 'fit')
    factor_columns <- colnames(contrast_df)[1:(which(colnames(contrast_df) == "fit") - 1)]
    num_factors <- length(factor_columns)

    # Handling different cases based on the number of factors
    if (num_factors == 1) {
      
      group_var <- contrast_df[[factor_columns[1]]]
      contrast_df$group_var <- group_var

      p2 <- ggplot(contrast_df, aes(x = group_var, 
          y = fit, ymin = 
          lower, 
          ymax = upper, group=1)) +
          geom_point(size=7) 
          
    } else if (num_factors > 1) {
      # facet should always be "method"
      # group var should be any column that isn't method
        group_var_columns <- factor_columns[factor_columns != "method"]
        contrast_df$group_var <- contrast_df[[group_var_columns[1]]]

        # If there are more than one factor, dynamically map them to different aesthetics
        p2 <- ggplot(contrast_df, aes(x = group_var, 
          y = fit, ymin = 
          lower, 
          ymax = upper, group = method, color=method)) +
          geom_point(size = 7) +
          facet_wrap(~ method) 
    } 
      p2 <- p2 + 
        theme(axis.text.x = element_text(angle = 90, hjust = 1, size = 20)) +
        labs(y = "Fitted weighted F1", title = "", x = "") +
        theme(legend.position = "right") 
      if (is.factor(contrast_df$group_var )) {
        
        p2 <- p2 + scale_x_discrete(labels = function(x) str_wrap(x, width = 20)) +
          geom_errorbar(aes(ymin = lower, ymax = upper), width = 0.2, size =2)         
      } else if (is.numeric(contrast_df$group_var)) {
        p2 <- p2 + geom_line(size=2) +
        geom_ribbon(aes(ymin = lower, ymax = upper, fill = method), alpha = 0.2) 
      }
        
      ggsave(file.path(key_dir, paste0(key, "_", contrast, "_effects.png")), p2, width = 25, height = 15, dpi = 250)
    }
}


run_drop1 <- function(model, key_dir) {
  d1 <- drop1(model, test = "Chisq")

  # Convert drop1 output to a data frame for plotting
  d1_df <- as.data.frame(d1)
  d1_df$Term <- rownames(d1_df)

  # pivot longer to plot all columns
  d1_df$logChiSquared <- log10(d1_df[["Pr(>Chi)"]])

  d1_df <- d1_df %>% 
  pivot_longer(cols = -c(Term,`Pr(>Chi)`),
         names_to = "metric", values_to = "value")

  # Plot
  p <- ggplot(d1_df, aes(x = Term, y = value, fill = metric)) +
    geom_col(position = "dodge") +  # Using geom_col for better clarity with bars
    coord_flip() +
    facet_wrap(~metric, scales = "free_x") +  # Facet by metric
    labs(x = "Term", y = "Value", title = "Drop1 Analysis: Term Metrics") +
    theme(axis.text.x = element_text(angle = 90, hjust = 1, size = 15)) +  # Rotate x-axis labels
    scale_fill_brewer(palette = "Set1")  # Color palette for better distinction


  # Save plot
  ggsave(file.path(key_dir, "drop1.png"), p, width = 20, height = 20, dpi = 300)
}

run_and_store_model <- function(df, formula, key_dir, key) {

  # Run the beta model using the run_beta_model function
  result <- run_beta_model(df, formula, group_var = "study")  # Adjust group_var as needed
  
  # Extract model summary coefficients and add additional info
  model_summary_coefs <- result$summary
  model_summary_coefs$formula <- result$formula
  model_summary_coefs$key <- key
  model_summary_coefs$LogLik <- result$stats$LogLik
  model_summary_coefs$AIC <- result$stats$AIC
  model_summary_coefs$BIC <- result$stats$BIC

  model = result$model
  run_drop1(model, key_dir) 
  run_emmeans(model, model_summary_coefs, key_dir)

  ae <- allEffects(model)
  plot_contrasts(ae, key_dir, key)
  # Save the model summary and coefficients summary to files
  write.table(model_summary_coefs, file = file.path(key_dir, paste0(key,"_model_summary_coefs_combined.tsv")), sep = "\t", row.names = FALSE)
  # Plot the model summary (assuming plot_model_summary is defined)
  plot_model_summary(model_summary = model_summary_coefs, outdir = key_dir, key = key)
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
  formula_wrapped <- unique(factor(str_wrap(formula, width = 20)))  # Adjust width as needed

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
        
  # Add error bars
  p <- p + geom_errorbar(aes(xmin = estimate - 2 * std.error, xmax = estimate + 2 * std.error), width = 0.2)

  # Save the plot to the specified directory
  ggsave(filename = file.path(outdir, paste0(key, "_", formula, "_lm_coefficients.png")), 
  plot = p, width = 35, height = max(10, nrow(model_summary) * 0.7))
}



parser <- argparse::ArgumentParser()
parser$add_argument("--weighted_f1_results", help = "Path to the weighted_f1_results file", default="/space/grp/rschwartz/rschwartz/evaluation_summary.nf/SCT_integrated_mmus/aggregated_results/weighted_f1_results.tsv")
args <- parser$parse_args()


# Reading the weighted_f1_results file
weighted_f1_results <- read.table(args$weighted_f1_results, sep="\t", header=TRUE, stringsAsFactors = TRUE)
# fill NA with none
weighted_f1_results[is.na(weighted_f1_results)] <- "None"
# Extract organism (assuming only one unique value in the 'organism' column)
organism <- unique(weighted_f1_results$organism)[1]

# Defining factor names
factor_names <- c("reference", "method", "cutoff")



if (organism == "homo_sapiens") {
  all_factors = c(factor_names, "disease_state","sex","region_match")
  # Defining the formulas
  formulas <- list(
    paste("weighted_f1 ~", paste(c(factor_names, "method:cutoff"), collapse = " + ")),
    paste("weighted_f1 ~", paste(c(factor_names, "method:cutoff","disease_state"),collapse = "+")),
    paste("weighted_f1 ~", paste(c(factor_names, "method:cutoff","sex"),collapse = "+")),
    paste("weighted_f1 ~", paste(c(factor_names, "method:cutoff","region_match"),collapse = "+")),
    paste("weighted_f1 ~", paste(c(factor_names, "method:cutoff", "reference:method"), collapse = " + ")),
    paste("weighted_f1 ~", paste(c(all_factors, "method:cutoff", "reference:method"), collapse = " + "))
    )
} else if (organism == "mus_musculus") {
    # full interactive model
  all_factors <- c(factor_names, "treatment","sex")
  formulas <- list(
    paste("weighted_f1 ~", paste(c(factor_names, "method:cutoff"), collapse = " + ")),
    paste("weighted_f1 ~", paste(c(factor_names, "method:cutoff","treatment"),collapse = " + ")),
    paste("weighted_f1 ~", paste(c(factor_names, "method:cutoff","sex"),collapse = " + ")),
    paste("weighted_f1 ~", paste(c(factor_names, "method:cutoff", "reference:method"), collapse = " + ")),
    paste("weighted_f1 ~", paste(c(all_factors, "method:cutoff", "reference:method"), collapse = " + "))

    
  )
}

weighted_f1_results$weighted_f1 <-  pmax(pmin(weighted_f1_results$weighted_f1, 1 - 1e-6), 1e-6)


# Grouping the data by 'key' column and creating a list of data frames
df_list <- split(weighted_f1_results, weighted_f1_results$key)

plot_model_metrics(df_list, formulas)



for (df in df_list) {
  for (formula in formulas) {
    formula_dir <- formula %>% gsub(" ", "_", .)

    dir.create(formula_dir, showWarnings = FALSE,recursive=TRUE)
    df$method <- factor(df$method, levels=c("seurat","scvi"))
    
    # set baseline
    if (organism == "homo_sapiens") {

        df$reference <- factor(df$reference)
        # use relevel instead of this
        ref_ref = "Dissection Dorsolateral prefrontal cortex DFC"
        df$reference <- relevel(df$reference, ref = ref_ref)
         } # need to set default levels for mmus
    
    if (organism == "mus_musculus") {

      df$reference <- factor(df$reference)
      ref_ref <- "Single-cell RNA-seq for all cortical  hippocampal regions SMART-Seq v4"
      df$reference <- relevel(df$reference, ref = ref_ref)

      df$study <- factor(df$study)
      study_ref <- "GSE152715.2"
      df$study <- relevel(df$study, ref=study_ref)


    }
    key = df$key[1]
    key_dir = file.path(formula_dir, key)
    # make dir
    dir.create(key_dir, showWarnings = FALSE,recursive=TRUE)

    run_and_store_model(df, formula, key_dir = key_dir, key = key)
  }
}
