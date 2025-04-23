#library(multcomp)
`%||%` <- function(a, b) if (!is.null(a)) a else b

# Set global theme for background
theme_set(
  theme_minimal(base_size =30 ) +  # Base theme
    theme(
      plot.background = element_rect(fill = "white", color = NA),  # Plot background color
      panel.background = element_rect(fill = "white", color = NA), # Panel background color
      legend.background = element_rect(fill = "white", color = NA) # Legend background color
    )
)

plot_qq <- function(model, key_dir) {
  # Create a QQ plot using the DHARMa package
  qq_residuals <- simulateResiduals(model, plot = FALSE)
  
  png(file.path(key_dir, "qq_plot.png"), width = 800, height = 800)
  plot(qq_residuals, quantreg = TRUE)
  dev.off()

  png(file.path(key_dir, "dispersion_plot.png"), width = 800, height = 800)
  testDispersion(qq_residuals,plot = TRUE)
  dev.off()

}


run_beta_model <- function(df, formula, group_var = "study") {
  nt <- min(parallel::detectCores(),5)

  # Ensure the outcome is within (0,1) for Beta regression
  outcome_var <- all.vars(as.formula(formula))[1]

  random_effect_formula <- paste(formula, "+ (1 |", group_var, ")")
    # Add random effects directly to the formula
  model <- glmmTMB(as.formula(random_effect_formula), data = df, family = beta_family(link = "logit"),
              control=glmmTMBControl(parallel = nt))

  
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


run_emmeans_weighted <- function(model, key_dir) {

  fig.dir <- file.path(key_dir, "figures")
  if (!dir.exists(fig.dir)) {
    dir.create(fig.dir)
  }
  file.dir <- file.path(key_dir, "files")
  if (!dir.exists(file.dir)) {
    dir.create(file.dir)
  }
  # Estimate for reference * method * cutoff
  emm_reference_method <- emmeans(model, specs = ~ reference * method, at = list(cutoff = 0), type = "response")
  summary_emm_reference_method_df <- as.data.frame(summary(emm_reference_method))
  estimate_reference_method_df <- as.data.frame(pairs(emm_reference_method))
  #plot_contrasts(summary_emm_reference_method_df, key_dir=fig.dir, contrast ="reference:method")
  # Save emmeans summary and estimates for reference * method * cutoff
  write.table(summary_emm_reference_method_df, file = file.path(file.dir, "reference_method_emmeans_summary.tsv"), sep = "\t", row.names = FALSE)
  write.table(estimate_reference_method_df, file = file.path(file.dir, "reference_method_emmeans_estimates.tsv"), sep = "\t", row.names = FALSE)

  # Estimate for method * cutoff
  emm_method <- emmeans(model, specs = ~ method, at = list(cutoff = 0), type = "response")
  summary_emm_method_df <- as.data.frame(summary(emm_method))
  estimate_method_df <- as.data.frame(pairs(emm_method))
  # Save emmeans summary and estimates for method * cutoff
  write.table(summary_emm_method_df, file = file.path(file.dir, "method_emmeans_summary.tsv"), sep = "\t", row.names = FALSE)
  write.table(estimate_method_df, file = file.path(file.dir, "method_emmeans_estimates.tsv"), sep = "\t", row.names = FALSE)
  #plot_contrasts(summary_emm_method_df, key_dir=fig.dir, contrast ="method")

# subsample ref 
  emm_subsample_ref <- emmeans(model, specs = ~ subsample_ref, at = list(cutoff = 0), type = "response")
  summary_emm_subsample_ref_df <- as.data.frame(summary(emm_subsample_ref))
  estimate_subsample_ref_df <- as.data.frame(pairs(emm_subsample_ref))
  #plot_contrasts(summary_emm_subsample_ref_df, key_dir=fig.dir, contrast ="subsample_ref")
  # Save emmeans summary and estimates for subsample_ref
  write.table(summary_emm_subsample_ref_df, file = file.path(file.dir, "subsample_ref_emmeans_summary.tsv"), sep = "\t", row.names = FALSE)
  write.table(estimate_subsample_ref_df, file = file.path(file.dir, "subsample_ref_emmeans_estimates.tsv"), sep = "\t", row.names = FALSE)


  if ("sex" %in% colnames(model$frame) ) {
    emm_sex <- emmeans(model, specs = ~ sex, at = list(cutoff = 0), type = "response")
    summary_emm_sex_df <- as.data.frame(summary(emm_sex))
    estimate_sex_df <- as.data.frame(pairs(emm_sex))
    #plot_contrasts(summary_emm_sex_df, key_dir=fig.dir, contrast="sex")
    # Save emmeans summary and estimates for sex
    write.table(summary_emm_sex_df, file = file.path(file.dir, "sex_emmeans_summary.tsv"), sep = "\t", row.names = FALSE)
    write.table(estimate_sex_df, file = file.path(file.dir, "sex_emmeans_estimates.tsv"), sep = "\t", row.names = FALSE)
  }

  if ("disease_state" %in% colnames(model$frame) ) {
    emm_disease_state <- emmeans(model, specs = ~ disease_state, at = list(cutoff = 0), type = "response")
    summary_emm_disease_state_df <- as.data.frame(summary(emm_disease_state))
    estimate_disease_state_df <- as.data.frame(pairs(emm_disease_state))
    #plot_contrasts(summary_emm_disease_state_df, key_dir=fig.dir, contrast="disease_state")
    write.table(summary_emm_disease_state_df, file = file.path(file.dir, "disease_state_emmeans_summary.tsv"), sep = "\t", row.names = FALSE)
    write.table(estimate_disease_state_df, file = file.path(file.dir, "disease_state_emmeans_estimates.tsv"), sep = "\t", row.names = FALSE)
  }

  if ("treatment_state" %in% colnames(model$frame) ) {
    emm_treatment <- emmeans(model, specs = ~ treatment_state, at = list(cutoff = 0), type = "response")
    summary_emm_treatment_df <- as.data.frame(summary(emm_treatment))
    estimate_treatment_df <- as.data.frame(pairs(emm_treatment))
    #plot_contrasts(summary_emm_treatment_df, key_dir=fig.dir, contrast="treatment_state")
    write.table(summary_emm_treatment_df, file = file.path(file.dir, "treatment_emmeans_summary.tsv"), sep = "\t", row.names = FALSE)
    write.table(estimate_treatment_df, file = file.path(file.dir, "treatment_emmeans_estimates.tsv"), sep = "\t", row.names = FALSE)
    
  }

  if ("region_match" %in% colnames(model$frame)) {
    emm_region_match <- emmeans(model, specs = ~ region_match , at = list(cutoff = 0), type = "response")
    summary_emm_region_match_df <- as.data.frame(summary(emm_region_match))
    estimate_region_match_df <- as.data.frame(pairs(emm_region_match))
    #plot_contrasts(summary_emm_region_match_df, key_dir=fig.dir, contrast="region_match")
    write.table(summary_emm_region_match_df, file = file.path(file.dir, "region_match_emmeans_summary.tsv"), sep = "\t", row.names = FALSE)
    write.table(estimate_region_match_df, file = file.path(file.dir, "region_match_emmeans_estimates.tsv"), sep = "\t", row.names = FALSE)
  
  }

  # get marginal mean across all contrasts
  emm_summary_df <- as.data.frame(summary(emmeans(model, type = "response", specs=~1, at = list(cutoff = 0))))
  write.table(emm_summary_df, file = file.path(file.dir, "summary_emmeans.tsv"), sep = "\t", row.names = FALSE)
}


run_emmeans_label <- function(model, key_dir) {

  fig.dir <- file.path(key_dir, "figures")
  if (!dir.exists(fig.dir)) {
    dir.create(fig.dir)
  }
  file.dir <- file.path(key_dir, "files")
  if (!dir.exists(file.dir)) {
    dir.create(file.dir)
  }
  emm_label <- emmeans(model, specs = ~ label, at = list(cutoff = 0), type = "response")
  summary_emm_label_df <- as.data.frame(summary(emm_label))
  estimate_label_df <- as.data.frame(pairs(emm_label))
  plot_contrasts(summary_emm_label_df, key_dir=fig.dir, contrast="label")
}



plot_contrasts <- function(emm_summary_df, key_dir, contrast) {
  
  factors = colnames(emm_summary_df)[1:(which(colnames(emm_summary_df) == "response") - 1)]
  emm_summary_df$factor <- emm_summary_df[[1]]

  if ("method" %in% factors) {

    # check if method is the only factor
    # Plot for summary effects
    p1 <- ggplot(emm_summary_df, aes(x = factor, y = response, 
                                    ymin = asymp.LCL, ymax = asymp.UCL, 
                                    group = method, color = method)) +
      geom_point(size = 7) +
      geom_errorbar(width = 0.2, size = 2) +
      theme() +
      theme(axis.text.x = element_text(angle = 90, hjust = 1, size = 20)) +
      labs(y = "Estimate", title = paste0("Marginal Means for ", contrast), x = "") +
      theme(legend.position = "right") +
      scale_x_discrete(labels = function(x) str_wrap(x, width = 20))

      if (length(factors) > 1) {
        p1 <- p1 + facet_wrap(~ method)
      }

  } else {

    # do the same thing without method
   p1 <- ggplot(emm_summary_df, aes(x = factor, y = response, 
                                   ymin = asymp.LCL, ymax = asymp.UCL)) +
    geom_point(size = 7) +
    geom_errorbar(width = 0.2, size = 2) +
    theme() +
    theme(axis.text.x = element_text(angle = 90, hjust = 1, size = 20)) +
    labs(y = "Estimate", title = paste0("Marginal Means for ", contrast), x = "") +
    theme(legend.position = "right") +
    scale_x_discrete(labels = function(x) str_wrap(x, width = 20)) 
 
  }

  # Save the summary plot
  ggsave(file.path(key_dir, paste0(contrast, "_summary_effects.png")), p1, width = 25, height = 15, dpi = 250)
}


plot_continuous_effects <- function(ae_contrast, key_dir) {
# only deals with method:cutoff for now
  contrast = names(ae_contrast)[1]
  contrast_df = as.data.frame(ae_contrast)[[1]]
  factor_columns <- colnames(contrast_df)[1:(which(colnames(contrast_df) == "fit") - 1)]
  num_factors <- length(factor_columns)
  group_var_columns <- factor_columns[factor_columns != "method"]
  contrast_df$group_var <- contrast_df[[group_var_columns[1]]]

      p2 <- ggplot(contrast_df, aes(x = group_var, 
        y = fit, ymin = 
        lower, 
        ymax = upper, group = method, color=method)) +
        geom_point(size = 7) +
        facet_wrap(~ method) +
      theme(legend.position = "right") +     
      theme(axis.text.x = element_text(angle = 90, hjust = 1, size = 20),
            panel.grid.major = element_blank(),
            panel.grid.minor = element_blank()) +
      labs(y = "F1", title = "", x = colnames(contrast_df)[1]) +
      geom_line(size=2) +
      geom_ribbon(aes(ymin = lower, ymax = upper, fill = method), alpha = 0.2)    
      
    ggsave(file.path(key_dir, paste0(key, "_", contrast, "_effects.png")), p2, width = 25, height = 15, dpi = 250)
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

run_and_store_model <- function(df, formula, key_dir, key, type="weighted", group_var="study") {
  fig.dir <- file.path(key_dir, "figures")
  if (!dir.exists(fig.dir)) {
    dir.create(fig.dir)
  }
  file.dir <- file.path(key_dir, "files")
  if (!dir.exists(file.dir)) {
    dir.create(file.dir)
  }
  # Run the beta model using the run_beta_model function
  result <- run_beta_model(df, formula, group_var = group_var)  # Adjust group_var as needed
  
  # Extract model summary coefficients and add additional info
  model_summary_coefs <- result$summary
  model_summary_coefs$formula <- result$formula
  model_summary_coefs$key <- key
  model_summary_coefs$LogLik <- result$stats$LogLik
  model_summary_coefs$AIC <- result$stats$AIC
  model_summary_coefs$BIC <- result$stats$BIC
  model = result$model

  run_drop1(model, fig.dir) 
  plot_qq(model, fig.dir)
  if (type == "weighted") {
    # Run emmeans for weighted F1
    run_emmeans_weighted(model, key_dir)
    alleffects <- allEffects(model, xlevels = list(cutoff = c(0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.5, 0.75)))

    ae_contrast <- alleffects["method:cutoff"]

  } else if (type == "label") {
    # Run emmeans for label F1
    run_emmeans_label(model, key_dir)
    alleffects <- allEffects(model, xlevels = list(cutoff = c(0, 0.05)))
 
    ae_contrast <- alleffects["support:method"]
    plot_continuous_effects(ae_contrast, fig.dir)
    ae_support <- as.data.frame(ae_contrast[[1]])
    write.table(ae_support, file = file.path(file.dir, "label_support_effects.tsv"), sep = "\t", row.names = FALSE)
    alleffects <- allEffects(model, xlevels = list(cutoff = c(0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.5, 0.75)))
    ae_contrast <- alleffects["cutoff:method"]
  }


  plot_continuous_effects(ae_contrast, fig.dir)
  ae_contrast <- as.data.frame(ae_contrast[[1]])
  write.table(ae_contrast, file = file.path(file.dir, "method_cutoff_effects.tsv"), sep = "\t", row.names = FALSE)


  # Save the model summary and coefficients summary to files
  write.table(model_summary_coefs, file = file.path(file.dir, paste0(key,"_model_summary_coefs_combined.tsv")), sep = "\t", row.names = FALSE)
  # Plot the model summary (assuming plot_model_summary is defined)
  plot_model_summary(model_summary = model_summary_coefs, outdir = fig.dir, key = key)
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

