#library(multcomp)
library(patchwork)
`%||%` <- function(a, b) if (!is.null(a)) a else b

# Set global theme for background
theme <- theme_set(
  theme_bw(base_size = 45) +  # Base theme
    theme(
      plot.background = element_rect(fill = "white", color = NA),  # Plot background color
      panel.background = element_rect(fill = "white", color = NA), # Panel background color
      legend.background = element_rect(fill = "white", color = NA) # Legend background color

    )
)

plot_qq <- function(model, key_dir) {
  # Create a QQ plot using the DHARua package
  qq_residuals <- simulateResiduals(model, plot = FALSE)
  
  png(file.path(key_dir, "qq_plot.png"), width = 800, height = 800)
  plot(qq_residuals, quantreg = TRUE)
  dev.off()

  png(file.path(key_dir, "dispersion_plot.png"), width = 800, height = 800)
  testDispersion(qq_residuals,plot = TRUE)
  dev.off()

}


run_beta_model <- function(df, formula, group_var = "study", type="weighted", mixed=TRUE) {
  outcome_var <- all.vars(as.formula(formula))[1]

  if (mixed) {
    # add study as a random effect (intercept may vary)
    model_formula <- paste(formula, "+ (1 |", group_var, ")")
    nt <- min(parallel::detectCores(), 10)
  } else {
    # fixed-effects only beta regression (no random effect)
    model_formula <- formula
    nt <- 1L
  }

  model <- glmmTMB(as.formula(model_formula), data = df, family = beta_family(link = "logit"),
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
  return(list(model = model, summary = summary_df, stats = model_stats, formula = model_formula))
}


run_emmeans_weighted <- function(model, key) {
  results <- list()

  # Estimate for reference * method
  emm_reference_method <- emmeans(model, specs = ~ reference * method, at = list(cutoff = 0, subsample_ref = "500"), type = "response")
  results$reference_method_emmeans_summary <- as.data.frame(summary(emm_reference_method)) %>% mutate(key = key)
  results$reference_method_emmeans_estimates <- as.data.frame(pairs(emm_reference_method)) %>% mutate(key = key)

  # Estimate for method
  emm_method <- emmeans(model, specs = ~ method, at = list(cutoff = 0, subsample_ref="500", reference="whole cortex"), type = "response")
  results$method_emmeans_summary <- as.data.frame(summary(emm_method)) %>% mutate(key = key)
  results$method_emmeans_estimates <- as.data.frame(pairs(emm_method)) %>% mutate(key = key)

  # subsample ref
  emm_subsample_ref <- emmeans(model, specs = ~ subsample_ref, at = list(cutoff = 0, reference="whole cortex"), type = "response")
  results$subsample_ref_emmeans_summary <- as.data.frame(summary(emm_subsample_ref)) %>% mutate(key = key)
  results$subsample_ref_emmeans_estimates <- as.data.frame(pairs(emm_subsample_ref)) %>% mutate(key = key)

  if ("sex" %in% colnames(model$frame)) {
    emm_sex <- emmeans(model, specs = ~ sex, at = list(cutoff = 0, subsample_ref="500", reference="whole cortex"), type = "response")
    results$sex_emmeans_summary <- as.data.frame(summary(emm_sex)) %>% mutate(key = key)
    results$sex_emmeans_estimates <- as.data.frame(pairs(emm_sex)) %>% mutate(key = key)
  }

  if ("disease_state" %in% colnames(model$frame)) {
    emm_disease_state <- emmeans(model, specs = ~ disease_state, at = list(cutoff = 0, subsample_ref="500", reference="whole cortex"), type = "response")
    results$disease_state_emmeans_summary <- as.data.frame(summary(emm_disease_state)) %>% mutate(key = key)
    results$disease_state_emmeans_estimates <- as.data.frame(pairs(emm_disease_state)) %>% mutate(key = key)
  }

  if ("treatment_state" %in% colnames(model$frame)) {
    emm_treatment <- emmeans(model, specs = ~ treatment_state, at = list(cutoff = 0, subsample_ref="500", reference="whole cortex"), type = "response")
    results$treatment_emmeans_summary <- as.data.frame(summary(emm_treatment)) %>% mutate(key = key)
    results$treatment_emmeans_estimates <- as.data.frame(pairs(emm_treatment)) %>% mutate(key = key)
  }

  if ("region_match" %in% colnames(model$frame)) {
    emm_region_match <- emmeans(model, specs = ~ region_match, at = list(cutoff = 0), type = "response")
    results$region_match_emmeans_summary <- as.data.frame(summary(emm_region_match)) %>% mutate(key = key)
    results$region_match_emmeans_estimates <- as.data.frame(pairs(emm_region_match)) %>% mutate(key = key)
  }

  # get marginal mean across all contrasts
  results$summary_emmeans <- as.data.frame(summary(emmeans(model, type = "response", specs=~1, at = list(cutoff = 0)))) %>% mutate(key = key)

  return(results)
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
      theme + 
      theme(axis.text.x = element_text(angle = 90, hjust = 1, size=45)) +
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
    theme +
    theme(axis.text.x = element_text(angle = 90, hjust = 1, size=45)) +
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
      theme +
      theme(legend.position = "right") +     
      theme(axis.text.x = element_text(angle = 90, hjust = 1, size=45)) +
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
    scale_fill_brewer(palette = "Set1") +  # Color palette for better distinction
    theme

  # Save plot
  ggsave(file.path(key_dir, "drop1.png"), p, width = 20, height = 20, dpi = 300)
}

run_and_store_model <- function(df, formula, fig_dir, key, type="label", group_var="study", mixed=TRUE) {
  all_results <- list()

  if (!dir.exists(fig_dir)) {
    dir.create(fig_dir, recursive = TRUE)
  }

  # Run the beta model
  result <- run_beta_model(df, formula, group_var = group_var, type=type, mixed=mixed)

  # Extract model summary coefficients and add additional info
  model_summary_coefs <- result$summary
  model_summary_coefs$formula <- result$formula
  model_summary_coefs$key <- key
  model_summary_coefs$LogLik <- as.numeric(result$stats$LogLik)
  model_summary_coefs$AIC <- as.numeric(result$stats$AIC)
  model_summary_coefs$BIC <- as.numeric(result$stats$BIC)
  model <- result$model

  all_results$model_coefs <- model_summary_coefs

  plot_qq(model, fig_dir)

  if (type == "weighted") {
    emmeans_results <- run_emmeans_weighted(model, key)
    all_results <- c(all_results, emmeans_results)

    alleffects <- allEffects(model, xlevels = list(cutoff = c(0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.5, 0.75)))
    ae_contrast <- alleffects["method:cutoff"]
    plot_continuous_effects(ae_contrast, fig_dir)
    all_results$method_cutoff_effects <- as.data.frame(ae_contrast[[1]]) %>% mutate(key = key)

  }

  plot_model_summary(model_summary = model_summary_coefs, outdir = fig_dir, key = key)

  return(all_results)
}



# Function to plot model metrics
plot_model_metrics <- function(model, formula, key) {
  # Initialize an empty list to store results
      stats <- model$stats
      # Add key column for each unique 'key' in df
        results_df <- stats %>% 
          mutate(key = key) %>% 
          mutate(formula = model$formula)
  
  # Combine the results into a single data frame
  #results_df <- do.call(rbind, results)
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
    theme +
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
    
  outdir = key
  dir.create(outdir, recursive = TRUE)
  # Save the plot
  ggsave(file.path(ourdir,"model_metrics.png"), p, width = 20, height = 15, dpi = 250)
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

