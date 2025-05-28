# hierarchical_functions.R
# Helper functions for the hierarchical model analysis

library(tidyverse)
library(cmdstanr)
library(posterior)
library(bayesplot)

# Function to run the complete hierarchical analysis
run_hierarchical_analysis <- function(
  data_file = "tables/hierarchical_data.csv",
  stan_file = "hierarchical_delayed_tau.stan",
  chains = 4,
  iter_sampling = 2000,
  iter_warmup = 1000,
  adapt_delta = 0.95,
  max_treedepth = 12,
  output_dir = "hierarchical_results"
) {

  # Create output directory
  dir.create(output_dir, showWarnings = FALSE, recursive = TRUE)

  # Load data
  message("Loading data...")
  galaxy_data <- read_csv(data_file, show_col_types = FALSE)
  stan_data <- jsonlite::fromJSON("tables/stan_data.json")
  group_info <- read_csv("tables/hierarchical_groups.csv", show_col_types = FALSE)

  # Compile model
  message("Compiling Stan model...")
  model <- cmdstan_model(stan_file)

  # Initialize
  init_fun <- function() {
    list(
      mu_t_sf_pop = 12,
      sigma_t_sf_pop = 1,
      mu_tau_pop = 4,
      sigma_tau_pop = 1,
      mu_zeta_pop = 1.3,
      sigma_zeta_pop = 0.05,

      mu_t_sf_group = rep(12, stan_data$N_groups),
      sigma_t_sf_group = rep(0.5, stan_data$N_groups),

      mu_tau_group = rep(4, stan_data$N_groups),
      sigma_tau_group = rep(0.5, stan_data$N_groups),

      mu_zeta_group = rep(1.3, stan_data$N_groups),
      sigma_zeta_group = rep(0.01, stan_data$N_groups),

      t_sf_raw = rnorm(stan_data$N, 0, 0.1),
      tau_raw = rnorm(stan_data$N, 0, 0.1),
      zeta_raw = rnorm(stan_data$N, 0, 0.1),

      sigma_obs = 0.5
    )
  }

  # Fit model
  message("Fitting hierarchical model...")
  message(sprintf("Running %d chains with %d warmup and %d sampling iterations each",
                  chains, iter_warmup, iter_sampling))

  fit <- model$sample(
    data = stan_data,
    chains = chains,
    parallel_chains = chains,
    iter_sampling = iter_sampling,
    iter_warmup = iter_warmup,
    #init = init_fun,
    refresh = 100,
    adapt_delta = adapt_delta,
    max_treedepth = max_treedepth
  )

  # Save fit object
  fit$save_object(file.path(output_dir, "hierarchical_fit.rds"))

  # Process results
  message("Processing results...")
  process_hierarchical_results(fit, galaxy_data, group_info, stan_data, output_dir)

  # Create diagnostics
  message("Creating diagnostic plots...")
  create_hierarchical_diagnostics(fit, output_dir)

  message("Analysis complete!")
  return(fit)
}

# Process and save results
process_hierarchical_results <- function(fit, galaxy_data, group_info, stan_data, output_dir) {

  # Galaxy-level parameters
  galaxy_params <- c("t_sf", "tau", "zeta", "x", "A", "logSFR_today_pred")

  # Extract summaries for each parameter
  galaxy_summaries <- map_dfr(galaxy_params, function(param) {
    fit$summary(variables = param) %>%
      mutate(
        ID = rep(stan_data$ID, each = 1),
        parameter = param
      ) %>%
      filter(variable == paste0(param, "[", row_number(), "]"))
  })

  # Pivot to wide format
  galaxy_wide <- galaxy_summaries %>%
    select(ID, parameter, mean, sd, q5, q95, rhat, ess_bulk) %>%
    pivot_wider(
      names_from = parameter,
      values_from = c(mean, sd, q5, q95, rhat, ess_bulk),
      names_sep = "_"
    )

  # Merge with original data
  galaxy_results <- galaxy_data %>%
    inner_join(galaxy_wide, by = "ID")

  write_csv(galaxy_results, file.path(output_dir, "galaxy_parameters.csv"))

  # Group-level parameters
  group_params <- c("mu_t_sf_group", "mu_tau_group", "mu_zeta_group",
                   "sigma_t_sf_group", "sigma_tau_group", "sigma_zeta_group",
                   "mean_x_group", "mean_sSFR_group")

  group_summaries <- map_dfr(group_params, function(param) {
    fit$summary(variables = param) %>%
      filter(str_detect(variable, paste0("^", param, "\\["))) %>%
      mutate(
        group_id = as.integer(str_extract(variable, "\\d+"))
      )
  })

  # Pivot group results
  group_wide <- group_summaries %>%
    mutate(param_name = str_remove(variable, "\\[\\d+\\]")) %>%
    select(group_id, param_name, mean, sd, q5, q95, rhat, ess_bulk) %>%
    pivot_wider(
      names_from = param_name,
      values_from = c(mean, sd, q5, q95, rhat, ess_bulk),
      names_sep = "_"
    )

  # Merge with group info
  group_results <- group_info %>%
    left_join(group_wide, by = "group_id")

  write_csv(group_results, file.path(output_dir, "group_parameters.csv"))

  # Population parameters
  pop_params <- c("mu_t_sf_pop", "sigma_t_sf_pop",
                 "mu_tau_pop", "sigma_tau_pop",
                 "mu_zeta_pop", "sigma_zeta_pop", "sigma_obs")

  pop_summary <- fit$summary(variables = pop_params)
  write_csv(pop_summary, file.path(output_dir, "population_parameters.csv"))
}

# Create diagnostic plots
create_hierarchical_diagnostics <- function(fit, output_dir) {

  # Save diagnostic summary
  diag_summary <- fit$diagnostic_summary()
  capture.output(diag_summary, file = file.path(output_dir, "diagnostic_summary.txt"))

  # Create plots
  pdf(file.path(output_dir, "diagnostic_plots.pdf"), width = 10, height = 8)

  # Population parameter traces
  pop_params <- c("mu_t_sf_pop", "mu_tau_pop", "mu_zeta_pop", "sigma_obs")
  pop_draws <- fit$draws(variables = pop_params)

  print(mcmc_trace(pop_draws) +
        ggtitle("Population Parameter Traces"))

  print(mcmc_rank_overlay(pop_draws) +
        ggtitle("Rank Plots"))

  # Energy diagnostic
  np <- nuts_params(fit)
  print(mcmc_nuts_energy(np) +
        ggtitle("Energy Diagnostic"))

  # Sample group parameters
  group_draws <- fit$draws(variables = c("mu_t_sf_group[1]", "mu_tau_group[1]",
                                         "mu_t_sf_group[2]", "mu_tau_group[2]"))
  print(mcmc_pairs(group_draws) +
        ggtitle("Group Parameter Correlations (Groups 1-2)"))

  dev.off()

  message("Diagnostic plots saved to ", file.path(output_dir, "diagnostic_plots.pdf"))
}

# Quick diagnostic check
check_convergence <- function(fit) {
  # Get all rhats
  all_summary <- fit$summary()

  # Check rhats
  bad_rhat <- all_summary %>%
    filter(rhat > 1.01) %>%
    arrange(desc(rhat))

  if (nrow(bad_rhat) > 0) {
    message("Warning: ", nrow(bad_rhat), " parameters have Rhat > 1.01")
    message("Worst Rhat values:")
    print(head(bad_rhat))
  } else {
    message("All Rhat values < 1.01 ✓")
  }

  # Check ESS
  bad_ess <- all_summary %>%
    filter(ess_bulk < 400) %>%
    arrange(ess_bulk)

  if (nrow(bad_ess) > 0) {
    message("Warning: ", nrow(bad_ess), " parameters have ESS < 400")
    message("Worst ESS values:")
    print(head(bad_ess))
  } else {
    message("All ESS values > 400 ✓")
  }

  # Check divergences
  sampler_diag <- fit$sampler_diagnostics()
  divergences <- sum(sampler_diag[,,"divergent__"])

  if (divergences > 0) {
    message("Warning: ", divergences, " divergent transitions detected")
  } else {
    message("No divergent transitions ✓")
  }

  return(list(
    bad_rhat = bad_rhat,
    bad_ess = bad_ess,
    n_divergences = divergences
  ))
}
