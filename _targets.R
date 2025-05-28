# _targets_hierarchical.R: Hierarchical model pipeline

library(targets)
library(tarchetypes)

# Source helper functions
tar_source()

# MCMC settings
chains <- 4
parallel_chains <- 4
threads_chain <- 2
sample_size <- 2000  # Reduced for hierarchical model
warmup <- 1000

options(tidyverse.quiet = TRUE)
tar_option_set(
  packages = c(
    "readr", "dplyr", "tidyr", "tibble", "stringr", "here",
    "cmdstanr", "posterior", "bayesplot", "ggplot2", "jsonlite"
  ),
  garbage_collection = TRUE,
  format = "qs"
)

list(
  # Load prepared data ------------------------------------------------------
  tar_target(
    hierarchical_data,
    {
      # Load the prepared data from Python
      data <- read_csv("tables/hierarchical_data.csv", show_col_types = FALSE)
      stan_data <- fromJSON("tables/stan_data.json")
      group_info <- read_csv("tables/hierarchical_groups.csv", show_col_types = FALSE)

      list(
        data = data,
        stan_data = stan_data,
        group_info = group_info
      )
    }
  ),

  # Compile Stan model ------------------------------------------------
  tar_target(
    hierarchical_model,
    cmdstan_model("hierarchical_delayed_tau.stan")
  ),

  # Fit hierarchical model --------------------------------------------
  tar_target(
    hierarchical_fit,
    {
      # Initialize values for better convergence
      init_fun <- function() {
        list(
          mu_t_sf_pop = 10,
          sigma_t_sf_pop = 2,
          mu_tau_pop = 4,
          sigma_tau_pop = 1,
          mu_zeta_pop = 1.3,
          sigma_zeta_pop = 0.05,

          mu_t_sf_group = rep(10, hierarchical_data$stan_data$N_groups),
          sigma_t_sf_group = rep(0.5, hierarchical_data$stan_data$N_groups),

          mu_tau_group = rep(4, hierarchical_data$stan_data$N_groups),
          sigma_tau_group = rep(0.5, hierarchical_data$stan_data$N_groups),

          mu_zeta_group = rep(1.3, hierarchical_data$stan_data$N_groups),
          sigma_zeta_group = rep(0.01, hierarchical_data$stan_data$N_groups),

          t_sf_raw = rnorm(hierarchical_data$stan_data$N, 0, 0.1),
          tau_raw = rnorm(hierarchical_data$stan_data$N, 0, 0.1),
          zeta_raw = rnorm(hierarchical_data$stan_data$N, 0, 0.1),

          sigma_obs = 0.1
        )
      }

      # Fit the model
      fit <- hierarchical_model$sample(
        data = hierarchical_data$stan_data,
        chains = chains,
        parallel_chains = parallel_chains,
        threads_per_chain = threads_chain,
        iter_sampling = sample_size,
        iter_warmup = warmup,
        init = init_fun,
        refresh = 100,
        max_treedepth = 12,
        adapt_delta = 0.95  # Higher for hierarchical models
      )

      fit
    }
  ),

  # Extract and save results ------------------------------------------
  tar_target(
    hierarchical_results,
    {
      # Create output directory
      output_dir <- here::here("hierarchical_results")
      dir.create(output_dir, showWarnings = FALSE, recursive = TRUE)

      # Extract galaxy-level parameters
      galaxy_params <- c("t_sf", "tau", "zeta", "x", "A", "logSFR_today_pred")
      galaxy_summary <- hierarchical_fit$summary(variables = galaxy_params)

      # Add galaxy IDs and merge with original data
      galaxy_summary$ID <- rep(hierarchical_data$stan_data$ID,
                               each = length(galaxy_params))
      galaxy_summary$parameter <- rep(galaxy_params,
                                      times = hierarchical_data$stan_data$N)

      # Reshape to wide format
      galaxy_wide <- galaxy_summary %>%
        select(ID, parameter, mean, sd, q5, q95, rhat, ess_bulk) %>%
        pivot_wider(
          names_from = parameter,
          values_from = c(mean, sd, q5, q95, rhat, ess_bulk),
          names_sep = "_"
        )

      # Merge with original data
      galaxy_results <- hierarchical_data$data %>%
        left_join(galaxy_wide, by = c("ID" = "ID"))

      write_csv(galaxy_results, file.path(output_dir, "galaxy_parameters.csv"))

      # Extract group-level parameters
      group_params <- c("mu_t_sf_group", "mu_tau_group", "mu_zeta_group",
                        "sigma_t_sf_group", "sigma_tau_group", "sigma_zeta_group",
                        "mean_x_group", "mean_sSFR_group")

      group_summary <- hierarchical_fit$summary(variables = group_params)

      # Reshape group results
      group_wide <- group_summary %>%
        mutate(
          group_id = as.integer(str_extract(variable, "\\[(\\d+)\\]") %>%
                                  str_extract("\\d+")),
          param_name = str_remove(variable, "\\[\\d+\\]")
        ) %>%
        select(group_id, param_name, mean, sd, q5, q95, rhat, ess_bulk) %>%
        pivot_wider(
          names_from = param_name,
          values_from = c(mean, sd, q5, q95, rhat, ess_bulk),
          names_sep = "_"
        )

      # Merge with group info
      group_results <- hierarchical_data$group_info %>%
        left_join(group_wide, by = "group_id")

      write_csv(group_results, file.path(output_dir, "group_parameters.csv"))

      # Extract population-level parameters
      pop_params <- c("mu_t_sf_pop", "sigma_t_sf_pop",
                      "mu_tau_pop", "sigma_tau_pop",
                      "mu_zeta_pop", "sigma_zeta_pop", "sigma_obs")

      pop_summary <- hierarchical_fit$summary(variables = pop_params)
      write_csv(pop_summary, file.path(output_dir, "population_parameters.csv"))

      # Save full posterior draws for selected parameters
      draws <- hierarchical_fit$draws(variables = c(pop_params, "lp__"))
      write_rds(draws, file.path(output_dir, "posterior_draws.rds"))

      list(
        galaxy_results = galaxy_results,
        group_results = group_results,
        pop_summary = pop_summary,
        output_dir = output_dir
      )
    }
  ),

  # Diagnostics -------------------------------------------------------
  tar_target(
    hierarchical_diagnostics,
    {
      output_dir <- hierarchical_results$output_dir

      # MCMC diagnostics
      diag_summary <- hierarchical_fit$diagnostic_summary()
      capture.output(diag_summary,
                     file = file.path(output_dir, "diagnostic_summary.txt"))

      # Sampler diagnostics
      sampler_diag <- hierarchical_fit$sampler_diagnostics()
      write_csv(as.data.frame(sampler_diag),
                file.path(output_dir, "sampler_diagnostics.csv"))

      # Create diagnostic plots
      pdf(file.path(output_dir, "diagnostic_plots.pdf"), width = 10, height = 8)

      # Trace plots for population parameters
      pop_draws <- hierarchical_fit$draws(
        variables = c("mu_t_sf_pop", "mu_tau_pop", "mu_zeta_pop", "sigma_obs")
      )
      print(mcmc_trace(pop_draws))

      # Rank plots
      print(mcmc_rank_overlay(pop_draws))

      # Energy diagnostic
      print(mcmc_nuts_energy(sampler_diag))

      dev.off()

      file.path(output_dir, "diagnostic_plots.pdf")
    },
    format = "file"
  ),

  # Create analysis plots ---------------------------------------------
  tar_target(
    hierarchical_plots,
    {
      output_dir <- hierarchical_results$output_dir

      # Load results
      galaxy_res <- hierarchical_results$galaxy_results
      group_res <- hierarchical_results$group_results

      # Create comprehensive plot set
      pdf(file.path(output_dir, "analysis_plots.pdf"), width = 12, height = 10)

      # 1. Parameter distributions by group
      p1 <- galaxy_res %>%
        filter(!is.na(mean_t_sf)) %>%
        ggplot(aes(x = factor(group_id), y = mean_t_sf)) +
        geom_boxplot(aes(fill = Activity_class)) +
        theme_minimal() +
        theme(axis.text.x = element_text(angle = 45, hjust = 1)) +
        labs(title = "Formation Time Distribution by Group",
             x = "Group ID", y = "Formation Time (Gyr)")
      print(p1)

      # 2. Group-level parameter relationships
      p2 <- group_res %>%
        filter(n_galaxies > 10) %>%  # Only well-populated groups
        ggplot(aes(x = mean_mu_tau_group, y = mean_mu_t_sf_group)) +
        geom_point(aes(size = n_galaxies, color = mean_mean_sSFR_group)) +
        geom_errorbar(aes(ymin = mean_mu_t_sf_group - sd_mu_t_sf_group,
                          ymax = mean_mu_t_sf_group + sd_mu_t_sf_group),
                      alpha = 0.3) +
        geom_errorbarh(aes(xmin = mean_mu_tau_group - sd_mu_tau_group,
                           xmax = mean_mu_tau_group + sd_mu_tau_group),
                       alpha = 0.3) +
        scale_color_viridis_c(name = "Mean sSFR") +
        theme_minimal() +
        labs(title = "Group Parameter Relationships",
             x = "Mean τ (Gyr)", y = "Mean Formation Time (Gyr)",
             size = "N Galaxies")
      print(p2)

      # 3. Observed vs Predicted SFR
      p3 <- galaxy_res %>%
        filter(!is.na(mean_logSFR_today_pred)) %>%
        ggplot(aes(x = logSFR_HECv2, y = mean_logSFR_today_pred)) +
        geom_point(alpha = 0.5, size = 0.5) +
        geom_abline(intercept = 0, slope = 1, color = "red", linetype = "dashed") +
        facet_wrap(~Activity_class, labeller = labeller(
          Activity_class = c("0" = "Star-forming", "1" = "AGN",
                             "2" = "LINER", "3" = "Composite", "4" = "Passive")
        )) +
        theme_minimal() +
        labs(title = "Observed vs Predicted log(SFR) by Activity Class",
             x = "Observed log(SFR)", y = "Predicted log(SFR)")
      print(p3)

      # 4. Parameter evolution with mass
      p4 <- galaxy_res %>%
        filter(!is.na(mean_x)) %>%
        ggplot(aes(x = logM_star_HECv2, y = mean_x)) +
        geom_point(aes(color = ms_group), alpha = 0.5, size = 0.5) +
        geom_smooth(method = "loess", se = TRUE) +
        scale_color_brewer(palette = "Set1", name = "MS Group") +
        theme_minimal() +
        labs(title = "x = t_sf/τ vs Stellar Mass",
             x = "log(M*)", y = "x = t_sf/τ")
      print(p4)

      # 5. Group hierarchy visualization
      if (nrow(group_res) <= 50) {  # Only if not too many groups
        p5 <- group_res %>%
          arrange(desc(n_galaxies)) %>%
          head(30) %>%
          ggplot(aes(x = reorder(group_name, n_galaxies), y = n_galaxies)) +
          geom_bar(stat = "identity", aes(fill = mean_mean_x_group)) +
          coord_flip() +
          scale_fill_viridis_c(name = "Mean x") +
          theme_minimal() +
          labs(title = "Top 30 Groups by Galaxy Count",
               x = "Group", y = "Number of Galaxies")
        print(p5)
      }

      dev.off()

      file.path(output_dir, "analysis_plots.pdf")
    },
    format = "file"
  )
)