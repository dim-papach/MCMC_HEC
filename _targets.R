# _targets.R: multi-model, multi-dataset pipeline using explicit cross pattern

library(targets)
library(tarchetypes)

tar_source()

options(tidyverse.quiet = TRUE)
tar_option_set(
  packages = c(
    "readr", "dplyr", "tidyr", "tibble", "stringr", "here",
    "cmdstanr", "posterior", "bayesplot", "ggplot2"
  ),
  # set multiprocessing to 4 cores

  garbage_collection = TRUE,
  format = "qs"
)

# Define datasets and models
list(
# Dataset definitions ------------------------------------------------------
  tar_target(
    data_files,
    c(
      "tables/Hec_10.csv",
      "tables/Hec_11-20.csv"
      #"tables/Hec_21-30.csv",
      #"tables/Hec_31-40.csv",
      #"tables/Hec_41-50.csv"
    )
  ),

  tar_target(
    data_names,
    c(
      "Hec_10",
      "Hec_11-20"
      #"Hec_21-30",
      #"Hec_31-40",
      #"Hec_41-50"
    )
  ),

  tar_target(
    data_list,
    {
      data <- read_csv(data_files, show_col_types = FALSE)
      list(
        data_name = data_names,
        data = prepare_data(data)
      )
    },
    pattern = map(data_files, data_names),
  ),

  # Model definitions ------------------------------------------------
  tar_target(
    model_names,
    c("np", "uni")#, "skew")
  ),

  tar_target(
    stan_files,
    c(
      np = "x.stan",
      uni = "uniform.stan"
      #skew = "skew.stan"
    )
  ),

  tar_target(
    model_list,
    list(
    model = cmdstan_model(stan_files),
    names = model_names
    ),
    pattern = map(stan_files, model_names),
  ),


  # Fit models to datasets ------------------------------------------
  tar_target(
    fit_model,
    {
      # Extract data and model
      model <- model_list$model
      # Prepare Stan data
      stan_data <- list(
        N = nrow(data_list$data),
        logSFR_total = data_list$data$logSFR_total,
        ID = data_list$data$ID,
        M_star = data_list$data$M_total
      )

      # Fit the model
      chains <- 1
      init_vals <- init_function(chains, nrow(data_list$data))
      fit <- model$sample(
        data            = stan_data,
        chains          = chains,
        parallel_chains = 6,
        iter_sampling   = 700,
        iter_warmup     = 300,
        init            = init_vals,
        refresh         = 100,
        max_treedepth   = 12,
      )
      # Save the fit object to the output directory .mcmc_models
      output_dir <- here::here(".mcmc_models",data_list$data_name)
      dir.create(output_dir, showWarnings = FALSE, recursive = TRUE)

      fit$summary()

      qs::qsave(
        x=fit,
        file = file.path(output_dir, paste0(model_list$names, ".qs")),
        nthread = 3
      )

      # Return fit object
      list(
        fit = fit,
        data_name = data_list$data_name,
        parameters = fit$draws(),
        model_name = model_list$names
      )
    },
    pattern = cross(data_list, model_list),
  ),


  # Process results --------------------------------
  ## Configuration target --------------------------------
  tar_target(
    summary_config,
    {
      # Verify required components exist
      stopifnot(
        !is.null(fit_model$data_name),
        !is.null(fit_model$model_name)
      )

      list(
        params = c("t_sf", "tau", "A", "x", "id", "logSFR_today_pred"),
        suffix = fit_model$model_name,
        data_name = fit_model$data_name,
        output_dir = here::here(fit_model$data_name, fit_model$model_name)
      )
    },
    pattern = map(fit_model)
  ),
  # Summary data target --------------------------------
  tar_target(
    summary_data,
    {
      dir.create(summary_config$output_dir, showWarnings = FALSE, recursive = TRUE)

      # Extract the summary of the parameters
      summary

    },
    pattern = map(summary_config)
  ),

  # Write individual parameter files --------------------------------
  tar_target(
    param_files,
    {
      purrr::map_chr(summary_config$params, function(param) {
        filename <- file.path(
          summary_config$output_dir,
          paste0(param, "_", summary_config$suffix, ".csv")
        )
        summary_data |>
          readr::write_csv(filename)
        filename
      })
    },
    pattern = map(summary_config, summary_data),
    format = "file"
  ),

  # Write combined summary file --------------------------------
  tar_target(
    combined_file,
    {
      combined <- purrr::map_dfc(summary_config$params, function(param) {
        param_data <- summary_data |>
        tibble::tibble(
          "{param}" := param_data$mean,
          "{param}_sd" := param_data$sd
        )
      })
      filename <- file.path(
        summary_config$output_dir,
        paste0("combined_summary_", summary_config$suffix, ".csv")
      )
      readr::write_csv(combined, filename)
      filename
    },
    pattern = map(summary_config, summary_data),
    format = "file"
  ),


  # Diagnostics --------------------------------
  tar_target(
    diagnostic_checks,
    {
      # Verify required components exist
      stopifnot(
        !is.null(fit_model$data_name),
        !is.null(fit_model$model_name)
      )

      # Create output directory
      output_dir <- here::here(fit_model$data_name, fit_model$model_name)
      dir.create(output_dir, showWarnings = FALSE, recursive = TRUE)

      # Process and save sampler diagnostics
      diagnostics_file <- file.path(output_dir,
                                    paste0("diagnostics_", fit_model$model_name, ".csv"))
      fit_model$fit$sampler_diagnostics() %>%
        as.data.frame() %>%
        readr::write_csv(diagnostics_file)

      # Process and save diagnostic summary
      summary_file <- file.path(output_dir,
                                paste0("summary_", fit_model$model_name, ".txt"))
      capture.output(
        fit_model$fit$diagnostic_summary(),
        file = summary_file
      )

      # Return both file paths
      c(diagnostics_file, summary_file)
    },
    pattern = map(fit_model),
    format = "file"
  )

)