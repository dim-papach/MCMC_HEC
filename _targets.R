# _targets.R: multi-model, multi-dataset pipeline using explicit cross pattern

library(targets)
library(tarchetypes)

tar_source()

chains <- 6
parallel_chains <- 6
sample_size <- 5000
warmup <- 2500

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
      "tables/Hec_11-20.csv",
      "tables/Hec_21-30.csv",
      "tables/Hec_31-40.csv",
      "tables/Hec_41-50.csv"
    )
  ),

  tar_target(
    data_names,
    c(
      "Hec_10",
      "Hec_11-20",
      "Hec_21-30",
      "Hec_31-40",
      "Hec_41-50"
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
    c("np", "uni", "skew")
  ),

  tar_target(
    stan_files,
    c(
      "x.stan",
      "uniform.stan",
      "skew.stan"
    )
  ),

  tar_target(
    model_list,
    {
    file_times <- file.mtime(stan_files)

    list(
      model = cmdstan_model(stan_files),
      names = model_names
    )},
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
      init_vals <- init_function(chains, nrow(data_list$data))
      fit <- model$sample(
        data            = stan_data,
        chains          = chains,
        parallel_chains = parallel_chains,
        iter_sampling   = sample_size,
        iter_warmup     = warmup,
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
  tar_target(
    write_summary,
    {
      # Verify required components exist
      stopifnot(
        !is.null(fit_model$data_name),
        !is.null(fit_model$model_name)
      )

      params <- c("t_sf", "tau", "A", "x", "id", "logSFR_today_pred")
      suffix <- fit_model$model_name
      data_name <- fit_model$data_name

      output_dir <- here::here(data_name, suffix)
      dir.create(output_dir, showWarnings = FALSE, recursive = TRUE)

      # Process individual parameters
      individual_files <- purrr::map_chr(params, function(param) {
        filename <- file.path(output_dir, paste0(param, "_", suffix, ".csv"))
        fit_model$fit$summary(variable = param) |>
          readr::write_csv(filename)
        filename
      })

      # Create combined summary with param and param_sd columns
      combined_summary <- purrr::map_dfc(params, function(param) {
        summary_row <- fit_model$fit$summary(variable = param)
        tibble::tibble(
          "{param}_{suffix}" := summary_row$mean,
          "{param}_sd" := summary_row$sd
        )
      })

      # Save combined summary
      combined_file <- file.path(output_dir, paste0("combined_summary_", suffix, ".csv"))
      readr::write_csv(combined_summary, combined_file)

      # Return all created files
      c(individual_files, combined_file)
    },
    pattern = map(fit_model),
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