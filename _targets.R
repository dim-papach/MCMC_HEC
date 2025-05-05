# _targets.R: multi-model, multi-dataset pipeline using explicit cross pattern

library(targets)
library(tarchetypes)
library(future)
library(cmdstanr)
library(tidyverse)

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
      "tables/Hec_2-10.csv",
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
      "Hec_2-10",
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
      np = "x.stan",
      uni = "uniform.stan",
      skew = "skew.stan"
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
      init_vals <- init_function(6, nrow(data_list$data))
      fit <- model$sample(
        data            = stan_data,
        chains          = 6,
        parallel_chains = 6,
        iter_sampling   = 4000,
        iter_warmup     = 1500,
        init            = init_vals,
        refresh         = 100
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

      params <- c("t_sf", "tau", "A", "x", "id")
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
          "{param}" := summary_row$mean,
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
  ),

# Combined summaries ------------------------------------------------------
tar_target(
  combined_model_summaries,
  {
    # Define base datasets and their cumulative groupings
    base_groups <- list(
      "WholeHEC" = c("Hec_10", "Hec_11-20", "Hec_21-30", "Hec_31-40", "Hec_41-50"),
      "WholeHEC-2Mpc" = c("Hec_2-10", "Hec_11-20", "Hec_21-30", "Hec_31-40", "Hec_41-50")
    )

    # For each model and base group, combine cumulative summaries
    map_dfr(names(base_groups), function(base_name) {
      map_dfr(model_names, function(current_model) {
        # Get all relevant dataset groups for this base
        cumulative_groups <- accumulate(base_groups[[base_name]], ~ c(.x, .y))

        # Process each cumulative combination
        map_dfr(seq_along(cumulative_groups), function(n) {
          current_datasets <- cumulative_groups[[n]]

          # Collect all summary files for these datasets
          summary_files <- file.path(
            here::here(current_datasets, current_model),
            paste0("combined_summary_", current_model, ".csv")
          )

          # Read and vertically concatenate summaries
          combined <- map_dfr(summary_files, ~ {
            read_csv(.x, show_col_types = FALSE) |>
              mutate(source_dataset = str_extract(.x, "Hec_[^/]+"))
          }) |>
            mutate(
              cumulative_step = n,
              model = current_model,
              base_group = base_name
            )

          # Save combined summary with new directory structure
          output_dir <- here::here("results", current_model, base_name)
          dir.create(output_dir, showWarnings = FALSE, recursive = TRUE)
          output_file <- file.path(
            output_dir,
            paste0("stacked_step", n, "_", current_model, ".csv")
          )
          write_csv(combined, output_file)

          tibble(file = output_file, model = current_model, group = base_name)
        })
      })
    }) |>
      pull(file) |>
      unlist()
  },
  pattern = map(model_names),
  cue = tar_cue(mode = "always"),
  format = "file"
),


# Visualizations --------------------------------

  tar_target(
    histograms,
    #Histograms of models per 10 Mpc
    {
      # Validate inputs
      stopifnot(
        !is.null(fit_model$data_name),
        !is.null(fit_model$model_name)
      )

      # Set base directory using write_summary's output structure
      base_dir <- here::here(fit_model$data_name, fit_model$model_name)

      # Generate histograms from saved files
      create_file_based_histograms(fit_model, base_dir)
    },
    pattern = map(fit_model),
    format = "file"
  ),

# Faceted Histograms ------------------------------------------------------
# Faceted Histograms ------------------------------------------------------
tar_target(
  cumulative_histograms,
  {
    # Get all stacked summary files
    stacked_files <- list.files(
      here::here("results"),
      pattern = "stacked_step\\d+_.+\\.csv",
      recursive = TRUE,
      full.names = TRUE
    )

    # Create one plot per stacked file
    map(stacked_files, function(file_path) {
      # Read data and extract metadata
      data <- read_csv(file_path, show_col_types = FALSE)
      model <- str_extract(file_path, "(?<=results/)[^/]+")
      base_group <- str_extract(file_path, "(?<=/)[^/]+(?=/)")
      step <- str_extract(file_path, "(?<=stacked_step)\\d+")

      # Prepare data for plotting
      plot_data <- data |>
        pivot_longer(
          cols = c(t_sf, tau, A, x, t_sf_sd, tau_sd, A_sd, x_sd),
          names_to = "parameter",
          values_to = "value"
        )

      # Create faceted histogram
      p <- ggplot(plot_data, aes(x = value, fill = source_dataset)) +
        geom_histogram(alpha = 0.7, bins = 30, position = "identity") +
        facet_wrap(vars(parameter), ncol = 4, scales = "free") +
        labs(
          title = paste("Parameter Distributions:", model, "-", base_group),
          subtitle = paste("Cumulative Step", step),
          x = "Value",
          y = "Count"
        ) +
        theme_minimal() +
        theme(
          legend.position = "bottom",
          strip.text = element_text(size = 8)
        )

      # Save in same directory as data
      output_dir <- dirname(file_path)
      output_file <- file.path(output_dir, paste0("histograms_step", step, ".png"))
      ggsave(output_file, plot = p, width = 12, height = 8, dpi = 300)

      output_file
    }) |> unlist()
  },
  format = "file"
),

# Combine /results/model_name/WholeHEC/stacked_step5_model_name.csv with
# tables/Hec_50.csv based in the stack$id and Hec$ID columns
tar_target(
  combine_results,
  {
    # Process each model
    combined_results <- map_dfr(model_names, function(model) {
      # Read model's final cumulative data
      stacked_file <- here::here("results", model, "WholeHEC", paste0("stacked_step5_", model, ".csv"))
      stacked_data <- read_csv(
        stacked_file,
        col_types = cols(
          .default = col_double(),
          id = col_integer(),
          source_dataset = col_character(),
          model = col_character(),
          base_group = col_character()
        )
      )
      # Alternative renaming that definitely preserves types
      new_names <- names(stacked_data) %>%
        if_else(. == "id", ., paste0(., "_", model))
      names(stacked_data) <- new_names
      # Read Hec_50 data
      hec_50_data <- read_csv(here::here("tables", "Hec_50.csv"), show_col_types = FALSE)

      # Merge based on ID columns
      left_join(stacked_data, hec_50_data, by = c("id" = "ID")) %>%
        mutate(model = model)
    })

    # Save combined results
    output_file <- here::here("tables", "MCMC_results.csv")
    write_csv(combined_results, output_file)
    output_file
  },
  format = "file"
)

)

