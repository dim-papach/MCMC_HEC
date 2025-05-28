# functions.R

# Function to load and prepare the data
prepare_data <- function(data_csv) {
  dt <- data_csv %>%
    select(logSFR_total, ID, logM_total, M_total, sSFR) %>%
    filter(
      # iogSFR_total >=-3,
      !is.na(logSFR_total),
      !is.nan(logSFR_total),
      is.finite(logSFR_total),
      !is.na(logM_total),
      !is.nan(logM_total),
      is.finite(logM_total)
    ) %>%
    return(dt)
}

# Function to define initial values for the Stan model
init_function <- function(chains, N) {
  lapply(1:chains, function(i) {
    list(
      t_sf = rep(13.6, N),
      # x = rep(3, N),
      logtau = rep(log10(4), N), # MS galaxies 3.5<tau<4.5
      # logA = rep(5.5, N),
      tau = rep(4, N),
      zeta = rep(1.3, N)
    )
  })
}

mean_vs_median <- function(summary_data, summary_variables) {
  ggplot(summary_data, aes(x = mean, y = median)) +
    geom_point() +
    geom_abline(intercept = 0, slope = 1, color = "red", linetype = "dashed") +
    labs(
      title = paste("Mean vs. Median of", summary_variables),
      x = "Mean",
      y = "Median"
    )
  ggsave(sprintf("plots/Mean-Median/%s.png", summary_variables))
}

# Function to create histograms of the mcmc results, with mean value highlighted
plot_histograms <- function(summary_data, summary_variables, summary_names) {
  max_count <- max(ggplot_build(
    ggplot(summary_data, aes(x = mean)) +
      geom_histogram(bins = 30)
  )$data[[1]]$count, na.rm = TRUE)

  p <- ggplot(summary_data, aes(x = mean)) +
    geom_histogram(alpha = 0.7, bins = 40, fill = "blue", color = "black") +
    geom_vline(aes(xintercept = mean(mean)), color = "red", linetype = "dashed", size = 1) +
    geom_text(
      aes(
        x = mean(mean),
        label = format(mean(mean), scientific = TRUE, digits = 2) # Scientific notation, 2 decimal points
      ),
      y = max_count + 1, # Place the text slightly above the maximum bar
      inherit.aes = FALSE,
      vjust = 0, # Above the bar
      hjust = 0.5 # Centered above the mean line
    ) +
    labs(
      title = "Histogram of the mean values from the MCMC run",
      x = summary_names,
      y = "Number of galaxies"
    ) +
    ggeasy::easy_center_title()

  ggsave(sprintf("plots/Hists/%s.png", summary_variables), p)
}




sfr_comparison_plot <- function(sfr_diff, logSFR_pred, logSFR_total) {
  default_colors <- scales::hue_pal()(2)
  # Create a scatter plot of logSFR_pred vs logSFR_total
  ggplot(sfr_diff, aes(y = logSFR_pred, x = logSFR_total, color = flag)) +
    geom_point(size = 0.9) +
    # Error bar for x based on _sigma
    geom_errorbar(
      aes(
        ymin = logSFR_pred - logSFR_pred_sigma,
        ymax = logSFR_pred + logSFR_pred_sigma
      ),
      width = 0.1
    ) +
    # legend for keep and discard = "kept data", "discarded data"+ the number of points for eac
    scale_color_manual(
      values = c("keep" = default_colors[2], "discard" = default_colors[1]),
      breaks = c("keep", "discard"),
      labels = c(
        paste("kept data (n =", sum(sfr_diff$flag == "keep"), ")"),
        paste("discarded data (n =", sum(sfr_diff$flag == "discard"), ")")
      )
     ) +
    geom_abline(intercept = 0, slope = 1, color = "red", linetype = "dashed") +
    xlim(-8, 1) +
    labs(
      title = TeX("Comparison of $log_{10}SFR_{pred}$ and $log_{10}SFR_{obs}$"),
      # x = TeX("$log_{10}\\left[\\frac{SFR_{obs}}{M_\u2299/yr} \\right]$"),
      x = expression(log[10] * bgroup("[", frac(SFR[obs], M["\u2299"] / yr), "]")),
      y = expression(log[10] * bgroup("[", frac(SFR[pred], M["\u2299"] / yr), "]"))
    ) +
    ggeasy::easy_center_title() +
    ggeasy::easy_legend_at("top")

  ggsave("plots/sfr_diff_plot.png")
}


# Helper function for histograms using saved files
create_file_based_histograms <- function(fit_model, base_dir) {
  params <- c("t_sf", "tau", "A", "x")
  output_files <- character(0)

  purrr::walk(params, function(param) {
    # Read from saved summary files
    summary_file <- here::here(
      fit_model$data_name,
      fit_model$model_name,
      paste0(param, "_", fit_model$model_name, ".csv")
    )

    if(file.exists(summary_file)) {
      # Create parameter directory
      param_dir <- file.path(base_dir, "plots")
      dir.create(param_dir, showWarnings = FALSE, recursive = TRUE)

      # Read summary data
      summary_data <- read_csv(summary_file, show_col_types = FALSE)

      # Create histogram from saved means
      plot <- ggplot(summary_data, aes(x = mean)) +
        geom_histogram(fill = "skyblue", color = "black", bins = 30) +
        labs(
          title = paste("Parameter:", param),
          subtitle = paste("Model:", fit_model$model_name,
                           "| Data:", fit_model$data_name,
                           "|ESS bulk:", round(summary_data$ess_bulk, 1),
                           "| Rhat:", round(summary_data$rhat, 3)),
          x = "Posterior Mean Value"
        ) +
        geom_vline(aes(xintercept = mean(mean)), color = "red", linetype = "dashed") +
        geom_text(
          aes(
            x = mean(mean),
            label = format(mean(mean), scientific = TRUE, digits = 2)
          ),
          # Place the text at the bottom of the histogram
          y = -1,
          vjust = -0.5,
          color = "red"
        ) +
        # white bg, not transparent
        theme_minimal() +
        theme(
          panel.background = element_rect(fill = "white"),
          plot.background = element_rect(fill = "white"),
          panel.grid.major = element_line(color = "grey80"),
          panel.grid.minor = element_blank(),
          axis.title.x = element_text(size = 14),
          axis.title.y = element_text(size = 14),
          plot.title = element_text(size = 16, face = "bold"),
          plot.subtitle = element_text(size = 12)
        )


      # Save plot
      plot_file <- file.path(param_dir, paste0(param, "_histogram.png"))
      ggsave(plot_file, plot, width = 8, height = 6)
      output_files <<- c(output_files, plot_file)
    }
  })

  output_files
}

combine_model_results <- function(model_names, results_dir = "results",
                                  stop_on_error = TRUE)
{
  tryCatch({
    # Read and rename each model's data
    model_data_list <- purrr::map(model_names, function(model) {
      stacked_file <- here::here(results_dir, model, "WholeHEC",
                            paste0("stacked_step5_", model, ".csv"))

      readr::read_csv(
        stacked_file,
        col_types = readr::cols(
          .default = readr::col_double(),
          id = readr::col_integer(),
          source_dataset = readr::col_character(),
          model = readr::col_character(),
          base_group = readr::col_character()
        )
      ) %>%
        dplyr::rename_with(~if_else(.x == "id", .x, paste0(.x, "_", model)), -id)
    })

    # Merge all models by id
    combined_results <- purrr::reduce(model_data_list, ~dplyr::full_join(.x, .y, by = "id"))

    # Merge with Hec_50 data
    hec_50_data <- readr::read_csv(
      here::here("tables", "Hec_50.csv"),
      show_col_types = FALSE
    ) %>%
      dplyr::rename(id = ID)

    final_data <- dplyr::left_join(combined_results, hec_50_data, by = "id")

    # Save output
    return(final_data)
  }, error = function(e) {
    message("Error in combine_model_results: ", e$message)
    if (stop_on_error) {
      stop(e)
    } else {
      return(NULL)
    }
  })
}