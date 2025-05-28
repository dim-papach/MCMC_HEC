#!/usr/bin/env Rscript
# run_hierarchical_analysis.R
# Simple script to run the complete hierarchical analysis

# Load required libraries
library(tidyverse)
library(cmdstanr)
library(here)

# Set working directory to script location
setwd(here::here())

# Source helper functions
source("R/hierarchical_functions.R")

# Check if data preparation has been done
if (!file.exists("tables/hierarchical_data.csv")) {
  stop("Please run 'python prepare_hierarchical_data.py' first to prepare the data!")
}

# Print system info
cat("Running hierarchical analysis on:", Sys.info()["nodename"], "\n")
cat("Using", parallel::detectCores(), "cores\n")
cat("CmdStan path:", cmdstan_path(), "\n\n")

# Run the analysis with default parameters
# You can modify these parameters as needed
fit <- run_hierarchical_analysis(
  data_file = "tables/hierarchical_data.csv",
  stan_file = "hierarchical_delayed_tau.stan",
  chains = 4,                  # Number of MCMC chains
  iter_sampling = 2000,        # Sampling iterations per chain
  iter_warmup = 1000,          # Warmup iterations per chain
  adapt_delta = 0.95,          # Target acceptance rate
  max_treedepth = 12,          # Maximum tree depth
  output_dir = "hierarchical_results"
)

# Check convergence
cat("\n=== Convergence Check ===\n")
convergence_check <- check_convergence(fit)

# Print summary
cat("\n=== Analysis Complete ===\n")
cat("Results saved in: hierarchical_results/\n")
cat("Next steps:\n")
cat("1. Run 'python analyze_hierarchical_results.py' for detailed analysis\n")
cat("2. Check diagnostic plots in hierarchical_results/diagnostic_plots.pdf\n")
cat("3. Review convergence warnings above (if any)\n")

# Optional: Create a quick summary plot
if (require(ggplot2)) {
  galaxy_results <- read_csv("hierarchical_results/galaxy_parameters.csv",
                            show_col_types = FALSE)

  p <- ggplot(galaxy_results, aes(x = logSFR_HECv2, y = mean_logSFR_today_pred)) +
    geom_point(alpha = 0.5, size = 0.5) +
    geom_abline(intercept = 0, slope = 1, color = "red") +
    facet_wrap(~Activity_class) +
    theme_minimal() +
    labs(title = "Quick Check: Observed vs Predicted SFR",
         x = "Observed log(SFR)",
         y = "Predicted log(SFR)")

  ggsave("hierarchical_quick_check.png", p, width = 10, height = 8)
  cat("\nQuick check plot saved as: hierarchical_quick_check.png\n")
}
