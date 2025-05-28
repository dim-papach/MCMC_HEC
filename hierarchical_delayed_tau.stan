/*
  Hierarchical Stan model for delayed-tau star formation histories.
  
  This model allows parameters to vary by galaxy group, with hyperpriors
  controlling the group-level distributions. The hierarchical structure
  enables partial pooling of information across groups.
  
  Model structure:
  - Galaxy level: Individual galaxy parameters
  - Group level: Mean and variance for each group's parameters
  - Population level: Hyperpriors on group parameters
*/

data {
  int<lower=1> N;                          // Number of galaxies
  int<lower=1> N_groups;                   // Number of hierarchical groups
  
  vector[N] logSFR_total;                  // Observed log SFR
  vector[N] M_star;                        // Stellar masses
  array[N] int<lower=1, upper=N_groups> group;  // Group assignment for each galaxy
  vector[N] ID;                            // Galaxy IDs
}

parameters {
  // Population-level hyperparameters
  real<lower=0, upper=13.8> mu_t_sf_pop;      // Population mean formation time
  real<lower=0> sigma_t_sf_pop;               // Population SD formation time
  
  real<lower=0, upper=10> mu_tau_pop;         // Population mean tau
  real<lower=0> sigma_tau_pop;                // Population SD tau
  
  real<lower=1, upper=2> mu_zeta_pop;         // Population mean mass-loss
  real<lower=0> sigma_zeta_pop;               // Population SD mass-loss
  
  // Group-level parameters
  vector<lower=0, upper=13.8>[N_groups] mu_t_sf_group;   // Group mean formation times
  vector<lower=0>[N_groups] sigma_t_sf_group;            // Group SD formation times
  
  vector<lower=0, upper=20>[N_groups] mu_tau_group;      // Group mean tau
  vector<lower=0>[N_groups] sigma_tau_group;             // Group SD tau
  
  vector<lower=1, upper=2>[N_groups] mu_zeta_group;      // Group mean zeta
  vector<lower=0>[N_groups] sigma_zeta_group;            // Group SD zeta
  
  // Galaxy-level parameters (non-centered parameterization)
  vector[N] t_sf_raw;                      // Raw formation time (standardized)
  vector[N] tau_raw;                       // Raw tau (standardized)
  vector[N] zeta_raw;                      // Raw zeta (standardized)
  
  // Observation error
  real<lower=0> sigma_obs;                 // Observation error in log(SFR)
}

transformed parameters {
  // Galaxy-level parameters (actual scale)
  vector<lower=0, upper=13.8>[N] t_sf;
  vector<lower=0, upper=20>[N] tau;
  vector<lower=1, upper=2>[N] zeta;
  
  vector[N] x;                             // t_sf / tau
  vector[N] A;                             // Normalization
  vector[N] logSFR_today;                  // Modeled log SFR
  
  real log10_e = log10(exp(1));            // Conversion constant
  
  // Non-centered parameterization for better sampling
  for (i in 1:N) {
    // Transform from standardized to actual parameters
    t_sf[i] = mu_t_sf_group[group[i]] + sigma_t_sf_group[group[i]] * t_sf_raw[i];
    tau[i] = mu_tau_group[group[i]] + sigma_tau_group[group[i]] * tau_raw[i];
    zeta[i] = mu_zeta_group[group[i]] + sigma_zeta_group[group[i]] * zeta_raw[i];
    
    // Apply bounds
    t_sf[i] = fmax(0.1, fmin(13.8, t_sf[i]));
    tau[i] = fmax(0.1, fmin(20, tau[i]));
    zeta[i] = fmax(1.0, fmin(2.0, zeta[i]));
    
    // Calculate derived quantities
    x[i] = t_sf[i] / tau[i];
    
    // Normalization with numerical stability check
    if (x[i] < 10) {  // exp(-x) is not too small
      A[i] = (M_star[i] * zeta[i]) / (1 - (x[i] + 1) * exp(-x[i]));
    } else {  // For large x, use approximation
      A[i] = (M_star[i] * zeta[i]) / 1.0;
    }
    
    // Calculate log SFR
    logSFR_today[i] = log10(A[i]) - 9 + log10(x[i]) - log10(tau[i]) - x[i] * log10_e;
  }
}

model {
  // Population-level priors
  mu_t_sf_pop ~ normal(10, 2);            // Most galaxies form around 10 Gyr ago
  sigma_t_sf_pop ~ normal(0, 2)T[0.1,];          // Half-normal for scale
  
  mu_tau_pop ~ normal(4, 1);              // Typical tau around 4 Gyr
  sigma_tau_pop ~ normal(0, 1)T[0.1,];
  
  mu_zeta_pop ~ normal(1.3, 0.1)T[0,];         // Mass-loss factor
  sigma_zeta_pop ~ normal(0, 0.1)T[0.1,];
  
  // Group-level priors
  mu_t_sf_group ~ normal(mu_t_sf_pop, sigma_t_sf_pop);
  sigma_t_sf_group ~ normal(0, 1)T[0.1,];
  
  mu_tau_group ~ normal(mu_tau_pop, sigma_tau_pop);
  sigma_tau_group ~ normal(0, 0.5)T[0.1,];
  
  mu_zeta_group ~ normal(mu_zeta_pop, sigma_zeta_pop);
  sigma_zeta_group ~ normal(0, 0.05)T[0.1,];
  
  // Galaxy-level priors (standardized)
  t_sf_raw ~ std_normal();
  tau_raw ~ std_normal();
  zeta_raw ~ std_normal();
  
  // Observation error prior
  sigma_obs ~ normal(0, 0.5)T[0.1,];
  
  // Likelihood
  logSFR_total ~ normal(logSFR_today, sigma_obs);
}

generated quantities {
  vector[N] logSFR_today_pred;            // Predicted log SFR
  vector[N] log_lik;                      // Log likelihood for LOO
  vector[N] id;                           // Galaxy IDs
  
  // Group-level summaries
  vector[N_groups] mean_x_group;          // Mean x = t_sf/tau per group
  vector[N_groups] mean_sSFR_group;       // Mean specific SFR per group
  
  // Calculate predictions and diagnostics
  for (i in 1:N) {
    logSFR_today_pred[i] = normal_rng(logSFR_today[i], sigma_obs);
    log_lik[i] = normal_lpdf(logSFR_total[i] | logSFR_today[i], sigma_obs);
    id[i] = ID[i];
  }
  
  // Calculate group summaries
  for (g in 1:N_groups) {
    int n_g = 0;
    real sum_x = 0;
    real sum_sSFR = 0;
    
    for (i in 1:N) {
      if (group[i] == g) {
        n_g += 1;
        sum_x += x[i];
        sum_sSFR += 10^(logSFR_today[i] - log10(M_star[i]));
      }
    }
    
    if (n_g > 0) {
      mean_x_group[g] = sum_x / n_g;
      mean_sSFR_group[g] = sum_sSFR / n_g;
    } else {
      mean_x_group[g] = 0;
      mean_sSFR_group[g] = 0;
    }
  }
}
