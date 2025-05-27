/*
  Stan model for estimating the star formation rate (SFR) of galaxies
  based on individual parameters: time of star formation (t_sf),
  tau (logtau), and A (logA). The model predicts log SFR and compares
  it with observed data to infer parameter values.

  Key Components:
  - t_sf: Time of star formation in Gyr.
  - logtau: log10(tau) in Gyr.
  - logA: log10(A) (scaling factor).
  - logSFR_today: Modeled logarithmic SFR at the present time.
*/

data {
  int<lower=1> N;                   // Number of data points
  vector[N] logSFR_total;        // Observed log SFR in Gyr
  vector[N] ID;             // Row identifiers for the data points
  vector[N] M_star;                 // Stellar masses (input data)
}

parameters {
  vector<lower=1, upper=13.8>[N] t_sf;     // Time of star formation for each galaxy in Gyr
  vector<lower=1, upper=20>[N] tau;      // log10(tau) for each galaxy
  vector<lower=1,upper=2>[N] zeta;        // Mass-loss
}

transformed parameters {

  vector[N] x = t_sf./tau;
  vector[N] logSFR_today;                   // Modeled log SFR for each data point
  real log10_e = log10(exp(1));             // Constant for ln to log10 conversion (log10(e))

  // Compute A for each galaxy using the full normalization equation
  vector[N]  A = (M_star .* zeta) ./ (1 - (x + 1) .* exp(-x));

  // Calculate modeled log SFR at the present time for each galaxy
  logSFR_today = log10(A) -9 + log10(x)- log10(tau)- (x) * log10_e;
  // 1/yr=10^9/Gyr from SFR// Subtract exponential decay term
}

model {
  /*
    Bayesian model specification:
    - Priors on parameters (t_sf, logtau).
    - Likelihood: Observed log SFR is modeled as normally distriiiibuted around the predicted value.
  */

  // Priors: Uniform distributions for each parameter
  tau ~  normal(4,1); //                // Prior for log10(tau)
  //t_sf ~ uniform(1, 13.8);                // Prior for t_sf (time of star formation)
  t_sf ~ skew_normal(13.0,0.5,-8)T[,13.8];

  zeta ~ normal(1.3, 0.01);

  // Likelihood: Observed log SFR compared to modeled log SFR
  logSFR_total ~ normal(logSFR_today, 0.1); // Assume small measurement error (std = 0.1)
}

generated quantities {
  /*
    Generated quantities:
    - Predicted log SFR for posterior predictive checks.
    - Pass-through of row identifiers for post-modeling analysis.
  */
  vector[N] logSFR_today_pred;            // Predicted log SFR for each data point
  vector[N] id;                          // Row identifiers

  // Predict log SFR using the same formula as in `transformed parameters`
  logSFR_today_pred = log10(A) -9
                      + log(x)
                      - log(tau)
                      - (t_sf ./ tau) * log10_e;
  // Assign row identifiers for reference
  id = ID;
}

