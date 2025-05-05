import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import scipy.integrate
from astropy.cosmology import Planck18 as cosmo
import astropy.units as u
import os

# set wd as the directory of the script
script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)


# Create output directory
os.makedirs('sfrd_plots', exist_ok=True)

# Load data once at start
data = pd.read_csv('tables/MCMC_results.csv')
T0 = 13.8 * 1e9  # Age of the Universe in years

# Load ALL parameters once (before volume loops)
# Uniform Prior
A_uni = data['A_uni'].values
tau_uni = (data['tau_uni']*u.Gyr.to(u.yr)).values
tsf_uni = (data['t_sf_uni']*u.Gyr.to(u.yr)).values
tstart_uni = T0 - tsf_uni
sd_A_uni = data['A_sd_uni'].values
sd_tau_uni = (data['tau_sd_uni']*u.Gyr.to(u.yr)).values
sd_tsf_uni = (data['t_sf_sd_uni']*u.Gyr.to(u.yr)).values
sd_tstart_uni = sd_tsf_uni

# Normal Prior
A_norm = data['A_np'].values
tau_norm = (data['tau_np']*u.Gyr.to(u.yr)).values
tsf_norm = (data['t_sf_np']*u.Gyr.to(u.yr)).values
tstart_norm = T0 - tsf_norm
sd_A_norm = data['A_sd_np'].values
sd_tau_norm = (data['tau_sd_np']*u.Gyr.to(u.yr)).values
sd_tsf_norm = (data['t_sf_sd_np']*u.Gyr.to(u.yr)).values
sd_tstart_norm = sd_tsf_norm

# Skew Prior
A_skew = data['A_skew'].values
tau_skew = (data['tau_skew']*u.Gyr.to(u.yr)).values
tsf_skew = (data['t_sf_skew']*u.Gyr.to(u.yr)).values
tstart_skew = T0 - tsf_skew
sd_A_skew = data['A_sd_skew'].values
sd_tau_skew = (data['tau_sd_skew']*u.Gyr.to(u.yr)).values
sd_tsf_skew = (data['t_sf_sd_skew']*u.Gyr.to(u.yr)).values
sd_tstart_skew = sd_tsf_skew

# Define volume parameters
D_values = [(10, 8), (20, 18), (30, 28), (50, 48)]
redshifts = np.linspace(0, 10, 100)

# Core functions (unchanged)
def sfr_function(t, A, tau, V):
    return A * (t / tau**2) * np.exp(-t / tau) / V

def sfr_error(T_current, t_start, A, tau, sd_A, sd_tau, sd_t_start, V):
    t_Gyr = T_current - t_start
    t_years = t_Gyr * 1e9
    dSFR_dA = (t_years / tau**2) * np.exp(-t_years / tau) / V
    dSFR_dtau = A * np.exp(-t_years / tau) * ((2 * t_years / tau**3) - (t_years**2 / tau**4)) / V
    dSFR_dt = -A * np.exp(-t_years / tau) * ((1 / tau) - (t_years / tau**2)) / V
    sd_t_start_years = sd_t_start * 1e9
    return np.sqrt((dSFR_dA * sd_A)**2 + (dSFR_dtau * sd_tau)**2 + (dSFR_dt * sd_t_start_years)**2)

def compute_total_sfr_and_error(z_array, t_start, sd_t_start, A_values, tau_values, V, 
                              sd_A_values=None, sd_tau_values=None):
    total_sfr = np.zeros_like(z_array)
    total_sfr_error = np.zeros_like(z_array)
    for i, z in enumerate(z_array):
        T_current = cosmo.age(z).value
        t = T_current*10**9 - t_start 
        valid = t > 0
        t_years = t[valid]
        sfr_i = np.zeros_like(t)
        if np.any(valid):
            sfr_i[valid] = sfr_function(t_years, A_values[valid], tau_values[valid], V)
        total_sfr[i] = np.sum(sfr_i)
        if sd_A_values is not None and sd_tau_values is not None:
            sd_sfr_i = np.zeros_like(t)
            if np.any(valid):
                sd_sfr_i[valid] = sfr_error(T_current, t_start[valid], A_values[valid],
                                             tau_values[valid], sd_A_values[valid],
                                             sd_tau_values[valid], sd_t_start[valid], V)
            total_sfr_error[i] = np.sqrt(np.sum(sd_sfr_i**2))
    return total_sfr, total_sfr_error

def compute_log_sfr_and_error(total_sfr, total_sfr_error):
    log_sfr = np.log10(total_sfr)
    log_sfr_error = (total_sfr_error / total_sfr) / np.log(10)
    return log_sfr, log_sfr_error

def lilly_madau(z):
    return 0.015 * ((1 + z)**2.7) / (1 + ((1 + z)/2.9)**5.6)

def comoving_distance(z):
    return cosmo.comoving_distance(z).value

def ratio_error(total_sfr, total_sfr_error, sfrd_lilly_madau):
    return np.sqrt(np.abs(sfrd_lilly_madau/total_sfr**2*total_sfr_error**2))

# Process volumes
for D, D_minus_2 in D_values:
    for current_D in [D, D_minus_2]:
        V = (4/3 * np.pi * current_D**3)
        # From all the data, keep only the rows with D_v2<= current_D
        # get a fresh DataFrame slice for this volume
        sub = data[data['D_v2'] <= current_D]
        
        # now pull arrays from that slice
        A_uni     = sub['A_uni'].values
        tau_uni   = (sub['tau_uni']*u.Gyr.to(u.yr)).values
        tstart_uni= T0 - (sub['t_sf_uni']*u.Gyr.to(u.yr)).values
        sd_A_uni  = sub['A_sd_uni'].values
        sd_tau_uni= (sub['tau_sd_uni']*u.Gyr.to(u.yr)).values
        sd_tstart_uni = (sub['t_sf_sd_uni']*u.Gyr.to(u.yr)).values

        A_norm     = sub['A_np'].values
        tau_norm   = (sub['tau_np']*u.Gyr.to(u.yr)).values
        tstart_norm= T0 - (sub['t_sf_np']*u.Gyr.to(u.yr)).values
        sd_A_norm  = sub['A_sd_np'].values
        sd_tau_norm= (sub['tau_sd_np']*u.Gyr.to(u.yr)).values
        sd_tstart_norm = (sub['t_sf_sd_np']*u.Gyr.to(u.yr)).values

        A_skew     = sub['A_skew'].values
        tau_skew   = (sub['tau_skew']*u.Gyr.to(u.yr)).values
        tstart_skew= T0 - (sub['t_sf_skew']*u.Gyr.to(u.yr)).values
        sd_A_skew  = sub['A_sd_skew'].values
        sd_tau_skew= (sub['tau_sd_skew']*u.Gyr.to(u.yr)).values
        sd_tstart_skew = (sub['t_sf_sd_skew']*u.Gyr.to(u.yr)).values
        
        # Compute SFRs using pre-loaded arrays
        total_sfr_uni, total_sfr_error_uni = compute_total_sfr_and_error(
            redshifts, tstart_uni, sd_tstart_uni,
            A_uni, tau_uni, V,
            sd_A_uni, sd_tau_uni
        )
        
        total_sfr_norm, total_sfr_error_norm = compute_total_sfr_and_error(
            redshifts, tstart_norm, sd_tstart_norm,
            A_norm, tau_norm, V,
            sd_A_norm, sd_tau_norm
        )
        
        total_sfr_skew, total_sfr_error_skew = compute_total_sfr_and_error(
            redshifts, tstart_skew, sd_tstart_skew,
            A_skew, tau_skew, V,
            sd_A_skew, sd_tau_skew
        )

        # Compute log values
        log_sfr_uni, log_sfr_error_uni = compute_log_sfr_and_error(total_sfr_uni, total_sfr_error_uni)
        log_sfr_norm, log_sfr_error_norm = compute_log_sfr_and_error(total_sfr_norm, total_sfr_error_norm)
        log_sfr_skew, log_sfr_error_skew = compute_log_sfr_and_error(total_sfr_skew, total_sfr_error_skew)

        # Lilly-Madau calculations
        sfrd_lilly_madau = lilly_madau(redshifts)
        log_sfrd_lm = np.log10(sfrd_lilly_madau)

        # Define redshifts and corresponding lookback times
        redshift_ticks = np.array([0, 2, 4, 6, 8, 10])
        lookback_times = cosmo.lookback_time(redshift_ticks).value
        co_moving_distances = comoving_distance(redshift_ticks) / 1000

        # Main SFRD Plot
        plt.style.use('bmh')
        plt.rcParams["axes.prop_cycle"] = plt.cycler("color", plt.cm.viridis(np.linspace(0, 1, 4)))
        
        fig, ax1 = plt.subplots(figsize=(10, 6))
        ax2 = ax1.twiny()
        ax3 = ax1.twiny()
        ax3.spines['top'].set_position(('outward', 40))

        # Plot data
        ax1.errorbar(redshifts, log_sfr_uni, yerr=log_sfr_error_uni,
                    fmt='o-', capsize=3, label=r"Uniform Prior", markersize=4)
        ax1.errorbar(redshifts, log_sfr_norm, yerr=log_sfr_error_norm,
                    fmt='s-', capsize=3, label=r"Normal Prior", markersize=4)
        ax1.errorbar(redshifts, log_sfr_skew, yerr=log_sfr_error_skew,
                    fmt='v-', capsize=3, label=r"Skew Prior", markersize=4)
        ax1.plot(redshifts, log_sfrd_lm, 'k--', linewidth=2, label="Lilly-Madau (2014)")

        # Axes configuration
        ax1.set_xlabel("Redshift $z$")
        ax1.set_ylabel(r"$\log_{10}\left(\text{SFRD}\ \left[\text{M}_\odot \text{yr}^{-1} \text{Mpc}^{-3}\right]\right)$")
        ax1.set_title(f"SFRD Comparison (D = {current_D} Mpc)")
        ax1.axvline(x=1.86, color='r', linestyle='--', alpha=0.5)
        ax1.legend()
        ax1.grid(True)
        ax1.invert_xaxis()
        # Secondary x-axis (Lookback Time)
        ax2.set_xlabel("Lookback time [Gyr]")
        ax2.set_xticks(redshift_ticks)
        ax2.set_xticklabels([f"{t:.1f}" for t in lookback_times])

        # Tertiary x-axis (Co-moving Distance)
        ax3.set_xlabel("Co-moving distance [cGpc]")
        ax3.set_xticks(redshift_ticks)
        ax3.set_xticklabels([f"{d:.1f}" for d in co_moving_distances])

        # Invert all x-axes
        for ax in [ax1, ax2, ax3]:
            ax.set_xlim(10, 0)
            ax.set_xticks(redshift_ticks)
         
        # Invert all x-axes
        ax1.invert_xaxis()
        ax2.invert_xaxis()
        ax3.invert_xaxis()
         
        plt.tight_layout()
        plt.savefig(f'sfrd_plots/sfrd_comparison_D{current_D}.png', dpi=300)
        plt.close()

        # Residuals Plot
        residuals_uni = log_sfrd_lm - log_sfr_uni
        residuals_norm = log_sfrd_lm - log_sfr_norm
        residuals_skew = log_sfrd_lm - log_sfr_skew

        fig, ax1 = plt.subplots(figsize=(10, 6))
        ax2 = ax1.twiny()
        ax3 = ax1.twiny()
        ax3.spines['top'].set_position(('outward', 40))

        ax1.errorbar(redshifts, residuals_uni, yerr=log_sfr_error_uni,
                    fmt='o-', capsize=3, label="Uniform Prior", markersize=4)
        ax1.errorbar(redshifts, residuals_norm, yerr=log_sfr_error_norm,
                    fmt='s-', capsize=3, label="Normal Prior", markersize=4)
        ax1.errorbar(redshifts, residuals_skew, yerr=log_sfr_error_skew,
                    fmt='v-', capsize=3, label="Skew Prior", markersize=4)
        ax1.axhline(0, color='k', linestyle='--', alpha=0.7)

        ax1.set_xlabel("Redshift $z$")
        ax1.set_ylabel(r"$\Delta \log_{10}(\text{SFRD})$")
        ax1.set_title(f"SFRD Residuals (D = {current_D} Mpc)")
        ax1.legend()
        ax1.grid(True)
        ax1.invert_xaxis()

        # Configure all x-axes
        for ax in [ax1, ax2, ax3]:
            ax.set_xlim(10, 0)
            ax.set_xticks(redshift_ticks)
        ax2.set_xticklabels([f"{t:.1f}" for t in lookback_times])
        ax3.set_xticklabels([f"{d:.1f}" for d in co_moving_distances])
        ax2.set_xlabel("Lookback time [Gyr]")
        ax3.set_xlabel("Co-moving distance [cGpc]")

        # Invert all x-axes
        ax1.invert_xaxis()
        ax2.invert_xaxis()
        ax3.invert_xaxis()

        plt.tight_layout()
        plt.savefig(f'sfrd_plots/sfrd_residuals_D{current_D}.png', dpi=300)
        plt.close()

        # Ratio Plot
        ratio_uni = sfrd_lilly_madau / total_sfr_uni
        ratio_norm = sfrd_lilly_madau / total_sfr_norm
        ratio_skew = sfrd_lilly_madau / total_sfr_skew
        ratio_err_uni = ratio_error(total_sfr_uni, total_sfr_error_uni, sfrd_lilly_madau)
        ratio_err_norm = ratio_error(total_sfr_norm, total_sfr_error_norm, sfrd_lilly_madau)
        ratio_err_skew = ratio_error(total_sfr_skew, total_sfr_error_skew, sfrd_lilly_madau)

        fig, ax1 = plt.subplots(figsize=(10, 6))
        ax2 = ax1.twiny()
        ax3 = ax1.twiny()
        ax3.spines['top'].set_position(('outward', 40))

        ax1.errorbar(redshifts, ratio_uni, yerr=ratio_err_uni,
                    fmt='o-', capsize=3, label="Uniform Prior", markersize=4)
        ax1.errorbar(redshifts, ratio_norm, yerr=ratio_err_norm,
                    fmt='s-', capsize=3, label="Normal Prior", markersize=4)
        ax1.errorbar(redshifts, ratio_skew, yerr=ratio_err_skew,
                    fmt='v-', capsize=3, label="Skew Prior", markersize=4)
        ax1.axhline(1, color='k', linestyle='--', alpha=0.7)

        ax1.set_xlabel("Redshift $z$")
        ax1.set_ylabel(r"$\frac{\text{SFRD}_{\text{LM}}}{\text{SFRD}_{\text{Data}}}$")
        ax1.set_title(f"SFRD Ratio (D = {current_D} Mpc)")
        ax1.legend()
        ax1.grid(True)
        ax1.invert_xaxis()

        # Configure all x-axes
        for ax in [ax1, ax2, ax3]:
            ax.set_xlim(10, 0)
            ax.set_xticks(redshift_ticks)
        ax2.set_xticklabels([f"{t:.1f}" for t in lookback_times])
        ax3.set_xticklabels([f"{d:.1f}" for d in co_moving_distances])
        ax2.set_xlabel("Lookback time [Gyr]")
        ax3.set_xlabel("Co-moving distance [cGpc]")

        # Invert all x-axes
        ax1.invert_xaxis()
        ax2.invert_xaxis()
        ax3.invert_xaxis()

        plt.tight_layout()
        plt.savefig(f'sfrd_plots/sfrd_ratio_D{current_D}.png', dpi=300)
        plt.close()

        # z=1.86 calculations
        z_186 = np.array([1.86])
        sfr_uni_186, _ = compute_total_sfr_and_error(
            z_186, tstart_uni, sd_tstart_uni,
            A_uni, tau_uni, V,
            sd_A_uni, sd_tau_uni
        )
        sfr_norm_186, _ = compute_total_sfr_and_error(
            z_186, tstart_norm, sd_tstart_norm,
            A_norm, tau_norm, V,
            sd_A_norm, sd_tau_norm
        )
        sfr_skew_186, _ = compute_total_sfr_and_error(
            z_186, tstart_skew, sd_tstart_skew,
            A_skew, tau_skew, V,
            sd_A_skew, sd_tau_skew
        )
        sfrd_lm_186 = lilly_madau(1.86)
        
        print(f"\nD = {current_D} Mpc results at z=1.86:")
        print(f"Uniform Prior ratio: {sfrd_lm_186/sfr_uni_186[0]:.2f}")
        print(f"Normal Prior ratio: {sfrd_lm_186/sfr_norm_186[0]:.2f}")
        print(f"Skew Prior ratio: {sfrd_lm_186/sfr_skew_186[0]:.2f}")
