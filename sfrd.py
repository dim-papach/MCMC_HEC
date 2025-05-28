import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from astropy.cosmology import Planck18 as cosmo
import astropy.units as u
import os

# Set working directory to the script location
script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)

# Create output directory
os.makedirs('sfrd_plots', exist_ok=True)

# Global constants
T0 = 13.8 * 1e9  # Age of the Universe in years
D_VALUES = [(10, 5), (20, 15), (30, 25), (50, 45)]
REDSHIFTS = np.linspace(0, 10, 100)
REDSHIFT_TICKS = np.array([0, 2, 4, 6, 8, 10])


def sfr_function(t, A, tau, V):
    """Calculate the star formation rate."""
    return A * (t / tau**2) * np.exp(-t / tau) / V


def sfr_error(T_current, t_start, A, tau, sd_A, sd_tau, sd_t_start, V):
    """Calculate the error in star formation rate using error propagation."""
    t_Gyr = T_current - t_start
    t_years = t_Gyr * 1e9
    
    dSFR_dA = (t_years / tau**2) * np.exp(-t_years / tau) / V
    dSFR_dtau = A * np.exp(-t_years / tau) * ((2 * t_years / tau**3) - (t_years**2 / tau**4)) / V
    dSFR_dt = -A * np.exp(-t_years / tau) * ((1 / tau) - (t_years / tau**2)) / V
    
    sd_t_start_years = sd_t_start * 1e9
    return np.sqrt((dSFR_dA * sd_A)**2 + (dSFR_dtau * sd_tau)**2 + (dSFR_dt * sd_t_start_years)**2)


def compute_total_sfr_and_error(z_array, t_start, sd_t_start, A_values, tau_values, V, 
                              sd_A_values=None, sd_tau_values=None):
    """Compute total SFR, error, and track valid galaxies for each redshift."""
    total_sfr = np.zeros_like(z_array)
    total_sfr_error = np.zeros_like(z_array)
    n_valid_galaxies = np.zeros_like(z_array, dtype=int)
    
    for i, z in enumerate(z_array):
        T_current = cosmo.age(z).to(u.yr).value
        t = T_current - t_start 
        valid = t > 0
        n_valid_galaxies[i] = np.sum(valid)
        
        if np.any(valid):
            t_years = t[valid]
            sfr_i = np.zeros_like(t)
            sfr_i[valid] = sfr_function(t_years, A_values[valid], tau_values[valid], V)
            total_sfr[i] = np.sum(sfr_i)
            
            if sd_A_values is not None and sd_tau_values is not None:
                sd_sfr_i = np.zeros_like(t)
                sd_sfr_i[valid] = sfr_error(
                    T_current, t_start[valid], A_values[valid],
                    tau_values[valid], sd_A_values[valid],
                    sd_tau_values[valid], sd_t_start[valid], V
                )
                total_sfr_error[i] = np.sqrt(np.sum(sd_sfr_i**2))
    
    return total_sfr, total_sfr_error, n_valid_galaxies


def compute_log_sfr_and_error(total_sfr, total_sfr_error):
    """Convert SFR and its error to log scale."""
    log_sfr = np.log10(total_sfr)
    log_sfr_error = (total_sfr_error / total_sfr) / np.log(10)
    return log_sfr, log_sfr_error


def lilly_madau(z):
    """Lilly-Madau SFRD parameterization."""
    return 0.015 * ((1 + z)**2.7) / (1 + ((1 + z)/2.9)**5.6)


def comoving_distance(z):
    """Calculate comoving distance for a given redshift."""
    return cosmo.comoving_distance(z).value


def ratio_error(total_sfr, total_sfr_error, sfrd_lilly_madau):
    """Calculate error in the ratio of SFRD values."""
    return np.sqrt(np.abs(sfrd_lilly_madau/total_sfr**2*total_sfr_error**2))


def extract_parameters(df, D):
    """Extract and filter parameters for a given diameter."""
    # Filter data by diameter
    sub = df[df['D_v2'] <= D].copy()
    
    # Initialize parameter dictionary
    params = {}
    
    # Extract parameters for each prior type
    for prior in ['uni', 'np', 'skew']:
        # Extract parameters - corrected column names based on combined_columns.txt
        params[f'A_{prior}'] = sub[f'A_{prior}'].values
        params[f'tau_{prior}'] = (sub[f'tau_{prior}'] * 1e9).values  # Convert from Gyr to years
        params[f'tstart_{prior}'] = T0 - (sub[f't_sf_{prior}'] * 1e9).values  # Convert to years and calculate start time
        params[f'sd_A_{prior}'] = sub[f'A_sd'].values if f'A_sd_{prior}' not in sub.columns else sub[f'A_sd_{prior}'].values
        params[f'sd_tau_{prior}'] = sub[f'tau_sd'].values if f'tau_sd_{prior}' not in sub.columns else sub[f'tau_sd_{prior}'].values
        params[f'sd_tstart_{prior}'] = (sub[f't_sf_sd'].values if f't_sf_sd_{prior}' not in sub.columns else sub[f't_sf_sd_{prior}'].values) * 1e9
        
        # Create predicted SFR for filtering if not available
        if f'logSFR_today_pred_{prior}' in sub.columns:
            pred_sfr_col = f'logSFR_today_pred_{prior}'
        elif 'logSFR_today_pred_np' in sub.columns and prior == 'np':
            pred_sfr_col = 'logSFR_today_pred_np'
        else:
            # If no prediction column, skip filtering
            continue
        
        # Filter by log difference criterion if we have both observed and predicted SFR
        if 'logSFR_total' in sub.columns and pred_sfr_col in sub.columns:
            mask = np.abs(sub[pred_sfr_col] - sub['logSFR_total']) < 10
        else:
            # If no filtering possible, use all data
            mask = np.ones(len(sub), dtype=bool)
        
        # Apply mask to parameters
        for param_name in [f'A_{prior}', f'tau_{prior}', f'tstart_{prior}', 
                          f'sd_A_{prior}', f'sd_tau_{prior}', f'sd_tstart_{prior}']:
            if param_name in params:
                params[param_name][~mask] = np.nan
    
    return params


def plot_sfrd_comparison(redshifts, log_sfr_data, log_sfr_errors, log_sfrd_lm, D, 
                        lookback_times, co_moving_distances, ir, uv):
    """Create SFRD comparison plot."""
    fig, ax1 = plt.subplots(figsize=(10, 6))
    ax2 = ax1.twiny()
    ax3 = ax1.twiny()
    ax3.spines['top'].set_position(('outward', 40))

    # Plot data for each prior
    markers = ['o-', 's-', 'v-']
    prior_labels = {'uni': 'Uniform', 'np': 'Normal', 'skew': 'Skewed'}
    for i, prior in enumerate(['uni', 'np', 'skew']):
        if prior in log_sfr_data:
            ax1.errorbar(
                redshifts, log_sfr_data[prior], yerr=log_sfr_errors[prior],
                fmt=markers[i], capsize=3, label=f"{prior_labels[prior]} Prior", markersize=4
            )
    
    # Plot Lilly-Madau SFRD from ir and uv data
    if ir is not None and not ir.empty:
        log_v = ir["log_v"]
        log_v_upper = ir["log_v_upper_error"]
        log_v_lower = ir["log_v_lower_error"]
        z_min = ir['z_min']
        z_max = ir['z_max']
        z = np.mean([z_min, z_max], axis=0)
        z_error = (z_max- z_min).replace(0, 0.001) / 2  # Avoid division by zero
        ax1.errorbar(z, log_v,xerr=z_error , yerr=[log_v_lower, log_v_upper], fmt='o', capsize=3, label='MD14 (IR)', markersize=4, color='orange')

    if uv is not None and not uv.empty:
        log_v = uv["log_v"]
        log_v_upper = uv["log_v_upper_error"]
        log_v_lower = uv["log_v_lower_error"]
        z_min = uv['z_min']
        z_max = uv['z_max']
        z = np.mean([z_min, z_max], axis=0)
        z_error = (z_max - z_min).replace(0, 0.001) / 2  # Avoid division by zero
        ax1.errorbar(z, log_v,xerr=z_error , yerr=[log_v_lower, log_v_upper], fmt='o', capsize=3, label='MD14 (UV)', markersize=4, color='red')

    # Plot Lilly-Madau reference
    ax1.plot(redshifts, log_sfrd_lm, 'k--', linewidth=2, label="Lilly-Madau (2014)")

    # Configure axes
    ax1.set_xlabel("Redshift $z$")
    ax1.set_ylabel(r"$\log_{10}\left(\text{SFRD}\ \left[\text{M}_\odot \text{yr}^{-1} \text{Mpc}^{-3}\right]\right)$")
    ax1.set_title(f"SFRD Comparison (D = {D} Mpc)")
    ax1.axvline(x=1.86, color='r', linestyle='--', alpha=0.5)
    ax1.legend()
    ax1.grid(True)
    
    # Secondary and tertiary x-axes
    configure_multiple_x_axes(ax1, ax2, ax3, lookback_times, co_moving_distances)
    
    plt.tight_layout()
    plt.savefig(f'sfrd_plots/sfrd_comparison_D{D}.png', dpi=300)
    plt.close()


def plot_residuals(redshifts, log_sfrd_lm, log_sfr_data, log_sfr_errors, D, 
                 lookback_times, co_moving_distances):
    """Create SFRD residuals plot."""
    fig, ax1 = plt.subplots(figsize=(10, 6))
    ax2 = ax1.twiny()
    ax3 = ax1.twiny()
    ax3.spines['top'].set_position(('outward', 40))

    # Calculate and plot residuals for each prior
    markers = ['o-', 's-', 'v-']
    prior_labels = {'uni': 'Uniform', 'np': 'Normal', 'skew': 'Skewed'}
    for i, prior in enumerate(['uni', 'np', 'skew']):
        if prior in log_sfr_data:
            residuals = log_sfrd_lm - log_sfr_data[prior]
            ax1.errorbar(
                redshifts, residuals, yerr=log_sfr_errors[prior],
                fmt=markers[i], capsize=3, label=f"{prior_labels[prior]} Prior", markersize=4
            )
    
    ax1.axhline(0, color='k', linestyle='--', alpha=0.7)
    ax1.axvline(x=1.86, color='r', linestyle='--', alpha=0.5)
    ax1.set_xlabel("Redshift $z$")
    ax1.set_ylabel(r"$\Delta \log_{10}(\text{SFRD})$")
    ax1.set_title(f"SFRD Residuals (D = {D} Mpc)")
    ax1.legend()
    ax1.grid(True)
    
    # Configure axes
    configure_multiple_x_axes(ax1, ax2, ax3, lookback_times, co_moving_distances)
    
    plt.tight_layout()
    plt.savefig(f'sfrd_plots/sfrd_residuals_D{D}.png', dpi=300)
    plt.close()


def plot_ratio(redshifts, sfrd_lilly_madau, total_sfr_data, total_sfr_errors, D, 
             lookback_times, co_moving_distances):
    """Create SFRD ratio plot."""
    fig, ax1 = plt.subplots(figsize=(10, 6))
    ax2 = ax1.twiny()
    ax3 = ax1.twiny()
    ax3.spines['top'].set_position(('outward', 40))

    # Calculate and plot ratios for each prior
    markers = ['o-', 's-', 'v-']
    prior_labels = {'uni': 'Uniform', 'np': 'Normal', 'skew': 'Skewed'}
    for i, prior in enumerate(['uni', 'np', 'skew']):
        if prior in total_sfr_data:
            # Calculate ratio and its error avoiding division by zero
            ratio = sfrd_lilly_madau / total_sfr_data[prior]
            ratio[total_sfr_data[prior] == 0] = np.nan
            
            ratio_err = ratio_error(total_sfr_data[prior], total_sfr_errors[prior], sfrd_lilly_madau)
            ax1.errorbar(
                redshifts, ratio, yerr=ratio_err,
                fmt=markers[i], capsize=3, label=f"{prior_labels[prior]} Prior", markersize=4
            )
    
    ax1.axhline(1, color='k', linestyle='--', alpha=0.7)
    ax1.axvline(x=1.86, color='r', linestyle='--', alpha=0.5)
    ax1.set_xlabel("Redshift $z$")
    ax1.set_ylabel(r"$\frac{\text{SFRD}_{\text{LM}}}{\text{SFRD}_{\text{Data}}}$")
    ax1.set_ylim(-1, 5)
    ax1.set_title(f"SFRD Ratio (D = {D} Mpc)")
    ax1.legend()
    ax1.grid(True)
    
    # Configure axes
    configure_multiple_x_axes(ax1, ax2, ax3, lookback_times, co_moving_distances)
    
    plt.tight_layout()
    plt.savefig(f'sfrd_plots/sfrd_ratio_D{D}.png', dpi=300)
    plt.close()


def plot_valid_galaxies(redshifts, total_sfr_data, n_valid_data, D):
    """Plot SFRD vs. number of valid galaxies."""
    fig, ax1 = plt.subplots(figsize=(10, 6))
    ax2 = ax1.twinx()
    
    # SFRD (left axis) - use 'np' prior if available
    if 'np' in total_sfr_data:
        ax1.plot(redshifts, np.log10(total_sfr_data['np']), 'b-', label='SFRD (Normal Prior)')
        ax2.plot(redshifts, n_valid_data['np'], 'r--', label='Valid Galaxies (t > 0)')
    elif 'uni' in total_sfr_data:
        ax1.plot(redshifts, np.log10(total_sfr_data['uni']), 'b-', label='SFRD (Uniform Prior)')
        ax2.plot(redshifts, n_valid_data['uni'], 'r--', label='Valid Galaxies (t > 0)')
    
    ax1.set_xlabel("Redshift")
    ax1.set_ylabel(r"$\log_{10}(\text{SFRD})$ [M$_\odot$ yr$^{-1}$ Mpc$^{-3}$]")
    ax2.set_ylabel("Number of Valid Galaxies")
    
    ax1.legend(loc='upper left')
    ax2.legend(loc='upper right')
    plt.title(f"SFRD vs. Valid Galaxies (D = {D} Mpc)")
    plt.savefig(f"sfrd_plots/valid_galaxies_D{D}.png")
    plt.close()


def configure_multiple_x_axes(ax1, ax2, ax3, lookback_times, co_moving_distances):
    """Configure multiple x-axes with proper labels and ticks."""
    # Configure all x-axes
    for ax in [ax1, ax2, ax3]:
        ax.set_xlim(10, 0)
        ax.set_xticks(REDSHIFT_TICKS)
        ax.invert_xaxis()
    
    # Add labels for secondary axes
    ax2.set_xticklabels([f"{t:.1f}" for t in lookback_times])
    ax3.set_xticklabels([f"{d:.1f}" for d in co_moving_distances])
    ax2.set_xlabel("Lookback time [Gyr]")
    ax3.set_xlabel("Co-moving distance [cGpc]")


def process_volume(D, data, ir, uv):
    """Process data for a specific volume diameter."""
    V = (4/3 * np.pi * D**3)
    
    # Extract parameters
    params = extract_parameters(data, D)
    
    # Compute SFRs for each prior
    results = {}
    available_priors = []
    
    for prior in ['uni', 'np', 'skew']:
        # Check if parameters exist for this prior
        required_params = [f'tstart_{prior}', f'sd_tstart_{prior}', f'A_{prior}', f'tau_{prior}']
        if all(f'{param}' in params for param in required_params):
            available_priors.append(prior)
            
            total_sfr, total_sfr_error, n_valid = compute_total_sfr_and_error(
                REDSHIFTS, 
                params[f'tstart_{prior}'], params[f'sd_tstart_{prior}'],
                params[f'A_{prior}'], params[f'tau_{prior}'], V,
                params.get(f'sd_A_{prior}'), params.get(f'sd_tau_{prior}')
            )
            
            # Store results
            results[f'total_sfr_{prior}'] = total_sfr
            results[f'total_sfr_error_{prior}'] = total_sfr_error
            results[f'n_valid_{prior}'] = n_valid
            
            # Compute log values
            log_sfr, log_sfr_error = compute_log_sfr_and_error(total_sfr, total_sfr_error)
            results[f'log_sfr_{prior}'] = log_sfr
            results[f'log_sfr_error_{prior}'] = log_sfr_error
    
    if not available_priors:
        print(f"Warning: No valid priors found for D = {D} Mpc")
        return
    
    # Lilly-Madau calculations
    sfrd_lilly_madau = lilly_madau(REDSHIFTS)
    log_sfrd_lm = np.log10(sfrd_lilly_madau)
    
    # Prepare data for plotting
    lookback_times = cosmo.lookback_time(REDSHIFT_TICKS).value
    co_moving_distances = comoving_distance(REDSHIFT_TICKS) / 1000
    
    # Group data for plotting
    log_sfr_data = {}
    log_sfr_errors = {}
    total_sfr_data = {}
    total_sfr_errors = {}
    n_valid_data = {}
    
    for prior in available_priors:
        log_sfr_data[prior] = results[f'log_sfr_{prior}']
        log_sfr_errors[prior] = results[f'log_sfr_error_{prior}']
        total_sfr_data[prior] = results[f'total_sfr_{prior}']
        total_sfr_errors[prior] = results[f'total_sfr_error_{prior}']
        n_valid_data[prior] = results[f'n_valid_{prior}']
    
    # Create plots
    plt.style.use('bmh')
    plt.rcParams["axes.prop_cycle"] = plt.cycler("color", plt.cm.viridis(np.linspace(0, 1, 4)))
    
    # Main SFRD comparison plot
    plot_sfrd_comparison(
        REDSHIFTS, log_sfr_data, log_sfr_errors, log_sfrd_lm, D, 
        lookback_times, co_moving_distances, ir, uv
    )
    
    # Residuals plot
    plot_residuals(
        REDSHIFTS, log_sfrd_lm, log_sfr_data, log_sfr_errors, D,
        lookback_times, co_moving_distances
    )
    
    # Ratio plot
    plot_ratio(
        REDSHIFTS, sfrd_lilly_madau, total_sfr_data, total_sfr_errors, D,
        lookback_times, co_moving_distances
    )
    
    # Valid galaxies plot
    plot_valid_galaxies(REDSHIFTS, total_sfr_data, n_valid_data, D)
    
    # Calculate z=1.86 values
    z_186 = np.array([1.86])
    sfrd_lm_186 = lilly_madau(1.86)
    
    # Calculate SFR at z=1.86 for each available prior
    ratios_186 = {}
    for prior in available_priors:
        sfr_186, _, _ = compute_total_sfr_and_error(
            z_186, 
            params[f'tstart_{prior}'], params[f'sd_tstart_{prior}'],
            params[f'A_{prior}'], params[f'tau_{prior}'], V,
            params.get(f'sd_A_{prior}'), params.get(f'sd_tau_{prior}')
        )
        if sfr_186[0] > 0:
            ratios_186[prior] = sfrd_lm_186 / sfr_186[0]
        else:
            ratios_186[prior] = np.nan
    
    # Print results at z=1.86
    print(f"\nD = {D} Mpc results at z=1.86:")
    prior_labels = {'uni': 'Uniform', 'np': 'Normal', 'skew': 'Skewed'}
    for prior in available_priors:
        if not np.isnan(ratios_186[prior]):
            print(f"{prior_labels[prior]} Prior ratio: {ratios_186[prior]:.2f}")
        else:
            print(f"{prior_labels[prior]} Prior ratio: undefined (SFR = 0)")
    
    # Print when n_valid_galaxies is 0
    for prior in available_priors:
        for i, count in enumerate(results[f'n_valid_{prior}']):
            if count == 0:
                print(f"At redshift {REDSHIFTS[i]:.2f}, no valid galaxies (t > 0) for D = {D} Mpc ({prior_labels[prior]} prior).")
                break
        else:
            print(f"All redshifts have valid galaxies (t > 0) for D = {D} Mpc ({prior_labels[prior]} prior).")


def main():
    """Main function to process all volumes."""
    # Try to load the main combined data file
    data_files = [
        'tables/MCMC_results_12.csv',  # Original expected file
        'combined_columns.csv',         # Based on provided column info
        'tables/combined_results.csv'   # Alternative location
    ]
    
    data = None
    for file_path in data_files:
        try:
            if os.path.exists(file_path):
                print(f"Loading data from: {file_path}")
                data = pd.read_csv(file_path)
                break
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
            continue
    
    if data is None:
        print("Error: Could not load main data file. Please check file paths.")
        print("Expected files:", data_files)
        return
    
    # Try to load reference data
    ir = None
    uv = None
    
    try:
        if os.path.exists('tables/LM_ir.csv'):
            ir = pd.read_csv('tables/LM_ir.csv')
    except Exception as e:
        print(f"Warning: Could not load IR reference data: {e}")
    
    try:
        if os.path.exists('tables/LM_uv.csv'):
            uv = pd.read_csv('tables/LM_uv.csv')
    except Exception as e:
        print(f"Warning: Could not load UV reference data: {e}")
    
    print(f"Data shape: {data.shape}")
    print(f"Available columns: {list(data.columns)}")
    
    # Process each diameter combination
    for D_pair in D_VALUES:
        for D in D_pair:
            print(f"\nProcessing D = {D} Mpc...")
            try:
                process_volume(D, data, ir, uv)
            except Exception as e:
                print(f"Error processing D = {D} Mpc: {e}")
                continue


if __name__ == "__main__":
    main()
