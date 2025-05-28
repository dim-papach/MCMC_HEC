"""
Prepare galaxy data for hierarchical MCMC modeling.
Creates galaxy groups based on physical properties and HECATE classifications.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.mixture import GaussianMixture
import warnings
warnings.filterwarnings('ignore')

def load_and_clean_data(filepath='tables/Hec_50.csv'):
    """Load HECATE data and perform basic cleaning."""
    print("Loading HECATE catalog...")
    df = pd.read_csv(filepath)
    
    # Filter out bad data
    df = df[
        (df['logSFR_HECv2'].notna()) & 
        (df['logM_star_HECv2'].notna()) &
        (df['logM_star_HECv2'] > 5) &  # Reasonable mass range
        (df['logM_star_HECv2'] < 12) &
        (df['D_v2'] > 0)
    ].copy()
    
    # Calculate derived quantities
    df['sSFR'] = 10**(df['logSFR_HECv2'] - df['logM_star_HECv2'])
    df['log_sSFR'] = np.log10(df['sSFR'])
    
    # Calculate colors where photometry exists
    df['U_G'] = df['U_v2'] - df['G_v2']
    df['G_R'] = df['G_v2'] - df['R_v2']
    df['R_I'] = df['R_v2'] - df['I_v2']
    
    print(f"Loaded {len(df)} galaxies with complete SFR and mass data")
    return df


def create_mass_groups(df, n_groups=3):
    """Create mass-based galaxy groups."""
    print("\nCreating mass groups...")
    
    # Define mass bins based on physical understanding
    # Low mass: < 10^9 M_sun
    # Intermediate: 10^9 - 10^10.5 M_sun  
    # High mass: > 10^10.5 M_sun
    
    mass_bins = [5, 9, 10.5, 12]
    df['mass_group'] = pd.cut(
        df['logM_star_HECv2'], 
        bins=mass_bins,
        labels=['low_mass', 'intermediate_mass', 'high_mass']
    )
    
    # Print statistics
    print("Mass group distribution:")
    print(df['mass_group'].value_counts())
    
    return df


def create_main_sequence_groups(df):
    """
    Create groups based on position relative to the star-forming main sequence.
    The main sequence is the tight correlation between SFR and stellar mass.
    """
    print("\nCreating main sequence groups...")
    
    # Define the main sequence relation (from literature)
    # log(SFR) = α * log(M*) + β
    # Typical values: α ≈ 0.8-1.0, β ≈ -8.5 to -9.5
    
    # Fit the main sequence to star-forming galaxies only
    sf_galaxies = df[df['Activity_class'] == 0]  # Star-forming from HECATE
    
    if len(sf_galaxies) > 100:
        # Robust linear fit to find main sequence
        from scipy import stats
        slope, intercept, _, _, _ = stats.linregress(
            sf_galaxies['logM_star_HECv2'], 
            sf_galaxies['logSFR_HECv2']
        )
        
        # Calculate distance from main sequence
        df['MS_expected_logSFR'] = slope * df['logM_star_HECv2'] + intercept
        df['delta_MS'] = df['logSFR_HECv2'] - df['MS_expected_logSFR']
        
        # Define groups based on distance from MS
        # Starburst: > 0.3 dex above MS
        # Main sequence: within ±0.3 dex
        # Below MS: < -0.3 dex below MS
        
        df['ms_group'] = pd.cut(
            df['delta_MS'],
            bins=[-10, -0.3, 0.3, 10],
            labels=['below_ms', 'main_sequence', 'starburst']
        )
        
        print(f"Main sequence fit: log(SFR) = {slope:.2f} * log(M*) + {intercept:.2f}")
    else:
        # Fallback if not enough star-forming galaxies
        df['ms_group'] = 'unknown'
    
    print("Main sequence group distribution:")
    print(df['ms_group'].value_counts())
    
    return df


def create_color_groups(df, n_groups=3):
    """
    Create groups based on optical colors.
    Red galaxies are typically quiescent, blue galaxies are star-forming.
    """
    print("\nCreating color groups...")
    
    # Use galaxies with good photometry
    color_data = df[
        df['G_R'].notna() & 
        (df['Optical_phot_QF_v2'] < 200000)  # Good quality photometry
    ].copy()
    
    if len(color_data) > 100:
        # Use Gaussian Mixture Model to find color groups
        gmm = GaussianMixture(n_components=n_groups, random_state=42)
        
        # Use G-R color as primary discriminant
        color_features = color_data[['G_R']].values
        color_groups = gmm.fit_predict(color_features)
        
        # Assign group names based on mean colors
        group_means = []
        for i in range(n_groups):
            mean_color = color_data[color_groups == i]['G_R'].mean()
            group_means.append((i, mean_color))
        
        # Sort by color (bluest to reddest)
        group_means.sort(key=lambda x: x[1])
        
        # Create mapping
        group_names = ['blue_cloud', 'green_valley', 'red_sequence'][:n_groups]
        group_mapping = {old_id: group_names[new_id] 
                        for new_id, (old_id, _) in enumerate(group_means)}
        
        # Apply to full dataset
        df['color_group'] = 'unknown'
        color_data['color_group'] = [group_mapping[g] for g in color_groups]
        df.loc[color_data.index, 'color_group'] = color_data['color_group']
    else:
        df['color_group'] = 'unknown'
    
    print("Color group distribution:")
    print(df['color_group'].value_counts())
    
    return df


def create_morphology_proxy_groups(df):
    """
    Create morphological proxy groups based on observable properties.
    Since HECATE doesn't have morphological types, we use proxies:
    - High sSFR + blue colors → likely late-type (spiral)
    - Low sSFR + red colors → likely early-type (elliptical)
    """
    print("\nCreating morphology proxy groups...")
    
    # Initialize with unknown
    df['morph_proxy'] = 'unknown'
    
    # Define criteria for morphological proxies
    # Early-type: low sSFR, red colors, often passive
    early_mask = (
        (df['log_sSFR'] < -11) & 
        (df['G_R'] > 0.6) & 
        (df['Activity_class'].isin([4]))  # Passive
    )
    
    # Late-type: high sSFR, blue colors, star-forming
    late_mask = (
        (df['log_sSFR'] > -10.5) & 
        (df['G_R'] < 0.5) & 
        (df['Activity_class'] == 0)  # Star-forming
    )
    
    # Intermediate: everything else with data
    has_data = df['log_sSFR'].notna() & df['G_R'].notna()
    
    df.loc[early_mask, 'morph_proxy'] = 'early_type_proxy'
    df.loc[late_mask, 'morph_proxy'] = 'late_type_proxy'
    df.loc[has_data & ~early_mask & ~late_mask, 'morph_proxy'] = 'intermediate_proxy'
    
    print("Morphology proxy distribution:")
    print(df['morph_proxy'].value_counts())
    
    return df


def create_hierarchical_groups(df):
    """
    Create hierarchical group structure combining all classifications.
    This will be used for the hierarchical model.
    """
    print("\nCreating hierarchical group structure...")
    
    # Create a unique group ID combining all classifications
    df['hierarchical_group'] = (
        df['Activity_class'].astype(str) + '_' +
        df['mass_group'].astype(str) + '_' +
        df['color_group'].astype(str) + '_' +
        df['morph_proxy'].astype(str) + '_' +
        df['ms_group'].astype(str)
    )
    
    # Create numeric group ID for Stan
    unique_groups = df['hierarchical_group'].unique()
    group_mapping = {group: i+1 for i, group in enumerate(unique_groups)}
    df['group_id'] = df['hierarchical_group'].map(group_mapping)
    
    # Count galaxies per group
    group_counts = df['hierarchical_group'].value_counts()
    print(f"\nCreated {len(unique_groups)} hierarchical groups")
    print("\nTop 10 groups by galaxy count:")
    print(group_counts.head(10))
    
    # Save group mapping
    group_info = pd.DataFrame({
        'group_name': unique_groups,
        'group_id': [group_mapping[g] for g in unique_groups],
        'n_galaxies': [group_counts.get(g, 0) for g in unique_groups]
    })
    group_info.to_csv('tables/hierarchical_groups.csv', index=False)
    
    return df, group_info


def prepare_stan_data(df):
    """Prepare data for Stan hierarchical model."""
    print("\nPreparing data for Stan...")
    
    # Filter to galaxies with all required data
    stan_df = df[
        df['logSFR_HECv2'].notna() & 
        df['logM_star_HECv2'].notna() &
        df['group_id'].notna() &
        (df['group_id'] > 0)
    ].copy()
    
    # Calculate total masses and SFRs for Stan
    stan_df['M_total'] = 10**stan_df['logM_star_HECv2']
    stan_df['logSFR_total'] = stan_df['logSFR_HECv2']
    
    # Create Stan data dictionary
    stan_data = {
        'N': len(stan_df),
        'N_groups': stan_df['group_id'].nunique(),
        'logSFR_total': stan_df['logSFR_total'].values,
        'M_star': stan_df['M_total'].values,
        'group': stan_df['group_id'].values.astype(int),
        'ID': stan_df.index.values
    }
    
    # Save processed data
    stan_df.to_csv('tables/hierarchical_data.csv', index=False)
    
    # Save Stan data for R
    import json
    with open('tables/stan_data.json', 'w') as f:
        # Convert numpy arrays to lists for JSON serialization
        stan_data_json = {k: v.tolist() if isinstance(v, np.ndarray) else v 
                         for k, v in stan_data.items()}
        json.dump(stan_data_json, f, indent=2)
    
    print(f"Prepared {stan_data['N']} galaxies in {stan_data['N_groups']} groups for Stan")
    
    return stan_df, stan_data


def create_diagnostic_plots(df):
    """Create diagnostic plots for the groupings."""
    print("\nCreating diagnostic plots...")
    
    fig, axes = plt.subplots(2, 3, figsize=(30, 10))
    
    # 1. SFR-Mass diagram with activity classes
    ax = axes[0, 0]
    for activity in df['Activity_class'].unique():
        if pd.notna(activity):
            mask = df['Activity_class'] == activity
            activity_names = {0: 'SF', 1: 'AGN', 2: 'LINER', 3: 'Comp', 4: 'Passive'}
            ax.scatter(df[mask]['logM_star_HECv2'], df[mask]['logSFR_HECv2'], 
                      alpha=0.5, s=10, label=activity_names.get(activity, f'Class {activity}'))
    ax.set_xlabel('log(M*)')
    ax.set_ylabel('log(SFR)')
    ax.legend()
    ax.set_title('Activity Classes')
    
    # 2. Color-magnitude diagram
    ax = axes[0, 1]
    color_mask = df['G_R'].notna()
    scatter = ax.scatter(df[color_mask]['logM_star_HECv2'], df[color_mask]['G_R'], 
                        c=df[color_mask]['log_sSFR'], cmap='viridis', s=10, alpha=0.5)
    ax.set_xlabel('log(M*)')
    ax.set_ylabel('G-R color')
    ax.set_title('Color-Magnitude Diagram')
    plt.colorbar(scatter, ax=ax, label='log(sSFR)')
    
    # 3. Main sequence groups
    ax = axes[0, 2]
    for group in df['ms_group'].unique():
        if pd.notna(group) and group != 'unknown':
            mask = df['ms_group'] == group
            ax.scatter(df[mask]['logM_star_HECv2'], df[mask]['logSFR_HECv2'],
                      alpha=0.5, s=10, label=group)
    ax.set_xlabel('log(M*)')
    ax.set_ylabel('log(SFR)')
    ax.legend()
    ax.set_title('Main Sequence Groups')
    
    # 4. Mass groups
    ax = axes[1, 0]
    df['mass_group'].value_counts().plot(kind='bar', ax=ax)
    ax.set_title('Mass Group Distribution')
    ax.set_ylabel('Count')
    
    # 5. Color groups
    ax = axes[1, 1]
    color_groups = df[df['color_group'] != 'unknown']['color_group'].value_counts()
    if len(color_groups) > 0:
        color_groups.plot(kind='bar', ax=ax)
    ax.set_title('Color Group Distribution')
    ax.set_ylabel('Count')
    
    # 6. Hierarchical groups (top 15)
    ax = axes[1, 2]
    top_groups = df['hierarchical_group'].value_counts().head(15)
    top_groups.plot(kind='barh', ax=ax)
    ax.set_title('Top 15 Hierarchical Groups')
    ax.set_xlabel('Count')
    
    plt.tight_layout()
    plt.savefig('diagnostic_plots.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("Saved diagnostic plots to diagnostic_plots.png")


def main():
    """Main workflow for preparing hierarchical data."""
    
    # Load and clean data
    df = load_and_clean_data()
    
    # Create individual group classifications
    df = create_mass_groups(df)
    df = create_main_sequence_groups(df)
    df = create_color_groups(df)
    df = create_morphology_proxy_groups(df)
    
    # Create hierarchical groups
    df, group_info = create_hierarchical_groups(df)
    
    # Prepare Stan data
    stan_df, stan_data = prepare_stan_data(df)
    
    # Create diagnostic plots
    create_diagnostic_plots(df)
    
    print("\n=== Summary ===")
    print(f"Total galaxies processed: {len(df)}")
    print(f"Galaxies with complete data for Stan: {stan_data['N']}")
    print(f"Number of hierarchical groups: {stan_data['N_groups']}")
    print("\nFiles created:")
    print("- tables/hierarchical_data.csv (processed galaxy data)")
    print("- tables/hierarchical_groups.csv (group definitions)")
    print("- tables/stan_data.json (data for Stan)")
    print("- diagnostic_plots.png (visualization)")
    
    return df, stan_data, group_info


if __name__ == "__main__":
    df, stan_data, group_info = main()