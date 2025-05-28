#!/usr/bin/env python3
"""
Modified MCMC Results Combiner
First combines results with original HECATE data per folder,
then filters based on |logSFR_total - logSFR_today_pred_*| < 1,
and finally combines everything together.
"""

import pandas as pd
import numpy as np
import os
import glob
from pathlib import Path
import re
import sys

def find_combined_summary_files(base_dir="."):
    """Find all combined_summary_*.csv files in the directory structure."""
    print(f"Searching for files in: {os.path.abspath(base_dir)}")
    
    # Try multiple search patterns
    patterns = [
        os.path.join(base_dir, "**/combined_summary_*.csv"),
        os.path.join(base_dir, "*/*/combined_summary_*.csv"),
        os.path.join(base_dir, "Hec_*/*/combined_summary_*.csv"),
    ]
    
    files = []
    for pattern in patterns:
        found = glob.glob(pattern, recursive=True)
        files.extend(found)
        if found:
            print(f"  Pattern '{pattern}' found {len(found)} files")
    
    # Remove duplicates
    files = list(set(files))
    
    # Also try os.walk as fallback
    if not files:
        print("  Trying os.walk method...")
        for root, dirs, filenames in os.walk(base_dir):
            for filename in filenames:
                if filename.startswith('combined_summary_') and filename.endswith('.csv'):
                    filepath = os.path.join(root, filename)
                    files.append(filepath)
                    print(f"    Found: {filepath}")
    
    return files

def extract_model_and_dataset(filepath):
    """Extract model name and dataset name from file path."""
    # Example path: "Hec_10/np/combined_summary_np.csv"
    path_parts = Path(filepath).parts
    
    print(f"  Extracting from path: {filepath}")
    print(f"  Path parts: {path_parts}")
    
    # Get dataset (parent directory of parent directory)
    dataset = None
    for i, part in enumerate(path_parts):
        if part.startswith('Hec_'):
            dataset = part
            break
    
    if dataset is None:
        dataset = path_parts[-3] if len(path_parts) >= 3 else "unknown"
    
    # Get model name from filename
    filename = Path(filepath).stem  # "combined_summary_np"
    model_match = re.search(r'combined_summary_(.+)', filename)
    model = model_match.group(1) if model_match else "unknown"
    
    print(f"  Dataset: {dataset}, Model: {model}")
    
    return model, dataset

def combine_with_original_and_filter(mcmc_file, model_name, dataset_name, output_dir, original_df):
    """
    Combine MCMC results with original HECATE data and apply filtering.
    Returns the filtered dataframe.
    """
    try:
        # Read MCMC file
        print(f"\n  Reading MCMC file: {mcmc_file}")
        mcmc_df = pd.read_csv(mcmc_file)
        
        print(f"  Combining {dataset_name}/{model_name}:")
        print(f"    MCMC data: {len(mcmc_df)} rows, {len(mcmc_df.columns)} columns")
        print(f"    Original data: {len(original_df)} rows (from Hec_50.csv)")
        
        # Debug: show first few columns of each dataframe
        print(f"    MCMC columns (first 10): {list(mcmc_df.columns[:10])}")
        print(f"    Original columns (first 10): {list(original_df.columns[:10])}")
        
        # Find the id column in MCMC results
        id_col_mcmc = f'id_{model_name}'
        if id_col_mcmc not in mcmc_df.columns:
            # Try without model suffix
            if 'id' in mcmc_df.columns:
                id_col_mcmc = 'id'
            elif 'id_np' in mcmc_df.columns and model_name == 'np':
                id_col_mcmc = 'id_np'
            elif 'id_uni' in mcmc_df.columns and model_name == 'uni':
                id_col_mcmc = 'id_uni'
            elif 'id_skew' in mcmc_df.columns and model_name == 'skew':
                id_col_mcmc = 'id_skew'
            else:
                print(f"    Warning: No id column found in MCMC results")
                print(f"    Available columns: {[col for col in mcmc_df.columns if 'id' in col.lower()]}")
                return None
        
        print(f"    Using MCMC id column: {id_col_mcmc}")
        
        # Check for ID in original
        if 'ID' not in original_df.columns:
            print(f"    Warning: No ID column in original file")
            print(f"    Available columns: {[col for col in original_df.columns if 'id' in col.lower()]}")
            return None
        
        # Convert id columns to same type (int)
        try:
            mcmc_df[id_col_mcmc] = mcmc_df[id_col_mcmc].astype(int)
            original_df['ID'] = original_df['ID'].astype(int)
        except:
            print(f"    Warning: Could not convert ID columns to int")
        
        # Perform outer join
        print(f"    Merging on original['ID'] == mcmc['{id_col_mcmc}']")
        merged_df = pd.merge(
            original_df,
            mcmc_df,
            left_on='ID',
            right_on=id_col_mcmc,
            how='outer',
            suffixes=('', f'_{model_name}_mcmc')
        )
        
        print(f"    After merge: {len(merged_df)} rows")
        
        # Apply filtering: |logSFR_total - logSFR_today_pred_*| < 1
        logSFR_pred_col = f'logSFR_today_pred_{model_name}'
        
        # Debug: check if columns exist
        print(f"    Looking for columns: 'logSFR_total' and '{logSFR_pred_col}'")
        
        if 'logSFR_total' in merged_df.columns and logSFR_pred_col in merged_df.columns:
            # Calculate difference
            diff = np.abs(merged_df['logSFR_total'] - merged_df[logSFR_pred_col])
            
            # Count rows before filtering
            valid_before = merged_df[logSFR_pred_col].notna().sum()
            
            # Set values to NaN where difference >= 1
            mask_invalid = (diff >= 1) | diff.isna()
            merged_df.loc[mask_invalid, logSFR_pred_col] = np.nan
            
            # Also set related columns to NaN for consistency
            related_cols = [col for col in merged_df.columns if 
                          col.startswith(f't_sf_{model_name}') or 
                          col.startswith(f'tau_{model_name}') or 
                          col.startswith(f'A_{model_name}') or 
                          col.startswith(f'x_{model_name}') or
                          col.startswith(f'logSFR_today_pred_sd_{model_name}')]
            
            print(f"    Related columns to filter: {related_cols}")
            
            for col in related_cols:
                if col in merged_df.columns:
                    merged_df.loc[mask_invalid, col] = np.nan
            
            # Count rows after filtering
            valid_after = merged_df[logSFR_pred_col].notna().sum()
            
            print(f"    Filtering results:")
            print(f"      Valid before: {valid_before}")
            print(f"      Valid after: {valid_after}")
            print(f"      Removed: {valid_before - valid_after}")
        else:
            print(f"    Warning: Cannot apply filtering - missing required columns")
            if 'logSFR_total' not in merged_df.columns:
                print(f"      Missing: logSFR_total")
            if logSFR_pred_col not in merged_df.columns:
                print(f"      Missing: {logSFR_pred_col}")
                print(f"      Available logSFR columns: {[col for col in merged_df.columns if 'logSFR' in col]}")
        
        # Add metadata
        merged_df['dataset'] = dataset_name
        merged_df['model'] = model_name
        
        # Save intermediate result
        output_file = os.path.join(output_dir, f"{dataset_name}_{model_name}_filtered.csv")
        merged_df.to_csv(output_file, index=False)
        print(f"    Saved to: {output_file}")
        
        return merged_df
        
    except Exception as e:
        print(f"    Error processing {mcmc_file}: {e}")
        import traceback
        traceback.print_exc()
        return None

def combine_all_filtered_results(filtered_dir, output_file):
    """
    Combine all filtered results into a single file.
    """
    # Find all filtered files
    pattern = os.path.join(filtered_dir, "*_filtered.csv")
    filtered_files = glob.glob(pattern)
    
    if not filtered_files:
        print("No filtered files found!")
        return None
    
    print(f"\nCombining {len(filtered_files)} filtered files...")
    
    # Group files by dataset to avoid duplicates
    dataset_files = {}
    for file in sorted(filtered_files):
        # Extract dataset name from filename (e.g., "Hec_10_np_filtered.csv" -> "Hec_10")
        basename = os.path.basename(file)
        parts = basename.split('_')
        if len(parts) >= 2 and parts[0] == 'Hec':
            dataset = parts[0] + '_' + parts[1]
        else:
            dataset = 'unknown'
        
        if dataset not in dataset_files:
            dataset_files[dataset] = []
        dataset_files[dataset].append(file)
    
    print(f"  Found datasets: {list(dataset_files.keys())}")
    
    # For each dataset, combine all models
    combined_datasets = []
    
    for dataset, files in dataset_files.items():
        if len(files) == 1:
            # Only one model for this dataset
            df = pd.read_csv(files[0])
            combined_datasets.append(df)
            print(f"  {dataset}: {len(df)} rows (1 model)")
        else:
            # Multiple models for this dataset - need to merge
            print(f"  {dataset}: merging {len(files)} models")
            
            # Start with the first file
            base_df = pd.read_csv(files[0])
            
            # Merge with other models
            for file in files[1:]:
                model_df = pd.read_csv(file)
                
                # Extract model name from filename
                basename = os.path.basename(file)
                parts = basename.replace('_filtered.csv', '').split('_')
                if len(parts) >= 3:
                    model_name = parts[2]
                else:
                    model_name = 'unknown'
                
                print(f"    Merging model: {model_name}")
                
                # Drop duplicate columns except for model-specific ones
                cols_to_keep = ['ID'] + [col for col in model_df.columns 
                                       if model_name in col or col not in base_df.columns]
                model_df = model_df[cols_to_keep]
                
                # Merge on ID
                base_df = pd.merge(base_df, model_df, on='ID', how='outer', 
                                 suffixes=('', f'_{model_name}_dup'))
            
            combined_datasets.append(base_df)
            print(f"    Combined: {len(base_df)} rows")
    
    # Combine all datasets
    if not combined_datasets:
        print("No valid data to combine!")
        return None
    
    combined_df = pd.concat(combined_datasets, ignore_index=True)
    
    print(f"\nCombined result: {len(combined_df)} total rows")
    
    # Remove duplicate rows based on ID if present
    if 'ID' in combined_df.columns:
        before_dedup = len(combined_df)
        combined_df = combined_df.drop_duplicates(subset=['ID'], keep='first')
        after_dedup = len(combined_df)
        if before_dedup != after_dedup:
            print(f"Removed {before_dedup - after_dedup} duplicate rows")
    
    # Sort by ID for consistency
    if 'ID' in combined_df.columns:
        combined_df = combined_df.sort_values('ID').reset_index(drop=True)
    
    # Save final result
    combined_df.to_csv(output_file, index=False)
    print(f"\nFinal combined results saved to: {output_file}")
    print(f"Final shape: {combined_df.shape}")
    
    return combined_df

def create_summary_statistics(final_df, output_dir):
    """
    Create summary statistics for the final combined dataset.
    """
    summary_file = os.path.join(output_dir, "summary_statistics.txt")
    
    with open(summary_file, 'w') as f:
        f.write("=== MCMC RESULTS SUMMARY STATISTICS ===\n\n")
        
        f.write(f"Total rows: {len(final_df)}\n")
        f.write(f"Total columns: {len(final_df.columns)}\n\n")
        
        # Dataset distribution
        if 'dataset' in final_df.columns:
            f.write("Dataset distribution:\n")
            dataset_counts = final_df['dataset'].value_counts()
            for dataset, count in dataset_counts.items():
                f.write(f"  {dataset}: {count} rows\n")
            f.write("\n")
        
        # Model coverage
        if 'model' in final_df.columns:
            f.write("Model distribution:\n")
            model_counts = final_df['model'].value_counts()
            for model, count in model_counts.items():
                f.write(f"  {model}: {count} rows\n")
            f.write("\n")
        
        # Valid predictions per model
        f.write("Valid predictions per model:\n")
        for model in ['np', 'uni', 'skew']:
            logSFR_col = f'logSFR_today_pred_{model}'
            if logSFR_col in final_df.columns:
                valid_count = final_df[logSFR_col].notna().sum()
                f.write(f"  {model}: {valid_count} valid predictions\n")
        f.write("\n")
        
        # Missing data summary
        f.write("Columns with most missing data:\n")
        missing_counts = final_df.isnull().sum().sort_values(ascending=False).head(20)
        for col, count in missing_counts.items():
            if count > 0:
                pct = (count / len(final_df)) * 100
                f.write(f"  {col}: {count} missing ({pct:.1f}%)\n")
    
    print(f"Summary statistics saved to: {summary_file}")

def main():
    """Main function to combine MCMC results."""
    print("=== Modified MCMC Results Combiner ===")
    print(f"Current working directory: {os.getcwd()}")
    print("Using Hec_50.csv as the source for all original HECATE data\n")
    
    # Load the original HECATE data once
    original_file = "tables/Hec_50.csv"
    
    # Also check alternative location
    if not os.path.exists(original_file):
        original_file = "Hec_50.csv"
    
    if not os.path.exists(original_file):
        print(f"Error: Original HECATE file not found!")
        print(f"Looked for: 'tables/Hec_50.csv' and 'Hec_50.csv'")
        print(f"Please ensure the file exists in one of these locations.")
        return
    
    try:
        print(f"Loading original HECATE data from: {original_file}")
        original_df = pd.read_csv(original_file)
        print(f"Loaded original HECATE data: {len(original_df)} rows, {len(original_df.columns)} columns\n")
    except Exception as e:
        print(f"Error loading original HECATE data: {e}")
        return
    
    # Find all combined_summary files
    print("Finding MCMC combined summary files...")
    files = find_combined_summary_files()
    
    if not files:
        print("\nNo combined_summary_*.csv files found!")
        print("Please check that:")
        print("1. You are in the correct directory")
        print("2. The MCMC results have been generated")
        print("3. The files follow the pattern 'combined_summary_*.csv'")
        return
    
    print(f"\nFound {len(files)} files:")
    for f in sorted(files):
        print(f"  {f}")
    
    # Create output directories
    filtered_dir = "tables/filtered_results"
    os.makedirs(filtered_dir, exist_ok=True)
    os.makedirs("tables", exist_ok=True)
    
    # Process each file: combine with original and filter
    print("\n=== STEP 1: Combining with original and filtering ===")
    processed_count = 0
    
    for filepath in sorted(files):
        model, dataset = extract_model_and_dataset(filepath)
        result_df = combine_with_original_and_filter(
            filepath, model, dataset, filtered_dir, original_df
        )
        if result_df is not None:
            processed_count += 1
    
    if processed_count == 0:
        print("\nNo files were successfully processed!")
        return
    
    print(f"\nSuccessfully processed {processed_count} files")
    
    # Combine all filtered results
    print("\n=== STEP 2: Combining all filtered results ===")
    final_output = os.path.join("tables", "MCMC_results_combined_filtered.csv")
    final_df = combine_all_filtered_results(filtered_dir, final_output)
    
    if final_df is not None:
        # Create summary statistics
        print("\n=== STEP 3: Creating summary statistics ===")
        create_summary_statistics(final_df, "tables")
        
        print(f"\n=== COMPLETED SUCCESSFULLY ===")
        print(f"Final combined file: {final_output}")
        print(f"Summary statistics: tables/summary_statistics.txt")
    else:
        print("Failed to create final combined file!")

if __name__ == "__main__":
    main()
