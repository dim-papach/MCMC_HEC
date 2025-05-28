#!/usr/bin/env python3
"""
Combine MCMC results by model name.
Stack datasets for each model into tables/{model_name}_summary.csv
Then join all models into a single comprehensive results file.
"""

import pandas as pd
import numpy as np
import os
import glob
from pathlib import Path
import re

def find_combined_summary_files(base_dir="."):
    """Find all combined_summary_*.csv files in the directory structure."""
    pattern = os.path.join(base_dir, "**/combined_summary_*.csv")
    files = glob.glob(pattern, recursive=True)
    return files

def extract_model_and_dataset(filepath):
    """Extract model name and dataset name from file path."""
    # Example path: "Hec_10/np/combined_summary_np.csv"
    path_parts = Path(filepath).parts
    
    # Get dataset (parent directory of parent directory)
    dataset = path_parts[-3] if len(path_parts) >= 3 else "unknown"
    
    # Get model name from filename
    filename = Path(filepath).stem  # "combined_summary_np"
    model_match = re.search(r'combined_summary_(.+)', filename)
    model = model_match.group(1) if model_match else "unknown"
    
    return model, dataset

def combine_files_by_model(files, output_dir="tables"):
    """Combine files by model, stacking datasets one under the other."""
    
    # Create output directory
    Path(output_dir).mkdir(exist_ok=True)
    
    # Group files by model
    model_files = {}
    
    for filepath in files:
        model, dataset = extract_model_and_dataset(filepath)
        
        if model not in model_files:
            model_files[model] = []
        
        model_files[model].append({
            'filepath': filepath,
            'dataset': dataset
        })
    
    created_files = []
    
    # Process each model
    for model, file_info_list in model_files.items():
        print(f"\nProcessing model: {model}")
        
        # Read and stack all datasets for this model
        model_data_list = []
        
        for file_info in file_info_list:
            filepath = file_info['filepath']
            dataset = file_info['dataset']
            
            try:
                # Read the CSV file
                df = pd.read_csv(filepath)
                
                # Add metadata columns
                df['dataset'] = dataset
                df['source_file'] = os.path.basename(filepath)
                
                model_data_list.append(df)
                print(f"  Added {filepath}: {len(df)} rows from dataset {dataset}")
                
            except Exception as e:
                print(f"  Warning: Could not read {filepath}: {e}")
                continue
        
        if not model_data_list:
            print(f"  No valid files found for model {model}")
            continue
        
        # Stack all datasets for this model
        stacked_data = pd.concat(model_data_list, ignore_index=True)
        
        # Define output file
        output_file = os.path.join(output_dir, f"{model}_summary.csv")
        
        # Save the combined data (overwrite existing)
        stacked_data.to_csv(output_file, index=False)
        created_files.append(output_file)
        
        print(f"  Created {output_file}: {len(stacked_data)} total rows from {len(file_info_list)} datasets")
    
    return created_files

def join_all_the_files(output_dir="tables"):
    """
    Join all model summary files into a single comprehensive results file.
    Merges on the id_{model} columns to combine results from different models.
    """
    models = ["np", "uni", "skew"]
    combined_data = None
    
    print("\nJoining all model files...")
    
    for i, model in enumerate(models):
        file_path = os.path.join(output_dir, f"{model}_summary.csv")
        
        if not os.path.exists(file_path):
            print(f"Warning: {file_path} does not exist. Skipping model {model}.")
            continue
        
        try:
            df = pd.read_csv(file_path)
            print(f"Reading {file_path}: {len(df)} rows, {len(df.columns)} columns")
            
            if combined_data is None:
                # First model - use as base
                combined_data = df.copy()
                print(f"  Using {model} as base dataframe")
            else:
                # Subsequent models - join on id columns
                
                # Identify the id column for current model
                id_col_current = f"id_{model}"
                
                # Find the id column in the base dataframe (should be from first model)
                base_id_cols = [col for col in combined_data.columns if col.startswith('id_')]
                if not base_id_cols:
                    print(f"Warning: No id column found in base dataframe")
                    continue
                
                base_id_col = base_id_cols[0]  # Use the first id column found
                
                # Check if both id columns exist
                if id_col_current not in df.columns:
                    print(f"Warning: {id_col_current} not found in {file_path}")
                    continue
                
                if base_id_col not in combined_data.columns:
                    print(f"Warning: {base_id_col} not found in combined dataframe")
                    continue
                
                # Perform the join
                print(f"  Joining on {base_id_col} = {id_col_current}")
                
                # Use outer join to keep all records from both dataframes
                combined_data = pd.merge(
                    combined_data, 
                    df, 
                    left_on=base_id_col, 
                    right_on=id_col_current, 
                    how='outer',
                    suffixes=('', f'_{model}')
                )
                
                print(f"  After joining {model}: {len(combined_data)} rows, {len(combined_data.columns)} columns")
                
        except Exception as e:
            print(f"Error reading {file_path}: {e}")
            continue
    
    if combined_data is None:
        print("Error: No valid model files found to combine!")
        return None
    
    # Save the final combined results
    output_file = os.path.join(output_dir, "MCMC_results_combined.csv")
    combined_data.to_csv(output_file, index=False)
    
    print(f"\nFinal combined results saved to: {output_file}")
    print(f"Final dataframe: {len(combined_data)} rows, {len(combined_data.columns)} columns")
    
    # Print column summary for verification
    print("\nColumn summary:")
    for col in sorted(combined_data.columns):
        print(f"  {col}")
    
    return output_file

def verify_results(output_file):
    """
    Verify the combined results and provide a summary.
    """
    if not os.path.exists(output_file):
        print(f"Output file {output_file} does not exist!")
        return
    
    try:
        df = pd.read_csv(output_file)
        print(f"\n=== VERIFICATION SUMMARY ===")
        print(f"Total rows: {len(df)}")
        print(f"Total columns: {len(df.columns)}")
        
        # Check for missing data
        missing_summary = df.isnull().sum()
        print(f"\nMissing data summary:")
        for col, missing_count in missing_summary[missing_summary > 0].items():
            print(f"  {col}: {missing_count} missing ({missing_count/len(df)*100:.1f}%)")
        
        # Check for ID columns
        id_cols = [col for col in df.columns if col.startswith('id_')]
        print(f"\nID columns found: {id_cols}")
        
        # Check datasets
        if 'dataset' in df.columns:
            dataset_counts = df['dataset'].value_counts()
            print(f"\nDataset distribution:")
            for dataset, count in dataset_counts.items():
                print(f"  {dataset}: {count} rows")
        
    except Exception as e:
        print(f"Error verifying results: {e}")
        
        
def combine_with_original():
    """
    Combine the original HECATE with the combined summary files.
    """
    original_file = "tables/Hec_50.csv"
    combined_file = "tables/MCMC_results_combined.csv"
    output_file = "tables/HEC_50_combined.csv"
    
    if not os.path.exists(original_file):
        print(f"Original file {original_file} does not exist!")
        return
    if not os.path.exists(combined_file):
        print(f"Combined file {combined_file} does not exist!")
        return
    try:
        original_df = pd.read_csv(original_file)
        combined_df = pd.read_csv(combined_file)
        
        # Merge on id columns
        id_cols = [col for col in combined_df.columns if col.startswith('id_')]
        if not id_cols:
            print("No ID columns found in combined file!")
            return
        id_col = id_cols[0]  # Use the first ID column found
        
        # Merge original on id column original['ID']
        original_df.rename(columns={'ID': id_col}, inplace=True)
        final_df = pd.merge(original_df, combined_df, on=id_col, how='outer', suffixes=('', '_combined'))
        
        # Save the final combined results
        final_df.to_csv(output_file, index=False)
        print(f"Combined original and MCMC results saved to: {output_file}")
        
        # check if |logSFR_total- logSFR_today_pred_{model}| < 0.1
        if 'logSFR_total' in final_df.columns:
            for model in ['np', 'uni', 'skew']:
                log_col = f"logSFR_today_pred_{model}"
                if log_col in final_df.columns:
                    diff_col = f"logSFR_diff_{model}"
                    final_df[diff_col] = np.abs(final_df['logSFR_total'] - final_df[log_col])
                    print(f"  Added difference column {diff_col} for model {model}")
                    # Check if the difference is within the threshold
                    within_threshold = final_df[diff_col] < 0.1
                    print(f"  {model}: {within_threshold.sum()} rows within threshold (|logSFR_total - logSFR_today_pred_{model}| < 0.1)")
                    # if over threshold, make them NaN
                    final_df.loc[~within_threshold, log_col] = np.nan
                    
        else:            
            print("Warning: logSFR_total column not found in original file!")
        # Save the final combined results
        final_df.to_csv(output_file, index=False)
        print(f"Final combined results saved to: {output_file}")
        
        
    except Exception as e:
        print(f"Error combining original and MCMC results: {e}")
        return

def main():
    """Main function to combine MCMC results."""
    print("=== MCMC Results Combiner ===")
    print("Finding MCMC combined summary files...")
    
    # Find all combined_summary files
    files = find_combined_summary_files()
    
    if not files:
        print("No combined_summary_*.csv files found!")
        return
    
    print(f"Found {len(files)} files:")
    for f in files:
        print(f"  {f}")
    
    # Combine files by model
    print("\n=== STEP 1: Combining files by model ===")
    created_files = combine_files_by_model(files)
    
    if not created_files:
        print("No model summary files were created!")
        return
    
    print(f"\nSuccessfully created {len(created_files)} model summary files:")
    for f in created_files:
        print(f"  {f}")
    
    # Join all model files
    print("\n=== STEP 2: Joining all models ===")
    final_output = join_all_the_files()
    
    if final_output:
        print("\n=== STEP 3: Verification ===")
        verify_results(final_output)
        print(f"\n=== COMPLETED SUCCESSFULLY ===")
        print(f"Final combined file: {final_output}")
    else:
        print("Failed to create final combined file!")
        
    # Combine with original HECATE results
    print("\n=== STEP 4: Combining with original HECATE results ===")
    combine_with_original()
    print("Original HECATE results combined with MCMC results.")

if __name__ == "__main__":
    main()
