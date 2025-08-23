import pandas as pd

def main():
    # Define file paths
    input_file = "data/2025_08_17_results_all.csv"  # Update this path as needed
    output_file = "data/output_filtered.csv"  # Update this path as needed
    
    # Read the CSV file
    df = pd.read_csv(input_file)
    
    # Filter out rows where method equals "prep"
    filtered_df = df[df['eval_method'] != 'prep']
    
    # Export to new file
    filtered_df.to_csv(output_file, index=False)
    
    print(f"Filtered data saved to {output_file}")
    print(f"Removed {len(df) - len(filtered_df)} rows with method='prep'")
    print(f"Remaining rows: {len(filtered_df)}")

if __name__ == "__main__":
    main()


