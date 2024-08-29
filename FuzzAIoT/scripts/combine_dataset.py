import pandas as pd

def combine_csv_files(benign_csv, ddos_csv, output_csv):
    # Load the Benign and DDoS CSV files
    benign_df = pd.read_csv(benign_csv)
    ddos_df = pd.read_csv(ddos_csv)
    
    # Determine the minimum number of rows between the two datasets
    min_rows = min(len(benign_df), len(ddos_df))
    
    # Trim both datasets to have the same number of rows
    benign_df = benign_df.sample(n=min_rows, random_state=42).reset_index(drop=True)
    ddos_df = ddos_df.sample(n=min_rows, random_state=42).reset_index(drop=True)
    
    # Combine the datasets
    combined_df = pd.concat([benign_df, ddos_df]).reset_index(drop=True)
    
    # Shuffle the combined dataset to mix benign and DDoS rows
    combined_df = combined_df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    # Save the combined dataset to a new CSV file
    combined_df.to_csv(output_csv, index=False)
    print(f"Combined CSV file saved to {output_csv}")

def main():
    # Paths to the individual Benign and DDoS CSV files
    benign_csv = "C:/Users/Shaurya/Downloads/FuzzAIoT/data/BenignTraffic.csv"
    ddos_csv = "C:/Users/Shaurya/Downloads/FuzzAIoT/data/DDoSTraffic.csv"
    
    # Output path for the combined CSV file
    output_csv = "C:/Users/Shaurya/Downloads/FuzzAIoT/data/CombinedTraffic.csv"
    
    # Combine the CSV files
    combine_csv_files(benign_csv, ddos_csv, output_csv)

if __name__ == "__main__":
    main()
