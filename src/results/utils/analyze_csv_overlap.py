import argparse
import csv
import sys
import os

def get_ids_from_csv(csv_path):
    ids = set()
    try:
        with open(csv_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            if 'id' not in reader.fieldnames:
                print(f"Error: Column 'id' not found in {csv_path}. Available columns: {reader.fieldnames}")
                return None
            
            for row in reader:
                if row['id']:
                    ids.add(row['id'].strip())
    except FileNotFoundError:
        print(f"Error: CSV file not found at {csv_path}")
        return None
    except Exception as e:
        print(f"Error reading CSV {csv_path}: {e}")
        return None
    return ids

def analyze_csv_overlap(csv1_path, csv2_path):
    print(f"Analyzing overlap between:\n  1. {csv1_path}\n  2. {csv2_path}\n")

    ids1 = get_ids_from_csv(csv1_path)
    if ids1 is None: return

    ids2 = get_ids_from_csv(csv2_path)
    if ids2 is None: return

    print(f"CSV 1 has {len(ids1)} unique IDs.")
    print(f"CSV 2 has {len(ids2)} unique IDs.")

    overlap = ids1.intersection(ids2)
    unique_to_1 = ids1 - ids2
    unique_to_2 = ids2 - ids1

    print("\n" + "="*50)
    print("OVERLAP ANALYSIS")
    print("="*50)
    
    print(f"Overlapping IDs count: {len(overlap)}")
    if overlap:
        print(f"Overlapping IDs (First 10): {list(sorted(overlap))[:10]}")
        if len(overlap) > 10:
            print(f"... and {len(overlap) - 10} more.")
    
    print("-" * 30)
    print(f"Unique to CSV 1: {len(unique_to_1)}")
    print(f"Unique to CSV 2: {len(unique_to_2)}")
    print("="*50)

def main():
    parser = argparse.ArgumentParser(description="Find overlapping unique IDs between two CSV files.")
    parser.add_argument("csv1_path", help="Path to the first CSV file.")
    parser.add_argument("csv2_path", help="Path to the second CSV file.")

    args = parser.parse_args()
    analyze_csv_overlap(args.csv1_path, args.csv2_path)

if __name__ == "__main__":
    main()




