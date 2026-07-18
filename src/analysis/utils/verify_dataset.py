import argparse
import csv
import os
import sys

def verify_dataset(csv_path, parent_folder_path):
    # 1. Read CSV and get unique IDs
    csv_ids = set()
    try:
        with open(csv_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            if 'id' not in reader.fieldnames:
                print(f"Error: Column 'id' not found in {csv_path}. Available columns: {reader.fieldnames}")
                return

            for row in reader:
                if row['id']:
                    csv_ids.add(row['id'].strip())
    except FileNotFoundError:
        print(f"Error: CSV file not found at {csv_path}")
        return
    except Exception as e:
        print(f"Error reading CSV: {e}")
        return

    print(f"Found {len(csv_ids)} unique IDs in CSV.")

    # 2. Get folder names from parent directory
    if not os.path.isdir(parent_folder_path):
        print(f"Error: Parent folder not found at {parent_folder_path}")
        return

    folder_names = set()
    try:
        for entry in os.listdir(parent_folder_path):
            full_path = os.path.join(parent_folder_path, entry)
            if os.path.isdir(full_path):
                folder_names.add(entry)
    except Exception as e:
        print(f"Error reading directory: {e}")
        return

    print(f"Found {len(folder_names)} folders in parent directory.")

    # 3. Compare and find mismatches
    ids_in_csv_no_folder = csv_ids - folder_names
    folders_no_csv_id = folder_names - csv_ids

    # 4. Output results
    print("\n" + "="*50)
    print("MISMATCH REPORT")
    print("="*50)

    if not ids_in_csv_no_folder and not folders_no_csv_id:
        print("Success! All IDs in CSV match folders, and all folders match IDs in CSV.")
    else:
        if ids_in_csv_no_folder:
            print(f"\nIDs present in CSV but MISSING folder ({len(ids_in_csv_no_folder)}):")
            for missing_id in sorted(list(ids_in_csv_no_folder)):
                print(f"  - {missing_id}")
        
        if folders_no_csv_id:
            print(f"\nFolders present but MISSING from CSV ({len(folders_no_csv_id)}):")
            for extra_folder in sorted(list(folders_no_csv_id)):
                print(f"  - {extra_folder}")

    print("\n" + "="*50)


def main():
    parser = argparse.ArgumentParser(description="Verify dataset consistency between a CSV file and a directory of folders.")
    parser.add_argument("csv_path", help="Path to the CSV file containing an 'id' column.")
    parser.add_argument("parent_folder_path", help="Path to the parent folder containing subfolders to check.")

    args = parser.parse_args()

    print(f"Verifying:\n  CSV: {args.csv_path}\n  Folder: {args.parent_folder_path}\n")
    verify_dataset(args.csv_path, args.parent_folder_path)

if __name__ == "__main__":
    main()



