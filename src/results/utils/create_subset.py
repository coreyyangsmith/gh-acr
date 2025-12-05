import argparse
import csv
import os
import sys

def create_subset(input_csv, folder_path, output_csv):
    # 1. Read folder names from directory
    if not os.path.isdir(folder_path):
        print(f"Error: Folder path not found at {folder_path}")
        return

    folder_names = set()
    try:
        for entry in os.listdir(folder_path):
            full_path = os.path.join(folder_path, entry)
            if os.path.isdir(full_path):
                folder_names.add(entry)
    except Exception as e:
        print(f"Error reading directory: {e}")
        return

    print(f"Found {len(folder_names)} folders in directory.")

    # 2. Read CSV, identify IDs, and filter
    csv_ids = set()
    valid_rows = []
    
    # Store IDs that are in CSV but missing folders for reporting
    ids_missing_folders = set()

    try:
        with open(input_csv, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            if 'id' not in reader.fieldnames:
                print(f"Error: Column 'id' not found in {input_csv}. Available columns: {reader.fieldnames}")
                return
            
            fieldnames = reader.fieldnames
            
            for row in reader:
                row_id = row['id'].strip()
                if row_id:
                    csv_ids.add(row_id)
                    
                    if row_id in folder_names:
                        valid_rows.append(row)
                    else:
                        ids_missing_folders.add(row_id)
                        
    except FileNotFoundError:
        print(f"Error: CSV file not found at {input_csv}")
        return
    except Exception as e:
        print(f"Error reading CSV: {e}")
        return

    # 3. Calculate statistics
    # IDs in CSV but missing folder (already collected in ids_missing_folders)
    # Folders present but missing from CSV
    folders_missing_from_csv = folder_names - csv_ids

    # 4. Report statistics (matching verify_dataset.py style)
    print("\n" + "="*50)
    print("SUBSET CREATION REPORT")
    print("="*50)
    
    print(f"Total IDs in original CSV: {len(csv_ids)}")
    print(f"Total folders found: {len(folder_names)}")
    
    if ids_missing_folders:
        print(f"\nIDs present in CSV but MISSING folder ({len(ids_missing_folders)}):")
        print(f"  (These will be REMOVED from the new CSV)")
        # Uncomment to list them if needed, keeping concise for now as per "identify"
        # for missing_id in sorted(list(ids_missing_folders)):
        #     print(f"  - {missing_id}")
    else:
        print("\nNo IDs found in CSV that are missing folders.")

    if folders_missing_from_csv:
        print(f"\nFolders present but MISSING from CSV ({len(folders_missing_from_csv)}):")
        print(f"  (These are ignored as they are not in the source CSV)")
        # for extra_folder in sorted(list(folders_missing_from_csv)):
        #     print(f"  - {extra_folder}")
    else:
        print("\nNo extra folders found that are missing from CSV.")

    print(f"\nNew CSV will contain {len(valid_rows)} rows.")

    # 5. Write new CSV
    try:
        with open(output_csv, 'w', encoding='utf-8', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(valid_rows)
        print(f"\nSuccessfully wrote subset CSV to: {output_csv}")
    except Exception as e:
        print(f"Error writing output CSV: {e}")

def main():
    parser = argparse.ArgumentParser(description="Create a subset CSV by removing IDs that do not have a corresponding folder.")
    parser.add_argument("input_csv", help="Path to the source CSV file containing an 'id' column.")
    parser.add_argument("folder_path", help="Path to the parent folder containing subfolders to check against.")
    parser.add_argument("output_csv", help="Path where the new subset CSV will be saved.")

    args = parser.parse_args()

    print(f"Processing:\n  Input CSV: {args.input_csv}\n  Folder: {args.folder_path}\n  Output CSV: {args.output_csv}\n")
    create_subset(args.input_csv, args.folder_path, args.output_csv)

if __name__ == "__main__":
    main()



