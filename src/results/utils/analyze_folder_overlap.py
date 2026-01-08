import argparse
import os
import sys

def get_folder_names(parent_path):
    folders = set()
    if not os.path.isdir(parent_path):
        print(f"Error: Path not found or not a directory: {parent_path}")
        return None
        
    try:
        for entry in os.listdir(parent_path):
            full_path = os.path.join(parent_path, entry)
            if os.path.isdir(full_path):
                folders.add(entry)
    except Exception as e:
        print(f"Error reading directory {parent_path}: {e}")
        return None
    return folders

def analyze_folder_overlap(path1, path2):
    print(f"Analyzing folder overlap between:\n  1. {path1}\n  2. {path2}\n")

    folders1 = get_folder_names(path1)
    if folders1 is None: return

    folders2 = get_folder_names(path2)
    if folders2 is None: return

    print(f"Path 1 has {len(folders1)} folders.")
    print(f"Path 2 has {len(folders2)} folders.")

    overlap = folders1.intersection(folders2)
    unique_to_1 = folders1 - folders2
    unique_to_2 = folders2 - folders1

    print("\n" + "="*50)
    print("FOLDER OVERLAP ANALYSIS")
    print("="*50)
    
    print(f"Overlapping Folders count: {len(overlap)}")
    if overlap:
        print(f"Overlapping Folders (First 10): {list(sorted(overlap))[:10]}")
        if len(overlap) > 10:
            print(f"... and {len(overlap) - 10} more.")
            
    print("-" * 30)
    print(f"Unique to Path 1: {len(unique_to_1)}")
    print(f"Unique to Path 2: {len(unique_to_2)}")
    print("="*50)

def main():
    parser = argparse.ArgumentParser(description="Find overlapping folder names between two parent directories.")
    parser.add_argument("path1", help="Path to the first parent directory.")
    parser.add_argument("path2", help="Path to the second parent directory.")

    args = parser.parse_args()
    analyze_folder_overlap(args.path1, args.path2)

if __name__ == "__main__":
    main()




