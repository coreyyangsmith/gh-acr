"""
Script to check for valid bypass7 folders and .txt files within ID subfolders.

Reports all unique IDs that do NOT have:
1. A bypass7 subfolder, OR
2. Any .txt files within bypass7, OR
3. .txt files with content (non-empty)
"""

import csv
import os
import sys
from pathlib import Path


def check_bypass_validity(folder_path: str) -> dict:
    """
    Check all subfolders in the given folder for valid bypass7 directories
    and .txt files with content.
    
    Args:
        folder_path: Path to the folder to check
        
    Returns:
        Dictionary with results containing:
        - invalid_ids: List of IDs without valid bypass7 or .txt files
        - no_bypass7: IDs missing bypass7 folder
        - no_txt_files: IDs with bypass7 but no .txt files
        - empty_txt_files: IDs with bypass7 and .txt files but all are empty
        - valid_ids: IDs with valid bypass7 and non-empty .txt files
    """
    folder = Path(folder_path)
    
    if not folder.exists():
        print(f"Error: Folder '{folder_path}' does not exist.")
        sys.exit(1)
    
    if not folder.is_dir():
        print(f"Error: '{folder_path}' is not a directory.")
        sys.exit(1)
    
    results = {
        'no_bypass7': [],
        'no_txt_files': [],
        'empty_txt_files': [],
        'valid_ids': [],
        'invalid_ids': []
    }
    
    # Get all subdirectories (IDs)
    subdirs = [d for d in folder.iterdir() if d.is_dir()]
    total_ids = len(subdirs)
    
    print(f"Checking {total_ids} ID folders in: {folder_path}")
    print("-" * 60)
    
    for subdir in subdirs:
        id_name = subdir.name
        bypass7_path = subdir / "bypass7"
        
        # Check 1: Does bypass7 folder exist?
        if not bypass7_path.exists():
            results['no_bypass7'].append(id_name)
            results['invalid_ids'].append(id_name)
            continue
            
        if not bypass7_path.is_dir():
            results['no_bypass7'].append(id_name)
            results['invalid_ids'].append(id_name)
            continue
        
        # Check 2: Are there any .txt files in bypass7 (including nested)?
        txt_files = list(bypass7_path.rglob("*.txt"))
        
        if not txt_files:
            results['no_txt_files'].append(id_name)
            results['invalid_ids'].append(id_name)
            continue
        
        # Check 3: Do any .txt files have content?
        has_content = False
        for txt_file in txt_files:
            try:
                if txt_file.stat().st_size > 0:
                    # Double-check by reading
                    content = txt_file.read_text(encoding='utf-8', errors='ignore').strip()
                    if content:
                        has_content = True
                        break
            except Exception:
                continue
        
        if not has_content:
            results['empty_txt_files'].append(id_name)
            results['invalid_ids'].append(id_name)
        else:
            results['valid_ids'].append(id_name)
    
    return results


def print_results(results: dict):
    """Print the results in a formatted way."""
    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)
    
    total_checked = (len(results['valid_ids']) + len(results['invalid_ids']))
    
    print(f"\nTotal IDs checked: {total_checked}")
    print(f"Valid IDs (have bypass7 with non-empty .txt files): {len(results['valid_ids'])}")
    print(f"Invalid IDs: {len(results['invalid_ids'])}")
    
    print("\n" + "-" * 60)
    print("BREAKDOWN OF INVALID IDs:")
    print("-" * 60)
    
    print(f"\n1. Missing bypass7 folder: {len(results['no_bypass7'])}")
    if results['no_bypass7']:
        for id_name in sorted(results['no_bypass7'])[:20]:
            print(f"   - {id_name}")
        if len(results['no_bypass7']) > 20:
            print(f"   ... and {len(results['no_bypass7']) - 20} more")
    
    print(f"\n2. Has bypass7 but no .txt files: {len(results['no_txt_files'])}")
    if results['no_txt_files']:
        for id_name in sorted(results['no_txt_files'])[:20]:
            print(f"   - {id_name}")
        if len(results['no_txt_files']) > 20:
            print(f"   ... and {len(results['no_txt_files']) - 20} more")
    
    print(f"\n3. Has bypass7 and .txt files but all are empty: {len(results['empty_txt_files'])}")
    if results['empty_txt_files']:
        for id_name in sorted(results['empty_txt_files'])[:20]:
            print(f"   - {id_name}")
        if len(results['empty_txt_files']) > 20:
            print(f"   ... and {len(results['empty_txt_files']) - 20} more")
    
    print("\n" + "=" * 60)
    print("ALL INVALID IDs (sorted):")
    print("=" * 60)
    
    if results['invalid_ids']:
        for id_name in sorted(results['invalid_ids']):
            print(id_name)
    else:
        print("None - all IDs are valid!")
    
    print("\n" + "=" * 60)


def save_csv(results: dict, output_path: str):
    """Save invalid IDs to a CSV file with their reason."""
    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['id', 'reason'])
        
        for id_name in sorted(results['no_bypass7']):
            writer.writerow([id_name, 'missing_bypass7'])
        
        for id_name in sorted(results['no_txt_files']):
            writer.writerow([id_name, 'no_txt_files'])
        
        for id_name in sorted(results['empty_txt_files']):
            writer.writerow([id_name, 'empty_txt_files'])
    
    print(f"\nCSV saved to: {output_path}")


def main():
    if len(sys.argv) < 2:
        print("Usage: python check_bypass_validity.py <folder_path>")
        print("Example: python check_bypass_validity.py data/llama-3.1-8b")
        sys.exit(1)
    
    folder_path = sys.argv[1]
    results = check_bypass_validity(folder_path)
    print_results(results)
    
    # Save CSV with invalid IDs
    folder_name = Path(folder_path).name
    output_path = f"data/{folder_name}_invalid_bypass7.csv"
    save_csv(results, output_path)
    
    # Return exit code based on whether there are invalid IDs
    return 1 if results['invalid_ids'] else 0


if __name__ == "__main__":
    sys.exit(main())
