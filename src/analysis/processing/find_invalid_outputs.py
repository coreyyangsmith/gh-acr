"""Find IDs with invalid/empty bypass7 outputs by scanning folder structure.

An ID is considered INVALID if:
  1. The bypass7 folder doesn't exist, OR
  2. Any ``final/resolved.txt`` under bypass7 is missing or empty (0 bytes).
     Falls back to scanning all ``.txt`` files when no ``final/`` dirs exist
     (legacy flat layout).

How to run (from repo root, PowerShell single-line commands):

- Module form (recommended):
  python -m src.analysis.processing.find_invalid_outputs data\llama-3.1-8b

- With optional output file:
  python -m src.analysis.processing.find_invalid_outputs data\llama-3.1-8b --output data\invalid_ids.txt

Example:
  python -m src.analysis.processing.find_invalid_outputs data\llama-3.1-8b
"""

from pathlib import Path
from typing import Optional
import tyro
from dataclasses import dataclass


@dataclass
class Args:
    input_folder: Path  # Folder containing ID subfolders
    output: Optional[Path] = None  # Optional file to save invalid IDs


def find_invalid_ids(input_folder: Path) -> tuple[list[str], list[str], int]:
    """Scan folder for IDs with invalid bypass7 outputs.
    
    Returns:
        Tuple of (invalid_ids, valid_ids, total_ids)
    """
    invalid_ids = []
    valid_ids = []
    
    # Iterate through all subdirectories (each is an ID)
    for id_folder in input_folder.iterdir():
        if not id_folder.is_dir():
            continue
        
        id_name = id_folder.name
        bypass7_folder = id_folder / "bypass7"
        
        # Check if bypass7 folder exists
        if not bypass7_folder.exists():
            invalid_ids.append(id_name)
            continue
        
        final_resolved = list(bypass7_folder.rglob("final/resolved.txt"))
        if final_resolved:
            has_empty = any(p.stat().st_size == 0 for p in final_resolved)
            if has_empty:
                invalid_ids.append(id_name)
            else:
                valid_ids.append(id_name)
            continue

        # Legacy flat layout: any empty .txt under bypass7 marks invalid
        has_empty_txt = False
        txt_files_found = False
        
        for txt_file in bypass7_folder.rglob("*.txt"):
            txt_files_found = True
            if txt_file.stat().st_size == 0:
                has_empty_txt = True
                break
        
        if not txt_files_found or has_empty_txt:
            invalid_ids.append(id_name)
        else:
            valid_ids.append(id_name)
    
    return invalid_ids, valid_ids, len(invalid_ids) + len(valid_ids)


def main(args: Args) -> None:
    print(f"\n{'='*60}")
    print("Scanning for Invalid Bypass7 Outputs")
    print(f"{'='*60}")
    print(f"Input folder: {args.input_folder}")
    
    invalid_ids, valid_ids, total_ids = find_invalid_ids(args.input_folder)
    
    print(f"\n{'='*60}")
    print("Results:")
    print(f"  Total IDs scanned: {total_ids}")
    print(f"  Valid IDs: {len(valid_ids)}")
    print(f"  Invalid IDs: {len(invalid_ids)}")
    
    if invalid_ids:
        print(f"\n{'='*60}")
        print(f"Invalid IDs ({len(invalid_ids)} total):")
        for id_val in invalid_ids[:50]:
            print(f"  {id_val}")
        if len(invalid_ids) > 50:
            print(f"  ... and {len(invalid_ids) - 50} more")
    
    # Save to file if requested
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, 'w') as f:
            for id_val in invalid_ids:
                f.write(f"{id_val}\n")
        print(f"\n{'='*60}")
        print(f"Invalid IDs saved to: {args.output}")
    
    print(f"\n{'='*60}")
    print("Summary:")
    print({
        "input_folder": str(args.input_folder),
        "total_ids": total_ids,
        "valid_ids": len(valid_ids),
        "invalid_ids": len(invalid_ids),
        "output": str(args.output) if args.output else None,
    })


if __name__ == "__main__":
    args = tyro.cli(Args)
    main(args)
