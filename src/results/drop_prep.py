"""Utility to drop rows with `eval_method == 'prep'` from a results CSV."""

import pandas as pd
from pathlib import Path
import tyro


def main(input_file: Path, output_file: Path) -> None:
    df = pd.read_csv(input_file)
    filtered_df = df[df['eval_method'] != 'prep']
    output_file.parent.mkdir(parents=True, exist_ok=True)
    filtered_df.to_csv(output_file, index=False)
    print({
        "input": str(input_file),
        "output": str(output_file),
        "removed": int(len(df) - len(filtered_df)),
        "remaining": int(len(filtered_df)),
    })


if __name__ == "__main__":
    args = tyro.cli(tuple[Path, Path])
    main(*args)


