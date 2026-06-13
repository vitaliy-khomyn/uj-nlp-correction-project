"""Utility script to inspect the contents of generated Parquet files."""
import argparse
import pandas as pd
import os
import sys
import random


def main():
    """Inspects a generated Parquet file via CLI, printing out pairs or head rows."""
    parser = argparse.ArgumentParser(description="Inspect a generated Parquet file.")
    parser.add_argument("file_path", type=str, nargs="?", help="Path to the .parquet file")
    parser.add_argument("--head", type=int, default=10, help="Number of rows to display (default: 10)")
    parser.add_argument("--random", action="store_true", help="Randomly sample pairs instead of taking the first N")

    args = parser.parse_args()

    if not args.file_path:
        parser.print_help()
        sys.exit(1)

    if not os.path.exists(args.file_path):
        print(f"Error: File not found at {args.file_path}")
        sys.exit(1)

    try:
        df = pd.read_parquet(args.file_path)
        print(f"\n=== Parquet File Info: {args.file_path} ===")
        print(f"Shape: {df.shape[0]} rows, {df.shape[1]} columns")
        print(f"Columns: {list(df.columns)}")

        if 'pair_id' in df.columns:
            pair_counts = df['pair_id'].value_counts()
            valid_pair_ids = pair_counts[pair_counts > 1].index.tolist()

            if args.random:
                display_pairs = random.sample(valid_pair_ids, min(args.head, len(valid_pair_ids)))
            else:
                display_pairs = valid_pair_ids[:args.head]

            print(f"\n--- Extracting {len(display_pairs)} TRUE PAIRS ---")
            for pid in display_pairs:
                pair_df = df[df['pair_id'] == pid].sort_values(by='label', ascending=False)
                lemma = pair_df['error_lemma'].iloc[0] if 'error_lemma' in df.columns else 'N/A'
                print(f"\n[Pair ID {pid}] | Lemma: {lemma}")
                for idx, row in pair_df.iterrows():
                    if 'label' in row and 'sentence' in row:
                        prefix = "Source (L2 Error)" if row['label'] == 1 else "✅ Target (Correct)"
                        print(f"  {prefix}: {row['sentence']}")
                    else:
                        print(f"  --- Row {idx} (Label {row.get('label', 'N/A')}) ---")
                        for col in df.columns:
                            print(f"    {col}: {row[col]}")
        else:
            print(f"\n--- First {args.head} rows ---")
            for idx, row in df.head(args.head).iterrows():
                print(f"\n[Row {idx}]")
                for col in df.columns:
                    print(f"  {col}: {row[col]}")
    except Exception as e:
        print(f"Failed to read {args.file_path}: {e}")


if __name__ == "__main__":
    main()
